"""Per-sample PSNR/SSIM evaluation engine for contract C-05."""

from dataclasses import dataclass, field
import math
from typing import Any, Protocol, Sequence

import numpy as np

from .errors import EvaluationInputError, MetricExecutionError, ProtocolUnavailableError
from .protocol import GROUP_BASICSR_CPU, MetricProtocol, REALBLUR_OFFICIAL, UPSTREAM_BASICSR, get_metric_protocol


class BasicSRAdapter(Protocol):
    """Explicit adapter for the audited BasicSR metric call boundary."""

    def calculate_psnr(self, image1: np.ndarray, image2: np.ndarray, *, crop_border: int,
                       test_y_channel: bool) -> float: ...

    def calculate_ssim(self, image1: np.ndarray, image2: np.ndarray, *, crop_border: int,
                       test_y_channel: bool, ssim3d: bool) -> float: ...


class CpuBasicSRAdapter:
    """CPU implementation of audited BasicSR PSNR and non-3D SSIM math."""

    def calculate_psnr(self, image1: np.ndarray, image2: np.ndarray, *, crop_border: int,
                       test_y_channel: bool) -> float:
        _validate_metric_arrays(image1, image2, crop_border, test_y_channel)
        first, second = _crop(image1, image2, crop_border)
        if test_y_channel:
            first, second = _to_y(first), _to_y(second)
        mse = float(np.mean((first.astype(np.float64) - second.astype(np.float64)) ** 2))
        return math.inf if mse == 0.0 else 20.0 * math.log10(255.0 / math.sqrt(mse))

    def calculate_ssim(self, image1: np.ndarray, image2: np.ndarray, *, crop_border: int,
                       test_y_channel: bool, ssim3d: bool) -> float:
        if ssim3d:
            raise ProtocolUnavailableError("CPU adapter implements only ssim3d=False; use upstream-basicsr with a CUDA adapter")
        _validate_metric_arrays(image1, image2, crop_border, test_y_channel)
        first, second = _crop(image1, image2, crop_border)
        if test_y_channel:
            first, second = _to_y(first), _to_y(second)
        if first.shape[0] < 11 or first.shape[1] < 11:
            raise MetricExecutionError("SSIM requires post-crop height and width of at least 11 pixels")
        try:
            import cv2
        except ImportError as exc:
            raise ProtocolUnavailableError("group-basicsr-cpu SSIM requires OpenCV") from exc
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
        one, two = first.astype(np.float64), second.astype(np.float64)
        mu1 = cv2.filter2D(one, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(two, -1, window)[5:-5, 5:-5]
        sigma1 = cv2.filter2D(one ** 2, -1, window)[5:-5, 5:-5] - mu1 ** 2
        sigma2 = cv2.filter2D(two ** 2, -1, window)[5:-5, 5:-5] - mu2 ** 2
        sigma12 = cv2.filter2D(one * two, -1, window)[5:-5, 5:-5] - mu1 * mu2
        c1, c2 = (0.01 * 255.0) ** 2, (0.03 * 255.0) ** 2
        return float(np.mean(((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) /
                             ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))))


@dataclass(frozen=True)
class SampleEvaluation:
    """One C-05 measurement and its exact metric representation provenance."""

    sample_id: str
    psnr: float
    ssim: float
    protocol: MetricProtocol
    model_metadata: dict[str, Any] | None = None
    measurement_kind: str = "group-measured"

    def to_dict(self) -> dict[str, Any]:
        """Serialize infinity explicitly, keeping JSON metadata unambiguous."""
        return {
            "sample_id": self.sample_id,
            "psnr": "inf" if math.isinf(self.psnr) else self.psnr,
            "ssim": self.ssim,
            "measurement_kind": self.measurement_kind,
            "protocol": self.protocol.to_dict(),
            "model_metadata": self.model_metadata,
        }


@dataclass(frozen=True)
class EvaluationSummary:
    """Ordered batch aggregate with explicit non-GT exclusions."""

    evaluations: tuple[SampleEvaluation, ...]
    excluded: tuple[dict[str, str], ...] = field(default_factory=tuple)

    @property
    def valid_count(self) -> int:
        return len(self.evaluations)

    @property
    def excluded_count(self) -> int:
        return len(self.excluded)

    @property
    def mean_psnr(self) -> float | None:
        return None if not self.evaluations else float(np.mean([item.psnr for item in self.evaluations]))

    @property
    def mean_ssim(self) -> float | None:
        return None if not self.evaluations else float(np.mean([item.ssim for item in self.evaluations]))

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready aggregate without converting a valid infinite PSNR to null."""
        return {
            "valid_count": self.valid_count,
            "excluded_count": self.excluded_count,
            "mean_psnr": None if self.mean_psnr is None else ("inf" if math.isinf(self.mean_psnr) else self.mean_psnr),
            "mean_ssim": self.mean_ssim,
            "evaluations": [item.to_dict() for item in self.evaluations],
            "excluded": list(self.excluded),
        }


def _as_chw(value: Any, *, context: str) -> np.ndarray:
    """Validate C-02 invariants and return a detached NumPy view/copy."""
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        array = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        array = value
    else:
        raise EvaluationInputError(f"{context}: expected RGB float32 CHW tensor")
    if array.dtype != np.float32 or array.ndim != 3 or array.shape[0] != 3 or any(size <= 0 for size in array.shape):
        raise EvaluationInputError(f"{context}: expected nonempty float32 CHW [3,H,W], got {array.shape} {array.dtype}")
    if not np.isfinite(array).all():
        raise EvaluationInputError(f"{context}: non-finite values (NaN/Inf)")
    if float(array.min()) < 0.0 or float(array.max()) > 1.0:
        raise EvaluationInputError(f"{context}: expected range [0,1]")
    return array


def _to_hwc_bgr(value: Any, *, context: str) -> np.ndarray:
    chw = _as_chw(value, context=context)
    return _chw_to_hwc_bgr_uint8(chw)


def _chw_to_hwc_bgr_uint8(chw: np.ndarray) -> np.ndarray:
    """Apply the clamp/round conversion used by BasicSR tensor2img."""
    return np.rint(np.clip(chw, 0.0, 1.0).transpose(1, 2, 0)[..., ::-1] * 255.0).astype(np.uint8)


def _crop(one: np.ndarray, two: np.ndarray, border: int) -> tuple[np.ndarray, np.ndarray]:
    if border == 0:
        return one, two
    if one.shape[0] <= 2 * border or one.shape[1] <= 2 * border:
        raise MetricExecutionError(f"crop_border={border} leaves no pixels")
    return one[border:-border, border:-border, ...], two[border:-border, border:-border, ...]


def _to_y(image: np.ndarray) -> np.ndarray:
    # BasicSR's Y conversion expects BGR arrays in [0,255].
    return (image[..., 2:3] * 65.481 + image[..., 1:2] * 128.553 + image[..., :1] * 24.966) / 255.0 + 16.0


def _validate_metric_arrays(one: np.ndarray, two: np.ndarray, border: int, y_channel: bool) -> None:
    if one.shape != two.shape or one.ndim != 3 or one.shape[2] != 3:
        raise MetricExecutionError("metric arrays must be matched HWC three-channel images")
    if border < 0 or not isinstance(y_channel, bool):
        raise MetricExecutionError("invalid crop or Y-channel policy")


def _resolve_adapter(protocol: MetricProtocol, adapter: BasicSRAdapter | None) -> BasicSRAdapter:
    if protocol.protocol_id == GROUP_BASICSR_CPU.protocol_id:
        if adapter is not None:
            return adapter
        return CpuBasicSRAdapter()
    if protocol.protocol_id == UPSTREAM_BASICSR.protocol_id:
        if adapter is None:
            raise ProtocolUnavailableError(
                "upstream-basicsr requires an explicit CUDA-capable BasicSR adapter; no CPU fallback is selected"
            )
        return adapter
    raise ProtocolUnavailableError(protocol.unavailable_reason or f"Protocol {protocol.protocol_id!r} is unavailable")


def evaluate_sample(result: Any, sample: Any, protocol: str | MetricProtocol, *,
                    basicsr_adapter: BasicSRAdapter | None = None) -> SampleEvaluation:
    """Evaluate one aligned C-04 result against one C-03 sample with ground truth."""
    resolved = get_metric_protocol(protocol)
    if resolved.protocol_id == REALBLUR_OFFICIAL.protocol_id or not resolved.supported:
        raise ProtocolUnavailableError(resolved.unavailable_reason or "Protocol is unavailable")
    result_id, sample_id = getattr(result, "sample_id", None), getattr(sample, "sample_id", None)
    if not isinstance(result_id, str) or result_id != sample_id:
        raise EvaluationInputError(f"sample_id mismatch: prediction={result_id!r}, target={sample_id!r}")
    if not bool(getattr(sample, "has_ground_truth", False)) or getattr(sample, "target", None) is None:
        raise EvaluationInputError(f"sample {sample_id!r}: ground truth is required for evaluation")
    prediction = _as_chw(getattr(result, "restored_image", None), context=f"prediction {sample_id!r}")
    target = _as_chw(sample.target, context=f"target {sample_id!r}")
    if prediction.shape != target.shape:
        raise EvaluationInputError(f"sample {sample_id!r}: prediction shape {prediction.shape} != target shape {target.shape}")
    predicted_image = _to_hwc_bgr(prediction, context=f"prediction {sample_id!r}")
    target_image = _to_hwc_bgr(target, context=f"target {sample_id!r}")
    adapter = _resolve_adapter(resolved, basicsr_adapter)
    try:
        psnr = float(adapter.calculate_psnr(predicted_image, target_image, crop_border=resolved.crop_border,
                                            test_y_channel=resolved.test_y_channel))
        ssim = float(adapter.calculate_ssim(predicted_image, target_image, crop_border=resolved.crop_border,
                                            test_y_channel=resolved.test_y_channel,
                                            ssim3d=resolved.ssim_variant == "ssim3d=True"))
    except ProtocolUnavailableError:
        raise
    except Exception as exc:
        raise MetricExecutionError(f"metric execution failed for sample {sample_id!r}: {exc}") from exc
    if math.isnan(psnr) or not math.isfinite(ssim):
        raise MetricExecutionError(f"metric returned invalid result for sample {sample_id!r}: psnr={psnr}, ssim={ssim}")
    metadata = getattr(result, "model_metadata", None)
    return SampleEvaluation(sample_id, psnr, ssim, resolved, metadata if isinstance(metadata, dict) else None)


def evaluate_batch(results: Sequence[Any], samples: Sequence[Any], protocol: str | MetricProtocol, *,
                   basicsr_adapter: BasicSRAdapter | None = None) -> EvaluationSummary:
    """Evaluate matched ordered sequences; non-GT samples are explicit exclusions."""
    if isinstance(results, (str, bytes)) or isinstance(samples, (str, bytes)):
        raise EvaluationInputError("results and samples must be sequences, not strings")
    result_ids = [getattr(item, "sample_id", None) for item in results]
    sample_ids = [getattr(item, "sample_id", None) for item in samples]
    if any(not isinstance(item, str) or not item for item in result_ids + sample_ids):
        raise EvaluationInputError("every result and sample must have a non-empty string sample_id")
    if len(set(result_ids)) != len(result_ids) or len(set(sample_ids)) != len(sample_ids):
        raise EvaluationInputError("duplicate sample_id in evaluation batch")
    missing, excess = set(sample_ids) - set(result_ids), set(result_ids) - set(sample_ids)
    if missing or excess:
        raise EvaluationInputError(f"batch sample IDs differ: missing predictions={sorted(missing)!r}, excess predictions={sorted(excess)!r}")
    by_id = {getattr(item, "sample_id"): item for item in samples}
    evaluated: list[SampleEvaluation] = []
    excluded: list[dict[str, str]] = []
    for result in results:
        sample = by_id[getattr(result, "sample_id")]
        if not bool(getattr(sample, "has_ground_truth", False)) or getattr(sample, "target", None) is None:
            excluded.append({"sample_id": getattr(result, "sample_id"), "reason": "missing-ground-truth"})
            continue
        evaluated.append(evaluate_sample(result, sample, protocol, basicsr_adapter=basicsr_adapter))
    return EvaluationSummary(tuple(evaluated), tuple(excluded))
