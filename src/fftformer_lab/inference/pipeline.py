"""Deterministic single-image and batch inference pipeline adhering to C-04."""

from collections.abc import Sequence
import copy
import time
from typing import Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore

from .padding import calculate_reflection_padding, pad_image, unpad_image
from .types import (
    InferenceError,
    InferenceExecutionError,
    InferenceInputError,
    InferenceRequest,
    InferenceResult,
    PaddingMetadata,
)

_FACTOR = 32  # upstream FFTformer requires H and W to be multiples of 32 (C-02)


def validate_input_tensor(image: Any) -> None:
    """Validate that an input tensor strictly complies with C-02 specifications."""
    if torch is None:
        raise InferenceError("Input validation requires PyTorch")
    if not isinstance(image, torch.Tensor):
        raise InferenceInputError(f"Expected torch.Tensor, got {type(image).__name__}")
    if image.dtype != torch.float32:
        raise InferenceInputError(f"Expected torch.float32 dtype, got {image.dtype}")
    if image.ndim != 3:
        raise InferenceInputError(
            f"Expected 3D tensor [3, H, W] for a single sample at inference boundary, got ndim={image.ndim}"
        )
    if image.shape[0] != 3:
        raise InferenceInputError(f"Expected 3 channels (RGB), got {image.shape[0]} channels (shape={tuple(image.shape)})")
    if image.shape[1] <= 0 or image.shape[2] <= 0:
        raise InferenceInputError(f"Spatial dimensions must be positive, got {tuple(image.shape)}")
    if not bool(torch.isfinite(image).all()):
        raise InferenceInputError("Input image contains non-finite values (NaN or Inf)")
    if bool((image < 0.0).any()) or bool((image > 1.0).any()):
        raise InferenceInputError("Input image values must be in range [0, 1]; explicit normalization required")


def _canonical_device(value: Any) -> str:
    """Return a canonical device string so that 'cuda' and 'cuda:<current>' compare equal."""
    try:
        device = torch.device(value)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise InferenceInputError(f"Invalid device {value!r}: {exc}") from exc
    if device.type == "cuda" and device.index is None and torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
    return str(device)


def _is_training(model: Any) -> bool:
    """True if the model, or a wrapped ``.model`` (e.g. ModelAdapter), is an nn.Module in train mode."""
    candidates = [model]
    inner = getattr(model, "model", None)
    if inner is not None:
        candidates.append(inner)
    return any(isinstance(c, torch.nn.Module) and c.training for c in candidates)


def _module_device(model: Any) -> str | None:
    """Device of an nn.Module's parameters, or None if it has no parameters."""
    for parameter in model.parameters():
        return str(parameter.device)
    return None


class InferencePipeline:
    """Inference orchestrator for single-image and batch deblurring (Contract C-04).

    The pipeline does not read or write image files (that belongs to the C-02 boundary).
    It accepts either a WS1 ``ModelAdapter`` (device fixed by its metadata), a ``MockModel``
    (test double, ``is_mock=True``), or a plain callable/``nn.Module`` in evaluation mode.
    """

    def __init__(self, model: Any, device: str | None = None) -> None:
        if torch is None:
            raise InferenceError("InferencePipeline requires PyTorch")
        if _is_training(model):
            raise InferenceExecutionError(
                "Model is in training mode; call model.eval() before inference (C-01 requires evaluation mode)"
            )
        self.model = model
        self.is_mock = bool(getattr(model, "is_mock", False))

        # fixed_device is set only when the model itself determines its device (ModelAdapter, nn.Module).
        fixed_device: str | None = None
        default_device = "cpu"
        if hasattr(model, "metadata"):
            meta = model.metadata
            if hasattr(meta, "resolved_device"):
                fixed_device = str(meta.resolved_device)
                self.model_id = f"{meta.architecture_id}:{meta.implementation_version}"
                self.model_metadata = copy.deepcopy(meta.to_dict())
            elif isinstance(meta, dict):
                default_device = str(meta.get("device", "cpu"))
                self.model_id = str(meta.get("architecture_id", "mock_model"))
                self.model_metadata = copy.deepcopy(meta)
            else:
                self.model_id = "custom_model"
                self.model_metadata = None
        else:
            if isinstance(model, torch.nn.Module):
                fixed_device = _module_device(model) or "cpu"
            else:
                default_device = str(getattr(model, "device", "cpu"))
            self.model_id = getattr(model, "__name__", "custom_model")
            self.model_metadata = None

        resolved = _canonical_device(fixed_device if fixed_device is not None else default_device)
        if device is not None:
            requested = _canonical_device(device)
            if fixed_device is not None and requested != resolved:
                raise InferenceInputError(
                    f"Requested device {device!r} conflicts with the model's device {resolved!r}; "
                    "a model's device cannot be overridden by the pipeline"
                )
            resolved = requested
        self.resolved_device: str = resolved
        self._device = torch.device(resolved)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _unpack_request(request: Any, sample_id: str | None) -> tuple[Any, str]:
        """Extract (image, sample_id) from InferenceRequest, C-03-like sample, or a raw tensor."""
        if isinstance(request, InferenceRequest):
            image, default_sid = request.image, request.sample_id
        elif hasattr(request, "image"):
            image, default_sid = request.image, getattr(request, "sample_id", None)
        elif isinstance(request, torch.Tensor):
            image, default_sid = request, None
        else:
            raise InferenceInputError(
                f"Unsupported request type: {type(request).__name__}; expected InferenceRequest, Sample or torch.Tensor"
            )
        return image, sample_id or default_sid or "sample"

    @staticmethod
    def _synchronize(device: "torch.device") -> None:
        """Wait for queued CUDA work so that elapsed time reflects real compute."""
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # --------------------------------------------------------------- public API
    def run_single(
        self,
        request: InferenceRequest | Any,
        sample_id: str | None = None,
    ) -> InferenceResult:
        """Run deblurring inference on a single image adhering to C-04."""
        if torch is None:
            raise InferenceError("Inference execution requires PyTorch")

        image, sid = self._unpack_request(request, sample_id)
        validate_input_tensor(image)
        orig_h, orig_w = image.shape[-2:]

        # Reflection-pad right/bottom to a multiple of 32 (C-02); metadata is kept for the crop.
        padding: PaddingMetadata = calculate_reflection_padding(orig_h, orig_w, factor=_FACTOR)
        padded_chw = pad_image(image, padding)
        padded_nchw = padded_chw.unsqueeze(0)  # explicit CHW -> NCHW with batch size 1
        expected_out_shape = (1, 3, padding.padded_shape[0], padding.padded_shape[1])

        try:
            padded_nchw = padded_nchw.to(self._device)
        except (RuntimeError, TypeError) as exc:
            raise InferenceExecutionError(
                f"Cannot move sample '{sid}' to device {self.resolved_device}: {exc}"
            ) from exc

        self._synchronize(self._device)
        start_time = time.perf_counter()
        try:
            with torch.no_grad():
                output_nchw = self.model(padded_nchw)
        except Exception as exc:
            raise InferenceExecutionError(
                f"Model execution failed on device {self.resolved_device} for sample '{sid}': {exc}"
            ) from exc
        self._synchronize(self._device)
        elapsed_time = time.perf_counter() - start_time

        # Validate the model output before any crop: a wrong shape must never be cropped silently.
        if not isinstance(output_nchw, torch.Tensor):
            raise InferenceExecutionError(
                f"Model returned {type(output_nchw).__name__}, expected torch.Tensor for sample '{sid}'"
            )
        if tuple(output_nchw.shape) != expected_out_shape:
            raise InferenceExecutionError(
                f"Model returned shape {tuple(output_nchw.shape)} for sample '{sid}'; "
                f"expected {expected_out_shape} (NCHW, same spatial size as the padded input)"
            )
        if output_nchw.dtype != torch.float32:
            raise InferenceExecutionError(
                f"Model returned dtype {output_nchw.dtype} for sample '{sid}'; C-02 requires torch.float32"
            )
        if not bool(torch.isfinite(output_nchw).all()):
            raise InferenceExecutionError(f"Model returned non-finite values (NaN/Inf) for sample '{sid}'")

        output_chw = output_nchw.squeeze(0)
        restored = unpad_image(output_chw, padding)  # crop back to the original H x W
        restored = torch.clamp(restored, 0.0, 1.0)   # clamp policy per C-02/C-04

        return InferenceResult(
            restored_image=restored,
            sample_id=sid,
            input_shape=tuple(image.shape),
            padded_shape=tuple(padded_chw.shape),
            output_shape=tuple(restored.shape),
            padding=padding,
            device=self.resolved_device,
            model_id=self.model_id,
            is_mock=self.is_mock,
            elapsed_time_seconds=elapsed_time,
            model_metadata=copy.deepcopy(self.model_metadata),
        )

    def run_batch(self, requests: Sequence[Any]) -> list[InferenceResult]:
        """Process a sequence of samples sequentially, preserving sample order (C-04).

        Each element is an independent single-sample request; no batching of images is performed.
        Failures are re-raised with the batch index so the failing sample can be identified.
        """
        if isinstance(requests, (str, bytes)) or not isinstance(requests, Sequence):
            raise InferenceInputError(f"Expected sequence of requests, got {type(requests).__name__}")

        results: list[InferenceResult] = []
        for idx, req in enumerate(requests):
            has_identity = isinstance(req, InferenceRequest) or hasattr(req, "sample_id")
            default_sid = None if has_identity else f"sample_{idx:04d}"
            try:
                results.append(self.run_single(req, sample_id=default_sid))
            except InferenceError as exc:
                raise type(exc)(f"batch index {idx}: {exc}") from exc
        return results
