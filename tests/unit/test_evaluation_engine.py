"""Deterministic correctness and contract tests for the WS4-02 evaluation engine."""

from dataclasses import dataclass
import math
import unittest

import numpy as np

from fftformer_lab.evaluation import engine
from fftformer_lab.evaluation import (
    EvaluationInputError,
    GROUP_BASICSR_CPU,
    ProtocolUnavailableError,
    REALBLUR_OFFICIAL,
    UPSTREAM_BASICSR,
    evaluate_batch,
    evaluate_sample,
)


@dataclass
class _Result:
    restored_image: np.ndarray
    sample_id: str = "sample"
    model_metadata: dict[str, object] | None = None


@dataclass
class _Sample:
    target: np.ndarray | None
    sample_id: str = "sample"
    has_ground_truth: bool = True


class _CapturingAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, np.ndarray, dict[str, object]]] = []

    def calculate_psnr(self, first, second, **kwargs):
        self.calls.append(("psnr", first.copy(), kwargs))
        return 20.0

    def calculate_ssim(self, first, second, **kwargs):
        self.calls.append(("ssim", first.copy(), kwargs))
        return 0.75


def _image(value: float = 0.0, *, size: int = 16) -> np.ndarray:
    return np.full((3, size, size), value, dtype=np.float32)


class EvaluationEngineTests(unittest.TestCase):
    def test_identical_images_have_infinite_psnr_and_unit_ssim(self) -> None:
        evaluation = evaluate_sample(_Result(_image(0.25)), _Sample(_image(0.25)), GROUP_BASICSR_CPU)
        self.assertTrue(math.isinf(evaluation.psnr))
        self.assertAlmostEqual(evaluation.ssim, 1.0, places=12)
        self.assertEqual(evaluation.to_dict()["psnr"], "inf")

    def test_psnr_matches_independent_closed_form_reference(self) -> None:
        prediction, target = _image(0.0), _image(1.0)
        evaluation = evaluate_sample(_Result(prediction), _Sample(target), GROUP_BASICSR_CPU)
        self.assertAlmostEqual(evaluation.psnr, 0.0, places=12)

    def test_cpu_ssim_has_deterministic_golden_reference(self) -> None:
        prediction, target = _image(0.0), _image(0.0)
        prediction[:, 4:12, 4:12] = 1.0
        evaluation = evaluate_sample(_Result(prediction), _Sample(target), GROUP_BASICSR_CPU)
        # Golden value was calculated from the audited BasicSR non-3D Gaussian-window formula.
        self.assertAlmostEqual(evaluation.ssim, 1.6063435018736121e-06, places=12)

    def test_basic_sr_representation_is_bgr_uint8_with_rounding_and_clip(self) -> None:
        prediction = _image(0.0)
        prediction[0, 0, 0], prediction[1, 0, 0], prediction[2, 0, 0] = 0.1, 0.5, 0.9
        adapter = _CapturingAdapter()
        evaluate_sample(_Result(prediction), _Sample(prediction.copy()), UPSTREAM_BASICSR, basicsr_adapter=adapter)
        _, image, kwargs = adapter.calls[0]
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(image.shape, (16, 16, 3))
        self.assertEqual(image[0, 0].tolist(), [230, 128, 26])
        self.assertEqual(kwargs, {"crop_border": 0, "test_y_channel": False})
        self.assertEqual(adapter.calls[1][2], {"crop_border": 0, "test_y_channel": False, "ssim3d": True})

    def test_internal_conversion_clips_before_rounding(self) -> None:
        raw = np.zeros((3, 1, 1), dtype=np.float32)
        raw[:, 0, 0] = [-0.2, 0.5, 1.2]
        self.assertEqual(engine._chw_to_hwc_bgr_uint8(raw)[0, 0].tolist(), [255, 128, 0])

    def test_upstream_requires_explicit_adapter_without_fallback(self) -> None:
        with self.assertRaises(ProtocolUnavailableError):
            evaluate_sample(_Result(_image()), _Sample(_image()), UPSTREAM_BASICSR)

    def test_realblur_is_unavailable_not_substituted(self) -> None:
        with self.assertRaises(ProtocolUnavailableError):
            evaluate_sample(_Result(_image()), _Sample(_image()), REALBLUR_OFFICIAL)

    def test_missing_gt_and_identity_mismatch_are_rejected(self) -> None:
        with self.assertRaises(EvaluationInputError):
            evaluate_sample(_Result(_image()), _Sample(None, has_ground_truth=False), GROUP_BASICSR_CPU)
        with self.assertRaises(EvaluationInputError):
            evaluate_sample(_Result(_image(), "result"), _Sample(_image(), "sample"), GROUP_BASICSR_CPU)

    def test_shape_dtype_range_channel_and_nonfinite_are_rejected(self) -> None:
        valid = _image()
        invalids = [
            np.zeros((2, 16, 16), dtype=np.float32),
            np.zeros((3, 16, 16), dtype=np.float64),
            np.full((3, 16, 16), 1.1, dtype=np.float32),
            np.full((3, 16, 16), np.nan, dtype=np.float32),
        ]
        for invalid in invalids:
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype):
                with self.assertRaises(EvaluationInputError):
                    evaluate_sample(_Result(invalid), _Sample(valid), GROUP_BASICSR_CPU)
        with self.assertRaises(EvaluationInputError):
            evaluate_sample(_Result(valid), _Sample(np.zeros((3, 15, 16), dtype=np.float32)), GROUP_BASICSR_CPU)

    def test_batch_preserves_result_order_and_records_non_gt_exclusion(self) -> None:
        results = [_Result(_image(), "b"), _Result(_image(), "a"), _Result(_image(), "c")]
        samples = [_Sample(_image(), "a"), _Sample(_image(), "b"), _Sample(None, "c", False)]
        summary = evaluate_batch(results, samples, GROUP_BASICSR_CPU)
        self.assertEqual([item.sample_id for item in summary.evaluations], ["b", "a"])
        self.assertEqual(summary.valid_count, 2)
        self.assertEqual(summary.excluded, ({"sample_id": "c", "reason": "missing-ground-truth"},))
        self.assertTrue(math.isinf(summary.mean_psnr or 0.0))

    def test_batch_rejects_duplicate_missing_and_excess_ids(self) -> None:
        with self.assertRaises(EvaluationInputError):
            evaluate_batch([_Result(_image(), "a"), _Result(_image(), "a")], [_Sample(_image(), "a")], GROUP_BASICSR_CPU)
        with self.assertRaises(EvaluationInputError):
            evaluate_batch([_Result(_image(), "a")], [_Sample(_image(), "b")], GROUP_BASICSR_CPU)
        with self.assertRaises(EvaluationInputError):
            evaluate_batch([_Result(_image(), "")], [_Sample(_image(), "")], GROUP_BASICSR_CPU)
