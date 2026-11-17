"""Unit and pipeline integration tests for WS3 inference using MockModel."""

from dataclasses import dataclass
from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from fftformer_lab.inference import (  # noqa: E402
    InferenceExecutionError,
    InferenceInputError,
    InferencePaddingError,
    InferencePipeline,
    InferenceRequest,
    MockModel,
    calculate_reflection_padding,
    pad_image,
    unpad_image,
    validate_input_tensor,
)


@dataclass
class DummyDuckSample:
    """Mock Sample conforming to C-03 duck-typing."""
    image: "torch.Tensor"
    sample_id: str = "duck_sample_123"


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class InferencePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_model = MockModel(mode="identity", device="cpu")
        self.pipeline = InferencePipeline(self.mock_model)

    def test_validate_input_tensor_valid(self) -> None:
        valid_img = torch.rand(3, 64, 64, dtype=torch.float32)
        # Should not raise
        validate_input_tensor(valid_img)

    def test_validate_input_tensor_invalid_types_and_shapes(self) -> None:
        cases = [
            ("not a tensor", "Expected torch.Tensor"),
            (torch.rand(3, 64, 64, dtype=torch.float64), "Expected torch.float32 dtype"),
            (torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8), "Expected torch.float32 dtype"),
            (torch.rand(64, 64, dtype=torch.float32), "Expected 3D tensor"),
            (torch.rand(1, 3, 64, 64, dtype=torch.float32), "Expected 3D tensor"),
            (torch.rand(1, 64, 64, dtype=torch.float32), "Expected 3 channels"),
            (torch.rand(4, 64, 64, dtype=torch.float32), "Expected 3 channels"),
        ]
        for val, err_msg in cases:
            with self.subTest(val=val), self.assertRaisesRegex(InferenceInputError, err_msg):
                validate_input_tensor(val)

    def test_validate_input_tensor_out_of_range_and_non_finite(self) -> None:
        # Out of [0, 1] range
        below_zero = torch.tensor([[[-0.1]]], dtype=torch.float32).repeat(3, 32, 32)
        above_one = torch.tensor([[[1.05]]], dtype=torch.float32).repeat(3, 32, 32)
        with self.assertRaisesRegex(InferenceInputError, "range"):
            validate_input_tensor(below_zero)
        with self.assertRaisesRegex(InferenceInputError, "range"):
            validate_input_tensor(above_one)

        # Non-finite values
        has_nan = torch.rand(3, 32, 32, dtype=torch.float32)
        has_nan[0, 5, 5] = float("nan")
        with self.assertRaisesRegex(InferenceInputError, "non-finite"):
            validate_input_tensor(has_nan)

        has_inf = torch.rand(3, 32, 32, dtype=torch.float32)
        has_inf[0, 5, 5] = float("inf")
        with self.assertRaisesRegex(InferenceInputError, "non-finite"):
            validate_input_tensor(has_inf)

    def test_pad_and_unpad_roundtrip(self) -> None:
        # Image size 50x70 -> padded to 64x96
        img = torch.rand(3, 50, 70, dtype=torch.float32)
        padding = calculate_reflection_padding(50, 70, factor=32)
        padded = pad_image(img, padding)
        self.assertEqual(tuple(padded.shape), (3, 64, 96))

        unpadded = unpad_image(padded, padding)
        self.assertEqual(tuple(unpadded.shape), (3, 50, 70))
        # Round-trip unpadding must match the original content exactly
        self.assertTrue(torch.equal(img, unpadded))

    def test_mock_model_modes(self) -> None:
        img_nchw = torch.rand(1, 3, 32, 32, dtype=torch.float32)
        
        identity_model = MockModel(mode="identity")
        self.assertTrue(torch.equal(identity_model(img_nchw), img_nchw))

        invert_model = MockModel(mode="invert")
        inverted = invert_model(img_nchw)
        self.assertTrue(torch.allclose(inverted, 1.0 - img_nchw, atol=1e-6))

        shift_model = MockModel(mode="shift")
        shifted = shift_model(img_nchw)
        self.assertTrue((shifted >= 0.0).all() and (shifted <= 1.0).all())

    def test_mock_model_rejects_non_multiple_of_32(self) -> None:
        model = MockModel()
        bad_nchw = torch.rand(1, 3, 30, 32, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "multiples of 32"):
            model(bad_nchw)

    def test_pipeline_run_single_with_identity_mock(self) -> None:
        # Input shape 45x55 (not multiples of 32)
        img = torch.rand(3, 45, 55, dtype=torch.float32)
        res = self.pipeline.run_single(img, sample_id="sample_test_1")

        self.assertEqual(res.sample_id, "sample_test_1")
        self.assertEqual(res.input_shape, (3, 45, 55))
        self.assertEqual(res.padded_shape, (3, 64, 64))
        self.assertEqual(res.output_shape, (3, 45, 55))
        self.assertEqual(res.padding.original_shape, (45, 55))
        self.assertEqual(res.padding.padded_shape, (64, 64))
        self.assertTrue(res.is_mock)
        self.assertIsNotNone(res.elapsed_time_seconds)
        self.assertGreater(res.elapsed_time_seconds, 0.0)

        # Under identity mock, restored image matches original exactly
        self.assertTrue(torch.allclose(res.restored_image, img, atol=1e-6))

    def test_pipeline_run_single_supports_inference_request_and_duck_sample(self) -> None:
        img = torch.rand(3, 32, 32, dtype=torch.float32)
        req = InferenceRequest(image=img, sample_id="req_42")
        res1 = self.pipeline.run_single(req)
        self.assertEqual(res1.sample_id, "req_42")

        duck = DummyDuckSample(image=img, sample_id="duck_99")
        res2 = self.pipeline.run_single(duck)
        self.assertEqual(res2.sample_id, "duck_99")

    def test_pipeline_run_batch_deterministic_order(self) -> None:
        imgs = [
            torch.rand(3, 32, 32, dtype=torch.float32),
            torch.rand(3, 40, 50, dtype=torch.float32),
            torch.rand(3, 64, 32, dtype=torch.float32),
        ]
        results = self.pipeline.run_batch(imgs)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].sample_id, "sample_0000")
        self.assertEqual(results[1].sample_id, "sample_0001")
        self.assertEqual(results[2].sample_id, "sample_0002")

        self.assertEqual(results[0].output_shape, (3, 32, 32))
        self.assertEqual(results[1].output_shape, (3, 40, 50))
        self.assertEqual(results[2].output_shape, (3, 64, 32))

    def test_pipeline_propagates_model_execution_error(self) -> None:
        def broken_model(x: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("GPU out of memory simulation")

        broken_pipeline = InferencePipeline(broken_model)
        img = torch.rand(3, 32, 32, dtype=torch.float32)
        with self.assertRaisesRegex(InferenceExecutionError, "GPU out of memory simulation"):
            broken_pipeline.run_single(img)
