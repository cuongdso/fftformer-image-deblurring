"""CPU source integration tests; no checkpoint, dataset, GPU or benchmark."""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from fftformer_lab.model import (  # noqa: E402
    ModelBackendError, ModelInputError, ModelSpec, build_model,
)

RUN_MODEL_TESTS = os.environ.get("FFTFORMER_RUN_MODEL_TESTS") == "1"
if RUN_MODEL_TESTS:
    import torch


@unittest.skipUnless(RUN_MODEL_TESTS, "requires FFTFORMER_RUN_MODEL_TESTS=1, torch, einops and upstream source")
class ModelAdapterCPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.spec = ModelSpec(architecture_file=REPO / "third_party/FFTformer/basicsr/models/archs/fftformer_arch.py")
        cls.adapter = build_model(cls.spec)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def test_constructor_metadata_and_eval_mode(self) -> None:
        metadata = self.adapter.metadata
        self.assertEqual(metadata.parameter_count, 16_560_474)
        self.assertEqual(metadata.state_dict_entry_count, 535)
        self.assertEqual(metadata.resolved_device, "cpu")
        self.assertFalse(self.adapter.model.training)
        self.assertTrue(all(not module.training for module in self.adapter.model.modules()))
        self.assertFalse(metadata.checkpoint_loaded)
        self.assertIsNone(metadata.checkpoint_sha256)
        self.assertEqual(metadata.initialization, "random")
        second = build_model(self.spec)
        self.assertEqual(second.metadata.to_dict(), metadata.to_dict())

    def test_valid_shapes_including_batch_and_rectangle(self) -> None:
        for shape in ((1, 3, 32, 32), (1, 3, 64, 96), (2, 3, 32, 32)):
            with self.subTest(shape=shape), torch.inference_mode():
                image = torch.zeros(shape, dtype=torch.float32)
                result = self.adapter(image)
                self.assertEqual(tuple(result.shape), shape)
                self.assertTrue(bool(torch.isfinite(result).all()))

    def test_forward_preserves_upstream_output_and_input(self) -> None:
        image = torch.linspace(0, 1, 3 * 32 * 32).reshape(1, 3, 32, 32)
        original = image.clone()
        with torch.inference_mode():
            expected = self.adapter.model(image)
            actual = self.adapter(image)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(image, original, rtol=0, atol=0)

    def test_invalid_shape_is_rejected_before_upstream(self) -> None:
        for shape in ((3, 32, 32), (1, 1, 32, 32), (1, 4, 32, 32),
                      (0, 3, 32, 32), (1, 3, 0, 32), (1, 3, 40, 40)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ModelInputError, "shape"):
                self.adapter(torch.zeros(shape))

    def test_wrong_type_and_dtype_are_rejected(self) -> None:
        with self.assertRaisesRegex(ModelInputError, "torch.Tensor"):
            self.adapter([[0]])
        for dtype in (torch.float64, torch.float16, torch.uint8):
            with self.subTest(dtype=dtype), self.assertRaisesRegex(ModelInputError, "dtype"):
                self.adapter(torch.zeros(1, 3, 32, 32, dtype=dtype))

    def test_non_finite_and_out_of_range_inputs_are_rejected(self) -> None:
        for value, message in ((float("nan"), "finite"), (float("inf"), "finite"),
                               (-0.1, "range"), (1.1, "range")):
            with self.subTest(value=value), self.assertRaisesRegex(ModelInputError, message):
                self.adapter(torch.full((1, 3, 32, 32), value))

    def test_wrong_device_is_rejected(self) -> None:
        with self.assertRaisesRegex(ModelInputError, "device"):
            self.adapter(torch.empty(1, 3, 32, 32, device="meta"))

    def test_unavailable_cuda_does_not_fall_back_to_cpu(self) -> None:
        spec = ModelSpec(architecture_file=self.spec.architecture_file, device="cuda")
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ModelBackendError, "CUDA is unavailable"):
                build_model(spec)

    def test_cuda_index_is_validated_before_model_construction(self) -> None:
        spec = ModelSpec(architecture_file=self.spec.architecture_file, device="cuda:3")
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=1):
            with self.assertRaisesRegex(ModelBackendError, "index 3"):
                build_model(spec)

    def test_invalid_model_output_is_reported(self) -> None:
        image = torch.zeros(1, 3, 32, 32)
        for output in (None, torch.zeros(1, 3, 16, 16), image.double(), torch.full_like(image, float("nan"))):
            with self.subTest(output_type=type(output).__name__), patch.object(self.adapter.model, "forward", return_value=output):
                with self.assertRaisesRegex(ModelBackendError, "output"):
                    self.adapter(image)

    def test_output_is_not_clamped_or_normalized(self) -> None:
        image = torch.zeros(1, 3, 32, 32)
        output = torch.full_like(image, 1.5)
        with patch.object(self.adapter.model, "forward", return_value=output):
            self.assertIs(self.adapter(image), output)

    def test_forward_failure_includes_shape_and_device(self) -> None:
        with patch.object(self.adapter.model, "forward", side_effect=RuntimeError("test forward failure")):
            with self.assertRaisesRegex(ModelBackendError, r"cpu, input \(1, 3, 32, 32\)"):
                self.adapter(torch.zeros(1, 3, 32, 32))


if __name__ == "__main__":
    unittest.main()
