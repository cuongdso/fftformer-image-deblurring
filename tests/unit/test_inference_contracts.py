"""WS3 (M1-WS3-02) C-04 contract tests: output validation, device handling, batch error context.

Requires PyTorch (CPU is enough). Without torch these tests are skipped, as in test_inference_pipeline.py.
Only the mock model and in-test callables are used; no FFTformer checkpoint is involved.
"""

from pathlib import Path
from types import SimpleNamespace
import json
import sys
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    HAS_TORCH = False

from fftformer_lab.inference import (  # noqa: E402
    InferenceExecutionError,
    InferenceInputError,
    InferencePaddingError,
    InferencePipeline,
    InferenceRequest,
    MockModel,
    unpad_image,
    PaddingMetadata,
)


class _CallableModel:
    """Plain callable (no metadata): device defaults to cpu, not a mock."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, x):
        return self.fn(x)


class _FixedDeviceModel:
    """Mimics the WS1 ModelAdapter metadata surface: device is decided by the model."""

    def __init__(self, device: str) -> None:
        self.metadata = SimpleNamespace(
            resolved_device=device,
            architecture_id="fake_arch",
            implementation_version="v0",
            to_dict=lambda: {"resolved_device": device},
        )

    def __call__(self, x):
        return x


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class InferenceContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.img = torch.rand(3, 45, 55, dtype=torch.float32)

    # ---- model output validation (C-04: never crop a wrong output silently)
    def test_non_finite_model_output_is_rejected(self) -> None:
        def nan_model(x):
            out = x.clone()
            out[0, 0, 0, 0] = float("nan")
            return out

        with self.assertRaisesRegex(InferenceExecutionError, "non-finite"):
            InferencePipeline(_CallableModel(nan_model)).run_single(self.img)

    def test_wrong_spatial_output_shape_is_rejected(self) -> None:
        def cropped_model(x):
            return x[:, :, :-1, :]

        with self.assertRaisesRegex(InferenceExecutionError, "expected"):
            InferencePipeline(_CallableModel(cropped_model)).run_single(self.img)

    def test_batch_dimension_in_output_is_rejected(self) -> None:
        def doubled_model(x):
            return torch.cat([x, x], dim=0)

        with self.assertRaisesRegex(InferenceExecutionError, "expected"):
            InferencePipeline(_CallableModel(doubled_model)).run_single(self.img)

    def test_wrong_output_dtype_is_rejected(self) -> None:
        def double_model(x):
            return x.double()

        with self.assertRaisesRegex(InferenceExecutionError, "float32"):
            InferencePipeline(_CallableModel(double_model)).run_single(self.img)

    def test_non_tensor_output_is_rejected(self) -> None:
        with self.assertRaisesRegex(InferenceExecutionError, "expected torch.Tensor"):
            InferencePipeline(_CallableModel(lambda x: x.numpy())).run_single(self.img)

    # ---- padding and values
    def test_too_small_image_raises_padding_error(self) -> None:
        with self.assertRaises(InferencePaddingError):
            InferencePipeline(MockModel()).run_single(torch.rand(3, 10, 10))

    def test_invert_values_survive_padding_and_crop(self) -> None:
        res = InferencePipeline(MockModel(mode="invert")).run_single(self.img)
        self.assertEqual(tuple(res.restored_image.shape), (3, 45, 55))
        self.assertEqual(res.restored_image.dtype, torch.float32)
        self.assertTrue(torch.allclose(res.restored_image, 1.0 - self.img, atol=1e-6))

    def test_result_has_positive_elapsed_time_and_cpu_device(self) -> None:
        res = InferencePipeline(MockModel()).run_single(self.img)
        self.assertEqual(res.device, "cpu")
        self.assertIsNotNone(res.elapsed_time_seconds)
        self.assertGreater(res.elapsed_time_seconds, 0.0)

    # ---- request / sample identity
    def test_invalid_request_type_is_rejected(self) -> None:
        with self.assertRaisesRegex(InferenceInputError, "Unsupported request type"):
            InferencePipeline(MockModel()).run_single("not an image")

    def test_sample_id_fallbacks(self) -> None:
        pipeline = InferencePipeline(MockModel())
        self.assertEqual(pipeline.run_single(self.img).sample_id, "sample")
        req = InferenceRequest(image=self.img, sample_id="req_a")
        self.assertEqual(pipeline.run_single(req).sample_id, "req_a")
        self.assertEqual(pipeline.run_single(req, sample_id="override").sample_id, "override")

    # ---- batch orchestration
    def test_batch_empty_returns_empty_list(self) -> None:
        self.assertEqual(InferencePipeline(MockModel()).run_batch([]), [])

    def test_batch_rejects_string_and_stacked_tensor(self) -> None:
        pipeline = InferencePipeline(MockModel())
        with self.assertRaisesRegex(InferenceInputError, "Expected sequence"):
            pipeline.run_batch("abc")
        with self.assertRaisesRegex(InferenceInputError, "Expected sequence"):
            pipeline.run_batch(torch.rand(2, 3, 32, 32))

    def test_batch_mixed_identity_keeps_order_and_ids(self) -> None:
        reqs = [InferenceRequest(image=self.img, sample_id="a"), self.img]
        results = InferencePipeline(MockModel()).run_batch(reqs)
        self.assertEqual([r.sample_id for r in results], ["a", "sample_0001"])

    def test_batch_error_reports_failing_index(self) -> None:
        calls = {"n": 0}

        def fails_on_second(x):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("boom")
            return x.clone()

        with self.assertRaisesRegex(InferenceExecutionError, "batch index 1"):
            InferencePipeline(_CallableModel(fails_on_second)).run_batch([self.img, self.img])

    # ---- model state and device (C-01)
    def test_training_mode_module_is_rejected_until_eval(self) -> None:
        module = torch.nn.Identity()
        self.assertTrue(module.training)
        with self.assertRaisesRegex(InferenceExecutionError, "training mode"):
            InferencePipeline(module)
        module.eval()
        res = InferencePipeline(module).run_single(self.img)
        self.assertEqual(res.output_shape, (3, 45, 55))

    def test_device_override_conflicting_with_model_device_is_rejected(self) -> None:
        # No CUDA is needed: the conflict is detected from the device strings alone.
        with self.assertRaisesRegex(InferenceInputError, "conflicts"):
            InferencePipeline(_FixedDeviceModel("cpu"), device="cuda:0")

    def test_fixed_device_metadata_is_used(self) -> None:
        pipeline = InferencePipeline(_FixedDeviceModel("cpu"))
        self.assertEqual(pipeline.resolved_device, "cpu")
        self.assertEqual(pipeline.model_id, "fake_arch:v0")

    def test_device_override_allowed_for_plain_callable(self) -> None:
        pipeline = InferencePipeline(_CallableModel(lambda x: x), device="cpu")
        self.assertEqual(pipeline.resolved_device, "cpu")

    def test_invalid_device_string_is_rejected(self) -> None:
        with self.assertRaisesRegex(InferenceInputError, "Invalid device"):
            InferencePipeline(_CallableModel(lambda x: x), device="not_a_device")

    # ---- serialization and metadata isolation (C-04)
    def test_result_to_dict_is_json_serializable(self) -> None:
        res = InferencePipeline(MockModel()).run_single(self.img)
        payload = json.loads(json.dumps(res.to_dict()))
        self.assertEqual(payload["sample_id"], "sample")
        self.assertEqual(payload["output_shape"], [3, 45, 55])
        self.assertNotIn("restored_image", payload)

    def test_model_metadata_copy_is_isolated(self) -> None:
        pipeline = InferencePipeline(MockModel())
        res = pipeline.run_single(self.img)
        res.model_metadata["mode"] = "tampered"  # type: ignore[index]
        self.assertEqual(pipeline.model_metadata["mode"], "identity")  # type: ignore[index]

    def test_unpad_rejects_five_dimensional_tensor(self) -> None:
        pad = PaddingMetadata(bottom=0, right=0, original_shape=(32, 32), padded_shape=(32, 32))
        with self.assertRaisesRegex(InferencePaddingError, "ndim"):
            unpad_image(torch.rand(1, 1, 3, 32, 32), pad)


if __name__ == "__main__":
    unittest.main()
