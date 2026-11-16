"""Opt-in checkpoint tests: tiny mock fixtures and separately marked GoPro integration."""

from dataclasses import replace
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from fftformer_lab.model import (  # noqa: E402
    CheckpointCompatibilityError, CheckpointError, CheckpointSchemaError,
    ModelAdapter, ModelBackendError, ModelConfig, ModelConfigurationError, ModelMetadata, ModelSpec, load_model,
)

RUN_MODEL_TESTS = os.environ.get("FFTFORMER_RUN_MODEL_TESTS") == "1"
GOPRO_CHECKPOINT = os.environ.get("FFTFORMER_GOPRO_CHECKPOINT")
if RUN_MODEL_TESTS:
    import torch


@unittest.skipUnless(RUN_MODEL_TESTS, "requires FFTFORMER_RUN_MODEL_TESTS=1 and torch")
class MockCheckpointLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.checkpoint = Path(directory.name) / "mock-conv2d.pth"
        self.spec = ModelSpec(architecture_file="mock-source.py", checkpoint=self.checkpoint)
        model = torch.nn.Conv2d(3, 3, 1, bias=True)
        metadata = ModelMetadata(
            architecture_id="mock-conv2d", implementation_id="mock-fixture-only",
            implementation_version="fixture-v1", architecture_sha256="mock-source",
            config=ModelConfig(bias=True), requested_device="cpu", resolved_device="cpu",
            parameter_count=12, state_dict_entry_count=2,
        )
        self.mock_adapter = ModelAdapter(model, metadata)
        factory = patch("fftformer_lab.model.checkpoint.build_model", return_value=self.mock_adapter)
        self.mock_build = factory.start()
        self.addCleanup(factory.stop)
        self.state = {name: torch.full_like(value, 0.25) for name, value in model.state_dict().items()}

    def save(self, state: object) -> None:
        """Write a tiny generated fixture only inside the temporary test directory."""
        torch.save(state, self.checkpoint)

    def test_valid_fixture_strict_load_and_metadata(self) -> None:
        self.save(self.state)
        loaded = load_model(self.spec)
        for name, value in loaded.model.state_dict().items():
            torch.testing.assert_close(value, self.state[name], rtol=0, atol=0)
        self.assertFalse(loaded.model.training)
        self.assertTrue(loaded.metadata.checkpoint_loaded)
        self.assertEqual(loaded.metadata.initialization, "checkpoint")
        self.assertEqual(loaded.metadata.implementation_id, "mock-fixture-only")
        self.assertEqual(loaded.metadata.checkpoint_path, self.checkpoint.as_posix())
        self.assertEqual(loaded.metadata.checkpoint_schema, "raw_state_dict")
        self.assertEqual(loaded.metadata.checkpoint_map_location, "cpu")
        self.assertEqual(loaded.metadata.checkpoint_size_bytes, self.checkpoint.stat().st_size)
        self.assertEqual(loaded.metadata.checkpoint_sha256, hashlib.sha256(self.checkpoint.read_bytes()).hexdigest())
        self.assertIsNone(self.mock_build.call_args.args[0].checkpoint)
        self.assertEqual(self.spec.checkpoint, self.checkpoint)
        self.assertFalse(self.mock_adapter.metadata.checkpoint_loaded)

    def test_missing_file_has_context_before_construction(self) -> None:
        with self.assertRaisesRegex(CheckpointError, "cannot read file") as error:
            load_model(self.spec)
        self.assertIn("mock-conv2d.pth", str(error.exception))
        self.mock_build.assert_not_called()

    def test_directory_path_is_rejected(self) -> None:
        with self.assertRaisesRegex(CheckpointError, "cannot read file"):
            load_model(replace(self.spec, checkpoint=self.checkpoint.parent))

    def test_corrupt_file_has_deserialization_context(self) -> None:
        self.checkpoint.write_bytes(b"not a PyTorch checkpoint")
        with self.assertRaisesRegex(CheckpointError, "weights_only=True"):
            load_model(self.spec)

    def test_wrapper_schemas_are_not_guessed(self) -> None:
        for key in ("params", "params_ema", "state_dict"):
            with self.subTest(key=key):
                self.save({key: self.state})
                with self.assertRaisesRegex(CheckpointSchemaError, "wrapper schemas"):
                    load_model(self.spec)

    def test_non_tensor_empty_and_non_mapping_schemas(self) -> None:
        for state in ({}, [1, 2], {"weight": 123}, {42: torch.ones(1)}):
            with self.subTest(state_type=type(state).__name__):
                self.save(state)
                with self.assertRaises(CheckpointSchemaError):
                    load_model(self.spec)

    def test_missing_key_is_rejected_before_weights_are_mutated(self) -> None:
        self.save({"bias": self.state["bias"]})
        before = self.mock_adapter.model.weight.detach().clone()
        with self.assertRaisesRegex(CheckpointCompatibilityError, "missing keys.*weight"):
            load_model(self.spec)
        torch.testing.assert_close(self.mock_adapter.model.weight, before, rtol=0, atol=0)

    def test_unexpected_key_is_rejected(self) -> None:
        self.save({**self.state, "audit.unexpected.weight": torch.empty(1)})
        with self.assertRaisesRegex(CheckpointCompatibilityError, "unexpected keys.*audit.unexpected.weight"):
            load_model(self.spec)

    def test_wrong_shape_reports_name_actual_and_expected(self) -> None:
        self.save({**self.state, "weight": torch.zeros(2, 3, 1, 1)})
        with self.assertRaisesRegex(CheckpointCompatibilityError, r"weight: shape \(2, 3, 1, 1\).*\(3, 3, 1, 1\)"):
            load_model(self.spec)

    def test_wrong_dtype_is_not_silently_cast(self) -> None:
        self.save({**self.state, "weight": self.state["weight"].double()})
        with self.assertRaisesRegex(CheckpointCompatibilityError, "weight: dtype torch.float64.*torch.float32"):
            load_model(self.spec)

    def test_missing_checkpoint_request_is_rejected(self) -> None:
        with self.assertRaisesRegex(ModelConfigurationError, "checkpoint path is required"):
            load_model(replace(self.spec, checkpoint=None))

    def test_load_uses_resolved_device_weights_only_and_strict(self) -> None:
        self.save(self.state)
        with patch("torch.load", wraps=torch.load) as deserialize:
            with patch.object(self.mock_adapter.model, "load_state_dict", wraps=self.mock_adapter.model.load_state_dict) as strict_load:
                load_model(self.spec)
        self.assertEqual(deserialize.call_args.kwargs, {"map_location": "cpu", "weights_only": True})
        self.assertEqual(strict_load.call_args.kwargs, {"strict": True})

    def test_device_error_propagates_without_deserializing(self) -> None:
        self.save(self.state)
        self.mock_build.side_effect = ModelBackendError("CUDA unavailable (mock)")
        with patch("torch.load") as deserialize:
            with self.assertRaisesRegex(ModelBackendError, "CUDA unavailable"):
                load_model(replace(self.spec, device="cuda"))
        deserialize.assert_not_called()

    def test_checkpoint_file_is_not_modified_on_failure(self) -> None:
        self.save({**self.state, "weight": torch.zeros(2, 3, 1, 1)})
        before = self.checkpoint.read_bytes()
        with self.assertRaises(CheckpointCompatibilityError):
            load_model(self.spec)
        self.assertEqual(self.checkpoint.read_bytes(), before)


@unittest.skipUnless(RUN_MODEL_TESTS and GOPRO_CHECKPOINT,
                     "requires model opt-in and FFTFORMER_GOPRO_CHECKPOINT pointing to the audited artifact")
class GoProCheckpointIntegrationTests(unittest.TestCase):
    def test_caller_defaults_do_not_reject_audited_checkpoint(self) -> None:
        previous_dtype = torch.get_default_dtype()
        previous_device = torch.get_default_device()
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(2)
            torch.set_default_dtype(torch.float64)
            torch.set_default_device("meta")
            adapter = load_model(ModelSpec(
                architecture_file=REPO / "third_party/FFTformer/basicsr/models/archs/fftformer_arch.py",
                checkpoint=Path(GOPRO_CHECKPOINT), device="cpu",
            ))
            self.assertTrue(adapter.metadata.checkpoint_loaded)
            self.assertEqual(adapter.metadata.checkpoint_map_location, "cpu")
            self.assertEqual({value.dtype for value in adapter.model.state_dict().values()}, {torch.float32})
            self.assertEqual({str(value.device) for value in adapter.model.state_dict().values()}, {"cpu"})
            with torch.inference_mode():
                output = adapter(torch.zeros(1, 3, 32, 32, dtype=torch.float32, device="cpu"))
            self.assertTrue(bool(torch.isfinite(output).all()))
            self.assertEqual(torch.get_default_dtype(), torch.float64)
            self.assertEqual(str(torch.get_default_device()), "meta")
        finally:
            torch.set_default_dtype(previous_dtype)
            torch.set_default_device(previous_device)
            torch.set_num_threads(previous_threads)

    def test_changed_architecture_is_rejected(self) -> None:
        spec = ModelSpec(
            architecture_file=REPO / "third_party/FFTformer/basicsr/models/archs/fftformer_arch.py",
            checkpoint=Path(GOPRO_CHECKPOINT), config=ModelConfig(dim=32),
        )
        with self.assertRaisesRegex(CheckpointCompatibilityError, "tensor mismatches.*shape .*expected"):
            load_model(spec)

    def test_audited_gopro_load_metadata_and_forward(self) -> None:
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, previous_threads)
        spec = ModelSpec(
            architecture_file=REPO / "third_party/FFTformer/basicsr/models/archs/fftformer_arch.py",
            checkpoint=Path(GOPRO_CHECKPOINT), device="cpu",
        )
        adapter = load_model(spec)
        self.assertEqual(adapter.metadata.checkpoint_sha256,
                         "336717b7c4e97b66af112cc7878106cb185b698bd872c2a8247911507540b2a3")
        self.assertEqual(adapter.metadata.parameter_count, 16_560_474)
        self.assertEqual(adapter.metadata.state_dict_entry_count, 535)
        self.assertTrue(adapter.metadata.checkpoint_loaded)
        self.assertFalse(adapter.model.training)
        for shape in ((1, 3, 32, 32), (1, 3, 64, 96)):
            with self.subTest(shape=shape), torch.inference_mode():
                output = adapter(torch.zeros(shape, dtype=torch.float32))
                self.assertEqual(tuple(output.shape), shape)
                self.assertTrue(bool(torch.isfinite(output).all()))
