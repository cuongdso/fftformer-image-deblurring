"""Dependency-free tests for WS1 specification, validation and source selection."""

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from fftformer_lab.model import (  # noqa: E402
    ModelBackendError, ModelConfig, ModelConfigurationError, ModelMetadata, ModelSpec, build_model,
)
from fftformer_lab.model.spec import UPSTREAM_COMMIT  # noqa: E402


class ModelSpecificationTests(unittest.TestCase):
    def test_default_config_matches_audited_gopro(self) -> None:
        config = ModelConfig.from_mapping({"type": "fftformer", "num_blocks": [6, 6, 12]})
        self.assertEqual(config, ModelConfig())
        self.assertEqual(config.dim, 48)
        self.assertEqual(config.num_refinement_blocks, 4)
        self.assertEqual(config.ffn_expansion_factor, 3)
        self.assertFalse(config.bias)

    def test_constructor_mapping_does_not_alias_mutable_input(self) -> None:
        blocks = [6, 6, 12]
        config = ModelConfig.from_mapping({"num_blocks": blocks})
        blocks[0] = 99
        exported = config.to_constructor_kwargs()
        exported["num_blocks"][1] = 99
        self.assertEqual(config.num_blocks, (6, 6, 12))
        with self.assertRaises(FrozenInstanceError):
            config.dim = 99

    def test_invalid_constructor_values_include_field_context(self) -> None:
        cases = [
            ("dim", 0), ("dim", True), ("dim", "48"),
            ("inp_channels", 1), ("out_channels", 4),
            ("num_refinement_blocks", -1), ("bias", "false"),
            ("ffn_expansion_factor", 0), ("ffn_expansion_factor", True),
            ("ffn_expansion_factor", float("nan")), ("ffn_expansion_factor", float("inf")),
            ("ffn_expansion_factor", 0.0001), ("ffn_expansion_factor", 1e308),
            ("ffn_expansion_factor", 10 ** 1000),
            ("num_blocks", [6, 6]), ("num_blocks", [6, 6, 12, 8]),
            ("num_blocks", [6, 0, 12]), ("num_blocks", [6, True, 12]),
        ]
        for name, value in cases:
            message = name if name not in ("inp_channels", "out_channels") else "RGB"
            with self.subTest(name=name, value=value), self.assertRaisesRegex(ModelConfigurationError, message):
                ModelConfig.from_mapping({name: value})

    def test_unknown_architecture_and_fields_are_rejected(self) -> None:
        for config in ({"type": "other"}, {"unknown": 123}, ["not-a-mapping"]):
            with self.subTest(config=config), self.assertRaises(ModelConfigurationError):
                ModelConfig.from_mapping(config)

    def test_spec_is_immutable_and_retains_checkpoint_request(self) -> None:
        spec = ModelSpec(architecture_file="architecture.py", checkpoint="weights.pth")
        self.assertEqual(spec.architecture_file, Path("architecture.py"))
        self.assertEqual(spec.checkpoint, Path("weights.pth"))
        self.assertEqual(spec.implementation_version, UPSTREAM_COMMIT)
        with self.assertRaises(FrozenInstanceError):
            spec.device = "cuda"

    def test_spec_rejects_unaudited_identity_and_wrong_config(self) -> None:
        for values in (
            {"architecture_id": "other"}, {"implementation_id": "unknown"},
            {"implementation_version": "main"}, {"config": {}},
        ):
            with self.subTest(values=values), self.assertRaises(ModelConfigurationError):
                ModelSpec(architecture_file="architecture.py", **values)

    def test_spec_rejects_invalid_paths(self) -> None:
        for field in ("architecture_file", "checkpoint"):
            for value in ("", " ", 123):
                values = {"architecture_file": "architecture.py", field: value}
                with self.subTest(values=values), self.assertRaisesRegex(ModelConfigurationError, field):
                    ModelSpec(**values)

    def test_devices_have_explicit_syntax(self) -> None:
        for device in ("cpu", "cuda", "cuda:0", "cuda:2"):
            self.assertEqual(ModelSpec(architecture_file="architecture.py", device=device).device, device)
        for device in ("auto", "mps", "cuda:-1", "cuda:01", "cpu:2", "", None):
            with self.subTest(device=device), self.assertRaisesRegex(ModelConfigurationError, "device"):
                ModelSpec(architecture_file="architecture.py", device=device)

    def test_non_strict_requests_are_rejected(self) -> None:
        for value in (False, 1, "True", None):
            with self.subTest(value=value), self.assertRaisesRegex(ModelConfigurationError, "strict"):
                ModelSpec(architecture_file="architecture.py", strict=value)

    def test_checkpoint_request_is_never_silently_ignored(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "M1-WS1-03"):
            build_model(ModelSpec(architecture_file="architecture.py", checkpoint="weights.pth"))

    def test_missing_or_changed_source_cannot_execute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "architecture.py"
            with self.assertRaisesRegex(ModelBackendError, "architecture_file"):
                build_model(ModelSpec(architecture_file=source))
            source.write_text("raise AssertionError('unverified source executed')", encoding="utf-8")
            with self.assertRaisesRegex(ModelBackendError, "SHA-256"):
                build_model(ModelSpec(architecture_file=source))

    def test_metadata_exports_json_and_independent_config(self) -> None:
        metadata = ModelMetadata("fftformer", "upstream-fftformer", UPSTREAM_COMMIT, "source-hash",
                                 ModelConfig(), "cpu", "cpu", 16_560_474, 535)
        exported = metadata.to_dict()
        exported["config"]["num_blocks"][0] = 99
        self.assertEqual(metadata.config.num_blocks, (6, 6, 12))
        self.assertFalse(json.loads(json.dumps(metadata.to_dict()))["checkpoint_loaded"])
        self.assertEqual(metadata.initialization, "random")

    def test_package_import_does_not_require_torch_or_einops(self) -> None:
        script = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from fftformer_lab.model import ModelSpec; "
            "assert 'torch' not in sys.modules and 'einops' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", script, str(REPO / "src")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
