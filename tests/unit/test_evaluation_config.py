"""Standard-library C-06 config tests."""
import json
from pathlib import Path
import tempfile
import unittest

from fftformer_lab.evaluation import ConfigurationError, load_evaluation_config


def _document() -> dict:
    return {"schema_version": 1, "contract_version": "C-06-v1", "experiment_id": "unit-eval",
            "protocol_id": "group-basicsr-cpu", "dataset": {"prediction_root": "pred", "target_root": "gt"},
            "output_path": "out.json", "checkpoint": None, "model": None, "provenance": {"group_commit": None}}


class EvaluationConfigTests(unittest.TestCase):
    def write(self, root: Path, document: dict | None = None) -> Path:
        (root / "pred").mkdir(); (root / "gt").mkdir()
        path = root / "job.json"; path.write_text(json.dumps(document or _document()), encoding="utf-8")
        return path

    def test_valid_config_resolves_paths_relative_to_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config = load_evaluation_config(self.write(root))
            self.assertEqual(config.prediction_root, (root / "pred").resolve())
            self.assertEqual(config.output_path, (root / "out.json").resolve())
            with self.assertRaises(AttributeError): config.experiment_id = "other"  # type: ignore[misc]

    def test_rejects_unknown_schema_protocol_and_missing_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mutate in (lambda d: d.update({"schema_version": 2}), lambda d: d.update({"protocol_id": "unknown"}), lambda d: d["dataset"].update({"prediction_root": "none"}), lambda d: d.update({"extra": True})):
                document = _document(); mutate(document)
                path = root / f"{len(list(root.glob('*.json')))}.json"; path.write_text(json.dumps(document), encoding="utf-8")
                with self.assertRaises(ConfigurationError): load_evaluation_config(path)

    def test_duplicate_json_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "pred").mkdir(); (root / "gt").mkdir()
            path = root / "job.json"; path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
            with self.assertRaises(ConfigurationError): load_evaluation_config(path)
