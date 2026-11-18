"""Metadata serialization tests without CV dependencies."""
import json
from pathlib import Path
import tempfile
import unittest

from fftformer_lab.evaluation import EvaluationSummary, GROUP_BASICSR_CPU, SampleEvaluation, build_experiment_metadata, load_evaluation_config, write_experiment_metadata


class EvaluationMetadataTests(unittest.TestCase):
    def test_optional_identity_and_infinite_psnr_are_json_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "pred").mkdir(); (root / "gt").mkdir()
            config_path = root / "job.json"; config_path.write_text('{"schema_version":1,"contract_version":"C-06-v1","experiment_id":"x","protocol_id":"group-basicsr-cpu","dataset":{"prediction_root":"pred","target_root":"gt"},"output_path":"out.json","checkpoint":null,"model":null,"provenance":{"group_commit":null}}', encoding="utf-8")
            summary = EvaluationSummary((SampleEvaluation("a", float("inf"), 1.0, GROUP_BASICSR_CPU),))
            target = write_experiment_metadata(root / "out.json", build_experiment_metadata(load_evaluation_config(config_path), summary))
            document = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(document["summary"]["mean_psnr"], "inf")
            self.assertIsNone(document["model"]["identity"])
            with self.assertRaises(Exception): write_experiment_metadata(target, document)
