"""CLI parser/config tests; image execution is deliberately dependency-gated."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from fftformer_lab.evaluation import cli


class EvaluationCliTests(unittest.TestCase):
    def config(self, root: Path) -> Path:
        (root / "pred").mkdir(); (root / "gt").mkdir()
        path = root / "job.json"; path.write_text('{"schema_version":1,"contract_version":"C-06-v1","experiment_id":"x","protocol_id":"group-basicsr-cpu","dataset":{"prediction_root":"pred","target_root":"gt"},"output_path":"out.json","checkpoint":null,"model":null,"provenance":{"group_commit":null}}', encoding="utf-8")
        return path

    def test_validate_config_returns_success_without_cv_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("sys.stdout"):
                self.assertEqual(cli.main(["validate-config", str(self.config(Path(directory)))]), 0)

    def test_help_and_invalid_config_exit(self) -> None:
        with self.assertRaises(SystemExit) as help_exit: cli.main(["--help"])
        self.assertEqual(help_exit.exception.code, 0)
        with self.assertRaises(SystemExit) as invalid_exit: cli.main(["validate-config", "missing.json"])
        self.assertEqual(invalid_exit.exception.code, 2)

    def test_run_reports_dataset_runtime_failure_as_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit) as run_exit:
                cli.main(["run", str(self.config(Path(directory)))])
        self.assertEqual(run_exit.exception.code, 2)
