"""CLI parser/config tests; image execution is deliberately dependency-gated."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from fftformer_lab.evaluation import cli

try:
    import cv2
    import numpy as np
    from PIL import Image
    import torch
except ImportError:
    cv2 = np = Image = torch = None  # type: ignore[assignment]


class EvaluationCliTests(unittest.TestCase):
    def config(self, root: Path) -> Path:
        (root / "pred").mkdir(exist_ok=True); (root / "gt").mkdir(exist_ok=True)
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

    @unittest.skipUnless(
        cv2 is not None and np is not None and Image is not None and torch is not None,
        "CLI image smoke test requires Torch, NumPy, Pillow, and OpenCV",
    )
    def test_run_evaluates_real_synthetic_png_pairs_and_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); prediction = root / "pred"; target = root / "gt"
            prediction.mkdir(); target.mkdir()
            rows, columns = np.indices((64, 64))
            for index, offset in enumerate((0, 29)):
                rgb = np.stack(((columns * 3 + offset) % 256, (rows * 5 + offset) % 256,
                                ((rows + columns) * 2 + offset) % 256), axis=2).astype(np.uint8)
                Image.fromarray(rgb, mode="RGB").save(target / f"pair_{index}.png")
                blurred = cv2.GaussianBlur(rgb, (5, 5), 1.1)
                Image.fromarray(blurred, mode="RGB").save(prediction / f"pair_{index}.png")
            config = self.config(root)
            config.write_text('{"schema_version":1,"contract_version":"C-06-v1","experiment_id":"synthetic-cli","protocol_id":"group-basicsr-cpu","dataset":{"prediction_root":"pred","target_root":"gt","identity":"synthetic-ws4-smoke"},"output_path":"out.json","checkpoint":null,"model":null,"provenance":{"group_commit":null}}', encoding="utf-8")
            with mock.patch("sys.stdout"):
                self.assertEqual(cli.main(["run", str(config)]), 0)
            document = __import__("json").loads((root / "out.json").read_text(encoding="utf-8"))
            self.assertEqual(document["evaluation_source"], "precomputed-prediction-images")
            self.assertEqual(document["protocol"]["protocol_id"], "group-basicsr-cpu")
            self.assertEqual(document["summary"]["valid_count"], 2)
            self.assertEqual(document["summary"]["excluded_count"], 0)
            self.assertTrue(all(isinstance(item["psnr"], float) for item in document["summary"]["evaluations"]))
            with self.assertRaises(SystemExit):
                cli.main(["run", str(config)])
            with mock.patch("sys.stdout"):
                self.assertEqual(cli.main(["run", str(config), "--overwrite"]), 0)
