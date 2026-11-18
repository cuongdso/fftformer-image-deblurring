"""Unit and CLI integration tests for M1-WS3-03 inference orchestration."""

import json
from pathlib import Path
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO / "src"))

try:
    import torch
    import numpy as np
    import PIL.Image
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from fftformer_lab.inference import (  # noqa: E402
    InferenceIOError,
    InferenceInputError,
    run_cli,
    write_image,
)
from fftformer_lab.inference.cli import create_parser  # noqa: E402
from fftformer_lab.data import read_image, build_single_manifest, save_manifest  # noqa: E402


@unittest.skipUnless(HAS_DEPS, "requires torch, numpy, PIL")
class InferenceIOTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.dir_path = Path(self.tmp_dir.name)

    def test_write_image_valid_png(self) -> None:
        img = torch.rand(3, 40, 50, dtype=torch.float32)
        out_path = self.dir_path / "test_out.png"
        saved = write_image(img, out_path)
        self.assertTrue(saved.exists())
        self.assertGreater(saved.stat().st_size, 0)

        # Read back via WS2 read_image
        reloaded = read_image(saved)
        self.assertEqual(tuple(reloaded.shape), (3, 40, 50))
        self.assertEqual(reloaded.dtype, torch.float32)
        self.assertTrue((reloaded >= 0.0).all() and (reloaded <= 1.0).all())

    def test_write_image_valid_jpeg(self) -> None:
        img = torch.rand(3, 32, 32, dtype=torch.float32)
        out_path = self.dir_path / "test_out.jpg"
        saved = write_image(img, out_path, quality=90)
        self.assertTrue(saved.exists())
        reloaded = read_image(saved)
        self.assertEqual(tuple(reloaded.shape), (3, 32, 32))

    def test_write_image_rejects_unsupported_extension(self) -> None:
        img = torch.rand(3, 32, 32, dtype=torch.float32)
        with self.assertRaisesRegex(InferenceIOError, "Unsupported output image format"):
            write_image(img, self.dir_path / "bad.bmp")

    def test_write_image_rejects_invalid_inputs(self) -> None:
        cases = [
            ("not a tensor", "Expected torch.Tensor"),
            (torch.rand(3, 32, 32, dtype=torch.float64), "Expected torch.float32"),
            (torch.rand(1, 32, 32, dtype=torch.float32), "Expected 3D RGB tensor"),
            (torch.rand(4, 32, 32, dtype=torch.float32), "Expected 3D RGB tensor"),
        ]
        for val, err in cases:
            with self.subTest(val=val), self.assertRaisesRegex(InferenceInputError, err):
                write_image(val, self.dir_path / "out.png")

    def test_write_image_rejects_non_finite(self) -> None:
        img = torch.rand(3, 32, 32, dtype=torch.float32)
        img[0, 2, 2] = float("nan")
        with self.assertRaisesRegex(InferenceInputError, "non-finite"):
            write_image(img, self.dir_path / "out.png")


@unittest.skipUnless(HAS_DEPS, "requires torch, numpy, PIL")
class InferenceCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.dir_path = Path(self.tmp_dir.name)

        # Create a sample test image (35x45)
        self.sample_img_path = self.dir_path / "sample_blur.png"
        sample_tensor = torch.rand(3, 35, 45, dtype=torch.float32)
        write_image(sample_tensor, self.sample_img_path)

    def test_cli_help(self) -> None:
        parser = create_parser()
        with self.assertRaises(SystemExit) as cm:
            parser.parse_args(["--help"])
        self.assertEqual(cm.exception.code, 0)

    def test_cli_missing_input_arguments(self) -> None:
        with self.assertRaises(SystemExit) as cm:
            parser = create_parser()
            parser.parse_args([])
        self.assertNotEqual(cm.exception.code, 0)

    def test_cli_single_image_inference(self) -> None:
        out_file = self.dir_path / "restored.png"
        code = run_cli([
            "--input", str(self.sample_img_path),
            "--output", str(out_file),
            "--mock",
            "--mock-mode", "identity",
            "--quiet",
        ])
        self.assertEqual(code, 0)
        self.assertTrue(out_file.exists())

        reloaded = read_image(out_file)
        self.assertEqual(tuple(reloaded.shape), (3, 35, 45))

    def test_cli_single_image_with_metadata(self) -> None:
        out_file = self.dir_path / "restored_meta.png"
        code = run_cli([
            "--input", str(self.sample_img_path),
            "--output", str(out_file),
            "--mock",
            "--save-metadata",
            "--quiet",
        ])
        self.assertEqual(code, 0)
        self.assertTrue(out_file.exists())

        meta_file = self.dir_path / "restored_meta_metadata.json"
        self.assertTrue(meta_file.exists())
        data = json.loads(meta_file.read_text(encoding="utf-8"))
        self.assertEqual(data["sample_id"], "sample_blur")
        self.assertEqual(data["input_shape"], [3, 35, 45])
        self.assertEqual(data["output_shape"], [3, 35, 45])
        self.assertTrue(data["is_mock"])

    def test_cli_single_image_to_output_dir(self) -> None:
        out_dir = self.dir_path / "single_out_dir"
        code = run_cli([
            "--input", str(self.sample_img_path),
            "--output-dir", str(out_dir),
            "--mock",
            "--quiet",
        ])
        self.assertEqual(code, 0)
        expected = out_dir / "sample_blur_restored.png"
        self.assertTrue(expected.exists())

    def test_cli_single_image_invert_mode(self) -> None:
        out_file = self.dir_path / "inverted.png"
        code = run_cli([
            "--input", str(self.sample_img_path),
            "--output", str(out_file),
            "--mock",
            "--mock-mode", "invert",
            "--quiet",
        ])
        self.assertEqual(code, 0)
        self.assertTrue(out_file.exists())

    def test_cli_nonexistent_input_file(self) -> None:
        nonexistent = self.dir_path / "missing.png"
        out_file = self.dir_path / "out.png"
        code = run_cli([
            "--input", str(nonexistent),
            "--output", str(out_file),
            "--mock",
            "--quiet",
        ])
        self.assertEqual(code, 1)

    def test_cli_batch_directory_mode(self) -> None:
        in_dir = self.dir_path / "batch_in"
        in_dir.mkdir()
        img1 = in_dir / "img1.png"
        img2 = in_dir / "img2.jpg"
        write_image(torch.rand(3, 32, 32), img1)
        write_image(torch.rand(3, 40, 50), img2)

        out_dir = self.dir_path / "batch_out"
        code = run_cli([
            "--input-dir", str(in_dir),
            "--output-dir", str(out_dir),
            "--mock",
            "--save-metadata",
            "--quiet",
        ])
        self.assertEqual(code, 0)

        out1 = out_dir / "img1_restored.png"
        out2 = out_dir / "img2_restored.jpg"
        self.assertTrue(out1.exists())
        self.assertTrue(out2.exists())
        self.assertTrue((out_dir / "img1_restored_metadata.json").exists())

    def test_cli_manifest_mode(self) -> None:
        in_dir = self.dir_path / "manifest_in"
        in_dir.mkdir()
        write_image(torch.rand(3, 32, 32), in_dir / "a.png")
        write_image(torch.rand(3, 32, 32), in_dir / "b.png")

        manifest_file = self.dir_path / "manifest.json"
        records = build_single_manifest(in_dir)
        save_manifest(records, manifest_file)

        out_dir = self.dir_path / "manifest_out"
        code = run_cli([
            "--manifest", str(manifest_file),
            "--output-dir", str(out_dir),
            "--mock",
            "--quiet",
        ])
        self.assertEqual(code, 0)
        self.assertTrue((out_dir / "a_restored.png").exists())
        self.assertTrue((out_dir / "b_restored.png").exists())

    def test_cli_empty_input_directory_returns_error(self) -> None:
        empty_dir = self.dir_path / "empty_dir"
        empty_dir.mkdir()
        out_dir = self.dir_path / "out"
        code = run_cli([
            "--input-dir", str(empty_dir),
            "--output-dir", str(out_dir),
            "--mock",
            "--quiet",
        ])
        self.assertEqual(code, 1)
