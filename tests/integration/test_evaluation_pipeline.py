"""Real WS2 -> WS3 -> WS4 CPU integration tests; MockModel is the only double."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

try:
    import cv2
    import numpy as np
    from PIL import Image
    import torch
except ImportError:
    cv2 = np = Image = torch = None  # type: ignore[assignment]

from fftformer_lab.data import DatasetValidationError, ImageDataset, build_paired_manifest, build_single_manifest
from fftformer_lab.evaluation import EvaluationInputError, ProtocolUnavailableError, UnknownProtocolError, evaluate_batch, evaluate_sample
from fftformer_lab.inference import InferenceInputError, InferencePipeline, MockModel


@unittest.skipUnless(
    all(item is not None for item in (cv2, np, Image, torch)),
    "WS2->WS3->WS4 integration requires Torch, NumPy, Pillow, and OpenCV",
)
class EvaluationPipelineIntegrationTests(unittest.TestCase):
    def make_pairs(self, root: Path, specs: list[tuple[str, int, int, int]]) -> tuple[Path, Path]:
        inputs, targets = root / "input", root / "target"; inputs.mkdir(); targets.mkdir()
        for name, height, width, offset in specs:
            rows, columns = np.indices((height, width))
            image = np.stack(((columns * 3 + offset) % 256, (rows * 5 + offset) % 256,
                              ((rows + columns) * 2 + offset) % 256), axis=2).astype(np.uint8)
            Image.fromarray(image, mode="RGB").save(inputs / name)
            Image.fromarray(image, mode="RGB").save(targets / name)
        return inputs, targets

    def test_single_identity_preserves_pixels_and_has_infinite_psnr(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs, targets = self.make_pairs(Path(directory), [("single.png", 50, 70, 13)])
            sample = ImageDataset(build_paired_manifest(inputs, targets))[0]
            result = InferencePipeline(MockModel("identity")).run_single(sample)
            measurement = evaluate_sample(result, sample, "group-basicsr-cpu")
            self.assertEqual(result.sample_id, sample.sample_id)
            self.assertEqual(tuple(result.restored_image.shape), (3, 50, 70))
            self.assertEqual(result.padding.padded_shape, (64, 96))
            self.assertTrue(torch.equal(result.restored_image, sample.image))
            self.assertTrue(np.isinf(measurement.psnr))
            self.assertAlmostEqual(measurement.ssim, 1.0, places=12)

    def test_batch_shift_keeps_order_alignment_and_real_finite_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs, targets = self.make_pairs(Path(directory), [("b.png", 64, 64, 3), ("a.png", 50, 70, 29)])
            samples = list(ImageDataset(build_paired_manifest(inputs, targets)))
            results = InferencePipeline(MockModel("shift")).run_batch(samples)
            summary = evaluate_batch(results, samples, "group-basicsr-cpu")
            self.assertEqual([item.sample_id for item in results], ["a.png", "b.png"])
            self.assertEqual([item.sample_id for item in summary.evaluations], ["a.png", "b.png"])
            self.assertEqual(summary.valid_count, 2)
            self.assertEqual(summary.excluded_count, 0)
            self.assertTrue(all(np.isfinite(item.psnr) for item in summary.evaluations))
            self.assertTrue(all(item.ssim < 1.0 for item in summary.evaluations))
            self.assertTrue(all(result.is_mock and result.model_metadata["is_mock"] for result in results))
            self.assertEqual(summary.evaluations[0].model_metadata, results[0].model_metadata)

    def test_non_gt_is_inferred_but_explicitly_excluded_from_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"; root.mkdir()
            rows, columns = np.indices((64, 64)); image = np.stack((rows, columns, rows + columns), axis=2).astype(np.uint8)
            Image.fromarray(image, mode="RGB").save(root / "only.png")
            sample = ImageDataset(build_single_manifest(root))[0]
            result = InferencePipeline(MockModel("identity")).run_single(sample)
            summary = evaluate_batch([result], [sample], "group-basicsr-cpu")
            self.assertEqual(summary.valid_count, 0)
            self.assertEqual(summary.excluded, ({"sample_id": "only.png", "reason": "missing-ground-truth"},))

    def test_contract_errors_propagate_without_default_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); inputs, targets = self.make_pairs(root, [("a.png", 64, 64, 1)])
            Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB").save(targets / "extra.png")
            with self.assertRaises(DatasetValidationError): build_paired_manifest(inputs, targets)
            (targets / "extra.png").unlink()
            sample = ImageDataset(build_paired_manifest(inputs, targets))[0]
            result = InferencePipeline(MockModel("identity")).run_single(sample)
            with self.assertRaises(EvaluationInputError): evaluate_sample(replace(result, sample_id="wrong"), sample, "group-basicsr-cpu")
            with self.assertRaises(InferenceInputError): InferencePipeline(MockModel()).run_single(torch.zeros(2, 64, 64))
            with self.assertRaises(UnknownProtocolError): evaluate_sample(result, sample, "unknown")
            with self.assertRaises(ProtocolUnavailableError): evaluate_sample(result, sample, "realblur-official")
