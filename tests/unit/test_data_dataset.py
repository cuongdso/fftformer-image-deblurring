"""C-03 tests for deterministic pairing, portable manifests and lazy samples."""

from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from fftformer_lab.data import (  # noqa: E402
    DatasetValidationError, ImageDataset, ImageReadError, ImageValidationError,
    SampleRecord, build_paired_manifest, build_single_manifest,
    load_manifest, save_manifest,
)

FIXTURES = REPO / "tests/fixtures/ws2"
HAS_DEPS = all(importlib.util.find_spec(name) is not None for name in ("torch", "numpy", "PIL"))
if HAS_DEPS:
    import torch
    from PIL import Image


class DatasetFiles(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.inputs = self.root/"input"
        self.targets = self.root/"target"
        self.inputs.mkdir()
        self.targets.mkdir()

    def copy_image(self, directory: Path, name: str, fixture: str = "rgb.png") -> Path:
        path = directory/name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(FIXTURES/fixture, path)
        return path

    def pair(self, name: str) -> None:
        self.copy_image(self.inputs, name)
        self.copy_image(self.targets, name)


class ManifestTests(DatasetFiles):
    def test_paired_order_and_ids_use_full_relative_filename(self) -> None:
        for name in ("z.png", "b.PNG", "a.png"):
            self.pair(name)
        first = build_paired_manifest(self.inputs, self.targets)
        second = build_paired_manifest(self.inputs, self.targets)
        self.assertEqual(first, second)
        self.assertEqual([r.sample_id for r in first], ["a.png", "b.PNG", "z.png"])
        self.assertTrue(all(r.has_ground_truth for r in first))
        self.assertEqual(first[0].source_path.name, first[0].target_path.name)

    def test_recursive_layout_preserves_sequence_identity(self) -> None:
        self.pair("sequence_b/frame.png")
        self.pair("sequence_a/frame.png")
        records = build_paired_manifest(self.inputs, self.targets, recursive=True)
        self.assertEqual([r.sample_id for r in records], ["sequence_a/frame.png", "sequence_b/frame.png"])

    def test_missing_target_and_orphan_target_are_errors(self) -> None:
        self.pair("ok.png")
        self.copy_image(self.inputs, "missing.png")
        self.copy_image(self.targets, "orphan.png")
        with self.assertRaisesRegex(DatasetValidationError, "missing.png.*orphan.png"):
            build_paired_manifest(self.inputs, self.targets)

    def test_same_stem_different_extension_is_not_a_pair(self) -> None:
        self.copy_image(self.inputs, "a.png")
        self.copy_image(self.targets, "a.jpg")
        with self.assertRaisesRegex(DatasetValidationError, "missing targets"):
            build_paired_manifest(self.inputs, self.targets)

    def test_single_manifest_has_no_implicit_ground_truth(self) -> None:
        self.pair("a.png")
        records = build_single_manifest(self.inputs)
        self.assertFalse(records[0].has_ground_truth)
        self.assertIsNone(records[0].target_path)

    def test_invalid_or_empty_roots_and_scan_policy(self) -> None:
        for path in (self.root/"missing", self.inputs, ""):
            with self.subTest(path=path), self.assertRaises(DatasetValidationError):
                build_single_manifest(path)
        self.copy_image(self.inputs, "a.png")
        (self.inputs/"notes.txt").write_text("not an image", encoding="utf-8")
        with self.assertRaisesRegex(DatasetValidationError, "unsupported extension"):
            build_single_manifest(self.inputs)
        with self.assertRaisesRegex(DatasetValidationError, "recursive"):
            build_single_manifest(self.targets, recursive="yes")

    def test_invalid_ids_and_extensions_are_rejected(self) -> None:
        for sid in (None, "", " ", "a\0b"):
            with self.subTest(sid=sid), self.assertRaisesRegex(DatasetValidationError, "sample_id"):
                SampleRecord(sid, self.inputs/"a.png")
        with self.assertRaisesRegex(DatasetValidationError, "extension"):
            SampleRecord("sample", self.inputs/"a.bmp")

    def test_duplicate_id_source_and_target_are_rejected(self) -> None:
        self.pair("a.png")
        self.pair("b.png")
        a, b = build_paired_manifest(self.inputs, self.targets)
        cases = [([a, a], "duplicate sample_id"),
                 ([a, replace(b, source_path=a.source_path)], "duplicate source"),
                 ([a, replace(b, target_path=a.target_path)], "duplicate target")]
        for records, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(DatasetValidationError, message):
                ImageDataset(records)

    def test_missing_files_and_nonrecords_are_rejected(self) -> None:
        for records in ([], None, ["not-a-record"], [SampleRecord("missing", self.inputs/"missing.png")]):
            with self.subTest(records=records), self.assertRaises(DatasetValidationError):
                ImageDataset(records)
        self.pair("a.png")
        record = build_paired_manifest(self.inputs, self.targets)[0]
        record.target_path.unlink()
        with self.assertRaisesRegex(DatasetValidationError, "missing target"):
            ImageDataset([record])

    def test_dataset_is_lazy_and_records_are_sorted_without_image_dependencies(self) -> None:
        self.copy_image(self.inputs, "z.png")
        self.copy_image(self.inputs, "a.png")
        records = list(reversed(build_single_manifest(self.inputs)))
        dataset = ImageDataset(records)
        self.assertEqual(len(dataset), 2)
        self.assertIsInstance(dataset.records, tuple)
        self.assertEqual(dataset.records[0].sample_id, "a.png")

    def test_manifest_roundtrip_is_relative_deterministic_and_relocatable(self) -> None:
        self.pair("z.png")
        self.pair("a.png")
        records = build_paired_manifest(self.inputs, self.targets)
        path = self.root/"manifest.json"
        save_manifest(reversed(records), path)
        text = path.read_text(encoding="utf-8")
        self.assertNotIn(b"\r\n", path.read_bytes())
        self.assertNotIn(str(self.root), text)
        self.assertEqual(load_manifest(path), records)
        save_manifest(records, path)
        self.assertEqual(path.read_text(encoding="utf-8"), text)
        destination = self.root/"relocated"
        destination.mkdir()
        shutil.copytree(self.inputs, destination/"input")
        shutil.copytree(self.targets, destination/"target")
        shutil.copyfile(path, destination/"manifest.json")
        moved = load_manifest(destination/"manifest.json")
        self.assertEqual([r.sample_id for r in moved], [r.sample_id for r in records])
        self.assertTrue(all(r.source_path.is_relative_to(destination) for r in moved))

    def test_manifest_without_gt_and_explicit_custom_ids(self) -> None:
        path = self.copy_image(self.inputs, "a.png")
        records = [SampleRecord("custom-id", path)]
        manifest = self.root/"single.json"
        save_manifest(records, manifest)
        loaded = load_manifest(manifest)
        self.assertEqual(loaded[0].sample_id, "custom-id")
        self.assertFalse(loaded[0].has_ground_truth)

    def test_invalid_json_and_schema_are_reported(self) -> None:
        path = self.root/"manifest.json"
        for text in ("not-json", "[]", "{}", '{"schema_version":2,"samples":[]}',
                     '{"schema_version":true,"samples":[]}', '{"schema_version":1,"samples":{}}',
                     '{"schema_version":1,"samples":[]}'):
            path.write_text(text, encoding="utf-8")
            with self.subTest(text=text), self.assertRaises(DatasetValidationError):
                load_manifest(path)
        with self.assertRaisesRegex(DatasetValidationError, "cannot read JSON"):
            load_manifest(self.root/"missing.json")

    def test_duplicate_json_fields_do_not_silently_override_ground_truth(self) -> None:
        self.pair("a.png")
        path = self.root/"manifest.json"
        path.write_text(
            '{"schema_version":1,"samples":[{"sample_id":"a","source_path":"input/a.png",'
            '"target_path":null,"has_ground_truth":true,"has_ground_truth":false}]}', encoding="utf-8",
        )
        with self.assertRaisesRegex(DatasetValidationError, "duplicate JSON field"):
            load_manifest(path)

    def test_manifest_rejects_invalid_gt_flags_paths_and_duplicate_ids(self) -> None:
        self.pair("a.png")
        manifest = self.root/"manifest.json"
        item = {"sample_id":"id", "source_path":"input/a.png", "target_path":None, "has_ground_truth":False}
        bad_items = [dict(item, has_ground_truth=True), dict(item, has_ground_truth=0),
                     dict(item, source_path=""), dict(item, source_path=123),
                     dict(item, target_path=""), dict(item, unknown=True)]
        for bad in bad_items:
            manifest.write_text(json.dumps({"schema_version":1,"samples":[bad]}), encoding="utf-8")
            with self.subTest(bad=bad), self.assertRaises(DatasetValidationError):
                load_manifest(manifest)
        manifest.write_text(json.dumps({"schema_version":1,"samples":[item,item]}), encoding="utf-8")
        with self.assertRaisesRegex(DatasetValidationError, "duplicate sample_id"):
            load_manifest(manifest)

    def test_manifest_cannot_overwrite_image_file(self) -> None:
        self.pair("a.png")
        records = build_paired_manifest(self.inputs, self.targets)
        before = records[0].source_path.read_bytes()
        with self.assertRaisesRegex(DatasetValidationError, ".json"):
            save_manifest(records, records[0].source_path)
        self.assertEqual(records[0].source_path.read_bytes(), before)


@unittest.skipUnless(HAS_DEPS, "requires torch, numpy and Pillow (WS2 image dependencies)")
class DatasetRuntimeTests(DatasetFiles):
    def test_pair_returns_contract_sample_without_modifying_sources(self) -> None:
        self.pair("a.png")
        records = build_paired_manifest(self.inputs, self.targets)
        paths = [records[0].source_path, records[0].target_path]
        hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        dataset = ImageDataset(records)
        sample = dataset[0]
        self.assertEqual(sample.sample_id, "a.png")
        self.assertTrue(sample.has_ground_truth)
        self.assertEqual(sample.source_path, paths[0])
        self.assertEqual(sample.target_path, paths[1])
        self.assertEqual(tuple(sample.image.shape), (3,2,2))
        self.assertEqual(sample.image.dtype, torch.float32)
        torch.testing.assert_close(sample.image, sample.target, rtol=0, atol=0)
        self.assertNotEqual(sample.image.data_ptr(), sample.target.data_ptr())
        meta = sample.preprocessing_metadata.to_dict()
        self.assertEqual(meta["original_shape"], [3,2,2])
        self.assertEqual(meta["padding"], [0,0,0,0])
        self.assertIsNone(meta["resize"])
        self.assertIsNone(meta["crop"])
        json.dumps(meta)
        self.assertEqual(hashes, [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths])

    def test_single_has_none_target_and_dataset_slices_preserve_order(self) -> None:
        self.copy_image(self.inputs, "b.png")
        self.copy_image(self.inputs, "a.png")
        dataset = ImageDataset(build_single_manifest(self.inputs))
        self.assertEqual([s.sample_id for s in dataset[:]], ["a.png", "b.png"])
        self.assertEqual([s.sample_id for s in dataset], ["a.png", "b.png"])
        sample = dataset[-1]
        self.assertFalse(sample.has_ground_truth)
        self.assertIsNone(sample.target)
        self.assertIsNone(sample.target_path)
        with self.assertRaises(IndexError):
            dataset[2]

    def test_size_mismatch_is_not_cropped_or_resized(self) -> None:
        self.pair("wrong-size.png")
        Image.new("RGB", (3,2)).save(self.targets/"wrong-size.png")
        dataset = ImageDataset(build_paired_manifest(self.inputs, self.targets))
        with self.assertRaisesRegex(DatasetValidationError, "wrong-size.png.*size mismatch"):
            dataset[0]

    def test_channel_and_corrupt_image_errors_include_sample_context(self) -> None:
        self.pair("bad.png")
        for fixture in ("grayscale.png", "rgba.png"):
            self.copy_image(self.targets, "bad.png", fixture)
            dataset = ImageDataset(build_paired_manifest(self.inputs, self.targets))
            with self.subTest(fixture=fixture), self.assertRaisesRegex(ImageValidationError, "sample 'bad.png'.*target"):
                dataset[0]
        (self.targets/"bad.png").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ImageReadError, "sample 'bad.png'"):
            dataset[0]

    def test_later_file_failure_keeps_context_instead_of_returning_placeholder(self) -> None:
        self.copy_image(self.inputs, "a.png")
        dataset = ImageDataset(build_single_manifest(self.inputs))
        (self.inputs/"a.png").unlink()
        with self.assertRaisesRegex(ImageReadError, "sample 'a.png'"):
            dataset[0]

    def test_samples_reject_inconsistent_gt_and_metadata(self) -> None:
        self.copy_image(self.inputs, "a.png")
        sample = ImageDataset(build_single_manifest(self.inputs))[0]
        for fields in ({"has_ground_truth":True}, {"has_ground_truth":0}, {"target_path":self.inputs/"a.png"},
                       {"preprocessing_metadata":None}):
            with self.subTest(fields=fields), self.assertRaises(DatasetValidationError):
                replace(sample, **fields)

    def test_reloading_returns_fresh_tensors_without_random_augmentation(self) -> None:
        self.copy_image(self.inputs, "a.png")
        dataset = ImageDataset(build_single_manifest(self.inputs))
        first, second = dataset[0], dataset[0]
        torch.testing.assert_close(first.image, second.image, rtol=0, atol=0)
        first.image[:] = 0
        self.assertGreater(float(second.image.max()), 0)

    def test_sample_sequence_is_consumable_by_existing_ws3_mock_pipeline(self) -> None:
        from fftformer_lab.inference import InferencePipeline, MockModel

        Image.new("RGB", (70,50), (255,64,0)).save(self.inputs/"b.png")
        Image.new("RGB", (64,64), (0,128,255)).save(self.inputs/"a.png")
        dataset = ImageDataset(build_single_manifest(self.inputs))
        results = InferencePipeline(MockModel(mode="identity", device="cpu")).run_batch(dataset)
        self.assertEqual([result.sample_id for result in results], ["a.png", "b.png"])
        self.assertEqual(results[1].padding.right, 26)
        self.assertEqual(results[1].padding.bottom, 14)
        for result, sample in zip(results, dataset):
            torch.testing.assert_close(result.restored_image, sample.image, rtol=0, atol=0)
            self.assertTrue(result.is_mock)
            self.assertEqual(result.output_shape, tuple(sample.image.shape))
