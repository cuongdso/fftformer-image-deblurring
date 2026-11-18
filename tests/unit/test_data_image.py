"""C-02 regression tests using locally generated fixtures, not benchmark data."""

import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from fftformer_lab.data import (  # noqa: E402
    DataDependencyError, ImageReadError, ImageValidationError,
    array_to_tensor, read_image, validate_image_tensor,
)

HAS_DEPS = all(importlib.util.find_spec(name) is not None for name in ("torch", "numpy", "PIL"))
if HAS_DEPS:
    import numpy as np
    import torch
    from PIL import Image

FIXTURES = REPO / "tests/fixtures/ws2"


class DataImportTests(unittest.TestCase):
    def test_import_needs_only_standard_library(self) -> None:
        code = (
            f"import sys; sys.path.insert(0, {str(REPO / 'src')!r}); "
            "import fftformer_lab.data; "
            "assert not any(x in sys.modules for x in ('torch','numpy','PIL','cv2'))"
        )
        result = subprocess.run([sys.executable, "-B", "-S", "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_dependency_has_actionable_error(self) -> None:
        with patch("fftformer_lab.data.image.importlib.import_module", side_effect=ImportError("missing")):
            with self.assertRaisesRegex(DataDependencyError, "requires torch"):
                validate_image_tensor(None)


@unittest.skipUnless(HAS_DEPS, "requires torch, numpy and Pillow (WS2 image dependencies)")
class ImageBoundaryTests(unittest.TestCase):
    def test_fixture_exact_colors_and_source_bytes_unchanged(self) -> None:
        path = FIXTURES / "rgb.png"
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        tensor = read_image(path)
        self.assertEqual(tuple(tensor.shape), (3, 2, 2))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(str(tensor.device), "cpu")
        self.assertTrue(tensor.is_contiguous())
        expected = torch.tensor([[[1, 0], [0, 1]], [[0, 1], [0, 1]], [[0, 0], [1, 1]]], dtype=torch.float32)
        torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_uint8_conversion_roundtrip_all_levels(self) -> None:
        arr = np.arange(256, dtype=np.uint8).reshape(16,16,1).repeat(3, axis=2)
        tensor = array_to_tensor(arr, color_order="RGB")
        roundtrip = (tensor.permute(1,2,0).numpy() * 255).round().astype(np.uint8)
        np.testing.assert_array_equal(roundtrip, arr)

    def test_bgr_conversion_is_explicit_and_does_not_mutate(self) -> None:
        bgr = np.array([[[17, 64, 255]]], dtype=np.uint8)
        original = bgr.copy()
        tensor = array_to_tensor(bgr, color_order="BGR")
        torch.testing.assert_close(tensor[:,0,0], torch.tensor([255,64,17])/255, rtol=0, atol=0)
        np.testing.assert_array_equal(bgr, original)
        with self.assertRaises(TypeError):
            array_to_tensor(bgr)
        for value in ("bgr", "auto", None):
            with self.subTest(value=value), self.assertRaises(ImageValidationError):
                array_to_tensor(bgr, color_order=value)

    def test_float32_input_is_not_divided_or_aliased(self) -> None:
        arr = np.full((2,3,3), 0.5, dtype=np.float32)
        tensor = array_to_tensor(arr, color_order="RGB")
        arr[:] = 0
        self.assertTrue(bool((tensor == 0.5).all()))
        tensor[:] = 1
        self.assertTrue(bool((arr == 0).all()))

    def test_noncontiguous_and_readonly_arrays_are_supported(self) -> None:
        arr = np.arange(36, dtype=np.uint8).reshape(3,4,3)[:,::-1]
        arr.flags.writeable = False
        converted = array_to_tensor(arr, color_order="RGB")
        np.testing.assert_array_equal((converted.permute(1,2,0).numpy()*255).round().astype(np.uint8), arr)

    def test_invalid_arrays_report_shape_dtype_range_or_finiteness(self) -> None:
        cases = [([], "numpy"), (np.zeros((2,2)), "HWC"),
                 (np.zeros((2,2,1)), "HWC"), (np.zeros((2,2,4)), "HWC"),
                 (np.zeros((0,2,3)), "HWC"), (np.zeros((1,1,3), dtype=np.uint16), "uint8"),
                 (np.zeros((1,1,3), dtype=np.float64), "float32")]
        for value in (-0.1, 255.0, float("nan"), float("inf")):
            cases.append((np.full((2,2,3), value, dtype=np.float32), "finite" if not np.isfinite(value) else "range"))
        for arr, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ImageValidationError, message):
                array_to_tensor(arr, color_order="RGB")

    def test_tensor_validation_rejects_invalid_boundary(self) -> None:
        cases = [(None, "Tensor"), (torch.zeros(3,2,2,dtype=torch.float64), "float32"),
                 (torch.zeros(1,3,2,2), "CHW"), (torch.zeros(4,2,2), "CHW"),
                 (torch.zeros(3,0,2), "CHW")]
        for value in (-0.1, 1.1, float("nan"), float("inf")):
            cases.append((torch.full((3,2,2), value), "finite" if not np.isfinite(value) else "range"))
        for tensor, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ImageValidationError, message):
                validate_image_tensor(tensor)

    def test_grayscale_and_alpha_are_not_silently_converted(self) -> None:
        for filename in ("grayscale.png", "rgba.png"):
            with self.subTest(filename=filename), self.assertRaisesRegex(ImageValidationError, "expected RGB"):
                read_image(FIXTURES / filename)

    def test_missing_corrupt_directory_and_unsupported_extension(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "corrupt.png").write_bytes(b"not an image")
            (root / "folder.png").mkdir()
            for name in ("missing.png", "corrupt.png", "folder.png"):
                with self.subTest(name=name), self.assertRaisesRegex(ImageReadError, name):
                    read_image(root / name)
            with self.assertRaisesRegex(ImageValidationError, "extension"):
                read_image(root / "image.bmp")

    def test_jpeg_and_uppercase_extension_preserve_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"image.JPG"
            Image.new("RGB", (37,19), (24,128,220)).save(path)
            tensor = read_image(path)
            self.assertEqual(tuple(tensor.shape), (3,19,37))
            validate_image_tensor(tensor)

    def test_high_bit_depth_png_is_rejected_before_lossy_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"image.png"
            Image.fromarray(np.array([[0,65535]], dtype=np.uint16)).save(path)
            with self.assertRaisesRegex(ImageValidationError, "8-bit PNG"):
                read_image(path)

    def test_animated_png_and_disguised_unsupported_format_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"animated.png"
            first = Image.new("RGB", (4,4), "red")
            second = Image.new("RGB", (4,4), "blue")
            first.save(path, save_all=True, append_images=[second], duration=100, loop=0)
            with self.assertRaisesRegex(ImageValidationError, "multi-frame"):
                read_image(path)
            disguised = Path(temp)/"not-really-a-png.png"
            first.save(disguised, format="BMP")
            with self.assertRaisesRegex(ImageValidationError, "decoded format"):
                read_image(disguised)

    def test_caller_default_meta_device_does_not_move_image_off_cpu(self) -> None:
        with torch.device("meta"):
            tensor = read_image(FIXTURES/"rgb.png")
        self.assertEqual(str(tensor.device), "cpu")
        self.assertEqual(float(tensor.max()), 1.0)

    def test_caller_float64_default_does_not_change_output_dtype(self) -> None:
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            tensor = read_image(FIXTURES/"rgb.png")
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertEqual(torch.get_default_dtype(), torch.float64)
        finally:
            torch.set_default_dtype(previous)
