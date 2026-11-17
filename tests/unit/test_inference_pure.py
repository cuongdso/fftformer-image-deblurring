"""Dependency-free unit tests for WS3 inference data structures, padding and contracts."""

from dataclasses import FrozenInstanceError
from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from fftformer_lab.inference import (  # noqa: E402
    InferenceError,
    InferenceExecutionError,
    InferenceInputError,
    InferencePaddingError,
    InferenceRequest,
    InferenceResult,
    PaddingMetadata,
    calculate_reflection_padding,
)


class InferencePureTests(unittest.TestCase):
    def test_padding_metadata_attributes_and_immutability(self) -> None:
        pad = PaddingMetadata(top=0, bottom=12, left=0, right=14, mode="reflect", original_shape=(52, 50), padded_shape=(64, 64))
        self.assertEqual(pad.top, 0)
        self.assertEqual(pad.bottom, 12)
        self.assertEqual(pad.left, 0)
        self.assertEqual(pad.right, 14)
        self.assertEqual(pad.mode, "reflect")
        self.assertEqual(pad.original_shape, (52, 50))
        self.assertEqual(pad.padded_shape, (64, 64))
        with self.assertRaises(FrozenInstanceError):
            pad.bottom = 99  # type: ignore

        as_dict = pad.to_dict()
        self.assertEqual(as_dict["bottom"], 12)
        self.assertEqual(as_dict["right"], 14)
        self.assertEqual(as_dict["mode"], "reflect")

    def test_calculate_reflection_padding_multiples_of_32(self) -> None:
        # Exact multiples require zero padding
        pad = calculate_reflection_padding(64, 128, factor=32)
        self.assertEqual(pad.top, 0)
        self.assertEqual(pad.bottom, 0)
        self.assertEqual(pad.left, 0)
        self.assertEqual(pad.right, 0)
        self.assertEqual(pad.original_shape, (64, 128))
        self.assertEqual(pad.padded_shape, (64, 128))

    def test_calculate_reflection_padding_arbitrary_sizes(self) -> None:
        # 100 -> next multiple is 128, diff is 28
        # 70 -> next multiple is 96, diff is 26
        pad = calculate_reflection_padding(100, 70, factor=32)
        self.assertEqual(pad.bottom, 28)
        self.assertEqual(pad.right, 26)
        self.assertEqual(pad.original_shape, (100, 70))
        self.assertEqual(pad.padded_shape, (128, 96))

    def test_calculate_reflection_padding_invalid_dimensions(self) -> None:
        cases = [
            (0, 100), (-5, 100), (100, 0), (100, -10),
            ("100", 100), (100, 32.5), (True, 64)
        ]
        for h, w in cases:
            with self.subTest(h=h, w=w), self.assertRaises(InferencePaddingError):
                calculate_reflection_padding(h, w)  # type: ignore

    def test_calculate_reflection_padding_too_small_for_reflection(self) -> None:
        # When height <= pad_h or width <= pad_w, reflection padding is invalid
        # E.g. height=10 -> pad_h = (32 - 10) = 22. 10 <= 22 -> must error
        with self.assertRaisesRegex(InferencePaddingError, "too small for reflection padding"):
            calculate_reflection_padding(10, 64)

        with self.assertRaisesRegex(InferencePaddingError, "too small for reflection padding"):
            calculate_reflection_padding(64, 10)

    def test_exception_hierarchy(self) -> None:
        self.assertTrue(issubclass(InferenceInputError, InferenceError))
        self.assertTrue(issubclass(InferenceInputError, ValueError))
        self.assertTrue(issubclass(InferencePaddingError, InferenceError))
        self.assertTrue(issubclass(InferencePaddingError, ValueError))
        self.assertTrue(issubclass(InferenceExecutionError, InferenceError))

    def test_inference_result_structure_and_dict(self) -> None:
        pad = PaddingMetadata(top=0, bottom=4, left=0, right=6, original_shape=(60, 58), padded_shape=(64, 64))
        result = InferenceResult(
            restored_image="mock_tensor",  # type: ignore
            sample_id="test_001",
            input_shape=(3, 60, 58),
            padded_shape=(3, 64, 64),
            output_shape=(3, 60, 58),
            padding=pad,
            device="cpu",
            model_id="mock:1.0",
            is_mock=True,
            elapsed_time_seconds=0.015,
            model_metadata={"mock": True},
        )
        self.assertEqual(result.sample_id, "test_001")
        self.assertTrue(result.is_mock)
        d = result.to_dict()
        self.assertEqual(d["sample_id"], "test_001")
        self.assertEqual(d["output_shape"], (3, 60, 58))
        self.assertNotIn("restored_image", d)
        self.assertEqual(d["padding"]["bottom"], 4)


class InferenceMockValidationTests(unittest.TestCase):
    """Torch-free checks for the mock configuration and padding factor."""

    def test_mock_model_rejects_unknown_mode(self) -> None:
        from fftformer_lab.inference import MockModel

        with self.assertRaisesRegex(ValueError, "Unknown mock mode"):
            MockModel(mode="bogus")

    def test_calculate_reflection_padding_custom_factor(self) -> None:
        pad = calculate_reflection_padding(30, 30, factor=16)
        self.assertEqual((pad.bottom, pad.right), (2, 2))
        self.assertEqual(pad.padded_shape, (32, 32))

    def test_calculate_reflection_padding_rejects_invalid_factor(self) -> None:
        with self.assertRaisesRegex(InferencePaddingError, "Invalid factor"):
            calculate_reflection_padding(64, 64, factor=0)
