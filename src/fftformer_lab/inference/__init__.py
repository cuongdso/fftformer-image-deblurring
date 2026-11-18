"""WS3 Inference & Demonstration module (Contract C-04)."""

from .cli import run_cli
from .io import SUPPORTED_WRITE_EXTENSIONS, write_image
from .mock import MockModel
from .padding import calculate_reflection_padding, pad_image, unpad_image
from .pipeline import InferencePipeline, validate_input_tensor
from .types import (
    InferenceError,
    InferenceExecutionError,
    InferenceIOError,
    InferenceInputError,
    InferencePaddingError,
    InferenceRequest,
    InferenceResult,
    PaddingMetadata,
)

__all__ = [
    "InferenceError",
    "InferenceExecutionError",
    "InferenceIOError",
    "InferenceInputError",
    "InferencePaddingError",
    "InferencePipeline",
    "InferenceRequest",
    "InferenceResult",
    "MockModel",
    "PaddingMetadata",
    "SUPPORTED_WRITE_EXTENSIONS",
    "calculate_reflection_padding",
    "pad_image",
    "run_cli",
    "unpad_image",
    "validate_input_tensor",
    "write_image",
]
