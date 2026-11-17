"""WS3 Inference & Demonstration module (Contract C-04)."""

from .mock import MockModel
from .padding import calculate_reflection_padding, pad_image, unpad_image
from .pipeline import InferencePipeline, validate_input_tensor
from .types import (
    InferenceError,
    InferenceExecutionError,
    InferenceInputError,
    InferencePaddingError,
    InferenceRequest,
    InferenceResult,
    PaddingMetadata,
)

__all__ = [
    "InferenceError",
    "InferenceExecutionError",
    "InferenceInputError",
    "InferencePaddingError",
    "InferencePipeline",
    "InferenceRequest",
    "InferenceResult",
    "MockModel",
    "PaddingMetadata",
    "calculate_reflection_padding",
    "pad_image",
    "unpad_image",
    "validate_input_tensor",
]
