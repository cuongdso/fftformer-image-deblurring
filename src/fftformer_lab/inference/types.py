"""Data structures and exceptions for M1-WS3-02 / C-04."""

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class InferenceError(RuntimeError):
    """Base exception for inference operations."""


class InferenceInputError(InferenceError, ValueError):
    """Input tensor or request violates C-02/C-04 specifications."""


class InferencePaddingError(InferenceError, ValueError):
    """Padding calculation or padding operation failed."""


class InferenceExecutionError(InferenceError):
    """Underlying model execution or device transfer failed."""


class InferenceIOError(InferenceError):
    """Image read/write or filesystem serialization failed."""


@dataclass(frozen=True)
class PaddingMetadata:
    """Reflection padding metadata conforming to Contract C-02."""

    top: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0
    mode: str = "reflect"
    original_shape: tuple[int, int] = field(default_factory=lambda: (0, 0))  # (H, W)
    padded_shape: tuple[int, int] = field(default_factory=lambda: (0, 0))    # (H_pad, W_pad)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable dictionary."""
        return {
            "top": self.top,
            "bottom": self.bottom,
            "left": self.left,
            "right": self.right,
            "mode": self.mode,
            "original_shape": self.original_shape,
            "padded_shape": self.padded_shape,
        }


@dataclass(frozen=True)
class InferenceRequest:
    """Single-image inference request conforming to C-02/C-04."""

    image: "torch.Tensor"
    sample_id: str = "sample"


@dataclass(frozen=True)
class InferenceResult:
    """Structured inference output for a single sample conforming to C-04."""

    restored_image: "torch.Tensor"
    sample_id: str
    input_shape: tuple[int, int, int]
    padded_shape: tuple[int, int, int]
    output_shape: tuple[int, int, int]
    padding: PaddingMetadata
    device: str
    model_id: str
    is_mock: bool = False
    elapsed_time_seconds: float | None = None
    model_metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable metadata without raw image tensor."""
        return {
            "sample_id": self.sample_id,
            "input_shape": self.input_shape,
            "padded_shape": self.padded_shape,
            "output_shape": self.output_shape,
            "padding": self.padding.to_dict(),
            "device": self.device,
            "model_id": self.model_id,
            "is_mock": self.is_mock,
            "elapsed_time_seconds": self.elapsed_time_seconds,
            "model_metadata": self.model_metadata,
        }
