"""Explicit image decoding and normalization for C-02 (M1-WS2-02)."""

import importlib
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING

from .errors import DataDependencyError, ImageReadError, ImageValidationError

if TYPE_CHECKING:
    import numpy as np
    import torch


SUPPORTED_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg"})


def _dependency(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise DataDependencyError(f"WS2 image processing requires {name}: {exc}") from exc


def validate_image_tensor(image: "torch.Tensor", *, context: str = "image") -> None:
    """Validate an RGB float32 CHW image without modifying or padding it."""
    torch = _dependency("torch")
    if not isinstance(image, torch.Tensor):
        raise ImageValidationError(f"{context}: expected a torch.Tensor")
    if image.dtype != torch.float32:
        raise ImageValidationError(f"{context}: expected float32, got {image.dtype}")
    if image.ndim != 3 or image.shape[0] != 3 or any(size <= 0 for size in image.shape):
        raise ImageValidationError(f"{context}: expected nonempty CHW [3,H,W], got {tuple(image.shape)}")
    if not bool(torch.isfinite(image).all()):
        raise ImageValidationError(f"{context}: non-finite values (NaN/Inf)")
    if bool((image < 0).any()) or bool((image > 1).any()):
        raise ImageValidationError(f"{context}: expected range [0,1]")


def array_to_tensor(
    image: "np.ndarray", *, color_order: Literal["RGB", "BGR"],
) -> "torch.Tensor":
    """Convert explicit RGB/BGR HWC uint8 or normalized float32 to RGB CHW."""
    np = _dependency("numpy")
    torch = _dependency("torch")
    if color_order not in ("RGB", "BGR"):
        raise ImageValidationError(f"color_order: expected explicit RGB or BGR, got {color_order!r}")
    if not isinstance(image, np.ndarray):
        raise ImageValidationError("image array: expected a numpy.ndarray")
    if image.ndim != 3 or image.shape[2] != 3 or any(size <= 0 for size in image.shape):
        raise ImageValidationError(f"image array: expected nonempty HWC [H,W,3], got {image.shape}")
    if image.dtype not in (np.dtype("uint8"), np.dtype("float32")):
        raise ImageValidationError(f"image array: expected uint8 or normalized float32, got {image.dtype}")
    if image.dtype == np.float32:
        if not np.isfinite(image).all():
            raise ImageValidationError("image array: non-finite values (NaN/Inf)")
        if np.any(image < 0) or np.any(image > 1):
            raise ImageValidationError("image array: float32 values must already be in range [0,1]")
    # Copy even contiguous arrays: callers must not share mutable storage with a sample.
    rgb = image[..., ::-1] if color_order == "BGR" else image
    copied = np.array(rgb.transpose(2, 0, 1), order="C", copy=True)
    tensor = torch.from_numpy(copied).to(dtype=torch.float32)
    if image.dtype == np.uint8:
        tensor.div_(255.0)
    validate_image_tensor(tensor)
    return tensor


def read_image(path: str | Path) -> "torch.Tensor":
    """Read one 8-bit RGB PNG/JPEG as CPU float32 CHW, preserving its pixels."""
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise ImageReadError(f"image path: expected a nonempty filesystem path, got {path!r}")
    source = Path(path).expanduser()
    if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageValidationError(f"image {source}: unsupported extension {source.suffix!r}")
    pil = _dependency("PIL.Image")
    np = _dependency("numpy")
    try:
        with source.open("rb") as stream:
            # Pillow would silently reduce 16-bit RGB PNGs to uint8. Reject them
            # rather than changing the source precision without an approved policy.
            header = stream.read(25)
            if header[:8] == b"\x89PNG\r\n\x1a\n" and len(header) == 25 and header[24] != 8:
                raise ImageValidationError(f"image {source}: expected 8-bit PNG, got bit depth {header[24]}")
            stream.seek(0)
            with pil.open(stream) as decoded:
                if decoded.format not in ("PNG", "JPEG"):
                    raise ImageValidationError(f"image {source}: unsupported decoded format {decoded.format!r}")
                if decoded.mode != "RGB":
                    raise ImageValidationError(f"image {source}: expected RGB, got mode {decoded.mode!r}")
                if getattr(decoded, "n_frames", 1) != 1:
                    raise ImageValidationError(f"image {source}: animated or multi-frame images are unsupported")
                decoded.load()
                pixels = np.array(decoded, copy=True)
    except (OSError, SyntaxError, ValueError, pil.DecompressionBombError) as exc:
        if isinstance(exc, ImageValidationError):
            raise
        raise ImageReadError(f"image {source}: cannot read/decode file: {exc}") from exc
    return array_to_tensor(pixels, color_order="RGB")
