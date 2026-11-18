"""Image output serialization conforming to Contract C-02 / C-04."""

from pathlib import Path
from typing import Any
import importlib

from .types import InferenceIOError, InferenceInputError

SUPPORTED_WRITE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg"})


def _import_dependency(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise InferenceIOError(f"Image writing requires dependency '{name}': {exc}") from exc


def write_image(
    image: Any,
    path: str | Path,
    *,
    quality: int = 95,
) -> Path:
    """Save an RGB float32 CHW [3, H, W] tensor in [0, 1] as an 8-bit PNG/JPEG image.

    Conforms to Contract C-02 / C-04:
    - Serialization format: 8-bit RGB PNG or JPEG.
    - Clamp policy: values are clamped to [0.0, 1.0].
    - Quantization: round(clamped * 255.0) converted to uint8.
    """
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise InferenceIOError(f"Output path must be a non-empty string or Path, got {path!r}")

    dest = Path(path).expanduser().resolve()
    ext = dest.suffix.lower()
    if ext not in SUPPORTED_WRITE_EXTENSIONS:
        raise InferenceIOError(
            f"Unsupported output image format {dest.suffix!r}; supported formats: {sorted(SUPPORTED_WRITE_EXTENSIONS)}"
        )

    torch = _import_dependency("torch")
    np = _import_dependency("numpy")
    pil = _import_dependency("PIL.Image")

    if not isinstance(image, torch.Tensor):
        raise InferenceInputError(f"Expected torch.Tensor for write_image, got {type(image).__name__}")
    if image.dtype != torch.float32:
        raise InferenceInputError(f"Expected torch.float32 tensor for write_image, got {image.dtype}")
    if image.ndim != 3 or image.shape[0] != 3 or any(dim <= 0 for dim in image.shape):
        raise InferenceInputError(f"Expected 3D RGB tensor [3, H, W] with positive dimensions, got shape {tuple(image.shape)}")
    if not bool(torch.isfinite(image).all()):
        raise InferenceInputError("Cannot write image containing non-finite values (NaN/Inf)")

    # Explicit clamp to [0.0, 1.0] and quantization to 8-bit uint8 [0, 255]
    clamped = torch.clamp(image.detach().cpu(), 0.0, 1.0)
    quantized = (clamped * 255.0).round().to(torch.uint8)

    # Convert CHW -> HWC for PIL
    hwc = quantized.permute(1, 2, 0).numpy()

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        pil_img = pil.fromarray(hwc, mode="RGB")
        fmt = "PNG" if ext == ".png" else "JPEG"
        save_kwargs: dict[str, Any] = {}
        if fmt == "JPEG":
            save_kwargs["quality"] = quality
        pil_img.save(dest, format=fmt, **save_kwargs)
    except (OSError, ValueError) as exc:
        raise InferenceIOError(f"Failed to write image to {dest}: {exc}") from exc

    return dest
