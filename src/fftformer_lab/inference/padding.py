"""Reflection padding and cropping conforming to C-02/C-04 specifications."""

from typing import TYPE_CHECKING
from .types import InferencePaddingError, PaddingMetadata

if TYPE_CHECKING:
    import torch


def calculate_reflection_padding(height: int, width: int, factor: int = 32) -> PaddingMetadata:
    """Calculate right and bottom reflection padding required to reach a multiple of `factor`.

    Per C-02 and upstream baseline, padding is applied to the right and bottom using
    reflection mode so that H and W become multiples of 32.
    """
    if type(height) is not int or height <= 0:
        raise InferencePaddingError(f"Invalid height: expected positive int, got {height!r}")
    if type(width) is not int or width <= 0:
        raise InferencePaddingError(f"Invalid width: expected positive int, got {width!r}")
    if type(factor) is not int or factor <= 0:
        raise InferencePaddingError(f"Invalid factor: expected positive int, got {factor!r}")

    pad_h = (factor - (height % factor)) % factor
    pad_w = (factor - (width % factor)) % factor

    # Reflection padding in PyTorch requires each dimension to be strictly greater
    # than the padding size applied to that dimension.
    if pad_h > 0 and height <= pad_h:
        raise InferencePaddingError(
            f"Image spatial height {height} is too small for reflection padding of bottom={pad_h} "
            f"(reflection requires height > padding size {pad_h})"
        )
    if pad_w > 0 and width <= pad_w:
        raise InferencePaddingError(
            f"Image spatial width {width} is too small for reflection padding of right={pad_w} "
            f"(reflection requires width > padding size {pad_w})"
        )

    return PaddingMetadata(
        top=0,
        bottom=pad_h,
        left=0,
        right=pad_w,
        mode="reflect",
        original_shape=(height, width),
        padded_shape=(height + pad_h, width + pad_w),
    )


def pad_image(image: "torch.Tensor", padding: PaddingMetadata) -> "torch.Tensor":
    """Apply padding to an image tensor [C, H, W] or batch [N, C, H, W]."""
    import torch.nn.functional as F

    if padding.bottom == 0 and padding.right == 0 and padding.top == 0 and padding.left == 0:
        return image

    pad_tuple = (padding.left, padding.right, padding.top, padding.bottom)
    try:
        return F.pad(image, pad_tuple, mode=padding.mode)
    except RuntimeError as exc:
        raise InferencePaddingError(f"Padding operation failed with pad={pad_tuple}, mode={padding.mode}: {exc}") from exc


def unpad_image(image: "torch.Tensor", padding: PaddingMetadata) -> "torch.Tensor":
    """Crop image tensor back to its original (H, W) spatial dimensions."""
    orig_h, orig_w = padding.original_shape
    if image.ndim == 3:
        if image.shape[-2] < orig_h or image.shape[-1] < orig_w:
            raise InferencePaddingError(
                f"Cannot unpad tensor of shape {tuple(image.shape)} to original shape ({orig_h}, {orig_w})"
            )
        return image[:, :orig_h, :orig_w]
    elif image.ndim == 4:
        if image.shape[-2] < orig_h or image.shape[-1] < orig_w:
            raise InferencePaddingError(
                f"Cannot unpad tensor of shape {tuple(image.shape)} to original shape ({orig_h}, {orig_w})"
            )
        return image[:, :, :orig_h, :orig_w]
    else:
        raise InferencePaddingError(f"Expected 3D or 4D tensor for unpadding, got ndim={image.ndim}")
