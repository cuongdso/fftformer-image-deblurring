"""Group-owned image I/O and dataset adapters; see contracts C-02 and C-03."""

from .errors import (
    DataDependencyError, DataError, DatasetValidationError,
    ImageReadError, ImageValidationError,
)
from .image import SUPPORTED_EXTENSIONS, array_to_tensor, read_image, validate_image_tensor
from .dataset import (
    ImageDataset, PreprocessingMetadata, Sample, SampleRecord,
    build_paired_manifest, build_single_manifest, load_manifest, save_manifest,
)

__all__ = [
    "DataDependencyError", "DataError", "DatasetValidationError", "ImageReadError",
    "ImageValidationError", "SUPPORTED_EXTENSIONS", "array_to_tensor", "read_image",
    "validate_image_tensor",
    "ImageDataset", "PreprocessingMetadata", "Sample", "SampleRecord",
    "build_paired_manifest", "build_single_manifest", "load_manifest", "save_manifest",
]
