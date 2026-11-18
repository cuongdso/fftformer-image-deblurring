"""Errors owned by the WS2 data boundary."""


class DataError(RuntimeError):
    """Base error for data preparation and loading."""


class DataDependencyError(DataError):
    """An optional image or tensor dependency is unavailable."""


class ImageReadError(DataError):
    """An image file cannot be read or decoded."""


class ImageValidationError(DataError, ValueError):
    """An image violates the C-02 boundary."""


class DatasetValidationError(DataError, ValueError):
    """A manifest or sample violates the C-03 boundary."""
