"""Deterministic folder/JSON manifests and lazy C-03 samples (M1-WS2-03)."""

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING, overload

from .errors import DataError, DatasetValidationError
from .image import SUPPORTED_EXTENSIONS, read_image, validate_image_tensor

if TYPE_CHECKING:
    import torch


def _image_path(value: str | Path, context: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise DatasetValidationError(f"{context}: expected a nonempty image path")
    try:
        path = Path(value).expanduser().resolve()
    except (OSError, ValueError, RuntimeError) as exc:
        raise DatasetValidationError(f"{context}: invalid image path {value!r}: {exc}") from exc
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise DatasetValidationError(f"{context} {path}: unsupported extension {path.suffix!r}")
    return path


@dataclass(frozen=True)
class SampleRecord:
    """Identify one source and its optional ground truth without decoding either."""

    sample_id: str
    source_path: Path
    target_path: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip() or "\0" in self.sample_id:
            raise DatasetValidationError("sample_id: expected a nonempty string without NUL characters")
        object.__setattr__(self, "source_path", _image_path(self.source_path, f"sample {self.sample_id!r} source"))
        if self.target_path is not None:
            object.__setattr__(self, "target_path", _image_path(self.target_path, f"sample {self.sample_id!r} target"))

    @property
    def has_ground_truth(self) -> bool:
        """True only when an explicit target path was provided."""
        return self.target_path is not None


@dataclass(frozen=True)
class PreprocessingMetadata:
    """Describe the fixed, non-geometric WS2 file-reading policy."""

    original_shape: tuple[int, int, int]

    def to_dict(self) -> dict[str, Any]:
        """Return metadata suitable for experiment logging by consumers."""
        return {
            "policy_version": "ws2-image-v1",
            "reader": "pillow",
            "source_color_order": "RGB",
            "color_space": "RGB", "dtype": "float32", "layout": "CHW",
            "range": [0.0, 1.0], "normalization": "uint8-divide-255",
            "original_shape": list(self.original_shape),
            "resize": None, "crop": None, "padding": [0, 0, 0, 0],
            "exif_orientation": "stored-pixels",
        }


@dataclass(frozen=True)
class Sample:
    """One decoded C-03 sample, ready for WS3/WS4 consumers."""

    image: "torch.Tensor"
    target: "torch.Tensor | None"
    sample_id: str
    source_path: Path
    preprocessing_metadata: PreprocessingMetadata
    has_ground_truth: bool
    target_path: Path | None = None

    def __post_init__(self) -> None:
        record = SampleRecord(self.sample_id, self.source_path, self.target_path)
        object.__setattr__(self, "source_path", record.source_path)
        object.__setattr__(self, "target_path", record.target_path)
        if type(self.has_ground_truth) is not bool or self.has_ground_truth != (self.target is not None):
            raise DatasetValidationError(f"sample {self.sample_id!r}: has_ground_truth must match target presence")
        if not self.has_ground_truth and self.target_path is not None:
            raise DatasetValidationError(f"sample {self.sample_id!r}: non-GT sample cannot have a target_path")
        validate_image_tensor(self.image, context=f"sample {self.sample_id!r} input")
        if self.target is not None:
            validate_image_tensor(self.target, context=f"sample {self.sample_id!r} target")
            if self.image.shape != self.target.shape:
                raise DatasetValidationError(
                    f"sample {self.sample_id!r}: size mismatch between input {self.source_path} "
                    f"{tuple(self.image.shape)} and target {self.target_path} {tuple(self.target.shape)}"
                )
        if not isinstance(self.preprocessing_metadata, PreprocessingMetadata):
            raise DatasetValidationError(f"sample {self.sample_id!r}: expected PreprocessingMetadata")
        if tuple(self.image.shape) != self.preprocessing_metadata.original_shape:
            raise DatasetValidationError(f"sample {self.sample_id!r}: preprocessing original_shape mismatch")


def _validated_records(records: Iterable[SampleRecord]) -> tuple[SampleRecord, ...]:
    try:
        entries = tuple(records)
    except TypeError as exc:
        raise DatasetValidationError("manifest: expected an iterable of SampleRecord") from exc
    if not entries:
        raise DatasetValidationError("manifest: no samples")
    ids: set[str] = set()
    sources: set[Path] = set()
    targets: set[Path] = set()
    for entry in entries:
        if not isinstance(entry, SampleRecord):
            raise DatasetValidationError("manifest: every entry must be a SampleRecord")
        if entry.sample_id in ids:
            raise DatasetValidationError(f"manifest: duplicate sample_id {entry.sample_id!r}")
        ids.add(entry.sample_id)
        for role, path, seen in (("source", entry.source_path, sources), ("target", entry.target_path, targets)):
            if path is None:
                continue
            if path in seen:
                raise DatasetValidationError(f"sample {entry.sample_id!r}: duplicate {role} path {path}")
            seen.add(path)
            if not path.is_file():
                raise DatasetValidationError(f"sample {entry.sample_id!r}: missing {role} file {path}")
    return tuple(sorted(entries, key=lambda entry: entry.sample_id))


def _scan(root: str | Path, *, recursive: bool) -> dict[str, Path]:
    if not isinstance(root, (str, Path)) or not str(root).strip():
        raise DatasetValidationError("dataset root: expected a nonempty path")
    try:
        directory = Path(root).expanduser().resolve()
    except (OSError, ValueError, RuntimeError) as exc:
        raise DatasetValidationError(f"dataset root: invalid path {root!r}: {exc}") from exc
    if not directory.is_dir():
        raise DatasetValidationError(f"dataset root {directory}: missing or not a directory")
    if type(recursive) is not bool:
        raise DatasetValidationError("recursive: expected bool")
    paths: dict[str, Path] = {}
    try:
        for path in sorted(directory.rglob("*") if recursive else directory.iterdir()):
            if path.is_dir():
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise DatasetValidationError(f"dataset root {directory}: unsupported extension for {path}")
            paths[path.relative_to(directory).as_posix()] = path
    except OSError as exc:
        raise DatasetValidationError(f"dataset root {directory}: cannot scan: {exc}") from exc
    if not paths:
        raise DatasetValidationError(f"dataset root {directory}: no images")
    return paths


def build_paired_manifest(
    input_root: str | Path, target_root: str | Path, *, recursive: bool = False,
) -> tuple[SampleRecord, ...]:
    """Match RGB file candidates 1:1 by full relative filename, including extension."""
    inputs = _scan(input_root, recursive=recursive)
    targets = _scan(target_root, recursive=recursive)
    missing = sorted(inputs.keys() - targets.keys())
    extra = sorted(targets.keys() - inputs.keys())
    if missing or extra:
        raise DatasetValidationError(f"paired manifest: missing targets {missing}; targets without inputs {extra}")
    return _validated_records(SampleRecord(sid, inputs[sid], targets[sid]) for sid in sorted(inputs))


def build_single_manifest(input_root: str | Path, *, recursive: bool = False) -> tuple[SampleRecord, ...]:
    """List images without inventing ground truth, sorted by relative filename."""
    return _validated_records(SampleRecord(sid, path) for sid, path in _scan(input_root, recursive=recursive).items())


def save_manifest(records: Iterable[SampleRecord], path: str | Path) -> None:
    """Save local manifest schema v1 with paths relative to its JSON file."""
    entries = _validated_records(records)
    destination = Path(path).expanduser().resolve()
    if destination.suffix.lower() != ".json":
        raise DatasetValidationError(f"manifest {destination}: expected .json extension")
    try:
        def relative(image_path: Path | None) -> str | None:
            return None if image_path is None else Path(os.path.relpath(image_path, destination.parent)).as_posix()

        samples = [
            {"sample_id": entry.sample_id, "source_path": relative(entry.source_path),
             "target_path": relative(entry.target_path), "has_ground_truth": entry.has_ground_truth}
            for entry in entries
        ]
        with destination.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({"schema_version": 1, "samples": samples}, indent=2) + "\n")
    except (OSError, ValueError) as exc:
        raise DatasetValidationError(f"manifest {destination}: cannot save relative paths: {exc}") from exc


def load_manifest(path: str | Path) -> tuple[SampleRecord, ...]:
    """Read schema v1, resolving paths relative to the manifest rather than CWD."""
    source = Path(path).expanduser().resolve()
    try:
        def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON field {key!r}")
                result[key] = value
            return result

        document = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=unique_keys)
    except (OSError, ValueError) as exc:
        raise DatasetValidationError(f"manifest {source}: cannot read JSON: {exc}") from exc
    if not isinstance(document, dict) or set(document) != {"schema_version", "samples"}:
        raise DatasetValidationError(f"manifest {source}: expected schema_version and samples fields")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise DatasetValidationError(f"manifest {source}: unsupported schema_version")
    if not isinstance(document["samples"], list):
        raise DatasetValidationError(f"manifest {source}: samples must be a list")
    records = []
    fields = {"sample_id", "source_path", "target_path", "has_ground_truth"}
    for index, item in enumerate(document["samples"]):
        context = f"manifest {source}, sample index {index}"
        if not isinstance(item, dict) or set(item) != fields:
            raise DatasetValidationError(f"{context}: expected fields {sorted(fields)}")
        target = item["target_path"]
        if type(item["has_ground_truth"]) is not bool or item["has_ground_truth"] != (target is not None):
            raise DatasetValidationError(f"{context}: has_ground_truth must match target_path presence")
        for name in ("source_path", "target_path"):
            value = item[name]
            if name == "target_path" and value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise DatasetValidationError(f"{context}: {name} must be a nonempty path string")
        records.append(SampleRecord(item["sample_id"], source.parent/item["source_path"],
                                    None if target is None else source.parent/target))
    return _validated_records(records)


class ImageDataset(Sequence[Sample]):
    """Lazily decode a validated manifest in sample_id order, without augmentation."""

    def __init__(self, records: Iterable[SampleRecord]) -> None:
        self._records = _validated_records(records)

    @property
    def records(self) -> tuple[SampleRecord, ...]:
        """Expose immutable ordered paths for logging and manifest export."""
        return self._records

    def __len__(self) -> int:
        return len(self._records)

    @overload
    def __getitem__(self, index: int) -> Sample: ...

    @overload
    def __getitem__(self, index: slice) -> list[Sample]: ...

    def __getitem__(self, index: int | slice) -> Sample | list[Sample]:
        """Read one sample (or a slice), reporting its ID and file paths on errors."""
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        record = self._records[index]
        try:
            image = read_image(record.source_path)
            target = None if record.target_path is None else read_image(record.target_path)
            return Sample(image, target, record.sample_id, record.source_path,
                          PreprocessingMetadata(tuple(image.shape)), record.has_ground_truth, record.target_path)
        except DataError as exc:
            raise type(exc)(f"sample {record.sample_id!r} (input={record.source_path}, target={record.target_path}): {exc}") from exc

    def __iter__(self) -> Iterator[Sample]:
        """Decode each sample once in manifest order."""
        for index in range(len(self)):
            yield self[index]
