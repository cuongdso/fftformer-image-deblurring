"""Standard-library C-06 parser for reproducible evaluation jobs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .errors import ConfigurationError
from .protocol import MetricProtocol, get_metric_protocol

SCHEMA_VERSION = 1
CONTRACT_VERSION = "C-06-v1"
_TOP_LEVEL = {"schema_version", "contract_version", "experiment_id", "protocol_id", "dataset", "output_path", "checkpoint", "model", "provenance"}
_DATASET = {"prediction_root", "target_root", "identity"}
_OPTIONAL_IDENTITY = {"identity", "sha256"}


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _object(value: Any, allowed: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) - allowed:
        unknown = sorted(set(value) - allowed) if isinstance(value, dict) else []
        raise ConfigurationError(f"{context}: expected object with known keys; unknown={unknown}")
    return value


def _text(value: Any, context: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{context}: expected non-empty string")
    return value


def _resolve(value: Any, base: Path, context: str, *, directory: bool) -> Path:
    text = _text(value, context)
    assert text is not None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base / path
    try:
        path = path.resolve()
    except OSError as exc:
        raise ConfigurationError(f"{context}: invalid path {text!r}: {exc}") from exc
    if directory and not path.is_dir():
        raise ConfigurationError(f"{context}: directory does not exist: {path}")
    return path


@dataclass(frozen=True)
class ResolvedEvaluationConfig:
    """Validated C-06 config with paths resolved relative to its source file."""

    source_path: Path
    schema_version: int
    contract_version: str
    experiment_id: str
    protocol: MetricProtocol
    prediction_root: Path
    target_root: Path
    output_path: Path
    dataset_identity: str | None = None
    checkpoint_identity: str | None = None
    checkpoint_sha256: str | None = None
    model_identity: str | None = None
    group_commit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "contract_version": self.contract_version,
            "experiment_id": self.experiment_id, "protocol_id": self.protocol.protocol_id,
            "prediction_root": str(self.prediction_root), "target_root": str(self.target_root),
            "output_path": str(self.output_path), "dataset_identity": self.dataset_identity,
            "checkpoint_identity": self.checkpoint_identity, "checkpoint_sha256": self.checkpoint_sha256,
            "model_identity": self.model_identity, "group_commit": self.group_commit,
        }


def load_evaluation_config(path: str | Path) -> ResolvedEvaluationConfig:
    """Load schema v1 JSON and validate C-06 paths without CV/runtime dependencies."""
    source = Path(path).expanduser().resolve()
    try:
        document = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"config {source}: cannot read JSON: {exc}") from exc
    if not isinstance(document, dict) or set(document) != _TOP_LEVEL:
        missing = sorted(_TOP_LEVEL - set(document)) if isinstance(document, dict) else sorted(_TOP_LEVEL)
        unknown = sorted(set(document) - _TOP_LEVEL) if isinstance(document, dict) else []
        raise ConfigurationError(f"config {source}: expected exact schema keys; missing={missing}, unknown={unknown}")
    if document["schema_version"] != SCHEMA_VERSION:
        raise ConfigurationError(f"config {source}: unsupported schema_version {document['schema_version']!r}")
    if document["contract_version"] != CONTRACT_VERSION:
        raise ConfigurationError(f"config {source}: expected contract_version {CONTRACT_VERSION!r}")
    dataset = _object(document["dataset"], _DATASET, "dataset")
    if set(dataset) not in ({"prediction_root", "target_root"}, _DATASET):
        raise ConfigurationError("dataset: prediction_root and target_root are required")
    checkpoint = _object(document["checkpoint"], _OPTIONAL_IDENTITY, "checkpoint") if document["checkpoint"] is not None else {}
    model = _object(document["model"], {"identity"}, "model") if document["model"] is not None else {}
    provenance = _object(document["provenance"], {"group_commit"}, "provenance") if document["provenance"] is not None else {}
    try:
        protocol = get_metric_protocol(document["protocol_id"])
    except Exception as exc:
        raise ConfigurationError(f"config {source}: invalid protocol_id: {exc}") from exc
    output = _resolve(document["output_path"], source.parent, "output_path", directory=False)
    if not output.parent.is_dir():
        raise ConfigurationError(f"output_path: parent directory does not exist: {output.parent}")
    return ResolvedEvaluationConfig(
        source, SCHEMA_VERSION, CONTRACT_VERSION, _text(document["experiment_id"], "experiment_id") or "", protocol,
        _resolve(dataset["prediction_root"], source.parent, "dataset.prediction_root", directory=True),
        _resolve(dataset["target_root"], source.parent, "dataset.target_root", directory=True), output,
        _text(dataset.get("identity"), "dataset.identity", optional=True),
        _text(checkpoint.get("identity"), "checkpoint.identity", optional=True),
        _text(checkpoint.get("sha256"), "checkpoint.sha256", optional=True),
        _text(model.get("identity"), "model.identity", optional=True),
        _text(provenance.get("group_commit"), "provenance.group_commit", optional=True),
    )
