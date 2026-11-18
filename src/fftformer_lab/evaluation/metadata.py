"""C-06 JSON metadata writer for C-05 evaluation summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ResolvedEvaluationConfig
from .engine import EvaluationSummary
from .errors import ConfigurationError


def build_experiment_metadata(config: ResolvedEvaluationConfig, summary: EvaluationSummary) -> dict[str, Any]:
    """Build traceable metadata without fabricating absent model/checkpoint facts."""
    return {
        "experiment_id": config.experiment_id, "measurement_kind": "group-measured",
        "evaluation_source": "precomputed-prediction-images", "config": config.to_dict(),
        "protocol": config.protocol.to_dict(), "dataset": {
            "prediction_root": str(config.prediction_root), "target_root": str(config.target_root),
            "identity": config.dataset_identity,
        },
        "model": {"identity": config.model_identity, "checkpoint_identity": config.checkpoint_identity,
                  "checkpoint_sha256": config.checkpoint_sha256},
        "provenance": {"group_commit": config.group_commit}, "summary": summary.to_dict(),
    }


def write_experiment_metadata(path: str | Path, metadata: dict[str, Any], *, overwrite: bool = False) -> Path:
    """Write strict JSON atomically, preserving the WS4 infinity-as-string policy."""
    destination = Path(path).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise ConfigurationError(f"output metadata already exists: {destination}")
    try:
        text = json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8", newline="\n")
        temporary.replace(destination)
    except (OSError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"cannot write metadata {destination}: {exc}") from exc
    return destination
