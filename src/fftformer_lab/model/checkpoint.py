"""Strict raw-state-dict checkpoint loading for M1-WS1-03 / C-01."""

from collections.abc import Mapping
from dataclasses import replace
import hashlib
from typing import Any

from .adapter import ModelAdapter, ModelBackendError, _torch, build_model
from .spec import ModelConfigurationError, ModelSpec


class CheckpointError(ModelBackendError):
    """The checkpoint file cannot be read or deserialized."""


class CheckpointSchemaError(CheckpointError):
    """The artifact is not the supported raw tensor state dictionary."""


class CheckpointCompatibilityError(CheckpointError):
    """Checkpoint keys, tensor shapes or dtypes differ from the requested model."""


def _validate_state(state: Any, model_state: Mapping[str, Any], torch: Any, context: str) -> None:
    if not isinstance(state, Mapping) or not state:
        raise CheckpointSchemaError(f"{context}: expected a nonempty raw tensor state_dict, got {type(state).__name__}")
    if any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in state.items()):
        raise CheckpointSchemaError(f"{context}: expected string keys and tensor values; wrapper schemas are not supported")
    missing = sorted(set(model_state) - set(state))
    unexpected = sorted(set(state) - set(model_state))
    mismatches = []
    for key in sorted(set(model_state) & set(state)):
        expected, actual = model_state[key], state[key]
        if actual.shape != expected.shape:
            mismatches.append(f"{key}: shape {tuple(actual.shape)} != expected {tuple(expected.shape)}")
        elif actual.dtype != expected.dtype:
            mismatches.append(f"{key}: dtype {actual.dtype} != expected {expected.dtype}")
    details = []
    if missing:
        details.append(f"missing keys ({len(missing)}): {missing[:10]}")
    if unexpected:
        details.append(f"unexpected keys ({len(unexpected)}): {unexpected[:10]}")
    if mismatches:
        details.append(f"tensor mismatches ({len(mismatches)}): {mismatches[:10]}")
    if details:
        raise CheckpointCompatibilityError(f"{context}: " + "; ".join(details))


def load_model(spec: ModelSpec) -> ModelAdapter:
    """Build and strictly load a raw state_dict, returning an eval model and checkpoint identity."""
    if not isinstance(spec, ModelSpec) or spec.checkpoint is None:
        raise ModelConfigurationError("load_model: ModelSpec with an explicit checkpoint path is required")
    checkpoint = spec.checkpoint.resolve()
    context = f"checkpoint {spec.checkpoint} for {spec.architecture_id} ({spec.implementation_version}), device {spec.device}"
    try:
        with checkpoint.open("rb") as artifact:
            adapter = build_model(replace(spec, checkpoint=None))
            torch = _torch()
            digest = hashlib.sha256()
            for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
                digest.update(chunk)
            size_bytes = artifact.tell()
            artifact.seek(0)
            try:
                state = torch.load(artifact, map_location=adapter.metadata.resolved_device, weights_only=True)
            except Exception as exc:
                raise CheckpointError(f"{context}: cannot deserialize with weights_only=True: {exc}") from exc
    except OSError as exc:
        raise CheckpointError(f"{context}: cannot read file: {exc}") from exc

    _validate_state(state, adapter.model.state_dict(), torch, context)
    try:
        result = adapter.model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise CheckpointCompatibilityError(f"{context}: strict state_dict loading failed: {exc}") from exc
    if result.missing_keys or result.unexpected_keys:
        raise CheckpointCompatibilityError(f"{context}: strict loading reported incompatible keys: {result}")
    adapter.model.eval()
    metadata = replace(
        adapter.metadata, checkpoint_path=spec.checkpoint.as_posix(),
        checkpoint_sha256=digest.hexdigest(), checkpoint_size_bytes=size_bytes,
        checkpoint_schema="raw_state_dict", checkpoint_map_location=adapter.metadata.resolved_device,
        checkpoint_loaded=True, initialization="checkpoint",
    )
    return ModelAdapter(adapter.model, metadata)
