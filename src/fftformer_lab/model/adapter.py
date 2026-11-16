"""Explicit upstream construction and C-02 tensor validation for M1-WS1-02."""

from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
from types import ModuleType
from typing import Any, TYPE_CHECKING

from .spec import ModelConfigurationError, ModelConfig, ModelSpec, UPSTREAM_ARCHITECTURE_SHA256

if TYPE_CHECKING:
    import torch


class ModelBackendError(RuntimeError):
    """A dependency, device, constructor or model forward failed."""


class ModelInputError(ValueError):
    """An input tensor violates the model-facing C-02 boundary."""


@dataclass(frozen=True)
class ModelMetadata:
    """Construction identity and explicit checkpoint loading state."""

    architecture_id: str
    implementation_id: str
    implementation_version: str
    architecture_sha256: str
    config: ModelConfig
    requested_device: str
    resolved_device: str
    parameter_count: int
    state_dict_entry_count: int
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    checkpoint_loaded: bool = False
    initialization: str = "random"
    checkpoint_size_bytes: int | None = None
    checkpoint_schema: str | None = None
    checkpoint_map_location: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable metadata without exposing mutable config state."""
        return {
            "architecture_id": self.architecture_id,
            "implementation_id": self.implementation_id,
            "implementation_version": self.implementation_version,
            "architecture_sha256": self.architecture_sha256,
            "config": self.config.to_constructor_kwargs(),
            "requested_device": self.requested_device, "resolved_device": self.resolved_device,
            "parameter_count": self.parameter_count, "state_dict_entry_count": self.state_dict_entry_count,
            "checkpoint_path": self.checkpoint_path, "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_loaded": self.checkpoint_loaded, "initialization": self.initialization,
            "checkpoint_size_bytes": self.checkpoint_size_bytes,
            "checkpoint_schema": self.checkpoint_schema,
            "checkpoint_map_location": self.checkpoint_map_location,
        }


class ModelAdapter:
    """Wrap an evaluation-mode model; padding, image conversion and saving belong to callers."""

    def __init__(self, model: "torch.nn.Module", metadata: ModelMetadata) -> None:
        self.model = model
        self.metadata = metadata

    def __call__(self, image: "torch.Tensor") -> "torch.Tensor":
        """Validate and forward one NCHW tensor without modifying its values."""
        return self.forward(image)

    def forward(self, image: "torch.Tensor") -> "torch.Tensor":
        """Check C-02 input and output shape/dtype/finiteness, without clamp or padding."""
        torch = _torch()
        if not isinstance(image, torch.Tensor):
            raise ModelInputError("model input: expected a torch.Tensor (RGB float32 NCHW)")
        if image.dtype != torch.float32:
            raise ModelInputError(f"model input dtype: expected torch.float32, got {image.dtype}")
        if image.ndim != 4 or image.shape[1] != 3 or any(size <= 0 for size in image.shape):
            raise ModelInputError(f"model input shape: expected nonempty [N,3,H,W], got {tuple(image.shape)}")
        height, width = image.shape[-2:]
        if height % 32 or width % 32:
            raise ModelInputError(f"model input shape {height}x{width}: H/W must be multiples of 32; WS3 supplies padding")
        if str(image.device) != self.metadata.resolved_device:
            raise ModelInputError(f"model input device: expected {self.metadata.resolved_device}, got {image.device}")
        if not bool(torch.isfinite(image).all()):
            raise ModelInputError("model input: values must be finite (no NaN/Inf)")
        if bool((image < 0).any()) or bool((image > 1).any()):
            raise ModelInputError("model input range: expected [0,1]; conversion must be explicit")
        try:
            output = self.model(image)
        except RuntimeError as exc:
            raise ModelBackendError(f"fftformer forward on {image.device}, input {tuple(image.shape)}: {exc}") from exc
        if not isinstance(output, torch.Tensor):
            raise ModelBackendError("fftformer output: expected a torch.Tensor")
        if output.shape != image.shape or output.dtype != image.dtype or output.device != image.device:
            raise ModelBackendError("fftformer output: shape, dtype and device must match the model-facing input")
        if not bool(torch.isfinite(output).all()):
            raise ModelBackendError("fftformer output: non-finite values")
        return output


def _torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise ModelBackendError("model construction/forward requires PyTorch; ModelSpec itself does not") from exc


def _resolve_device(torch: Any, requested: str) -> "torch.device":
    if requested == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise ModelBackendError(f"device {requested!r}: CUDA is unavailable; no automatic CPU fallback")
    index = torch.cuda.current_device() if requested == "cuda" else int(requested.split(":")[1])
    if index >= torch.cuda.device_count():
        raise ModelBackendError(f"device {requested!r}: CUDA index {index} is unavailable")
    return torch.device("cuda", index)


def build_model(spec: ModelSpec) -> ModelAdapter:
    """Construct an uninitialized upstream model; checkpoint loading is reserved for WS1-03."""
    if not isinstance(spec, ModelSpec):
        raise ModelConfigurationError("build_model: expected ModelSpec")
    if spec.checkpoint is not None:
        raise NotImplementedError("checkpoint loading belongs to M1-WS1-03; build_model never ignores a checkpoint request")
    architecture_file: Path = spec.architecture_file.resolve()
    try:
        source_bytes = architecture_file.read_bytes()
    except OSError as exc:
        raise ModelBackendError(f"architecture_file {spec.architecture_file}: cannot read upstream source: {exc}") from exc
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    if source_hash != UPSTREAM_ARCHITECTURE_SHA256:
        raise ModelBackendError(f"architecture_file {spec.architecture_file}: SHA-256 differs from the WS1-01 audited source")
    torch = _torch()
    device = _resolve_device(torch, spec.device)
    # Execute only the explicitly selected, fingerprint-verified source. No BasicSR
    # package import, sys.path changes, submodule edits or bytecode writes are needed.
    module = ModuleType("_fftformer_lab_audited_upstream")
    try:
        exec(compile(source_bytes, str(architecture_file), "exec"), module.__dict__)
        # Override an enclosing default-device context without changing caller
        # defaults. C-02 also requires float32 even when its default dtype differs.
        with torch.device("cpu"):
            model = module.fftformer(**spec.config.to_constructor_kwargs())
        model = model.to(device=device, dtype=torch.float32).eval()
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        raise ModelBackendError(f"fftformer construction from {spec.architecture_file}, device {device}: {exc}") from exc
    metadata = ModelMetadata(
        architecture_id=spec.architecture_id, implementation_id=spec.implementation_id,
        implementation_version=spec.implementation_version, architecture_sha256=source_hash,
        config=spec.config, requested_device=spec.device, resolved_device=str(device),
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        state_dict_entry_count=len(model.state_dict()),
    )
    return ModelAdapter(model, metadata)
