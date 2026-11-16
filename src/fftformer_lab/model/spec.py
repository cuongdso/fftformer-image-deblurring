"""Validated, dependency-free model specifications for M1-WS1-02."""

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import Any


UPSTREAM_COMMIT = "01306b5d4072146a011c8c6e9d4927cf4607f1d5"
UPSTREAM_ARCHITECTURE_SHA256 = "dc0e1775eed3c3358f637084e9e345ec0246b66084487dc4220aad0a5cb66ddf"


class ModelConfigurationError(ValueError):
    """The requested model specification is invalid or unsupported."""


@dataclass(frozen=True)
class ModelConfig:
    """Constructor configuration; defaults match the audited GoPro network_g."""

    inp_channels: int = 3
    out_channels: int = 3
    dim: int = 48
    num_blocks: tuple[int, int, int] = (6, 6, 12)
    num_refinement_blocks: int = 4
    ffn_expansion_factor: float = 3.0
    bias: bool = False

    def __post_init__(self) -> None:
        for name in ("inp_channels", "out_channels", "dim", "num_refinement_blocks"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ModelConfigurationError(f"network_g.{name}: expected a positive integer, got {value!r}")
        if self.inp_channels != 3 or self.out_channels != 3:
            raise ModelConfigurationError("network_g: C-02 requires inp_channels=out_channels=3 (RGB)")
        if not isinstance(self.num_blocks, (tuple, list)) or len(self.num_blocks) != 3:
            raise ModelConfigurationError("network_g.num_blocks: expected three positive block counts")
        if any(type(count) is not int or count <= 0 for count in self.num_blocks):
            raise ModelConfigurationError("network_g.num_blocks: every block count must be a positive integer")
        object.__setattr__(self, "num_blocks", tuple(self.num_blocks))
        factor = self.ffn_expansion_factor
        if type(factor) not in (int, float):
            raise ModelConfigurationError("network_g.ffn_expansion_factor: expected a finite positive number")
        try:
            expanded = self.dim * factor
            valid_expansion = math.isfinite(factor) and factor > 0 and math.isfinite(expanded) and expanded >= 1
        except OverflowError:
            valid_expansion = False
        if not valid_expansion:
            raise ModelConfigurationError("network_g.ffn_expansion_factor: expanded channel count must be finite and positive")
        if type(self.bias) is not bool:
            raise ModelConfigurationError("network_g.bias: expected bool")

    @classmethod
    def from_mapping(cls, network_g: Mapping[str, Any]) -> "ModelConfig":
        """Validate a network_g mapping without loading YAML or resolving paths."""
        if not isinstance(network_g, Mapping):
            raise ModelConfigurationError("network_g: expected a mapping")
        values = dict(network_g)
        architecture = values.pop("type", "fftformer")
        if architecture != "fftformer":
            raise ModelConfigurationError(f"network_g.type: unsupported architecture {architecture!r}")
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ModelConfigurationError(f"network_g: unknown constructor fields {sorted(unknown, key=str)!r}")
        return cls(**values)

    def to_constructor_kwargs(self) -> dict[str, Any]:
        """Return a fresh constructor mapping with the canonical three levels."""
        return {
            "inp_channels": self.inp_channels, "out_channels": self.out_channels,
            "dim": self.dim, "num_blocks": list(self.num_blocks),
            "num_refinement_blocks": self.num_refinement_blocks,
            "ffn_expansion_factor": self.ffn_expansion_factor, "bias": self.bias,
        }


@dataclass(frozen=True)
class ModelSpec:
    """Identify the explicit upstream source, constructor, device and checkpoint request."""

    architecture_file: Path
    architecture_id: str = "fftformer"
    implementation_id: str = "upstream-fftformer"
    implementation_version: str = UPSTREAM_COMMIT
    config: ModelConfig = field(default_factory=ModelConfig)
    checkpoint: Path | None = None
    device: str = "cpu"
    strict: bool = True

    def __post_init__(self) -> None:
        if self.architecture_id != "fftformer":
            raise ModelConfigurationError(f"architecture_id: unsupported value {self.architecture_id!r}")
        if self.implementation_id != "upstream-fftformer" or self.implementation_version != UPSTREAM_COMMIT:
            raise ModelConfigurationError("implementation: only the WS1-01 audited upstream version is supported")
        if not isinstance(self.config, ModelConfig):
            raise ModelConfigurationError("config: expected ModelConfig; use ModelConfig.from_mapping for network_g")
        for name in ("architecture_file", "checkpoint"):
            value = getattr(self, name)
            if name == "checkpoint" and value is None:
                continue
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ModelConfigurationError(f"{name}: expected a nonempty filesystem path")
            object.__setattr__(self, name, Path(value).expanduser())
        if not isinstance(self.device, str) or re.fullmatch(r"cpu|cuda(?::(?:0|[1-9][0-9]*))?", self.device) is None:
            raise ModelConfigurationError(f"device: expected cpu, cuda or cuda:<index>, got {self.device!r}")
        if self.strict is not True:
            raise ModelConfigurationError("strict: only True is supported; checkpoint mismatches must not be hidden")
