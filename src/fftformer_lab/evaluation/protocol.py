"""Immutable, explicit metric protocol definitions for WS4-02."""

from dataclasses import dataclass
from typing import Final

from .errors import UnknownProtocolError


@dataclass(frozen=True)
class MetricProtocol:
    """Describe one reproducible metric representation and evaluator variant."""

    protocol_id: str
    version: str
    implementation_source: str
    color_order: str
    input_layout: str
    numeric_range: tuple[int, int]
    clamp_policy: str
    quantization_policy: str
    crop_border: int
    test_y_channel: bool
    ssim_variant: str
    device_requirement: str
    supported: bool
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.protocol_id or not self.version:
            raise ValueError("protocol_id and version must be non-empty")
        if self.color_order not in {"BGR"} or self.input_layout != "HWC":
            raise ValueError("WS4-02 BasicSR profiles require HWC BGR metric inputs")
        if self.numeric_range != (0, 255) or self.crop_border < 0:
            raise ValueError("numeric_range must be (0, 255) and crop_border must be non-negative")
        if not self.supported and not self.unavailable_reason:
            raise ValueError("unsupported protocol requires unavailable_reason")

    def to_dict(self) -> dict[str, object]:
        """Return stable, JSON-ready protocol provenance."""
        return {
            "protocol_id": self.protocol_id,
            "version": self.version,
            "implementation_source": self.implementation_source,
            "color_order": self.color_order,
            "input_layout": self.input_layout,
            "numeric_range": list(self.numeric_range),
            "clamp_policy": self.clamp_policy,
            "quantization_policy": self.quantization_policy,
            "crop_border": self.crop_border,
            "test_y_channel": self.test_y_channel,
            "ssim_variant": self.ssim_variant,
            "device_requirement": self.device_requirement,
            "supported": self.supported,
            "unavailable_reason": self.unavailable_reason,
        }


UPSTREAM_BASICSR: Final = MetricProtocol(
    protocol_id="upstream-basicsr",
    version="fftformer-pin-01306b5",
    implementation_source="third_party/FFTformer basicsr.metrics.psnr_ssim; validation tensor2img path",
    color_order="BGR", input_layout="HWC", numeric_range=(0, 255),
    clamp_policy="clamp RGB [0,1] before conversion",
    quantization_policy="round(value * 255) then uint8, matching tensor2img",
    crop_border=0, test_y_channel=False, ssim_variant="ssim3d=True",
    device_requirement="CUDA required by upstream _ssim_3d", supported=True,
)

GROUP_BASICSR_CPU: Final = MetricProtocol(
    protocol_id="group-basicsr-cpu",
    version="ws4-02-v1",
    implementation_source="group CPU port of audited BasicSR PSNR and calculate_ssim(..., ssim3d=False)",
    color_order="BGR", input_layout="HWC", numeric_range=(0, 255),
    clamp_policy="clamp RGB [0,1] before conversion",
    quantization_policy="round(value * 255) then uint8, matching tensor2img",
    crop_border=0, test_y_channel=False, ssim_variant="ssim3d=False",
    device_requirement="CPU", supported=True,
)

REALBLUR_OFFICIAL: Final = MetricProtocol(
    protocol_id="realblur-official",
    version="rimchang-RealBlur-ecc",
    implementation_source="rimchang/RealBlur evaluation_RealBlur_ecc.py",
    color_order="BGR", input_layout="HWC", numeric_range=(0, 255),
    clamp_policy="official file-based evaluator policy; not implemented in WS4-02",
    quantization_policy="official evaluator reads saved image files; not implemented in WS4-02",
    crop_border=0, test_y_channel=False, ssim_variant="ECC-aligned masked skimage SSIM",
    device_requirement="CPU with pinned legacy skimage/OpenCV", supported=False,
    unavailable_reason="RealBlur official evaluation requires ECC alignment, dataset layout, and file workflow outside WS4-02.",
)

_REGISTRY: Final[dict[str, MetricProtocol]] = {
    item.protocol_id: item for item in (UPSTREAM_BASICSR, GROUP_BASICSR_CPU, REALBLUR_OFFICIAL)
}


def get_metric_protocol(protocol: str | MetricProtocol) -> MetricProtocol:
    """Resolve a registered protocol without silently selecting a fallback."""
    if isinstance(protocol, MetricProtocol):
        registered = _REGISTRY.get(protocol.protocol_id)
        if registered is None:
            raise UnknownProtocolError(f"Unknown metric protocol {protocol.protocol_id!r}")
        if protocol != registered:
            raise UnknownProtocolError(
                f"Protocol {protocol.protocol_id!r} does not match its registered immutable definition"
            )
        return registered
    if not isinstance(protocol, str) or not protocol:
        raise UnknownProtocolError("protocol must be a non-empty registered ID or MetricProtocol")
    try:
        return _REGISTRY[protocol]
    except KeyError as exc:
        raise UnknownProtocolError(f"Unknown metric protocol {protocol!r}") from exc


def registered_metric_protocols() -> tuple[MetricProtocol, ...]:
    """Return registered profiles in stable declaration order."""
    return tuple(_REGISTRY.values())
