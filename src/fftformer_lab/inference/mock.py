"""Mock model test double for inference pipeline verification without GPU/weights."""

from typing import Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore


class MockModel:
    """Mock model implementing C-01/C-02 model-facing protocol for testing.

    Clearly identified as a mock double; does not perform real FFTformer restoration.
    """

    def __init__(self, mode: str = "identity", device: str = "cpu") -> None:
        if mode not in ("identity", "invert", "shift"):
            raise ValueError(f"Unknown mock mode: {mode!r}")
        self.mode = mode
        self.device = device
        self.is_mock = True
        self.architecture_id = "mock_fftformer"
        self.implementation_id = "fftformer_lab.mock"
        self.implementation_version = "mock-1.0"
        self.metadata = {
            "architecture_id": self.architecture_id,
            "implementation_id": self.implementation_id,
            "implementation_version": self.implementation_version,
            "is_mock": True,
            "mode": self.mode,
            "device": self.device,
        }

    def __call__(self, image: Any) -> Any:
        return self.forward(image)

    def forward(self, image: Any) -> Any:
        """Validate model-facing NCHW input and apply mock transformation."""
        if torch is None:
            raise RuntimeError("MockModel forward requires PyTorch")
        if not isinstance(image, torch.Tensor):
            raise ValueError("Expected torch.Tensor input")
        if image.dtype != torch.float32:
            raise ValueError(f"Expected torch.float32, got {image.dtype}")
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected [N, 3, H, W], got {tuple(image.shape)}")
        h, w = image.shape[-2:]
        if h % 32 != 0 or w % 32 != 0:
            raise ValueError(f"Spatial dimensions {h}x{w} must be multiples of 32")
        if not bool(torch.isfinite(image).all()):
            raise ValueError("Input contains non-finite values (NaN/Inf)")
        if bool((image < 0.0).any()) or bool((image > 1.0).any()):
            raise ValueError("Input values must be in range [0, 1]")

        if self.mode == "identity":
            return image.clone()
        elif self.mode == "invert":
            return 1.0 - image
        elif self.mode == "shift":
            return torch.clamp(image + 0.05, 0.0, 1.0)
        return image.clone()
