#!/usr/bin/env python3
"""Report environment facts without installing packages or requiring a GPU."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import sys


def package_version(name: str) -> str:
    """Return installed distribution version or a stable missing marker."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-torch",
        action="store_true",
        help="return non-zero when PyTorch is unavailable",
    )
    args = parser.parse_args()

    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"Platform: {platform.platform()}")
    for name in ("torch", "torchvision", "numpy", "opencv-python", "einops"):
        print(f"{name}: {package_version(name)}")

    try:
        import torch
    except ImportError:
        print("CUDA: unavailable (PyTorch is not installed)")
        return 2 if args.require_torch else 0

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA runtime: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device 0: {torch.cuda.get_device_name(0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
