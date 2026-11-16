#!/usr/bin/env python3
"""Independently reproduce WS1-01 CPU architecture/checkpoint smoke checks.

Usage (from the repository root):
    python -B scripts/ws1_01_verify.py --repo . --output docs/research/ws1-01-runtime-evidence.json

Requires: torch, einops, PyYAML. Does not modify any repository source files.
This is NOT an official benchmark or an implementation of WS1-02/WS1-03.
Only open checkpoint files you trust; weights_only=True reduces pickle risk.
"""


from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import einops
import torch
import yaml


def checksum_sha256(path: Path) -> str:
    """Return the SHA-256 of an audit input or script."""
    sha = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def evidence_path(path: Path, repo: Path) -> str:
    """Record a repository-relative path or redact an external directory."""
    try:
        return path.relative_to(repo).as_posix()
    except ValueError:
        return f"<external>/{path.name}"


def main() -> int:
    """Run the CPU integration audit and optionally write portable evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Path to group repository root")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report destination")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override checkpoint path if stored outside repository")
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    upstream = repo / "third_party" / "FFTformer"
    arch_py = upstream / "basicsr" / "models" / "archs" / "fftformer_arch.py"
    gopro_yml = upstream / "options" / "train" / "GoPro.yml"
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else (upstream / "pretrain_model" / "fftformer_GoPro.pth")
    for required in (arch_py, gopro_yml, checkpoint):
        if not required.is_file():
            parser.error(f"Missing {required}. Did you initialize the submodule and get checkpoint weights?")

    torch.set_num_threads(2)
    # Importing the upstream file must not create __pycache__ in the submodule.
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("ws1_pinned_fftformer_arch", arch_py)
    assert spec is not None and spec.loader is not None
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)

    config_raw = yaml.safe_load(gopro_yml.read_text(encoding="utf-8"))["network_g"]
    assert config_raw["type"] == "fftformer", f"Unexpected architecture: {config_raw['type']}"
    model_args = {key: val for key, val in config_raw.items() if key != "type"}
    model = source.fftformer(**model_args).cpu()
    model.eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)

    evidence: dict = {
        "audit": "WS1-01", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_source": "group-measured",
        "test_kind": "integration-cpu-checkpoint",
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "einops": einops.__version__, "pyyaml": yaml.__version__,
            "os": platform.system(), "machine": platform.machine(),
            "device": "cpu", "torch_num_threads": torch.get_num_threads(),
        },
        "paths": {
            "upstream_architecture": evidence_path(arch_py, repo),
            "config": evidence_path(gopro_yml, repo),
            "checkpoint": evidence_path(checkpoint, repo),
            "verification_script": evidence_path(Path(__file__).resolve(), repo),
        },
        "upstream_commit": subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True,
        ).strip(),
        "architecture_sha256": checksum_sha256(arch_py),
        "config_sha256": checksum_sha256(gopro_yml),
        "verification_script_sha256": checksum_sha256(Path(__file__).resolve()),
        "checkpoint_sha256": checksum_sha256(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_tensor_dtypes": sorted({str(v.dtype) for v in state.values() if isinstance(v, torch.Tensor)})
        if isinstance(state, Mapping) else [],
        "runtime_policy": {"map_location": "cpu", "eval_mode": not model.training,
                           "inference_mode": True, "input_kind": "zeros_float32"},
        "config": model_args, "parameter_count": sum(p.numel() for p in model.parameters()),
        "model_state_dict_entry_count": len(model.state_dict()),
        "checkpoint_state_dict_entry_count": len(state) if isinstance(state, Mapping) else None,
        "module_counts": {
            "FSAS": sum(isinstance(module, source.FSAS) for module in model.modules()),
            "DFFN": sum(isinstance(module, source.DFFN) for module in model.modules()),
        },
        "fuse_fft_shapes": {
            name: list(getattr(model, name).att_channel.ffn.fft.shape)
            for name in ("fuse1", "fuse2")
        },
        "checks": {},
    }

    def check(name: str, passed: bool, **details: Any) -> bool:
        status = "PASS" if passed else "FAIL"
        evidence["checks"][name] = {"status": status, **details}
        print(f"[{status}] {name}")
        return passed

    check("parameter_count_16560474", evidence["parameter_count"] == 16_560_474,
          actual=evidence["parameter_count"], expected=16_560_474)
    check("checkpoint_raw_tensor_state_dict", isinstance(state, Mapping) and all(isinstance(v, torch.Tensor) for v in state.values()),
          actual_type=type(state).__name__)

    # Strict loading: the actual original checkpoint must match all model names and tensor shapes.
    try:
        compatibility = model.load_state_dict(state, strict=True)
        ok = not compatibility.missing_keys and not compatibility.unexpected_keys
        check("strict_checkpoint_load", ok,
              missing_keys=list(compatibility.missing_keys),
              unexpected_keys=list(compatibility.unexpected_keys))
    except Exception as exc:
        check("strict_checkpoint_load", False, error=repr(exc)[:1200])

    # The official test.py uses fftformer() without reading the YAML config.
    try:
        default_model = source.fftformer().cpu().eval()
        compatibility = default_model.load_state_dict(state, strict=True)
        check("default_constructor_strict_checkpoint_load",
              not compatibility.missing_keys and not compatibility.unexpected_keys,
              missing_keys=list(compatibility.missing_keys),
              unexpected_keys=list(compatibility.unexpected_keys),
              parameter_count=sum(p.numel() for p in default_model.parameters()),
              model_state_dict_entry_count=len(default_model.state_dict()),
              module_counts={
                  "FSAS": sum(isinstance(module, source.FSAS) for module in default_model.modules()),
                  "DFFN": sum(isinstance(module, source.DFFN) for module in default_model.modules()),
              })
        del default_model
    except Exception as exc:
        check("default_constructor_strict_checkpoint_load", False, error=repr(exc)[:1200])

    # Positive smoke: valid shapes, output finite, no gradient calculation.
    for height, width in ((32, 32), (64, 96)):
        try:
            with torch.inference_mode():
                inp = torch.zeros(1, 3, height, width, dtype=torch.float32)
                output = model(inp)
            correct_shape = tuple(output.shape) == tuple(inp.shape)
            finite = bool(torch.isfinite(output).all().item())
            check(f"forward_{height}x{width}", correct_shape and finite,
                  input_shape=list(inp.shape), output_shape=list(output.shape), output_finite=finite)
        except Exception as exc:
            check(f"forward_{height}x{width}", False, error=repr(exc)[:1200])

    # Invalid unpadded input must fail in patch rearrangement, not for an unrelated reason.
    try:
        with torch.inference_mode():
            model(torch.zeros(1, 3, 40, 40, dtype=torch.float32))
        check("forward_40x40_rejected", False,
              reason="Unexpected success: unpadded 40x40 input was accepted")
    except einops.EinopsError as exc:
        message = str(exc)
        expected_fragment = "can't divide axis of length 20 in chunks of 8"
        check("forward_40x40_rejected", expected_fragment in message,
              input_shape=[1, 3, 40, 40], error_type=type(exc).__name__,
              expected_error_contains=expected_fragment, message_excerpt=message[:1200])
    except Exception as exc:
        check("forward_40x40_rejected", False, error=repr(exc)[:1200])

    # Negative tests: intentionally corrupt only an IN-MEMORY shallow copy of checkpoint mapping.
    # Expected failures are PASS when strict=True rejects the corruption with the right key in error.
    if isinstance(state, Mapping):
        key = "patch_embed.proj.weight"
        for name in ("missing_key", "unexpected_key", "wrong_shape"):
            modified = state.copy()
            expected_fragment = key
            if name == "missing_key":
                modified.pop(key, None)
            elif name == "unexpected_key":
                expected_fragment = "audit.unexpected.weight"
                modified[expected_fragment] = torch.empty(1)
            else:
                modified[key] = torch.empty(47, 3, 3, 3)
            try:
                model.load_state_dict(modified, strict=True)
                check(name + "_rejected", False, reason="Unexpected success: invalid checkpoint was accepted")
            except RuntimeError as exc:
                err = str(exc)
                check(name + "_rejected", expected_fragment in err,
                      expected_error_contains=expected_fragment, message_excerpt=err[:400])

    failures = [name for name, record in evidence["checks"].items() if record["status"] != "PASS"]
    evidence["summary"] = {"total": len(evidence["checks"]), "passed": len(evidence["checks"]) - len(failures),
                           "failed": len(failures), "status": "PASS" if not failures else "FAIL",
                           "limitations": "CPU synthetic smoke + strict state-dict checks only; no image-quality benchmark"}

    if args.output is not None:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote JSON evidence: {evidence_path(destination, repo)}")
    print(f"Summary: {evidence['summary']['passed']}/{evidence['summary']['total']} passed")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
