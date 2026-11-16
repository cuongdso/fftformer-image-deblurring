#!/usr/bin/env python3
"""Run WS1 tests including GoPro checkpoint integration and optionally save JSON evidence.

Usage from repository root:
    python -B scripts/ws1_03_verify.py --repo . --output docs/research/ws1-03-runtime-evidence.json
Requires PyTorch, einops, initialized upstream source and the audited GoPro checkpoint.
Does not download artifacts or write checkpoint/source files.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any
import unittest


class RecordingResult(unittest.TextTestResult):
    """Capture the actual unittest outcomes for the evidence artifact."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.records: dict[str, dict[str, str]] = {}

    def addSuccess(self, test: unittest.TestCase) -> None:
        """Record a successful test."""
        super().addSuccess(test)
        self.records[test.id()] = {"status": "PASS"}

    def addSkip(self, test: unittest.TestCase, reason: str) -> None:
        """Record a skipped test and its reason."""
        super().addSkip(test, reason)
        self.records[test.id()] = {"status": "SKIP", "reason": reason}

    def addFailure(self, test: unittest.TestCase, err: Any) -> None:
        """Record an assertion failure."""
        super().addFailure(test, err)
        self.records[test.id()] = {"status": "FAIL", "message": str(err[1])}

    def addError(self, test: unittest.TestCase, err: Any) -> None:
        """Record a test/setup error."""
        super().addError(test, err)
        self.records[test.id()] = {"status": "ERROR", "message": str(err[1])}


def main() -> int:
    """Run opt-in CPU tests and save metadata/hashes without personal directory paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    checkpoint = (args.checkpoint.expanduser().resolve() if args.checkpoint else
                  repo / "third_party/FFTformer/pretrain_model/fftformer_GoPro.pth")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(repo / "src"))
    os.environ["FFTFORMER_RUN_MODEL_TESTS"] = "1"
    os.environ["FFTFORMER_GOPRO_CHECKPOINT"] = str(checkpoint)
    import torch
    torch.set_num_threads(2)
    suite = unittest.defaultTestLoader.discover(str(repo / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult).run(suite)

    def reference(path: Path) -> str:
        try:
            return path.resolve().relative_to(repo).as_posix()
        except ValueError:
            return f"<external>/{path.name}"

    def clean_error(message: str) -> str:
        return message.replace(str(repo), ".").replace(repo.as_posix(), ".").replace(str(Path.home()), "<home>")

    import einops
    from fftformer_lab.model import ModelSpec, load_model

    records = {name: {key: clean_error(value) for key, value in record.items()}
               for name, record in result.records.items()}
    evidence: dict[str, Any] = {
        "audit": "M1-WS1-03", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_source": "group-measured", "test_kind": "cpu-unit-and-checkpoint-integration",
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "einops": einops.__version__, "os": platform.system(),
                        "device": "cpu", "torch_num_threads": torch.get_num_threads()},
        "checkpoint": reference(checkpoint), "checks": records,
        "summary": {"total": result.testsRun, "passed": sum(r["status"] == "PASS" for r in records.values()),
                    "failed": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
                    "status": "PASS" if result.wasSuccessful() and not result.skipped else "FAIL"},
        "limitations": "CPU smoke only; fixtures use a mock Conv2d, real integration uses only the audited GoPro checkpoint; no GPU or image-quality benchmark",
    }
    if result.wasSuccessful() and not result.skipped:
        adapter = load_model(ModelSpec(
            architecture_file=repo / "third_party/FFTformer/basicsr/models/archs/fftformer_arch.py",
            checkpoint=checkpoint,
        ))
        evidence["model_metadata"] = adapter.metadata.to_dict()
        evidence["model_metadata"]["checkpoint_path"] = reference(checkpoint)
    files = [
        "src/fftformer_lab/model/spec.py", "src/fftformer_lab/model/adapter.py",
        "src/fftformer_lab/model/checkpoint.py", "src/fftformer_lab/model/__init__.py",
        "tests/unit/test_model_spec.py", "tests/model/test_model_adapter.py",
        "tests/model/test_checkpoint_loader.py", "scripts/ws1_03_verify.py",
    ]
    evidence["file_sha256"] = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in files}
    if args.output:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote evidence: {reference(destination)}")
    print(f"Summary: {evidence['summary']['passed']}/{result.testsRun} passed")
    return 0 if evidence["summary"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
