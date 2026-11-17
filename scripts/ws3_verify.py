#!/usr/bin/env python3
"""Run WS3 inference pipeline verification tests and save JSON evidence.

Usage from repository root:
    python -B scripts/ws3_verify.py --repo . --output docs/research/ws3-02-runtime-evidence.json

Follows the verification pattern and evidence schema established by WS1 (ADR-0002).
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
    """Capture individual unittest outcomes for the evidence artifact."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.records: dict[str, dict[str, str]] = {}
        self.subtests: dict[str, dict[str, str]] = {}

    def _record(self, test: unittest.TestCase, status: str, **details: str) -> None:
        priority = {"PASS": 0, "SKIP": 1, "FAIL": 2, "ERROR": 3}
        previous = self.records.get(test.id())
        if previous is None or priority[status] > priority[previous["status"]]:
            self.records[test.id()] = {"status": status, **details}

    def _record_subtest(
        self, parent: unittest.TestCase, subtest: unittest.TestCase, status: str, **details: str
    ) -> None:
        original_id = subtest.id()
        key = original_id
        occurrence = 1
        while key in self.subtests:
            occurrence += 1
            key = f"{original_id} [occurrence {occurrence}]"
        self.subtests[key] = {"id": original_id, "parent": parent.id(), "status": status, **details}

    def addSuccess(self, test: unittest.TestCase) -> None:
        super().addSuccess(test)
        self._record(test, "PASS")

    def addSkip(self, test: unittest.TestCase, reason: str) -> None:
        super().addSkip(test, reason)
        parent = getattr(test, "test_case", test)
        self._record(parent, "SKIP", reason=reason)
        if parent is not test:
            self._record_subtest(parent, test, "SKIP", reason=reason)

    def addSubTest(self, test: unittest.TestCase, subtest: unittest.TestCase, err: Any) -> None:
        super().addSubTest(test, subtest, err)
        if err is not None:
            status = "FAIL" if issubclass(err[0], test.failureException) else "ERROR"
            self._record(test, status, message=str(err[1]))
            self._record_subtest(test, subtest, status, message=str(err[1]))

    def addFailure(self, test: unittest.TestCase, err: Any) -> None:
        super().addFailure(test, err)
        self._record(test, "FAIL", message=str(err[1]))

    def addError(self, test: unittest.TestCase, err: Any) -> None:
        super().addError(test, err)
        self._record(test, "ERROR", message=str(err[1]))


def summarize_result(result: RecordingResult) -> dict[str, Any]:
    """Require a nonempty complete test run before reporting an all-pass audit."""
    passed = sum(record["status"] == "PASS" for record in result.records.values())
    complete = (
        result.testsRun > 0
        and len(result.records) == result.testsRun
        and passed == result.testsRun
        and result.wasSuccessful()
        and not result.skipped
    )
    return {
        "total": result.testsRun,
        "passed": passed,
        "failed": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "status": "PASS" if complete else "FAIL",
    }


def main() -> int:
    """Run WS3 unit and pipeline tests, collecting metadata and portable hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(repo / "src"))

    try:
        import torch
        torch.set_num_threads(2)
        torch_version = torch.__version__
        device_str = "cpu"
    except ImportError:
        torch_version = "not-installed"
        device_str = "none"

    suite = unittest.defaultTestLoader.discover(str(repo / "tests"), pattern="test_inference_*.py")
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult).run(suite)

    def reference(path: Path) -> str:
        try:
            return path.resolve().relative_to(repo).as_posix()
        except ValueError:
            return f"<external>/{path.name}"

    def clean_error(message: str) -> str:
        return message.replace(str(repo), ".").replace(repo.as_posix(), ".").replace(str(Path.home()), "<home>")

    records = {
        name: {key: clean_error(value) for key, value in record.items()}
        for name, record in result.records.items()
    }
    subtests = {
        name: {key: clean_error(value) for key, value in record.items()}
        for name, record in result.subtests.items()
    }
    summary = summarize_result(result)

    files = [
        "src/fftformer_lab/inference/__init__.py",
        "src/fftformer_lab/inference/types.py",
        "src/fftformer_lab/inference/padding.py",
        "src/fftformer_lab/inference/mock.py",
        "src/fftformer_lab/inference/pipeline.py",
        "src/fftformer_lab/inference/README.md",
        "tests/unit/test_inference_pure.py",
        "tests/unit/test_inference_pipeline.py",
        "tests/unit/test_inference_contracts.py",
        "scripts/ws3_verify.py",
    ]
    file_hashes = {}
    for name in files:
        target = repo / name
        if target.exists():
            file_hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()

    evidence: dict[str, Any] = {
        "audit": "M1-WS3-02",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_source": "group-measured",
        "test_kind": "cpu-unit-and-mock-pipeline",
        "contract_compliance": ["C-02", "C-04"],
        "environment": {
            "python": platform.python_version(),
            "torch": torch_version,
            "os": platform.system(),
            "device": device_str,
            "torch_num_threads": 2 if torch_version != "not-installed" else 0,
        },
        "checks": records,
        "subtests": subtests,
        "summary": summary,
        "file_sha256": file_hashes,
        "limitations": (
            "CPU mock pipeline and reflection padding tests only; "
            "no GPU forward, no checkpoint weights, no file-based image I/O CLI (reserved for M1-WS3-03)"
        ),
    }

    if args.output:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print(f"Wrote evidence: {reference(destination)}")

    print(f"Summary: {evidence['summary']['passed']}/{result.testsRun} passed")
    return 0 if evidence["summary"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
