"""Command line entry point for validating and evaluating saved prediction images."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from typing import Any

from fftformer_lab.data import DataError

from .config import load_evaluation_config
from .engine import evaluate_batch
from .errors import EvaluationError
from .metadata import build_experiment_metadata, write_experiment_metadata


@dataclass(frozen=True)
class ExistingPrediction:
    """WS4 adapter for a saved prediction, deliberately not an InferenceResult."""

    restored_image: Any
    sample_id: str
    model_metadata: dict[str, Any] | None = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fftformer-evaluate", description="Validate or evaluate saved RGB prediction/GT image pairs.")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-config", help="validate C-06 JSON without image/runtime dependencies")
    validate.add_argument("config", help="path to evaluation JSON config")
    run = commands.add_parser("run", help="evaluate existing predictions against ground truth and write metadata JSON")
    run.add_argument("config", help="path to evaluation JSON config")
    run.add_argument("--overwrite", action="store_true", help="allow replacement of an existing output JSON")
    return parser


def _run(config_path: str, *, overwrite: bool) -> str:
    config = load_evaluation_config(config_path)
    if config.output_path.exists() and not overwrite:
        raise EvaluationError(f"output exists: {config.output_path}; use --overwrite to replace it")
    try:
        from fftformer_lab.data import ImageDataset, build_paired_manifest
    except ImportError as exc:
        raise EvaluationError("image evaluation requires WS2 runtime dependencies (torch, NumPy, Pillow)") from exc
    records = build_paired_manifest(config.prediction_root, config.target_root)
    samples = list(ImageDataset(records))
    predictions = [ExistingPrediction(sample.image, sample.sample_id) for sample in samples]
    summary = evaluate_batch(predictions, samples, config.protocol)
    write_experiment_metadata(config.output_path, build_experiment_metadata(config, summary), overwrite=overwrite)
    return str(config.output_path)


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation CLI and return a process-style exit code."""
    parser = _parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "validate-config":
            config = load_evaluation_config(args.config)
            print(f"valid: {config.experiment_id} ({config.protocol.protocol_id})")
            return 0
        print(f"wrote: {_run(args.config, overwrite=args.overwrite)}")
        return 0
    except (EvaluationError, DataError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    sys.exit(main())
