"""Command-line interface and orchestration for FFTformer inference (M1-WS3-03)."""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

from .io import SUPPORTED_WRITE_EXTENSIONS, write_image
from .mock import MockModel
from .pipeline import InferencePipeline
from .types import (
    InferenceError,
    InferenceInputError,
    InferenceResult,
)


def create_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser for inference orchestration."""
    parser = argparse.ArgumentParser(
        prog="fftformer-inference",
        description="Run FFTformer image deblurring inference on a single image, directory, or manifest.",
    )

    # Input group (mutually exclusive)
    input_group = parser.add_argument_group("Input options (choose one)")
    input_mutex = input_group.add_mutually_exclusive_group(required=True)
    input_mutex.add_argument(
        "--input", "-i", type=Path, help="Path to a single input image (.png, .jpg, .jpeg)"
    )
    input_mutex.add_argument(
        "--input-dir", type=Path, help="Directory containing input images to deblur"
    )
    input_mutex.add_argument(
        "--manifest", "-m", type=Path, help="Path to a dataset manifest JSON (from fftformer_lab.data)"
    )

    # Output options
    output_group = parser.add_argument_group("Output options")
    output_group.add_argument(
        "--output", "-o", type=Path, help="Path for restored output image (single image mode)"
    )
    output_group.add_argument(
        "--output-dir", type=Path, help="Directory to save restored images (batch or directory mode)"
    )
    output_group.add_argument(
        "--save-metadata", action="store_true", help="Save JSON metadata alongside restored image(s)"
    )

    # Model options
    model_group = parser.add_argument_group("Model options")
    model_group.add_argument(
        "--checkpoint", "-c", type=Path, default=None, help="Path to pretrained .pth checkpoint"
    )
    model_group.add_argument(
        "--architecture-file",
        type=Path,
        default=None,
        help="Path to upstream fftformer_arch.py (required when loading checkpoint)",
    )
    model_group.add_argument(
        "--mock",
        action="store_true",
        help="Use mock model test double (default when no checkpoint is specified)",
    )
    model_group.add_argument(
        "--mock-mode",
        choices=["identity", "invert", "shift"],
        default="identity",
        help="Mock transformation mode (default: identity)",
    )
    model_group.add_argument(
        "--device", default="cpu", help="Target device (cpu or cuda, default: cpu)"
    )

    # Misc options
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress informative console logs")

    return parser


def build_model_for_cli(args: argparse.Namespace, repo: Path) -> Any:
    """Build either a MockModel or strict loaded ModelAdapter based on CLI arguments."""
    if args.checkpoint is not None:
        from fftformer_lab.model import ModelSpec, load_model

        arch_file = (
            args.architecture_file
            if args.architecture_file is not None
            else repo / "third_party/FFTformer/basicsr/models/archs/fftformer_arch.py"
        )
        spec = ModelSpec(
            architecture_file=arch_file,
            checkpoint=args.checkpoint,
            device=args.device,
        )
        return load_model(spec)

    # Default to MockModel for lightweight / verification execution
    return MockModel(mode=args.mock_mode, device=args.device)


def run_cli(args_list: list[str] | None = None) -> int:
    """Execute inference CLI orchestration."""
    parser = create_parser()
    args = parser.parse_args(args_list)

    repo = Path.cwd().resolve()

    # Validate output target
    if args.input is not None:
        if args.output is None and args.output_dir is None:
            parser.error("Single image mode requires either --output <file> or --output-dir <dir>")
        if args.output is not None and args.output_dir is not None:
            parser.error("Specify either --output or --output-dir, not both")
    else:
        if args.output_dir is None:
            parser.error("Batch/directory/manifest mode requires --output-dir <dir>")

    try:
        from fftformer_lab.data import (
            ImageDataset,
            build_single_manifest,
            load_manifest,
            read_image,
        )
    except ImportError as exc:
        sys.stderr.write(f"Error: WS2 data provider is required for CLI execution: {exc}\n")
        return 1

    try:
        model = build_model_for_cli(args, repo)
        pipeline = InferencePipeline(model, device=args.device)
    except Exception as exc:
        sys.stderr.write(f"Model initialization error: {exc}\n")
        return 1

    # ------------------------------------------------------------- Single image mode
    if args.input is not None:
        input_path = args.input.expanduser().resolve()
        if not input_path.exists() or not input_path.is_file():
            sys.stderr.write(f"Error: input file does not exist: {input_path}\n")
            return 1

        if args.output is not None:
            output_path = args.output.expanduser().resolve()
        else:
            out_dir = args.output_dir.expanduser().resolve()
            output_path = out_dir / f"{input_path.stem}_restored{input_path.suffix}"

        try:
            image_tensor = read_image(input_path)
            sample_id = input_path.stem
            result = pipeline.run_single(image_tensor, sample_id=sample_id)
            saved_path = write_image(result.restored_image, output_path)

            if args.save_metadata:
                meta_path = saved_path.parent / f"{saved_path.stem}_metadata.json"
                meta_path.write_text(
                    json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )

            if not args.quiet:
                print(f"[WS3] Deblurred: {input_path.name} -> {saved_path.name}")
                print(f"      Input shape:  {result.input_shape}")
                print(f"      Padded shape: {result.padded_shape}")
                print(f"      Output shape: {result.output_shape}")
                print(f"      Elapsed time: {result.elapsed_time_seconds:.4f}s")
                print(f"      Device:       {result.device} (is_mock={result.is_mock})")

        except Exception as exc:
            sys.stderr.write(f"Inference error on {input_path}: {exc}\n")
            return 1

        return 0

    # --------------------------------------------------- Batch / Directory / Manifest mode
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.manifest is not None:
            manifest_path = args.manifest.expanduser().resolve()
            if not manifest_path.exists():
                sys.stderr.write(f"Error: manifest file does not exist: {manifest_path}\n")
                return 1
            records = load_manifest(manifest_path)
            dataset = ImageDataset(records)
        else:
            in_dir = args.input_dir.expanduser().resolve()
            if not in_dir.exists() or not in_dir.is_dir():
                sys.stderr.write(f"Error: input directory does not exist: {in_dir}\n")
                return 1
            records = build_single_manifest(in_dir)
            if not records:
                sys.stderr.write(f"Error: no supported images (.png, .jpg, .jpeg) found in {in_dir}\n")
                return 1
            dataset = ImageDataset(records)

        start_batch_time = time.perf_counter()
        results: list[InferenceResult] = pipeline.run_batch(dataset)
        total_time = time.perf_counter() - start_batch_time

        for idx, (record, res) in enumerate(zip(records, results)):
            stem = Path(record.source_path).stem
            suffix = Path(record.source_path).suffix or ".png"
            if suffix.lower() not in SUPPORTED_WRITE_EXTENSIONS:
                suffix = ".png"
            dest_file = out_dir / f"{stem}_restored{suffix}"
            write_image(res.restored_image, dest_file)

            if args.save_metadata:
                meta_path = out_dir / f"{stem}_restored_metadata.json"
                meta_path.write_text(
                    json.dumps(res.to_dict(), indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )

        if not args.quiet:
            print(f"[WS3] Processed {len(results)} samples into {out_dir}")
            print(f"      Total batch time: {total_time:.4f}s")
            if results:
                avg_time = total_time / len(results)
                print(f"      Average latency:  {avg_time:.4f}s/sample")

    except Exception as exc:
        sys.stderr.write(f"Batch inference error: {exc}\n")
        return 1

    return 0


def main() -> None:
    raise SystemExit(run_cli(sys.argv[1:]))


if __name__ == "__main__":
    main()
