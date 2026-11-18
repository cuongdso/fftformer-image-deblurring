"""Verify a local WS2 handoff; downloads are opt-in and limited to four examples."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import urllib.request


SOURCE_REVISION = "2d5a698560e658718f0520e48dfb15bd52c80118"
SOURCE_REPOSITORY = "https://github.com/SeungjunNah/DeepDeblur_release"
DATASET_PAGE = "https://seungjunnah.github.io/Datasets/gopro.html"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
EXAMPLES = (
    ("Flower", "input", "Flower_blur1.png", 1403470, "c58a81f9d4d6cbb393f4255dd69b00135a54b62a"),
    ("Flower", "target", "Flower_sharp1.png", 1572256, "331b4dcbf51f964046b1e54092f2bc61637da26f"),
    ("Istanbul", "input", "Istanbul_blur1.png", 932252, "9f268b61fa19691bdd8877c561b402097fe165b6"),
    ("Istanbul", "target", "Istanbul_sharp1.png", 1108064, "570e4439bf980c043a23ff9d1e9cf9b62fd55326"),
)


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def _download_examples(root: Path) -> None:
    for sample_id, role, filename, size, expected_blob in EXAMPLES:
        destination = root / role / f"{sample_id}.png"
        if destination.exists():
            data = destination.read_bytes()
        else:
            url = f"https://raw.githubusercontent.com/SeungjunNah/DeepDeblur_release/{SOURCE_REVISION}/images/{filename}"
            request = urllib.request.Request(url, headers={"User-Agent": "fftformer-lab-ws2-verifier"})
            with urllib.request.urlopen(request, timeout=45) as response:
                data = response.read(size + 1)
        if len(data) != size or _git_blob(data) != expected_blob:
            raise ValueError(f"example {destination}: source size/Git blob mismatch; existing files are not overwritten")
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(data)
        print(f"Example verified: {role}/{sample_id}.png ({size} bytes)", flush=True)
    attribution = root / "ATTRIBUTION.md"
    text = (
        "# GoPro author-published example pairs\n\n"
        "Creators: Seungjun Nah, Tae Hyun Kim, Kyoung Mu Lee.\n"
        "Work: Deep Multi-Scale Convolutional Neural Network for Dynamic Scene Deblurring, CVPR 2017.\n"
        f"Dataset/source terms: {DATASET_PAGE}\nLicense: CC BY 4.0, {LICENSE_URL}\n"
        f"Source: {SOURCE_REPOSITORY}/tree/{SOURCE_REVISION}/images\n\n"
        "These are the Flower and Istanbul blurry/sharp illustrations linked from the author's README.\n"
        "Changes: filenames mapped to input/target pairing names; image bytes are unchanged.\n"
        "Split and CRF variant: not asserted; this is not the full GoPro test split.\n"
        "Used only for local loader verification. Keep images outside Git.\n"
    )
    if not attribution.exists():
        with attribution.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)


def _check_examples(root: Path) -> dict:
    import torch
    from fftformer_lab.data import ImageDataset, build_paired_manifest, build_single_manifest

    torch.set_num_threads(2)
    expected = {}
    provenance = []
    for sample_id, role, filename, size, expected_blob in EXAMPLES:
        path = root / role / f"{sample_id}.png"
        data = path.read_bytes()
        if len(data) != size or _git_blob(data) != expected_blob:
            raise ValueError(f"example {path}: differs from the pinned author source")
        expected[path] = hashlib.sha256(data).hexdigest()
        provenance.append({
            "sample_id": sample_id + ".png", "role": role,
            "source_url": f"https://raw.githubusercontent.com/SeungjunNah/DeepDeblur_release/{SOURCE_REVISION}/images/{filename}",
            "source_git_blob": expected_blob, "bytes": size, "sha256": expected[path],
        })
    paired = ImageDataset(build_paired_manifest(root / "input", root / "target"))
    single = ImageDataset(build_single_manifest(root / "input"))
    paired_ids, single_ids, samples = [], [], []
    for sample in paired:
        if not sample.has_ground_truth or sample.target is None or sample.image.shape != sample.target.shape:
            raise ValueError(f"sample {sample.sample_id}: invalid paired output")
        if torch.equal(sample.image, sample.target):
            raise ValueError(f"sample {sample.sample_id}: input unexpectedly equals target")
        paired_ids.append(sample.sample_id)
        samples.append({"sample_id": sample.sample_id, "shape": list(sample.image.shape),
                        "dtype": str(sample.image.dtype), "has_ground_truth": sample.has_ground_truth,
                        "preprocessing_metadata": sample.preprocessing_metadata.to_dict()})
    for sample in single:
        if sample.has_ground_truth or sample.target is not None or sample.target_path is not None:
            raise ValueError(f"sample {sample.sample_id}: non-GT output invented ground truth")
        single_ids.append(sample.sample_id)
    if paired_ids != ["Flower.png", "Istanbul.png"] or paired_ids != single_ids:
        raise ValueError("author example ordering/count differs from expected pairs")
    for path, digest in expected.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"source bytes were changed: {path}")
    return {"status": "PASS", "source_repository": SOURCE_REPOSITORY, "source_revision": SOURCE_REVISION,
            "dataset_page": DATASET_PAGE, "license": "CC-BY-4.0", "license_url": LICENSE_URL,
            "scope": "Two author README example pairs; split/CRF variant not asserted; no benchmark",
            "files": provenance, "paired_samples": samples, "non_gt_sample_ids": single_ids,
            "source_bytes_unchanged": True, "quality_metrics_computed": False}


def _run(repo: Path, arguments: list[str], env: dict[str, str]) -> dict:
    print("Running:", subprocess.list2cmdline(arguments), flush=True)
    result = subprocess.run(arguments, cwd=repo, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    output = result.stdout + result.stderr
    print(output, end="", flush=True)
    check = {"arguments": arguments[1:], "interpreter": Path(arguments[0]).name,
             "exit_code": result.returncode, "status": "PASS" if result.returncode == 0 else "FAIL",
             "output": output}
    run = re.search(r"Ran (\d+) tests? in", output)
    skipped = re.search(r"skipped=(\d+)", output)
    if run:
        total, skips = int(run.group(1)), int(skipped.group(1)) if skipped else 0
        check.update(tests_run=total, skipped=skips)
        # On failure preserve raw unittest output; subtest failure events cannot
        # safely be subtracted from the number of parent tests.
        if result.returncode == 0 and 0 <= skips <= total:
            check["passed"] = total - skips
        if total == 0:
            check["status"] = "FAIL"
    return check


def _check_model_boundary(repo: Path, checkpoint: Path) -> dict:
    import torch
    from PIL import Image
    from fftformer_lab.data import ImageDataset, build_single_manifest
    from fftformer_lab.inference import InferencePipeline
    from fftformer_lab.model import ModelSpec, load_model

    torch.set_num_threads(2)
    adapter = load_model(ModelSpec(
        architecture_file=repo / "third_party/FFTformer/basicsr/models/archs/fftformer_arch.py",
        checkpoint=checkpoint, device="cpu",
    ))
    samples = []
    with tempfile.TemporaryDirectory(prefix="ws2-model-boundary-") as temporary:
        root = Path(temporary)
        for index, size in enumerate(((64, 64), (96, 64), (70, 50)), start=1):
            Image.new("RGB", size, (32 * index, 128, 192)).save(root / f"sample_{index:03d}.png")
        dataset = ImageDataset(build_single_manifest(root))
        before = {r.source_path: hashlib.sha256(r.source_path.read_bytes()).hexdigest() for r in dataset.records}
        results = InferencePipeline(adapter).run_batch(dataset)
        if len(results) != len(dataset):
            raise ValueError("model smoke: result count mismatch")
        for sample, result in zip(dataset, results):
            if result.is_mock or result.sample_id != sample.sample_id or result.restored_image.shape != sample.image.shape:
                raise ValueError(f"model smoke {sample.sample_id}: identity/shape/mock mismatch")
            image = result.restored_image
            if image.dtype != torch.float32 or not bool(torch.isfinite(image).all()) or bool(((image < 0) | (image > 1)).any()):
                raise ValueError(f"model smoke {sample.sample_id}: dtype/finiteness/range mismatch")
            samples.append({"sample_id": sample.sample_id, "input_output_shape": list(image.shape),
                            "padding": result.padding.to_dict(), "is_mock": result.is_mock})
        for path, digest in before.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"model smoke: source bytes changed: {path.name}")
    return {"status": "PASS", "device": "cpu", "is_mock": False,
            "dataset_identity": "three temporary synthetic constant-RGB images without GT",
            "flow": "WS2 file reader/ImageDataset -> WS1 strict GoPro checkpoint -> WS3 pipeline",
            "samples": samples, "checkpoint_sha256": adapter.metadata.checkpoint_sha256,
            "source_bytes_unchanged": True, "quality_metrics_computed": False}


def main() -> int:
    """Run WS2, repository and standard-library suites and write actual evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("docs/research/ws2-runtime-evidence.json"))
    parser.add_argument("--examples-dir", type=Path, help="Validate the two pinned GoPro author example pairs")
    parser.add_argument("--download-examples", action="store_true", help="Fetch only the four small author PNGs; requires --examples-dir")
    parser.add_argument("--checkpoint", type=Path, help="Enable existing WS1 model/checkpoint tests with this local artifact")
    args = parser.parse_args()
    if args.download_examples and args.examples_dir is None:
        parser.error("--download-examples requires --examples-dir")
    repo = args.repo.resolve()
    if not (repo / "tests/unit/test_data_image.py").is_file():
        parser.error(f"not a WS2 checkout: {repo}")
    sys.path.insert(0, str(repo / "src"))
    output = args.output if args.output.is_absolute() else repo / args.output
    if output.suffix.lower() != ".json":
        parser.error("--output must be a .json evidence file")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    env.pop("FFTFORMER_RUN_MODEL_TESTS", None)
    env.pop("FFTFORMER_GOPRO_CHECKPOINT", None)
    full_env = env.copy()
    if args.checkpoint is not None:
        checkpoint = args.checkpoint if args.checkpoint.is_absolute() else repo / args.checkpoint
        if not checkpoint.is_file():
            parser.error(f"checkpoint does not exist: {checkpoint}")
        full_env.update(FFTFORMER_RUN_MODEL_TESTS="1", FFTFORMER_GOPRO_CHECKPOINT=str(checkpoint.resolve()))
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
    verified_at = datetime.now(timezone(timedelta(hours=7)))
    evidence = {"verification_date": verified_at.date().isoformat(), "verified_at": verified_at.isoformat(),
                "timezone": "Asia/Saigon", "owner": "phanthek2",
                "tasks": ["M1-WS2-01", "M1-WS2-02", "M1-WS2-03"], "base_commit": base,
                "status": "LOCAL_VERIFICATION_RUNNING", "implementation_commit": None, "pr_url": None,
                "environment": {"python": platform.python_version(), "platform": platform.system(),
                                "cpu_threads": 2, "packages": {}}, "checks": []}
    evidence["model_tests"] = {"enabled": args.checkpoint is not None}
    if args.checkpoint is not None:
        evidence["model_tests"].update(checkpoint_name=checkpoint.name,
                                       checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    for package in ("torch", "numpy", "Pillow", "einops"):
        try:
            evidence["environment"]["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            evidence["environment"]["packages"][package] = None
    commands = [
        ([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests/unit", "-p", "test_data_*.py"], env),
        ([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"], full_env),
        ([sys.executable, "-B", "-S", "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"], env),
        ([sys.executable, "-m", "compileall", "-q", "scripts", "src", "tests"], env),
        ([sys.executable, "-m", "pip", "check"], env),
        (["git", "diff", "--check"], env),
    ]
    for command, command_env in commands:
        evidence["checks"].append(_run(repo, command, command_env))
    try:
        if args.examples_dir is not None:
            examples = args.examples_dir if args.examples_dir.is_absolute() else repo / args.examples_dir
            if args.download_examples:
                _download_examples(examples)
            evidence["author_examples"] = _check_examples(examples)
        else:
            evidence["author_examples"] = {"status": "NOT_RUN", "reason": "--examples-dir not provided"}
    except Exception as exc:
        evidence["author_examples"] = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
    try:
        evidence["boundary_smoke"] = (
            _check_model_boundary(repo, checkpoint) if args.checkpoint is not None
            else {"status": "NOT_RUN", "reason": "--checkpoint not provided"}
        )
    except Exception as exc:
        evidence["boundary_smoke"] = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
    files = [repo / "pyproject.toml", repo / "scripts/ws2_verify.py"]
    files += sorted((repo / "src/fftformer_lab/data").glob("*.py"))
    files += sorted((repo / "tests/unit").glob("test_data_*.py"))
    files += sorted((repo / "tests/fixtures/ws2").glob("*.png"))
    evidence["sha256"] = {}
    evidence["sha256_lf"] = {}
    for path in files:
        path = path if path.is_absolute() else repo / path
        data = path.read_bytes()
        name = path.relative_to(repo).as_posix()
        evidence["sha256"][name] = hashlib.sha256(data).hexdigest()
        evidence["sha256_lf"][name] = hashlib.sha256(data.replace(b"\r\n", b"\n") if path.suffix != ".png" else data).hexdigest()
    passed = all(check["status"] == "PASS" for check in evidence["checks"])
    if evidence["checks"][0].get("skipped", 0):
        passed = False
        evidence["image_dependency_error"] = "WS2 runtime tests were skipped; install the project data extra"
    passed = passed and evidence["author_examples"]["status"] != "FAIL"
    passed = passed and evidence["boundary_smoke"]["status"] != "FAIL"
    evidence["status"] = "LOCAL_VERIFIED_AWAITING_REVIEW" if passed else "FAIL"
    evidence["limitations"] = ["No full GoPro/RealBlur benchmark, CUDA, LMDB or training validation",
                               "No external team review, commit, push, PR or merge performed"]
    _write_json(output, evidence)
    print(f"WS2 verification: {evidence['status']}; evidence: {output}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
