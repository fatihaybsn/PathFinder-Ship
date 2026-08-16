from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import yaml


RESUME_FILES = {"optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin", "trainer_state.json"}
Q_V_ADAPTER_CONFIG = {
    "base_model_name_or_path": "google/flan-t5-large",
    "bias": "none",
    "fan_in_fan_out": False,
    "inference_mode": True,
    "init_lora_weights": True,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "peft_type": "LORA",
    "r": 16,
    "target_modules": ["q", "v"],
    "task_type": "SEQ_2_SEQ_LM",
    "use_dora": False,
    "use_rslora": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_directory(source: Path, destination: Path, predicate=None) -> None:
    for path in source.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if predicate and not predicate(path):
            continue
        copy_file(path, destination / path.relative_to(source))


def stage_model(experiment: dict, archive_root: Path, models_root: Path) -> None:
    source = archive_root / Path(experiment["source"])
    destination = models_root / experiment["id"]
    if not source.exists():
        raise FileNotFoundError(source)

    if experiment["family"] == "minilm_intent":
        for onnx_path in source.glob("*.onnx"):
            copy_file(onnx_path, destination / onnx_path.name)
        tokenizer_dir = source / "best"
        copy_directory(tokenizer_dir, destination, lambda path: path.suffix.lower() != ".safetensors")
        return

    if experiment["format"] == "transformers_full":
        copy_directory(source, destination, lambda path: path.name not in RESUME_FILES)
        return

    if experiment["format"] == "peft_lora":
        for name in ("adapter_model.safetensors", "adapter_config.json", "README.md"):
            path = source / name
            if path.exists():
                copy_file(path, destination / name)
        if experiment.get("reconstructed_adapter_config") and not (destination / "adapter_config.json").exists():
            (destination / "adapter_config.json").write_text(
                json.dumps(Q_V_ADAPTER_CONFIG, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        if onnx_rel := experiment.get("onnx_source"):
            onnx_source = archive_root / Path(onnx_rel)
            copy_directory(onnx_source, destination / "onnx")
        return

    if experiment["family"] == "vision_integration":
        for path in source.glob("yolo11*.pt"):
            copy_file(path, destination / path.name)
        onnx = source / "yolo11l.onnx"
        if onnx.exists():
            copy_file(onnx, destination / onnx.name)
        return

    raise ValueError(f"No staging rule for {experiment['id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the read-only upload bundle for Lightning AI.")
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    archive_root = args.archive_root.resolve()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    marker = output_dir / ".pathfinder_lightning_bundle"
    if output_dir.exists():
        if not marker.exists():
            raise RuntimeError(f"Refusing to replace unmarked directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    marker.write_text("PathFinder Lightning bundle v1\n", encoding="utf-8")

    manifest_path = repo_root / "benchmarks" / "config" / "experiments.yaml"
    experiments = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))["experiments"]
    for experiment in experiments:
        print(f"[stage] {experiment['id']}")
        stage_model(experiment, archive_root, output_dir / "models")

    copy_directory(repo_root / "benchmarks", output_dir / "benchmarks")
    copy_file(repo_root / "BENCHMARK_PLAN.md", output_dir / "BENCHMARK_PLAN.md")
    copy_file(repo_root / "docs" / "model-development" / "DATASET_CARD.md", output_dir / "DATASET_CARD.md")
    for evidence_name in (
        "historical_extraction.json",
        "training_data_audit.json",
        "TRAINING_DATA_AUDIT.md",
        "artifact_inventory.json",
        "ARTIFACT_INVENTORY.md",
    ):
        copy_file(
            repo_root / "docs" / "model-development" / "evidence" / evidence_name,
            output_dir / "historical_evidence" / evidence_name,
        )
    copy_file(
        repo_root / "benchmarks" / "lightning" / "PathFinder_Lightning_Benchmark.ipynb",
        output_dir / "PathFinder_Lightning_Benchmark.ipynb",
    )
    copy_file(
        repo_root / "benchmarks" / "lightning" / "LIGHTNING_INSTRUCTIONS_TR.md",
        output_dir / "LIGHTNING_INSTRUCTIONS_TR.md",
    )

    artifacts = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path == marker:
            continue
        artifacts.append(
            {
                "relative_path": path.relative_to(output_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    bundle_manifest = {
        "schema_version": 1,
        "file_count": len(artifacts),
        "total_size_bytes": sum(item["size_bytes"] for item in artifacts),
        "files": artifacts,
    }
    (output_dir / "UPLOAD_MANIFEST.json").write_text(
        json.dumps(bundle_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[stage] complete: {bundle_manifest['file_count']} files, "
        f"{bundle_manifest['total_size_bytes'] / 1024**3:.2f} GiB"
    )


if __name__ == "__main__":
    main()
