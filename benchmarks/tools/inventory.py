from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


MODEL_EXTENSIONS = {".safetensors", ".onnx", ".pt", ".pth", ".bin"}
DATA_EXTENSIONS = {".json", ".jsonl", ".csv", ".parquet"}
NOTEBOOK_EXTENSIONS = {".ipynb", ".py"}
EVIDENCE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".txt", ".md"}
SKIP_PARTS = {".git", ".venv", ".venv_passenger", "__pycache__", "node_modules"}
SPECIAL_JSON = {
    "trainer_state.json",
    "adapter_config.json",
    "config.json",
    "all_results.json",
    "eval_results.json",
    "metrics_full.json",
    "metrics_report.json",
}


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def category(path: Path) -> str | None:
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix in MODEL_EXTENSIONS:
        if name in {"optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin"}:
            return "training_resume_state"
        return "model_artifact"
    if suffix in NOTEBOOK_EXTENSIONS:
        return "training_or_test_code"
    if name in SPECIAL_JSON:
        return "configuration_or_metric"
    if suffix in DATA_EXTENSIONS:
        return "dataset_or_tokenizer"
    if suffix in EVIDENCE_EXTENSIONS:
        return "documentation_or_visual_evidence"
    return None


def iter_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
            continue
        if category(path):
            yield path


def build_inventory(root: Path) -> dict:
    rows = []
    duplicate_index: dict[tuple[int, str], list[str]] = defaultdict(list)
    for index, path in enumerate(iter_files(root), 1):
        rel = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = sha256(path)
        row = {
            "relative_path": rel,
            "category": category(path),
            "size_bytes": size,
            "sha256": digest,
            "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        }
        rows.append(row)
        duplicate_index[(size, digest)].append(rel)
        if index % 25 == 0:
            print(f"[inventory] hashed {index} relevant files")

    rows.sort(key=lambda item: item["relative_path"].casefold())
    duplicates = [
        {"size_bytes": size, "sha256": digest, "paths": sorted(paths, key=str.casefold)}
        for (size, digest), paths in duplicate_index.items()
        if len(paths) > 1
    ]
    duplicates.sort(key=lambda group: (-group["size_bytes"], group["sha256"]))
    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_root_label": root.name,
        "source_root_disclosed": False,
        "file_count": len(rows),
        "total_size_bytes": sum(row["size_bytes"] for row in rows),
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "artifacts": rows,
        "duplicate_groups": duplicates,
    }


def write_outputs(inventory: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "artifact_inventory.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "artifact_inventory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "category", "size_bytes", "sha256", "modified_utc"],
        )
        writer.writeheader()
        writer.writerows(inventory["artifacts"])

    lines = [
        "# Local Artifact Inventory",
        "",
        f"- Relevant files: {inventory['file_count']:,}",
        f"- Total relevant size: {inventory['total_size_bytes'] / 1024**3:.2f} GiB",
        f"- Duplicate groups: {len(inventory['duplicate_groups']):,}",
        "- Absolute source paths are intentionally omitted from publishable outputs.",
        "",
        "## Categories",
        "",
    ]
    lines.extend(f"- `{key}`: {value:,}" for key, value in sorted(inventory["category_counts"].items()))
    lines.extend(["", "## Largest duplicate groups", ""])
    for group in inventory["duplicate_groups"][:20]:
        lines.append(
            f"- `{group['sha256'][:12]}` — {group['size_bytes'] / 1024**2:.2f} MiB — "
            f"{len(group['paths'])} copies"
        )
        lines.extend(f"  - `{path}`" for path in group["paths"])
    (output_dir / "ARTIFACT_INVENTORY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hash the read-only PathFindership source archive.")
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inventory = build_inventory(args.archive_root.resolve())
    write_outputs(inventory, args.output_dir.resolve())
    print(f"[inventory] complete: {inventory['file_count']} files")


if __name__ == "__main__":
    main()
