from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def find_run_root(root: Path) -> Path:
    if (root / "RUN_COMPLETE.txt").exists():
        return root
    candidates = [path.parent for path in root.rglob("RUN_COMPLETE.txt")]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one result root, found {len(candidates)}")
    return candidates[0]


def validate(run_root: Path) -> dict:
    required = ["metrics", "predictions", "figures", "logs", "status", "environment.json", "artifact_manifest.json", "RUN_COMPLETE.txt"]
    missing = [name for name in required if not (run_root / name).exists()]
    manifest = json.loads((run_root / "artifact_manifest.json").read_text(encoding="utf-8")) if not missing else {"files": []}
    hash_errors = []
    for record in manifest.get("files", []):
        path = run_root / Path(record["relative_path"])
        if not path.exists():
            hash_errors.append({"path": record["relative_path"], "error": "missing"})
        elif path.stat().st_size != record["size_bytes"]:
            hash_errors.append({"path": record["relative_path"], "error": "size_mismatch"})
        elif sha256(path) != record["sha256"]:
            hash_errors.append({"path": record["relative_path"], "error": "sha256_mismatch"})
    statuses = []
    if (run_root / "status").exists():
        for path in (run_root / "status").glob("*.json"):
            statuses.append(json.loads(path.read_text(encoding="utf-8")))
    return {
        "valid": not missing and not hash_errors,
        "run_root": str(run_root),
        "missing": missing,
        "hash_errors": hash_errors,
        "complete_experiments": [item.get("experiment_id") for item in statuses if item.get("status") == "complete"],
        "failed_experiments": [item for item in statuses if item.get("status") == "failed"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the ZIP/folder returned from Lightning AI.")
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    if args.bundle.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory(prefix="pathfinder_result_validation_") as directory:
            with zipfile.ZipFile(args.bundle) as archive:
                archive.extractall(directory)
            result = validate(find_run_root(Path(directory)))
    else:
        result = validate(find_run_root(args.bundle.resolve()))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
