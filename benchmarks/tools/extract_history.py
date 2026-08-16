from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


KEYWORDS = re.compile(
    r"(?i)(train|valid|eval|loss|accuracy|macro.?f1|rouge|sacrebleu|exact match|\bem\b|"
    r"epoch|runtime|trainable params|early stopping|perplexity|\bppl\b|onnx|lora)"
)
SKIP_PARTS = {".git", ".venv", ".venv_passenger", "__pycache__", "node_modules"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def extract_trainer_states(root: Path) -> list[dict]:
    records = []
    for path in root.rglob("trainer_state.json"):
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            records.append({"source": relative(root, path), "error": str(error)})
            continue
        history = []
        for item in state.get("log_history", []):
            selected = {
                key: value
                for key, value in item.items()
                if key in {"epoch", "step", "loss", "eval_loss", "learning_rate", "train_loss", "train_runtime"}
            }
            if selected:
                history.append(selected)
        records.append(
            {
                "source": relative(root, path),
                "sha256": digest(path),
                "best_metric": state.get("best_metric"),
                "best_model_checkpoint": state.get("best_model_checkpoint"),
                "epoch": state.get("epoch"),
                "global_step": state.get("global_step"),
                "max_steps": state.get("max_steps"),
                "log_history": history,
            }
        )
    return sorted(records, key=lambda item: item["source"].casefold())


def output_text(output: dict) -> str:
    kind = output.get("output_type")
    if kind == "stream":
        value = output.get("text", "")
        return "".join(value) if isinstance(value, list) else str(value)
    if kind in {"execute_result", "display_data"}:
        value = output.get("data", {}).get("text/plain", "")
        return "".join(value) if isinstance(value, list) else str(value)
    if kind == "error":
        return f"{output.get('ename', 'Error')}: {output.get('evalue', '')}"
    return ""


def extract_notebooks(root: Path) -> list[dict]:
    notebooks = []
    for path in root.rglob("*.ipynb"):
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            notebooks.append({"source": relative(root, path), "error": str(error)})
            continue
        evidence = []
        executed = 0
        code_cells = 0
        for cell_index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            code_cells += 1
            if cell.get("execution_count") is not None:
                executed += 1
            text = "\n".join(output_text(item) for item in cell.get("outputs", []))
            matching = [line.strip() for line in text.splitlines() if KEYWORDS.search(line)]
            if matching:
                evidence.append(
                    {
                        "cell": cell_index,
                        "execution_count": cell.get("execution_count"),
                        "matched_output": matching[:80],
                    }
                )
        notebooks.append(
            {
                "source": relative(root, path),
                "sha256": digest(path),
                "code_cells": code_cells,
                "executed_code_cells": executed,
                "evidence_cells": evidence,
            }
        )
    return sorted(notebooks, key=lambda item: item["source"].casefold())


def extract_metric_files(root: Path) -> list[dict]:
    names = {"metrics_full.json", "metrics_report.json", "all_results.json", "eval_results.json"}
    records = []
    for path in root.rglob("*.json"):
        if path.name not in names or any(part in SKIP_PARTS for part in path.parts):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            records.append({"source": relative(root, path), "sha256": digest(path), "payload": payload})
        except (OSError, json.JSONDecodeError) as error:
            records.append({"source": relative(root, path), "error": str(error)})
    return sorted(records, key=lambda item: item["source"].casefold())


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract historical evidence without executing notebooks.")
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.archive_root.resolve()
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_root_label": root.name,
        "trainer_states": extract_trainer_states(root),
        "metric_files": extract_metric_files(root),
        "notebooks": extract_notebooks(root),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[history] {len(payload['trainer_states'])} trainer states, "
        f"{len(payload['metric_files'])} metric files, {len(payload['notebooks'])} notebooks"
    )


if __name__ == "__main__":
    main()
