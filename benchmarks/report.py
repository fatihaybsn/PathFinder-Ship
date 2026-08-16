from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


def read_metrics(metrics_dir: Path) -> list[dict]:
    records = []
    for path in sorted(metrics_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            records.append({"file": path.name, "payload": payload})
        except json.JSONDecodeError:
            records.append({"file": path.name, "payload": {"status": "invalid_json"}})
    return records


def scalar(value):
    if isinstance(value, dict) and "estimate" in value:
        return value["estimate"]
    return value if isinstance(value, (int, float, str, bool)) or value is None else None


def flatten(prefix: str, value, output: dict) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            flatten(f"{prefix}.{key}" if prefix else key, child, output)
    elif isinstance(value, (int, float, str, bool)) or value is None:
        output[prefix] = value


def write_tables(run_dir: Path, metrics: list[dict]) -> None:
    rows = []
    for record in metrics:
        flattened = {"metrics_file": record["file"]}
        flatten("", record["payload"], flattened)
        rows.append(flattened)
    fieldnames = sorted({key for row in rows for key in row})
    with (run_dir / "tables" / "all_metrics_flat.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for record in metrics:
        payload = record["payload"]
        experiment_id = payload.get("experiment_id", record["file"].removesuffix(".json"))
        if "accuracy" in payload:
            summary_rows.append(
                {"experiment": experiment_id, "suite": "intent_v1", "metric": "accuracy", "value": payload["accuracy"]}
            )
        for suite, suite_metrics in payload.get("suites", {}).items():
            for metric_name in (
                "exact_match", "token_f1", "rouge_l", "subset_accuracy",
                "mean_output_token_f1_vs_pytorch", "quality_delta_onnx_minus_pytorch",
            ):
                if metric_name in suite_metrics:
                    value = scalar(suite_metrics[metric_name])
                    if value is not None:
                        summary_rows.append(
                            {"experiment": experiment_id, "suite": suite, "metric": metric_name, "value": value}
                        )
            for mode in ("strict", "loose"):
                if mode in suite_metrics:
                    value = scalar(suite_metrics[mode].get("prompt_accuracy"))
                    summary_rows.append(
                        {"experiment": experiment_id, "suite": suite, "metric": f"{mode}_prompt_accuracy", "value": value}
                    )
        for model_name, model_metrics in payload.get("models", {}).items():
            if "map50_95" in model_metrics:
                summary_rows.append(
                    {"experiment": model_name, "suite": "vision_coco_v1", "metric": "map50_95", "value": model_metrics["map50_95"]}
                )
    with (run_dir / "tables" / "benchmark_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment", "suite", "metric", "value"])
        writer.writeheader()
        writer.writerows(summary_rows)


def metric_estimate(payload: dict, suite: str, metric: str):
    value = payload.get("suites", {}).get(suite, {}).get(metric)
    return scalar(value)


def write_focused_comparison(
    run_dir: Path, metrics: list[dict], experiment_manifest: Path | None, statuses: list[dict] | None = None
) -> None:
    """Join frozen training metadata with new shared-suite scores.

    The table deliberately includes rejected and intermediate runs. Winners are
    marked per task metric; no cross-task composite score is invented.
    """
    if experiment_manifest is None or not experiment_manifest.exists():
        return
    import yaml

    registry = yaml.safe_load(experiment_manifest.read_text(encoding="utf-8"))["experiments"]
    payloads = {
        record["payload"].get("experiment_id"): record["payload"]
        for record in metrics
        if not record["file"].endswith("__onnx_parity.json")
    }
    status_by_id = {item.get("experiment_id"): item for item in (statuses or [])}
    rows = []
    for experiment in registry:
        payload = payloads.get(experiment["id"], {})
        row = {
            "experiment_id": experiment["id"],
            "display_name": experiment.get("display_name", experiment["id"]),
            "original_folder": experiment.get("original_folder", experiment.get("source", "")),
            "development_status": experiment.get("status", ""),
            "training_change": experiment.get("training_change", ""),
            "historical_result": experiment.get("historical_result", ""),
            "historical_eval_loss": experiment.get("historical_eval_loss"),
            "benchmark_status": payload.get("status", status_by_id.get(experiment["id"], {}).get("status", "not_run")),
            "benchmark_error": status_by_id.get(experiment["id"], {}).get("error", ""),
            "intent_accuracy": payload.get("accuracy"),
            "intent_macro_f1": payload.get("macro_f1"),
            "chat_token_f1": metric_estimate(payload, "chat_reference_v1", "token_f1"),
            "chat_rouge_l": metric_estimate(payload, "chat_reference_v1", "rouge_l"),
            "rag_token_f1": metric_estimate(payload, "rag_project_v1", "token_f1"),
            "rag_rouge_l": metric_estimate(payload, "rag_project_v1", "rouge_l"),
            "rag_exact_match": metric_estimate(payload, "rag_project_v1", "exact_match"),
        }
        rows.append(row)

    winner_columns = ("intent_accuracy", "intent_macro_f1", "chat_token_f1", "chat_rouge_l", "rag_token_f1", "rag_rouge_l", "rag_exact_match")
    for column in winner_columns:
        valid = [row[column] for row in rows if isinstance(row[column], (int, float))]
        best = max(valid) if valid else None
        for row in rows:
            row[f"best_{column}"] = best is not None and row[column] == best

    csv_path = run_dir / "tables" / "focused_model_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def show(value) -> str:
        return "—" if not isinstance(value, (int, float)) else f"{value:.4f}"

    markdown = [
        "# Focused Evidence v1 — Model Comparison",
        "",
        "Rejected, intermediate, milestone, and final-candidate runs are all retained. **Best** is assigned independently within each compatible task; there is no composite score.",
        "",
        "| Experiment | Status | Training change | Historical eval loss | Chat token F1 | RAG token F1 | RAG EM |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["experiment_id"] == "minilm_intent_int8":
            chat = f"Intent accuracy {show(row['intent_accuracy'])}"
            rag_f1 = f"Intent macro-F1 {show(row['intent_macro_f1'])}"
            rag_em = "—"
        else:
            chat = show(row["chat_token_f1"]) + (" **BEST**" if row["best_chat_token_f1"] else "")
            rag_f1 = show(row["rag_token_f1"]) + (" **BEST**" if row["best_rag_token_f1"] else "")
            rag_em = show(row["rag_exact_match"]) + (" **BEST**" if row["best_rag_exact_match"] else "")
        markdown.append(
            f"| {row['display_name']} | {row['development_status']} | {row['training_change']} | "
            f"{show(row['historical_eval_loss'])} | {chat} | {rag_f1} | {rag_em} |"
        )
    (run_dir / "tables" / "focused_model_comparison.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        flan_rows = [row for row in rows if row["experiment_id"].startswith("flan_")]
        labels = [row["display_name"] for row in flan_rows]
        figure, axes = plt.subplots(1, 2, figsize=(15, max(5, len(labels) * 0.75)), sharey=True)
        colors = ["#c0392b" if row["development_status"] == "rejected" else "#4c78a8" for row in flan_rows]
        for axis, column, title in (
            (axes[0], "chat_token_f1", "Chat token F1"),
            (axes[1], "rag_token_f1", "RAG token F1"),
        ):
            values = [row[column] if isinstance(row[column], (int, float)) else 0.0 for row in flan_rows]
            bars = axis.barh(labels, values, color=colors)
            if values:
                best_index = max(range(len(values)), key=values.__getitem__)
                bars[best_index].set_color("#2ca02c")
            axis.set_xlim(0, 1)
            axis.set_title(title)
            axis.grid(axis="x", alpha=0.25)
            axis.bar_label(bars, fmt="%.3f", padding=3)
        figure.suptitle("Focused Evidence v1 — Same frozen tests, all Flan-T5 LoRA attempts")
        figure.tight_layout()
        figure.savefig(run_dir / "figures" / "focused_flan_model_comparison.png", dpi=180, bbox_inches="tight")
        plt.close(figure)
    except Exception as error:
        (run_dir / "logs" / "focused_comparison_figure_error.txt").write_text(str(error), encoding="utf-8")


def write_summary(run_dir: Path, statuses: list[dict], metric_count: int) -> None:
    complete = [item for item in statuses if item.get("status") == "complete"]
    failed = [item for item in statuses if item.get("status") == "failed"]
    lines = [
        "PathFinderShip Model Evidence Result Summary",
        "============================================",
        f"Completed experiments: {len(complete)}",
        f"Failed experiments: {len(failed)}",
        f"Metric files: {metric_count}",
        "",
        "Completed:",
        *[f"- {item.get('experiment_id')}" for item in complete],
        "",
        "Failed:",
        *[f"- {item.get('experiment_id')}: {item.get('error_type')} — {item.get('error')}" for item in failed],
        "",
        "A failed model is retained as evidence and is not removed from the report.",
    ]
    (run_dir / "summaries" / "benchmark_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures(run_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd

        table = pd.read_csv(run_dir / "tables" / "benchmark_summary.csv")
        for suite in sorted(table["suite"].dropna().unique()):
            subset = table[table["suite"] == suite]
            if subset.empty:
                continue
            pivot = subset.pivot_table(index="experiment", columns="metric", values="value", aggfunc="first")
            axis = pivot.plot(kind="bar", figsize=(max(9, len(pivot) * 1.2), 5))
            axis.set_title(f"PathFinderShip Model Evidence — {suite}")
            axis.set_ylabel("Metric value")
            axis.set_xlabel("")
            axis.grid(axis="y", alpha=0.25)
            axis.figure.tight_layout()
            safe_suite = "".join(character if character.isalnum() or character in "-_" else "_" for character in suite)
            axis.figure.savefig(run_dir / "figures" / f"benchmark_{safe_suite}.png", dpi=180)
            plt.close(axis.figure)
    except Exception as error:
        (run_dir / "logs" / "figure_generation_error.txt").write_text(str(error), encoding="utf-8")


def write_historical_figures(run_dir: Path, historical_evidence: Path | None) -> None:
    if historical_evidence is None or not historical_evidence.exists():
        return
    try:
        import matplotlib.pyplot as plt

        payload = json.loads(historical_evidence.read_text(encoding="utf-8"))
        train_points = []
        validation_points = []
        for notebook in payload.get("notebooks", []):
            if not notebook.get("source", "").endswith("NLPP/Multi Task Models/Flan T5 Base/Flan_T5_early_Base.ipynb"):
                continue
            for cell in notebook.get("evidence_cells", []):
                current_epoch = None
                for line in cell.get("matched_output", []):
                    if match := re.search(r"Epoch (\d+) Train Loss: ([0-9.]+)", line):
                        current_epoch = int(match.group(1))
                        train_points.append((current_epoch, float(match.group(2))))
                    elif current_epoch and (match := re.search(r"Validation Loss: ([0-9.]+)", line)):
                        validation_points.append((current_epoch, float(match.group(1))))
        if train_points and validation_points:
            figure, axis = plt.subplots(figsize=(9, 5))
            axis.plot(*zip(*train_points), marker="o", label="Train loss")
            axis.plot(*zip(*validation_points), marker="o", label="Validation loss")
            best_epoch, best_loss = min(validation_points, key=lambda item: item[1])
            axis.scatter([best_epoch], [best_loss], color="red", zorder=4, label=f"Best: epoch {best_epoch}, {best_loss:.4f}")
            axis.set_title("Historical Early Flan-T5 Base Training Curve")
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Loss")
            axis.grid(alpha=0.25)
            axis.legend()
            figure.tight_layout()
            figure.savefig(run_dir / "figures" / "historical_early_flan_t5_base_loss.png", dpi=180)
            plt.close(figure)

        figure, axis = plt.subplots(figsize=(10, 6))
        plotted = False
        for state in payload.get("trainer_states", []):
            points = [
                (item.get("step"), item.get("eval_loss"))
                for item in state.get("log_history", [])
                if item.get("step") is not None and item.get("eval_loss") is not None
            ]
            if not points:
                continue
            label = Path(state["source"]).parent.as_posix().replace("Primee/Models/", "")
            axis.plot(*zip(*points), marker="o", label=label)
            plotted = True
        if plotted:
            axis.set_title("Historical Flan-T5 Large LoRA Evaluation Loss")
            axis.set_xlabel("Training step")
            axis.set_ylabel("Evaluation loss")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
            figure.tight_layout()
            figure.savefig(run_dir / "figures" / "historical_flan_large_eval_loss.png", dpi=180)
        plt.close(figure)
    except Exception as error:
        (run_dir / "logs" / "historical_figure_error.txt").write_text(str(error), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(run_dir: Path) -> dict:
    manifest_path = run_dir / "artifact_manifest.json"
    files = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        files.append(
            {
                "relative_path": path.relative_to(run_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate and package a Lightning benchmark run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path, required=True)
    parser.add_argument("--historical-evidence", type=Path)
    parser.add_argument("--experiment-manifest", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    metrics = read_metrics(run_dir / "metrics")
    statuses = []
    for path in sorted((run_dir / "status").glob("*.json")):
        statuses.append(json.loads(path.read_text(encoding="utf-8")))
    write_tables(run_dir, metrics)
    write_focused_comparison(run_dir, metrics, args.experiment_manifest, statuses)
    write_summary(run_dir, statuses, len(metrics))
    write_figures(run_dir)
    write_historical_figures(run_dir, args.historical_evidence)
    completion = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "complete": sum(item.get("status") == "complete" for item in statuses),
        "failed": sum(item.get("status") == "failed" for item in statuses),
        "note": "Orchestration completed; failed experiments are documented in status/.",
    }
    (run_dir / "RUN_COMPLETE.txt").write_text(json.dumps(completion, indent=2), encoding="utf-8")
    manifest = write_manifest(run_dir)
    args.zip_path.parent.mkdir(parents=True, exist_ok=True)
    archive_base = args.zip_path.with_suffix("")
    created = Path(shutil.make_archive(str(archive_base), "zip", root_dir=run_dir.parent, base_dir=run_dir.name))
    if created != args.zip_path:
        if args.zip_path.exists():
            args.zip_path.unlink()
        created.replace(args.zip_path)
    print(
        json.dumps(
            {"run_dir": str(run_dir), "zip": str(args.zip_path), "files": manifest["file_count"], **completion},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
