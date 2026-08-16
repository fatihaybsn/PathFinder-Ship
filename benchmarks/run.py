from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.metrics import (
    bootstrap_ci,
    classification_summary,
    exact_match,
    expected_calibration_error,
    lexical_context_support,
    no_answer_match,
    repetition_rate,
    rouge_l_f1,
    tag_leak,
    token_f1,
)


CHAT_INSTRUCTION = (
    "Be helpful, friendly, and concise. Answer in English. If the user's request is ambiguous, "
    "ask exactly one clarifying question first; otherwise answer directly. Do not fabricate."
)
RAG_INSTRUCTION = (
    "Answer strictly using only the information in the Context. If the answer is not in the Context, "
    "say \"I don't know.\" Be brief and direct. Do not use outside knowledge. Answer in English."
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def load_experiments(path: Path) -> list[dict]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload["experiments"]


def setup_run(run_dir: Path) -> logging.Logger:
    for name in ("metrics", "tables", "summaries", "predictions", "figures", "logs", "status"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pathfinder_benchmark")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(run_dir / "logs" / "benchmark.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    environment_path = run_dir / "environment.json"
    if not environment_path.exists():
        try:
            freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()
        except Exception:
            freeze = []
        environment = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            "packages": freeze,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        try:
            import torch

            environment.update(
                {
                    "torch": torch.__version__,
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_version": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                }
            )
        except Exception as error:
            environment["torch_error"] = str(error)
        write_json(environment_path, environment)
    return logger


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percent / 100
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def file_size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def build_prompt(style: str, suite: str, record: dict) -> str:
    if suite == "command_v1":
        return f"command: {record['text']}"
    if suite == "chat_reference_v1" or suite == "chat_ifeval_v1":
        user_prompt = record["prompt"]
        if style == "early_task_prefix":
            return f"chat: {user_prompt}"
        raw_input = f"User: {user_prompt}\nAssistant:"
        instruction = CHAT_INSTRUCTION
        task = "chat"
        tag = "Assistant:"
    elif suite == "rag_v1":
        raw_input = record["input"]
        instruction = RAG_INSTRUCTION
        task = "rag_qa"
        tag = "Answer:"
    else:
        raise ValueError(f"Unsupported prompt suite: {suite}")

    cleaned = re.sub(rf"\s*{re.escape(tag)}\s*$", "", raw_input).rstrip()
    if style == "task_wrapped_v1":
        return f"Task: {task}\nInstruction: {instruction}\n{cleaned}\n{tag}"
    if style == "instruction_prefix":
        return f"Instruction: {instruction}\n{cleaned}\n{tag}"
    return f"{instruction}\n\n{cleaned}\n{tag}"


def load_suite(data_dir: Path, suite: str) -> list[dict]:
    if suite == "intent_v1":
        return read_jsonl(data_dir / "intent_v1.jsonl")
    if suite == "command_v1":
        return read_jsonl(data_dir / "command_v1.jsonl")
    if suite == "chat_reference_v1":
        return read_jsonl(data_dir / "chat_reference_v1.jsonl")
    if suite == "chat_ifeval_v1":
        return read_jsonl(data_dir / "ifeval_v1.jsonl")
    if suite == "rag_v1":
        return read_jsonl(data_dir / "ragbench_v1.jsonl") + read_jsonl(data_dir / "rag_project_v1.jsonl")
    if suite == "rag_project_v1":
        return read_jsonl(data_dir / "rag_project_v1.jsonl")
    raise ValueError(f"Unknown suite: {suite}")


def load_seq2seq(experiment: dict, model_dir: Path):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    if experiment["format"] == "transformers_full":
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_dir, torch_dtype=dtype, device_map="auto" if torch.cuda.is_available() else None
        )
    else:
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(experiment["base_model"], use_fast=True)
        base = AutoModelForSeq2SeqLM.from_pretrained(
            experiment["base_model"], torch_dtype=dtype, device_map="auto" if torch.cuda.is_available() else None
        )
        model = PeftModel.from_pretrained(base, model_dir)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def generate_texts(model, tokenizer, prompts: list[str], max_new_tokens: int, beams: int, batch_size: int = 4):
    import torch

    predictions = []
    latencies = []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for offset in range(0, len(prompts), batch_size):
            batch = prompts[offset : offset + batch_size]
            encoded = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=480
            ).to(device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=beams,
                no_repeat_ngram_size=3,
                early_stopping=beams > 1,
                use_cache=True,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            latencies.extend([elapsed / len(batch)] * len(batch))
            predictions.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return predictions, latencies


def metric_ci(values: list[float]) -> dict:
    return bootstrap_ci(values, lambda sample: statistics.mean(sample), iterations=1000, seed=42)


def score_reference_chat(records: list[dict], predictions: list[str]) -> dict:
    references = [record["reference"] for record in records]
    em = [exact_match(prediction, reference) for prediction, reference in zip(predictions, references)]
    f1 = [token_f1(prediction, reference) for prediction, reference in zip(predictions, references)]
    rouge = [rouge_l_f1(prediction, reference) for prediction, reference in zip(predictions, references)]
    result = {
        "count": len(records),
        "exact_match": metric_ci(em),
        "token_f1": metric_ci(f1),
        "rouge_l": metric_ci(rouge),
        "diagnostics": {
            "empty_rate": statistics.mean(not prediction.strip() for prediction in predictions),
            "tag_leak_rate": statistics.mean(tag_leak(prediction) for prediction in predictions),
            "mean_repetition_3gram": statistics.mean(repetition_rate(prediction) for prediction in predictions),
            "mean_prediction_reference_word_ratio": statistics.mean(
                len(prediction.split()) / max(1, len(reference.split()))
                for prediction, reference in zip(predictions, references)
            ),
        },
    }
    try:
        import sacrebleu

        result["sacrebleu"] = sacrebleu.corpus_bleu(predictions, [references]).score
    except Exception as error:
        result["sacrebleu_error"] = str(error)
    try:
        from bert_score import score as bert_score

        _, _, bert_f1 = bert_score(predictions, references, lang="en", verbose=False, batch_size=16)
        result["bertscore_f1"] = float(bert_f1.mean().item())
    except Exception as error:
        result["bertscore_error"] = str(error)
    return result


def score_rag(records: list[dict], predictions: list[str]) -> dict:
    references = [record["reference"] for record in records]
    em = [exact_match(prediction, reference) for prediction, reference in zip(predictions, references)]
    f1 = [token_f1(prediction, reference) for prediction, reference in zip(predictions, references)]
    rouge = [rouge_l_f1(prediction, reference) for prediction, reference in zip(predictions, references)]
    no_answer_indices = [index for index, record in enumerate(records) if not record.get("answerable", True)]
    return {
        "count": len(records),
        "exact_match": metric_ci(em),
        "token_f1": metric_ci(f1),
        "rouge_l": metric_ci(rouge),
        "no_answer_count": len(no_answer_indices),
        "no_answer_accuracy": (
            statistics.mean(no_answer_match(predictions[index]) for index in no_answer_indices)
            if no_answer_indices
            else None
        ),
        "lexical_context_support_proxy": statistics.mean(
            lexical_context_support(prediction, record.get("context", ""))
            for record, prediction in zip(records, predictions)
        ),
        "tag_leak_rate": statistics.mean(tag_leak(prediction) for prediction in predictions),
    }


def parse_command(text: str) -> list[int] | None:
    match = re.search(r"\[\s*([01])\s*,\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\]", text)
    return [int(value) for value in match.groups()] if match else None


def score_command(records: list[dict], predictions: list[str]) -> dict:
    command_pairs = [
        (record, prediction)
        for record, prediction in zip(records, predictions)
        if record.get("task") == "command"
    ]
    parsed = [parse_command(prediction) for _, prediction in command_pairs]
    expected = [record["labels"] for record, _ in command_pairs]
    valid = [value is not None for value in parsed]
    filled = [value if value is not None else [0, 0, 0, 0] for value in parsed]
    subset = [prediction == reference for prediction, reference in zip(filled, expected)]
    hamming = [
        sum(predicted_bit != expected_bit for predicted_bit, expected_bit in zip(prediction, reference)) / 4
        for prediction, reference in zip(filled, expected)
    ]
    bit_f1 = []
    for bit in range(4):
        labels = [str(row[bit]) for row in expected]
        predictions_for_bit = [str(row[bit]) for row in filled]
        bit_f1.append(classification_summary(labels, predictions_for_bit)["macro_f1"])
    return {
        "count": len(command_pairs),
        "format_valid_rate": statistics.mean(valid),
        "subset_accuracy": metric_ci([float(value) for value in subset]),
        "hamming_loss": statistics.mean(hamming),
        "mean_bit_macro_f1": statistics.mean(bit_f1),
        "per_bit_macro_f1": dict(zip(["open_camera", "take_photo", "close_camera", "object_detect"], bit_f1)),
    }


def score_ifeval(
    records: list[dict], predictions: list[str], data_dir: Path, output_dir: Path, code_root: Path | None
) -> dict:
    official_input_path = output_dir / "ifeval_official_input.jsonl"
    write_jsonl(
        official_input_path,
        [
            {
                "key": record["key"],
                "prompt": record["prompt"],
                "instruction_id_list": record["instruction_id_list"],
                "kwargs": record["kwargs"],
            }
            for record in records
        ],
    )
    response_path = output_dir / "ifeval_responses.jsonl"
    write_jsonl(
        response_path,
        [{"prompt": record["prompt"], "response": prediction} for record, prediction in zip(records, predictions)],
    )
    if code_root is None:
        return {"count": len(records), "scoring_error": "--ifeval-code-root was not supplied"}
    env = os.environ.copy()
    env["PYTHONPATH"] = str(code_root) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "instruction_following_eval.evaluation_main",
            f"--input_data={official_input_path}",
            f"--input_response_data={response_path}",
            f"--output_dir={output_dir}",
        ],
        check=True,
        env=env,
    )
    result = {"count": len(records)}
    for mode in ("strict", "loose"):
        rows = read_jsonl(output_dir / f"eval_results_{mode}.jsonl")
        prompt_values = [float(row["follow_all_instructions"]) for row in rows]
        instruction_values = [float(value) for row in rows for value in row["follow_instruction_list"]]
        result[mode] = {
            "prompt_accuracy": metric_ci(prompt_values),
            "instruction_accuracy": metric_ci(instruction_values),
            "instruction_count": len(instruction_values),
        }
    return result


def prediction_records(records: list[dict], predictions: list[str], latencies: list[float]) -> list[dict]:
    return [
        {
            "id": record["id"],
            "source": record.get("source"),
            "reference": record.get("reference"),
            "prediction": prediction,
            "latency_seconds": latency,
            "category": record.get("category"),
            "answerable": record.get("answerable"),
        }
        for record, prediction, latency in zip(records, predictions, latencies)
    ]


def run_flan(
    experiment: dict,
    models_root: Path,
    data_dir: Path,
    run_dir: Path,
    only_suite: str | None,
    ifeval_code_root: Path | None,
    logger: logging.Logger,
    run_beam_comparison: bool = True,
    generation_batch_size: int = 4,
) -> None:
    import torch

    model_dir = models_root / experiment["id"]
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)
    logger.info("loading %s from %s", experiment["id"], model_dir)
    model, tokenizer = load_seq2seq(experiment, model_dir)
    metrics = {
        "schema_version": 1,
        "experiment_id": experiment["id"],
        "family": experiment["family"],
        "display_name": experiment.get("display_name", experiment["id"]),
        "development_status": experiment.get("status"),
        "training_change": experiment.get("training_change"),
        "protocol": experiment.get("protocol", "benchmark_v1"),
        "status": "complete",
        "artifact_size_bytes": file_size_bytes(model_dir),
        "suites": {},
    }
    suites = [suite for suite in experiment["suites"] if suite != "onnx_parity_v1"]
    if only_suite:
        suites = [suite for suite in suites if suite == only_suite]
    for suite in suites:
        records = load_suite(data_dir, suite)
        prompts = [build_prompt(experiment["prompt_style"], suite, record) for record in records]
        max_new_tokens = 512 if suite == "chat_ifeval_v1" else (256 if suite == "chat_reference_v1" else 64)
        batch_size = 1 if suite == "chat_ifeval_v1" else generation_batch_size
        logger.info("%s: %s examples", suite, len(records))
        predictions, latencies = generate_texts(model, tokenizer, prompts, max_new_tokens, beams=1, batch_size=batch_size)
        prediction_path = run_dir / "predictions" / f"{experiment['id']}__{suite}.jsonl"
        write_jsonl(prediction_path, prediction_records(records, predictions, latencies))
        if suite == "command_v1":
            suite_metrics = score_command(records, predictions)
        elif suite == "chat_reference_v1":
            suite_metrics = score_reference_chat(records, predictions)
        elif suite in {"rag_v1", "rag_project_v1"}:
            suite_metrics = score_rag(records, predictions)
        elif suite == "chat_ifeval_v1":
            suite_metrics = score_ifeval(
                records,
                predictions,
                data_dir,
                run_dir / "metrics" / f"{experiment['id']}__ifeval_detail",
                ifeval_code_root,
            )
        else:
            raise ValueError(suite)
        suite_metrics["runtime"] = {
            "device": str(next(model.parameters()).device),
            "batch_size": batch_size,
            "p50_latency_seconds_per_example": percentile(latencies, 50),
            "p95_latency_seconds_per_example": percentile(latencies, 95),
            "total_generation_seconds": sum(latencies),
        }
        metrics["suites"][suite] = suite_metrics

        if run_beam_comparison and experiment["id"] == "flan_large_lora_second_try" and suite in {"chat_reference_v1", "rag_v1", "rag_project_v1"}:
            beam_predictions, beam_latencies = generate_texts(
                model, tokenizer, prompts, max_new_tokens, beams=4, batch_size=1
            )
            beam_path = run_dir / "predictions" / f"{experiment['id']}__{suite}__beam4.jsonl"
            write_jsonl(beam_path, prediction_records(records, beam_predictions, beam_latencies))
            beam_metrics = score_reference_chat(records, beam_predictions) if suite == "chat_reference_v1" else score_rag(records, beam_predictions)
            beam_metrics["runtime"] = {
                "device": str(next(model.parameters()).device),
                "batch_size": 1,
                "p50_latency_seconds_per_example": percentile(beam_latencies, 50),
                "p95_latency_seconds_per_example": percentile(beam_latencies, 95),
            }
            metrics["suites"][f"{suite}__beam4"] = beam_metrics
        write_json(run_dir / "metrics" / f"{experiment['id']}.json", metrics)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    write_json(run_dir / "metrics" / f"{experiment['id']}.json", metrics)


def run_minilm(experiment: dict, models_root: Path, data_dir: Path, run_dir: Path) -> None:
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    model_dir = models_root / experiment["id"]
    model_path = next(model_dir.rglob("*.onnx"))
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    id2label = {int(key): value for key, value in config["id2label"].items()}
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_names = {item.name for item in session.get_inputs()}
    records = load_suite(data_dir, "intent_v1")
    labels = [record["intent"] for record in records]
    predictions = []
    confidences = []
    latencies = []
    output_rows = []
    for index, record in enumerate(records):
        encoded = tokenizer(record["text"], return_tensors="np", truncation=True, max_length=96, padding="max_length")
        inputs = {name: encoded[name].astype("int64") for name in input_names if name in encoded}
        if index < 10:
            session.run(None, inputs)
        started = time.perf_counter()
        logits = session.run(None, inputs)[0][0]
        elapsed = time.perf_counter() - started
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        predicted_index = int(probabilities.argmax())
        prediction = id2label[predicted_index]
        predictions.append(prediction)
        confidences.append(float(probabilities[predicted_index]))
        latencies.append(elapsed)
        output_rows.append(
            {
                "id": record["id"], "text": record["text"], "reference": record["intent"],
                "prediction": prediction, "confidence": confidences[-1], "latency_seconds": elapsed,
            }
        )
    summary = classification_summary(labels, predictions)
    summary.update(
        {
            "schema_version": 1,
            "experiment_id": experiment["id"],
            "family": experiment["family"],
            "status": "complete",
            "count": len(records),
            "ece_10_bins": expected_calibration_error(
                confidences, [expected == predicted for expected, predicted in zip(labels, predictions)]
            ),
            "runtime": {
                "provider": "CPUExecutionProvider", "batch_size": 1,
                "p50_latency_seconds": percentile(latencies, 50),
                "p95_latency_seconds": percentile(latencies, 95),
            },
            "artifact_size_bytes": file_size_bytes(model_dir),
        }
    )
    write_json(run_dir / "metrics" / f"{experiment['id']}.json", summary)
    write_jsonl(run_dir / "predictions" / f"{experiment['id']}__intent_v1.jsonl", output_rows)
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

        ordered_labels = [id2label[index] for index in sorted(id2label)]
        matrix = confusion_matrix(labels, predictions, labels=ordered_labels)
        figure, axis = plt.subplots(figsize=(8, 7))
        ConfusionMatrixDisplay(matrix, display_labels=ordered_labels).plot(ax=axis, cmap="Blues", colorbar=False)
        axis.set_title("MiniLM Intent — Focused Evidence v1 Confusion Matrix")
        figure.tight_layout()
        figure.savefig(run_dir / "figures" / "minilm_intent_confusion_matrix.png", dpi=180)
        plt.close(figure)
    except Exception as error:
        summary["figure_error"] = str(error)
        write_json(run_dir / "metrics" / f"{experiment['id']}.json", summary)


def run_onnx_parity(
    experiment: dict, models_root: Path, data_dir: Path, run_dir: Path, examples_per_suite: int = 100
) -> None:
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    from transformers import AutoTokenizer

    onnx_dir = models_root / experiment["id"] / "onnx"
    if not onnx_dir.exists():
        raise FileNotFoundError(onnx_dir)
    tokenizer = AutoTokenizer.from_pretrained(onnx_dir, use_fast=True)
    model = ORTModelForSeq2SeqLM.from_pretrained(onnx_dir, provider="CPUExecutionProvider")
    all_metrics = {
        "schema_version": 1,
        "experiment_id": experiment["id"],
        "display_name": experiment.get("display_name", experiment["id"]),
        "protocol": experiment.get("protocol", "benchmark_v1"),
        "status": "complete",
        "suites": {},
    }
    parity_suites = [suite for suite in experiment["suites"] if suite in {"chat_reference_v1", "rag_v1", "rag_project_v1"}]
    for suite in parity_suites:
        records = load_suite(data_dir, suite)[:examples_per_suite]
        pytorch_rows = read_jsonl(run_dir / "predictions" / f"{experiment['id']}__{suite}.jsonl")[:examples_per_suite]
        prompts = [build_prompt(experiment["prompt_style"], suite, record) for record in records]
        max_new = 256 if suite == "chat_reference_v1" else 64
        predictions = []
        latencies = []
        for prompt in prompts:
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=480)
            started = time.perf_counter()
            output = model.generate(**encoded, max_new_tokens=max_new, do_sample=False, num_beams=1)
            latencies.append(time.perf_counter() - started)
            predictions.append(tokenizer.decode(output[0], skip_special_tokens=True))
        parity = [token_f1(prediction, row["prediction"]) for prediction, row in zip(predictions, pytorch_rows)]
        references = [record["reference"] for record in records]
        pytorch_quality = statistics.mean(token_f1(row["prediction"], ref) for row, ref in zip(pytorch_rows, references))
        onnx_quality = statistics.mean(token_f1(prediction, ref) for prediction, ref in zip(predictions, references))
        all_metrics["suites"][suite] = {
            "count": len(records),
            "mean_output_token_f1_vs_pytorch": statistics.mean(parity),
            "exact_output_agreement": statistics.mean(
                exact_match(prediction, row["prediction"]) for prediction, row in zip(predictions, pytorch_rows)
            ),
            "pytorch_reference_token_f1": pytorch_quality,
            "onnx_reference_token_f1": onnx_quality,
            "quality_delta_onnx_minus_pytorch": onnx_quality - pytorch_quality,
            "p50_latency_seconds": percentile(latencies, 50),
            "p95_latency_seconds": percentile(latencies, 95),
        }
        write_jsonl(
            run_dir / "predictions" / f"{experiment['id']}__{suite}__onnx_int8.jsonl",
            prediction_records(records, predictions, latencies),
        )
    all_metrics["artifact_size_bytes"] = file_size_bytes(onnx_dir)
    write_json(run_dir / "metrics" / f"{experiment['id']}__onnx_parity.json", all_metrics)


def run_yolo(experiment: dict, models_root: Path, run_dir: Path, logger: logging.Logger) -> None:
    from ultralytics import YOLO

    model_dir = models_root / experiment["id"]
    candidates = sorted(list(model_dir.glob("yolo11*.pt")) + list(model_dir.glob("yolo11l.onnx")))
    if not candidates:
        raise FileNotFoundError(f"No YOLO11 artifacts in {model_dir}")
    metrics = {
        "schema_version": 1,
        "experiment_id": experiment["id"],
        "claim": "pretrained_integration_only",
        "status": "complete",
        "models": {},
    }
    for artifact in candidates:
        logger.info("YOLO COCO validation: %s", artifact.name)
        model = YOLO(str(artifact))
        result = model.val(
            data="coco.yaml", split="val", imgsz=640, batch=16, device=0,
            plots=True, project=str(run_dir / "figures" / "yolo_runs"), name=artifact.stem, exist_ok=True,
        )
        metrics["models"][artifact.name] = {
            "size_bytes": artifact.stat().st_size,
            "map50_95": float(result.box.map),
            "map50": float(result.box.map50),
            "precision": float(result.box.mp),
            "recall": float(result.box.mr),
            "speed_ms": dict(result.speed),
        }
        write_json(run_dir / "metrics" / f"{experiment['id']}.json", metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a PathFinderShip model-evidence manifest in Lightning AI.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--experiment", default="all")
    parser.add_argument("--only-suite")
    parser.add_argument("--ifeval-code-root", type=Path)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--skip-beam-comparison", action="store_true")
    parser.add_argument("--onnx-parity-examples-per-suite", type=int, default=100)
    parser.add_argument("--force", action="store_true", help="Rerun even when status is already complete.")
    args = parser.parse_args()
    logger = setup_run(args.run_dir)
    experiments = load_experiments(args.manifest)
    selected = experiments if args.experiment == "all" else [item for item in experiments if item["id"] == args.experiment]
    if not selected:
        raise SystemExit(f"Unknown experiment: {args.experiment}")

    for experiment in selected:
        status_path = args.run_dir / "status" / f"{experiment['id']}.json"
        if status_path.exists() and not args.force:
            previous = json.loads(status_path.read_text(encoding="utf-8"))
            if previous.get("status") == "complete":
                logger.info("SKIP already complete %s (use --force to rerun)", experiment["id"])
                continue
        try:
            logger.info("START %s", experiment["id"])
            if experiment["family"] == "minilm_intent":
                run_minilm(experiment, args.models_root, args.data_dir, args.run_dir)
            elif experiment["family"] in {"flan_early_chat_command", "flan_large_chat_rag"}:
                run_flan(
                    experiment, args.models_root, args.data_dir, args.run_dir,
                    args.only_suite, args.ifeval_code_root, logger,
                    run_beam_comparison=not args.skip_beam_comparison,
                    generation_batch_size=args.generation_batch_size,
                )
                if experiment["id"] == "flan_large_lora_second_try" and not args.only_suite:
                    run_onnx_parity(
                        experiment, args.models_root, args.data_dir, args.run_dir,
                        examples_per_suite=args.onnx_parity_examples_per_suite,
                    )
            elif experiment["family"] == "vision_integration":
                run_yolo(experiment, args.models_root, args.run_dir, logger)
            else:
                raise ValueError(f"Unsupported family: {experiment['family']}")
            write_json(status_path, {"experiment_id": experiment["id"], "status": "complete"})
            logger.info("COMPLETE %s", experiment["id"])
        except Exception as error:
            failure = {
                "experiment_id": experiment["id"],
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            write_json(status_path, failure)
            logger.exception("FAILED %s", experiment["id"])
            continue


if __name__ == "__main__":
    main()
