# PathFinderShip Model Evidence and Benchmark Plan

This document separates evidence recovered from historical training runs from new, reproducible evaluation results. Historical numbers are never mixed into the Benchmark v1 leaderboard.

## Claims policy

- MiniLM and Flan-T5 are pretrained checkpoints fine-tuned on project datasets. They were not trained from scratch.
- Failed and intermediate Flan-T5 experiments remain visible so that the development path is auditable.
- YOLO11 and YOLO-NAS are pretrained integration/ONNX work. No custom YOLO training claim is made because no training dataset, run directory, `best.pt`, or training metrics were found.
- Identical files are collapsed by SHA-256 and listed as aliases; a copied weight is not counted as a separate trained model.
- Files in the source archive are read-only. Deletion is outside this benchmark and requires a later explicit approval.

## Evidence layers

1. **Historical evidence** — executed notebook outputs, trainer states, configuration files, saved weights, old metrics, and their hashes.
2. **Benchmark v1** — one frozen, decontaminated test set used consistently within each compatible model family.
3. **Artifact evidence** — model size, SHA-256, load test, inference environment, ONNX parity, and downloadable publication revision.

## Suites

| Suite | Models | Primary measurements |
|---|---|---|
| `intent_v1` | MiniLM INT8 ONNX | accuracy, macro/weighted F1, per-class metrics, confusion matrix, ECE |
| `command_v1` | early Flan-T5 Small/Base variants | exact label-set accuracy, bit F1, Hamming loss, output validity |
| `chat_ifeval_v1` | compatible Flan-T5 variants | IFEval strict/loose instruction accuracy |
| `chat_reference_v1` | compatible Flan-T5 variants | BERTScore F1, ROUGE-L, SacreBLEU, repetition and truncation diagnostics |
| `rag_v1` | Flan-T5 Large LoRA variants | EM, token F1, ROUGE-L, no-answer accuracy, lexical context-support proxy |
| `onnx_parity_v1` | final LoRA, merged PyTorch and INT8 ONNX | task-metric delta, output validity, size, latency and memory |
| `vision_coco_v1` | YOLO11 n/s/m/l/x and YOLO11l ONNX | COCO mAP50-95, mAP50, precision, recall, size and latency |

There is no composite score across unrelated tasks. New quality metrics receive paired bootstrap 95% confidence intervals. Latency results are comparable only when runtime, provider, batch size, and hardware match.

## Reproducibility contract

- Random seed: `42`.
- Main generation: greedy (`do_sample=false`, `num_beams=1`).
- Chat maximum: 512 new tokens for IFEval and 256 for reference chat.
- RAG and command maximum: 64 new tokens.
- Final-model decoding ablation: greedy versus four beams.
- Latency: batch 1, 10 warm-ups, at least 100 measured calls.
- Every run writes metrics, predictions, tables, figures, logs, environment metadata, artifact hashes, and `RUN_COMPLETE.txt`.

## Publication gate

The README is finalized only after the Lightning result bundle passes schema and hash validation. Hugging Face upload is manual and limited to the final Second Try LoRA/ONNX, MiniLM INT8, the 1.2x step-1980 adapter, and the 2x-RAG step-1320 adapter. No local source model is deleted until its published revision can be downloaded and smoke-tested in a clean environment.
