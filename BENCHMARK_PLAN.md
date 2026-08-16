# PathFinderShip Model Evidence and Benchmark Plan

This document separates evidence recovered from historical training runs from new, reproducible evaluation results. Historical numbers are never mixed into the Focused Evidence v1 leaderboard.

## Claims policy

- MiniLM and Flan-T5 are pretrained checkpoints fine-tuned on project datasets. They were not trained from scratch.
- Failed and intermediate Flan-T5 experiments remain visible so that the development path is auditable.
- YOLO is outside this first evidence run. No custom YOLO training claim is made.
- Identical files are collapsed by SHA-256 and listed as aliases; a copied weight is not counted as a separate trained model.
- Files in the source archive are read-only. Deletion is outside this benchmark and requires a later explicit approval.

## Evidence layers

1. **Historical evidence** — executed notebook outputs, trainer states, configuration files, saved weights, old metrics, and their hashes.
2. **Focused Evidence v1** — one frozen project test set used consistently within each compatible model family.
3. **Artifact evidence** — model size, SHA-256, load test, inference environment, ONNX parity, and downloadable publication revision.

## Suites

| Suite | Models | Primary measurements |
|---|---|---|
| `intent_v1` | MiniLM INT8 ONNX | accuracy, macro/weighted F1, per-class metrics, confusion matrix, ECE |
| `chat_reference_v1` | six Flan-T5 Large LoRA attempts | token F1, ROUGE-L, BERTScore, SacreBLEU, repetition and truncation diagnostics |
| `rag_project_v1` | six Flan-T5 Large LoRA attempts | EM, token F1, ROUGE-L, no-answer accuracy, lexical context-support proxy |
| `onnx_parity_v1` | final LoRA, merged PyTorch and INT8 ONNX | task-metric delta, output validity, size, latency and memory |

There is no composite score across unrelated tasks. New quality metrics receive paired bootstrap 95% confidence intervals. Latency results are comparable only when runtime, provider, batch size, and hardware match.

## Reproducibility contract

- Random seed: `42`.
- Main generation: greedy (`do_sample=false`, `num_beams=1`).
- Chat maximum: 256 new tokens; RAG maximum: 64 new tokens.
- Generation uses batch 16 on the recommended H100; MiniLM and ONNX timing runs on CPU.
- ONNX parity uses the first frozen 25 Chat and 25 RAG rows. The indices are deterministic and not hand-picked.
- Every run writes metrics, predictions, tables, figures, logs, environment metadata, artifact hashes, and `RUN_COMPLETE.txt`.

## Publication gate

The README is finalized only after the Lightning result bundle passes schema and hash validation. The comparison retains rejected `kötü`, both `1.2x` checkpoints, `2x rag`, First Try, and Second Try, then highlights the best score separately for Chat and RAG. Hugging Face upload remains manual. No local source model is deleted until its published revision can be downloaded and smoke-tested in a clean environment.
