# PathFinderShip Model Evidence Protocol

## Claims policy

- MiniLM and Flan-T5 are pretrained checkpoints fine-tuned on project data; they were not trained from scratch.
- Rejected, intermediate, runner-up, and selected experiments remain visible.
- `1.2x Chat` and `2x RAG` are original aliases for task-loss weights (`1.2` and `2.0`), not model scaling or dataset duplication.
- Compatible tasks are reported separately; no composite score combines Chat, RAG, and intent classification.
- Historical training-loss values remain separate from Retraining Evaluation v2 metrics.
- YOLO is pretrained integration work and is outside the custom-training evidence claim.

## Published suites

| Suite | Models | Count | Primary measurements |
|---|---|---:|---|
| `intent_v1` | MiniLM INT8 ONNX | 1,000 | accuracy, macro/weighted F1, ECE, confusion matrix |
| `chat_reference_v1` | six updated Flan-T5 Large LoRA adapters | 300 | token-F1 |
| `rag_project_v1` | the same six adapters | 160 | token-F1, exact match |

## Experiment comparison

Retraining Evaluation v2 compares q/v-only LoRA, RAG-loss weight 2.0, Chat-loss weight 1.2 at two checkpoints, and two custom trainer/loss iterations. `My Class/Second Try` is selected because it leads every reported metric, not because of a synthetic cross-task score.

## Evidence contract

Publication includes machine-readable JSON/CSV, an auditable workbook, figures, an experiment log, a dataset card, and explicit limitations. Old raw prediction files are not paired with retrained-v2 summary values. ONNX is labeled as the retrained final model only after its export/hash/parity evidence is confirmed against the updated Second Try weights.

See [`docs/model-development/results/retraining-v2/RESULT_CARD.md`](docs/model-development/results/retraining-v2/RESULT_CARD.md).
