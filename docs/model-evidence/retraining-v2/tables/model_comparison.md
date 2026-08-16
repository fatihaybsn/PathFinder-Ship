# Retraining Evaluation v2 — Model Comparison

All Flan-T5 entries are pretrained `google/flan-t5-large` checkpoints fine-tuned with LoRA on project data. They are not models trained from scratch.

`1.2x Chat` and `2x RAG` are original folder aliases. They mean **task-loss weights**, not model-size scaling or dataset duplication.

| Experiment | Deliberate training change | Chat token-F1 | RAG token-F1 | RAG EM | Decision |
|---|---|---:|---:|---:|---|
| LoRA q/v — `kötü` | q/v-only LoRA, r=16, alpha=32 | 0.3942 | 0.8127 | 0.7063 | Rejected: RAG is strong, but Chat is weak |
| RAG-loss-weight 2.0 — step 1320 | RAG loss weight 2.0; Chat 1.0 | 0.3778 | 0.5864 | 0.4625 | Milestone; weighting did not improve the shared evaluation |
| Chat-loss-weight 1.2 — step 1485 | Chat loss weight 1.2; RAG 1.0 | 0.4695 | 0.6719 | 0.5563 | Intermediate checkpoint |
| Chat-loss-weight 1.2 — step 1980 | Same weighting, more training steps | 0.4684 | 0.6587 | 0.5438 | Slight regression versus step 1485 |
| My Class — First Try | First custom trainer/loss iteration | 0.4937 | 0.8421 | 0.7438 | Runner-up; strong on both tasks |
| **My Class — Second Try** | Chat weight 1.7, task smoothing, partial R-Drop | **0.5216** | **0.8894** | **0.7938** | **Selected final: best on all reported metrics** |

There is no single composite score. The selected model leads every reported task-compatible metric.

## Preserved MiniLM result

| Model | Evaluation set | Accuracy | Macro-F1 | Weighted-F1 | ECE (10 bins) |
|---|---:|---:|---:|---:|---:|
| MiniLM-L6 Intent INT8 ONNX | 1,000 | 1.0000 | 1.0000 | 1.0000 | 0.1813 |

The perfect classification result applies only to the frozen, balanced project stress set. ECE shows that probability calibration is not perfect even though the predicted labels are correct.
