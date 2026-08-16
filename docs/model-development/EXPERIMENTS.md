# Model Development Record

PathFinderShip's MiniLM and Flan-T5 models are pretrained checkpoints fine-tuned on project data; they were not trained from scratch. Original folder aliases are retained for traceability, while publication names describe the actual training change.

## Retraining Evaluation v2

| Experiment ID | Original folder | Deliberate change | Chat token-F1 | RAG token-F1 | RAG EM | Interpretation |
|---|---|---|---:|---:|---:|---|
| `flan_large_lora_qv_failed` | `kötü` | LoRA q/v only, r=16, alpha=32 | 0.3942 | 0.8127 | 0.7063 | Rejected because the shared Chat result remained weak. |
| `flan_large_lora_rag2_step1320` | `2x rag/checkpoint_1320` | RAG task-loss weight 2.0; Chat 1.0 | 0.3778 | 0.5864 | 0.4625 | Task weighting did not improve the shared evaluation. |
| `flan_large_lora_chat12_step1485` | `1.2x chat/1485` | Chat task-loss weight 1.2; RAG 1.0 | 0.4695 | 0.6719 | 0.5563 | Useful intermediate checkpoint. |
| `flan_large_lora_chat12_step1980` | `1.2x chat/1980` | Same task weighting, more optimization steps | 0.4684 | 0.6587 | 0.5438 | Slight regression versus step 1485. |
| `flan_large_lora_first_try` | `My Class/First Try` | First custom trainer/loss iteration | 0.4937 | 0.8421 | 0.7438 | Strong runner-up on both tasks. |
| **`flan_large_lora_second_try`** | **`My Class/Second Try`** | **Chat weight 1.7, task smoothing, partial R-Drop** | **0.5216** | **0.8894** | **0.7938** | **Selected final; best on every reported metric.** |

`1.2x` and `2x` refer to task-loss multipliers, not model size or repeated copies of the dataset. No composite score is constructed across tasks.

## Preserved MiniLM evaluation

The MiniLM-L6 intent classifier was not changed by the Flan-T5 retraining cycle. Its verified INT8 ONNX result remains accuracy `1.0000`, macro-F1 `1.0000`, weighted-F1 `1.0000`, and ECE `0.1813` on 1,000 balanced project stress examples.

## Historical training evidence

Executed notebooks and Trainer states retain older training-loss evidence, including the rejected q/v run, the task-weighting milestones, and early full-fine-tuning experiments. Those loss values describe their own historical splits and are not compared numerically with Retraining Evaluation v2.

Publication assets and claims boundaries are documented in [`results/retraining-v2/RESULT_CARD.md`](results/retraining-v2/RESULT_CARD.md).
