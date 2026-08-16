# Model Development Record

This page is generated from source notebooks, trainer states, artifact hashes, and the frozen experiment manifest. Until Benchmark v1 is executed, all numbers in the table below are historical and must be read with their listed limitations.

| Experiment | Deliberate change | Historical evidence | Interpretation |
|---|---|---|---|
| MiniLM-L6 INT8 | Five-class intent fine-tuning, 4 epochs, LR 2e-5 | test accuracy 1.000; macro-F1 1.000 on 600 examples | Strong internal result, but split duplicates mean it is not the publication headline until `intent_v1` is run. |
| Early Flan-T5 Base | Chat+four-bit command generation, early stopping | best validation loss 0.7815 at epoch 13; stopped at epoch 16 | Establishes the early multitask full-fine-tuning stage. |
| Large LoRA q/v | LoRA only on q/v, LR 1e-4, label smoothing 0.1 | best trainer eval loss 1.8437; separate full evaluation loss 32.1334 | Rejected experiment; configuration and conflicting evaluation remain visible. |
| Large LoRA 2x RAG | Seven LoRA target modules; RAG example weight 2.0 | best eval loss 0.9845; historical task-loss improvements of roughly 14–20% | Useful RAG-weighting milestone; old prompt construction contains known duplicated tags. |
| Large LoRA 1.2x Chat | Seven LoRA targets; Chat weight 1.2 | step 1980 eval loss 0.9585; historical RAG EM 0.742 and F1 0.8609 | Strong historical RAG result; will be rerun on decontaminated Benchmark v1. |
| My Class Second Try | Chat weight 1.7, task-specific smoothing, partial R-Drop | eval loss 1.0937; quick RAG EM 0.805/F1 0.9148 | Final candidate. The old chat run used a 1.2-second generation cap and was severely truncated, so its chat score is not a headline result. |

The final report will add Benchmark v1 scores, confidence intervals, load status, environment, artifact SHA-256, and exact Hugging Face revision to this record.
