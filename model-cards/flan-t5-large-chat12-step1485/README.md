---
base_model: google/flan-t5-large
library_name: peft
pipeline_tag: text2text-generation
license: apache-2.0
tags: [peft, lora, flan-t5, rag, chat, experimental]
---

# PathFinder Flan-T5 Large — Chat Loss Weight 1.2, Step 1485

Intermediate LoRA checkpoint from the experiment that weighted Chat loss by `1.2` and RAG loss by `1.0`. The original folder alias `1.2x chat/1485` refers to task weighting, not model scaling or dataset duplication.

Retraining Evaluation v2:

| Chat token-F1 | RAG token-F1 | RAG exact match |
|---:|---:|---:|
| 0.4695 | 0.6719 | 0.5563 |

This checkpoint slightly outperformed the step-1980 checkpoint on all three reported metrics, demonstrating that additional optimization did not improve generalization in this run.
