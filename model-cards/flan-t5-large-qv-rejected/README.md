---
base_model: google/flan-t5-large
library_name: peft
pipeline_tag: text2text-generation
license: apache-2.0
tags: [peft, lora, flan-t5, rag, chat, rejected-experiment]
---

# PathFinder Flan-T5 Large — q/v-only Rejected Experiment

Rejected experiment preserved from the original `kötü` folder. LoRA targeted q/v only with r=16 and alpha=32.

Retraining Evaluation v2:

| Chat token-F1 | RAG token-F1 | RAG exact match |
|---:|---:|---:|
| 0.3942 | 0.8127 | 0.7063 |

Although RAG overlap was strong, Chat performance remained weak. It was rejected because PathFinderShip requires a balanced Chat+RAG model, not because every task metric was poor.
