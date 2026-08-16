---
base_model: google/flan-t5-large
library_name: peft
pipeline_tag: text2text-generation
license: apache-2.0
tags:
- peft
- lora
- flan-t5
- rag
- chat
---

# PathFinder Flan-T5 Large — 2x RAG Step 1320

Milestone LoRA adapter from the experiment that weighted RAG examples by 2.0 and Chat examples by 1.0.

- LoRA r=16, alpha=32, dropout=0.05
- Targets: q, k, v, o, wi_0, wi_1, wo
- Approximately 18.28M trainable parameters (2.28%)
- Two epochs, learning rate 2e-4
- Historical runtime: 7,024.8 seconds on the recorded L40S workflow

`2x RAG` means that RAG examples received a task-loss weight of `2.0` while Chat remained `1.0`; it is not dataset duplication or model scaling.

Retraining Evaluation v2 reports Chat token-F1 `0.3778`, RAG token-F1 `0.5864`, and RAG exact match `0.4625`. The heavier RAG loss weight did not outperform the balanced custom trainer variants on the shared evaluation suites, so this adapter is published as an experimental milestone rather than the selected final model.
