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

# PathFinder Flan-T5 Large — 1.2x Chat Step 1980

Milestone LoRA adapter from the experiment that weighted Chat examples by 1.2 and RAG examples by 1.0.

- LoRA r=16, alpha=32, dropout=0.05
- Targets: q, k, v, o, wi_0, wi_1, wo
- Approximately 18.28M trainable parameters (2.28%)
- Three planned epochs; checkpoint step 1980 corresponds to epoch 3
- Learning rate 2e-4 with cosine schedule

`1.2x Chat` means that Chat examples received a task-loss weight of `1.2` while RAG remained `1.0`; it is not a model-size or dataset multiplier.

Retraining Evaluation v2 reports Chat token-F1 `0.4684`, RAG token-F1 `0.6587`, and RAG exact match `0.5438`. The earlier step-1485 checkpoint scored `0.4695`, `0.6719`, and `0.5563`, respectively, so additional optimization to step 1980 produced a small regression on the shared suites.
