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

The Trainer state reports best eval loss 0.9845. Historical pre/post measurements recorded loss improvements of approximately 14.1% overall, 14.4% on Chat, and 20.5% on RAG. The historical prompt builder has known duplicated-tag cases, which are documented rather than hidden. Benchmark v1 uses a normalized prompt builder and separate result table.
