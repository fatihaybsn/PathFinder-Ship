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

Historical evaluation reported eval loss 0.9585, RAG EM 0.742, RAG F1 0.8609, and RAG ROUGE-L 0.8573. These historical values came from the experiment's saved validation data and remain separate from decontaminated Benchmark v1 results.
