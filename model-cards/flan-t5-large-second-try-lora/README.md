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

# PathFinder Flan-T5 Large — Second Try LoRA

This is the final PathFinderShip Chat+RAG LoRA adapter. It fine-tunes `google/flan-t5-large`; it is not a model trained from scratch.

## Training configuration

- LoRA: r=16, alpha=32, dropout=0.05
- Targets: q, k, v, o, wi_0, wi_1, wo
- Trainable parameters: approximately 18.28M (2.28% of the base model)
- Data: 100,000 Chat+RAG records, deterministic 95/5 split with seed 42
- Epochs: 1
- Learning rate: 1e-4; warmup ratio 0.06
- Task weighting: Chat 1.7, RAG 1.0
- Chat label smoothing: 0.02; RAG label smoothing: 0.00
- Partial R-Drop: probability 0.15, lambda 0.25

## Evidence

Historical executed-notebook results include eval loss 1.0937 and a quick RAG subset result of EM 0.805/F1 0.9148. The quick historical chat run used a 1.2-second generation cap and was truncated, so that chat score is not a headline metric.

Benchmark v1 results must be inserted here only after the Lightning result ZIP passes manifest validation. See the GitHub experiment record for raw predictions, hashes, environment, and limitations.

## Use

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

base_id = "google/flan-t5-large"
adapter_id = "fatihaybsn/pathfinder-flan-t5-large-second-try-lora"
tokenizer = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForSeq2SeqLM.from_pretrained(base_id)
model = PeftModel.from_pretrained(base, adapter_id)
```

The model is designed for the prompt templates documented in the PathFinderShip repository. Automatic reference metrics do not replace human evaluation.
