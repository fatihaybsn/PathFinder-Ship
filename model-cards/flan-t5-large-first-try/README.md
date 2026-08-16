---
base_model: google/flan-t5-large
library_name: peft
pipeline_tag: text2text-generation
license: apache-2.0
tags: [peft, lora, flan-t5, rag, chat, experimental]
---

# PathFinder Flan-T5 Large — My Class First Try

First custom trainer/loss iteration in the PathFinderShip Chat+RAG development chain.

Retraining Evaluation v2:

| Chat token-F1 | RAG token-F1 | RAG exact match |
|---:|---:|---:|
| 0.4937 | 0.8421 | 0.7438 |

First Try is the runner-up and remains useful evidence that the custom trainer direction improved both tasks before the final Second Try configuration.
