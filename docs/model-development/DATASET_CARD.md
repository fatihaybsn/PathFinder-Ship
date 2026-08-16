# Benchmark v1 Dataset Card

## Purpose

Benchmark v1 is an evaluation-only collection for PathFinderShip's intent, Chat+Command, Chat+RAG, ONNX parity, and pretrained vision-integration experiments. It must not be used for further fine-tuning after results have been published.

## Components

- `intent_v1`: 1,000 deterministic project-authored stress examples, balanced across the five configured intent labels.
- `command_v1`: 600 deterministic project-authored examples covering the valid four-bit command combinations and chat negatives.
- `chat_ifeval_v1`: the official Google IFEval prompts (Apache-2.0), evaluated with its strict and loose rule checkers.
- `chat_reference_v1`: 300 deterministic project-authored single-turn prompts with references and diagnostic tags.
- `rag_v1`: 440 examples sampled with seed 42 from the test splits of eleven RAGBench subsets (HotpotQA excluded) plus 160 project-authored examples split evenly between answerable and unanswerable context.
- `vision_coco_v1`: COCO 2017 validation images and annotations, used only to verify pretrained integration.

## Decontamination

Normalized exact hashes and word-ngram similarity signatures are built from all known historical training and validation inputs. Public and project-authored candidates are rejected when they exactly match or exceed the configured near-duplicate threshold. Every retained sample receives a stable ID and source field.

## Limitations

- Project-authored examples are deterministic stress tests, not a representative sample of all user traffic.
- Reference-overlap metrics do not fully measure conversational quality.
- The lexical RAG support score is a transparent proxy, not a semantic hallucination judge.
- IFEval measures verifiable instruction following and does not replace human conversation review.
- COCO results validate pretrained inference integration; they do not demonstrate custom object-detector training.
