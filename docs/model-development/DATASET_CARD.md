# PathFinderShip Model Evidence Dataset Card

## Purpose

The publication evaluation compares project-fine-tuned models on task-compatible, frozen project suites. The evaluation data is not used to claim universal model accuracy and should not be reused for further tuning after publication.

## Headline components

- `intent_v1`: 1,000 deterministic project-authored stress examples, balanced across five configured MiniLM intent labels.
- `chat_reference_v1`: 300 deterministic project-authored Chat prompts with reference responses.
- `rag_project_v1`: 160 project-authored grounded RAG examples, including answerable and unanswerable contexts.

Retraining Evaluation v2 compares six updated Flan-T5 Large LoRA adapters on the same Chat and RAG suites. MiniLM retains its previously verified intent result.

## Metric definitions

- Chat token-F1: token overlap against the reference Chat response.
- RAG token-F1: token overlap against the grounded reference answer.
- RAG exact match: normalized exact match against the grounded reference answer.
- Intent accuracy/macro-F1/weighted-F1: five-class classification quality.
- ECE: confidence calibration error across ten bins.

## Decontamination record

Known historical training and validation inputs were inventoried with normalized exact and near-duplicate fingerprints. The full audit remains in [`evidence/TRAINING_DATA_AUDIT.md`](evidence/TRAINING_DATA_AUDIT.md).

## Limitations

- Project-authored deterministic stress tests do not represent all production traffic.
- Reference-overlap metrics can penalize semantically valid alternative wording and do not replace human review.
- The MiniLM suite is balanced and template-driven; a perfect label score must be interpreted within that scope.
- Retraining v2 reports aggregate values without new bootstrap confidence intervals.
- YOLO is outside this model-training evidence set because the project uses pretrained detector integration rather than a custom-trained YOLO checkpoint.
