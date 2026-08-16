# Evidence Provenance

## Current headline evidence

- Result set: `retraining_evaluation_v2`
- Flan-T5 values: supplied from the completed retraining evaluation summary and reflected in the updated figures placed in `lightning_results_incoming/pathfinder_focused_results_20260816T102211Z/figures`.
- Protocol identity: `chat_reference_v1` (300 examples) and `rag_project_v1` (160 examples), consistent across the six reported LoRA variants.
- Selected checkpoint: `My Class/Second Try`.

## Preserved evidence

- MiniLM values and confusion matrix are retained unchanged from the earlier verified 1,000-example run.
- Historical notebook losses remain historical training evidence and are not relabeled as retraining-v2 evaluation scores.

## Excluded from the v2 headline

- Old Flan prediction JSONL files and old aggregate metric JSON files are not copied into this publication directory because they correspond to superseded weights/results.
- Old ONNX parity numbers are not presented as v2 parity unless the exported ONNX files are confirmed to correspond to the retrained Second Try weights.
- No confidence interval was inferred from aggregate values.

This separation prevents a new summary table from being paired with incompatible old raw predictions or hashes.
