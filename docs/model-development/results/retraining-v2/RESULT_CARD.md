# PathFinderShip Retraining Evaluation v2

## Outcome

`My Class — Second Try` is the selected production candidate. It achieved the highest reported score on Chat token-F1 (`0.5216`), RAG token-F1 (`0.8894`), and RAG exact match (`0.7938`) among six retrained Flan-T5 Large LoRA variants.

![Retraining v2 dashboard](figures/flan_retraining_v2_dashboard.png)

## What changed across experiments

- `kötü`: limited LoRA targets to q/v. It retained strong RAG overlap but underperformed on Chat.
- `2x RAG`: assigned RAG examples a task-loss weight of `2.0`. This is a loss multiplier, not data duplication; it did not improve the shared evaluation.
- `1.2x Chat`: assigned Chat examples a task-loss weight of `1.2`. Step 1485 slightly outperformed step 1980, showing that more optimization steps did not improve generalization.
- `First Try`: introduced the first custom trainer/loss iteration and became the runner-up.
- `Second Try`: used Chat weight `1.7`, task-specific smoothing, and partial R-Drop; it led all reported metrics.

## Evidence files

- Machine-readable Flan metrics: [`metrics/flan_retraining_results.json`](metrics/flan_retraining_results.json)
- Editable CSV: [`metrics/flan_retraining_results.csv`](metrics/flan_retraining_results.csv)
- Auditable workbook: [`metrics/flan_retraining_results.xlsx`](metrics/flan_retraining_results.xlsx)
- Publication table: [`tables/model_comparison.md`](tables/model_comparison.md)
- Updated supplied figures: [`figures/source-updated/`](figures/source-updated/)
- SHA-256 manifest: [`artifact_manifest.json`](artifact_manifest.json)

MiniLM was not retrained in this update. Its verified 1,000-example result is preserved in [`metrics/minilm_intent_results.json`](metrics/minilm_intent_results.json) and its confusion matrix remains available under `figures/source-updated/`.

## Claims boundary

- These are fine-tuned pretrained checkpoints, not models trained from scratch.
- Scores apply to the named project evaluation suites; they are not universal accuracy percentages.
- Automatic reference metrics do not replace human review.
- Retraining v2 does not include new bootstrap confidence intervals.
- ONNX should be published as the current retrained model only when its files, hash, and parity evidence correspond to the new Second Try weights.
