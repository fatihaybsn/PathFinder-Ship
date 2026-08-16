---
base_model: google/flan-t5-large
library_name: optimum
pipeline_tag: text2text-generation
license: apache-2.0
tags:
- onnx
- int8
- flan-t5
- rag
- chat
---

# PathFinder Flan-T5 Large Second Try — INT8 ONNX

Quantized ONNX encoder, decoder, and decoder-with-past export of the PathFinderShip Second Try LoRA model after merging with `google/flan-t5-large`.

The export notebook completed successfully and recorded numerical tolerance warnings up to approximately 7.2e-5. Quality and generation parity against the source PyTorch+LoRA model are therefore reported explicitly instead of claiming bit-exact equivalence.

Benchmark v1 parity, CPU latency, memory, and metric deltas will be added after the Lightning result bundle is validated. The repository retains raw paired predictions.

Expected files:

- `encoder_model_int8.onnx`
- `decoder_model_int8.onnx`
- `decoder_with_past_model_int8.onnx`
- tokenizer and generation configuration files

Use Optimum ONNX Runtime `ORTModelForSeq2SeqLM` with `CPUExecutionProvider` for the published deployment path.
