---
base_model: nreimers/MiniLM-L6-H384-uncased
library_name: onnxruntime
pipeline_tag: text-classification
license: apache-2.0
tags:
- minilm
- onnx
- int8
- intent-classification
---

# PathFinder MiniLM-L6 Intent Classifier — INT8 ONNX

Five-class PathFinderShip intent classifier fine-tuned from `nreimers/MiniLM-L6-H384-uncased` and exported to dynamic INT8 ONNX.

Labels:

1. `open_camera`
2. `close_camera`
3. `take_photo`
4. `object_detect`
5. `chat`

Historical configuration: maximum length 96, learning rate 2e-5, four epochs, train batch 32, eval batch 64, CPU training. The executed notebook reported accuracy and macro-F1 of 1.000 on 600 historical test examples.

That historical split contains duplicates and cross-split overlaps. It is retained as historical evidence, not presented as the clean generalization result. Benchmark v1 evaluates 1,000 newly authored balanced stress examples and will be inserted after result validation.
