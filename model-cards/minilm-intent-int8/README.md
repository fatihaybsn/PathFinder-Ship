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

That historical split contains duplicates and cross-split overlaps, so it is retained only as historical evidence.

The preserved focused evaluation uses 1,000 newly authored, balanced project stress examples. The INT8 model achieved accuracy `1.0000`, macro-F1 `1.0000`, and weighted-F1 `1.0000`; ECE across ten bins was `0.1813`. The perfect label score applies only to this deterministic project suite, while ECE shows that probability calibration is not perfect.
