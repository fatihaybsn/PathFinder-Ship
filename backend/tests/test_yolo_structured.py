import sys
import types
import unittest
from types import MethodType
from unittest.mock import MagicMock

import numpy as np

if "onnxruntime" not in sys.modules:
    sys.modules["onnxruntime"] = types.ModuleType("onnxruntime")

if not hasattr(sys.modules["onnxruntime"], "SessionOptions"):
    class _SessionOptions:
        graph_optimization_level = None

    class _GraphOptimizationLevel:
        ORT_ENABLE_ALL = "ORT_ENABLE_ALL"

    sys.modules["onnxruntime"].SessionOptions = _SessionOptions
    sys.modules["onnxruntime"].GraphOptimizationLevel = _GraphOptimizationLevel
    sys.modules["onnxruntime"].InferenceSession = MagicMock()

from schemas.pipeline import (
    DETECTION_STATUS_INVALID_IMAGE,
    DETECTION_STATUS_MODEL_ERROR,
    DETECTION_STATUS_SUCCESS,
)
from services.yolo import YOLOService


def make_service():
    service = YOLOService.__new__(YOLOService)
    service.model_name = "fake-yolo.onnx"
    service.imgsz = 640
    service.conf_thr = 0.25
    service.iou_thr = 0.45
    service.names = ["person", "bottle"]
    return service


class YOLOStructuredTests(unittest.TestCase):
    def test_detect_structured_preserves_detection_metadata(self):
        service = make_service()

        def fake_detect_from_bgr(self, image_bgr):
            return [[1, 2, 3, 4]], ["bottle"], [0.77], [39]

        service.detect_from_bgr = MethodType(fake_detect_from_bgr, service)

        result = service.detect_structured(np.zeros((10, 20, 3), dtype=np.uint8), image_source="unit")

        self.assertEqual(result.status, DETECTION_STATUS_SUCCESS)
        self.assertEqual(result.image_source, "unit")
        self.assertEqual(result.model_name, "fake-yolo.onnx")
        self.assertIsNotNone(result.latency_ms)
        self.assertEqual(result.metadata["image_width"], 20)
        self.assertEqual(result.metadata["image_height"], 10)
        self.assertEqual(result.metadata["model_input_size"], 640)
        self.assertEqual(result.metadata["confidence_threshold"], 0.25)
        self.assertEqual(result.metadata["iou_threshold"], 0.45)
        self.assertEqual(result.metadata["bbox_format"], "xyxy")
        self.assertEqual(result.objects[0].label, "bottle")
        self.assertEqual(result.objects[0].confidence, 0.77)
        self.assertEqual(result.objects[0].bbox, [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result.objects[0].metadata["class_id"], 39)
        self.assertEqual(result.objects[0].metadata["raw_score"], 0.77)

    def test_detect_structured_rejects_invalid_image(self):
        service = make_service()

        result = service.detect_structured(None, image_source="unit")

        self.assertEqual(result.status, DETECTION_STATUS_INVALID_IMAGE)
        self.assertEqual(result.error, "image_missing")
        self.assertEqual(result.image_source, "unit")

    def test_detect_structured_returns_model_error_without_crashing(self):
        service = make_service()

        def fake_detect_from_bgr(self, image_bgr):
            raise RuntimeError("boom")

        service.detect_from_bgr = MethodType(fake_detect_from_bgr, service)

        result = service.detect_structured(np.zeros((10, 20, 3), dtype=np.uint8), image_source="unit")

        self.assertEqual(result.status, DETECTION_STATUS_MODEL_ERROR)
        self.assertEqual(result.error, "yolo_inference_failed")
        self.assertEqual(result.metadata["image_width"], 20)
        self.assertEqual(result.metadata["image_height"], 10)


if __name__ == "__main__":
    unittest.main()
