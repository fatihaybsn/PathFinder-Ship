import unittest

from schemas.pipeline import (
    ClientAction,
    DETECTION_STATUS_NO_OBJECTS,
    DETECTION_STATUS_SUCCESS,
    DetectionResult,
    GenerationResult,
    IndexingResult,
    IntentResult,
    RetrievedChunk,
    RetrievalResult,
    RunResult,
    detection_result_from_legacy,
    generation_result_from_text,
    to_serializable_dict,
)


class PipelineSchemaTests(unittest.TestCase):
    def test_intent_result_serializes(self):
        result = IntentResult(
            label="chat",
            confidence=0.91,
            threshold=0.6,
            is_confident=True,
            raw_scores={"chat": 0.91},
        )

        serialized = to_serializable_dict(result)

        self.assertEqual(serialized["label"], "chat")
        self.assertEqual(serialized["raw_scores"]["chat"], 0.91)
        self.assertTrue(serialized["is_confident"])

    def test_retrieval_result_serializes_nested_chunks(self):
        result = RetrievalResult(
            query="Where is the nearest exit?",
            chunks=[
                RetrievedChunk(
                    text="Exit guidance text",
                    source="local:safety.pdf",
                    score=0.82,
                    rank=1,
                    retrieval_type="local",
                )
            ],
            top_k=1,
            best_score=0.82,
            threshold=0.4,
            used_context=True,
            retrieval_mode="hybrid",
        )

        serialized = to_serializable_dict(result)

        self.assertEqual(serialized["chunks"][0]["text"], "Exit guidance text")
        self.assertEqual(serialized["chunks"][0]["source"], "local:safety.pdf")
        self.assertTrue(serialized["used_context"])

    def test_run_result_serializes_with_empty_optional_sections(self):
        result = RunResult(input_text="hello", status="ok")

        serialized = to_serializable_dict(result)

        self.assertEqual(serialized["input_text"], "hello")
        self.assertEqual(serialized["status"], "ok")
        self.assertIsNone(serialized["intent"])
        self.assertEqual(serialized["errors"], [])
        self.assertEqual(serialized["warnings"], [])

    def test_generation_result_serializes_runtime_metadata(self):
        result = GenerationResult(
            text="hello",
            model_name="local-t5-onnx",
            runtime="onnxruntime",
            device="cpu",
            prompt_type="chat",
            input_chars=5,
            output_chars=5,
            input_tokens=3,
            output_tokens=2,
            max_new_tokens=256,
            input_truncated=False,
            output_truncated=False,
            latency_ms=12,
        )

        serialized = to_serializable_dict(result)

        self.assertEqual(serialized["runtime"], "onnxruntime")
        self.assertEqual(serialized["device"], "cpu")
        self.assertEqual(serialized["input_tokens"], 3)
        self.assertFalse(serialized["input_truncated"])

    def test_generation_result_helper_accepts_optional_lengths(self):
        result = generation_result_from_text(
            "answer",
            model_name="local-t5-onnx",
            runtime="onnxruntime",
            device="cpu",
            prompt_type="rag_answer",
            input_chars=10,
            input_tokens=4,
            output_tokens=2,
            max_new_tokens=64,
            input_truncated=False,
            output_truncated=True,
        )

        self.assertEqual(result.output_chars, 6)
        self.assertEqual(result.output_tokens, 2)
        self.assertTrue(result.output_truncated)

    def test_client_action_is_separate_from_detection_result(self):
        action = ClientAction(action="open_camera", requires_user_permission=True)
        detection = DetectionResult(status="not_run")

        serialized_action = to_serializable_dict(action)
        serialized_detection = to_serializable_dict(detection)

        self.assertEqual(serialized_action["action"], "open_camera")
        self.assertTrue(serialized_action["requires_user_permission"])
        self.assertEqual(serialized_detection["status"], "not_run")
        self.assertEqual(serialized_detection["objects"], [])

    def test_detection_result_from_legacy_uses_standard_statuses_and_metadata(self):
        result = detection_result_from_legacy(
            labels=["bottle"],
            boxes=[[1, 2, 3, 4]],
            scores=[0.77],
            cls_ids=[39],
            model_name="fake-yolo",
        )

        self.assertEqual(result.status, DETECTION_STATUS_SUCCESS)
        self.assertEqual(result.objects[0].label, "bottle")
        self.assertEqual(result.objects[0].confidence, 0.77)
        self.assertEqual(result.objects[0].bbox, [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result.objects[0].metadata["class_id"], 39)
        self.assertEqual(result.objects[0].metadata["raw_score"], 0.77)

    def test_detection_result_from_legacy_marks_no_objects(self):
        result = detection_result_from_legacy(labels=[], boxes=[], scores=[], cls_ids=[])

        self.assertEqual(result.status, DETECTION_STATUS_NO_OBJECTS)
        self.assertEqual(result.objects, [])

    def test_indexing_result_serializes_upload_status(self):
        result = IndexingResult(
            filename="manual.txt",
            document_id="doc_abc",
            saved_path="data/rag/corpus/manual__abc.txt",
            indexed=True,
            indexed_chunk_count=2,
            warnings=[],
        )

        serialized = to_serializable_dict(result)

        self.assertEqual(serialized["filename"], "manual.txt")
        self.assertEqual(serialized["document_id"], "doc_abc")
        self.assertTrue(serialized["indexed"])
        self.assertEqual(serialized["indexed_chunk_count"], 2)


if __name__ == "__main__":
    unittest.main()
