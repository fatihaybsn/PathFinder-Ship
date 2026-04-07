import unittest

from schemas.pipeline import (
    ClientAction,
    DetectionResult,
    IntentResult,
    RetrievedChunk,
    RetrievalResult,
    RunResult,
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

    def test_client_action_is_separate_from_detection_result(self):
        action = ClientAction(action="open_camera", requires_user_permission=True)
        detection = DetectionResult(status="not_run")

        serialized_action = to_serializable_dict(action)
        serialized_detection = to_serializable_dict(detection)

        self.assertEqual(serialized_action["action"], "open_camera")
        self.assertTrue(serialized_action["requires_user_permission"])
        self.assertEqual(serialized_detection["status"], "not_run")
        self.assertEqual(serialized_detection["objects"], [])


if __name__ == "__main__":
    unittest.main()
