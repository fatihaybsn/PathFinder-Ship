import unittest

from schemas.pipeline import (
    DETECTION_STATUS_FAILED,
    GenerationResult,
    IntentResult,
    RetrievedChunk,
    RetrievalResult,
    to_serializable_dict,
)
from services.pipeline_orchestrator import PipelineOrchestrator


class StaticNLU:
    def __init__(self, label="chat", confidence=0.95, error=None):
        self.label = label
        self.confidence = confidence
        self.error = error
        self.last_error = error

    def classify_intent(self, text, threshold=None):
        is_confident = None
        if threshold is not None:
            is_confident = self.confidence >= threshold
        return IntentResult(
            label=self.label,
            confidence=self.confidence,
            threshold=threshold,
            is_confident=is_confident,
            raw_scores={self.label: self.confidence},
            latency_ms=1,
            error=self.error,
        )


class StructuredT5:
    max_new_chat = 256
    max_new_rag = 64

    def chat_structured(self, text):
        answer = f"chat answer: {text}"
        return GenerationResult(
            text=answer,
            model_name="fake-t5",
            runtime="onnxruntime",
            device="cpu",
            prompt_type="chat",
            input_chars=len(text),
            output_chars=len(answer),
            max_new_tokens=self.max_new_chat,
            latency_ms=1,
        )

    def answer_structured(self, question, context):
        answer = f"rag answer: {question}"
        return GenerationResult(
            text=answer,
            model_name="fake-t5",
            runtime="onnxruntime",
            device="cpu",
            prompt_type="rag_answer",
            input_chars=len(question),
            output_chars=len(answer),
            max_new_tokens=self.max_new_rag,
            latency_ms=1,
        )

    def answer_model_only_with_instruction_structured(self, question, instruction=None):
        answer = "I don't know."
        return GenerationResult(
            text=answer,
            model_name="fake-t5",
            runtime="onnxruntime",
            device="cpu",
            prompt_type="model_only",
            input_chars=len(question),
            output_chars=len(answer),
            max_new_tokens=self.max_new_chat,
            latency_ms=1,
        )

    def narrate_detection_structured(self, objects):
        answer = f"detected: {objects}"
        return GenerationResult(
            text=answer,
            model_name="fake-t5",
            runtime="onnxruntime",
            device="cpu",
            prompt_type="detection_narration",
            input_chars=len(str(objects)),
            output_chars=len(answer),
            max_new_tokens=self.max_new_rag,
            latency_ms=1,
        )


class EmptyOutputT5(StructuredT5):
    def chat_structured(self, text):
        return GenerationResult(
            text="I couldn't generate a response right now.",
            model_name="fake-t5",
            runtime="onnxruntime",
            device="cpu",
            prompt_type="chat",
            input_chars=len(text),
            output_chars=40,
            max_new_tokens=self.max_new_chat,
            latency_ms=1,
            empty_output=True,
            fallback_used=True,
            fallback_reason="empty_generation",
        )


class StructuredRAG:
    top_k = 2
    thr = 0.4
    max_ctx_tokens = 512

    def __init__(self, chunks=None):
        self.chunks = chunks

    def retrieve_structured(self, question, use_internet=False, web_only=False):
        if self.chunks is None:
            chunks = [
                RetrievedChunk(
                    text="Uploaded document context.",
                    source="local:uploaded_manual.txt",
                    score=0.91,
                    rank=1,
                    retrieval_type="local_hybrid",
                    metadata={"document_id": "doc_test"},
                )
            ]
        else:
            chunks = self.chunks

        return RetrievalResult(
            query=question,
            chunks=chunks,
            top_k=self.top_k,
            best_score=chunks[0].score if chunks else 0.0,
            threshold=self.thr,
            used_context=bool(chunks),
            retrieval_mode="local_only" if chunks else "empty",
            fallback_used=not bool(chunks),
            fallback_reason=None if chunks else "empty_retrieval",
            latency_ms=1,
        )


class Prompt10ReadinessTests(unittest.TestCase):
    def make_pipeline(self, nlu=None, t5=None, rag=None, yolo=None):
        return PipelineOrchestrator(
            {"CLS_ROUTE_THRESHOLD": 0.6},
            nlu or StaticNLU(),
            t5 or StructuredT5(),
            rag or StructuredRAG(),
            yolo,
        )

    def test_normal_chat_runresult_contract(self):
        result = self.make_pipeline(nlu=StaticNLU("chat", 0.95)).run("hello")

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.route.route, "chat")
        self.assertEqual(result.intent.label, "chat")
        self.assertEqual(result.generation.prompt_type, "chat")
        self.assertIsNotNone(result.generation.latency_ms)
        self.assertTrue(result.final_answer)
        self.assertEqual(result.errors, [])
        self.assertIsInstance(result.warnings, list)
        self.assertIsNotNone(result.duration_ms)

    def test_successful_rag_runresult_contract_from_chat_intent(self):
        result = self.make_pipeline(nlu=StaticNLU("chat", 0.95)).run(
            "what is in the uploaded document?"
        )

        self.assertEqual(result.route.route, "rag")
        self.assertFalse(result.route.fallback_used)
        self.assertTrue(result.retrieval.used_context)
        self.assertEqual(result.retrieval.chunks[0].source, "local:uploaded_manual.txt")
        self.assertEqual(result.retrieval.chunks[0].score, 0.91)
        self.assertEqual(result.retrieval.chunks[0].rank, 1)
        self.assertEqual(result.generation.prompt_type, "rag_answer")
        self.assertIsNotNone(result.generation.latency_ms)
        self.assertTrue(result.final_answer)

    def test_empty_retrieval_returns_safe_fallback(self):
        result = self.make_pipeline(
            nlu=StaticNLU("chat", 0.95),
            rag=StructuredRAG(chunks=[]),
        ).run("what is in the uploaded document?")

        self.assertEqual(result.status, "degraded")
        self.assertEqual(result.route.route, "rag")
        self.assertEqual(result.retrieval.chunks, [])
        self.assertFalse(result.retrieval.used_context)
        self.assertTrue(result.retrieval.fallback_used)
        self.assertEqual(result.retrieval.fallback_reason, "empty_retrieval")
        self.assertTrue(result.generation.fallback_used)
        self.assertEqual(result.generation.fallback_reason, "no_retrieval_context")
        self.assertEqual(result.final_answer, "I don't know.")

    def test_low_confidence_document_question_uses_rag_fallback(self):
        result = self.make_pipeline(nlu=StaticNLU("chat", 0.2)).run(
            "according to the manual, what should passengers do?"
        )

        self.assertEqual(result.route.route, "rag")
        self.assertTrue(result.route.fallback_used)
        self.assertEqual(result.route.fallback_reason, "low_confidence_rag_heuristic")
        self.assertTrue(result.retrieval.used_context)
        self.assertEqual(result.status, "degraded")

    def test_camera_client_actions_do_not_run_detection(self):
        open_result = self.make_pipeline(nlu=StaticNLU("open_camera", 0.95)).run("open camera")
        close_result = self.make_pipeline(nlu=StaticNLU("close_camera", 0.95)).run("close camera")

        self.assertEqual(open_result.route.route, "camera_action")
        self.assertEqual(open_result.client_action.action, "open_camera")
        self.assertTrue(open_result.client_action.requires_user_permission)
        self.assertIsNone(open_result.detection)
        self.assertEqual(close_result.route.route, "camera_action")
        self.assertEqual(close_result.client_action.action, "close_camera")
        self.assertFalse(close_result.client_action.requires_user_permission)
        self.assertIsNone(close_result.detection)

    def test_detection_without_image_requests_capture_photo(self):
        result = self.make_pipeline(nlu=StaticNLU("object_detect", 0.95)).run("detect objects")

        self.assertEqual(result.status, "degraded")
        self.assertEqual(result.route.route, "detect")
        self.assertEqual(result.client_action.action, "capture_photo")
        self.assertEqual(result.detection.status, DETECTION_STATUS_FAILED)
        self.assertEqual(result.detection.error, "missing_image_for_detection")
        self.assertIn("missing_image_for_detection", result.warnings)

    def test_t5_empty_output_degrades_without_crashing(self):
        result = self.make_pipeline(
            nlu=StaticNLU("chat", 0.95),
            t5=EmptyOutputT5(),
        ).run("hello")

        self.assertEqual(result.status, "degraded")
        self.assertEqual(result.route.route, "chat")
        self.assertTrue(result.generation.empty_output)
        self.assertTrue(result.generation.fallback_used)
        self.assertEqual(result.generation.fallback_reason, "empty_generation")
        self.assertEqual(result.final_answer, "I couldn't generate a response right now.")

    def test_runresult_serialization_exposes_diagent_ready_fields(self):
        result = self.make_pipeline(nlu=StaticNLU("chat", 0.95)).run(
            "what is in the uploaded document?"
        )
        body = to_serializable_dict(result)

        for key in (
            "input_text",
            "status",
            "intent",
            "route",
            "retrieval",
            "generation",
            "detection",
            "client_action",
            "errors",
            "warnings",
            "metadata",
            "duration_ms",
            "final_answer",
        ):
            self.assertIn(key, body)

        self.assertIsInstance(body["errors"], list)
        self.assertIsInstance(body["warnings"], list)
        self.assertIsNotNone(body["duration_ms"])
        self.assertEqual(body["route"]["route"], "rag")
        self.assertEqual(body["generation"]["prompt_type"], "rag_answer")


if __name__ == "__main__":
    unittest.main()
