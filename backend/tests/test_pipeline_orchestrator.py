import unittest
import warnings

from fastapi.testclient import TestClient

from schemas.pipeline import RunResult, to_serializable_dict
from services.pipeline_orchestrator import PipelineOrchestrator


class FakeNLU:
    def __init__(self, label="chat", score=0.95, should_raise=False):
        self.label = label
        self.score = score
        self.should_raise = should_raise
        self.last_error = None

    def predict(self, text):
        if self.should_raise:
            raise RuntimeError("intent failed")
        return self.label, self.score


class FakeT5:
    def __init__(self, should_raise_chat=False):
        self.should_raise_chat = should_raise_chat

    def chat(self, text):
        if self.should_raise_chat:
            raise RuntimeError("chat failed")
        return f"chat:{text}"

    def answer(self, question, context):
        return f"rag:{question}:{len(context)}"

    def answer_model_only_with_instruction(self, question, instruction=None):
        return f"fallback:{question}"


class FakeRAG:
    top_k = 2
    thr = 0.4

    def retrieve(self, question, use_internet=False, web_only=False):
        return ["context text"], 0.82, ["local:test.txt"]


class FakeYOLO:
    def __init__(self):
        self.detect_called = False

    def detect_from_bgr(self, image_bgr):
        self.detect_called = True
        return [[1, 2, 3, 4]], ["bottle"], [0.77], [39]


class PipelineOrchestratorTests(unittest.TestCase):
    def make_pipeline(self, nlu=None, t5=None, rag=None, yolo=None):
        return PipelineOrchestrator(
            {"CLS_ROUTE_THRESHOLD": 0.6},
            nlu or FakeNLU(),
            t5 or FakeT5(),
            rag or FakeRAG(),
            yolo or FakeYOLO(),
        )

    def test_chat_route_returns_generation_result(self):
        pipeline = self.make_pipeline(nlu=FakeNLU("chat", 0.96))

        result = pipeline.run("hello")

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.final_answer, "chat:hello")
        self.assertEqual(result.generation.prompt_type, "chat")
        self.assertIsNone(result.retrieval)

    def test_open_camera_returns_client_action_without_detection(self):
        yolo = FakeYOLO()
        pipeline = self.make_pipeline(nlu=FakeNLU("open_camera", 0.97), yolo=yolo)

        result = pipeline.run("open the camera")

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.route.route, "camera_action")
        self.assertEqual(result.client_action.action, "open_camera")
        self.assertTrue(result.client_action.requires_user_permission)
        self.assertIsNone(result.detection)
        self.assertFalse(yolo.detect_called)

    def test_object_detect_without_image_is_degraded_not_crashed(self):
        pipeline = self.make_pipeline(nlu=FakeNLU("object_detect", 0.98))

        result = pipeline.run("detect objects")

        self.assertEqual(result.status, "degraded")
        self.assertEqual(result.route.route, "detect")
        self.assertEqual(result.detection.status, "not_run")
        self.assertEqual(result.client_action.action, "capture_photo")
        self.assertTrue(result.warnings)

    def test_service_exception_returns_serializable_failed_result(self):
        pipeline = self.make_pipeline(nlu=FakeNLU("chat", 0.97), t5=FakeT5(should_raise_chat=True))

        with self.assertLogs("services.pipeline_orchestrator", level="ERROR"):
            result = pipeline.run("hello")
        serialized = to_serializable_dict(result)

        self.assertEqual(result.status, "failed")
        self.assertIn("chat service failed", result.errors)
        self.assertEqual(serialized["status"], "failed")

    def test_intent_exception_falls_back_to_chat(self):
        pipeline = self.make_pipeline(nlu=FakeNLU(should_raise=True))

        with self.assertLogs("services.pipeline_orchestrator", level="ERROR"):
            result = pipeline.run("hello")

        self.assertEqual(result.status, "degraded")
        self.assertEqual(result.route.route, "chat")
        self.assertTrue(result.route.fallback_used)
        self.assertEqual(result.final_answer, "chat:hello")
        self.assertTrue(result.warnings)


class RunEndpointTests(unittest.TestCase):
    def test_run_endpoint_returns_pipeline_result(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            import web.app as web_app

        class FakePipeline:
            def run(self, input_text, metadata=None, image_bgr=None):
                return RunResult(
                    input_text=input_text,
                    final_answer="ok",
                    status="completed",
                    metadata={"use_internet": bool((metadata or {}).get("use_internet"))},
                )

        previous_pipeline = web_app.PIPELINE
        web_app.PIPELINE = FakePipeline()
        try:
            client = TestClient(web_app.app)
            response = client.post("/api/run", json={"message": "hello", "metadata": {"use_internet": True}})
        finally:
            web_app.PIPELINE = previous_pipeline

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["input_text"], "hello")
        self.assertEqual(body["final_answer"], "ok")
        self.assertEqual(body["metadata"]["use_internet"], True)


if __name__ == "__main__":
    unittest.main()
