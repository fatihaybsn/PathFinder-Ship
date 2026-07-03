import sys
import types
import unittest
import warnings
from unittest.mock import MagicMock

for _mod_name in (
    "tokenizers",
    "onnxruntime",
    "transformers",
    "sentence_transformers",
    "chromadb",
    "chromadb.utils",
    "chromadb.utils.embedding_functions",
    "bs4",
    "ddgs",
    "docx",
    "fitz",
):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = types.ModuleType(_mod_name)

if not hasattr(sys.modules["tokenizers"], "Tokenizer"):
    sys.modules["tokenizers"].Tokenizer = MagicMock()

if not hasattr(sys.modules["onnxruntime"], "SessionOptions"):
    class _SessionOptions:
        graph_optimization_level = None

    class _GraphOptimizationLevel:
        ORT_ENABLE_ALL = "ORT_ENABLE_ALL"

    sys.modules["onnxruntime"].SessionOptions = _SessionOptions
    sys.modules["onnxruntime"].GraphOptimizationLevel = _GraphOptimizationLevel
    sys.modules["onnxruntime"].InferenceSession = MagicMock()

if not hasattr(sys.modules["transformers"], "AutoTokenizer"):
    _auto_tok = MagicMock()
    _auto_tok.from_pretrained = MagicMock(return_value=MagicMock())
    sys.modules["transformers"].AutoTokenizer = _auto_tok

if not hasattr(sys.modules["sentence_transformers"], "SentenceTransformer"):
    sys.modules["sentence_transformers"].SentenceTransformer = MagicMock()

if not hasattr(sys.modules["chromadb"], "PersistentClient"):
    sys.modules["chromadb"].PersistentClient = MagicMock()

if not hasattr(sys.modules["bs4"], "BeautifulSoup"):
    sys.modules["bs4"].BeautifulSoup = MagicMock()

if not hasattr(sys.modules["ddgs"], "DDGS"):
    sys.modules["ddgs"].DDGS = MagicMock()

from fastapi.testclient import TestClient

from schemas.pipeline import (
    ClientAction,
    DETECTION_STATUS_FAILED,
    DETECTION_STATUS_SUCCESS,
    DetectionObject,
    DetectionResult,
    GenerationResult,
    IntentResult,
    RetrievedChunk,
    RetrievalResult,
    RouteDecision,
    RunResult,
    to_serializable_dict,
)
from services.observability.diagent_config import DiagentConfig
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


class FakeStructuredT5(FakeT5):
    max_new_chat = 256
    max_new_rag = 64

    def chat_structured(self, text):
        answer = f"structured-chat:{text}"
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
        answer = f"structured-rag:{question}:{len(context)}"
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
        answer = f"structured-fallback:{question}"
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
        answer = f"structured-detection:{objects}"
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


class FakeRAG:
    top_k = 2
    thr = 0.4
    max_ctx_tokens = 512

    def retrieve(self, question, use_internet=False, web_only=False):
        return ["context text"], 0.82, ["local:test.txt"]

    def retrieve_structured(self, question, use_internet=False, web_only=False):
        return RetrievalResult(
            query=question,
            chunks=[
                RetrievedChunk(
                    text="context text",
                    source="local:test.txt",
                    score=0.82,
                    rank=1,
                    retrieval_type="local_hybrid",
                    metadata={"file_name": "test.txt"},
                ),
            ],
            top_k=self.top_k,
            best_score=0.82,
            threshold=self.thr,
            used_context=True,
            retrieval_mode="local_only",
            latency_ms=5,
        )


class FakeYOLO:
    def __init__(self):
        self.detect_called = False

    def detect_from_bgr(self, image_bgr):
        self.detect_called = True
        return [[1, 2, 3, 4]], ["bottle"], [0.77], [39]


class FakeStructuredYOLO(FakeYOLO):
    def detect_structured(self, image_bgr, image_source=None):
        self.detect_called = True
        return DetectionResult(
            objects=[
                DetectionObject(
                    label="bottle",
                    confidence=0.77,
                    bbox=[1, 2, 3, 4],
                    metadata={"class_id": 39, "raw_score": 0.77},
                )
            ],
            image_source=image_source,
            model_name="fake-yolo",
            latency_ms=2,
            status=DETECTION_STATUS_SUCCESS,
            metadata={"image_width": 20, "image_height": 10},
        )


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

    def test_chat_route_prefers_structured_t5_generation(self):
        pipeline = self.make_pipeline(nlu=FakeNLU("chat", 0.96), t5=FakeStructuredT5())

        result = pipeline.run("hello")

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.final_answer, "structured-chat:hello")
        self.assertEqual(result.generation.runtime, "onnxruntime")
        self.assertEqual(result.generation.device, "cpu")
        self.assertEqual(result.generation.max_new_tokens, 256)

    def test_rag_route_prefers_structured_t5_generation(self):
        class LegacyContextRAG:
            top_k = 2
            thr = 0.4

            def retrieve(self, question, use_internet=False, web_only=False):
                return ["context text"], 0.82, ["local:test.txt"]

        pipeline = self.make_pipeline(nlu=FakeNLU("rag", 0.95), t5=FakeStructuredT5(), rag=LegacyContextRAG())

        result = pipeline.run("where is the exit?")

        self.assertIn(result.status, ("completed", "degraded"))
        self.assertIsNotNone(result.retrieval)
        self.assertIsNotNone(result.generation)
        self.assertEqual(result.final_answer, "structured-rag:where is the exit?:1")
        self.assertEqual(result.generation.prompt_type, "rag_answer")
        self.assertEqual(result.generation.max_new_tokens, 64)
        self.assertFalse(result.generation.fallback_used)

    def test_document_question_routes_to_rag_when_nlu_returns_chat(self):
        pipeline = self.make_pipeline(nlu=FakeNLU("chat", 0.96), t5=FakeStructuredT5(), rag=FakeRAG())

        result = pipeline.run("what is in the uploaded document?")

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.route.route, "rag")
        self.assertEqual(result.route.source_intent, "chat")
        self.assertFalse(result.route.fallback_used)
        self.assertIn("heuristic", result.route.reason)
        self.assertIsNotNone(result.retrieval)
        self.assertTrue(result.retrieval.used_context)
        self.assertEqual(result.retrieval.chunks[0].source, "local:test.txt")
        self.assertEqual(result.generation.prompt_type, "rag_answer")
        self.assertEqual(result.final_answer, "structured-rag:what is in the uploaded document?:1")
        self.assertIsInstance(result.errors, list)
        self.assertIsInstance(result.warnings, list)
        self.assertIsNotNone(result.duration_ms)

    def test_low_confidence_document_question_routes_to_rag_fallback(self):
        pipeline = self.make_pipeline(nlu=FakeNLU("chat", 0.2), t5=FakeStructuredT5(), rag=FakeRAG())

        result = pipeline.run("according to the manual, what should passengers do?")

        self.assertEqual(result.status, "degraded")
        self.assertEqual(result.route.route, "rag")
        self.assertTrue(result.route.fallback_used)
        self.assertEqual(result.route.fallback_reason, "low_confidence_rag_heuristic")
        self.assertTrue(result.retrieval.used_context)
        self.assertEqual(result.generation.prompt_type, "rag_answer")

    def test_rag_route_model_only_structured_generation_is_marked_fallback(self):
        class EmptyRAG(FakeRAG):
            def retrieve_structured(self, question, use_internet=False, web_only=False):
                return RetrievalResult(
                    query=question,
                    chunks=[],
                    top_k=self.top_k,
                    best_score=0.1,
                    threshold=self.thr,
                    used_context=False,
                    retrieval_mode="empty",
                    fallback_used=True,
                    fallback_reason="empty_retrieval",
                    latency_ms=1,
                )

        pipeline = self.make_pipeline(nlu=FakeNLU("rag", 0.95), t5=FakeStructuredT5(), rag=EmptyRAG())

        result = pipeline.run("unknown topic?")

        self.assertEqual(result.status, "degraded")
        self.assertFalse(result.retrieval.used_context)
        self.assertEqual(result.final_answer, "structured-fallback:unknown topic?")
        self.assertEqual(result.generation.prompt_type, "model_only")
        self.assertTrue(result.generation.fallback_used)
        self.assertEqual(result.generation.fallback_reason, "no_retrieval_context")

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
        self.assertEqual(result.detection.status, DETECTION_STATUS_FAILED)
        self.assertEqual(result.detection.error, "missing_image_for_detection")
        self.assertEqual(result.client_action.action, "capture_photo")
        self.assertTrue(result.warnings)

    def test_object_detect_with_image_fills_detection_and_generation(self):
        yolo = FakeStructuredYOLO()
        pipeline = self.make_pipeline(nlu=FakeNLU("object_detect", 0.98), t5=FakeStructuredT5(), yolo=yolo)

        result = pipeline.run("detect objects", image_bgr=object())

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.route.route, "detect")
        self.assertTrue(yolo.detect_called)
        self.assertEqual(result.detection.status, DETECTION_STATUS_SUCCESS)
        self.assertEqual(result.detection.objects[0].label, "bottle")
        self.assertEqual(result.generation.prompt_type, "detection_narration")
        self.assertEqual(result.final_answer, "structured-detection:1 bottle")

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
    class RecordingDiagentClient:
        def __init__(self):
            self.config = DiagentConfig(
                enabled=True,
                max_chunk_chars=10,
                max_retrieval_chunks=1,
            )
            self.created = []
            self.spans = []
            self.retrievals = []
            self.tool_calls = []
            self.finished = []
            self.closed = False

        def create_run(self, input_text="", *, agent_name=None):
            self.created.append({"input_text": input_text, "agent_name": agent_name})
            return "run-123"

        def log_span(self, run_id, **kwargs):
            self.spans.append((run_id, kwargs))

        def log_retrieval(self, run_id, **kwargs):
            self.retrievals.append((run_id, kwargs))

        def log_tool_call(self, run_id, **kwargs):
            self.tool_calls.append((run_id, kwargs))

        def finish_run(self, run_id, **kwargs):
            self.finished.append((run_id, kwargs))

        def close(self):
            self.closed = True

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

    def test_run_endpoint_emits_diagent_telemetry_with_fake_client(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            import web.app as web_app

        class FakePipeline:
            def run(self, input_text, metadata=None, image_bgr=None):
                return RunResult(
                    input_text=input_text,
                    final_answer="telemetry ok",
                    status="completed",
                    intent=IntentResult(
                        label="rag",
                        confidence=0.93,
                        threshold=0.6,
                        is_confident=True,
                        latency_ms=3,
                    ),
                    route=RouteDecision(
                        route="rag",
                        reason="confident rag intent",
                        source_intent="rag",
                        confidence=0.93,
                    ),
                    retrieval=RetrievalResult(
                        query=input_text,
                        chunks=[
                            RetrievedChunk(
                                text="0123456789abcdef",
                                source="https://example.test/manual",
                                score=0.88,
                                rank=1,
                                retrieval_type="web",
                                metadata={"url": "https://example.test/manual"},
                            ),
                            RetrievedChunk(text="second chunk", source="local:manual.txt"),
                        ],
                        sources=["https://example.test/manual"],
                        top_k=2,
                        best_score=0.88,
                        used_context=True,
                        retrieval_mode="hybrid_local_web",
                    ),
                    generation=GenerationResult(
                        text="telemetry ok",
                        model_name="gemini-2.5-flash",
                        runtime="gemini_api",
                        device="remote",
                        prompt_type="rag_answer",
                        input_tokens=12,
                        output_tokens=4,
                        latency_ms=17,
                    ),
                    detection=DetectionResult(
                        objects=[
                            DetectionObject(label="bottle", confidence=0.91),
                        ],
                        image_source="pipeline",
                        model_name="fake-yolo",
                        latency_ms=5,
                        status=DETECTION_STATUS_SUCCESS,
                    ),
                    client_action=ClientAction(
                        action="capture_photo",
                        reason="frontend capture requested",
                        requires_user_permission=True,
                    ),
                    metadata={"use_internet": bool((metadata or {}).get("use_internet")), "web_only": False},
                )

        fake_client = self.RecordingDiagentClient()
        previous_pipeline = web_app.PIPELINE
        previous_factory = web_app.DIAGENT_CLIENT_FACTORY
        web_app.PIPELINE = FakePipeline()
        web_app.DIAGENT_CLIENT_FACTORY = lambda cfg: fake_client
        try:
            client = TestClient(web_app.app)
            response = client.post(
                "/api/run",
                json={
                    "message": "manual question",
                    "metadata": {"use_internet": True, "request_id": "req-1"},
                },
            )
        finally:
            web_app.PIPELINE = previous_pipeline
            web_app.DIAGENT_CLIENT_FACTORY = previous_factory

        self.assertEqual(response.status_code, 200)
        self.assertEqual(fake_client.created[0]["input_text"], "manual question")
        self.assertTrue(fake_client.closed)

        span_names = [span["name"] for _, span in fake_client.spans]
        self.assertIn("pathfindership.intent", span_names)
        self.assertIn("pathfindership.route_decision", span_names)
        self.assertIn("pathfindership.generation", span_names)
        self.assertIn("pathfindership.client_action", span_names)

        route_span = next(
            span for _, span in fake_client.spans if span["name"] == "pathfindership.route_decision"
        )
        self.assertTrue(route_span["payload"]["retrieval_executed"])
        self.assertTrue(route_span["payload"]["web_search_executed"])
        self.assertEqual(route_span["payload"]["sources_count"], 1)
        self.assertEqual(route_span["payload"]["best_score"], 0.88)
        self.assertTrue(route_span["payload"]["use_internet"])
        self.assertEqual(route_span["payload"]["request_id"], "req-1")

        self.assertEqual(len(fake_client.retrievals), 1)
        retrieved_chunks = fake_client.retrievals[0][1]["retrieved_chunks"]
        self.assertEqual(len(retrieved_chunks), 1)
        self.assertEqual(retrieved_chunks[0]["text"], "0123456...")
        self.assertEqual(retrieved_chunks[0]["source"], "https://example.test/manual")

        generation_span = next(
            span for _, span in fake_client.spans if span["name"] == "pathfindership.generation"
        )
        self.assertEqual(generation_span["payload"]["provider"], "gemini")
        self.assertEqual(generation_span["payload"]["runtime"], "gemini_api")

        self.assertEqual(fake_client.tool_calls[0][1]["tool_name"], "pathfindership.yolo.detect")
        self.assertEqual(fake_client.tool_calls[0][1]["status"], "success")
        self.assertEqual(fake_client.finished[0][1]["status"], "finished")
        self.assertEqual(fake_client.finished[0][1]["output"], "telemetry ok")

    def test_run_endpoint_finishes_diagent_run_on_pipeline_exception(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            import web.app as web_app

        class ExplodingPipeline:
            def run(self, input_text, metadata=None, image_bgr=None):
                raise RuntimeError("boom")

        fake_client = self.RecordingDiagentClient()
        previous_pipeline = web_app.PIPELINE
        previous_factory = web_app.DIAGENT_CLIENT_FACTORY
        web_app.PIPELINE = ExplodingPipeline()
        web_app.DIAGENT_CLIENT_FACTORY = lambda cfg: fake_client
        try:
            client = TestClient(web_app.app, raise_server_exceptions=False)
            response = client.post("/api/run", json={"message": "hello"})
        finally:
            web_app.PIPELINE = previous_pipeline
            web_app.DIAGENT_CLIENT_FACTORY = previous_factory

        self.assertEqual(response.status_code, 500)
        self.assertEqual(fake_client.finished[0][1]["status"], "failed")
        self.assertIn("RuntimeError: boom", fake_client.finished[0][1]["error"])
        self.assertTrue(fake_client.closed)

    def test_run_endpoint_uninitialized_pipeline_returns_structured_failure(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            import web.app as web_app

        previous_pipeline = web_app.PIPELINE
        web_app.PIPELINE = None
        try:
            client = TestClient(web_app.app)
            response = client.post("/api/run", json={"message": "hello", "metadata": {"use_internet": True}})
        finally:
            web_app.PIPELINE = previous_pipeline

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "failed")
        self.assertEqual(body["errors"], ["pipeline is not initialized"])
        self.assertEqual(body["metadata"]["use_internet"], True)
        self.assertEqual(body["metadata"]["web_only"], False)
        self.assertIsNotNone(body["duration_ms"])

    def test_intent_endpoint_returns_intent_without_t5_side_effect(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            import web.app as web_app

        class FakeNLU:
            def classify_intent(self, text, threshold=None):
                return IntentResult(
                    label="open_camera",
                    confidence=0.91,
                    threshold=threshold,
                    is_confident=True,
                    raw_scores={"open_camera": 0.91},
                    latency_ms=1,
                )

        class ExplodingT5:
            def __getattr__(self, name):
                raise AssertionError(f"T5 should not be called by /api/intent: {name}")

        previous_nlu = web_app.NLU
        previous_t5 = web_app.T5
        web_app.NLU = FakeNLU()
        web_app.T5 = ExplodingT5()
        try:
            client = TestClient(web_app.app)
            response = client.post("/api/intent", json={"text": "open camera"})
        finally:
            web_app.NLU = previous_nlu
            web_app.T5 = previous_t5

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["intent"], "open_camera")
        self.assertEqual(body["label"], "open_camera")
        self.assertEqual(body["score"], 0.91)
        self.assertIsNone(body["narration"])
        self.assertTrue(body["is_confident"])

    def test_detect_endpoint_invalid_image_returns_structured_error(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            import web.app as web_app

        client = TestClient(web_app.app)
        response = client.post(
            "/api/detect",
            files={"file": ("bad.txt", b"not an image", "text/plain")},
            data={"draw": "0"},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["labels"], [])
        self.assertEqual(body["summary"], "no objects")
        self.assertEqual(body["detection"]["status"], "invalid_image")
        self.assertEqual(body["detection"]["error"], "unsupported_image_format")
        self.assertEqual(body["error"], "unsupported_image_format")

    def test_detect_endpoint_valid_image_returns_structured_detection(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            import web.app as web_app

        import cv2
        import numpy as np

        class EndpointYOLO:
            def detect_structured(self, image_bgr, image_source=None):
                return DetectionResult(
                    objects=[
                        DetectionObject(
                            label="bottle",
                            confidence=0.88,
                            bbox=[1, 2, 3, 4],
                            metadata={"class_id": 39, "raw_score": 0.88},
                        )
                    ],
                    image_source=image_source,
                    model_name="fake-yolo",
                    latency_ms=1,
                    status=DETECTION_STATUS_SUCCESS,
                )

        ok, encoded = cv2.imencode(".png", np.zeros((4, 4, 3), dtype=np.uint8))
        self.assertTrue(ok)

        previous_yolo = web_app.YOLO
        web_app.YOLO = EndpointYOLO()
        try:
            client = TestClient(web_app.app)
            response = client.post(
                "/api/detect",
                files={"file": ("frame.png", encoded.tobytes(), "image/png")},
                data={"draw": "0"},
            )
        finally:
            web_app.YOLO = previous_yolo

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["labels"], ["bottle"])
        self.assertEqual(body["summary"], "1 bottle")
        self.assertEqual(body["detection"]["status"], "success")
        self.assertEqual(body["detection"]["objects"][0]["bbox"], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(body["detection"]["objects"][0]["confidence"], 0.88)
        self.assertIsNone(body["error"])


if __name__ == "__main__":
    unittest.main()
