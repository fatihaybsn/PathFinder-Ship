import unittest
from unittest.mock import patch

from schemas.pipeline import (
    GenerationResult,
    IntentResult,
    RetrievedChunk,
    RetrievalResult,
    RouteDecision,
    RunResult,
)
from services.observability.diagent_config import DiagentConfig
from services.observability.diagent_mapper import (
    clean_empty_fields,
    emit_run_result_telemetry,
    prepare_generation_metadata,
    sanitize_retrieval_chunks,
    truncate_text,
)
from services.observability.diagent_safe_client import DiagentSafeClient


class DiagentConfigTests(unittest.TestCase):
    def test_config_defaults_are_safe_for_local_development(self):
        cfg = DiagentConfig.from_mapping({})

        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.api_url, "http://localhost:8000")
        self.assertEqual(cfg.agent_name, "pathfindership")
        self.assertEqual(cfg.timeout_seconds, 5.0)
        self.assertTrue(cfg.fail_open)
        self.assertTrue(cfg.log_policy_spans)
        self.assertEqual(cfg.max_chunk_chars, 1200)
        self.assertEqual(cfg.max_retrieval_chunks, 5)

    def test_config_parses_string_values(self):
        cfg = DiagentConfig.from_mapping(
            {
                "DIAGENT_ENABLED": "true",
                "DIAGENT_API_URL": "http://localhost:18000/",
                "DIAGENT_AGENT_NAME": "ship-test",
                "DIAGENT_TIMEOUT_SECONDS": "2.5",
                "DIAGENT_FAIL_OPEN": "yes",
                "DIAGENT_LOG_POLICY_SPANS": "false",
                "DIAGENT_MAX_CHUNK_CHARS": "64",
                "DIAGENT_MAX_RETRIEVAL_CHUNKS": "2",
            }
        )

        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.api_url, "http://localhost:18000")
        self.assertEqual(cfg.agent_name, "ship-test")
        self.assertEqual(cfg.timeout_seconds, 2.5)
        self.assertTrue(cfg.fail_open)
        self.assertFalse(cfg.log_policy_spans)
        self.assertEqual(cfg.max_chunk_chars, 64)
        self.assertEqual(cfg.max_retrieval_chunks, 2)


class DiagentSafeClientTests(unittest.TestCase):
    def test_disabled_client_is_noop_and_never_builds_tracer(self):
        calls = []

        def factory(**kwargs):
            calls.append(kwargs)
            raise AssertionError("disabled mode must not build the tracer")

        client = DiagentSafeClient(
            DiagentConfig(enabled=False),
            tracer_factory=factory,
        )

        self.assertIsNone(client.create_run("hello"))
        self.assertIsNone(client.log_span("run-1", span_type="system", name="intent"))
        self.assertIsNone(client.log_retrieval("run-1", query="q", retrieved_chunks=[]))
        self.assertIsNone(client.log_tool_call("run-1", tool_name="camera"))
        self.assertIsNone(client.finish_run("run-1", output="ok", status="finished"))
        client.close()

        self.assertEqual(calls, [])

    def test_missing_sdk_import_is_noop_without_crashing(self):
        def failing_loader():
            raise ImportError("diagent is not installed")

        client = DiagentSafeClient(
            DiagentConfig(enabled=True),
            tracer_loader=failing_loader,
        )

        with self.assertLogs("services.observability.diagent_safe_client", level="WARNING"):
            result = client.create_run("hello")

        self.assertIsNone(result)
        self.assertFalse(client.sdk_available)

    def test_api_connection_error_is_swallowed(self):
        class FailingTracer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def create_run(self, agent_name, input_text=""):
                raise ConnectionError("connection refused")

        client = DiagentSafeClient(
            DiagentConfig(enabled=True, api_url="http://localhost:9999", timeout_seconds=1),
            tracer_factory=FailingTracer,
        )

        with self.assertLogs("services.observability.diagent_safe_client", level="WARNING"):
            result = client.create_run("hello")

        self.assertIsNone(result)
        self.assertTrue(client.sdk_available)


class DiagentReadinessTests(unittest.TestCase):
    def test_readiness_reports_diagent_without_degrading_core_status(self):
        from config import readiness_report

        cfg = {
            "GENERATION_PROVIDER": "gemini",
            "GEMINI_API_KEY": "fake-key",
            "T5_ENCODER": "assets/t5/encoder.onnx",
            "T5_DECODER": "assets/t5/decoder.onnx",
            "T5_TOKENIZER_DIR": "assets/t5/tokenizer",
            "CLS_ONNX": "assets/nlu/model.onnx",
            "CLS_TOKENIZER_DIR": "assets/nlu/tokenizer",
            "YOLO_ONNX": "assets/yolo/model.onnx",
            "YOLO_LABELS": "assets/yolo/labels.txt",
            "RAG_CORPUS_DIR": "data/rag/corpus",
            "CHROMA_PATH": "assets/rag/chroma_db",
            "RAG_SQLITE_PATH": "assets/rag/chroma_db/bm25.sqlite",
            "ENABLE_EMAIL": False,
            "ENABLE_WEB_SEARCH": False,
            "ENABLE_CAMERA_ACTIONS": True,
            "DIAGENT_ENABLED": True,
            "DIAGENT_API_URL": "http://localhost:8000",
            "DIAGENT_AGENT_NAME": "pathfindership",
            "DIAGENT_TIMEOUT_SECONDS": 5,
            "DIAGENT_FAIL_OPEN": True,
            "DIAGENT_LOG_POLICY_SPANS": True,
            "DIAGENT_MAX_CHUNK_CHARS": 1200,
            "DIAGENT_MAX_RETRIEVAL_CHUNKS": 5,
        }

        with (
            patch("config.CFG", cfg),
            patch("config._path_exists", return_value=True),
            patch(
                "services.observability.diagent_safe_client.is_diagent_sdk_available",
                return_value=False,
            ),
        ):
            report = readiness_report()

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["missing"], [])
        self.assertIn("diagent", report)
        self.assertTrue(report["diagent"]["enabled"])
        self.assertFalse(report["diagent"]["sdk_available"])


class DiagentMapperTests(unittest.TestCase):
    def test_truncate_text_uses_bounded_suffix(self):
        self.assertEqual(truncate_text("abcdef", 4), "a...")
        self.assertEqual(truncate_text("abcdef", 2), "ab")

    def test_retrieval_chunk_sanitization_truncates_and_preserves_source_url(self):
        retrieval = RetrievalResult(
            query="manual question",
            chunks=[
                RetrievedChunk(
                    text="0123456789abcdef",
                    source="local:manual.txt",
                    score=0.91,
                    rank=1,
                    retrieval_type="local_hybrid",
                    metadata={
                        "url": "https://example.test/manual",
                        "empty": "",
                        "api_key": "secret",
                        "image_base64": "not-for-telemetry",
                    },
                ),
                RetrievedChunk(text="second", source="local:second.txt"),
            ],
            top_k=2,
            used_context=True,
        )

        chunks = sanitize_retrieval_chunks(retrieval, max_chunks=1, max_chunk_chars=8)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], "01234...")
        self.assertEqual(chunks[0]["source"], "local:manual.txt")
        self.assertEqual(chunks[0]["url"], "https://example.test/manual")
        self.assertEqual(chunks[0]["score"], 0.91)
        self.assertNotIn("empty", chunks[0]["metadata"])
        self.assertNotIn("api_key", chunks[0]["metadata"])
        self.assertNotIn("image_base64", chunks[0]["metadata"])

    def test_generation_metadata_marks_api_vs_local_and_keeps_false_values(self):
        api_generation = GenerationResult(
            text="answer",
            model_name="gemini-2.5-flash",
            runtime="gemini_api",
            device="remote",
            prompt_type="rag_answer",
            input_tokens=10,
            output_tokens=5,
            empty_output=False,
            fallback_used=False,
        )
        local_generation = GenerationResult(
            text="answer",
            model_name="local-t5-onnx",
            runtime="onnxruntime",
            device="cpu",
            prompt_type="chat",
            input_tokens=4,
            output_tokens=2,
        )

        api_metadata = prepare_generation_metadata(api_generation)
        local_metadata = prepare_generation_metadata(local_generation)

        self.assertEqual(api_metadata["provider"], "gemini")
        self.assertEqual(api_metadata["provider_family"], "api")
        self.assertEqual(api_metadata["total_tokens"], 15)
        self.assertFalse(api_metadata["empty_output"])
        self.assertFalse(api_metadata["fallback_used"])
        self.assertEqual(local_metadata["provider"], "local_t5")
        self.assertEqual(local_metadata["provider_family"], "local")
        self.assertEqual(local_metadata["total_tokens"], 6)

    def test_emit_run_result_telemetry_logs_empty_retrieval_as_empty_chunk_list(self):
        class RecordingClient:
            def __init__(self):
                self.config = DiagentConfig(
                    enabled=True,
                    max_chunk_chars=8,
                    max_retrieval_chunks=1,
                )
                self.spans = []
                self.retrievals = []
                self.tool_calls = []

            def log_span(self, run_id, **kwargs):
                self.spans.append((run_id, kwargs))

            def log_retrieval(self, run_id, **kwargs):
                self.retrievals.append((run_id, kwargs))

            def log_tool_call(self, run_id, **kwargs):
                self.tool_calls.append((run_id, kwargs))

        result = RunResult(
            input_text="what is in the uploaded document?",
            final_answer="I don't know.",
            status="degraded",
            intent=IntentResult(label="chat", confidence=0.95, threshold=0.6, is_confident=True),
            route=RouteDecision(route="rag", reason="document query", source_intent="chat"),
            retrieval=RetrievalResult(
                query="what is in the uploaded document?",
                chunks=[],
                top_k=4,
                best_score=0.1,
                used_context=False,
                retrieval_mode="empty",
                fallback_used=True,
                fallback_reason="empty_retrieval",
            ),
            metadata={"use_internet": True, "web_only": False},
        )
        client = RecordingClient()

        emit_run_result_telemetry(
            client,
            "run-1",
            result,
            config=client.config,
            request_metadata={"request_id": "req-1", "use_internet": True},
            app_config={
                "ENABLE_WEB_SEARCH": True,
                "GENERATION_PROVIDER": "local_t5",
                "RAG_WEB_MIN_STRENGTH": 0.75,
            },
        )

        self.assertEqual(len(client.retrievals), 1)
        self.assertEqual(client.retrievals[0][1]["retrieved_chunks"], [])
        self.assertEqual(client.retrievals[0][1]["top_k"], 4)

        route_span = next(
            payload
            for _, payload in client.spans
            if payload["name"] == "pathfindership.route_decision"
        )
        self.assertTrue(route_span["payload"]["retrieval_executed"])
        self.assertFalse(route_span["payload"]["web_search_executed"])
        self.assertEqual(route_span["payload"]["sources_count"], 0)
        self.assertEqual(route_span["payload"]["best_score"], 0.1)
        self.assertEqual(route_span["payload"]["local_retrieval_score"], 0.1)
        self.assertEqual(route_span["payload"]["web_fallback_threshold"], 0.75)
        self.assertTrue(route_span["payload"]["use_internet"])

    def test_clean_empty_fields_keeps_zero_and_false(self):
        cleaned = clean_empty_fields(
            {
                "none": None,
                "empty": "",
                "zero": 0,
                "false": False,
                "nested": {"empty_list": [], "value": "x"},
            }
        )

        self.assertNotIn("none", cleaned)
        self.assertNotIn("empty", cleaned)
        self.assertEqual(cleaned["zero"], 0)
        self.assertFalse(cleaned["false"])
        self.assertEqual(cleaned["nested"], {"value": "x"})


if __name__ == "__main__":
    unittest.main()
