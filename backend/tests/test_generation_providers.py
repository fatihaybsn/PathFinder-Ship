import sys
import types
from unittest.mock import MagicMock, patch
import unittest

# Stub onnxruntime/transformers/tokenizers if not importable to keep test execution light and robust
if "onnxruntime" not in sys.modules:
    ort_stub = types.ModuleType("onnxruntime")
    class _SessionOptions:
        graph_optimization_level = None
    class _GraphOptimizationLevel:
        ORT_ENABLE_ALL = "ORT_ENABLE_ALL"
    ort_stub.SessionOptions = _SessionOptions
    ort_stub.GraphOptimizationLevel = _GraphOptimizationLevel
    ort_stub.InferenceSession = MagicMock()
    sys.modules["onnxruntime"] = ort_stub

if "transformers" not in sys.modules:
    transformers_stub = types.ModuleType("transformers")
    auto_tokenizer = MagicMock()
    auto_tokenizer.from_pretrained = MagicMock(return_value=MagicMock())
    transformers_stub.AutoTokenizer = auto_tokenizer
    sys.modules["transformers"] = transformers_stub

if "tokenizers" not in sys.modules:
    tokenizers_stub = types.ModuleType("tokenizers")
    tokenizers_stub.Tokenizer = MagicMock()
    sys.modules["tokenizers"] = tokenizers_stub

# Mock the new google-genai SDK
mock_genai_module = types.ModuleType("google.genai")
mock_types_submodule = types.ModuleType("google.genai.types")

class HttpOptionsMock:
    def __init__(self, **kwargs):
        pass

class GenerateContentConfigMock:
    def __init__(self, **kwargs):
        pass

mock_types_submodule.HttpOptions = HttpOptionsMock
mock_types_submodule.GenerateContentConfig = GenerateContentConfigMock
mock_genai_module.types = mock_types_submodule

# Mock Client
class ClientMock:
    def __init__(self, api_key=None, http_options=None):
        self.models = MagicMock()

mock_genai_module.Client = ClientMock

# Register stubs under sys.modules
sys.modules["google.genai"] = mock_genai_module
sys.modules["google.genai.types"] = mock_types_submodule

# Force HAS_GEMINI to True for tests
import services.generation.gemini_provider
services.generation.gemini_provider.HAS_GEMINI = True

from services.generation.factory import build_generation_provider
from services.generation.local_t5_provider import LocalT5Provider
from services.generation.gemini_provider import GeminiProvider
from schemas.pipeline import GenerationResult
from services.pipeline_orchestrator import PipelineOrchestrator
from config import readiness_report, CFG


class GenerationProvidersTests(unittest.TestCase):
    def setUp(self):
        self.cfg_local = {
            "GENERATION_PROVIDER": "local_t5",
            "BOT_NAME": "TestBot",
            "APP_NAME": "TestApp",
            "T5_TOKENIZER_DIR": "dummy/tokenizer",
            "T5_ENCODER": "dummy/encoder.onnx",
            "T5_DECODER": "dummy/decoder.onnx",
        }
        self.cfg_gemini = {
            "GENERATION_PROVIDER": "gemini",
            "BOT_NAME": "TestBot",
            "APP_NAME": "TestApp",
            "GEMINI_API_KEY": "fake-api-key",
            "GEMINI_MODEL": "gemini-2.5-flash",
            "GEMINI_TIMEOUT_SECONDS": 10,
            "GEMINI_MAX_OUTPUT_TOKENS": 256,
            "GEMINI_TEMPERATURE": 0.5,
        }

    @patch("services.generation.local_t5_provider.T5Service")
    def test_factory_builds_local_t5_provider(self, mock_t5_service):
        provider = build_generation_provider(self.cfg_local)
        self.assertIsInstance(provider, LocalT5Provider)

    def test_factory_builds_gemini_provider(self):
        provider = build_generation_provider(self.cfg_gemini)
        self.assertIsInstance(provider, GeminiProvider)
        self.assertEqual(provider.model_name, "gemini-2.5-flash")
        self.assertEqual(provider.runtime, "gemini_api")
        self.assertEqual(provider.device, "remote")

    def test_factory_raises_value_error_for_unknown_provider(self):
        cfg = {"GENERATION_PROVIDER": "invalid_provider"}
        with self.assertRaises(ValueError):
            build_generation_provider(cfg)

    @patch("services.generation.local_t5_provider.T5Service")
    def test_local_t5_provider_delegates_calls(self, mock_t5_service_cls):
        mock_service = MagicMock()
        mock_t5_service_cls.return_value = mock_service
        
        provider = LocalT5Provider(self.cfg_local)
        
        provider.chat_structured("hello")
        mock_service.chat_structured.assert_called_once_with("hello")
        
        provider.answer_structured("question", "context")
        mock_service.answer_structured.assert_called_once_with("question", "context")

    def test_gemini_provider_generate_structured_success(self):
        provider = GeminiProvider(self.cfg_gemini)
        
        # Mock total tokens response
        mock_count = MagicMock()
        mock_count.total_tokens = 10
        provider.client.models.count_tokens.return_value = mock_count

        # Mock generate content response
        mock_resp = MagicMock()
        mock_resp.text = "Hello Passenger!"
        
        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 12
        mock_usage.candidates_token_count = 8
        mock_resp.usage_metadata = mock_usage
        
        provider.client.models.generate_content.return_value = mock_resp

        result = provider.chat_structured("hello")
        
        self.assertIsInstance(result, GenerationResult)
        self.assertEqual(result.text, "Hello Passenger!")
        self.assertEqual(result.model_name, "gemini-2.5-flash")
        self.assertEqual(result.runtime, "gemini_api")
        self.assertEqual(result.device, "remote")
        self.assertEqual(result.input_tokens, 12)
        self.assertEqual(result.output_tokens, 8)
        self.assertFalse(result.fallback_used)

    def test_gemini_provider_generate_structured_empty_response(self):
        provider = GeminiProvider(self.cfg_gemini)
        
        mock_resp = MagicMock()
        mock_resp.text = "   "
        provider.client.models.generate_content.return_value = mock_resp

        result = provider.chat_structured("hello")
        
        self.assertEqual(result.text, "I couldn't generate a response right now.")
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.fallback_reason, "empty_generation")

    def test_gemini_provider_generate_structured_exception(self):
        provider = GeminiProvider(self.cfg_gemini)
        
        provider.client.models.generate_content.side_effect = RuntimeError("API rate limit exceeded")

        result = provider.chat_structured("hello")
        
        self.assertEqual(result.text, "I couldn't generate a response right now.")
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.fallback_reason, "api_error")
        self.assertIn("API rate limit exceeded", result.error)

    @patch("config.CFG")
    @patch("config._path_exists")
    def test_readiness_report_modes(self, mock_exists, mock_cfg):
        # Setup mock exists to return False for T5 assets, but True for NLU and YOLO
        def exists_side_effect(path, kind="any"):
            if "t5" in str(path):
                return False
            return True
        mock_exists.side_effect = exists_side_effect

        # 1. Local T5 mode
        mock_cfg.get.side_effect = lambda key, default=None: {
            "GENERATION_PROVIDER": "local_t5",
            "T5_ENCODER": "assets/t5/encoder.onnx",
            "T5_DECODER": "assets/t5/decoder.onnx",
            "T5_TOKENIZER_DIR": "assets/t5/tokenizer",
            "CLS_ONNX": "assets/nlu/nlu.onnx",
            "CLS_TOKENIZER_DIR": "assets/nlu/tokenizer",
            "YOLO_ONNX": "assets/yolo/yolo.onnx",
            "YOLO_LABELS": "assets/yolo/labels.txt",
            "RAG_CORPUS_DIR": "data/rag/corpus",
            "CHROMA_PATH": "assets/rag/chroma",
            "RAG_SQLITE_PATH": "assets/rag/chroma/bm25.sqlite",
            "GEMINI_API_KEY": "",
        }.get(key, default)

        report = readiness_report()
        self.assertEqual(report["generation_provider"], "local_t5")
        self.assertFalse(report["local_t5_ready"])
        self.assertEqual(report["status"], "degraded")
        self.assertIn("t5_model", report["missing"])

        # 2. Gemini mode (configured)
        mock_cfg.get.side_effect = lambda key, default=None: {
            "GENERATION_PROVIDER": "gemini",
            "T5_ENCODER": "assets/t5/encoder.onnx",
            "T5_DECODER": "assets/t5/decoder.onnx",
            "T5_TOKENIZER_DIR": "assets/t5/tokenizer",
            "CLS_ONNX": "assets/nlu/nlu.onnx",
            "CLS_TOKENIZER_DIR": "assets/nlu/tokenizer",
            "YOLO_ONNX": "assets/yolo/yolo.onnx",
            "YOLO_LABELS": "assets/yolo/labels.txt",
            "RAG_CORPUS_DIR": "data/rag/corpus",
            "CHROMA_PATH": "assets/rag/chroma",
            "RAG_SQLITE_PATH": "assets/rag/chroma/bm25.sqlite",
            "GEMINI_API_KEY": "some-key",
        }.get(key, default)

        report = readiness_report()
        self.assertEqual(report["generation_provider"], "gemini")
        self.assertTrue(report["gemini_configured"])
        # Status should be ok since T5 missing is ignored in gemini mode
        self.assertEqual(report["status"], "ok")
        self.assertNotIn("t5_model", report["missing"])

    def test_pipeline_orchestrator_runs_with_gemini(self):
        # Mock dependencies for orchestrator
        nlu = MagicMock()
        rag = MagicMock()
        yolo = MagicMock()
        
        provider = GeminiProvider(self.cfg_gemini)
        
        # Mock generate content response
        mock_resp = MagicMock()
        mock_resp.text = "Mocked chat response"
        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 10
        mock_usage.candidates_token_count = 5
        mock_resp.usage_metadata = mock_usage
        provider.client.models.generate_content.return_value = mock_resp

        orchestrator = PipelineOrchestrator(
            cfg={"CLS_ROUTE_THRESHOLD": 0.60},
            nlu=nlu,
            t5=provider,
            rag=rag,
            yolo=yolo
        )
        
        # Mock NLU classification to return chat route
        from schemas.pipeline import IntentResult
        mock_intent = IntentResult(
            label="chat",
            confidence=0.95,
            threshold=0.60,
            is_confident=True,
            error=None
        )
        nlu.classify_intent.return_value = mock_intent

        result = orchestrator.run("hello")
        
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.final_answer, "Mocked chat response")
        self.assertEqual(result.generation.model_name, "gemini-2.5-flash")
        self.assertEqual(result.generation.runtime, "gemini_api")


if __name__ == "__main__":
    unittest.main()
