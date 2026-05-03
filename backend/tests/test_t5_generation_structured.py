import sys
import types
import unittest
from unittest.mock import MagicMock


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


from services.t5 import T5DecodeError, T5Service


class FakeTokenizer:
    all_special_tokens = ["<pad>", "</s>"]


class T5StructuredGenerationTests(unittest.TestCase):
    def make_service(self, output="hello", meta=None, exc=None):
        service = T5Service.__new__(T5Service)
        service.model_name = "local-t5-onnx"
        service.runtime = "onnxruntime"
        service.device = "cpu"
        service.max_new_chat = 256
        service.max_new_rag = 64
        service.max_src_len = 512
        service.bot_name = "Passenger-Bot"
        service.app_name = "PathFinder-Ship"
        service.tok = FakeTokenizer()
        service.called = False

        def fake_generate(prompt, mode, max_new_tokens=None):
            service.called = True
            if exc:
                raise exc
            return output, meta or {
                "input_tokens": 4,
                "output_tokens": 2,
                "input_truncated": False,
                "output_truncated": False,
            }

        service._generate_text_with_metadata = fake_generate
        return service

    def test_generate_structured_returns_generation_result_metadata(self):
        service = self.make_service(output="hello passenger")

        result = service.generate_structured(
            "Prompt text",
            mode="chat",
            prompt_type="chat",
        )

        self.assertEqual(result.text, "hello passenger")
        self.assertEqual(result.model_name, "local-t5-onnx")
        self.assertEqual(result.runtime, "onnxruntime")
        self.assertEqual(result.device, "cpu")
        self.assertEqual(result.prompt_type, "chat")
        self.assertEqual(result.input_chars, len("Prompt text"))
        self.assertEqual(result.output_chars, len("hello passenger"))
        self.assertEqual(result.input_tokens, 4)
        self.assertEqual(result.output_tokens, 2)
        self.assertEqual(result.max_new_tokens, 256)
        self.assertFalse(result.input_truncated)
        self.assertFalse(result.output_truncated)
        self.assertFalse(result.empty_output)
        self.assertFalse(result.fallback_used)
        self.assertIsNone(result.fallback_reason)
        self.assertIsNone(result.error)
        self.assertIsInstance(result.latency_ms, int)

    def test_legacy_chat_returns_text_from_structured_result(self):
        service = self.make_service(output="legacy text")

        result = service.chat("hello")

        self.assertEqual(result, "legacy text")

    def test_empty_generation_uses_controlled_fallback(self):
        service = self.make_service(output="   ")

        result = service.generate_structured(
            "Prompt text",
            mode="rag",
            prompt_type="rag_answer",
            fallback_text="fallback answer",
        )

        self.assertEqual(result.text, "fallback answer")
        self.assertTrue(result.empty_output)
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.fallback_reason, "empty_generation")
        self.assertIsNone(result.error)

    def test_special_token_only_generation_is_invalid(self):
        service = self.make_service(output="<pad> </s>")

        result = service.generate_structured(
            "Prompt text",
            mode="chat",
            prompt_type="chat",
            fallback_text="fallback answer",
        )

        self.assertEqual(result.text, "fallback answer")
        self.assertTrue(result.empty_output)
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.fallback_reason, "invalid_generation")

    def test_inference_exception_becomes_structured_fallback(self):
        service = self.make_service(exc=RuntimeError("boom"))

        with self.assertLogs("services.t5", level="ERROR"):
            result = service.generate_structured(
                "Prompt text",
                mode="chat",
                prompt_type="chat",
            )

        self.assertTrue(result.empty_output)
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.fallback_reason, "inference_failed")
        self.assertEqual(result.error, "inference_failed")
        self.assertEqual(result.text, "I couldn't generate a response right now.")

    def test_decode_exception_becomes_structured_fallback(self):
        service = self.make_service(exc=T5DecodeError("decode failed"))

        result = service.generate_structured(
            "Prompt text",
            mode="rag",
            prompt_type="rag_answer",
        )

        self.assertTrue(result.empty_output)
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.fallback_reason, "decode_failed")
        self.assertEqual(result.error, "decode_failed")
        self.assertEqual(result.text, "I don't know.")

    def test_empty_prompt_does_not_call_model(self):
        service = self.make_service(output="should not run")

        result = service.generate_structured(
            "",
            mode="chat",
            prompt_type="unknown",
            fallback_text="empty fallback",
        )

        self.assertFalse(service.called)
        self.assertEqual(result.text, "empty fallback")
        self.assertTrue(result.empty_output)
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.fallback_reason, "empty_prompt")
        self.assertEqual(result.error, "empty_prompt")


if __name__ == "__main__":
    unittest.main()
