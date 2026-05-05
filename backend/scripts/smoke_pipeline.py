from __future__ import annotations

import json
import sys
import types
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

try:
    from tokenizers import Tokenizer as _Tokenizer  # noqa: F401
except ModuleNotFoundError:
    tokenizers_stub = types.ModuleType("tokenizers")

    class _TokenizerStub:
        @classmethod
        def from_file(cls, path: str):
            return cls()

        def encode(self, text: str):
            return types.SimpleNamespace(ids=list(range(min(len(text.split()), 32))))

    tokenizers_stub.Tokenizer = _TokenizerStub
    sys.modules["tokenizers"] = tokenizers_stub

from config import readiness_report
from schemas.pipeline import GenerationResult, IntentResult, to_serializable_dict
from services.pipeline_orchestrator import PipelineOrchestrator


class FakeNLU:
    last_error = None

    def classify_intent(self, text: str, threshold: float | None = None) -> IntentResult:
        label = "open_camera" if "open camera" in text.lower() else "chat"
        confidence = 0.95
        return IntentResult(
            label=label,
            confidence=confidence,
            threshold=threshold,
            is_confident=confidence >= float(threshold or 0.0),
            raw_scores={label: confidence},
            latency_ms=1,
        )


class FakeT5:
    max_new_chat = 256
    max_new_rag = 64

    def chat_structured(self, text: str) -> GenerationResult:
        answer = f"chat:{text}"
        return GenerationResult(
            text=answer,
            model_name="smoke-t5",
            runtime="fake",
            device="cpu",
            prompt_type="chat",
            input_chars=len(text),
            output_chars=len(answer),
            max_new_tokens=self.max_new_chat,
            latency_ms=1,
        )

    def answer_structured(self, question: str, context) -> GenerationResult:
        answer = "rag:uploaded document context is available"
        return GenerationResult(
            text=answer,
            model_name="smoke-t5",
            runtime="fake",
            device="cpu",
            prompt_type="rag_answer",
            input_chars=len(question),
            output_chars=len(answer),
            max_new_tokens=self.max_new_rag,
            latency_ms=1,
        )

    def answer_model_only_with_instruction_structured(self, question: str, instruction: str | None = None) -> GenerationResult:
        answer = "I don't know."
        return GenerationResult(
            text=answer,
            model_name="smoke-t5",
            runtime="fake",
            device="cpu",
            prompt_type="model_only",
            input_chars=len(question),
            output_chars=len(answer),
            max_new_tokens=self.max_new_chat,
            latency_ms=1,
            fallback_used=True,
            fallback_reason="no_retrieval_context",
        )


class FakeRAG:
    top_k = 1
    thr = 0.4
    max_ctx_tokens = 256

    def retrieve(self, question: str, use_internet: bool = False, web_only: bool = False):
        return ["The uploaded document contains smoke-test context."], 0.9, ["local:smoke-upload.txt"]


def _print_json(name: str, value) -> None:
    print(f"\n[{name}]")
    print(json.dumps(to_serializable_dict(value), ensure_ascii=False, indent=2))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    report = readiness_report()
    _print_json("readiness", {
        "status": report.get("status"),
        "assets": report.get("assets"),
        "missing": report.get("missing"),
    })
    _require(report.get("status") in {"ok", "degraded"}, "readiness status missing")

    pipeline = PipelineOrchestrator(
        {"CLS_ROUTE_THRESHOLD": 0.6},
        FakeNLU(),
        FakeT5(),
        FakeRAG(),
    )

    chat = pipeline.run("hello")
    _print_json("normal_chat", chat)
    _require(chat.route and chat.route.route == "chat", "normal chat did not route to chat")
    _require(bool(chat.final_answer), "normal chat final_answer missing")
    _require(chat.duration_ms is not None, "normal chat duration_ms missing")

    camera = pipeline.run("open camera")
    _print_json("camera_client_action", camera)
    _require(camera.client_action is not None, "camera client_action missing")
    _require(camera.client_action.action == "open_camera", "camera action mismatch")
    _require(camera.detection is None, "camera command should not run backend detection")

    rag = pipeline.run("what is in the uploaded document?")
    _print_json("rag_route_decision", rag)
    _require(rag.route and rag.route.route == "rag", "document question did not route to RAG")
    _require(rag.retrieval is not None and rag.retrieval.used_context, "RAG retrieval context missing")
    _require(rag.generation is not None and rag.generation.prompt_type == "rag_answer", "RAG generation missing")

    serialized = to_serializable_dict(rag)
    for key in ("input_text", "status", "intent", "route", "retrieval", "generation", "errors", "warnings", "duration_ms", "final_answer"):
        _require(key in serialized, f"RunResult serialization missing {key}")

    print("\nsmoke_pipeline: ok")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nsmoke_pipeline: failed: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
