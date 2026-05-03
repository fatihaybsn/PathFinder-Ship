from __future__ import annotations

import json
import sys

from config import CFG
from schemas.pipeline import GenerationResult, to_serializable_dict
from services.t5 import T5Service


def _print_result(name: str, result: GenerationResult) -> bool:
    body = to_serializable_dict(result)
    print(f"\n[{name}]")
    print(json.dumps(body, ensure_ascii=False, indent=2))

    ok = True
    if not isinstance(result.text, str) or result.text == "":
        print(f"{name}: text is empty")
        ok = False
    if result.runtime != "onnxruntime":
        print(f"{name}: runtime is not onnxruntime")
        ok = False
    if result.device != "cpu":
        print(f"{name}: device is not cpu")
        ok = False
    if result.latency_ms is None:
        print(f"{name}: latency_ms is missing")
        ok = False
    if result.max_new_tokens is None:
        print(f"{name}: max_new_tokens is missing")
        ok = False
    return ok


def main() -> int:
    service = T5Service(CFG)
    checks = [
        (
            "chat",
            lambda: service.chat_structured("Hello, who are you?"),
        ),
        (
            "rag_answer",
            lambda: service.answer_structured(
                "What should passengers do in an emergency?",
                ["Passengers should follow crew instructions and move to muster stations."],
            ),
        ),
        (
            "empty_prompt",
            lambda: service.generate_structured(
                "",
                mode="chat",
                prompt_type="unknown",
                fallback_text="Empty prompt fallback.",
            ),
        ),
    ]

    ok = True
    for name, run in checks:
        try:
            ok = _print_result(name, run()) and ok
        except Exception as exc:
            print(f"{name}: smoke check crashed: {type(exc).__name__}")
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
