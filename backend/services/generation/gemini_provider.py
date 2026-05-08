import logging
import time
from typing import Optional

try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

from schemas.pipeline import GenerationResult
from utils.text import (
    build_chat_prompt,
    build_rag_prompt,
    build_open_camera_prompt,
    build_close_camera_prompt,
    build_take_photo_prompt,
    build_detection_prompt,
    build_model_only_prompt,
    fallback_instruction,
)
from services.generation.base import BaseGenerationProvider

logger = logging.getLogger(__name__)

class GeminiProvider(BaseGenerationProvider):
    def __init__(self, cfg: dict):
        if not HAS_GEMINI:
            raise ImportError(
                "google-genai is not installed. "
                "Please run `pip install google-genai` to use Gemini provider."
            )

        self._model_name = cfg.get("GEMINI_MODEL", "gemini-2.5-flash")
        self.api_key = cfg.get("GEMINI_API_KEY", "")
        self.timeout = float(cfg.get("GEMINI_TIMEOUT_SECONDS", 30))
        self.max_output_tokens = int(cfg.get("GEMINI_MAX_OUTPUT_TOKENS", 512))
        self.temperature = float(cfg.get("GEMINI_TEMPERATURE", 0.2))
        self.bot_name = cfg.get("BOT_NAME", "Passenger-Bot")
        self.app_name = cfg.get("APP_NAME", "PathFinder-Ship")

        if self.api_key:
            # timeout is specified in milliseconds in types.HttpOptions
            timeout_ms = int(self.timeout * 1000)
            self.client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(timeout=timeout_ms)
            )
        else:
            self.client = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def runtime(self) -> str:
        return "gemini_api"

    @property
    def device(self) -> str:
        return "remote"

    @property
    def max_new_chat(self) -> int | None:
        return self.max_output_tokens

    @property
    def max_new_rag(self) -> int | None:
        return self.max_output_tokens

    def chat_structured(self, user_text: str) -> GenerationResult:
        prompt = build_chat_prompt(user_text, self.bot_name, self.app_name)
        return self.generate_structured(prompt, prompt_type="chat")

    def answer_structured(self, question: str, context: list[str] | str | None) -> GenerationResult:
        prompt = build_rag_prompt(question, context)
        return self.generate_structured(prompt, prompt_type="rag_answer")

    def answer_model_only_with_instruction_structured(
        self, question: str, instruction: str | None = None
    ) -> GenerationResult:
        inst = instruction if instruction is not None else fallback_instruction()
        prompt = build_model_only_prompt(question, inst)
        return self.generate_structured(prompt, prompt_type="model_only")

    def narrate_open_camera_structured(self) -> GenerationResult:
        prompt = build_open_camera_prompt(self.bot_name)
        return self.generate_structured(prompt, prompt_type="camera_narration")

    def narrate_close_camera_structured(self) -> GenerationResult:
        prompt = build_close_camera_prompt(self.bot_name)
        return self.generate_structured(prompt, prompt_type="camera_narration")

    def narrate_take_photo_structured(self) -> GenerationResult:
        prompt = build_take_photo_prompt(self.bot_name)
        return self.generate_structured(prompt, prompt_type="camera_narration")

    def narrate_detection_structured(self, objects: list[str] | str) -> GenerationResult:
        prompt = build_detection_prompt(objects, self.bot_name)
        return self.generate_structured(prompt, prompt_type="detection_narration")

    def generate_structured(
        self,
        prompt: str,
        *,
        prompt_type: str = "unknown",
        fallback_text: str | None = None,
    ) -> GenerationResult:
        started = time.perf_counter()
        prompt = prompt or ""

        base = {
            "model_name": self._model_name,
            "runtime": "gemini_api",
            "device": "remote",
            "prompt_type": prompt_type,
            "input_chars": len(prompt),
            "max_new_tokens": self.max_output_tokens,
        }

        def elapsed_ms() -> int:
            return int((time.perf_counter() - started) * 1000)

        if not prompt.strip():
            text = fallback_text or self._fallback_text(prompt_type)
            return GenerationResult(
                text=text,
                **base,
                output_chars=len(text),
                latency_ms=elapsed_ms(),
                empty_output=True,
                fallback_used=True,
                fallback_reason="empty_prompt",
                error="empty_prompt",
            )

        if not self.client:
            text = fallback_text or self._fallback_text(prompt_type)
            return GenerationResult(
                text=text,
                **base,
                output_chars=len(text),
                latency_ms=elapsed_ms(),
                empty_output=True,
                fallback_used=True,
                fallback_reason="missing_api_key",
                error="missing_api_key",
            )

        input_tokens = None
        output_tokens = None

        try:
            count_resp = self.client.models.count_tokens(
                model=self._model_name,
                contents=prompt
            )
            if count_resp and hasattr(count_resp, "total_tokens"):
                input_tokens = int(count_resp.total_tokens)
        except Exception:
            pass

        try:
            response = self.client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=self.max_output_tokens,
                    temperature=self.temperature,
                )
            )

            output_text = response.text if (response and response.text) else ""

            if response and hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = response.usage_metadata
                if hasattr(usage, "prompt_token_count") and usage.prompt_token_count is not None:
                    input_tokens = int(usage.prompt_token_count)
                if hasattr(usage, "candidates_token_count") and usage.candidates_token_count is not None:
                    output_tokens = int(usage.candidates_token_count)

            fallback_reason = self._invalid_generation_reason(output_text)
            result_text = output_text
            fallback_used = False
            empty_output = False

            if fallback_reason:
                result_text = fallback_text or self._fallback_text(prompt_type)
                fallback_used = True
                empty_output = True

            return GenerationResult(
                text=result_text,
                **base,
                output_chars=len(result_text or ""),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=elapsed_ms(),
                empty_output=empty_output,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
            )

        except Exception as exc:
            logger.exception("Gemini API call failed for prompt_type=%s", prompt_type)
            fallback_reason = "api_error"
            text = fallback_text or self._fallback_text(prompt_type)
            return GenerationResult(
                text=text,
                **base,
                output_chars=len(text),
                input_tokens=input_tokens,
                latency_ms=elapsed_ms(),
                empty_output=True,
                fallback_used=True,
                fallback_reason=fallback_reason,
                error=str(exc),
            )

    def _fallback_text(self, prompt_type: str | None) -> str:
        if prompt_type == "camera_narration":
            return "Okay, I will handle that now."
        if prompt_type == "detection_narration":
            return "I couldn't generate a narration right now."
        if prompt_type in {"rag_answer", "model_only"}:
            return "I don't know."
        return "I couldn't generate a response right now."

    def _invalid_generation_reason(self, text: str | None) -> str | None:
        if text is None:
            return "empty_generation"
        stripped = str(text).strip()
        if not stripped:
            return "empty_generation"
        return None
