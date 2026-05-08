from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Any, Mapping

from schemas.pipeline import (
    ClientAction,
    DETECTION_STATUS_FAILED,
    DETECTION_STATUS_MODEL_ERROR,
    DETECTION_STATUS_NO_OBJECTS,
    DETECTION_STATUS_SUCCESS,
    DetectionResult,
    GenerationResult,
    IntentResult,
    RetrievalResult,
    RouteDecision,
    RunResult,
    detection_result_from_legacy,
    generation_result_from_text,
    intent_result_from_prediction,
    retrieval_result_from_legacy,
)
from services.route_decision import CAMERA_ACTIONS, DEFAULT_INTENT_THRESHOLD, decide_route, normalize_intent_label
from services.generation.base import BaseGenerationProvider
from utils.text import fallback_instruction

logger = logging.getLogger(__name__)


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _metadata_bool(metadata: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = metadata.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _detection_summary(detection: DetectionResult) -> str:
    labels = [obj.label for obj in detection.objects]
    return ", ".join(f"{count} {label}" for label, count in Counter(labels).items()) if labels else "no objects"


class PipelineOrchestrator:
    """
    First central backend pipeline for a single user message.

    The existing services keep their legacy return shapes. This class maps them
    into the shared structured result models without changing service internals.
    """

    def __init__(
        self,
        cfg: Mapping[str, Any],
        nlu: Any,
        t5: BaseGenerationProvider,
        rag: Any,
        yolo: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.nlu = nlu
        self.t5 = t5
        self.rag = rag
        self.yolo = yolo
        self.intent_threshold = float(cfg.get("CLS_ROUTE_THRESHOLD", DEFAULT_INTENT_THRESHOLD))

    def run(
        self,
        input_text: str,
        metadata: Mapping[str, Any] | None = None,
        image_bgr: Any | None = None,
    ) -> RunResult:
        started = time.perf_counter()
        warnings: list[str] = []
        errors: list[str] = []
        text = (input_text or "").strip()
        request_options = self._request_options(metadata)

        if not text:
            return RunResult(
                input_text=input_text or "",
                status="failed",
                errors=["message must not be empty"],
                warnings=warnings,
                metadata=request_options,
                duration_ms=_elapsed_ms(started),
            )

        intent = self._predict_intent(text, warnings)
        route = self._decide_route(text, intent, request_options, image_bgr)

        try:
            result = self._execute_route(text, route, intent, request_options, image_bgr, warnings)
        except Exception:
            logger.exception("Pipeline route failed: %s", route.route)
            errors.append(f"{route.route} service failed")
            return RunResult(
                input_text=text,
                status="failed",
                intent=intent,
                route=route,
                errors=errors,
                warnings=warnings,
                metadata=request_options,
                duration_ms=_elapsed_ms(started),
            )

        result.duration_ms = _elapsed_ms(started)
        result.metadata = request_options
        if errors:
            result.errors.extend(errors)
        for warning in warnings:
            if warning not in result.warnings:
                result.warnings.append(warning)
        if warnings and result.status == "completed":
            result.status = "degraded"
        return result

    def _request_options(self, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
        metadata = metadata or {}
        use_internet = _metadata_bool(metadata, "use_internet", False)
        web_only = _metadata_bool(metadata, "web_only", False)
        return {"use_internet": use_internet, "web_only": web_only}

    def _predict_intent(self, text: str, warnings: list[str]) -> IntentResult:
        started = time.perf_counter()
        try:
            if hasattr(self.nlu, "classify_intent"):
                intent = self.nlu.classify_intent(text, threshold=self.intent_threshold)
                if intent.threshold is None:
                    intent.threshold = self.intent_threshold
                if intent.latency_ms is None:
                    intent.latency_ms = _elapsed_ms(started)
            else:
                label, confidence = self.nlu.predict(text)
                error = None
                if getattr(self.nlu, "last_error", None):
                    error = "intent service failed"
                intent = intent_result_from_prediction(
                    label=label,
                    confidence=float(confidence) if confidence is not None else None,
                    threshold=self.intent_threshold,
                    latency_ms=_elapsed_ms(started),
                    error=error,
                )

            if intent.error:
                warnings.append("Intent service degraded; routed with fallback.")
        except Exception:
            logger.exception("Intent prediction failed")
            warnings.append("Intent service failed; routed as chat fallback.")
            return intent_result_from_prediction(
                label="chat",
                confidence=0.0,
                threshold=self.intent_threshold,
                latency_ms=_elapsed_ms(started),
                error="intent service failed",
            )

        return intent

    def _decide_route(
        self,
        text: str,
        intent: IntentResult,
        request_options: Mapping[str, Any],
        image_bgr: Any | None,
    ) -> RouteDecision:
        route_metadata = dict(request_options)
        route_metadata["has_image"] = image_bgr is not None
        route_metadata["intent_threshold"] = self.intent_threshold
        return decide_route(text, intent, metadata=route_metadata)

    def _execute_route(
        self,
        text: str,
        route: RouteDecision,
        intent: IntentResult,
        request_options: Mapping[str, Any],
        image_bgr: Any | None,
        warnings: list[str],
    ) -> RunResult:
        if route.route == "camera_action":
            return self._run_client_action(text, route, intent)
        if route.route == "detect":
            return self._run_detection(text, route, intent, image_bgr, warnings)
        if route.route == "chat":
            return self._run_chat(text, route, intent)
        return self._run_rag(text, route, intent, request_options)

    def _run_client_action(self, text: str, route: RouteDecision, intent: IntentResult) -> RunResult:
        action = route.client_action or "none"
        requires_permission = CAMERA_ACTIONS.get(normalize_intent_label(intent.label), (action, False))[1]
        client_action = ClientAction(
            action=action,
            reason=route.reason,
            requires_user_permission=requires_permission,
        )

        return RunResult(
            input_text=text,
            status="completed",
            intent=intent,
            route=route,
            client_action=client_action,
        )

    def _run_detection(
        self,
        text: str,
        route: RouteDecision,
        intent: IntentResult,
        image_bgr: Any | None,
        warnings: list[str],
    ) -> RunResult:
        if image_bgr is None:
            warning = "missing_image_for_detection"
            if warning not in warnings:
                warnings.append(warning)
            return RunResult(
                input_text=text,
                final_answer="Please capture or upload an image for object detection.",
                status="degraded",
                intent=intent,
                route=route,
                detection=DetectionResult(status=DETECTION_STATUS_FAILED, error=warning),
                client_action=ClientAction(
                    action="capture_photo",
                    reason="object detection requires an image",
                    requires_user_permission=True,
                ),
                warnings=warnings,
            )

        if self.yolo is None:
            warning = "detection_service_unavailable"
            if warning not in warnings:
                warnings.append(warning)
            return RunResult(
                input_text=text,
                final_answer="Object detection is not available right now.",
                status="degraded",
                intent=intent,
                route=route,
                detection=DetectionResult(status=DETECTION_STATUS_MODEL_ERROR, error=warning),
                warnings=warnings,
            )

        if hasattr(self.yolo, "detect_structured"):
            detection = self.yolo.detect_structured(image_bgr, image_source="pipeline")
        else:
            started = time.perf_counter()
            boxes, labels, scores, cls_ids = self.yolo.detect_from_bgr(image_bgr)
            detection = detection_result_from_legacy(
                labels,
                boxes,
                scores,
                cls_ids,
                image_source="pipeline",
                model_name="yolo",
                latency_ms=_elapsed_ms(started),
            )

        summary = _detection_summary(detection)
        generation: GenerationResult | None = None
        answer = f"Object summary: {summary}"
        if detection.status in {DETECTION_STATUS_SUCCESS, DETECTION_STATUS_NO_OBJECTS}:
            if hasattr(self.t5, "narrate_detection_structured"):
                generation = self.t5.narrate_detection_structured(summary)
                answer = generation.text
        else:
            warning = detection.error or "detection_failed"
            if warning not in warnings:
                warnings.append(warning)
            answer = "Object detection failed."

        return RunResult(
            input_text=text,
            final_answer=answer,
            status="completed" if detection.status in {DETECTION_STATUS_SUCCESS, DETECTION_STATUS_NO_OBJECTS} else "degraded",
            intent=intent,
            route=route,
            detection=detection,
            generation=generation,
            warnings=warnings,
        )

    def _run_chat(self, text: str, route: RouteDecision, intent: IntentResult) -> RunResult:
        if hasattr(self.t5, "chat_structured"):
            generation = self.t5.chat_structured(text)
            answer = generation.text
        else:
            started = time.perf_counter()
            answer = self.t5.chat(text)
            generation = generation_result_from_text(
                answer,
                model_name="t5",
                runtime="onnxruntime",
                device="cpu",
                prompt_type="chat",
                input_chars=len(text),
                max_new_tokens=getattr(self.t5, "max_new_chat", None),
                latency_ms=_elapsed_ms(started),
            )

        return RunResult(
            input_text=text,
            final_answer=answer,
            status="degraded" if generation.fallback_used else "completed",
            intent=intent,
            route=route,
            generation=generation,
        )

    def _run_rag(
        self,
        text: str,
        route: RouteDecision,
        intent: IntentResult,
        request_options: Mapping[str, Any],
    ) -> RunResult:
        use_internet = bool(request_options["use_internet"])
        web_only = bool(request_options["web_only"] or (intent.label == "chat" and use_internet))

        # --- Structured retrieval (preferred) ---
        if hasattr(self.rag, "retrieve_structured"):
            retrieval_started = time.perf_counter()
            retrieval = self.rag.retrieve_structured(
                text,
                use_internet=use_internet,
                web_only=web_only,
            )
            if retrieval.latency_ms is None:
                retrieval.latency_ms = _elapsed_ms(retrieval_started)

            # Build context from structured chunks
            if retrieval.used_context and retrieval.chunks:
                from services.rag import build_context_from_chunks
                ctx_str = build_context_from_chunks(
                    retrieval.chunks,
                    max_tokens=getattr(self.rag, "max_ctx_tokens", 512),
                    question=text,
                )
                contexts = [ctx_str] if ctx_str else []
            else:
                contexts = []
        else:
            # --- Legacy fallback ---
            retrieval_started = time.perf_counter()
            contexts, best_score, sources = self.rag.retrieve(
                text,
                use_internet=use_internet,
                web_only=web_only,
            )
            retrieval = retrieval_result_from_legacy(
                text,
                contexts,
                best_score,
                sources,
                top_k=getattr(self.rag, "top_k", None),
                threshold=getattr(self.rag, "thr", None),
                retrieval_mode="web" if web_only else "hybrid",
                fallback_used=not bool(contexts),
                fallback_reason=None if contexts else "no retrieval context",
                latency_ms=_elapsed_ms(retrieval_started),
            )

        generation_started = time.perf_counter()
        if contexts:
            if hasattr(self.t5, "answer_structured"):
                generation = self.t5.answer_structured(text, contexts)
                answer = generation.text
            else:
                answer = self.t5.answer(text, contexts)
                generation = generation_result_from_text(
                    answer,
                    model_name="t5",
                    runtime="onnxruntime",
                    device="cpu",
                    prompt_type="rag_answer",
                    input_chars=len(text),
                    max_new_tokens=getattr(self.t5, "max_new_rag", None),
                    latency_ms=_elapsed_ms(generation_started),
                )
            fallback_used = False
            fallback_reason = None
        else:
            if hasattr(self.t5, "answer_model_only_with_instruction_structured"):
                generation = self.t5.answer_model_only_with_instruction_structured(
                    text,
                    instruction=fallback_instruction(),
                )
                answer = generation.text
            else:
                answer = self.t5.answer_model_only_with_instruction(text, instruction=fallback_instruction())
                generation = generation_result_from_text(
                    answer,
                    model_name="t5",
                    runtime="onnxruntime",
                    device="cpu",
                    prompt_type="model_only",
                    input_chars=len(text),
                    max_new_tokens=getattr(self.t5, "max_new_chat", None),
                    latency_ms=_elapsed_ms(generation_started),
                )
            fallback_used = True
            fallback_reason = "no_retrieval_context"

        if fallback_used and not generation.fallback_used:
            generation.fallback_used = True
            generation.fallback_reason = fallback_reason

        status = "degraded" if route.fallback_used or fallback_used or generation.fallback_used else "completed"

        return RunResult(
            input_text=text,
            final_answer=answer,
            status=status,
            intent=intent,
            route=route,
            retrieval=retrieval,
            generation=generation,
        )
