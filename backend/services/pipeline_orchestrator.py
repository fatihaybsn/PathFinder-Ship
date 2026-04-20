from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Any, Mapping

from schemas.pipeline import (
    ClientAction,
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
from utils.text import fallback_instruction

logger = logging.getLogger(__name__)


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _metadata_bool(metadata: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = metadata.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


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
        t5: Any,
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
            warnings.append("Object detection requires an image from the client.")
            return RunResult(
                input_text=text,
                final_answer="Please capture or upload an image for object detection.",
                status="degraded",
                intent=intent,
                route=route,
                detection=DetectionResult(status="not_run"),
                client_action=ClientAction(
                    action="capture_photo",
                    reason="object detection requires an image",
                    requires_user_permission=True,
                ),
                warnings=warnings,
            )

        if self.yolo is None:
            warnings.append("Detection service is not available.")
            return RunResult(
                input_text=text,
                final_answer="Object detection is not available right now.",
                status="degraded",
                intent=intent,
                route=route,
                detection=DetectionResult(status="not_run"),
                warnings=warnings,
            )

        started = time.perf_counter()
        boxes, labels, scores, cls_ids = self.yolo.detect_from_bgr(image_bgr)
        detection = detection_result_from_legacy(
            labels,
            boxes,
            scores,
            cls_ids,
            model_name="yolo",
            latency_ms=_elapsed_ms(started),
        )
        summary = ", ".join(f"{count} {label}" for label, count in Counter(labels).items()) if labels else "no objects"

        return RunResult(
            input_text=text,
            final_answer=f"Object summary: {summary}",
            status="completed",
            intent=intent,
            route=route,
            detection=detection,
        )

    def _run_chat(self, text: str, route: RouteDecision, intent: IntentResult) -> RunResult:
        started = time.perf_counter()
        answer = self.t5.chat(text)
        generation = generation_result_from_text(
            answer,
            model_name="t5",
            runtime="onnx",
            prompt_type="chat",
            input_chars=len(text),
            latency_ms=_elapsed_ms(started),
        )

        return RunResult(
            input_text=text,
            final_answer=answer,
            status="completed",
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
            answer = self.t5.answer(text, contexts)
            fallback_used = False
            fallback_reason = None
            prompt_type = "rag"
        else:
            answer = self.t5.answer_model_only_with_instruction(text, instruction=fallback_instruction())
            fallback_used = True
            fallback_reason = "no retrieval context"
            prompt_type = "model_only"

        generation = generation_result_from_text(
            answer,
            model_name="t5",
            runtime="onnx",
            prompt_type=prompt_type,
            input_chars=len(text),
            latency_ms=_elapsed_ms(generation_started),
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
        )
        status = "degraded" if route.fallback_used or fallback_used else "completed"

        return RunResult(
            input_text=text,
            final_answer=answer,
            status=status,
            intent=intent,
            route=route,
            retrieval=retrieval,
            generation=generation,
        )
