from __future__ import annotations

from typing import Any, Mapping

from schemas.pipeline import IntentResult, RouteDecision

DEFAULT_INTENT_THRESHOLD = 0.60
FALLBACK_ROUTE = "chat"

CAMERA_ACTIONS: dict[str, tuple[str, bool]] = {
    "open_camera": ("open_camera", True),
    "close_camera": ("close_camera", False),
    "take_photo": ("capture_photo", True),
    "capture_photo": ("capture_photo", True),
}

INTENT_ALIASES: dict[str, str] = {
    "open_cam": "open_camera",
    "camera_open": "open_camera",
    "close_cam": "close_camera",
    "camera_close": "close_camera",
    "photo": "capture_photo",
    "take_picture": "capture_photo",
    "capture_picture": "capture_photo",
    "detect": "object_detect",
    "object_detection": "object_detect",
    "detection": "object_detect",
    "rag": "rag",
    "document_question": "rag",
    "document_qa": "rag",
    "knowledge_query": "rag",
    "knowledge_question": "rag",
    "smalltalk": "chat",
}

DETECTION_LABELS = {"object_detect"}
RAG_LABELS = {"rag"}
CHAT_LABELS = {"chat"}
UNKNOWN_LABELS = {"unknown", "none", "other", ""}


def normalize_intent_label(label: str | None) -> str:
    normalized = (label or "").strip().lower().replace("-", "_").replace(" ", "_")
    return INTENT_ALIASES.get(normalized, normalized)


def _metadata_bool(metadata: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = metadata.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _metadata_float(metadata: Mapping[str, Any], key: str) -> float | None:
    value = metadata.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _threshold_for(intent: IntentResult, metadata: Mapping[str, Any]) -> float:
    if intent.threshold is not None:
        return float(intent.threshold)
    return (
        _metadata_float(metadata, "intent_threshold")
        or _metadata_float(metadata, "CLS_ROUTE_THRESHOLD")
        or DEFAULT_INTENT_THRESHOLD
    )


def _is_confident(intent: IntentResult, threshold: float) -> bool:
    if intent.is_confident is not None:
        return bool(intent.is_confident)
    if intent.confidence is None:
        return False
    return float(intent.confidence) >= threshold


def _fallback(
    *,
    intent: IntentResult,
    reason: str,
    fallback_reason: str,
    route: str = FALLBACK_ROUTE,
    client_action: str | None = None,
    requires_client_action: bool = False,
) -> RouteDecision:
    return RouteDecision(
        route=route,
        reason=reason,
        source_intent=intent.label,
        confidence=intent.confidence,
        requires_client_action=requires_client_action,
        client_action=client_action,
        fallback_used=True,
        fallback_reason=fallback_reason,
    )


def decide_route(
    message: str,
    intent: IntentResult,
    metadata: Mapping[str, Any] | None = None,
) -> RouteDecision:
    """
    IntentResult + request context -> backend route/client action decision.

    Bu katman karar verir; RAG/T5/YOLO/kamera/e-posta gibi yan etkili işleri
    çalıştırmaz.
    """
    del message  # Future rules can inspect text; current rules are intent driven.
    metadata = metadata or {}
    label = normalize_intent_label(intent.label)
    threshold = _threshold_for(intent, metadata)

    if intent.error:
        return _fallback(
            intent=intent,
            reason="classifier error fallback",
            fallback_reason="classifier_error",
        )

    if not _is_confident(intent, threshold):
        return _fallback(
            intent=intent,
            reason="intent confidence below threshold",
            fallback_reason="low_intent_confidence",
        )

    if label in UNKNOWN_LABELS:
        return _fallback(
            intent=intent,
            reason="unknown intent fallback",
            fallback_reason="unknown_intent",
        )

    if label in CAMERA_ACTIONS:
        action, _ = CAMERA_ACTIONS[label]
        return RouteDecision(
            route="camera_action",
            reason=f"confident {label} intent",
            source_intent=intent.label,
            confidence=intent.confidence,
            requires_client_action=True,
            client_action=action,
        )

    if label in DETECTION_LABELS:
        has_image = _metadata_bool(metadata, "has_image") or _metadata_bool(metadata, "image_available")
        if not has_image:
            return _fallback(
                intent=intent,
                reason="object detection requires an image from the client",
                fallback_reason="missing_image_for_detection",
                route="detect",
                client_action="capture_photo",
                requires_client_action=True,
            )

        return RouteDecision(
            route="detect",
            reason=f"confident {label} intent",
            source_intent=intent.label,
            confidence=intent.confidence,
        )

    if label in RAG_LABELS:
        return RouteDecision(
            route="rag",
            reason=f"confident {label} intent",
            source_intent=intent.label,
            confidence=intent.confidence,
        )

    if label in CHAT_LABELS:
        if _metadata_bool(metadata, "use_internet") or _metadata_bool(metadata, "web_only"):
            return RouteDecision(
                route="rag",
                reason="metadata requested retrieval for chat intent",
                source_intent=intent.label,
                confidence=intent.confidence,
            )

        return RouteDecision(
            route="chat",
            reason=f"confident {label} intent",
            source_intent=intent.label,
            confidence=intent.confidence,
        )

    return _fallback(
        intent=intent,
        reason="unsupported intent fallback",
        fallback_reason="unsupported_route",
    )
