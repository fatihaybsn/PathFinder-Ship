from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from uuid import UUID


EMPTY_VALUES = (None, "", [], {})
REQUEST_METADATA_KEYS = {
    "request_id",
    "conversation_id",
    "generation_provider",
    "web_enabled",
    "local_first_mode",
    "use_internet",
    "web_only",
}
SENSITIVE_KEY_PARTS = {
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "password",
    "secret",
    "token",
    "base64",
    "binary",
    "file_bytes",
    "image_bytes",
    "image_binary",
    "raw_image",
    "headers",
}


def safe_serialize(value: Any) -> Any:
    """Convert Pydantic/dataclass/nested values into JSON-ready primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (UUID, Path)):
        return str(value)
    if hasattr(value, "model_dump"):
        try:
            return safe_serialize(value.model_dump(mode="json"))
        except TypeError:
            return safe_serialize(value.model_dump())
    if hasattr(value, "dict"):
        return safe_serialize(value.dict())
    if is_dataclass(value):
        return safe_serialize(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): safe_serialize(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [safe_serialize(item) for item in value]

    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def clean_empty_fields(value: Any) -> Any:
    """Recursively remove null/empty fields while keeping 0 and False."""
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            cleaned_item = clean_empty_fields(item)
            if cleaned_item in EMPTY_VALUES:
                continue
            cleaned[str(key)] = cleaned_item
        return cleaned
    if isinstance(value, list):
        return [
            cleaned_item
            for item in value
            if (cleaned_item := clean_empty_fields(item)) not in EMPTY_VALUES
        ]
    return value


def _is_sensitive_key(key: Any) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return any(part in normalized for part in SENSITIVE_KEY_PARTS)


def sanitize_metadata(value: Any, *, max_string_chars: int = 500) -> Any:
    """Remove obvious secret/binary fields from nested metadata."""
    data = safe_serialize(value)
    if isinstance(data, Mapping):
        sanitized: dict[str, Any] = {}
        for key, item in data.items():
            if _is_sensitive_key(key):
                continue
            sanitized[str(key)] = sanitize_metadata(item, max_string_chars=max_string_chars)
        return clean_empty_fields(sanitized)
    if isinstance(data, list):
        return [
            item
            for item in (
                sanitize_metadata(item, max_string_chars=max_string_chars) for item in data
            )
            if item not in EMPTY_VALUES
        ]
    if isinstance(data, str):
        return truncate_text(data, max_string_chars)
    return data


def truncate_text(text: Any, max_chars: int = 1200) -> str:
    value = "" if text is None else str(text)
    max_chars = int(max_chars)
    if max_chars <= 0:
        return ""
    if len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return value[:max_chars]
    return f"{value[: max_chars - 3]}..."


def sanitize_retrieval_chunk(chunk: Any, *, max_chunk_chars: int = 1200) -> dict[str, Any]:
    data = safe_serialize(chunk)
    if not isinstance(data, Mapping):
        data = {"text": data}

    metadata = data.get("metadata") if isinstance(data.get("metadata"), Mapping) else {}
    text = data.get("text") or data.get("content") or data.get("chunk") or ""
    source = data.get("source") or metadata.get("source") or metadata.get("file_name")
    url = data.get("url") or metadata.get("url")

    sanitized: dict[str, Any] = {
        "text": truncate_text(text, max_chunk_chars),
        "source": source,
        "url": url,
        "score": data.get("score"),
        "rank": data.get("rank"),
        "retrieval_type": data.get("retrieval_type"),
    }

    cleaned_metadata = sanitize_metadata(metadata)
    if cleaned_metadata:
        sanitized["metadata"] = cleaned_metadata

    if not sanitized.get("source") and sanitized.get("url"):
        sanitized["source"] = sanitized["url"]

    return clean_empty_fields(sanitized)


def sanitize_retrieval_chunks(
    retrieval: Any,
    *,
    max_chunks: int = 5,
    max_chunk_chars: int = 1200,
) -> list[dict[str, Any]]:
    data = safe_serialize(retrieval)
    if isinstance(data, Mapping):
        chunks = data.get("chunks") or data.get("retrieved_chunks") or []
    elif isinstance(data, list):
        chunks = data
    else:
        chunks = []

    return [
        sanitize_retrieval_chunk(chunk, max_chunk_chars=max_chunk_chars)
        for chunk in list(chunks)[: max(0, int(max_chunks))]
    ]


def prepare_generation_metadata(generation: Any) -> dict[str, Any]:
    data = safe_serialize(generation)
    if not isinstance(data, Mapping):
        return {}

    runtime = data.get("runtime")
    device = data.get("device")
    runtime_text = str(runtime or "").lower()
    model_text = str(data.get("model_name") or "").lower()
    provider_family = None
    provider = None
    if "api" in runtime_text or device == "remote":
        provider_family = "api"
    elif runtime or data.get("model_name"):
        provider_family = "local"
    if "gemini" in runtime_text or "gemini" in model_text:
        provider = "gemini"
    elif "t5" in runtime_text or "t5" in model_text or "onnx" in runtime_text:
        provider = "local_t5"
    elif provider_family:
        provider = provider_family

    total_tokens = None
    input_tokens = data.get("input_tokens")
    output_tokens = data.get("output_tokens")
    if input_tokens is not None or output_tokens is not None:
        total_tokens = int(input_tokens or 0) + int(output_tokens or 0)

    return clean_empty_fields(
        {
            "provider": provider,
            "provider_family": provider_family,
            "model_name": data.get("model_name"),
            "runtime": runtime,
            "device": device,
            "prompt_type": data.get("prompt_type"),
            "input_chars": data.get("input_chars"),
            "output_chars": data.get("output_chars"),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "max_new_tokens": data.get("max_new_tokens"),
            "input_truncated": data.get("input_truncated"),
            "output_truncated": data.get("output_truncated"),
            "latency_ms": data.get("latency_ms"),
            "empty_output": data.get("empty_output"),
            "fallback_used": data.get("fallback_used"),
            "fallback_reason": data.get("fallback_reason"),
            "error": data.get("error"),
        }
    )


def prepare_request_metadata(
    request_metadata: Mapping[str, Any] | None = None,
    *,
    run_metadata: Mapping[str, Any] | None = None,
    app_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for source in (request_metadata or {}, run_metadata or {}):
        data = safe_serialize(source)
        if not isinstance(data, Mapping):
            continue
        for key in REQUEST_METADATA_KEYS:
            if key in data:
                payload[key] = data[key]
    if app_config:
        if "generation_provider" not in payload and app_config.get("GENERATION_PROVIDER") is not None:
            payload["generation_provider"] = app_config.get("GENERATION_PROVIDER")
        if "web_enabled" not in payload and app_config.get("ENABLE_WEB_SEARCH") is not None:
            payload["web_enabled"] = bool(app_config.get("ENABLE_WEB_SEARCH"))
    return clean_empty_fields(sanitize_metadata(payload))


def map_intent_span(run_result: Any) -> dict[str, Any] | None:
    data = safe_serialize(run_result)
    if not isinstance(data, Mapping):
        return None
    intent = data.get("intent")
    if not isinstance(intent, Mapping):
        return None

    payload = clean_empty_fields(
        {
            "label": intent.get("label"),
            "confidence": intent.get("confidence"),
            "accepted": intent.get("is_confident"),
            "threshold": intent.get("threshold"),
            "duration_ms": intent.get("latency_ms"),
            "model": intent.get("model"),
            "error": intent.get("error"),
        }
    )
    return {
        "span_type": "system",
        "name": "pathfindership.intent",
        "duration_ms": intent.get("latency_ms"),
        "payload": payload,
    }


def _retrieval_chunks(retrieval: Mapping[str, Any] | None) -> list[Any]:
    if not isinstance(retrieval, Mapping):
        return []
    chunks = retrieval.get("chunks")
    if chunks is None:
        chunks = retrieval.get("retrieved_chunks")
    return list(chunks or [])


def _sources_count(retrieval: Mapping[str, Any] | None) -> int:
    if not isinstance(retrieval, Mapping):
        return 0
    sources = retrieval.get("sources")
    if isinstance(sources, Sequence) and not isinstance(sources, (str, bytes, bytearray)):
        return len([source for source in sources if source])

    seen: set[str] = set()
    for chunk in _retrieval_chunks(retrieval):
        chunk_data = safe_serialize(chunk)
        if not isinstance(chunk_data, Mapping):
            continue
        metadata = chunk_data.get("metadata") if isinstance(chunk_data.get("metadata"), Mapping) else {}
        source = chunk_data.get("source") or chunk_data.get("url") or metadata.get("source") or metadata.get("url")
        if source:
            seen.add(str(source))
    return len(seen)


def _web_search_executed(retrieval: Mapping[str, Any] | None) -> bool:
    if not isinstance(retrieval, Mapping):
        return False
    mode = str(retrieval.get("retrieval_mode") or retrieval.get("mode") or "").lower()
    if "web" in mode:
        return True
    for chunk in _retrieval_chunks(retrieval):
        chunk_data = safe_serialize(chunk)
        if not isinstance(chunk_data, Mapping):
            continue
        metadata = chunk_data.get("metadata") if isinstance(chunk_data.get("metadata"), Mapping) else {}
        retrieval_type = str(chunk_data.get("retrieval_type") or metadata.get("retrieval_type") or "").lower()
        source = str(chunk_data.get("source") or chunk_data.get("url") or metadata.get("url") or "")
        if "web" in retrieval_type or source.startswith(("http://", "https://")):
            return True
    return False


def map_route_span(
    run_result: Any,
    *,
    request_metadata: Mapping[str, Any] | None = None,
    app_config: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    data = safe_serialize(run_result)
    if not isinstance(data, Mapping):
        return None
    route = data.get("route")
    if not isinstance(route, Mapping):
        return None

    retrieval = data.get("retrieval") if isinstance(data.get("retrieval"), Mapping) else None
    run_metadata = data.get("metadata") if isinstance(data.get("metadata"), Mapping) else {}
    request_payload = prepare_request_metadata(
        request_metadata,
        run_metadata=run_metadata,
        app_config=app_config,
    )
    best_score = retrieval.get("best_score") if retrieval else None
    if best_score is None and retrieval:
        scores = [
            chunk.get("score")
            for chunk in (safe_serialize(chunk) for chunk in _retrieval_chunks(retrieval))
            if isinstance(chunk, Mapping) and chunk.get("score") is not None
        ]
        best_score = max(scores) if scores else None

    payload = {
        "route": route.get("route"),
        "reason": route.get("reason"),
        "source_intent": route.get("source_intent"),
        "confidence": route.get("confidence"),
        "fallback_used": route.get("fallback_used"),
        "fallback_reason": route.get("fallback_reason"),
        "retrieval_required": route.get("route") == "rag",
        "web_search_required": bool(request_payload.get("use_internet") or request_payload.get("web_only")),
        "client_action_required": route.get("requires_client_action"),
        "client_action": route.get("client_action"),
        "retrieval_executed": retrieval is not None,
        "web_search_executed": _web_search_executed(retrieval),
        "sources_count": _sources_count(retrieval),
        "best_score": best_score,
        "local_retrieval_score": best_score,
        "web_fallback_threshold": (app_config or {}).get("RAG_WEB_MIN_STRENGTH"),
        **request_payload,
    }
    return {
        "span_type": "system",
        "name": "pathfindership.route_decision",
        "payload": clean_empty_fields(payload),
    }


def map_retrieval(run_result: Any, *, max_chunks: int = 5, max_chunk_chars: int = 1200) -> dict[str, Any] | None:
    data = safe_serialize(run_result)
    if not isinstance(data, Mapping):
        return None
    retrieval = data.get("retrieval")
    if not isinstance(retrieval, Mapping):
        return None

    chunks = sanitize_retrieval_chunks(
        retrieval,
        max_chunks=max_chunks,
        max_chunk_chars=max_chunk_chars,
    )
    top_k = retrieval.get("top_k")
    if top_k is None:
        top_k = len(_retrieval_chunks(retrieval))
    return {
        "query": str(retrieval.get("query") or data.get("input_text") or ""),
        "retrieved_chunks": chunks,
        "top_k": int(top_k or 0),
        "source_age_hours": retrieval.get("source_age_hours"),
    }


def map_generation_span(run_result: Any) -> dict[str, Any] | None:
    data = safe_serialize(run_result)
    if not isinstance(data, Mapping):
        return None
    generation = data.get("generation")
    if not isinstance(generation, Mapping):
        return None

    payload = prepare_generation_metadata(generation)
    return {
        "span_type": "llm_call",
        "name": "pathfindership.generation",
        "duration_ms": generation.get("latency_ms"),
        "payload": payload,
    }


def map_client_action_span(run_result: Any) -> dict[str, Any] | None:
    data = safe_serialize(run_result)
    if not isinstance(data, Mapping):
        return None
    client_action = data.get("client_action")
    if not isinstance(client_action, Mapping):
        return None

    payload = clean_empty_fields(
        {
            "action": client_action.get("action") or client_action.get("type"),
            "reason": client_action.get("reason"),
            "requires_frontend": True,
            "requires_user_permission": client_action.get("requires_user_permission"),
            "payload": sanitize_metadata(client_action.get("payload")),
        }
    )
    return {
        "span_type": "system",
        "name": "pathfindership.client_action",
        "payload": payload,
    }


def map_detection_tool(run_result: Any) -> dict[str, Any] | None:
    data = safe_serialize(run_result)
    if not isinstance(data, Mapping):
        return None
    detection = data.get("detection")
    if not isinstance(detection, Mapping):
        return None

    objects = detection.get("objects") if isinstance(detection.get("objects"), list) else []
    labels = [
        obj.get("label")
        for obj in objects
        if isinstance(obj, Mapping) and obj.get("label")
    ]
    confidences = [
        float(obj.get("confidence"))
        for obj in objects
        if isinstance(obj, Mapping) and obj.get("confidence") is not None
    ]
    status = str(detection.get("status") or "")
    is_success = status in {"success", "no_objects"}
    args = clean_empty_fields(
        {
            "status": status,
            "labels": labels,
            "detection_count": len(objects),
            "max_confidence": max(confidences) if confidences else None,
            "image_source": detection.get("image_source"),
            "model_name": detection.get("model_name"),
        }
    )
    return {
        "tool_name": "pathfindership.yolo.detect",
        "args": args,
        "status": "success" if is_success else "error",
        "error": detection.get("error") if not is_success else None,
        "duration_ms": detection.get("latency_ms"),
    }


def emit_run_result_telemetry(
    client: Any,
    run_id: str | None,
    run_result: Any,
    *,
    config: Any | None = None,
    request_metadata: Mapping[str, Any] | None = None,
    app_config: Mapping[str, Any] | None = None,
) -> None:
    """Emit best-effort Diagent telemetry for a completed PathFinder run."""
    if not run_id:
        return

    max_chunks = int(getattr(config, "max_retrieval_chunks", 5))
    max_chunk_chars = int(getattr(config, "max_chunk_chars", 1200))

    for span in (
        map_intent_span(run_result),
        map_route_span(
            run_result,
            request_metadata=request_metadata,
            app_config=app_config,
        ),
    ):
        if span:
            client.log_span(run_id, **span)

    retrieval = map_retrieval(
        run_result,
        max_chunks=max_chunks,
        max_chunk_chars=max_chunk_chars,
    )
    if retrieval:
        client.log_retrieval(run_id, **retrieval)

    for span in (
        map_generation_span(run_result),
        map_client_action_span(run_result),
    ):
        if span:
            client.log_span(run_id, **span)

    detection_tool = map_detection_tool(run_result)
    if detection_tool:
        client.log_tool_call(run_id, **detection_tool)
