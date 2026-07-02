from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from uuid import UUID


EMPTY_VALUES = (None, "", [], {})


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

    cleaned_metadata = clean_empty_fields(metadata)
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
    provider_family = None
    if "api" in runtime_text or device == "remote":
        provider_family = "api"
    elif runtime or data.get("model_name"):
        provider_family = "local"

    total_tokens = None
    input_tokens = data.get("input_tokens")
    output_tokens = data.get("output_tokens")
    if input_tokens is not None or output_tokens is not None:
        total_tokens = int(input_tokens or 0) + int(output_tokens or 0)

    return clean_empty_fields(
        {
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
