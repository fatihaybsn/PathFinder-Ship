from __future__ import annotations

from typing import Any, Mapping, Sequence

from pydantic import BaseModel, Field


DETECTION_STATUS_SUCCESS = "success"
DETECTION_STATUS_NO_OBJECTS = "no_objects"
DETECTION_STATUS_INVALID_IMAGE = "invalid_image"
DETECTION_STATUS_MODEL_ERROR = "model_error"
DETECTION_STATUS_FAILED = "failed"
DETECTION_STATUS_NOT_RUN = "not_run"
DETECTION_STATUSES = {
    DETECTION_STATUS_SUCCESS,
    DETECTION_STATUS_NO_OBJECTS,
    DETECTION_STATUS_INVALID_IMAGE,
    DETECTION_STATUS_MODEL_ERROR,
    DETECTION_STATUS_FAILED,
    DETECTION_STATUS_NOT_RUN,
}


class IntentResult(BaseModel):
    label: str
    confidence: float | None = None
    threshold: float | None = None
    is_confident: bool | None = None
    raw_scores: dict[str, float] | None = None
    latency_ms: int | None = None
    error: str | None = None


class RouteDecision(BaseModel):
    route: str
    reason: str | None = None
    source_intent: str | None = None
    confidence: float | None = None
    requires_client_action: bool = False
    client_action: str | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None


class RetrievedChunk(BaseModel):
    text: str
    source: str | None = None
    score: float | None = None
    rank: int | None = None
    retrieval_type: str | None = None
    metadata: dict[str, Any] | None = None


class RetrievalResult(BaseModel):
    query: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    top_k: int | None = None
    best_score: float | None = None
    threshold: float | None = None
    used_context: bool = False
    retrieval_mode: str | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None
    latency_ms: int | None = None
    error: str | None = None
    web_search_attempted: bool = False
    web_search_status: str | None = None
    web_candidate_count: int = 0
    web_error_type: str | None = None


class IndexingResult(BaseModel):
    filename: str
    document_id: str | None = None
    saved_path: str | None = None
    indexed: bool = False
    indexed_chunk_count: int = 0
    skipped: bool = False
    skip_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] | None = None


class GenerationResult(BaseModel):
    text: str
    model_name: str | None = None
    runtime: str | None = None
    device: str | None = None
    prompt_type: str | None = None
    input_chars: int | None = None
    output_chars: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    max_new_tokens: int | None = None
    input_truncated: bool | None = None
    output_truncated: bool | None = None
    latency_ms: int | None = None
    empty_output: bool = False
    fallback_used: bool = False
    fallback_reason: str | None = None
    error: str | None = None


class DetectionObject(BaseModel):
    label: str
    confidence: float | None = None
    bbox: list[float] | None = None
    metadata: dict[str, Any] | None = None


class DetectionResult(BaseModel):
    objects: list[DetectionObject] = Field(default_factory=list)
    image_source: str | None = None
    model_name: str | None = None
    latency_ms: int | None = None
    status: str = "not_run"
    error: str | None = None
    metadata: dict[str, Any] | None = None


class ClientAction(BaseModel):
    action: str
    reason: str | None = None
    payload: dict[str, Any] | None = None
    requires_user_permission: bool = False


class RunResult(BaseModel):
    input_text: str
    final_answer: str | None = None
    status: str
    intent: IntentResult | None = None
    route: RouteDecision | None = None
    retrieval: RetrievalResult | None = None
    generation: GenerationResult | None = None
    detection: DetectionResult | None = None
    client_action: ClientAction | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None
    duration_ms: int | None = None


def to_serializable_dict(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except TypeError:
            return value.model_dump()

    if hasattr(value, "dict"):
        return value.dict()

    if isinstance(value, Mapping):
        return {str(key): to_serializable_dict(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_serializable_dict(item) for item in value]

    return value


def intent_result_from_prediction(
    label: str,
    confidence: float | None,
    *,
    threshold: float | None = None,
    raw_scores: dict[str, float] | None = None,
    latency_ms: int | None = None,
    error: str | None = None,
) -> IntentResult:
    is_confident = None
    if confidence is not None and threshold is not None:
        is_confident = float(confidence) >= float(threshold)

    return IntentResult(
        label=label,
        confidence=confidence,
        threshold=threshold,
        is_confident=is_confident,
        raw_scores=raw_scores,
        latency_ms=latency_ms,
        error=error,
    )


def retrieval_result_from_legacy(
    query: str,
    contexts: Sequence[str] | None,
    best_score: float | None,
    sources: Sequence[str] | None = None,
    *,
    top_k: int | None = None,
    threshold: float | None = None,
    retrieval_mode: str | None = None,
    fallback_used: bool = False,
    fallback_reason: str | None = None,
    latency_ms: int | None = None,
    error: str | None = None,
) -> RetrievalResult:
    context_list = list(contexts or [])
    source_list = list(sources or [])
    chunks = [
        RetrievedChunk(
            text=text,
            source=source_list[index] if index < len(source_list) else None,
            score=best_score if index == 0 else None,
            rank=index + 1,
            retrieval_type=retrieval_mode,
        )
        for index, text in enumerate(context_list)
    ]

    return RetrievalResult(
        query=query,
        chunks=chunks,
        sources=source_list,
        top_k=top_k,
        best_score=best_score,
        threshold=threshold,
        used_context=bool(chunks),
        retrieval_mode=retrieval_mode,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        latency_ms=latency_ms,
        error=error,
    )


def retrieval_result_empty(
    query: str,
    *,
    top_k: int | None = None,
    best_score: float | None = None,
    threshold: float | None = None,
    retrieval_mode: str = "empty",
    fallback_reason: str = "empty_retrieval",
    latency_ms: int | None = None,
    error: str | None = None,
) -> RetrievalResult:
    """Convenience constructor for empty or failed retrieval results."""
    return RetrievalResult(
        query=query,
        chunks=[],
        sources=[],
        top_k=top_k,
        best_score=best_score,
        threshold=threshold,
        used_context=False,
        retrieval_mode=retrieval_mode,
        fallback_used=True,
        fallback_reason=fallback_reason,
        latency_ms=latency_ms,
        error=error,
    )


def generation_result_from_text(
    text: str | None,
    *,
    model_name: str | None = None,
    runtime: str | None = None,
    device: str | None = None,
    prompt_type: str | None = None,
    input_chars: int | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    max_new_tokens: int | None = None,
    input_truncated: bool | None = None,
    output_truncated: bool | None = None,
    latency_ms: int | None = None,
    fallback_used: bool = False,
    fallback_reason: str | None = None,
    error: str | None = None,
) -> GenerationResult:
    normalized_text = text or ""

    return GenerationResult(
        text=normalized_text,
        model_name=model_name,
        runtime=runtime,
        device=device,
        prompt_type=prompt_type,
        input_chars=input_chars,
        output_chars=len(normalized_text),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        max_new_tokens=max_new_tokens,
        input_truncated=input_truncated,
        output_truncated=output_truncated,
        latency_ms=latency_ms,
        empty_output=not bool(normalized_text.strip()),
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        error=error,
    )


def detection_result_from_legacy(
    labels: Sequence[str] | None,
    boxes: Sequence[Sequence[float]] | None = None,
    scores: Sequence[float] | None = None,
    cls_ids: Sequence[int] | None = None,
    *,
    image_source: str | None = None,
    model_name: str | None = None,
    latency_ms: int | None = None,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
    status: str | None = None,
) -> DetectionResult:
    label_list = list(labels or [])
    box_list = list(boxes or [])
    score_list = list(scores or [])
    cls_id_list = list(cls_ids or [])

    objects = []
    for index, label in enumerate(label_list):
        object_metadata: dict[str, Any] = {}
        if index < len(cls_id_list):
            object_metadata["class_id"] = cls_id_list[index]
        if index < len(score_list):
            object_metadata["raw_score"] = score_list[index]

        objects.append(
            DetectionObject(
                label=label,
                confidence=score_list[index] if index < len(score_list) else None,
                bbox=[float(value) for value in box_list[index]] if index < len(box_list) else None,
                metadata=object_metadata or None,
            )
        )

    resolved_status = status
    if resolved_status is None:
        resolved_status = DETECTION_STATUS_MODEL_ERROR if error else (
            DETECTION_STATUS_SUCCESS if objects else DETECTION_STATUS_NO_OBJECTS
        )

    return DetectionResult(
        objects=objects,
        image_source=image_source,
        model_name=model_name,
        latency_ms=latency_ms,
        status=resolved_status,
        error=error,
        metadata=metadata,
    )
