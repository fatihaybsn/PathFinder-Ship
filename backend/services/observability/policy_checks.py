from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

from schemas.pipeline import to_serializable_dict


PolicySeverity = Literal["info", "warning", "critical"]
PolicyStatus = Literal["passed", "failed"]

YOLO_TOOL_NAME = "pathfindership.yolo.detect"
EXPECTED_OPERATIONS_BY_ROUTE: dict[str, dict[str, Any]] = {
    "chat": {
        "retrieval_expected": False,
        "yolo_expected": False,
    },
    "rag": {
        "retrieval_expected": True,
        "yolo_expected": False,
    },
    "detect": {
        "retrieval_expected": False,
        "yolo_expected": "image_dependent",
    },
    "camera_action": {
        "retrieval_expected": False,
        "yolo_expected": False,
        "client_action_expected": True,
    },
}


class PolicyViolation(BaseModel):
    code: str
    severity: PolicySeverity
    message: str
    expected: dict[str, Any] = Field(default_factory=dict)
    actual: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)


class PolicyCheckResult(BaseModel):
    status: PolicyStatus
    violation_count: int = 0
    violations: list[PolicyViolation] = Field(default_factory=list)
    checks: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    diagnosis_visibility: str = "span_recorded_not_loaded_by_current_diagent_diagnostician"


def evaluate_policy(
    run_result: Any,
    derived_metadata: Mapping[str, Any] | None = None,
    app_config: Mapping[str, Any] | None = None,
    actual_tools: Sequence[str] | None = None,
) -> PolicyCheckResult:
    """Compare expected PathFinderShip behavior with observed telemetry evidence."""
    data = _as_mapping(to_serializable_dict(run_result))
    derived = _as_mapping(derived_metadata)
    config = _as_mapping(app_config)
    tools = [str(tool) for tool in (actual_tools or []) if tool]

    route = _route_name(data)
    intent = _intent_label(data)
    retrieval = _as_optional_mapping(data.get("retrieval"))
    generation = _as_optional_mapping(data.get("generation"))
    detection = _as_optional_mapping(data.get("detection"))
    client_action = _as_optional_mapping(data.get("client_action"))
    run_metadata = _as_mapping(data.get("metadata"))
    final_answer_present = _has_text(data.get("final_answer"))

    retrieval_executed = _derived_bool(
        derived,
        "retrieval_executed",
        default=retrieval is not None,
    )
    retrieval_chunks = _retrieval_chunks(retrieval)
    retrieval_has_chunks = bool(retrieval_chunks)
    sources_count = _sources_count(retrieval, derived)
    checks: dict[str, Any] = {}
    violations: list[PolicyViolation] = []
    codes: set[str] = set()

    def add_violation(violation: PolicyViolation) -> None:
        if violation.code in codes:
            checks[violation.code] = "deduped"
            return
        codes.add(violation.code)
        violations.append(violation)
        checks[violation.code] = "failed"

    # Primary retrieval precedence: no retrieval record on a RAG route is the clearest
    # route-level policy failure. Empty retrieval rows are left to Diagent's built-in
    # empty_retrieval detector to avoid duplicate alert/span noise.
    if route == "rag":
        if retrieval is None:
            add_violation(
                PolicyViolation(
                    code="rag_route_without_retrieval",
                    severity="warning",
                    message="RAG route was selected but no retrieval result was recorded.",
                    expected={"route": "rag", "retrieval": "present"},
                    actual={"route": route, "retrieval_executed": retrieval_executed},
                    evidence={
                        "intent": intent,
                        "route_reason": _route_reason(data),
                        "sources_count": sources_count,
                    },
                )
            )
        elif not retrieval_has_chunks:
            checks["rag_route_without_retrieval"] = "skipped_empty_retrieval_detector_covers"
            checks["empty_retrieval"] = "delegated_to_diagent_empty_retrieval_detector"
        else:
            checks["rag_route_without_retrieval"] = "passed"
    else:
        checks["rag_route_without_retrieval"] = "skipped_route_not_rag"

    if "retrieval_required" in derived and _bool_value(derived.get("retrieval_required")) and not retrieval_executed:
        if "rag_route_without_retrieval" in codes:
            checks["missing_required_retrieval"] = "deduped_by_rag_route_without_retrieval"
        else:
            add_violation(
                PolicyViolation(
                    code="missing_required_retrieval",
                    severity="warning",
                    message="Route required retrieval but no retrieval was executed.",
                    expected={"retrieval_required": True},
                    actual={
                        "route": route,
                        "retrieval_executed": retrieval_executed,
                    },
                    evidence={
                        "intent": intent,
                        "route_reason": _route_reason(data),
                        "sources_count": sources_count,
                    },
                )
            )
    elif "retrieval_required" not in derived:
        checks["missing_required_retrieval"] = "skipped_missing_retrieval_required_signal"
    else:
        checks["missing_required_retrieval"] = "passed"

    if route == "rag" and retrieval_has_chunks and final_answer_present:
        if sources_count == 0:
            add_violation(
                PolicyViolation(
                    code="rag_answer_without_sources",
                    severity="warning",
                    message="RAG answer was produced from retrieved chunks with no backend source evidence.",
                    expected={"sources_count": ">0"},
                    actual={"route": route, "sources_count": sources_count},
                    evidence={
                        "chunk_count": len(retrieval_chunks),
                        "retrieval_mode": retrieval.get("retrieval_mode") if retrieval else None,
                        "intent": intent,
                    },
                )
            )
        else:
            checks["rag_answer_without_sources"] = "passed"
    else:
        checks["rag_answer_without_sources"] = "skipped_not_applicable"

    _evaluate_missing_web_fallback(
        route=route,
        retrieval=retrieval,
        derived=derived,
        config=config,
        checks=checks,
        add_violation=add_violation,
        intent=intent,
        route_reason=_route_reason(data),
        sources_count=sources_count,
    )

    if route in {"chat", "rag"} and YOLO_TOOL_NAME in tools:
        add_violation(
            PolicyViolation(
                code="unexpected_yolo_for_route",
                severity="warning",
                message="YOLO detection ran on a route that does not expect vision tools.",
                expected={"route": route, "yolo_expected": False},
                actual={"actual_tools": tools},
                evidence={
                    "intent": intent,
                    "correlation_id": _correlation_id(run_metadata, client_action, detection),
                },
            )
        )
    else:
        checks["unexpected_yolo_for_route"] = "passed" if YOLO_TOOL_NAME in tools else "skipped_no_yolo_tool"

    image_received = _first_bool(
        derived.get("image_received"),
        run_metadata.get("image_received"),
    )
    yolo_attempted = _first_bool(
        derived.get("yolo_attempted"),
        run_metadata.get("yolo_attempted"),
        YOLO_TOOL_NAME in tools if YOLO_TOOL_NAME in tools else None,
    )
    if yolo_attempted is True and image_received is False:
        add_violation(
            PolicyViolation(
                code="detection_without_image",
                severity="critical",
                message="YOLO detection was attempted without a received image.",
                expected={"image_received": True},
                actual={"image_received": False, "yolo_attempted": True},
                evidence={
                    "route": route,
                    "intent": intent,
                    "correlation_id": _correlation_id(run_metadata, client_action, detection),
                },
            )
        )
    elif yolo_attempted is None or image_received is None:
        checks["detection_without_image"] = "skipped_missing_evidence"
    else:
        checks["detection_without_image"] = "passed"

    generation_empty = bool(generation and _bool_value(generation.get("empty_output")))
    generation_fallback = bool(generation and _bool_value(generation.get("fallback_used")))
    if generation_empty:
        severity: PolicySeverity = "warning"
        if not generation_fallback and not final_answer_present:
            severity = "critical"
        add_violation(
            PolicyViolation(
                code="generation_empty_output",
                severity=severity,
                message="Generation provider reported an empty output.",
                expected={"generation.empty_output": False},
                actual={
                    "generation.empty_output": True,
                    "fallback_used": generation_fallback,
                    "final_answer_present": final_answer_present,
                },
                evidence={
                    "route": route,
                    "prompt_type": generation.get("prompt_type") if generation else None,
                    "provider": generation.get("runtime") if generation else None,
                    "fallback_reason": generation.get("fallback_reason") if generation else None,
                },
            )
        )
    else:
        checks["generation_empty_output"] = "passed" if generation else "skipped_no_generation"

    if _should_check_final_answer(data, route, client_action) and not final_answer_present:
        if "generation_empty_output" in codes:
            checks["missing_final_answer"] = "deduped_by_generation_empty_output"
        else:
            add_violation(
                PolicyViolation(
                    code="missing_final_answer",
                    severity="critical",
                    message="Run completed without a final answer.",
                    expected={"final_answer": "non_empty"},
                    actual={"final_answer_present": False, "status": data.get("status")},
                    evidence={"route": route, "intent": intent},
                )
            )
    else:
        checks["missing_final_answer"] = "passed_or_skipped"

    summary = {
        "route": route,
        "intent": intent,
        "actual_tools": tools,
        "retrieval_executed": retrieval_executed,
        "retrieval_chunk_count": len(retrieval_chunks),
        "sources_count": sources_count,
        "final_answer_present": final_answer_present,
        "expected_operations": EXPECTED_OPERATIONS_BY_ROUTE.get(route, {}),
    }

    return PolicyCheckResult(
        status="failed" if violations else "passed",
        violation_count=len(violations),
        violations=violations,
        checks=_clean(checks),
        summary=_clean(summary),
    )


def _evaluate_missing_web_fallback(
    *,
    route: str | None,
    retrieval: Mapping[str, Any] | None,
    derived: Mapping[str, Any],
    config: Mapping[str, Any],
    checks: dict[str, Any],
    add_violation: Any,
    intent: str | None,
    route_reason: str | None,
    sources_count: int,
) -> None:
    if route != "rag":
        checks["missing_web_fallback"] = "skipped_route_not_applicable"
        return

    evidence: dict[str, Any] = {}
    missing: list[str] = []

    web_enabled = _first_bool(
        derived.get("web_enabled"),
        derived.get("web_search_enabled"),
        config.get("ENABLE_WEB_SEARCH"),
    )
    if web_enabled is None:
        missing.append("web_search_enabled")
    else:
        evidence["web_search_enabled"] = web_enabled

    web_search_required = _first_bool(
        derived.get("web_search_required"),
        derived.get("use_internet"),
        derived.get("web_only"),
    )
    if web_search_required is None:
        missing.append("web_search_required")
    else:
        evidence["web_search_required"] = web_search_required

    local_score = _first_float(
        derived.get("local_retrieval_score"),
        derived.get("best_score"),
        retrieval.get("best_score") if retrieval else None,
    )
    if local_score is None:
        missing.append("local_retrieval_score")
    else:
        evidence["local_retrieval_score"] = local_score

    threshold = _first_float(
        derived.get("web_fallback_threshold"),
        config.get("RAG_WEB_MIN_STRENGTH"),
    )
    if threshold is None:
        missing.append("web_fallback_threshold")
    else:
        evidence["web_fallback_threshold"] = threshold

    web_executed = _first_bool(
        derived.get("web_search_executed"),
        _web_search_executed(retrieval),
    )
    if web_executed is None:
        missing.append("web_search_executed")
    else:
        evidence["web_search_executed"] = web_executed

    if missing:
        checks["missing_web_fallback"] = {
            "status": "skipped_missing_evidence",
            "missing": missing,
        }
        return

    if not web_enabled:
        checks["missing_web_fallback"] = "skipped_web_search_disabled"
        return
    if not web_search_required:
        checks["missing_web_fallback"] = "skipped_web_search_not_requested"
        return
    if local_score >= threshold:
        checks["missing_web_fallback"] = "passed_local_score_not_weak"
        return
    if web_executed:
        checks["missing_web_fallback"] = "passed_web_executed"
        return

    add_violation(
        PolicyViolation(
            code="missing_web_fallback",
            severity="warning",
            message="Local retrieval was weak and web fallback evidence indicates web search did not run.",
            expected={
                "web_search_enabled": True,
                "web_search_required": True,
                "web_search_executed": True,
            },
            actual=evidence,
            evidence={
                "intent": intent,
                "route_reason": route_reason,
                "sources_count": sources_count,
            },
        )
    )


def _as_mapping(value: Any) -> dict[str, Any]:
    data = to_serializable_dict(value)
    return dict(data) if isinstance(data, Mapping) else {}


def _as_optional_mapping(value: Any) -> dict[str, Any] | None:
    data = to_serializable_dict(value)
    return dict(data) if isinstance(data, Mapping) else None


def _route_name(data: Mapping[str, Any]) -> str | None:
    route = _as_optional_mapping(data.get("route"))
    value = route.get("route") if route else None
    return str(value) if value else None


def _route_reason(data: Mapping[str, Any]) -> str | None:
    route = _as_optional_mapping(data.get("route"))
    value = route.get("reason") if route else None
    return str(value) if value else None


def _intent_label(data: Mapping[str, Any]) -> str | None:
    intent = _as_optional_mapping(data.get("intent"))
    value = intent.get("label") if intent else None
    return str(value) if value else None


def _retrieval_chunks(retrieval: Mapping[str, Any] | None) -> list[Any]:
    if not retrieval:
        return []
    chunks = retrieval.get("chunks")
    if chunks is None:
        chunks = retrieval.get("retrieved_chunks")
    return list(chunks or [])


def _sources_count(retrieval: Mapping[str, Any] | None, derived: Mapping[str, Any]) -> int:
    if "sources_count" in derived and derived.get("sources_count") is not None:
        try:
            return int(derived.get("sources_count") or 0)
        except (TypeError, ValueError):
            pass

    if not retrieval:
        return 0

    sources = retrieval.get("sources")
    if isinstance(sources, Sequence) and not isinstance(sources, (str, bytes, bytearray)):
        return len([source for source in sources if source])

    seen: set[str] = set()
    for chunk in _retrieval_chunks(retrieval):
        chunk_data = _as_mapping(chunk)
        metadata = _as_mapping(chunk_data.get("metadata"))
        source = (
            chunk_data.get("source")
            or chunk_data.get("url")
            or metadata.get("source")
            or metadata.get("url")
            or metadata.get("file_name")
        )
        if source:
            seen.add(str(source))
    return len(seen)


def _web_search_executed(retrieval: Mapping[str, Any] | None) -> bool | None:
    if retrieval is None:
        return None
    mode = str(retrieval.get("retrieval_mode") or retrieval.get("mode") or "").lower()
    if "web" in mode:
        return True
    for chunk in _retrieval_chunks(retrieval):
        chunk_data = _as_mapping(chunk)
        metadata = _as_mapping(chunk_data.get("metadata"))
        retrieval_type = str(chunk_data.get("retrieval_type") or metadata.get("retrieval_type") or "").lower()
        source = str(chunk_data.get("source") or chunk_data.get("url") or metadata.get("url") or "")
        if "web" in retrieval_type or source.startswith(("http://", "https://")):
            return True
    return False


def _has_text(value: Any) -> bool:
    return bool(str(value or "").strip())


def _bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _derived_bool(metadata: Mapping[str, Any], key: str, *, default: bool) -> bool:
    if key not in metadata:
        return default
    return _bool_value(metadata.get(key))


def _first_bool(*values: Any) -> bool | None:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off"}:
                return False
            continue
        return bool(value)
    return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _correlation_id(
    run_metadata: Mapping[str, Any],
    client_action: Mapping[str, Any] | None,
    detection: Mapping[str, Any] | None,
) -> str | None:
    if run_metadata.get("correlation_id"):
        return str(run_metadata.get("correlation_id"))
    if client_action:
        payload = _as_mapping(client_action.get("payload"))
        if payload.get("correlation_id"):
            return str(payload.get("correlation_id"))
    if detection:
        metadata = _as_mapping(detection.get("metadata"))
        if metadata.get("correlation_id"):
            return str(metadata.get("correlation_id"))
    return None


def _should_check_final_answer(
    data: Mapping[str, Any],
    route: str | None,
    client_action: Mapping[str, Any] | None,
) -> bool:
    status = str(data.get("status") or "").lower()
    if status in {"failed", "error"}:
        return False
    if route == "camera_action":
        return False
    if client_action:
        action = str(client_action.get("action") or client_action.get("type") or "")
        if action in {"capture_photo", "open_camera", "close_camera"}:
            return False
    return status in {"completed", "degraded", "finished"}


def _clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            cleaned_item = _clean(item)
            if cleaned_item in (None, "", [], {}):
                continue
            cleaned[str(key)] = cleaned_item
        return cleaned
    if isinstance(value, list):
        return [
            cleaned_item
            for item in value
            if (cleaned_item := _clean(item)) not in (None, "", [], {})
        ]
    return value
