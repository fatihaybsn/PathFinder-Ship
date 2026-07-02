from __future__ import annotations

import importlib.util
import logging
from collections.abc import Callable
from typing import Any

from services.observability.diagent_config import DiagentConfig, load_diagent_config

logger = logging.getLogger(__name__)

TracerFactory = Callable[..., Any]


def is_diagent_sdk_available() -> bool:
    """Return whether the Diagent package is importable without creating a tracer."""
    try:
        return importlib.util.find_spec("diagent") is not None
    except Exception:
        return False


def _load_tracer_factory() -> TracerFactory:
    from diagent.core.tracer import DiagentTracer

    return DiagentTracer


class DiagentSafeClient:
    """Fail-open bridge around DiagentTracer.

    The client is lazy by design: disabled mode does not import Diagent and does
    not create the underlying HTTP tracer.
    """

    def __init__(
        self,
        config: DiagentConfig | None = None,
        *,
        tracer_factory: TracerFactory | None = None,
        tracer_loader: Callable[[], TracerFactory] | None = None,
    ) -> None:
        self.config = config or load_diagent_config()
        self._tracer_factory = tracer_factory
        self._tracer_loader = tracer_loader or _load_tracer_factory
        self._tracer: Any | None = None
        self._load_failed = False
        self._sdk_available: bool | None = None
        self._warned_messages: set[str] = set()

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "DiagentSafeClient":
        return cls(DiagentConfig.from_mapping(cfg))

    @property
    def sdk_available(self) -> bool | None:
        return self._sdk_available

    def create_run(self, input_text: str = "", *, agent_name: str | None = None) -> str | None:
        return self._safe_call(
            "create_run",
            agent_name or self.config.agent_name,
            input_text or "",
        )

    def log_span(
        self,
        run_id: str | None,
        *,
        span_type: str,
        name: str,
        started_at: str | None = None,
        ended_at: str | None = None,
        duration_ms: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> str | None:
        if not run_id:
            return None
        return self._safe_call(
            "log_span",
            run_id,
            span_type=span_type,
            name=name,
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
            payload=payload,
        )

    def log_retrieval(
        self,
        run_id: str | None,
        *,
        query: str,
        retrieved_chunks: list[dict[str, Any]] | None = None,
        top_k: int = 5,
        source_age_hours: float | None = None,
    ) -> str | None:
        if not run_id:
            return None
        return self._safe_call(
            "log_retrieval",
            run_id,
            query=query,
            retrieved_chunks=retrieved_chunks,
            top_k=top_k,
            source_age_hours=source_age_hours,
        )

    def log_tool_call(
        self,
        run_id: str | None,
        *,
        tool_name: str,
        args: dict[str, Any] | None = None,
        status: str = "success",
        error: str | None = None,
        duration_ms: int | None = None,
    ) -> str | None:
        if not run_id:
            return None
        return self._safe_call(
            "log_tool_call",
            run_id,
            tool_name=tool_name,
            args=args,
            status=status,
            error=error,
            duration_ms=duration_ms,
        )

    def finish_run(
        self,
        run_id: str | None,
        *,
        output: str | None = None,
        status: str | None = None,
        error: str | None = None,
        total_tokens: int | None = None,
        cost_usd: float | None = None,
    ) -> dict[str, Any] | None:
        if not run_id:
            return None
        return self._safe_call(
            "finish_run",
            run_id,
            output=output,
            status=status,
            error=error,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
        )

    def close(self) -> None:
        tracer = self._tracer
        self._tracer = None
        if tracer is None:
            return
        try:
            close = getattr(tracer, "close", None)
            if callable(close):
                close()
        except Exception as exc:
            self._warn_once("Diagent close failed: %s", exc)

    def _safe_call(self, method_name: str, *args: Any, **kwargs: Any) -> Any | None:
        tracer = self._ensure_tracer()
        if tracer is None:
            return None

        try:
            method = getattr(tracer, method_name)
            return method(*args, **kwargs)
        except Exception as exc:
            self._warn_once("Diagent %s failed: %s", method_name, exc)
            return None

    def _ensure_tracer(self) -> Any | None:
        if not self.config.enabled:
            return None
        if self._tracer is not None:
            return self._tracer
        if self._load_failed:
            return None

        try:
            factory = self._tracer_factory or self._tracer_loader()
            self._sdk_available = True
            self._tracer = factory(
                base_url=self.config.api_url,
                timeout=self.config.timeout_seconds,
            )
            return self._tracer
        except Exception as exc:
            self._sdk_available = False
            self._load_failed = True
            self._warn_once("Diagent SDK unavailable; telemetry disabled: %s", exc)
            return None

    def _warn_once(self, message: str, *args: Any) -> None:
        key = message % args if args else message
        if key in self._warned_messages:
            return
        self._warned_messages.add(key)
        logger.warning(message, *args)
