from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "off"}


def _value(cfg: Mapping[str, Any], key: str, default: Any) -> Any:
    if key in cfg and cfg[key] is not None:
        return cfg[key]
    lower_key = key.lower()
    if lower_key in cfg and cfg[lower_key] is not None:
        return cfg[lower_key]
    return default


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class DiagentConfig:
    enabled: bool = False
    api_url: str = "http://localhost:8000"
    agent_name: str = "pathfindership"
    timeout_seconds: float = 5.0
    fail_open: bool = True
    log_policy_spans: bool = True
    max_chunk_chars: int = 1200
    max_retrieval_chunks: int = 5

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | None = None) -> "DiagentConfig":
        cfg = cfg or {}
        return cls(
            enabled=_as_bool(_value(cfg, "DIAGENT_ENABLED", False), False),
            api_url=str(_value(cfg, "DIAGENT_API_URL", "http://localhost:8000")).rstrip("/"),
            agent_name=str(_value(cfg, "DIAGENT_AGENT_NAME", "pathfindership")).strip()
            or "pathfindership",
            timeout_seconds=_as_float(_value(cfg, "DIAGENT_TIMEOUT_SECONDS", 5), 5.0),
            fail_open=_as_bool(_value(cfg, "DIAGENT_FAIL_OPEN", True), True),
            log_policy_spans=_as_bool(_value(cfg, "DIAGENT_LOG_POLICY_SPANS", True), True),
            max_chunk_chars=max(1, _as_int(_value(cfg, "DIAGENT_MAX_CHUNK_CHARS", 1200), 1200)),
            max_retrieval_chunks=max(0, _as_int(_value(cfg, "DIAGENT_MAX_RETRIEVAL_CHUNKS", 5), 5)),
        )

    def to_readiness_dict(self, *, sdk_available: bool | None = None) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "api_url": self.api_url,
            "agent_name": self.agent_name,
            "timeout_seconds": self.timeout_seconds,
            "fail_open": self.fail_open,
            "log_policy_spans": self.log_policy_spans,
            "max_chunk_chars": self.max_chunk_chars,
            "max_retrieval_chunks": self.max_retrieval_chunks,
            "sdk_available": sdk_available,
        }


def load_diagent_config(cfg: Mapping[str, Any] | None = None) -> DiagentConfig:
    if cfg is None:
        from config import CFG

        cfg = CFG
    return DiagentConfig.from_mapping(cfg)
