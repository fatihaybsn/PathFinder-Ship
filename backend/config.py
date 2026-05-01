from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

BACKEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_ROOT.parent
DATA_ROOT = BACKEND_ROOT / "data"


def _load_env_files() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    # Existing process env wins. backend/.env is the legacy local config file.
    for env_path in (BACKEND_ROOT / ".env", PROJECT_ROOT / ".env"):
        if env_path.exists():
            load_dotenv(env_path, override=False)


_load_env_files()


def _raw_env(name: str, aliases: Iterable[str] = ()) -> Optional[str]:
    for key in (name, *aliases):
        val = os.getenv(key)
        if val is not None and str(val).strip() != "":
            return str(val).strip()
    return None


def _has_env(name: str, aliases: Iterable[str] = ()) -> bool:
    return _raw_env(name, aliases) is not None


def _get_str(name: str, default: Optional[str] = None, aliases: Iterable[str] = ()) -> Optional[str]:
    val = _raw_env(name, aliases)
    return val if val is not None else default


def _get_int(name: str, default: int, aliases: Iterable[str] = ()) -> int:
    val = _raw_env(name, aliases)
    try:
        return int(str(val if val is not None else default).strip())
    except Exception:
        return default


def _get_float(name: str, default: float, aliases: Iterable[str] = ()) -> float:
    val = _raw_env(name, aliases)
    try:
        return float(str(val if val is not None else default).strip())
    except Exception:
        return default


def _get_bool(name: str, default: bool, aliases: Iterable[str] = ()) -> bool:
    val = _raw_env(name, aliases)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "y", "on")


def _resolve_path(path_value: str | Path | None) -> Optional[str]:
    if path_value is None:
        return None

    raw = str(path_value).strip()
    if not raw:
        return None

    expanded = Path(os.path.expandvars(raw)).expanduser()
    if expanded.is_absolute():
        return str(expanded.resolve()).replace("\\", "/")

    parts = expanded.parts
    base = PROJECT_ROOT if parts and parts[0].lower() == "backend" else BACKEND_ROOT
    return str((base / expanded).resolve()).replace("\\", "/")


def _get_path(name: str, default: str | Path, aliases: Iterable[str] = ()) -> str:
    return _resolve_path(_get_str(name, str(default), aliases)) or _resolve_path(default) or ""


def _path_exists(path_value: str | Path | None, *, kind: str = "any") -> bool:
    if not path_value:
        return False
    p = Path(path_value)
    if kind == "file":
        return p.is_file()
    if kind == "dir":
        return p.is_dir()
    return p.exists()


def _configured_email_enabled(cfg: Dict[str, Any]) -> bool:
    required = (
        cfg.get("EMAIL_SMTP_HOST"),
        cfg.get("EMAIL_USER"),
        cfg.get("EMAIL_PASSWORD"),
        cfg.get("EMAIL_TO_PHONE"),
    )
    return all(bool(str(v or "").strip()) for v in required)


def build_config() -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "PROJECT_ROOT": str(PROJECT_ROOT).replace("\\", "/"),
        "BACKEND_ROOT": str(BACKEND_ROOT).replace("\\", "/"),
        "DATA_ROOT": str(DATA_ROOT.resolve()).replace("\\", "/"),
    }

    cfg["APP_NAME"] = _get_str("APP_NAME", "PathFinder-Ship")
    cfg["DEFAULT_USER_NAME"] = _get_str("DEFAULT_USER_NAME", "Passenger")
    cfg["BOT_NAME"] = _get_str("BOT_NAME", "Passenger-Bot")
    cfg["DEBUG"] = _get_bool("DEBUG", False)

    cfg["API_HOST"] = _get_str("API_HOST", "0.0.0.0")
    cfg["API_PORT"] = _get_int("API_PORT", 8000)
    cfg["FRONTEND_ORIGIN"] = _get_str("FRONTEND_ORIGIN", "*")

    cfg["PHOTO_DIR"] = _get_path("PHOTO_DIR", DATA_ROOT / "web_out" / "photo")
    cfg["DETECT_DIR"] = _get_path("DETECT_DIR", DATA_ROOT / "web_out" / "detect")
    cfg["MAX_FILES_PER_DIR"] = _get_int("MAX_FILES_PER_DIR", 10)

    cfg["EMAIL_SMTP_HOST"] = _get_str("EMAIL_SMTP_HOST", "", aliases=("SMTP_HOST",))
    cfg["EMAIL_SMTP_PORT"] = _get_int("EMAIL_SMTP_PORT", 465, aliases=("SMTP_PORT",))
    cfg["EMAIL_USE_TLS"] = _get_bool("EMAIL_USE_TLS", False, aliases=("SMTP_USE_TLS",))
    cfg["EMAIL_USER"] = _get_str("EMAIL_USER", "", aliases=("SMTP_USER",))
    cfg["EMAIL_PASSWORD"] = _get_str("EMAIL_PASSWORD", "", aliases=("SMTP_PASSWORD",))
    cfg["EMAIL_FROM"] = _get_str("EMAIL_FROM", "", aliases=("SMTP_FROM",))
    cfg["EMAIL_TO_PHONE"] = _get_str("EMAIL_TO_PHONE", "", aliases=("SMTP_TO", "EMAIL_TO"))

    cfg["CAMERA_INDEX"] = _get_int("CAMERA_INDEX", 0)

    nlu_model_dir = _get_path("NLU_MODEL_DIR", BACKEND_ROOT / "assets" / "models" / "nlu")
    cfg["CLS_ONNX"] = _get_path(
        "CLS_ONNX",
        Path(nlu_model_dir) / "intent-minilm-int8.onnx",
        aliases=("NLU_ONNX", "NLU_MODEL_PATH"),
    )
    cfg["CLS_TOKENIZER_DIR"] = _get_path(
        "CLS_TOKENIZER_DIR",
        Path(nlu_model_dir) / "tokenizer",
        aliases=("NLU_TOKENIZER_DIR",),
    )
    cfg["CLS_MAX_LEN"] = _get_int("CLS_MAX_LEN", 64)
    cfg["CLS_ROUTE_THRESHOLD"] = _get_float("CLS_ROUTE_THRESHOLD", 0.60)

    t5_model_dir = _get_path("T5_MODEL_DIR", BACKEND_ROOT / "assets" / "models" / "t5")
    cfg["T5_TOKENIZER_DIR"] = _get_path("T5_TOKENIZER_DIR", Path(t5_model_dir) / "tokenizer")
    cfg["T5_ENCODER"] = _get_path(
        "T5_ENCODER",
        Path(t5_model_dir) / "encoder_model_int8.onnx",
        aliases=("T5_ENCODER_ONNX",),
    )
    cfg["T5_DECODER"] = _get_path(
        "T5_DECODER",
        Path(t5_model_dir) / "decoder_model_int8.onnx",
        aliases=("T5_DECODER_ONNX",),
    )
    cfg["T5_MAX_SRC_LEN"] = _get_int("T5_MAX_SRC_LEN", 512)
    cfg["T5_MAX_NEW_TOKENS_CHAT"] = _get_int("T5_MAX_NEW_TOKENS_CHAT", 256)
    cfg["T5_MAX_NEW_TOKENS_RAG"] = _get_int("T5_MAX_NEW_TOKENS_RAG", 64)

    cfg["YOLO_ONNX"] = _get_path(
        "YOLO_ONNX",
        BACKEND_ROOT / "assets" / "models" / "yolo_nas" / "yolo_nas_s_coco.onnx",
        aliases=("YOLO_MODEL_PATH",),
    )
    cfg["YOLO_LABELS"] = _get_path(
        "YOLO_LABELS",
        BACKEND_ROOT / "assets" / "models" / "yolo_nas" / "labels.txt",
        aliases=("YOLO_LABELS_PATH",),
    )
    cfg["YOLO_SIZE"] = _get_int("YOLO_SIZE", 640)
    cfg["YOLO_CONF"] = _get_float("YOLO_CONF", 0.25)
    cfg["YOLO_IOU"] = _get_float("YOLO_IOU", 0.45)
    cfg["YOLO_PREPROC_IN_MODEL"] = _get_bool("YOLO_PREPROC_IN_MODEL", False)

    cfg["RAG_CORPUS_DIR"] = _get_path("RAG_CORPUS_DIR", DATA_ROOT / "rag" / "corpus")
    cfg["UPLOAD_MAX_BYTES"] = _get_int("UPLOAD_MAX_BYTES", 10 * 1024 * 1024)
    cfg["RAG_UPLOAD_ALLOWED_EXTENSIONS"] = _get_str(
        "RAG_UPLOAD_ALLOWED_EXTENSIONS",
        ".pdf,.docx,.txt,.md,.html,.htm",
    )
    cfg["RAG_SCORE_THRESHOLD"] = _get_float("RAG_SCORE_THRESHOLD", 0.40)
    cfg["RAG_TOP_K"] = _get_int("RAG_TOP_K", 4)
    cfg["RAG_MAX_CTX_TOKENS"] = _get_int("RAG_MAX_CTX_TOKENS", 512)
    cfg["RAG_TOKENIZER_DIR"] = _get_path("RAG_TOKENIZER_DIR", cfg["T5_TOKENIZER_DIR"])
    cfg["RAG_CHUNK_TOKENS"] = _get_int("RAG_CHUNK_TOKENS", _get_int("RAG_CHUNK_SIZE", 90))
    cfg["RAG_CHUNK_OVERLAP_TOKENS"] = _get_int(
        "RAG_CHUNK_OVERLAP_TOKENS",
        _get_int("RAG_CHUNK_OVERLAP", 20),
    )
    cfg["RAG_WORD_CHUNK_SIZE"] = _get_int("RAG_WORD_CHUNK_SIZE", 90)
    cfg["RAG_WORD_CHUNK_OVERLAP"] = _get_int("RAG_WORD_CHUNK_OVERLAP", 20)

    cfg["EMBED_MODEL"] = _get_str("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    cfg["CHROMA_PATH"] = _get_path(
        "CHROMA_PATH",
        BACKEND_ROOT / "assets" / "rag" / "chroma_db",
        aliases=("RAG_CHROMA_DIR",),
    )
    cfg["RAG_SQLITE_PATH"] = _get_path("RAG_SQLITE_PATH", Path(cfg["CHROMA_PATH"]) / "bm25.sqlite")
    cfg["CHROMA_COLLECTION"] = _get_str("CHROMA_COLLECTION", "pathfinder_corpus")
    cfg["VECTOR_WEIGHT"] = _get_float("VECTOR_WEIGHT", 0.75)
    cfg["BM25_WEIGHT"] = _get_float("BM25_WEIGHT", 0.25)
    cfg["RAG_WEB_MIN_STRENGTH"] = _get_float("RAG_WEB_MIN_STRENGTH", 0.75)
    cfg["WEB_CHUNK_SUPPORT_THRESHOLD"] = _get_float("WEB_CHUNK_SUPPORT_THRESHOLD", 0.70)

    cfg["WEB_API_ENDPOINT"] = _get_str("WEB_API_ENDPOINT", "")
    cfg["WEB_API_KEY"] = _get_str("WEB_API_KEY", "")
    cfg["ENABLE_WEB_SEARCH"] = _get_bool("ENABLE_WEB_SEARCH", False)
    cfg["ENABLE_CAMERA_ACTIONS"] = _get_bool("ENABLE_CAMERA_ACTIONS", True)

    if _has_env("ENABLE_EMAIL"):
        cfg["ENABLE_EMAIL"] = _get_bool("ENABLE_EMAIL", False)
    else:
        cfg["ENABLE_EMAIL"] = _configured_email_enabled(cfg)

    return cfg


CFG = build_config()


def readiness_report() -> Dict[str, Any]:
    assets = {
        "t5_model": _path_exists(CFG.get("T5_ENCODER"), kind="file")
        and _path_exists(CFG.get("T5_DECODER"), kind="file"),
        "t5_tokenizer": _path_exists(CFG.get("T5_TOKENIZER_DIR"), kind="dir"),
        "nlu_model": _path_exists(CFG.get("CLS_ONNX"), kind="file"),
        "nlu_tokenizer": _path_exists(CFG.get("CLS_TOKENIZER_DIR"), kind="dir"),
        "yolo_model": _path_exists(CFG.get("YOLO_ONNX"), kind="file"),
        "yolo_labels": _path_exists(CFG.get("YOLO_LABELS"), kind="file"),
        "rag_corpus": _path_exists(CFG.get("RAG_CORPUS_DIR"), kind="dir"),
        "rag_index": _path_exists(CFG.get("CHROMA_PATH"), kind="dir"),
        "rag_sqlite": _path_exists(CFG.get("RAG_SQLITE_PATH"), kind="file"),
    }
    missing = [name for name, ok in assets.items() if not ok]

    return {
        "status": "ok" if not missing else "degraded",
        "config": "loaded",
        "assets": assets,
        "features": {
            "email": bool(CFG.get("ENABLE_EMAIL", False)),
            "web_search": bool(CFG.get("ENABLE_WEB_SEARCH", False)),
            "camera_actions": bool(CFG.get("ENABLE_CAMERA_ACTIONS", True)),
        },
        "missing": missing,
    }
