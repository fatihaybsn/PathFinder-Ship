# app/config.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    # .env'yi otomatik yükle (yoksa sessizce geçer)
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -------- Yardımcılar --------
def _get_str(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(name, default)
    if val is None:
        return None
    return str(val).strip()

def _get_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, default)).strip())
    except Exception:
        return default

def _get_float(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, default)).strip())
    except Exception:
        return default

def _get_bool(name: str, default: bool) -> bool:
    v = str(os.getenv(name, str(default))).strip().lower()
    return v in ("1", "true", "yes", "y", "on")

def _normpath(p: Optional[str]) -> Optional[str]:
    if p is None or p == "":
        return p
    return str(Path(p).as_posix())


# -------- Ana API --------
def build_config() -> Dict[str, Any]:
    """
    .env'den bütün ayarları toplar ve tek bir CFG sözlüğü üretir.
    Bu anahtarlar, projenin geri kalanında zaten kullanılan isimlerle uyumludur.
    """
    cfg: Dict[str, Any] = {}
    # ---- Storage (foto & detect) ----
    cfg["PHOTO_DIR"] = _normpath(_get_str("PHOTO_DIR", "data/web_out/photo"))
    cfg["DETECT_DIR"] = _normpath(_get_str("DETECT_DIR", "data/web_out/detect"))
    cfg["MAX_FILES_PER_DIR"] = _get_int("MAX_FILES_PER_DIR", 10)

    # ---- Email ----
    cfg["EMAIL_SMTP_HOST"] = _get_str("EMAIL_SMTP_HOST", "")
    cfg["EMAIL_SMTP_PORT"] = _get_int("EMAIL_SMTP_PORT", 465)
    cfg["EMAIL_USER"] = _get_str("EMAIL_USER", "")
    cfg["EMAIL_PASSWORD"] = _get_str("EMAIL_PASSWORD", "")
    cfg["EMAIL_FROM"] = _get_str("EMAIL_FROM", "")
    cfg["EMAIL_TO_PHONE"] = _get_str("EMAIL_TO_PHONE", "")

    # ---- Uygulama / Genel ----
    cfg["APP_NAME"] = _get_str("APP_NAME", "PathFinder-Ship")
    cfg["DEFAULT_USER_NAME"] = _get_str("DEFAULT_USER_NAME", "Passenger")

    # ---- Kamera ----
    cfg["CAMERA_INDEX"] = _get_int("CAMERA_INDEX", 0)

    # ---- NLU (MiniLM-L6 Intent Classifier, ONNX INT8) ----
    cfg["CLS_ONNX"] = _normpath(_get_str("CLS_ONNX", "assets/models/nlu/intent-minilm-int8.onnx"))
    cfg["CLS_TOKENIZER_DIR"] = _normpath(_get_str("CLS_TOKENIZER_DIR", "assets/models/nlu/tokenizer"))
    cfg["CLS_MAX_LEN"] = _get_int("CLS_MAX_LEN", 64)

    # Sınıflandırıcı yönlendirme eşiği (komut mı, soru mu?)
    cfg["CLS_ROUTE_THRESHOLD"] = _get_float("CLS_ROUTE_THRESHOLD", 0.60)

    # ---- T5 (Flan-T5 Large, ONNX INT8; encoder + decoder) ----
    # Not: T5'i sen de INT8 ONNX'a çevirdin; burada encoder/decoder yoluyla çalışıyoruz.
    cfg["T5_TOKENIZER_DIR"] = _normpath(_get_str("T5_TOKENIZER_DIR", "assets/models/t5/tokenizer"))
    cfg["T5_ENCODER"] = _normpath(_get_str("T5_ENCODER", "assets/models/t5/encoder_model_int8.onnx"))
    cfg["T5_DECODER"] = _normpath(_get_str("T5_DECODER", "assets/models/t5/decoder_model_int8.onnx"))

    # Eğer sadece decoder_with_past.onnx varsa yukarıdaki DECODER yolunu ona işaret et.
    cfg["BOT_NAME"] = _get_str("BOT_NAME", "Passenger-Bot")
    cfg["T5_MAX_SRC_LEN"] = _get_int("T5_MAX_SRC_LEN", 512)
    cfg["T5_MAX_NEW_TOKENS_CHAT"] = _get_int("T5_MAX_NEW_TOKENS_CHAT", 256)
    cfg["T5_MAX_NEW_TOKENS_RAG"] = _get_int("T5_MAX_NEW_TOKENS_RAG", 64)

    # ---- YOLO (Object Detection, ONNX INT8) ----
    cfg["YOLO_ONNX"] = _normpath(_get_str("YOLO_ONNX", "assets/models/yolo_nas/yolo_nas_s_coco.onnx"))
    cfg["YOLO_LABELS"] = _normpath(_get_str("YOLO_LABELS", "assets/models/yolo_nas/labels.txt"))
    cfg["YOLO_SIZE"] = _get_int("YOLO_SIZE", 640)
    cfg["YOLO_CONF"] = _get_float("YOLO_CONF", 0.25)
    cfg["YOLO_IOU"] = _get_float("YOLO_IOU", 0.45)

    # ---- RAG (Hybrid: Vector + BM25) ----
    # Router kuralın: rag_score >= 0.4 → context'li; altı → context'siz.
    cfg["RAG_SCORE_THRESHOLD"] = _get_float("RAG_SCORE_THRESHOLD", 0.40)
    cfg["RAG_TOP_K"] = _get_int("RAG_TOP_K", 4)
    cfg["RAG_MAX_CTX_TOKENS"] = _get_int("RAG_MAX_CTX_TOKENS", 512)

    # Backend tarafı (rag_backend) için tipik ayarlar:
    cfg["EMBED_MODEL"] = _get_str("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    # Chroma kalıcı depo yolu (Windows dostu, assets altında tutmak güvenli)
    cfg["CHROMA_PATH"] = _normpath(_get_str("CHROMA_PATH", "assets/rag/chroma_db"))
    # (Opsiyonel) Hibrit ağırlıklar — backend içinden de .env ile çekiliyor olabilir
    cfg["VECTOR_WEIGHT"] = _get_float("VECTOR_WEIGHT", 0.75)
    cfg["BM25_WEIGHT"] = _get_float("BM25_WEIGHT", 0.25)

    # ---- Web Arama (opsiyonel) ----
    cfg["WEB_API_ENDPOINT"] = _get_str("WEB_API_ENDPOINT", "")
    cfg["WEB_API_KEY"] = _get_str("WEB_API_KEY", "")

    # ---- Diğer bayraklar ----
    cfg["DEBUG"] = _get_bool("DEBUG", False)

    return cfg


# Tek satırda erişmek istersen:
CFG = build_config()
