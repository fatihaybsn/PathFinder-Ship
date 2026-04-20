# backend/web/app.py
# --- Üstte FastAPI ve standart importlar ---
import logging
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from collections import Counter
from pathlib import Path
import cv2, numpy as np

# 1) .env'yi yükleyen yer ÖNCE gelsin
from config import BACKEND_ROOT, CFG, DATA_ROOT, readiness_report  # load_dotenv() burada çalışır

# 2) Artık utils.* ve servisler
from utils.storage import ensure_dir, save_with_ring_buffer
from utils.mailer import send_image_via_email
from utils.text import fallback_instruction

from services.nlu_classifier import NLUClassifier
from services.pipeline_orchestrator import PipelineOrchestrator
from services.t5 import T5Service
from services.rag import RAGService
from services.yolo import YOLOService
from schemas.pipeline import RunResult, intent_result_from_prediction
from utils.vision import draw_dets

logger = logging.getLogger(__name__)

# ---------- Helpers ----------
def load_image_from_upload(upload: UploadFile) -> np.ndarray:
    """UploadFile -> BGR numpy image"""
    data = upload.file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Görüntü okunamadı")
    return img


def _static_url_for(path: str | Path) -> Optional[str]:
    try:
        rel = Path(path).resolve().relative_to(Path(DATA_ROOT).resolve())
        return f"/static/{rel.as_posix()}"
    except Exception:
        return None


def _stored_path_for(path: str | Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(BACKEND_ROOT).resolve()).as_posix()
    except Exception:
        return Path(path).name


# ============ ORTAK AYARLAR ============
PHOTO_DIR = CFG.get("PHOTO_DIR")
DETECT_DIR = CFG.get("DETECT_DIR")
MAX_FILES_PER_DIR = int(CFG.get("MAX_FILES_PER_DIR", 10))

# ---------- Schemas ----------
class IntentRequest(BaseModel):
    text: str


class IntentResponse(BaseModel):
    intent: str
    score: float
    threshold: Optional[float] = None
    narration: Optional[str] = None
    label: Optional[str] = None
    confidence: Optional[float] = None
    is_confident: Optional[bool] = None
    raw_scores: Optional[Dict[str, float]] = None
    latency_ms: Optional[int] = None
    error: Optional[str] = None

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str


class RagRequest(BaseModel):
    question: str
    use_internet: bool = False
    web_only: bool = False


class RagResponse(BaseModel):
    answer: str
    used_context: bool
    sources: List[str]


class RunRequest(BaseModel):
    message: str
    metadata: dict[str, Any] | None = None


class DetectResponse(BaseModel):
    labels: List[str]
    summary: str
    boxes: Optional[List[List[float]]] = None
    image_url: Optional[str] = None
    narration: Optional[str] = None

# ---------- App & Services ----------
app = FastAPI(title="PathFinder-Ship Web API")

# /static altında data'yı sun (çizilmiş görselleri göstermek için)
app.mount("/static", StaticFiles(directory=str(DATA_ROOT), check_dir=False), name="static")

# CORS
origins = [origin.strip() for origin in str(CFG.get("FRONTEND_ORIGIN", "*")).split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Servisleri tek sefer yükle (soğuk başlatma)
NLU: Optional[NLUClassifier] = None
T5: Optional[T5Service] = None
RAG: Optional[RAGService] = None
YOLO: Optional[YOLOService] = None
PIPELINE: Optional[PipelineOrchestrator] = None


@app.on_event("startup")
def startup_event():
    global NLU, T5, RAG, YOLO, PIPELINE
    report = readiness_report()
    if report["status"] != "ok":
        logger.warning("Startup readiness degraded; missing assets: %s", ", ".join(report["missing"]))
    NLU = NLUClassifier(CFG)
    T5 = T5Service(CFG)
    RAG = RAGService(CFG)
    YOLO = YOLOService(CFG)
    PIPELINE = PipelineOrchestrator(CFG, NLU, T5, RAG, YOLO)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/readiness")
def readiness():
    return readiness_report()


# ---------- Intent ----------
@app.post("/api/intent", response_model=IntentResponse)
def intent_api(body: IntentRequest):
    thr = float(CFG.get("CLS_ROUTE_THRESHOLD", 0.60))
    if NLU is None:
        result = intent_result_from_prediction(
            label="chat",
            confidence=0.0,
            threshold=thr,
            error="intent service is not initialized",
        )
    elif hasattr(NLU, "classify_intent"):
        result = NLU.classify_intent(body.text, threshold=thr)
    else:
        label, score = NLU.predict(body.text)
        result = intent_result_from_prediction(
            label=label,
            confidence=float(score),
            threshold=thr,
            error="intent service failed" if getattr(NLU, "last_error", None) else None,
        )

    return {
        "intent": result.label,
        "score": float(result.confidence or 0.0),
        "threshold": result.threshold,
        "narration": None,
        "label": result.label,
        "confidence": result.confidence,
        "is_confident": result.is_confident,
        "raw_scores": result.raw_scores,
        "latency_ms": result.latency_ms,
        "error": result.error,
    }


# ---------- Chat ----------
@app.post("/api/chat", response_model=ChatResponse)
def chat_api(body: ChatRequest):
    answer = T5.chat(body.message)  # T5'te chat(...) var
    return {"answer": answer}


# ---------- RAG ----------
# app.py (FastAPI)
@app.post("/api/rag", response_model=RagResponse)
def rag_api(body: RagRequest):
    # 1) RAG araması: context + en iyi skor + kaynaklar
    contexts, best_score, sources = RAG.retrieve(
        body.question,
        use_internet=body.use_internet,
        web_only=body.web_only,
    )

    used_ctx = bool(contexts)

    # 3) Cevabı üret
    if used_ctx:
        # RAG: context + question + RAG instruction
        answer = T5.answer(body.question, contexts)
    else:
        # Fallback: context yok → chat’ten farklı instruction (utils/text.py’deki senin metnin)
        # YENİ
        instr = fallback_instruction()  # text.py
        answer = T5.answer_model_only_with_instruction(body.question, instruction=instr)
        used_ctx = False
        sources = []


    # 4) UI'ya dön
    return {"answer": answer, "used_context": used_ctx, "sources": sources}


@app.post("/api/run", response_model=RunResult)
def run_api(body: RunRequest):
    if PIPELINE is None:
        logger.error("Pipeline is not initialized.")
        return RunResult(
            input_text=body.message,
            status="failed",
            errors=["pipeline is not initialized"],
        )

    return PIPELINE.run(body.message, metadata=body.metadata)


@app.post("/api/photo")
async def take_photo_api(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Sadece fotoğrafı kaydeder (tespit YOK) ve telefona e-posta ile gönderir.
    - Kayıt: PHOTO_DIR altında ring buffer (MAX_FILES_PER_DIR).
    - Mail: arka planda (BackgroundTasks).
    """
    img_bytes = await file.read()

    # ring buffer ile hedef dosya
    target_path = save_with_ring_buffer(PHOTO_DIR, filename_prefix="photo", ext="jpg", max_count=MAX_FILES_PER_DIR)
    ensure_dir(str(target_path))
    with open(target_path, "wb") as f:
        f.write(img_bytes)

    # telefona e-posta (arka planda)
    background_tasks.add_task(
        send_image_via_email,
        image_path=str(target_path),
        subject="New Photo",
        body="PathFinder-Ship posted a photo"
    )

    image_url = _static_url_for(target_path)

    return {"ok": True, "stored": _stored_path_for(target_path), "image_url": image_url}


narration = None
# ---------- Detect (YOLO) ----------
@app.post("/api/detect", response_model=DetectResponse)
async def detect_api(background_tasks: BackgroundTasks, file: UploadFile = File(...), draw: int = Form(1)):
    """
    Tarayıcıdan gönderilen bir görüntü (image/jpeg/png) bekler.
    YOLO çalışır, etiketleri ve kutuları döner.
    draw=1 ise çizilmiş görseli diske kaydeder ve URL döner.
    """
    narration = None
    img_bgr = load_image_from_upload(file)

    # YOLO: kutular, etiket adları, skorlar, sınıf id'leri
    boxes, labels, scores, cls_ids = YOLO.detect_from_bgr(img_bgr)

    # Özet ("2 person, 1 dog")
    summary = ", ".join([f"{c} {n}" for c, n in Counter(labels).items()]) if labels else "no objects"

    image_url = None
    if draw and len(boxes) > 0:
        # draw_dets için: dets = [x1,y1,x2,y2, conf, cls] biçimi
        dets = np.column_stack([
            np.array(boxes, dtype=np.float32),   # (N,4)
            np.array(scores, dtype=np.float32),  # (N,)
            np.array(cls_ids, dtype=np.float32)  # (N,)
        ])
        drawn = draw_dets(img_bgr.copy(), dets, YOLO.names)

        # YENİ BLOK (ring buffer + mail)
        target_path = save_with_ring_buffer(DETECT_DIR, filename_prefix="detect", ext="jpg", max_count=MAX_FILES_PER_DIR)
        ensure_dir(str(target_path))
        cv2.imwrite(str(target_path), drawn)

        narration = T5.narrate_detection(summary)
        # telefona e-posta (arka planda)
        subject = f"PathFinder-Ship reported a photo"  # ör. "2 person, 1 dog"
        background_tasks.add_task(
            send_image_via_email,
            image_path=str(target_path),
            subject=subject,
            body=f"{narration}\nDetail: {summary}" or f"Detail: {summary}"
        )

        image_url = _static_url_for(target_path)

    # Kutuları istemciye göndermek için listele
    boxes_list = [[float(x1), float(y1), float(x2), float(y2)] for (x1, y1, x2, y2) in boxes] if boxes else []
    return {
        "labels": labels,
        "summary": summary,
        "boxes": boxes_list,
        "image_url": image_url,
        "narration": narration,
    }


# (Opsiyonel) Belgeleri yükleyip indeksleme için bir uç nokta:
@app.post("/api/upload")
async def upload_doc(file: UploadFile = File(...)):
    # Dosyayı corpus'a koy; istersen burada indexer'ı tetikleyebilirsin.
    safe_name = Path(file.filename or "upload.bin").name
    out_path = Path(str(CFG.get("RAG_CORPUS_DIR"))) / safe_name
    ensure_dir(out_path)
    with open(out_path, "wb") as f:
        f.write(await file.read())
    return {"ok": True, "stored": _stored_path_for(out_path)}
