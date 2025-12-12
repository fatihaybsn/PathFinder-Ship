# backend/web/app.py
# --- Üstte FastAPI ve standart importlar ---
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from collections import Counter
from pathlib import Path
import os, cv2, numpy as np

# 1) .env'yi yükleyen yer ÖNCE gelsin
from config import CFG  # load_dotenv() burada çalışır ✅

# 2) Artık utils.* ve servisler
from utils.storage import ensure_dir, save_with_ring_buffer
from utils.mailer import send_image_via_email
from utils.text import fallback_instruction, build_model_only_prompt

from services.nlu_classifier import NLUClassifier
from services.t5 import T5Service
from services.rag import RAGService
from services.yolo import YOLOService
from utils.vision import draw_dets

# ---------- Helpers ----------
def load_image_from_upload(upload: UploadFile) -> np.ndarray:
    """UploadFile -> BGR numpy image"""
    data = upload.file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Görüntü okunamadı")
    return img

# ============ ORTAK AYARLAR ============
PHOTO_DIR = CFG.get("PHOTO_DIR")
DETECT_DIR = CFG.get("DETECT_DIR")
MAX_FILES_PER_DIR = int(CFG.get("MAX_FILES_PER_DIR", 10))

EMAIL_SMTP_HOST = CFG.get("EMAIL_SMTP_HOST", "")
EMAIL_SMTP_PORT = int(CFG.get("EMAIL_SMTP_PORT", 465))
EMAIL_USER = CFG.get("EMAIL_USER", "")
EMAIL_PASSWORD = CFG.get("EMAIL_PASSWORD", "")
EMAIL_FROM = CFG.get("EMAIL_FROM", EMAIL_USER)
EMAIL_TO_PHONE = CFG.get("EMAIL_TO_PHONE", "")

# ---------- Schemas ----------
class IntentRequest(BaseModel):
    text: str


class IntentResponse(BaseModel):
    intent: str
    score: float
    threshold: float
    narration: Optional[str] = None

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


class DetectResponse(BaseModel):
    labels: List[str]
    summary: str
    boxes: Optional[List[List[float]]] = None
    image_url: Optional[str] = None
    narration: Optional[str] = None

# ---------- App & Services ----------
app = FastAPI(title="PathFinder-Ship Web API")

# /static altında data'yı sun (çizilmiş görselleri göstermek için)
app.mount("/static", StaticFiles(directory="data"), name="static")

# CORS
origins = [os.getenv("FRONTEND_ORIGIN", "*")]
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


@app.on_event("startup")
def startup_event():
    global NLU, T5, RAG, YOLO
    NLU = NLUClassifier(CFG)
    T5 = T5Service(CFG)
    RAG = RAGService(CFG)
    YOLO = YOLOService(CFG)


@app.get("/api/health")
def health():
    return {"ok": True}


# ---------- Intent ----------
@app.post("/api/intent", response_model=IntentResponse)
def intent_api(body: IntentRequest):
    label, score = NLU.predict(body.text)

    narration = None
    thr = float(CFG.get("CLS_ROUTE_THRESHOLD", 0.7))

    if score >= thr:
        if label == "open_camera":
            narration = T5.narrate_open_camera()
        elif label == "close_camera":
            narration = T5.narrate_close_camera()
        elif label == "take_photo":
            narration = T5.narrate_take_photo()
        # object_detect: algılama yapılmadan sadece niyet var.
        # İstersen burada "hazırlanıyorum" tarzı bir narration da üretebilirsin
        # (örn. T5.narrate_detection("...")), ama tipik akışta detect sonrası üretiyoruz.

    return {"intent": label, "score": float(score), "threshold": thr, "narration": narration}


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

    # URL (StaticFiles(directory='data')) altında servis
    try:
        rel = Path(target_path).relative_to("data")
        image_url = f"/static/{rel.as_posix()}"
    except Exception:
        image_url = None

    return {"ok": True, "stored": str(target_path), "image_url": image_url}


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

        try:
            rel = Path(target_path).relative_to("data")
            image_url = f"/static/{rel.as_posix()}"
        except Exception:
            image_url = None

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
    out_path = f"backend/data/rag/corpus/{os.path.basename(file.filename)}"
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "wb") as f:
        f.write(await file.read())
    return {"ok": True, "stored": out_path}