# backend/web/app.py
# --- Üstte FastAPI ve standart importlar ---
import logging
import time
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List, Mapping, Optional
from collections import Counter
from pathlib import Path
from uuid import uuid4
import cv2, numpy as np

# 1) .env'yi yükleyen yer ÖNCE gelsin
from config import BACKEND_ROOT, CFG, DATA_ROOT, readiness_report  # load_dotenv() burada çalışır

# 2) Artık utils.* ve servisler
from utils.storage import ensure_dir, save_with_ring_buffer
from utils.mailer import send_image_via_email
from utils.text import fallback_instruction

from services.nlu_classifier import NLUClassifier
from services.pipeline_orchestrator import PipelineOrchestrator
from services.document_indexing import UploadIndexingError, index_upload_file, upload_error_response
from services.generation.base import BaseGenerationProvider
from services.generation.factory import build_generation_provider
from services.observability import DiagentSafeClient
from services.observability.diagent_mapper import emit_run_result_telemetry, sanitize_metadata
from services.rag import RAGService
from services.yolo import YOLOService
from schemas.pipeline import (
    DETECTION_STATUS_INVALID_IMAGE,
    DETECTION_STATUS_MODEL_ERROR,
    DetectionResult,
    GenerationResult,
    RunResult,
    detection_result_from_legacy,
    generation_result_from_text,
    intent_result_from_prediction,
    to_serializable_dict,
)
from utils.vision import draw_dets

logger = logging.getLogger(__name__)
DIAGENT_CLIENT_FACTORY = DiagentSafeClient.from_config

# ---------- Helpers ----------
def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _create_diagent_client() -> DiagentSafeClient:
    try:
        return DIAGENT_CLIENT_FACTORY(CFG)
    except Exception:
        logger.warning("Diagent client creation failed; telemetry disabled.", exc_info=True)
        return DiagentSafeClient.from_config({"DIAGENT_ENABLED": False})


def _diagent_finish_status(result: RunResult) -> str:
    return "failed" if result.status == "failed" else "finished"


def _diagent_error(result: RunResult) -> str | None:
    if result.status != "failed":
        return None
    if result.errors:
        return "; ".join(str(error) for error in result.errors)
    return "pathfindership run failed"


def _finish_diagent_run(
    client: DiagentSafeClient,
    run_id: str | None,
    result: RunResult,
    *,
    request_metadata: dict[str, Any] | None = None,
) -> None:
    try:
        emit_run_result_telemetry(
            client,
            run_id,
            result,
            config=client.config,
            request_metadata=request_metadata,
            app_config=CFG,
        )
    except Exception:
        logger.warning("Diagent telemetry mapping failed.", exc_info=True)
    finally:
        client.finish_run(
            run_id,
            output=result.final_answer,
            status=_diagent_finish_status(result),
            error=_diagent_error(result),
        )


CORRELATION_SOURCE_KEYS = ("correlation_id", "request_id", "conversation_id", "trace_id")


def _normalize_correlation_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:128]


def _correlation_from_metadata(metadata: Mapping[str, Any] | None) -> str | None:
    if not isinstance(metadata, Mapping):
        return None
    for key in CORRELATION_SOURCE_KEYS:
        correlation_id = _normalize_correlation_id(metadata.get(key))
        if correlation_id:
            return correlation_id
    return None


def _new_correlation_id() -> str:
    return str(uuid4())


def _ensure_client_action_correlation(
    result: RunResult,
    *,
    request_metadata: Mapping[str, Any] | None = None,
) -> None:
    client_action = result.client_action
    if client_action is None or client_action.action != "capture_photo":
        return

    payload = dict(client_action.payload or {})
    payload["correlation_id"] = (
        _normalize_correlation_id(payload.get("correlation_id"))
        or _correlation_from_metadata(request_metadata)
        or _correlation_from_metadata(result.metadata)
        or _new_correlation_id()
    )
    payload.setdefault("originating_action", client_action.action)
    if result.route is not None:
        payload.setdefault("originating_route", result.route.route)
    client_action.payload = payload


def _detection_run_output(summary: str, narration: str | None, detection: DetectionResult) -> str:
    if narration:
        return narration
    if detection.error:
        return f"Object detection completed with error: {detection.error}"
    return f"Object summary: {summary}"


def _finish_detection_diagent_run(
    client: DiagentSafeClient,
    run_id: str | None,
    *,
    detection: DetectionResult,
    generation: GenerationResult | None,
    summary: str,
    narration: str | None,
    correlation_id: str,
    correlation_id_provided: bool,
    image_metadata: Mapping[str, Any],
    email_scheduled: bool,
) -> None:
    if not run_id:
        return

    request_payload = sanitize_metadata(
        {
            "endpoint": "/api/detect",
            "correlation_id": correlation_id,
            "correlation_id_provided": correlation_id_provided,
            "origin": "frontend_snapshot",
            "image_received": bool(image_metadata.get("size_bytes", 0)),
            "image_decoded": detection.status != DETECTION_STATUS_INVALID_IMAGE,
            "image_metadata": image_metadata,
            "email_scheduled": email_scheduled,
        }
    )

    try:
        client.log_span(
            run_id,
            span_type="system",
            name="pathfindership.detect.request",
            payload=request_payload,
        )

        telemetry_result = RunResult(
            input_text="PathFinderShip detection request",
            final_answer=_detection_run_output(summary, narration, detection),
            status="degraded" if detection.error else "completed",
            detection=detection,
            generation=generation,
            metadata={
                "endpoint": "/api/detect",
                "correlation_id": correlation_id,
                "correlation_id_provided": correlation_id_provided,
                "origin": "frontend_snapshot",
                "image_metadata": sanitize_metadata(image_metadata),
                "email_scheduled": email_scheduled,
            },
        )
        emit_run_result_telemetry(
            client,
            run_id,
            telemetry_result,
            config=client.config,
            app_config=CFG,
        )
    except Exception:
        logger.warning("Diagent detection telemetry mapping failed.", exc_info=True)
    finally:
        client.finish_run(
            run_id,
            output=_detection_run_output(summary, narration, detection),
            status="finished",
        )


IMAGE_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
IMAGE_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _upload_metadata(upload: UploadFile, size_bytes: int | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "filename": Path(upload.filename or "").name or None,
        "content_type": upload.content_type,
        "max_size_bytes": int(CFG.get("UPLOAD_MAX_BYTES", 10 * 1024 * 1024)),
    }
    if size_bytes is not None:
        metadata["size_bytes"] = int(size_bytes)
    return metadata


def _invalid_image_result(
    upload: UploadFile,
    error: str,
    *,
    started: float,
    size_bytes: int | None = None,
) -> DetectionResult:
    return DetectionResult(
        objects=[],
        image_source="upload",
        model_name="yolo",
        latency_ms=_elapsed_ms(started),
        status=DETECTION_STATUS_INVALID_IMAGE,
        error=error,
        metadata=_upload_metadata(upload, size_bytes),
    )


async def read_image_upload(upload: UploadFile) -> tuple[np.ndarray | None, DetectionResult | None, dict[str, Any]]:
    """UploadFile -> validated BGR numpy image or structured detection error."""
    started = time.perf_counter()
    data = await upload.read()
    size_bytes = len(data)
    metadata = _upload_metadata(upload, size_bytes)
    if size_bytes == 0:
        return None, _invalid_image_result(upload, "image_empty", started=started, size_bytes=size_bytes), metadata

    max_bytes = int(CFG.get("UPLOAD_MAX_BYTES", 10 * 1024 * 1024))
    if size_bytes > max_bytes:
        return None, _invalid_image_result(upload, "image_too_large", started=started, size_bytes=size_bytes), metadata

    suffix = Path(upload.filename or "").suffix.lower()
    content_type = (upload.content_type or "").split(";")[0].strip().lower()
    if suffix not in IMAGE_ALLOWED_EXTENSIONS and content_type not in IMAGE_ALLOWED_CONTENT_TYPES:
        return None, _invalid_image_result(
            upload,
            "unsupported_image_format",
            started=started,
            size_bytes=size_bytes,
        ), metadata

    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, _invalid_image_result(upload, "image_decode_failed", started=started, size_bytes=size_bytes), metadata
    if img.size == 0 or img.shape[0] <= 0 or img.shape[1] <= 0:
        return None, _invalid_image_result(upload, "image_empty", started=started, size_bytes=size_bytes), metadata

    return img, None, metadata


def load_image_from_upload(upload: UploadFile) -> np.ndarray:
    """Legacy sync helper kept for callers that already decoded uploads."""
    data = upload.file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Goruntu okunamadi")
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
    retrieval: Optional[dict] = None  # structured RAG evidence (ek alan)


class RunRequest(BaseModel):
    message: str
    metadata: dict[str, Any] | None = None


class DetectResponse(BaseModel):
    labels: List[str]
    summary: str
    boxes: Optional[List[List[float]]] = None
    scores: Optional[List[float]] = None
    class_ids: Optional[List[int]] = None
    image_url: Optional[str] = None
    narration: Optional[str] = None
    detection: Optional[dict] = None
    generation: Optional[dict] = None
    error: Optional[str] = None

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
GENERATION: Optional[BaseGenerationProvider] = None
T5: Optional[BaseGenerationProvider] = None
RAG: Optional[RAGService] = None
YOLO: Optional[YOLOService] = None
PIPELINE: Optional[PipelineOrchestrator] = None


@app.on_event("startup")
def startup_event():
    global NLU, GENERATION, T5, RAG, YOLO, PIPELINE
    report = readiness_report()
    if report["status"] != "ok":
        logger.warning("Startup readiness degraded; missing assets: %s", ", ".join(report["missing"]))
    NLU = NLUClassifier(CFG)
    GENERATION = build_generation_provider(CFG)
    T5 = GENERATION
    RAG = RAGService(CFG)
    YOLO = YOLOService(CFG)
    PIPELINE = PipelineOrchestrator(CFG, NLU, GENERATION, RAG, YOLO)


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
    answer = GENERATION.chat(body.message)
    return {"answer": answer}


# ---------- RAG ----------
# app.py (FastAPI)
@app.post("/api/rag", response_model=RagResponse)
def rag_api(body: RagRequest):
    from schemas.pipeline import to_serializable_dict

    retrieval_dict = None

    if hasattr(RAG, "retrieve_structured"):
        # Structured retrieval — chunk-level evidence korunur
        retrieval = RAG.retrieve_structured(
            body.question,
            use_internet=body.use_internet,
            web_only=body.web_only,
        )
        used_ctx = retrieval.used_context
        sources = [c.source for c in retrieval.chunks if c.source] if retrieval.chunks else []

        if used_ctx and retrieval.chunks:
            from services.rag import build_context_from_chunks
            ctx_str = build_context_from_chunks(
                retrieval.chunks,
                max_tokens=RAG.max_ctx_tokens,
                question=body.question,
            )
            contexts = [ctx_str] if ctx_str else []
        else:
            contexts = []

        retrieval_dict = to_serializable_dict(retrieval)
    else:
        # Legacy fallback
        contexts, best_score, sources = RAG.retrieve(
            body.question,
            use_internet=body.use_internet,
            web_only=body.web_only,
        )
        used_ctx = bool(contexts)

    # Cevabı üret
    if used_ctx and contexts:
        answer = GENERATION.answer(body.question, contexts)
    else:
        instr = fallback_instruction()
        answer = GENERATION.answer_model_only_with_instruction(body.question, instruction=instr)
        used_ctx = False
        sources = []

    # UI'ya dön (eski alanlar korunur, retrieval opsiyonel ek alan)
    return {"answer": answer, "used_context": used_ctx, "sources": sources, "retrieval": retrieval_dict}


@app.post("/api/run", response_model=RunResult)
def run_api(body: RunRequest):
    started = time.perf_counter()
    diagent_client = _create_diagent_client()
    diagent_run_id = diagent_client.create_run(body.message)
    try:
        if PIPELINE is None:
            logger.error("Pipeline is not initialized.")
            metadata = body.metadata or {}
            result = RunResult(
                input_text=body.message,
                status="failed",
                errors=["pipeline is not initialized"],
                metadata={
                    "use_internet": bool(metadata.get("use_internet", False)),
                    "web_only": bool(metadata.get("web_only", False)),
                },
                duration_ms=_elapsed_ms(started),
            )
        else:
            result = PIPELINE.run(body.message, metadata=body.metadata)
    except Exception as exc:
        diagent_client.finish_run(
            diagent_run_id,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        diagent_client.close()
        raise

    _ensure_client_action_correlation(result, request_metadata=body.metadata)

    try:
        _finish_diagent_run(
            diagent_client,
            diagent_run_id,
            result,
            request_metadata=body.metadata,
        )
    finally:
        diagent_client.close()

    return result


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

# ---------- Detect (YOLO) ----------
@app.post("/api/detect", response_model=DetectResponse)
async def detect_api(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    draw: int = Form(1),
    correlation_id: str | None = Form(None),
):
    """
    Tarayıcıdan gönderilen bir görüntü (image/jpeg/png) bekler.
    YOLO çalışır, etiketleri ve kutuları döner.
    draw=1 ise çizilmiş görseli diske kaydeder ve URL döner.
    """
    provided_correlation_id = _normalize_correlation_id(correlation_id)
    telemetry_correlation_id = provided_correlation_id or _new_correlation_id()
    diagent_client = _create_diagent_client()
    diagent_run_id = diagent_client.create_run(
        f"YOLO detection follow-up for correlation_id={telemetry_correlation_id}"
    )

    try:
        img_bgr, image_error, image_metadata = await read_image_upload(file)
        generation: GenerationResult | None = None
        narration = None
        image_url = None
        email_scheduled = False

        if image_error is not None:
            detection = image_error
        elif YOLO is None:
            detection = DetectionResult(
                objects=[],
                image_source="upload",
                model_name="yolo",
                status=DETECTION_STATUS_MODEL_ERROR,
                error="detection_service_unavailable",
                metadata=image_metadata,
            )
        else:
            if hasattr(YOLO, "detect_structured"):
                detection = YOLO.detect_structured(img_bgr, image_source="upload")
            else:
                try:
                    boxes, labels, scores, cls_ids = YOLO.detect_from_bgr(img_bgr)
                    detection = detection_result_from_legacy(
                        labels,
                        boxes,
                        scores,
                        cls_ids,
                        image_source="upload",
                        model_name="yolo",
                        metadata=image_metadata,
                    )
                except Exception:
                    logger.exception("Legacy YOLO detection failed")
                    detection = DetectionResult(
                        objects=[],
                        image_source="upload",
                        model_name="yolo",
                        status=DETECTION_STATUS_MODEL_ERROR,
                        error="yolo_inference_failed",
                        metadata=image_metadata,
                    )

        labels = [obj.label for obj in detection.objects]
        boxes = [obj.bbox for obj in detection.objects if obj.bbox is not None]
        scores = [float(obj.confidence) for obj in detection.objects if obj.confidence is not None]
        cls_ids = [
            int(obj.metadata["class_id"])
            for obj in detection.objects
            if obj.metadata and obj.metadata.get("class_id") is not None
        ]

        # Özet ("2 person, 1 dog")
        summary = ", ".join([f"{n} {c}" for c, n in Counter(labels).items()]) if labels else "no objects"

        if img_bgr is not None and draw and len(boxes) > 0 and len(scores) == len(boxes) and len(cls_ids) == len(boxes):
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

            if GENERATION is not None and hasattr(GENERATION, "narrate_detection_structured"):
                try:
                    generation = GENERATION.narrate_detection_structured(summary)
                    narration = generation.text
                except Exception:
                    logger.exception("Detection narration failed")
                    generation = generation_result_from_text(
                        "",
                        model_name=getattr(GENERATION, "model_name", "t5"),
                        runtime=getattr(GENERATION, "runtime", "onnxruntime"),
                        device=getattr(GENERATION, "device", "cpu"),
                        prompt_type="detection_narration",
                        fallback_used=True,
                        fallback_reason="detection_narration_failed",
                        error="detection_narration_failed",
                    )
            elif GENERATION is not None and hasattr(GENERATION, "narrate_detection"):
                narration = GENERATION.narrate_detection(summary)
                generation = generation_result_from_text(
                    narration,
                    model_name=getattr(GENERATION, "model_name", "t5"),
                    runtime=getattr(GENERATION, "runtime", "onnxruntime"),
                    device=getattr(GENERATION, "device", "cpu"),
                    prompt_type="detection_narration",
                )

            # telefona e-posta (arka planda)
            subject = f"PathFinder-Ship reported a photo"  # ör. "2 person, 1 dog"
            background_tasks.add_task(
                send_image_via_email,
                image_path=str(target_path),
                subject=subject,
                body=f"{narration}\nDetail: {summary}" if narration else f"Detail: {summary}"
            )
            email_scheduled = True

            image_url = _static_url_for(target_path)

        # Kutuları istemciye göndermek için listele
        boxes_list = [[float(x1), float(y1), float(x2), float(y2)] for (x1, y1, x2, y2) in boxes] if boxes else []
        response_body = {
            "labels": labels,
            "summary": summary,
            "boxes": boxes_list,
            "scores": scores,
            "class_ids": cls_ids,
            "image_url": image_url,
            "narration": narration,
            "detection": to_serializable_dict(detection),
            "generation": to_serializable_dict(generation) if generation is not None else None,
            "error": detection.error,
        }
    except Exception as exc:
        diagent_client.finish_run(
            diagent_run_id,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    else:
        _finish_detection_diagent_run(
            diagent_client,
            diagent_run_id,
            detection=detection,
            generation=generation,
            summary=summary,
            narration=narration,
            correlation_id=telemetry_correlation_id,
            correlation_id_provided=provided_correlation_id is not None,
            image_metadata=image_metadata,
            email_scheduled=email_scheduled,
        )
        return response_body
    finally:
        diagent_client.close()


# (Opsiyonel) Belgeleri yükleyip indeksleme için bir uç nokta:
@app.post("/api/upload")
async def upload_doc(file: UploadFile = File(...)):
    try:
        result = await index_upload_file(file, CFG)
    except UploadIndexingError as exc:
        return JSONResponse(status_code=exc.status_code, content=upload_error_response(exc))

    body = to_serializable_dict(result)
    body["ok"] = bool(result.saved_path)
    body["stored"] = result.saved_path

    if result.indexed:
        body["message"] = "File uploaded and indexed successfully"
    elif result.skipped:
        body["message"] = "File uploaded but indexing skipped"
    else:
        body["message"] = "File uploaded but indexing failed"

    return body
