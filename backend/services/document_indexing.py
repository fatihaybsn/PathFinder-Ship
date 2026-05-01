from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from config import BACKEND_ROOT, CFG
from schemas.pipeline import IndexingResult

DEFAULT_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}
READ_CHUNK_SIZE = 1024 * 1024


class UploadIndexingError(Exception):
    def __init__(self, code: str, message: str, *, status_code: int = 400, filename: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.filename = filename


def allowed_extensions(cfg: dict[str, Any] | None = None) -> set[str]:
    raw = str((cfg or CFG).get("RAG_UPLOAD_ALLOWED_EXTENSIONS") or "")
    items = [item.strip().lower() for item in raw.split(",") if item.strip()]
    normalized = {item if item.startswith(".") else f".{item}" for item in items}
    return normalized or set(DEFAULT_ALLOWED_EXTENSIONS)


def sanitize_upload_name(filename: str | None) -> tuple[str, str, str]:
    original = Path(filename or "upload.bin").name
    suffix = Path(original).suffix.lower()
    stem = Path(original).stem or "upload"
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return original, safe_stem or "upload", suffix


def _relative_backend_path(path: str | Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(BACKEND_ROOT).resolve()).as_posix()
    except Exception:
        return Path(path).name


def _document_id_for_digest(digest: str) -> str:
    return f"doc_{digest[:16]}"


async def _save_upload(file: Any, cfg: dict[str, Any]) -> tuple[Path, str, str, str, int]:
    filename, safe_stem, suffix = sanitize_upload_name(getattr(file, "filename", None))
    if suffix not in allowed_extensions(cfg):
        raise UploadIndexingError(
            "unsupported_file_type",
            f"Unsupported file type: {suffix or 'none'}",
            status_code=400,
            filename=filename,
        )

    max_bytes = int(cfg.get("UPLOAD_MAX_BYTES") or 10 * 1024 * 1024)
    corpus_dir = Path(str(cfg.get("RAG_CORPUS_DIR") or CFG["RAG_CORPUS_DIR"])).resolve()
    corpus_dir.mkdir(parents=True, exist_ok=True)

    temp_path = corpus_dir / f".upload-{uuid4().hex}.tmp"
    digest = hashlib.sha256()
    total = 0

    try:
        with open(temp_path, "wb") as out:
            while True:
                chunk = await file.read(READ_CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise UploadIndexingError(
                        "file_too_large",
                        f"File exceeds upload limit of {max_bytes} bytes",
                        status_code=413,
                        filename=filename,
                    )
                digest.update(chunk)
                out.write(chunk)

        content_hash = digest.hexdigest()
        final_path = corpus_dir / f"{safe_stem}__{content_hash[:8]}{suffix}"
        temp_path.replace(final_path)
        return final_path, filename, content_hash, _document_id_for_digest(content_hash), total
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _add_chunks_to_index(chunks: list[dict[str, Any]], document_id: str) -> list[str]:
    from services.rag_backend.indexer import add_or_replace_document_chunks

    return add_or_replace_document_chunks(chunks, document_id=document_id)


def _failure_result(
    *,
    filename: str,
    document_id: str | None,
    saved_path: str | None,
    metadata: dict[str, Any],
    error: str | None = None,
    skipped: bool = False,
    skip_reason: str | None = None,
    warnings: list[str] | None = None,
) -> IndexingResult:
    return IndexingResult(
        filename=filename,
        document_id=document_id,
        saved_path=saved_path,
        indexed=False,
        indexed_chunk_count=0,
        skipped=skipped,
        skip_reason=skip_reason,
        warnings=list(warnings or []),
        error=error,
        metadata=metadata,
    )


async def index_upload_file(file: Any, cfg: dict[str, Any] | None = None) -> IndexingResult:
    cfg = cfg or CFG
    saved_path, filename, content_hash, document_id, byte_count = await _save_upload(file, cfg)
    relative_path = _relative_backend_path(saved_path)
    uploaded_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "document_id": document_id,
        "file_name": saved_path.name,
        "source": saved_path.name,
        "saved_path": relative_path,
        "content_hash": content_hash,
        "uploaded_at": uploaded_at,
        "byte_count": byte_count,
    }

    if byte_count == 0:
        return _failure_result(
            filename=filename,
            document_id=document_id,
            saved_path=relative_path,
            metadata=metadata,
            skipped=True,
            skip_reason="empty_file",
        )

    try:
        from services.rag_backend.io_loader import load_document_from_file
        from services.rag_backend.preprocess import preprocess_documents

        document = load_document_from_file(saved_path, metadata=metadata)
    except ValueError as exc:
        return _failure_result(
            filename=filename,
            document_id=document_id,
            saved_path=relative_path,
            metadata=metadata,
            skipped=True,
            skip_reason=str(exc) or "unsupported_file_type",
            error=str(exc) or "unsupported_file_type",
        )
    except Exception as exc:
        return _failure_result(
            filename=filename,
            document_id=document_id,
            saved_path=relative_path,
            metadata=metadata,
            error="text_extraction_failed",
            warnings=[str(exc)],
        )

    if not str(document.get("content") or document.get("text") or "").strip():
        return _failure_result(
            filename=filename,
            document_id=document_id,
            saved_path=relative_path,
            metadata=metadata,
            skipped=True,
            skip_reason="empty_extracted_text",
        )

    chunks = preprocess_documents([document])
    if not chunks:
        return _failure_result(
            filename=filename,
            document_id=document_id,
            saved_path=relative_path,
            metadata=metadata,
            skipped=True,
            skip_reason="no_chunks",
        )

    try:
        warnings = _add_chunks_to_index(chunks, document_id)
    except Exception as exc:
        return _failure_result(
            filename=filename,
            document_id=document_id,
            saved_path=relative_path,
            metadata=metadata,
            error="index_update_failed",
            warnings=[str(exc)],
        )

    return IndexingResult(
        filename=filename,
        document_id=document_id,
        saved_path=relative_path,
        indexed=True,
        indexed_chunk_count=len(chunks),
        warnings=warnings,
        metadata=metadata,
    )


def upload_error_response(exc: UploadIndexingError) -> dict[str, Any]:
    return {
        "ok": False,
        "message": exc.message,
        "filename": exc.filename,
        "indexed": False,
        "indexed_chunk_count": 0,
        "document_id": None,
        "warnings": [],
        "error": exc.code,
    }
