# app/services/rag_backend/indexer.py
from __future__ import annotations
from typing import Any, List, Dict, Tuple
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm

# .env üzerinden ayarlar (gerekirse)
from . import EMBED_MODEL, CHROMA_PATH
from config import CFG

logger = logging.getLogger(__name__)

# -------------------------
# Model & Chroma init
# -------------------------
embedding_model = SentenceTransformer(EMBED_MODEL, device="cpu")

# Chroma kalıcı istemci ve koleksiyon
CHROMA_COLLECTION = str(CFG.get("CHROMA_COLLECTION", "pathfinder_corpus"))
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)

# -------------------------
# SQLite (FTS5) init
# -------------------------
SQLITE_PATH = str(CFG.get("RAG_SQLITE_PATH") or Path(CHROMA_PATH) / "bm25.sqlite")
Path(SQLITE_PATH).parent.mkdir(parents=True, exist_ok=True)
conn = sqlite3.connect(SQLITE_PATH)
cursor = conn.cursor()
sqlite_cur = cursor
sqlite_conn = conn  # (istersen ileride lazım olur)
# FTS5 tablo (content ve metadata json)
cursor.execute("""
CREATE VIRTUAL TABLE IF NOT EXISTS documents
USING fts5(content, metadata, tokenize = 'porter');
""")
conn.commit()


def _metadata_for_chunk(c: Dict, file_name: str, text: str) -> Dict[str, Any]:
    metadata = dict(c.get("metadata") or {})
    for key in ("document_id", "source", "uploaded_at", "content_hash", "saved_path", "chunk_index"):
        if c.get(key) is not None:
            metadata.setdefault(key, c.get(key))

    metadata.setdefault("file_name", file_name)
    metadata.setdefault("source", c.get("source") or file_name)
    metadata.setdefault("type", "local")
    metadata.setdefault("timestamp", datetime.now().isoformat())

    # Chroma metadata values must be scalar.
    return {
        str(key): value
        for key, value in metadata.items()
        if value is not None and isinstance(value, (str, int, float, bool))
    }


def _normalize_record(c: Dict) -> Tuple[str, str, str, Dict[str, Any]]:
    """
    Bir chunk kaydını normalize et.
    Desteklenen giriş şemaları:
      1) {'file_name','chunk','order'}  -> chunk_id otomatik üretilecek
      2) {'file_name','content','chunk_id'} (senin eski şeman)
    Dönüş: (chunk_id, text, file_name, metadata)
    """
    incoming_metadata = dict(c.get("metadata") or {})
    file_name = c.get("file_name") or incoming_metadata.get("file_name") or c.get("source") or "unknown"

    if "chunk" in c:  # bizim preprocess.py şeması
        text = c["chunk"]
        order = c.get("chunk_index", c.get("order", 0))
    else:  # alternatif şema
        text = c.get("content", "")
        order = c.get("chunk_index", 0)

    document_id = c.get("document_id") or incoming_metadata.get("document_id")
    if c.get("chunk_id"):
        chunk_id = c["chunk_id"]
    elif document_id:
        chunk_id = f"{document_id}_chunk_{order}"
    else:
        chunk_id = f"{file_name}_chunk_{order}"

    metadata = _metadata_for_chunk(c, file_name, text)
    metadata.setdefault("chunk_index", int(order) if str(order).isdigit() else order)

    return chunk_id, text, file_name, metadata


def add_chunks_to_db(chunks: List[Dict], batch_size: int = 100) -> None:
    """
    Chunk'ları ChromaDB ve SQLite FTS5 veritabanına toplu (batch) şekilde ekler.
    Büyük veri setlerinde ciddi performans sağlar.
    """
    if not chunks:
        logger.info("No chunks to add.")
        return

    # Toplam sayaç
    total = len(chunks)

    for i in tqdm(range(0, total, batch_size), desc="Toplu embedding ve DB ekleme"):
        batch = chunks[i:i + batch_size]

        # Normalize et
        norm = [_normalize_record(c) for c in batch]  # [(chunk_id, text, file_name, metadata), ...]
        ids = [cid for cid, _, _, _ in norm]
        texts = [txt for _, txt, _, _ in norm]
        metadatas = [meta for _, _, _, meta in norm]

        # ---- Embedding (toplu)
        # show_progress_bar=False CPU'da da yeterli
        embs = embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()

        # ---- ChromaDB'ye ekle
        collection.add(
            ids=ids,
            embeddings=embs,
            metadatas=metadatas,
            documents=texts,
        )

        # ---- SQLite FTS5'e ekle (content + metadata)
        sql_rows = [(texts[j], json.dumps(metadatas[j])) for j in range(len(texts))]
        cursor.executemany(
            "INSERT INTO documents(content, metadata) VALUES (?, ?);",
            sql_rows,
        )

    conn.commit()
    logger.info(
        "%s chunks added to Chroma collection '%s' and SQLite '%s'.",
        total,
        CHROMA_COLLECTION,
        Path(SQLITE_PATH).name,
    )


def close():
    """Uygulama kapanırken çağırmak istersen."""
    try:
        conn.commit()
    finally:
        conn.close()


def _ignore_missing_delete_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "not found" in text or "does not exist" in text


def _delete_chroma_document(document_id: str, chunk_ids: List[str]) -> List[str]:
    warnings: List[str] = []
    if collection is None:
        return warnings

    if chunk_ids:
        try:
            collection.delete(ids=chunk_ids)
        except Exception as exc:
            if not _ignore_missing_delete_error(exc):
                warnings.append(f"chroma_delete_by_ids_failed: {exc}")

    try:
        collection.delete(where={"document_id": document_id})
    except Exception as exc:
        if not _ignore_missing_delete_error(exc):
            warnings.append(f"chroma_delete_by_document_id_failed: {exc}")

    return warnings


def _delete_sqlite_document(document_id: str) -> List[str]:
    warnings: List[str] = []
    if cursor is None:
        return warnings

    try:
        cursor.execute("SELECT rowid FROM documents WHERE metadata LIKE ?;", (f"%{document_id}%",))
        row_ids = [row[0] for row in cursor.fetchall()]
        if row_ids:
            cursor.executemany("DELETE FROM documents WHERE rowid = ?;", [(row_id,) for row_id in row_ids])
            conn.commit()
    except Exception as exc:
        warnings.append(f"sqlite_delete_by_document_id_failed: {exc}")

    return warnings


def add_or_replace_document_chunks(chunks: List[Dict], document_id: str, batch_size: int = 100) -> List[str]:
    """
    Upload akışı için duplicate-safe ekleme.
    Aynı document_id yeniden gelirse önce eski Chroma/FTS kayıtları temizlenir,
    sonra mevcut add_chunks_to_db yolu ile tekrar yazılır.
    """
    if not chunks:
        return []
    if not document_id:
        raise ValueError("document_id is required")

    normalized = [_normalize_record(c) for c in chunks]
    chunk_ids = [chunk_id for chunk_id, _, _, _ in normalized]

    warnings: List[str] = []
    warnings.extend(_delete_chroma_document(document_id, chunk_ids))
    warnings.extend(_delete_sqlite_document(document_id))

    add_chunks_to_db(chunks, batch_size=batch_size)
    return warnings

if __name__ == "__main__":
    import argparse
    from .io_loader import load_documents_from_folder
    from .preprocess import preprocess_documents

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Klasör: txt/pdf/docx belgeler")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--reset", action="store_true",
                        help="Tüm koleksiyonu ve FTS içeriğini temizle")
    args = parser.parse_args()

    if args.reset:
        # Chroma koleksiyonunu temizle/yık-yap
        try:
            chroma_client.delete_collection(CHROMA_COLLECTION)
        except Exception:
            pass
        collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)
        # FTS5 temizliği (tabloyu boşalt)
        cursor.execute("DELETE FROM documents;")
        conn.commit()
        logger.info("Indexer reset completed.")

    # Belgeleri yükle → chunk'la → ekle
    docs = load_documents_from_folder(args.src)
    chunks = preprocess_documents(docs)
    logger.info("%s documents produced %s chunks.", len(docs), len(chunks))
    add_chunks_to_db(chunks, batch_size=args.batch_size)
    logger.info("Indexing completed.")
