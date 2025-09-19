# app/services/rag_backend/indexer.py
from __future__ import annotations
from typing import List, Dict, Tuple
import os
import json
import sqlite3
from datetime import datetime

from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm

# .env üzerinden ayarlar (gerekirse)
from . import EMBED_MODEL, CHROMA_PATH

# -------------------------
# Model & Chroma init
# -------------------------
embedding_model = SentenceTransformer(EMBED_MODEL, device="cpu")

# Chroma kalıcı istemci ve koleksiyon
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "pathfinder_corpus")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)

# -------------------------
# SQLite (FTS5) init
# -------------------------
SQLITE_PATH = os.getenv("RAG_SQLITE_PATH", os.path.join(CHROMA_PATH, "bm25.sqlite"))
os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)
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


def _normalize_record(c: Dict) -> Tuple[str, str, str]:
    """
    Bir chunk kaydını normalize et.
    Desteklenen giriş şemaları:
      1) {'file_name','chunk','order'}  -> chunk_id otomatik üretilecek
      2) {'file_name','content','chunk_id'} (senin eski şeman)
    Dönüş: (chunk_id, text, file_name)
    """
    file_name = c.get("file_name") or c.get("source") or "unknown"

    if "chunk" in c:  # bizim preprocess.py şeması
        text = c["chunk"]
        order = c.get("order", 0)
        chunk_id = c.get("chunk_id") or f"{file_name}_chunk_{order}"
    else:  # alternatif şema
        text = c.get("content", "")
        chunk_id = c.get("chunk_id") or f"{file_name}_chunk_0"

    return chunk_id, text, file_name


def add_chunks_to_db(chunks: List[Dict], batch_size: int = 100) -> None:
    """
    Chunk'ları ChromaDB ve SQLite FTS5 veritabanına toplu (batch) şekilde ekler.
    Büyük veri setlerinde ciddi performans sağlar.
    """
    if not chunks:
        print("[indexer] Eklenecek chunk yok.")
        return

    # Toplam sayaç
    total = len(chunks)

    for i in tqdm(range(0, total, batch_size), desc="Toplu embedding ve DB ekleme"):
        batch = chunks[i:i + batch_size]

        # Normalize et
        norm = [_normalize_record(c) for c in batch]  # [(chunk_id, text, file_name), ...]
        ids = [cid for cid, _, _ in norm]
        texts = [txt for _, txt, _ in norm]
        sources = [fn for _, _, fn in norm]

        # ---- Embedding (toplu)
        # show_progress_bar=False CPU'da da yeterli
        embs = embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()

        # ---- ChromaDB'ye ekle
        metadatas = [{
            "source": src,
            "type": "local",
            "timestamp": datetime.now().isoformat()
        } for src in sources]

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
    print(f"[indexer] {total} chunk başarıyla eklendi → "
          f"Chroma('{CHROMA_COLLECTION}') ve SQLite('{SQLITE_PATH}')")


def close():
    """Uygulama kapanırken çağırmak istersen."""
    try:
        conn.commit()
    finally:
        conn.close()

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
        global Collection
        collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)
        # FTS5 temizliği (tabloyu boşalt)
        cursor.execute("DELETE FROM documents;")
        conn.commit()
        print("[indexer] Reset tamamlandı.")

    # Belgeleri yükle → chunk'la → ekle
    docs = load_documents_from_folder(args.src)
    chunks = preprocess_documents(docs)
    print(f"[indexer] {len(docs)} belge, {len(chunks)} chunk üretildi.")
    add_chunks_to_db(chunks, batch_size=args.batch_size)
    print("[indexer] İndeksleme bitti.")
