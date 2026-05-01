# app/rag_backend/search.py
from __future__ import annotations
from typing import List, Tuple, Dict, Any
import json
import math

from .indexer import (
    embedding_model,
    collection,
    sqlite_cur as cursor,  # FTS5 için sqlite cursor (indexer.py'de oluşturuluyor)
)
from . import TOP_K, VECTOR_WEIGHT, BM25_WEIGHT


# -----------------------------
# Yardımcılar
# -----------------------------
def _extract_fname(metadata_json: str | None) -> str:
    """metadata JSON içinden dosya adını çıkarmaya çalışır."""
    if not metadata_json:
        return "unknown"
    try:
        meta = json.loads(metadata_json)
        # yaygın alan adları
        return meta.get("file_name") or meta.get("source") or meta.get("path") or "unknown"
    except Exception:
        return "unknown"


def _min_max_scale(values: List[float]) -> List[float]:
    """[vmin..vmax] → [0..1] ölçekleme. Tüm değerler aynıysa 0.5 döndür."""
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if math.isclose(vmin, vmax):
        return [0.5 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


# -----------------------------
# Arama fonksiyonları
# -----------------------------
def chroma_search(query: str, top_k: int = TOP_K, include_metadata: bool = False) -> List[Tuple]:
    """
    ChromaDB üzerinde semantik arama.
    Dönüş: (chunk_text, distance, file_name)

    Not: Index tarafında embedding_model.encode() ile manuel embedding
    üretildiği için, query tarafında da aynı model kullanılır.
    query_texts yerine query_embeddings ile tutarlılık sağlanır.
    """
    if collection is None:
        return []

    query_embedding = embedding_model.encode([query], convert_to_numpy=True).tolist()
    res = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "distances", "metadatas"],
    )
    docs = (res.get("documents") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]

    out: List[Tuple] = []
    for doc, dist, meta in zip(docs, dists, metas):
        # meta dict ise doğrudan, string ise json.loads ile al
        if isinstance(meta, dict):
            fname = meta.get("file_name") or meta.get("source") or meta.get("path") or "unknown"
            metadata = meta
        else:
            fname = _extract_fname(meta)
            try:
                metadata = json.loads(meta) if meta else {}
            except Exception:
                metadata = {}
        if include_metadata:
            out.append((str(doc), float(dist), fname, metadata))
        else:
            out.append((str(doc), float(dist), fname))
    return out


def bm25_search(query: str, top_k: int = TOP_K, include_metadata: bool = False) -> List[Tuple]:
    """
    SQLite FTS5 üzerinde anahtar kelime araması.
    Dönüş: (chunk_text, bm25_score, file_name)  -- Not: bm25_score'da DÜŞÜK değer daha iyi.
    """
    if cursor is None:
        return []

    # FTS5 MATCH söz dizimi: basit halde, gelen metni doğrudan kullanıyoruz.
    sql = """
        SELECT content, bm25(documents) AS score, metadata
        FROM documents
        WHERE documents MATCH ?
        ORDER BY score ASC
        LIMIT ?
    """
    try:
        cursor.execute(sql, (query, int(top_k)))
        rows = cursor.fetchall()
    except Exception:
        return []

    out: List[Tuple] = []
    for content, score, metadata_json in rows:
        fname = _extract_fname(metadata_json)
        try:
            metadata = json.loads(metadata_json) if metadata_json else {}
        except Exception:
            metadata = {}
        if include_metadata:
            out.append((str(content), float(score), fname, metadata))
        else:
            out.append((str(content), float(score), fname))
    return out


def hybrid_search(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Chroma (semantic) + BM25 (keyword) skorlarını normalize edip ağırlıklarla birleştir.
    Dönüş: [{'chunk': str, 'score': float(0..1), 'file_name': str}, ...]  skora göre azalan
    """
    # 1) alt aramalar
    chroma_results = chroma_search(query, top_k=top_k, include_metadata=True)   # (text, distance, fname, metadata)
    bm25_results   = bm25_search(query, top_k=top_k, include_metadata=True)     # (text, bm25,   fname, metadata)

    # 2) Chroma: distance -> similarity (1 - d), ardından normalize
    ch_texts = [t for t, _, _, _ in chroma_results]
    ch_fnames = [f for _, _, f, _ in chroma_results]
    ch_metas = [m for _, _, _, m in chroma_results]
    ch_sims_raw = [max(0.0, 1.0 - d) for _, d, _, _ in chroma_results]  # d ~ [0,2], güvenli kırpma
    ch_sims = _min_max_scale(ch_sims_raw)

    # 3) BM25: küçük değer daha iyi → negatifine çevirip normalize et
    bm_texts = [t for t, _, _, _ in bm25_results]
    bm_fnames = [f for _, _, f, _ in bm25_results]
    bm_metas = [m for _, _, _, m in bm25_results]
    bm_scores_raw = [-s for _, s, _, _ in bm25_results]  # büyük değer daha iyi olacak
    bm_scores = _min_max_scale(bm_scores_raw)

    # 4) Birleştir (text bazında)
    combined: Dict[str, Dict[str, Any]] = {}

    # Chroma katkısı
    for text, sim, fname, metadata in zip(ch_texts, ch_sims, ch_fnames, ch_metas):
        if text not in combined:
            combined[text] = {"score": 0.0, "file_name": fname, "metadata": metadata}
        combined[text]["score"] = float(combined[text]["score"]) + float(sim) * float(VECTOR_WEIGHT)

    # BM25 katkısı
    for text, sc, fname, metadata in zip(bm_texts, bm_scores, bm_fnames, bm_metas):
        if text not in combined:
            combined[text] = {"score": 0.0, "file_name": fname, "metadata": metadata}
        combined[text]["score"] = float(combined[text]["score"]) + float(sc) * float(BM25_WEIGHT)
        if not combined[text].get("metadata") and metadata:
            combined[text]["metadata"] = metadata
        # Eğer farklı kaynak isimleri varsa ilkini koruyoruz; istersen burada tercih yapabilirsin.

    if not combined:
        return []

    # 5) Skoru [0,1] aralığına kırp
    results = [
        {
            "chunk": text,
            "score": max(0.0, min(1.0, float(data["score"]))),
            "file_name": str(data["file_name"]),
            "metadata": data.get("metadata") if isinstance(data.get("metadata"), dict) else {},
        }
        for text, data in combined.items()
    ]

    # 6) Sırala ve top_k
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


__all__ = ["chroma_search", "bm25_search", "hybrid_search"]
