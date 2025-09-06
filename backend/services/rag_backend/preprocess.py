# app/services/rag_backend/preprocess.py
from __future__ import annotations
from typing import List, Dict
import os
import re

# Tokenizer opsiyonel: yoksa kelime-bazlı chunking'e düşeceğiz
try:
    from transformers import AutoTokenizer  # local_files_only ile yükleyeceğiz
    _HAS_TRANSFORMERS = True
except Exception:
    AutoTokenizer = None
    _HAS_TRANSFORMERS = False

# -----------------------------
# .env'den konfig (token bazlı)
# -----------------------------
# T5 tokenizer klasörü (önce RAG özel, yoksa T5 ile aynı)
_RAG_TOK_DIR = os.getenv("RAG_TOKENIZER_DIR") or os.getenv("T5_TOKENIZER_DIR", "assets/models/t5/tokenizer")
# Token bazlı chunk parametreleri
CHUNK_TOKENS = int(os.getenv("RAG_CHUNK_TOKENS", os.getenv("RAG_CHUNK_SIZE", 90)))           # eski isim desteği
OVERLAP_TOKENS = int(os.getenv("RAG_CHUNK_OVERLAP_TOKENS", os.getenv("RAG_CHUNK_OVERLAP", 20)))

CHUNK_SIZE = CHUNK_TOKENS
CHUNK_OVERLAP = OVERLAP_TOKENS
# -----------------------------
# Yedek (kelime bazlı) parametreler
# -----------------------------
WORD_CHUNK_SIZE = int(os.getenv("RAG_WORD_CHUNK_SIZE", 90))
WORD_OVERLAP = int(os.getenv("RAG_WORD_CHUNK_OVERLAP", 20))

# -----------------------------
# Tokenizer'ı (varsa) yükle
# -----------------------------
_tokenizer = None
if _HAS_TRANSFORMERS:
    try:
        _tokenizer = AutoTokenizer.from_pretrained(_RAG_TOK_DIR, use_fast=True, local_files_only=True)
    except Exception:
        _tokenizer = None


def clean_text(text: str) -> str:
    """
    Basit temizlik: soft hyphen vs. kaldır, boşlukları normalize et.
    """
    if not text:
        return ""
    # soft hyphen, control chars vb.
    text = text.replace("\u00ad", "")
    # çoklu boşluk ve satır sonlarını sadeleştir
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# -----------------------------
# TOKEN BAZLI CHUNKING
# -----------------------------
def _chunk_by_tokens(text: str, chunk_tokens: int = CHUNK_TOKENS, overlap_tokens: int = OVERLAP_TOKENS) -> List[str]:
    """
    Tokenizer mevcutsa token bazlı chunking. Special tokens eklemiyoruz.
    """
    assert _tokenizer is not None, "Tokenizer yüklü değil."
    text = clean_text(text)
    # not: encode/decode sırasında special token eklemiyoruz
    input_ids = _tokenizer.encode(text, add_special_tokens=False)
    if not input_ids:
        return []

    chunks: List[str] = []
    start = 0
    L = len(input_ids)

    while start < L:
        end = min(start + chunk_tokens, L)
        piece_ids = input_ids[start:end]
        # T5 için decode ederken özel tokenları atla
        piece = _tokenizer.decode(piece_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if piece.strip():
            chunks.append(piece.strip())
        if end == L:
            break
        # overlap kadar geri sar
        start = max(0, end - overlap_tokens)

    return chunks


# -----------------------------
# KELİME BAZLI CHUNKING (YEDEK)
# -----------------------------
def _chunk_by_words(text: str, chunk_size: int = WORD_CHUNK_SIZE, overlap: int = WORD_OVERLAP) -> List[str]:
    """
    Tokenizer yoksa veya hata olursa kelime bazlı yedek chunking.
    """
    text = clean_text(text)
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    n = len(words)

    while start < n:
        end = min(start + chunk_size, n)
        piece = " ".join(words[start:end]).strip()
        if piece:
            chunks.append(piece)
        if end == n:
            break
        start = max(0, end - overlap)

    return chunks


def chunk_text(text: str,
               chunk_tokens: int = CHUNK_TOKENS,
               overlap_tokens: int = OVERLAP_TOKENS) -> List[str]:
    """
    Dışa açık chunking API:
      - Tokenizer yüklüyse token bazlı,
      - değilse kelime bazlı fallback.
    """
    if _tokenizer is not None:
        try:
            return _chunk_by_tokens(text, chunk_tokens, overlap_tokens)
        except Exception:
            # herhangi bir tokenizer hatasında kelime bazlıya düş
            pass
    return _chunk_by_words(text, WORD_CHUNK_SIZE, WORD_OVERLAP)


def preprocess_documents(documents: List[Dict]) -> List[Dict]:
    """
    Input:  [{'file_name': str, 'text': str}, ...]
    Output: [{'file_name': str, 'chunk': str, 'order': int}, ...]
    Not: Şemanın router/indexer/search ile uyumlu kalması için
         alan adlarını standartlaştırıyoruz.
    """
    results: List[Dict] = []
    for doc in documents:
        file_name = doc.get("file_name") or doc.get("name") or "unknown"
        raw_text = doc.get("text") or doc.get("content") or ""
        pieces = chunk_text(raw_text, CHUNK_TOKENS, OVERLAP_TOKENS)
        for i, p in enumerate(pieces):
            results.append({
                "file_name": file_name,
                "chunk": p,
                "order": i
            })
    return results
