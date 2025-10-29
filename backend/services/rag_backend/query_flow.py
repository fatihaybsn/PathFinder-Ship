# app/services/rag_backend/query_flow.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
import os

from services.t5 import T5Service
from services.rag_backend.search import hybrid_search
from services.rag_backend.prompt import create_context

# web araması opsiyonel
try:
    from services.rag_backend.websearch import process_web_results
    _HAS_WEB = True
except Exception:
    _HAS_WEB = False

# Eşikler (.__init__.py üzerinden; .env ile yönetilir)
from services.rag_backend import RAG_WEB_MIN_STRENGTH, WEB_CHUNK_SUPPORT_THRESHOLD


# ---- Yardımcı: cfg al (dotenv ile .env okunmuş olsun) ----
def _load_cfg() -> dict:
    # main.py zaten dotenv'i yükledi; yine de env'den okumak güvenli
    return {
        "RAG_SCORE_THRESHOLD": float(os.getenv("RAG_SCORE_THRESHOLD", "0.5")),
        "RAG_TOP_K": int(os.getenv("RAG_TOP_K", "4")),
        "RAG_MAX_CTX_TOKENS": int(os.getenv("RAG_MAX_CTX_TOKENS", "512")),
        # T5 için gerekenler de env'de olmalı (T5_TOKENIZER_DIR, T5_ENCODER, T5_DECODER, BOT_NAME, vb.)
        "T5_TOKENIZER_DIR": os.getenv("T5_TOKENIZER_DIR", "assets/models/t5/tokenizer"),
        "T5_ENCODER": os.getenv("T5_ENCODER", "assets/models/t5/encoder_model_int8.onnx"),
        "T5_DECODER": os.getenv("T5_DECODER", "assets/models/t5/decoder_model_int8.onnx"),
        "BOT_NAME": os.getenv("BOT_NAME", "Passenger-Bot"),
    }


def _web_strength(chunks: List[Dict[str, Any]]) -> float:
    """
    Web parçalarının güvenilirliğini tek sayıya indirger.
      - top: en yüksek skor
      - support: skoru >= WEB_CHUNK_SUPPORT_THRESHOLD olan parça sayısı
    Formül:
      strength = 0.5*top + 0.5*min(1, support/3)
    """
    if not chunks:
        return 0.0
    top = max(float(c.get("score", 0.0)) for c in chunks)
    support = sum(1 for c in chunks if float(c.get("score", 0.0)) >= WEB_CHUNK_SUPPORT_THRESHOLD)
    return 0.5 * top + 0.5 * min(1.0, support / 3.0)


# ---- Sorgu yeniden yazma: ONNX T5 ile ----
def rewrite_query(question: str, t5: Optional[T5Service] = None) -> str:
    """
    Kullanıcı sorusunu web aramaları için optimize eder.
    ONNX T5'i 'chat' modunda, katı bir talimatla kullanıyoruz.
    """
    if not question or not question.strip():
        return ""

    close_ended = (
        "Rewrite the user question into a concise web search query. "
        "Use plain keywords, remove filler words, no quotes, no URLs, no punctuation at the end. "
        "Return ONLY the rewritten query, nothing else."
    )
    prompt = f"{close_ended}\nUser: {question}\nAssistant:"
    try:
        _t5 = t5 or T5Service(_load_cfg())  # geçici oluştur; uygulamada genelde singleton reuse edilir
        optimized = _t5.chat(prompt)
        optimized = optimized.strip().splitlines()[0].strip()  # tek satıra indir
        return optimized or question
    except Exception:
        return question  # hata halinde orijinali kullan


# ---- Çekirdek: Soru cevaplama akışı ----
def ask_question(question: str, use_internet: bool = False, t5: Optional[T5Service] = None) -> Dict[str, Any]:
    """
    RAG kararı skor-bazlıdır:
      - local_ok := (hibrit en iyi skor >= RAG_SCORE_THRESHOLD)
      - web parçaları ancak web_strength >= RAG_WEB_MIN_STRENGTH ise dahil edilir.
    Dört olasılık:
      1) local_ok & eligible_web   -> hybrid_local_web
      2) local_ok & !eligible_web  -> local_only
      3) !local_ok & eligible_web  -> web_only
      4) !local_ok & !eligible_web -> model_only
    """
    cfg = _load_cfg()
    thr = cfg["RAG_SCORE_THRESHOLD"]
    top_k = cfg["RAG_TOP_K"]
    max_ctx_tokens = cfg["RAG_MAX_CTX_TOKENS"]

    if not question or not question.strip():
        return {"task": "qa", "question": question, "response": "", "rag_required": 0, "error": "empty question"}

    # 1) Sorguyu optimize et (web ve hibrit arama için faydalı)
    opt_query = rewrite_query(question, t5=t5)

    # 2) Hibrit arama (vektör + BM25 -> normalize + ağırlık)
    retrieved: List[Dict[str, Any]] = hybrid_search(opt_query, top_k=top_k)  # [{'chunk','score','file_name'?}, ...]
    top_score = float(max((r.get("score", 0.0) for r in retrieved), default=0.0))

    # 3) Gerekirse web sonuçlarını al
    web_chunks: List[Dict[str, Any]] = []
    if use_internet and _HAS_WEB:
        try:
            web_chunks = process_web_results(opt_query) or []  # [{'chunk','score'?, 'source'?...}]
        except Exception:
            web_chunks = []

    # 4) Web kapısı ve yerel uygunluk
    local_ok = bool(retrieved) and (top_score is not None) and (top_score >= thr)
    ws = _web_strength(web_chunks) if (use_internet and web_chunks) else 0.0
    eligible_web = web_chunks if (use_internet and ws >= RAG_WEB_MIN_STRENGTH) else []

    # 5) Cevabı üret (tokenizer-bilinçli context kesimi ile)
    _t5 = t5 or T5Service(cfg)

    def _ctx(items: List[Dict[str, Any]]) -> List[str]:
        ctx_str = create_context(items, max_tokens=max_ctx_tokens, question=question)
        return [ctx_str] if ctx_str else []

    if local_ok and eligible_web:
        # Hibrit: güçlü yerel + güçlü web
        all_chunks = retrieved + eligible_web
        ctx = _ctx(all_chunks)
        reply = _t5.answer(question, context=ctx)
        meta = {
            "task": "qa",
            "mode": "hybrid_local_web",
            "rag_required": 1,
            "top_score": round(float(top_score), 3),
            "threshold": thr,
            "optimized_query": opt_query,
            "num_chunks": len(all_chunks),
            "web_strength": round(float(ws), 3),
        }
        return {"question": question, "response": reply, "meta": meta}

    elif local_ok:
        # Sadece yerel yeterli
        ctx = _ctx(retrieved)
        reply = _t5.answer(question, context=ctx)
        meta = {
            "task": "qa",
            "mode": "local_only",
            "rag_required": 1,
            "top_score": round(float(top_score), 3),
            "threshold": thr,
            "optimized_query": opt_query,
            "num_chunks": len(retrieved),
            "web_strength": round(float(ws), 3),
        }
        return {"question": question, "response": reply, "meta": meta}

    elif eligible_web:
        # Sadece web yeterli
        ctx = _ctx(eligible_web)
        reply = _t5.answer(question, context=ctx)
        meta = {
            "task": "qa",
            "mode": "web_only",
            "rag_required": 1,
            "top_score": round(float(top_score), 3),
            "threshold": thr,
            "optimized_query": opt_query,
            "num_chunks": len(eligible_web),
            "web_strength": round(float(ws), 3),
        }
        return {"question": question, "response": reply, "meta": meta}

    else:
        # Hem yerel zayıf hem web zayıf → model-only (bağlam yok)
        reply = _t5.answer(question, context=None)
        meta = {
            "task": "qa",
            "mode": "model_only",
            "rag_required": 0,
            "top_score": round(float(top_score), 3),
            "threshold": thr,
            "optimized_query": opt_query,
            "num_chunks": 0,
            "web_strength": round(float(ws), 3),
        }
        return {"question": question, "response": reply, "meta": meta}
