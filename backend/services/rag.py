# app/services/rag.py
from __future__ import annotations
from typing import List, Tuple, Dict, Any

# Backend (defterden taşıdığın kodların modüler hali)
# Aşağıdaki importlar, app/services/rag_backend/ altına koyduğun dosyalardan gelmelidir.
from services.rag_backend.search import hybrid_search
from services.rag_backend.prompt import create_context
from services.rag_backend.websearch import process_web_results
from services.rag_backend import TOP_K as BACKEND_TOP_K, RAG_MAX_CTX_TOKENS as BACKEND_MAX_CTX_TOKENS # __init__.py dosyasından alıyor.

import os
# .env ile ayarlanabilir; yoksa bu varsayılanları kullanır
RAG_WEB_MIN_STRENGTH = float(os.getenv("RAG_WEB_MIN_STRENGTH", "0.75"))
WEB_CHUNK_SUPPORT_THRESHOLD = float(os.getenv("WEB_CHUNK_SUPPORT_THRESHOLD", "0.70"))

def _web_strength(chunks):
    """
    Web parçalarının güvenilirliğini tek sayıya indirger.
    - top: en yüksek skor
    - support: skoru >= WEB_CHUNK_SUPPORT_THRESHOLD olan parça sayısı
    Güç = 0.5*top + 0.5*min(1, support/3)
    """
    if not chunks:
        return 0.0
    top = max(float(c.get("score", 0.0)) for c in chunks)
    support = sum(1 for c in chunks if float(c.get("score", 0.0)) >= WEB_CHUNK_SUPPORT_THRESHOLD)
    return 0.5*top + 0.5*min(1.0, support/3.0)


class RAGService:
    """
    Router'ın konuştuğu *adaptör* katmanı.
    - Backend'den (hybrid_search) [{chunk, score, ...}] formatında sonuç alır.
    - Skorları 0..1 aralığında garanti altına alır (yüzde gelirse normalize eder).
    - Tek bir context string üretir (token sınırlı) ve T5Service'e list[str] olarak verir.
    - Router için (chunks_text_list, top_score_float) döndürür.
    """

    def __init__(self, cfg: Dict[str, Any]):
        # Eşik 0..1 aralığında. %50 için 0.5 yazmalısın.
        self.thr: float = float(cfg.get("RAG_SCORE_THRESHOLD", 0.4))
        # Kaç parça alalım?
        self.top_k: int = int(cfg.get("RAG_TOP_K", BACKEND_TOP_K))
        # Context token limiti (backend prompt.create_context ile uyumlu)
        self.max_ctx_tokens: int = int(cfg.get("RAG_MAX_CTX_TOKENS", BACKEND_MAX_CTX_TOKENS))


    def retrieve(self, question: str, use_internet: bool = False, web_only: bool = False):
        """
        DÖNÜŞ: (contexts:list[str], best_score:float|None, sources:list[str])
        - contexts: create_context(...) sonucu tek bir büyük metni [list] içinde döndürür (yoksa []).
        - best_score: lokal RAG top score (web-only’ken None olabilir).
        - sources: "local:<dosya>" ve/veya web URL başlıkları.
        """
        # 1) Lokal hibrit arama (her zaman çalıştırıyoruz; gerekirse sadece score için)
        retrieved = hybrid_search(question, top_k=self.top_k)  # [{'chunk','score','file_name'}, ...]
        top_score = float(max((r.get("score", 0.0) for r in retrieved), default=0.0))
        sources_local = [f"local:{r.get('file_name','unknown')}" for r in retrieved]

        # 2) Web chunk'ları (gerekirse)
        web_chunks = []
        sources_web = []
        if use_internet and question.strip():
            try:
                web_chunks = process_web_results(question) or []
                sources_web = [c.get('source','') for c in web_chunks if c.get('source')]
            except Exception:
                web_chunks = []
                sources_web = []

        # 3) Yardımcı (chunks -> context metni)
        def ctx_from(items):
            return [create_context(items, max_tokens=self.max_ctx_tokens, question=question)] if items else []

        # 4) Karar ağacı
        if not use_internet:
            # Web kapalı: klasik RAG
            if top_score >= self.thr:
                return ctx_from(retrieved), top_score, sources_local
            else:
                return [], top_score, []

        # --- YENİ: skor-bazlı web kapısı ---
        local_ok = bool(retrieved) and (top_score is not None) and (top_score >= self.thr)

        ws = _web_strength(web_chunks) if (use_internet and web_chunks) else 0.0
        eligible_web = web_chunks if (use_internet and ws >= RAG_WEB_MIN_STRENGTH) else []

        if web_only:
            # "web_only" istense bile zayıf web'i zorlamıyoruz
            if eligible_web:
                contexts = ctx_from(eligible_web)
                return contexts, (top_score if retrieved else None), sources_web
            else:
                # web zayıf: lokal iyiyse lokal, değilse model-only
                if local_ok:
                    contexts = ctx_from(retrieved)
                    return contexts, top_score, sources_local
                else:
                    return [], top_score, []
        else:
            # web_only değil: dört durumu kapsa
            if local_ok and eligible_web:
                all_chunks = retrieved + eligible_web
                contexts = ctx_from(all_chunks)
                return contexts, top_score, (sources_local + sources_web)
            elif local_ok:
                contexts = ctx_from(retrieved)
                return contexts, top_score, sources_local
            elif eligible_web:
                contexts = ctx_from(eligible_web)
                return contexts, (top_score if retrieved else None), sources_web
            else:
                # hem yerel zayıf hem web zayıf → model-only
                return [], top_score, []
