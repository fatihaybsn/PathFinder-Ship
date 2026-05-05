# app/services/rag.py
from __future__ import annotations
import time
from typing import List, Tuple, Dict, Any, Optional

# Backend (defterden taşıdığın kodların modüler hali)
# Aşağıdaki importlar, app/services/rag_backend/ altına koyduğun dosyalardan gelmelidir.
from services.rag_backend.search import hybrid_search
from services.rag_backend.prompt import create_context
from services.rag_backend.websearch import process_web_results
from services.rag_backend import TOP_K as BACKEND_TOP_K, RAG_MAX_CTX_TOKENS as BACKEND_MAX_CTX_TOKENS # __init__.py dosyasından alıyor.

from config import CFG
from schemas.pipeline import RetrievedChunk, RetrievalResult

RAG_WEB_MIN_STRENGTH = float(CFG.get("RAG_WEB_MIN_STRENGTH", 0.75))
WEB_CHUNK_SUPPORT_THRESHOLD = float(CFG.get("WEB_CHUNK_SUPPORT_THRESHOLD", 0.70))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def build_context_from_chunks(
    chunks: List[RetrievedChunk],
    max_tokens: int,
    question: str,
) -> str:
    """
    RetrievedChunk listesinden create_context-uyumlu dict listesi oluşturup
    context string üretir.  Evidence chunk'ları korunurken context string
    ayrıca üretilir.
    """
    raw = [{"chunk": c.text} for c in chunks]
    context = create_context(raw, max_tokens=max_tokens, question=question)
    if context.strip():
        return context

    # If tokenizer-based trimming cannot produce context, keep a bounded
    # evidence-preserving fallback instead of silently discarding retrieved chunks.
    fallback = "\n\n".join(c.text.strip() for c in chunks if c.text and c.text.strip())
    return fallback[: max(1, int(max_tokens)) * 4]


def _map_local_chunks(
    retrieved: List[Dict[str, Any]],
) -> List[RetrievedChunk]:
    """hybrid_search sonuçlarını RetrievedChunk listesine dönüştürür."""
    out: List[RetrievedChunk] = []
    for i, r in enumerate(retrieved):
        metadata = dict(r.get("metadata") or {})
        metadata.setdefault("file_name", r.get("file_name", "unknown"))
        source_name = metadata.get("source") or r.get("file_name", "unknown")
        out.append(RetrievedChunk(
            text=r.get("chunk", ""),
            source=f"local:{source_name}",
            score=float(r.get("score", 0.0)),
            rank=i + 1,
            retrieval_type="local_hybrid",
            metadata=metadata,
        ))
    return out


def _map_web_chunks(
    web_chunks: List[Dict[str, Any]],
    rank_offset: int = 0,
) -> List[RetrievedChunk]:
    """process_web_results sonuçlarını RetrievedChunk listesine dönüştürür."""
    out: List[RetrievedChunk] = []
    for i, c in enumerate(web_chunks):
        out.append(RetrievedChunk(
            text=c.get("chunk", ""),
            source=c.get("source", ""),
            score=float(c.get("score", 0.0)) if c.get("score") is not None else None,
            rank=rank_offset + i + 1,
            retrieval_type="web",
            metadata={"title": c.get("title", "")},
        ))
    return out


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

    # ------------------------------------------------------------------
    # Structured retrieval — chunk-level evidence korunur
    # ------------------------------------------------------------------

    def retrieve_structured(
        self,
        question: str,
        use_internet: bool = False,
        web_only: bool = False,
    ) -> RetrievalResult:
        """
        Structured RAG retrieval.

        Mevcut retrieve() ile aynı karar ağacını kullanır ama chunk-level
        evidence'i RetrievedChunk/RetrievalResult olarak döndürür.
        Score, rank, source, retrieval_type bilgileri korunur.

        Dönüş: RetrievalResult (Pydantic model)
        """
        started = time.perf_counter()

        try:
            return self._retrieve_structured_inner(question, use_internet, web_only, started)
        except Exception as exc:
            elapsed = int((time.perf_counter() - started) * 1000)
            return RetrievalResult(
                query=question,
                chunks=[],
                top_k=self.top_k,
                threshold=self.thr,
                used_context=False,
                retrieval_mode="error",
                fallback_used=True,
                fallback_reason="retrieval_exception",
                latency_ms=elapsed,
                error=str(exc),
            )

    def _retrieve_structured_inner(
        self,
        question: str,
        use_internet: bool,
        web_only: bool,
        started: float,
    ) -> RetrievalResult:
        """Core structured retrieval logic (called by retrieve_structured)."""

        # 1) Lokal hibrit arama
        retrieved = hybrid_search(question, top_k=self.top_k)
        top_score = float(max((r.get("score", 0.0) for r in retrieved), default=0.0))

        # 2) Web chunk'ları (gerekirse)
        raw_web: List[Dict[str, Any]] = []
        if use_internet and question.strip():
            try:
                raw_web = process_web_results(question) or []
            except Exception:
                raw_web = []

        # 3) Skor kapısı
        local_ok = bool(retrieved) and (top_score >= self.thr)
        ws = _web_strength(raw_web) if (use_internet and raw_web) else 0.0
        eligible_web = raw_web if (use_internet and ws >= RAG_WEB_MIN_STRENGTH) else []

        # 4) Karar ağacı — chunk mapping ve mode belirleme
        if not use_internet:
            if local_ok:
                chunks = _map_local_chunks(retrieved)
                mode = "local_only"
                fallback_used = False
                fallback_reason = None
            else:
                chunks = []
                mode = "empty"
                fallback_used = True
                fallback_reason = "local_retrieval_below_threshold"
        elif web_only:
            if eligible_web:
                chunks = _map_web_chunks(eligible_web)
                mode = "web_only"
                fallback_used = False
                fallback_reason = None
            elif local_ok:
                chunks = _map_local_chunks(retrieved)
                mode = "local_only"
                fallback_used = True
                fallback_reason = "web_retrieval_weak_fell_back_to_local"
            else:
                chunks = []
                mode = "empty"
                fallback_used = True
                fallback_reason = "empty_retrieval"
        else:
            if local_ok and eligible_web:
                local_chunks = _map_local_chunks(retrieved)
                web_chunks_mapped = _map_web_chunks(eligible_web, rank_offset=len(local_chunks))
                chunks = local_chunks + web_chunks_mapped
                mode = "hybrid_local_web"
                fallback_used = False
                fallback_reason = None
            elif local_ok:
                chunks = _map_local_chunks(retrieved)
                mode = "local_only"
                fallback_used = False
                fallback_reason = None
            elif eligible_web:
                chunks = _map_web_chunks(eligible_web)
                mode = "web_only"
                fallback_used = True
                fallback_reason = "local_retrieval_below_threshold"
            else:
                chunks = []
                mode = "empty"
                fallback_used = True
                fallback_reason = "empty_retrieval"

        elapsed = int((time.perf_counter() - started) * 1000)

        return RetrievalResult(
            query=question,
            chunks=chunks,
            top_k=self.top_k,
            best_score=top_score if retrieved else None,
            threshold=self.thr,
            used_context=bool(chunks),
            retrieval_mode=mode,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            latency_ms=elapsed,
        )
