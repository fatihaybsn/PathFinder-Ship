# app/services/rag_backend/websearch.py
from __future__ import annotations
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import time
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

# Defterdeki yardımcılar (temizlik + chunklama)
try:
    from .preprocess import CHUNK_SIZE, CHUNK_OVERLAP, chunk_text, clean_text
except Exception:
    CHUNK_SIZE, CHUNK_OVERLAP = 90, 20

    # Basit fallback'ler
    def clean_text(t: str) -> str:
        return " ".join((t or "").split())

    def chunk_text(t: str, cs: int, ov: int) -> List[str]:
        w = (t or "").split()
        out = []
        i = 0
        while i < len(w):
            out.append(" ".join(w[i:i+cs]))
            if i+cs >= len(w):
                break
            i = max(0, i+cs-ov)
        return out

# Basit ve nazik bir User-Agent
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

def _is_probably_html(resp: requests.Response) -> bool:
    ctype = resp.headers.get("Content-Type", "").lower()
    # html veya düz metin dışındakileri (pdf, octet-stream vs.) at
    return ("text/html" in ctype) or ("text/plain" in ctype)

def search_web(query: str, max_results: int = 4) -> List[Dict]:
    """
    DDG üzerinden web sonuçlarını döndürür.
    Dönüş elemanları tipik olarak {title, href, body, ...}
    """
    out: List[Dict] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            # ddgs.text dict döndürür; ihtiyacımız olan alanlar:
            href = r.get("href") or r.get("url")
            title = r.get("title", "")
            if not href:
                continue
            out.append({"title": title or "", "href": href})
    return out

def extract_web_content(url: str, timeout: int = 10) -> str:
    """
    URL'den HTML içeriği indir, <script>/<style> temizle, metni normalize et.
    Uygun değilse boş string döndür.
    """
    try:
        resp = requests.get(url, headers={"User-Agent": _UA}, timeout=timeout)
        if not _is_probably_html(resp):
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        for bad in soup(["script", "style", "noscript"]):
            bad.extract()
        text = soup.get_text(separator=" ")
        return clean_text(text)
    except Exception:
        return ""

def _norm_relevance(query: str, chunk: str) -> float:
    """
    Çok basit bir alaka skoru: sorgu kelimelerinin kaç tanesi chunk içinde var?
    0..1 aralığına normalize edilir.
    """
    q_tokens = [w for w in query.lower().split() if w]
    if not q_tokens:
        return 0.0
    hits = sum(1 for w in q_tokens if w in chunk.lower())
    return min(1.0, hits / len(q_tokens))

def process_web_results(query: str, max_results: int = 4) -> List[Dict]:
    """
    Arama → paralel içerik çekme → her sayfadan en fazla 3 kısa chunk üretme.
    Dönüş: [{"chunk": "...", "source": "https://...", "title": "...", "score": float(0..1)}, ...]
    """
    results = search_web(query, max_results=max_results)
    if not results:
        return []

    processed: List[Dict] = []
    max_workers = max(1, min(8, len(results)))  # aşırı thread açma

    # Aynı domain'den çok sonuç gelirse ilkini tercih edelim (noise azaltır)
    seen_domains = set()
    filtered = []
    for r in results:
        try:
            dom = urlparse(r["href"]).netloc
        except Exception:
            dom = ""
        if dom and dom not in seen_domains:
            seen_domains.add(dom)
            filtered.append(r)
    if not filtered:
        filtered = results

    # Paralel içerik çekme
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(extract_web_content, r["href"]): r for r in filtered}
        for fut in as_completed(futures):
            meta = futures[fut]
            try:
                content = fut.result()
            except Exception:
                content = ""
            if not content:
                continue

            title = meta.get("title", "")
            url = meta.get("href", "")

            # İçeriği kısa parçalara böl; ilk 2–3 parça genellikle yeterli
            chunks = chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
            for i, c in enumerate(chunks[:3]):
                formatted = f"[Web {i+1}: {title}] {c}".strip()
                score = _norm_relevance(query, c)
                processed.append({
                    "chunk": formatted,
                    "source": url,
                    "title": title,
                    "score": float(score),
                })

    # Basit sıralama: puana göre azalan, sonra ilk 8–10 parça
    processed.sort(key=lambda x: x["score"], reverse=True)
    return processed[:10]
