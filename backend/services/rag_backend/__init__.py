import os

EMBED_MODEL      = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_PATH      = os.getenv("CHROMA_PATH", "assets/rag/chroma_db")
TOP_K            = int(os.getenv("RAG_TOP_K", 4))
RAG_MAX_CTX_TOKENS = int(os.getenv("RAG_MAX_CTX_TOKENS", 512))
VECTOR_WEIGHT    = float(os.getenv("VECTOR_WEIGHT", 0.75))
BM25_WEIGHT      = float(os.getenv("BM25_WEIGHT", 0.25))

# Web parça yeterlilik eşiği (0..1). Öneri: 0.75
RAG_WEB_MIN_STRENGTH = float(os.getenv("RAG_WEB_MIN_STRENGTH", "0.75"))

# Bir web parçasını "destek" saymak için asgari skor
WEB_CHUNK_SUPPORT_THRESHOLD = float(os.getenv("WEB_CHUNK_SUPPORT_THRESHOLD", "0.70"))