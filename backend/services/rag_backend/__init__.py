from config import CFG

EMBED_MODEL = str(CFG.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
CHROMA_PATH = str(CFG.get("CHROMA_PATH", "assets/rag/chroma_db"))
TOP_K = int(CFG.get("RAG_TOP_K", 4))
RAG_MAX_CTX_TOKENS = int(CFG.get("RAG_MAX_CTX_TOKENS", 512))
VECTOR_WEIGHT = float(CFG.get("VECTOR_WEIGHT", 0.75))
BM25_WEIGHT = float(CFG.get("BM25_WEIGHT", 0.25))

# Web parça yeterlilik eşiği (0..1). Öneri: 0.75
RAG_WEB_MIN_STRENGTH = float(CFG.get("RAG_WEB_MIN_STRENGTH", 0.75))

# Bir web parçasını "destek" saymak için asgari skor
WEB_CHUNK_SUPPORT_THRESHOLD = float(CFG.get("WEB_CHUNK_SUPPORT_THRESHOLD", 0.70))
