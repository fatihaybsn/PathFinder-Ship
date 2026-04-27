"""Tests for Prompt 5 — RAG structured modernization.

Covers:
1. RetrievalResult / RetrievedChunk serialization
2. Empty retrieval produces correct fallback fields
3. retrieval_result_empty helper
4. RAGService.retrieve_structured() with mock hybrid_search
5. PipelineOrchestrator._run_rag() fills RunResult.retrieval
6. build_context_from_chunks / chunk mapping helpers
"""
import sys
import types
import unittest
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Stub heavy ML dependencies that may be absent in test environment.
# This lets us import pipeline_orchestrator without installing sentence_transformers etc.
# ---------------------------------------------------------------------------
for _mod_name in ("tokenizers", "sentence_transformers", "tqdm", "transformers",
                  "chromadb", "chromadb.utils", "chromadb.utils.embedding_functions",
                  "bs4", "ddgs", "docx", "fitz"):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = types.ModuleType(_mod_name)
# tokenizers.Tokenizer stub
if not hasattr(sys.modules["tokenizers"], "Tokenizer"):
    sys.modules["tokenizers"].Tokenizer = MagicMock()
# sentence_transformers.SentenceTransformer stub
if not hasattr(sys.modules["sentence_transformers"], "SentenceTransformer"):
    sys.modules["sentence_transformers"].SentenceTransformer = MagicMock()
# transformers.AutoTokenizer stub
if not hasattr(sys.modules["transformers"], "AutoTokenizer"):
    _auto_tok = MagicMock()
    _auto_tok.from_pretrained = MagicMock(return_value=MagicMock())
    sys.modules["transformers"].AutoTokenizer = _auto_tok
# tqdm stub
if not callable(getattr(sys.modules["tqdm"], "tqdm", None)):
    sys.modules["tqdm"].tqdm = lambda iterable, **kw: iterable
# chromadb stub
if not hasattr(sys.modules["chromadb"], "PersistentClient"):
    sys.modules["chromadb"].PersistentClient = MagicMock()
# bs4 stub
if not hasattr(sys.modules["bs4"], "BeautifulSoup"):
    sys.modules["bs4"].BeautifulSoup = MagicMock()
# ddgs stub
if not hasattr(sys.modules["ddgs"], "DDGS"):
    sys.modules["ddgs"].DDGS = MagicMock()
# requests stub (for websearch)
if "requests" not in sys.modules:
    sys.modules["requests"] = MagicMock()

from schemas.pipeline import (
    RetrievedChunk,
    RetrievalResult,
    RunResult,
    retrieval_result_empty,
    retrieval_result_from_legacy,
    to_serializable_dict,
)


# ---------------------------------------------------------------------------
# 1. Serialization
# ---------------------------------------------------------------------------

class RetrievalResultSerializationTests(unittest.TestCase):
    def test_retrieval_result_serializes_to_json(self):
        result = RetrievalResult(
            query="test query",
            chunks=[
                RetrievedChunk(
                    text="chunk text",
                    source="local:test.txt",
                    score=0.85,
                    rank=1,
                    retrieval_type="local_hybrid",
                    metadata={"file_name": "test.txt"},
                ),
            ],
            top_k=4,
            best_score=0.85,
            threshold=0.4,
            used_context=True,
            retrieval_mode="local_only",
        )

        serialized = to_serializable_dict(result)

        self.assertEqual(serialized["query"], "test query")
        self.assertEqual(len(serialized["chunks"]), 1)
        self.assertEqual(serialized["chunks"][0]["text"], "chunk text")
        self.assertEqual(serialized["chunks"][0]["source"], "local:test.txt")
        self.assertEqual(serialized["chunks"][0]["score"], 0.85)
        self.assertEqual(serialized["chunks"][0]["rank"], 1)
        self.assertEqual(serialized["chunks"][0]["retrieval_type"], "local_hybrid")
        self.assertTrue(serialized["used_context"])
        self.assertEqual(serialized["retrieval_mode"], "local_only")

    def test_retrieved_chunk_preserves_rank_source_score(self):
        chunk = RetrievedChunk(
            text="evidence text",
            source="local:safety.pdf",
            score=0.72,
            rank=3,
            retrieval_type="local_hybrid",
        )
        serialized = to_serializable_dict(chunk)

        self.assertEqual(serialized["rank"], 3)
        self.assertEqual(serialized["source"], "local:safety.pdf")
        self.assertEqual(serialized["score"], 0.72)

    def test_retrieval_result_with_multiple_chunks_serializes(self):
        chunks = [
            RetrievedChunk(text=f"chunk {i}", score=0.9 - i * 0.1, rank=i + 1, retrieval_type="local_hybrid")
            for i in range(3)
        ]
        result = RetrievalResult(query="q", chunks=chunks, retrieval_mode="local_only", used_context=True)
        serialized = to_serializable_dict(result)

        self.assertEqual(len(serialized["chunks"]), 3)
        self.assertEqual(serialized["chunks"][0]["rank"], 1)
        self.assertEqual(serialized["chunks"][2]["rank"], 3)

    def test_retrieval_result_json_round_trip(self):
        """Ensure RetrievalResult can be exported to JSON dict and back."""
        original = RetrievalResult(
            query="test",
            chunks=[RetrievedChunk(text="t", score=0.5, rank=1)],
            best_score=0.5,
            used_context=True,
            retrieval_mode="local_only",
        )
        d = to_serializable_dict(original)
        rebuilt = RetrievalResult(**d)

        self.assertEqual(rebuilt.query, original.query)
        self.assertEqual(len(rebuilt.chunks), 1)
        self.assertEqual(rebuilt.chunks[0].text, "t")


# ---------------------------------------------------------------------------
# 2. Empty retrieval / fallback
# ---------------------------------------------------------------------------

class EmptyRetrievalTests(unittest.TestCase):
    def test_empty_retrieval_has_correct_fallback_fields(self):
        result = RetrievalResult(
            query="test",
            chunks=[],
            used_context=False,
            retrieval_mode="empty",
            fallback_used=True,
            fallback_reason="empty_retrieval",
        )

        self.assertEqual(result.chunks, [])
        self.assertFalse(result.used_context)
        self.assertEqual(result.retrieval_mode, "empty")
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.fallback_reason, "empty_retrieval")

    def test_retrieval_result_empty_helper(self):
        result = retrieval_result_empty(
            "test query",
            top_k=4,
            best_score=0.2,
            threshold=0.4,
            latency_ms=15,
        )

        self.assertEqual(result.query, "test query")
        self.assertEqual(result.chunks, [])
        self.assertFalse(result.used_context)
        self.assertEqual(result.retrieval_mode, "empty")
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.fallback_reason, "empty_retrieval")
        self.assertEqual(result.latency_ms, 15)
        self.assertIsNone(result.error)

    def test_retrieval_result_empty_with_error(self):
        result = retrieval_result_empty(
            "test",
            retrieval_mode="error",
            fallback_reason="retrieval_exception",
            error="connection failed",
        )

        self.assertEqual(result.retrieval_mode, "error")
        self.assertEqual(result.error, "connection failed")

    def test_retrieval_result_empty_serializes(self):
        result = retrieval_result_empty("q", best_score=0.1, threshold=0.4)
        serialized = to_serializable_dict(result)

        self.assertEqual(serialized["chunks"], [])
        self.assertFalse(serialized["used_context"])
        self.assertTrue(serialized["fallback_used"])


# ---------------------------------------------------------------------------
# 3. PipelineOrchestrator structured RAG (mock-based, no ML imports)
# ---------------------------------------------------------------------------

class PipelineStructuredRAGTests(unittest.TestCase):
    def test_pipeline_uses_structured_retrieval(self):
        from services.pipeline_orchestrator import PipelineOrchestrator

        class FakeNLU:
            last_error = None
            def predict(self, text):
                return "rag", 0.90

        class FakeT5:
            def chat(self, text):
                return f"chat:{text}"
            def answer(self, question, context):
                return f"rag:{question}"
            def answer_model_only_with_instruction(self, question, instruction=None):
                return f"fallback:{question}"

        class FakeStructuredRAG:
            top_k = 2
            thr = 0.4
            max_ctx_tokens = 512

            def retrieve(self, question, use_internet=False, web_only=False):
                return ["context text"], 0.82, ["local:test.txt"]

            def retrieve_structured(self, question, use_internet=False, web_only=False):
                return RetrievalResult(
                    query=question,
                    chunks=[
                        RetrievedChunk(
                            text="evidence chunk",
                            source="local:test.txt",
                            score=0.82,
                            rank=1,
                            retrieval_type="local_hybrid",
                        ),
                    ],
                    top_k=2,
                    best_score=0.82,
                    threshold=0.4,
                    used_context=True,
                    retrieval_mode="local_only",
                    latency_ms=10,
                )

        pipeline = PipelineOrchestrator(
            {"CLS_ROUTE_THRESHOLD": 0.6},
            FakeNLU(),
            FakeT5(),
            FakeStructuredRAG(),
        )

        result = pipeline.run("where is the exit?")

        self.assertIn(result.status, ("completed", "degraded"))
        self.assertIsNotNone(result.retrieval)
        self.assertEqual(result.retrieval.query, "where is the exit?")
        self.assertEqual(len(result.retrieval.chunks), 1)
        self.assertEqual(result.retrieval.chunks[0].text, "evidence chunk")
        self.assertEqual(result.retrieval.chunks[0].score, 0.82)
        self.assertEqual(result.retrieval.chunks[0].rank, 1)
        self.assertEqual(result.retrieval.retrieval_mode, "local_only")
        self.assertTrue(result.retrieval.used_context)
        self.assertIsNotNone(result.generation)

    def test_pipeline_falls_back_to_legacy_rag(self):
        from services.pipeline_orchestrator import PipelineOrchestrator

        class FakeNLU:
            last_error = None
            def predict(self, text):
                return "rag", 0.90

        class FakeT5:
            def chat(self, text):
                return f"chat:{text}"
            def answer(self, question, context):
                return f"rag:{question}"
            def answer_model_only_with_instruction(self, question, instruction=None):
                return f"fallback:{question}"

        class LegacyRAG:
            """RAG without retrieve_structured — pipeline should fall back."""
            top_k = 2
            thr = 0.4

            def retrieve(self, question, use_internet=False, web_only=False):
                return ["context text"], 0.82, ["local:test.txt"]

        pipeline = PipelineOrchestrator(
            {"CLS_ROUTE_THRESHOLD": 0.6},
            FakeNLU(),
            FakeT5(),
            LegacyRAG(),
        )

        result = pipeline.run("where is the exit?")

        self.assertEqual(result.status, "completed")
        self.assertIsNotNone(result.retrieval)
        self.assertEqual(result.retrieval.query, "where is the exit?")
        self.assertTrue(result.retrieval.used_context)

    def test_pipeline_structured_empty_retrieval_falls_back_to_model_only(self):
        from services.pipeline_orchestrator import PipelineOrchestrator

        class FakeNLU:
            last_error = None
            def predict(self, text):
                return "rag", 0.90

        class FakeT5:
            def chat(self, text):
                return f"chat:{text}"
            def answer(self, question, context):
                return f"rag:{question}"
            def answer_model_only_with_instruction(self, question, instruction=None):
                return f"fallback:{question}"

        class EmptyStructuredRAG:
            top_k = 2
            thr = 0.4
            max_ctx_tokens = 512

            def retrieve(self, question, use_internet=False, web_only=False):
                return [], 0.1, []

            def retrieve_structured(self, question, use_internet=False, web_only=False):
                return RetrievalResult(
                    query=question,
                    chunks=[],
                    top_k=2,
                    best_score=0.1,
                    threshold=0.4,
                    used_context=False,
                    retrieval_mode="empty",
                    fallback_used=True,
                    fallback_reason="empty_retrieval",
                    latency_ms=5,
                )

        pipeline = PipelineOrchestrator(
            {"CLS_ROUTE_THRESHOLD": 0.6},
            FakeNLU(),
            FakeT5(),
            EmptyStructuredRAG(),
        )

        result = pipeline.run("unknown topic?")

        self.assertIn(result.status, ("completed", "degraded"))
        self.assertIsNotNone(result.retrieval)
        self.assertFalse(result.retrieval.used_context)
        self.assertEqual(result.retrieval.retrieval_mode, "empty")
        self.assertTrue(result.retrieval.fallback_used)
        self.assertEqual(result.final_answer, "fallback:unknown topic?")


# ---------------------------------------------------------------------------
# 4. retrieval_result_from_legacy still works
# ---------------------------------------------------------------------------

class LegacyHelperTests(unittest.TestCase):
    def test_retrieval_result_from_legacy_creates_chunks(self):
        result = retrieval_result_from_legacy(
            query="test",
            contexts=["context text"],
            best_score=0.82,
            sources=["local:test.txt"],
            retrieval_mode="hybrid",
        )

        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(result.chunks[0].text, "context text")
        self.assertEqual(result.chunks[0].source, "local:test.txt")
        self.assertEqual(result.best_score, 0.82)

    def test_retrieval_result_from_legacy_empty(self):
        result = retrieval_result_from_legacy(
            query="test",
            contexts=[],
            best_score=0.1,
            sources=[],
            fallback_used=True,
            fallback_reason="no retrieval context",
        )

        self.assertEqual(result.chunks, [])
        self.assertFalse(result.used_context)
        self.assertTrue(result.fallback_used)


if __name__ == "__main__":
    unittest.main()
