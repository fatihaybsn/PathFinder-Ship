import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from schemas.pipeline import GenerationResult, RetrievedChunk, RetrievalResult
from services.document_indexing import UploadIndexingError, index_upload_file
from services.pipeline_orchestrator import PipelineOrchestrator


class FakeUploadFile:
    def __init__(self, filename, data):
        self.filename = filename
        self._stream = io.BytesIO(data)

    async def read(self, size=-1):
        return self._stream.read(size)


def run(coro):
    return asyncio.run(coro)


def make_cfg(corpus_dir, max_bytes=1024 * 1024):
    return {
        "RAG_CORPUS_DIR": str(corpus_dir),
        "UPLOAD_MAX_BYTES": max_bytes,
        "RAG_UPLOAD_ALLOWED_EXTENSIONS": ".pdf,.docx,.txt,.md,.html,.htm",
    }


class UploadIndexingTests(unittest.TestCase):
    def test_supported_txt_upload_is_saved_chunked_and_indexed(self):
        with tempfile.TemporaryDirectory() as tmp:
            captured = {}

            def fake_add(chunks, document_id):
                captured["chunks"] = chunks
                captured["document_id"] = document_id
                return []

            upload = FakeUploadFile(
                "../Safety Manual.txt",
                b"alpha bravo unique-pathfinder-upload smoke text",
            )
            with patch("services.document_indexing._add_chunks_to_index", side_effect=fake_add):
                result = run(index_upload_file(upload, make_cfg(tmp)))

            self.assertTrue(result.indexed)
            self.assertGreater(result.indexed_chunk_count, 0)
            self.assertTrue(result.saved_path.endswith(".txt"))
            self.assertNotIn("..", result.saved_path)
            self.assertEqual(result.document_id, captured["document_id"])
            self.assertEqual(captured["chunks"][0]["metadata"]["document_id"], result.document_id)
            self.assertEqual(captured["chunks"][0]["metadata"]["chunk_index"], 0)

    def test_unsupported_extension_returns_safe_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            upload = FakeUploadFile("payload.exe", b"not a document")

            with self.assertRaises(UploadIndexingError) as ctx:
                run(index_upload_file(upload, make_cfg(tmp)))

            self.assertEqual(ctx.exception.code, "unsupported_file_type")
            self.assertEqual(ctx.exception.status_code, 400)
            self.assertEqual(list(Path(tmp).glob("*")), [])

    def test_file_too_large_returns_safe_error_without_saved_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            upload = FakeUploadFile("large.txt", b"0123456789")

            with self.assertRaises(UploadIndexingError) as ctx:
                run(index_upload_file(upload, make_cfg(tmp, max_bytes=4)))

            self.assertEqual(ctx.exception.code, "file_too_large")
            self.assertEqual(ctx.exception.status_code, 413)
            self.assertEqual(list(Path(tmp).glob("*")), [])

    def test_empty_file_is_saved_but_indexing_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            upload = FakeUploadFile("empty.txt", b"")

            result = run(index_upload_file(upload, make_cfg(tmp)))

            self.assertFalse(result.indexed)
            self.assertTrue(result.skipped)
            self.assertEqual(result.skip_reason, "empty_file")
            self.assertEqual(result.indexed_chunk_count, 0)
            self.assertEqual(len(list(Path(tmp).glob("empty__*.txt"))), 1)

    def test_duplicate_upload_uses_stable_document_id_and_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []

            def fake_add(chunks, document_id):
                calls.append((document_id, len(chunks)))
                return []

            cfg = make_cfg(tmp)
            with patch("services.document_indexing._add_chunks_to_index", side_effect=fake_add):
                first = run(index_upload_file(FakeUploadFile("manual.txt", b"same document body"), cfg))
                second = run(index_upload_file(FakeUploadFile("manual.txt", b"same document body"), cfg))

            self.assertEqual(first.document_id, second.document_id)
            self.assertEqual(first.saved_path, second.saved_path)
            self.assertEqual(len(list(Path(tmp).glob("manual__*.txt"))), 1)
            self.assertEqual(len(calls), 2)

    def test_uploaded_text_is_available_to_stubbed_retrieval_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            in_memory_index = []

            def fake_add(chunks, document_id):
                in_memory_index[:] = chunks
                return []

            upload = FakeUploadFile(
                "retrieval.txt",
                b"the emergency muster station is on deck seven",
            )
            with patch("services.document_indexing._add_chunks_to_index", side_effect=fake_add):
                result = run(index_upload_file(upload, make_cfg(tmp)))

            matches = [
                chunk for chunk in in_memory_index
                if "muster station" in chunk["chunk"].lower()
            ]
            self.assertTrue(result.indexed)
            self.assertTrue(matches)
            self.assertEqual(matches[0]["metadata"]["document_id"], result.document_id)

    def test_uploaded_document_can_feed_pipeline_rag_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            in_memory_index = []

            def fake_add(chunks, document_id):
                in_memory_index[:] = chunks
                return []

            upload = FakeUploadFile(
                "uploaded_manual.txt",
                b"the uploaded manual says the emergency muster station is on deck seven",
            )
            with patch("services.document_indexing._add_chunks_to_index", side_effect=fake_add):
                upload_result = run(index_upload_file(upload, make_cfg(tmp)))

            class FakeNLU:
                last_error = None

                def predict(self, text):
                    return "chat", 0.95

            class FakeT5:
                max_new_rag = 64

                def chat_structured(self, text):
                    raise AssertionError("uploaded document question should route to RAG")

                def answer_structured(self, question, context):
                    return GenerationResult(
                        text="The muster station is on deck seven.",
                        prompt_type="rag_answer",
                        output_chars=37,
                        latency_ms=1,
                    )

                def answer_model_only_with_instruction_structured(self, question, instruction=None):
                    raise AssertionError("uploaded document query should use retrieved context")

            class UploadedRAG:
                top_k = 2
                thr = 0.4
                max_ctx_tokens = 512

                def retrieve_structured(self, question, use_internet=False, web_only=False):
                    chunk = in_memory_index[0]
                    source = chunk["metadata"]["source"]
                    return RetrievalResult(
                        query=question,
                        chunks=[
                            RetrievedChunk(
                                text=chunk["chunk"],
                                source=f"local:{source}",
                                score=0.91,
                                rank=1,
                                retrieval_type="local_hybrid",
                                metadata=chunk["metadata"],
                            )
                        ],
                        top_k=self.top_k,
                        best_score=0.91,
                        threshold=self.thr,
                        used_context=True,
                        retrieval_mode="local_only",
                        latency_ms=1,
                    )

            pipeline = PipelineOrchestrator(
                {"CLS_ROUTE_THRESHOLD": 0.6},
                FakeNLU(),
                FakeT5(),
                UploadedRAG(),
            )

            result = pipeline.run("what is in the uploaded document?")

            self.assertTrue(upload_result.indexed)
            self.assertGreater(upload_result.indexed_chunk_count, 0)
            self.assertEqual(result.route.route, "rag")
            self.assertTrue(result.retrieval.used_context)
            self.assertIn("uploaded_manual", result.retrieval.chunks[0].source)
            self.assertEqual(result.retrieval.chunks[0].metadata["document_id"], upload_result.document_id)
            self.assertEqual(result.generation.prompt_type, "rag_answer")
            self.assertEqual(result.final_answer, "The muster station is on deck seven.")


if __name__ == "__main__":
    unittest.main()
