# app/services/rag_backend/prompt.py
from typing import List, Dict, Optional
from . import RAG_MAX_CTX_TOKENS
import json
import os
from transformers import AutoTokenizer
from utils.text import rag_instruction  # prompt iskeleti için

# T5 tokenizer ve sınır
_T5_TOK_DIR = os.getenv("T5_TOKENIZER_DIR", "assets/models/t5/tokenizer")
_T5_MAX = int(os.getenv("T5_MAX_SRC_LEN", "512"))
_tok = AutoTokenizer.from_pretrained(_T5_TOK_DIR, use_fast=True, local_files_only=True)


def _tok_len(s: str) -> int:
    return len(_tok.encode(s, add_special_tokens=False))

def create_context(
    chunks: List[Dict],
    max_tokens: int = RAG_MAX_CTX_TOKENS,
    question: Optional[str] = None
) -> str:
    """
    Context'i tokenizer bazlı BÜTÜN PROMPT bütçesine göre keser.
    Yani: (instruction + "Context:" + context + "Question:" + question + "Answer:")
    toplamı T5_MAX_SRC_LEN'i aşmaz.
    """
    # 1) Prompt iskeleti (context boşken) → soru ve talimatın token maliyeti
    skeleton = f"{rag_instruction()}\nContext: \nQuestion: {question or ''}\nAnswer:"
    overhead = _tok_len(skeleton)

    # 2) Bağlam için kalan bütçe = min(max_tokens, T5_MAX) - overhead
    budget = min(int(max_tokens), int(_T5_MAX)) - overhead
    if budget <= 0:
        return ""

    parts: List[str] = []
    used = 0
    for c in chunks:
        t = c.get("chunk", "") or ""
        need = _tok_len(t + "\n\n")
        if need <= 0:
            continue
        if used + need > budget:
            # Parça fazla geliyorsa token bazlı sağdan kırp
            ids = _tok.encode(t, add_special_tokens=False)
            keep = max(0, budget - used)
            if keep > 0 and len(ids) > keep:
                t = _tok.decode(ids[:keep], skip_special_tokens=True)
                parts.append(t)
            # bütçe doldu, çık
            break
        parts.append(t)
        used += need

    return "\n\n".join([p.strip() for p in parts if p.strip()])


def create_augmented_prompt(question: str, retrieved_chunks: List[Dict]) -> str:
    """
    Kullanıcının sorusu ve chunk'lar ile Flan-T5 formatında
    zenginleştirilmiş prompt oluşturur.
    """
    context = create_context(retrieved_chunks, max_tokens=RAG_MAX_CTX_TOKENS)
    
    prompt_data = {
        "task": "rag_qa",
        "context": context,
        "text": question,
        "response": "",
        "rag_required": 1
    }
    return json.dumps(prompt_data, ensure_ascii=False, indent=2)
