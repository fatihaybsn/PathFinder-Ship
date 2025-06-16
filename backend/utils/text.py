# app/utils/text.py
from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
from tokenizers import Tokenizer

_TOKENIZER_CACHE: Dict[str, Tokenizer] = {}

def load_hf_tokenizer(dir_path: str) -> Tokenizer:
    """
    HF tokenizers formatındaki tokenizer.json dosyasını dir_path'ten yükler.
    """
    if dir_path in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[dir_path]
    # tokenizer.json bekler; yoksa uygun dosya adını güncelleyin
    tok = Tokenizer.from_file(f"{dir_path}/tokenizer.json")
    _TOKENIZER_CACHE[dir_path] = tok
    return tok

def encode_for_minilm(tokenizer: Tokenizer, text: str, max_len: int) -> Dict[str, np.ndarray]:
    """
    MiniLM/BERT benzeri onnx sınıflandırıcı girişleri: input_ids, attention_mask
    int64 tensörler (1, seq_len)
    """
    text = (text or "").strip()
    if not text:
        text = "[EMPTY]"
    enc = tokenizer.encode(text)
    ids = enc.ids[:max_len]
    mask = [1] * len(ids)

    # pad
    if len(ids) < max_len:
        pad_len = max_len - len(ids)
        ids = ids + [0] * pad_len
        mask = mask + [0] * pad_len

    # (1, seq_len) ve int64
    input_ids = np.asarray([ids], dtype=np.int64)
    attention_mask = np.asarray([mask], dtype=np.int64)
    return {"input_ids": input_ids, "attention_mask": attention_mask}

def softmax_logits(logits: np.ndarray) -> Tuple[int, float]:
    """
    (1, num_labels) bekler. argmax index ve olasılık döner.
    """
    x = logits.astype(np.float32)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    p = e / e.sum(axis=-1, keepdims=True)
    idx = int(np.argmax(p, axis=-1)[0])
    score = float(p[0, idx])
    return idx, score


from typing import List

def chat_instruction(bot_name: str, app_name: str) -> str:
    # Senin verdiğin talimat metni bire bir:
    return (
        f"You are an AI assistant named {bot_name}."
        "Always use this name when referring to yourself."
        f"You serve a system called {app_name}."
        "You retrieve information from local sources and from the web. "
        "You assist users with visual analysis and live chat."
        "Be friendly and helpful. Write in fluent, natural English. "
        "You should give natural and sincere answers instead of short ones."
        "Do not fabricate facts; if unsure, say you don't know and suggest next steps. "
        "If the question is ambiguous, briefly ask for clarification.\n"
        
    )
# SİSTEMİN AMACI NE İSE YANİ İLERİDE GÖREVİ HER NE OLACAKSA BURAYA EKLERSİN (YUKARIYA) -- Kullanıcıya passenger diye seslensin istersen ekleyebilirsin.

def rag_instruction() -> str:
    return (
        "Answer strictly using only the information in the Context. "
        "If the answer is not in the Context, say \"I don't know.\" "
        "You should give natural and sincere answers instead of short ones."
        "Do not use outside knowledge. Answer in English."
    )

# dosyanın uygun bir yerine ekle (chat_instruction / rag_instruction yakınları güzel olur)
def fallback_instruction() -> str:
    """RAG skoru düşük olduğunda (top_score < 0.40) kullanılacak TALİMAT."""
    
    return "Be helpful and friendly. Tell me the answer to the user question, but don't make it up. Answer in English."

def build_model_only_prompt(user_text: str, instruction: str) -> str:
    inst = instruction
    return f"{inst}\nUser: {user_text}\nAssistant:"

def build_chat_prompt(user_text: str, bot_name: str, app_name: str) -> str:
    # İSTEDİĞİN: instruction + kullanıcı sorusu TEK PROMPT’TA
    return f"{chat_instruction(bot_name, app_name)}\nUser: {user_text}\nAssistant:"

def build_rag_prompt(question: str, context: List[str] | str | None) -> str:
    ctx = ""
    if isinstance(context, list):
        ctx = "\n".join(context)
    elif isinstance(context, str):
        ctx = context
    return f"{rag_instruction()}\nContext: {ctx}\nQuestion: {question}\nAnswer:"




# --- CAMERA OPEN ---
def open_camera_instruction(bot_name: str) -> str:
    return (
        f"You are an AI assistant named {bot_name}."
        "The user asked you to open the camera."
        "Acknowledge this action and tell the user you are opening it now, in natural and friendly language."
    )

def build_open_camera_prompt(bot_name: str) -> str:
    return f"{open_camera_instruction(bot_name)}\nAssistant:"

# --- CAMERA CLOSE ---
def close_camera_instruction(bot_name: str) -> str:
    return (
        f"You are an AI assistant named {bot_name}. "
        "The user asked you to close the camera. "
        "Confirm that you are closing it now, in natural and friendly language."
    )

def build_close_camera_prompt(bot_name: str) -> str:
    return f"{close_camera_instruction(bot_name)}\nAssistant:"

# --- TAKE PHOTO ---
def take_photo_instruction(bot_name: str) -> str:
    return (
        f"You are an AI assistant named {bot_name}. "
        "The user asked you to take a photo. "
        "Tell the user you will take the photo now (or have just taken it), in natural and friendly language."
    )

def build_take_photo_prompt(bot_name: str) -> str:
    return f"{take_photo_instruction(bot_name)}\nAssistant:"

def detection_instruction(objects_text: str) -> str:
    return (
        "Speak directly to the user about what the camera likely shows. "
        f"Use these detections only as signals: {objects_text}. "
        "Infer a plausible setting or activity (e.g., cafe, desk, street) without sounding certain. "
        "Write 2–3 short, conversational sentences. "
        "Start immediately; no preambles (e.g., 'Sure', 'Here are'). "
        "Avoid words like 'detected' or 'object' and avoid meta-talk. "
        "Use natural singular/plural forms. "
        "End with one light, context-fitting question."
    )

def build_detection_prompt(objects: list[str] | str, bot_name: str) -> str:
    if isinstance(objects, list):
        obj_text = ", ".join(objects) if objects else "no objects"
    else:
        obj_text = objects or "no objects"
    # Bot adını özellikle kullanmıyoruz ki “You are …” yankısı olmasın
    return f"{detection_instruction(obj_text)}\nAssistant:"




