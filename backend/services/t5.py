# app/services/t5.py
from __future__ import annotations
import os
from typing import Optional, List, Dict
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from utils.text import (
    build_chat_prompt,
    build_rag_prompt,
    build_open_camera_prompt,
    build_close_camera_prompt,
    build_take_photo_prompt,
    build_detection_prompt,
    build_model_only_prompt,  
    fallback_instruction,   
)

# ---------- utils: sampling ----------
def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.clip(e.sum(axis=-1, keepdims=True), 1e-8, None)

def _top_p_sample(logits: np.ndarray, top_p: float, temperature: float) -> int:
    if temperature and temperature > 0.0:
        logits = logits / float(temperature)
    probs = _softmax(logits[None, :])[0]
    idx = np.argsort(probs)[::-1]
    cdf = np.cumsum(probs[idx])
    keep = idx[cdf <= top_p]
    if keep.size == 0:
        keep = idx[:1]
    p = probs[keep]
    p = p / p.sum()
    return int(np.random.choice(keep, p=p))

def _greedy(logits: np.ndarray) -> int:
    return int(np.argmax(logits))

# ---------- utils: dtype helpers ----------
def _np_dtype_for_ort(ort_type: str):
    """
    Map ORT type string (e.g., 'tensor(int64)', 'tensor(float)', 'tensor(float16)')
    to numpy dtype.
    """
    if "int64" in ort_type:
        return np.int64
    if "float16" in ort_type:
        return np.float16
    # default float32
    return np.float32


class T5Service:
    """
    ONNX runtime wrapper for T5 (encoder + decoder). Produces English-only responses.
    """

    def __init__(self, cfg: dict):
        # paths / names
        self.tok_dir  = cfg.get("T5_TOKENIZER_DIR", "assets/models/t5/tokenizer")
        self.enc_path = cfg.get("T5_ENCODER", "assets/models/t5/encoder_model_int8.onnx")
        self.dec_path = cfg.get("T5_DECODER", "assets/models/t5/decoder_with_past_model_int8.onnx")  
        self.bot_name = cfg.get("BOT_NAME", "Passenger-Bot")
        self.app_name = cfg.get("APP_NAME", "PathFinder-Ship")

        # limits
        self.max_src_len  = int(cfg.get("T5_MAX_SRC_LEN", 512))
        self.max_new_chat = int(cfg.get("T5_MAX_NEW_TOKENS_CHAT", 256))  # Buralardaki değişkenlerin değeri yedek değerdir yani hiç bir şey yazmıyorsa .env dosyasında bu değer kullanılır.
        self.max_new_rag  = int(cfg.get("T5_MAX_NEW_TOKENS_RAG", 64))

        # sessions
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.encoder = ort.InferenceSession(self.enc_path, so, providers=["CPUExecutionProvider"])
        self.decoder = ort.InferenceSession(self.dec_path, so, providers=["CPUExecutionProvider"])

        # io schemas
        self.enc_inputs   = [i.name for i in self.encoder.get_inputs()]
        self.dec_inputs   = [i.name for i in self.decoder.get_inputs()]
        self.dec_outputs  = [o.name for o in self.decoder.get_outputs()]
        # name -> ORT type string (e.g., 'tensor(int64)')
        self.dec_input_types = {i.name: i.type for i in self.decoder.get_inputs()}

        # tokenizer / ids
        self.tok = AutoTokenizer.from_pretrained(self.tok_dir, use_fast=True, local_files_only=True)
        self.decoder_start_token_id = int(getattr(self.tok, "pad_token_id", 0) or 0)
        self.eos_token_id = int(getattr(self.tok, "eos_token_id", 1) or 1)

        # capabilities
        self._has_past = any(("past_key_values" in n) or ("pkv" in n) for n in self.dec_inputs)

    # ============ PUBLIC API (English-only prompts) ============
    def chat(self, user_text: str) -> str:
        """
        Small-talk or basic chat (no RAG). English output.
        """
        prompt = build_chat_prompt(user_text, self.bot_name, self.app_name)
        return self._generate_text(prompt, mode="chat")

    def answer(self, question: str, context: Optional[List[str] | str]) -> str:
        """
        QA mode: optionally uses RAG context. English output.
        """
        prompt = build_rag_prompt(question, context)
        return self._generate_text(prompt, mode="rag")
    
    def answer_model_only_with_instruction(self, question: str, instruction: str | None = None) -> str:
        """
        RAG skoru düşük olduğunda kullanılan fallback cevaplayıcı.
        Chat'ten farklı bir instruction ile tek-şut cevap üretir.
        """
        inst = instruction if instruction is not None else fallback_instruction()
        prompt = build_model_only_prompt(question, inst)
        return self._generate_text(prompt, mode="chat")  

    def narrate_open_camera(self) -> str:
        prompt = build_open_camera_prompt(self.bot_name)
        return self._generate_text(prompt, mode="chat")

    def narrate_close_camera(self) -> str:
        prompt = build_close_camera_prompt(self.bot_name)
        return self._generate_text(prompt, mode="chat")

    def narrate_take_photo(self) -> str:
        prompt = build_take_photo_prompt(self.bot_name)
        return self._generate_text(prompt, mode="chat")

    def narrate_detection(self, objects: list[str] | str) -> str:
        prompt = build_detection_prompt(objects, self.bot_name)
        # Greedy/temkinli üretim: gevezelik ve preamble eğilimini azaltır
        out = self._generate_text(prompt, mode="rag")
        return (out or "").strip()


    # ============ CORE ============
    def _encode(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run encoder. Return dict with 'encoder_hidden_states' and 'encoder_attention_mask'.
        We keep mask in the original int64 here; decoder side will cast as needed.
        """
        feed = {}
        # place by name order (fallback to positional if needed)
        feed[self.enc_inputs[0]] = input_ids
        if len(self.enc_inputs) > 1:
            feed[self.enc_inputs[1]] = attention_mask
        enc_out = self.encoder.run(None, feed)
        return {
            "encoder_hidden_states": enc_out[0],  # shape can be (T,H) or (1,T,H) depending on export
            "encoder_attention_mask": attention_mask,  # (1,T) int64
        }

    def _generate_text(self, prompt: str, mode: str) -> str:
        # tokenize
        enc = self.tok([prompt], padding=False, truncation=True,
                       max_length=self.max_src_len, return_tensors="np")
        input_ids     = enc["input_ids"].astype(np.int64)
        attention_mask = enc["attention_mask"].astype(np.int64)

        # encode
        ctx = self._encode(input_ids, attention_mask)

        # --- normalize encoder outputs (shapes + dtypes) ---
        enc_out = ctx["encoder_hidden_states"]
        if enc_out.ndim == 2:             # (T,H) -> (1,T,H)
            enc_out = enc_out[None, ...]
        enc_out_fp32 = enc_out.astype(np.float32)   # safe default for decoder

        enc_mask = ctx["encoder_attention_mask"]    # (1,T) int64 expected from tokenizer
        if enc_mask.ndim == 1:                      # (T,) -> (1,T) safeguard
            enc_mask = enc_mask[None, ...]
        enc_mask_fp32 = enc_mask.astype(np.float32)
        enc_mask_i64  = enc_mask.astype(np.int64)
        # ---------------------------------------------------

        # decoding config
        if mode == "chat":
            max_new = self.max_new_chat
            do_sample, top_p, temperature = True, 0.9, 0.7
        else:
            max_new = self.max_new_rag
            do_sample, top_p, temperature = False, None, None

        generated = [self.decoder_start_token_id]
        past = None

        for _ in range(max_new):
            if not self._has_past:
                # full-sequence decoding (no past)
                dec_inp = np.asarray([generated], dtype=np.int64)

                feed: Dict[str, np.ndarray] = {}
                for n in self.dec_inputs:
                    onnx_type = self.dec_input_types.get(n, "tensor(float)")
                    want = _np_dtype_for_ort(onnx_type)

                    if "input_ids" in n:
                        feed[n] = dec_inp.astype(np.int64)
                    elif "encoder_hidden_states" in n:
                        feed[n] = enc_out_fp32.astype(want)
                    elif "encoder_attention_mask" in n:
                        feed[n] = enc_mask_i64 if want == np.int64 else enc_mask_fp32

                outs = self.decoder.run(None, feed)
                logits = outs[0][:, -1, :][0]

            else:
                # step-by-step decoding with past
                last_id = np.asarray([[generated[-1]]], dtype=np.int64)

                feed: Dict[str, np.ndarray] = {}
                for n in self.dec_inputs:
                    onnx_type = self.dec_input_types.get(n, "tensor(float)")
                    want = _np_dtype_for_ort(onnx_type)

                    if "input_ids" in n:
                        feed[n] = last_id.astype(np.int64)
                    elif "encoder_hidden_states" in n:
                        feed[n] = enc_out_fp32.astype(want)
                    elif "encoder_attention_mask" in n:
                        feed[n] = enc_mask_i64 if want == np.int64 else enc_mask_fp32
                    elif "use_cache_branch" in n:
                        feed[n] = np.asarray([1], dtype=np.bool_)
                    elif ("past_key_values" in n or "pkv" in n) and past and n in past:
                        feed[n] = past[n]

                outs = self.decoder.run(None, feed)
                logits = outs[0][0]

                # collect new past
                new_past: Dict[str, np.ndarray] = {}
                for name, val in zip(self.dec_outputs[1:], outs[1:]):
                    new_past[name] = val
                past = new_past

            # choose next token
            next_id = _top_p_sample(logits, top_p, temperature) if do_sample else _greedy(logits)
            if next_id == self.eos_token_id:
                break
            generated.append(int(next_id))

        return self.tok.decode(
            generated[1:],  # drop start token
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()
