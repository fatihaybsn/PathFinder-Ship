# app/services/t5.py
from __future__ import annotations
import logging
import os
import time
from typing import Optional, List, Dict
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from schemas.pipeline import GenerationResult
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

logger = logging.getLogger(__name__)


class T5DecodeError(RuntimeError):
    """Raised when tokenizer decode fails after ONNX inference completed."""


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


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
        self.model_name = cfg.get("T5_MODEL_NAME") or "local-t5-onnx"
        self.runtime = "onnxruntime"
        self.device = "cpu"

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
        return self.chat_structured(user_text).text

    def chat_structured(self, user_text: str) -> GenerationResult:
        prompt = build_chat_prompt(user_text, self.bot_name, self.app_name)
        return self.generate_structured(
            prompt,
            mode="chat",
            prompt_type="chat",
        )

    def answer(self, question: str, context: Optional[List[str] | str]) -> str:
        """
        QA mode: optionally uses RAG context. English output.
        """
        return self.answer_structured(question, context).text

    def answer_structured(self, question: str, context: Optional[List[str] | str]) -> GenerationResult:
        prompt = build_rag_prompt(question, context)
        return self.generate_structured(
            prompt,
            mode="rag",
            prompt_type="rag_answer",
        )
    
    def answer_model_only_with_instruction(self, question: str, instruction: str | None = None) -> str:
        """
        RAG skoru düşük olduğunda kullanılan fallback cevaplayıcı.
        Chat'ten farklı bir instruction ile tek-şut cevap üretir.
        """
        return self.answer_model_only_with_instruction_structured(question, instruction=instruction).text

    def answer_model_only_with_instruction_structured(
        self,
        question: str,
        instruction: str | None = None,
    ) -> GenerationResult:
        inst = instruction if instruction is not None else fallback_instruction()
        prompt = build_model_only_prompt(question, inst)
        return self.generate_structured(
            prompt,
            mode="chat",
            prompt_type="model_only",
        )  

    def narrate_open_camera(self) -> str:
        return self.narrate_open_camera_structured().text

    def narrate_open_camera_structured(self) -> GenerationResult:
        prompt = build_open_camera_prompt(self.bot_name)
        return self.generate_structured(
            prompt,
            mode="chat",
            prompt_type="camera_narration",
        )

    def narrate_close_camera(self) -> str:
        return self.narrate_close_camera_structured().text

    def narrate_close_camera_structured(self) -> GenerationResult:
        prompt = build_close_camera_prompt(self.bot_name)
        return self.generate_structured(
            prompt,
            mode="chat",
            prompt_type="camera_narration",
        )

    def narrate_take_photo(self) -> str:
        return self.narrate_take_photo_structured().text

    def narrate_take_photo_structured(self) -> GenerationResult:
        prompt = build_take_photo_prompt(self.bot_name)
        return self.generate_structured(
            prompt,
            mode="chat",
            prompt_type="camera_narration",
        )

    def narrate_detection(self, objects: list[str] | str) -> str:
        return self.narrate_detection_structured(objects).text

    def narrate_detection_structured(self, objects: list[str] | str) -> GenerationResult:
        prompt = build_detection_prompt(objects, self.bot_name)
        # Greedy/temkinli üretim: gevezelik ve preamble eğilimini azaltır
        return self.generate_structured(
            prompt,
            mode="rag",
            prompt_type="detection_narration",
        )

    def generate_structured(
        self,
        prompt: str,
        *,
        mode: str = "chat",
        prompt_type: str = "unknown",
        max_new_tokens: int | None = None,
        fallback_text: str | None = None,
    ) -> GenerationResult:
        """
        Generate text and return a structured result without changing the core
        ONNX inference logic used by legacy string-returning methods.
        """
        prompt = prompt or ""
        resolved_max_new = self._max_new_for_mode(mode, max_new_tokens)
        started = time.perf_counter()

        base = {
            "model_name": self.model_name,
            "runtime": self.runtime,
            "device": self.device,
            "prompt_type": prompt_type,
            "input_chars": len(prompt),
            "max_new_tokens": resolved_max_new,
        }

        if not prompt.strip():
            text = fallback_text or self._fallback_text(prompt_type)
            return GenerationResult(
                text=text,
                **base,
                output_chars=len(text),
                latency_ms=_elapsed_ms(started),
                empty_output=True,
                fallback_used=True,
                fallback_reason="empty_prompt",
                error="empty_prompt",
            )

        try:
            output_text, meta = self._generate_text_with_metadata(
                prompt,
                mode=mode,
                max_new_tokens=resolved_max_new,
            )
            fallback_reason = self._invalid_generation_reason(output_text)
            result_text = output_text
            fallback_used = False
            empty_output = False

            if fallback_reason:
                result_text = fallback_text or self._fallback_text(prompt_type)
                fallback_used = True
                empty_output = True

            return GenerationResult(
                text=result_text,
                **base,
                output_chars=len(result_text or ""),
                input_tokens=meta.get("input_tokens"),
                output_tokens=meta.get("output_tokens"),
                input_truncated=meta.get("input_truncated"),
                output_truncated=meta.get("output_truncated"),
                latency_ms=_elapsed_ms(started),
                empty_output=empty_output,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
            )
        except T5DecodeError:
            fallback_reason = "decode_failed"
        except Exception:
            logger.exception("T5 generation failed for prompt_type=%s", prompt_type)
            fallback_reason = "inference_failed"

        text = fallback_text or self._fallback_text(prompt_type)
        return GenerationResult(
            text=text,
            **base,
            output_chars=len(text),
            latency_ms=_elapsed_ms(started),
            empty_output=True,
            fallback_used=True,
            fallback_reason=fallback_reason,
            error=fallback_reason,
        )


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
        return self._generate_text_with_metadata(prompt, mode=mode)[0]

    def _max_new_for_mode(self, mode: str, max_new_tokens: int | None = None) -> int:
        if max_new_tokens is not None:
            try:
                return max(0, int(max_new_tokens))
            except Exception:
                pass
        if mode == "chat":
            return max(0, int(self.max_new_chat))
        return max(0, int(self.max_new_rag))

    def _fallback_text(self, prompt_type: str | None) -> str:
        if prompt_type == "camera_narration":
            return "Okay, I will handle that now."
        if prompt_type == "detection_narration":
            return "I couldn't generate a narration right now."
        if prompt_type in {"rag_answer", "model_only"}:
            return "I don't know."
        return "I couldn't generate a response right now."

    def _count_prompt_tokens(self, prompt: str) -> int | None:
        try:
            enc = self.tok([prompt], padding=False, truncation=False, return_attention_mask=False)
            ids = enc.get("input_ids", [])
            if ids and hasattr(ids[0], "__len__"):
                return len(ids[0])
        except Exception:
            return None
        return None

    def _invalid_generation_reason(self, text: str | None) -> str | None:
        if text is None:
            return "empty_generation"
        stripped = str(text).strip()
        if not stripped:
            return "empty_generation"

        special_tokens = getattr(self.tok, "all_special_tokens", []) or []
        without_special = stripped
        for token in special_tokens:
            if token:
                without_special = without_special.replace(str(token), "")
        if not without_special.strip():
            return "invalid_generation"
        return None

    def _generate_text_with_metadata(
        self,
        prompt: str,
        mode: str,
        max_new_tokens: int | None = None,
    ) -> tuple[str, dict[str, int | bool | None]]:
        full_input_tokens = self._count_prompt_tokens(prompt)
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
            max_new = self._max_new_for_mode(mode, max_new_tokens)
            do_sample, top_p, temperature = True, 0.9, 0.7
        else:
            max_new = self._max_new_for_mode(mode, max_new_tokens)
            do_sample, top_p, temperature = False, None, None

        generated = [self.decoder_start_token_id]
        past = None
        output_truncated = True if max_new == 0 else False

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
                output_truncated = False
                break
            generated.append(int(next_id))

        if max_new > 0 and len(generated) - 1 < max_new:
            output_truncated = False

        try:
            text = self.tok.decode(
                generated[1:],  # drop start token
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            ).strip()
        except Exception as exc:
            raise T5DecodeError("tokenizer decode failed") from exc

        metadata = {
            "input_tokens": full_input_tokens,
            "output_tokens": max(0, len(generated) - 1),
            "input_truncated": (
                bool(full_input_tokens > self.max_src_len)
                if full_input_tokens is not None
                else None
            ),
            "output_truncated": output_truncated,
        }
        return text, metadata
