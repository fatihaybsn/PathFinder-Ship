# app/services/nlu_classifier.py
from __future__ import annotations
import time
import numpy as np
import onnxruntime as ort
from typing import Tuple, List, Dict, Any

try:
    from transformers import AutoTokenizer, AutoConfig
except Exception:
    try:
        from transformers import AutoTokenizer
    except Exception:
        AutoTokenizer = None
    AutoConfig = None

from schemas.pipeline import IntentResult, intent_result_from_prediction

_CANONICAL = {"open_camera", "close_camera", "take_photo", "object_detect", "chat"}

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

def _normalize_label(label: str) -> str:
    # "open camera" → "open_camera", vb.
    return (
        (label or "").strip().lower()
        .replace("-", "_").replace(" ", "_")
    )


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


class NLUClassifier:
    """
    MiniLM-L6 intent sınıflandırıcı (ONNX) + HF tokenizer/config.
    CLS_ONNX, CLS_TOKENIZER_DIR, CLS_MAX_LEN .env'den okunur.
    """
    def __init__(self, cfg: dict):
        self.model_path   = cfg.get("CLS_ONNX", "assets/models/nlu/intent-minilm-int8.onnx")
        self.tok_dir      = cfg.get("CLS_TOKENIZER_DIR", "assets/models/nlu/tokenizer")
        self.max_len      = int(cfg.get("CLS_MAX_LEN", 64))

        self._sess: ort.InferenceSession | None = None
        self._tok  = None
        self._labels: List[str] | None = None
        self._required_inputs: List[str] | None = None
        self.last_error: str | None = None

    # --- Lazy loaders ---
    @property
    def session(self) -> ort.InferenceSession:
        if self._sess is None:
            self._sess = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
            self._required_inputs = [i.name for i in self._sess.get_inputs()]
        return self._sess

    @property
    def tokenizer(self):
        if self._tok is None:
            if AutoTokenizer is None:
                raise RuntimeError("transformers AutoTokenizer is not available")
            # Yerel klasörden yükler (tokenizer + vocab dosyaları + merges vs.)
            self._tok = AutoTokenizer.from_pretrained(self.tok_dir, use_fast=True, local_files_only=True)
        return self._tok

    @property
    def labels(self) -> List[str]:
        if self._labels is None:
            try:
                if AutoConfig is None:
                    raise RuntimeError("transformers AutoConfig is not available")
                cfg = AutoConfig.from_pretrained(self.tok_dir, local_files_only=True)
                # id2label sözlüğünü sıraya göre listeye çevir
                id2label = getattr(cfg, "id2label", None) or {}
                labels = [id2label[i] for i in range(len(id2label))] if id2label else []
                self._labels = labels if labels else ["open_camera","close_camera","take_photo","object_detect","chat"]
            except Exception:
                self._labels = ["open_camera","close_camera","take_photo","object_detect","chat"]
        return self._labels

    # --- Core ---
    def classify_intent(self, text: str, threshold: float | None = None) -> IntentResult:
        """
        Metin -> IntentResult.

        Bu katman sadece sınıflandırma yapar; T5/RAG/YOLO/kamera/e-posta gibi
        yan etkili servisleri çağırmaz.
        """
        started = time.perf_counter()
        self.last_error = None
        try:
            # ORT ve giriş isimleri hazırla
            sess = self.session
            required = self._required_inputs or []

            # Tokenize (np tensörler ile)
            enc = self.tokenizer(
                [text],
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="np",
            )

            # Model 'token_type_ids' istiyorsa ve tokenizer üretmediyse sıfırlarla ekle
            if "token_type_ids" in required and "token_type_ids" not in enc:
                enc["token_type_ids"] = np.zeros_like(enc["input_ids"])

            # Sadece gereksinilen girişleri, doğru dtype (int64) ile besle
            feed: Dict[str, Any] = {}
            for name in required:
                arr = enc[name]
                if arr.dtype != np.int64:
                    arr = arr.astype(np.int64)
                feed[name] = arr

            # Çalıştır
            outputs = sess.run(None, feed)
            if not outputs:
                raise RuntimeError("ONNX çıktı listesi boş.")
            logits = outputs[0]  # (1, num_labels)
            probs = _softmax(logits)
            idx = int(np.argmax(probs, axis=-1)[0])
            score = float(probs[0, idx])

            raw_label = self.labels[idx] if 0 <= idx < len(self.labels) else "chat"
            canon = _normalize_label(raw_label)
            # Beklenen 5 etiketten biri değilse güvenli varsayılanı 'chat' yap
            if canon not in _CANONICAL:
                canon = "chat"

            raw_scores: Dict[str, float] = {}
            for label_index, probability in enumerate(probs[0]):
                label = self.labels[label_index] if 0 <= label_index < len(self.labels) else "chat"
                normalized = _normalize_label(label)
                if normalized not in _CANONICAL:
                    normalized = "chat"
                raw_scores[normalized] = max(raw_scores.get(normalized, 0.0), float(probability))

            return intent_result_from_prediction(
                label=canon,
                confidence=score,
                threshold=threshold,
                raw_scores=raw_scores,
                latency_ms=_elapsed_ms(started),
            )

        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            return intent_result_from_prediction(
                label="chat",
                confidence=0.0,
                threshold=threshold,
                raw_scores=None,
                latency_ms=_elapsed_ms(started),
                error=self.last_error,
            )

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Geriye uyumlu eski arayüz: Metin -> (kanonik etiket, olasılık).
        Yeni kod classify_intent(...) kullanmalıdır.
        """
        result = self.classify_intent(text)
        return result.label, float(result.confidence or 0.0)
