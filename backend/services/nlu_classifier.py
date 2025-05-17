# app/services/nlu_classifier.py
from __future__ import annotations
import numpy as np
import onnxruntime as ort
from typing import Tuple, List, Dict, Any
from transformers import AutoTokenizer, AutoConfig

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
            # Yerel klasörden yükler (tokenizer + vocab dosyaları + merges vs.)
            self._tok = AutoTokenizer.from_pretrained(self.tok_dir, use_fast=True, local_files_only=True)
        return self._tok

    @property
    def labels(self) -> List[str]:
        if self._labels is None:
            try:
                cfg = AutoConfig.from_pretrained(self.tok_dir, local_files_only=True)
                # id2label sözlüğünü sıraya göre listeye çevir
                id2label = getattr(cfg, "id2label", None) or {}
                labels = [id2label[i] for i in range(len(id2label))] if id2label else []
                self._labels = labels if labels else ["open_camera","close_camera","take_photo","object_detect","chat"]
            except Exception:
                self._labels = ["open_camera","close_camera","take_photo","object_detect","chat"]
        return self._labels

    # --- Core ---
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Metin → (kanonik etiket, olasılık)
        Hata halinde ('chat', 0.0) döner ve last_error set edilir.
        """
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
            return canon, score

        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            return "chat", 0.0
