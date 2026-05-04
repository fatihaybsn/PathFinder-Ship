# app/services/yolo.py
from __future__ import annotations
import logging
import os, time
from typing import Any, List, Tuple
import numpy as np
import cv2
import onnxruntime as ort
from schemas.pipeline import (
    DETECTION_STATUS_INVALID_IMAGE,
    DETECTION_STATUS_MODEL_ERROR,
    DetectionResult,
    detection_result_from_legacy,
)
from utils.vision import nms, draw_dets  # YOLO-NAS returns xyxy boxes

logger = logging.getLogger(__name__)


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)

class YOLOService:
    """
    YOLO-NAS (ONNX) inference helper.

    Expects 2 outputs (common export):
      - boxes:  (1, N, 4)  -> xyxy
      - scores: (1, N, C)  OR (1, C, N) depending on export

    This implementation:
      - Forces consistent preprocessing (RGB + [0,1], NCHW)
      - Robustly resolves score layout (N,C) vs (C,N)
      - Scales boxes back from letterbox space to original image space
      - Runs NMS and returns dets (x1,y1,x2,y2,conf,cls_id)
    """

    def __init__(self, cfg: dict):
        self.model_path  = cfg.get("YOLO_ONNX",   "assets/models/yolo_nas/yolo_nas_s_coco.onnx")
        self.labels_path = cfg.get("YOLO_LABELS", "assets/models/yolo_nas/labels.txt")
        self.model_name = cfg.get("YOLO_MODEL_NAME") or os.path.basename(os.fspath(self.model_path))
        self.imgsz   = int(cfg.get("YOLO_SIZE", 640))
        self.conf_thr = float(cfg.get("YOLO_CONF", 0.25))
        self.iou_thr  = float(cfg.get("YOLO_IOU", 0.45))

        providers = ["CPUExecutionProvider"]
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(self.model_path, sess_opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        self.names = self._load_names(self.labels_path)

    # ----------------------- utils -----------------------
    def _load_names(self, path: str) -> List[str]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                names = [ln.strip() for ln in f if ln.strip()]
            if names:
                return names
        except Exception:
            pass
        # fallback (unlikely to be correct for your model, but prevents crashes)
        return [f"cls_{i}" for i in range(1000)]

    def _image_metadata(self, image_bgr: Any) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "model_input_size": self.imgsz,
            "confidence_threshold": self.conf_thr,
            "iou_threshold": self.iou_thr,
            "bbox_format": "xyxy",
        }
        if isinstance(image_bgr, np.ndarray) and image_bgr.ndim >= 2:
            height, width = image_bgr.shape[:2]
            metadata["image_width"] = int(width)
            metadata["image_height"] = int(height)
        return metadata

    def _validate_bgr_image(self, image_bgr: Any) -> str | None:
        if image_bgr is None:
            return "image_missing"
        if not isinstance(image_bgr, np.ndarray):
            return "image_not_ndarray"
        if image_bgr.size == 0:
            return "image_empty"
        if image_bgr.ndim != 3:
            return "image_invalid_shape"
        if image_bgr.shape[0] <= 0 or image_bgr.shape[1] <= 0:
            return "image_empty"
        if image_bgr.shape[2] < 3:
            return "image_invalid_channels"
        return None

    def _letterbox(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Letterbox to a square canvas of size self.imgsz.
        Returns:
          - canvas (imgsz x imgsz x 3, BGR)
          - scale s
          - (pad_left, pad_top)
        """
        h, w = img_bgr.shape[:2]
        s = self.imgsz / max(h, w)
        nh, nw = int(round(h * s)), int(round(w * s))
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        pad_top = (self.imgsz - nh) // 2
        pad_left = (self.imgsz - nw) // 2
        canvas[pad_top:pad_top + nh, pad_left:pad_left + nw] = resized
        return canvas, s, (pad_left, pad_top)

    def _preprocess(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Force a consistent preprocessing that matches common YOLO-NAS exports:
          - Letterbox to self.imgsz
          - BGR -> RGB
          - Scale to [0,1]
          - NCHW float32, add batch
        """
        canvas, s, pad = self._letterbox(img_bgr)
        img = canvas[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, [0,1]
        img = np.transpose(img, (2, 0, 1))                   # HWC -> CHW
        img = np.expand_dims(img, 0)                         # (1,3,S,S)
        return img, s, pad

    def _pick_boxes_scores(self, outputs: list[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Robustly pick boxes (N,4) and scores (N,C) from outputs.
        Handles both (N,C) and (C,N) score layouts.
        """
        if not isinstance(outputs, (list, tuple)) or len(outputs) < 2:
            raise RuntimeError("YOLO-NAS ONNX expected two outputs (boxes, scores).")

        o1, o2 = outputs[0], outputs[1]

        def _to_2d(x: np.ndarray) -> np.ndarray:
            # (1,N,K) -> (N,K); (N,K) stays as is
            return x[0] if x.ndim == 3 else x

        a, b = _to_2d(o1), _to_2d(o2)

        def _maybe_transpose_scores(scores: np.ndarray, n_boxes: int) -> np.ndarray:
            # Want (N,C). If we see (C,N), transpose it.
            if scores.ndim != 2:
                return scores
            if scores.shape[0] == n_boxes and scores.shape[1] > 4:
                return scores                 # (N,C)
            if scores.shape[1] == n_boxes and scores.shape[0] > 4:
                return scores.T               # (C,N) -> (N,C)
            return scores

        # Identify which tensor is boxes and which is scores
        if a.ndim == 2 and a.shape[1] == 4:
            boxes, scores = a, _maybe_transpose_scores(b, a.shape[0])
        elif b.ndim == 2 and b.shape[1] == 4:
            boxes, scores = b, _maybe_transpose_scores(a, b.shape[0])
        else:
            raise RuntimeError(f"Unexpected shapes for YOLO-NAS outputs: {a.shape}, {b.shape}")

        if scores.ndim != 2 or scores.shape[0] != boxes.shape[0]:
            raise RuntimeError(f"Scores shape {scores.shape} not aligned with boxes {boxes.shape}")
        return boxes.astype(np.float32), scores.astype(np.float32)

    def _postprocess(self, outputs, scale: float, pad: Tuple[int, int], orig_shape: Tuple[int, int]) -> np.ndarray:
        """
        Postprocess to dets: (N,6) -> (x1,y1,x2,y2,conf,cls_id)
        """
        if not isinstance(outputs, (list, tuple)):
            raise RuntimeError("YOLO-NAS: list/tuple outputs were expected.")

        boxes, scores = self._pick_boxes_scores(outputs)  # boxes: (N,4) xyxy, scores: (N,C)

        # Optional sanity: labels.txt length vs model class count
        num_classes = int(scores.shape[1])
        if len(self.names) != num_classes:
            logger.warning(
                "labels.txt length (%s) != model classes (%s). Class names may be misaligned.",
                len(self.names),
                num_classes,
            )

        # class + confidence per detection
        cls_ids = np.argmax(scores, axis=1).astype(np.int32)
        conf = scores[np.arange(scores.shape[0]), cls_ids].astype(np.float32)

        # threshold
        keep = conf >= self.conf_thr
        if not np.any(keep):
            return np.zeros((0, 6), dtype=np.float32)

        boxes = boxes[keep]
        conf = conf[keep]
        cls_ids = cls_ids[keep]

        # de-letterbox back to original image coords
        pad_left, pad_top = pad
        scale_inv = 1.0 / scale
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) * scale_inv
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) * scale_inv

        # clip to image
        H, W = orig_shape
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H - 1)

        dets = np.concatenate([boxes, conf[:, None], cls_ids[:, None]], axis=1)  # (M,6)

        # NMS
        keep_idx = nms(dets[:, :4], dets[:, 4], iou_thr=self.iou_thr)
        return dets[keep_idx].astype(np.float32)

    # ----------------------- Public API -----------------------
    def detect(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Run detection once and return:
          dets:   (N,6) -> (x1,y1,x2,y2,conf,cls_id)
          labels: List[str] (class names per detection)
        """
        blob, s, pad = self._preprocess(img_bgr)
        outputs = self.session.run(None, {self.input_name: blob})
        dets = self._postprocess(outputs, s, pad, img_bgr.shape[:2])

        labels: List[str] = []
        if dets.size > 0:
            cls_ids = dets[:, 5].astype(int)
            for c in cls_ids:
                if 0 <= c < len(self.names):
                    labels.append(self.names[c])
                else:
                    labels.append(f"id_{int(c)}")
        return dets, labels

    def detect_draw_and_labels(self, img_bgr: np.ndarray, out_dir: str = "data") -> Tuple[str, List[str]]:
        """
        Draw detections, save to disk, and return (saved_path, labels).
        """
        dets, labels = self.detect(img_bgr)
        drawn = draw_dets(img_bgr.copy(), dets, self.names)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"detect_{int(time.time())}.jpg")
        cv2.imwrite(path, drawn)
        return path, labels

    def detect_and_draw(self, img_bgr: np.ndarray, out_dir: str = "data") -> str:
        """
        Backward-compat: only return drawn image path.
        """
        path, _ = self.detect_draw_and_labels(img_bgr, out_dir=out_dir)
        return path
    
    def detect_from_bgr(self, image_bgr):
        """
        Tarayıcıdan gelen BGR numpy resmi alır.
        _preprocess -> onnx -> _postprocess -> kutular/etiketler döner.
        """
        # 1) preprocess
        blob, s, pad = self._preprocess(image_bgr)
        # 2) onnx inference
        outputs = self.session.run(None, {self.input_name: blob})
        # 3) postprocess -> dets: (N,6) [x1,y1,x2,y2,conf,cls_id]
        dets = self._postprocess(outputs, scale=s, pad=pad, orig_shape=image_bgr.shape[:2])

        if dets.size == 0:
            # Boş sonuç dönerken tipler tutarlı olsun
            boxes = []
            labels = []
            scores = []
            cls_ids = []
            return boxes, labels, scores, cls_ids

        # Kutuları ayrı döndür (N,4)
        boxes = dets[:, :4].astype(float).tolist()
        # Skorlar (N,)
        scores = dets[:, 4].astype(float).tolist()
        # Sınıf id'leri (N,)
        cls_ids_np = dets[:, 5].astype(int)
        cls_ids = cls_ids_np.tolist()
        # Etiket isimleri
        labels = [self.names[c] if 0 <= c < len(self.names) else str(int(c)) for c in cls_ids_np]
        return boxes, labels, scores, cls_ids

    def detect_structured(self, image_bgr, image_source: str | None = None) -> DetectionResult:
        """
        Run YOLO and return the shared structured detection schema.

        Latency covers BGR validation, preprocessing, ONNX inference,
        postprocessing, and structured result construction.
        """
        started = time.perf_counter()
        metadata = self._image_metadata(image_bgr)
        validation_error = self._validate_bgr_image(image_bgr)
        if validation_error:
            return DetectionResult(
                objects=[],
                image_source=image_source,
                model_name=self.model_name,
                latency_ms=_elapsed_ms(started),
                status=DETECTION_STATUS_INVALID_IMAGE,
                error=validation_error,
                metadata=metadata,
            )

        try:
            boxes, labels, scores, cls_ids = self.detect_from_bgr(image_bgr)
        except Exception:
            logger.exception("YOLO structured detection failed")
            return DetectionResult(
                objects=[],
                image_source=image_source,
                model_name=self.model_name,
                latency_ms=_elapsed_ms(started),
                status=DETECTION_STATUS_MODEL_ERROR,
                error="yolo_inference_failed",
                metadata=metadata,
            )

        return detection_result_from_legacy(
            labels,
            boxes,
            scores,
            cls_ids,
            image_source=image_source,
            model_name=self.model_name,
            latency_ms=_elapsed_ms(started),
            metadata=metadata,
        )
