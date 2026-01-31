from __future__ import annotations

import os
import math
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort

LOGGER = logging.getLogger("boardvision.frame_inference")


# Providers / device selection #####################################
def _softmax(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax.
    """
    x = x.astype(np.float32)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def _clip_bbox_xyxy(x1, y1, x2, y2, w, h):
    """
    Clip bbox to image bounds and cast to int.
    """
    x1 = int(max(0, min(w - 1, x1)))
    y1 = int(max(0, min(h - 1, y1)))
    x2 = int(max(0, min(w - 1, x2)))
    y2 = int(max(0, min(h - 1, y2)))
    return [x1, y1, x2, y2]

def available_providers() -> List[str]:
    try:
        return list(ort.get_available_providers())
    except Exception:
        return ["CPUExecutionProvider"]


def _parse_device(device: str) -> Tuple[str, int]:
    """
    Accepts:
      "cpu"
      "cuda"
      "cuda:0", "cuda:1"
    Returns (kind, gpu_id)
    """
    d = (device or "cpu").strip().lower()
    if d.startswith("cuda"):
        gpu_id = 0
        if ":" in d:
            try:
                gpu_id = int(d.split(":", 1)[1])
            except Exception:
                gpu_id = 0
        return "cuda", gpu_id
    return "cpu", 0


def select_providers(device: str = "cpu") -> List[Any]:
    """
    Returns providers list in priority order with CPU fallback.
    """
    kind, gpu_id = _parse_device(device)
    avail = set(available_providers())
    providers: List[Any] = []

    if kind == "cuda" and "CUDAExecutionProvider" in avail:
        providers.append(("CUDAExecutionProvider", {"device_id": int(gpu_id)}))

    providers.append("CPUExecutionProvider")
    return providers

def _map_xyxy_to_original(meta: PreprocessMeta, x1, y1, x2, y2) -> Tuple[float, float, float, float]:
    """
    Map model-space xyxy coords back to original frame coords.
    - If preprocess was resize (scale_x/scale_y set): multiply by scale_x/scale_y.
    - Else assume letterbox: undo pad + scale.
    """
    if meta.scale_x is not None and meta.scale_y is not None:
        # resize_to_square path
        x1 = float(x1) * float(meta.scale_x)
        x2 = float(x2) * float(meta.scale_x)
        y1 = float(y1) * float(meta.scale_y)
        y2 = float(y2) * float(meta.scale_y)
        return x1, y1, x2, y2

    # letterbox path
    x1 = (float(x1) - float(meta.pad_x)) / float(meta.scale)
    x2 = (float(x2) - float(meta.pad_x)) / float(meta.scale)
    y1 = (float(y1) - float(meta.pad_y)) / float(meta.scale)
    y2 = (float(y2) - float(meta.pad_y)) / float(meta.scale)
    return x1, y1, x2, y2

# Geometry helpers #####################################
@dataclass
class PreprocessMeta:
    orig_h: int
    orig_w: int
    inp_h: int
    inp_w: int
    scale: float
    pad_x: float
    pad_y: float
    # For non-letterbox resize:
    scale_x: Optional[float] = None
    scale_y: Optional[float] = None


def _letterbox(
    bgr: np.ndarray,
    new_shape: Tuple[int, int],
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, PreprocessMeta]:
    """
    Standard YOLO letterbox: resize with aspect ratio, pad evenly on all sides.
    Returns RGB float32 CHW (1,3,H,W) plus mapping metadata to project boxes back.
    """
    orig_h, orig_w = bgr.shape[:2]
    inp_h, inp_w = int(new_shape[0]), int(new_shape[1])

    # Scale (min ratio)
    r = min(inp_w / orig_w, inp_h / orig_h)
    new_unpad_w = int(round(orig_w * r))
    new_unpad_h = int(round(orig_h * r))

    # Compute padding
    dw = inp_w - new_unpad_w
    dh = inp_h - new_unpad_h
    dw /= 2
    dh /= 2

    # Resize
    resized = cv2.resize(bgr, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

    # Pad
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    # To model input
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[None, ...]  # (1,3,H,W)

    meta = PreprocessMeta(
        orig_h=orig_h, orig_w=orig_w,
        inp_h=inp_h, inp_w=inp_w,
        scale=r, pad_x=left, pad_y=top,
    )
    return blob, meta


def _resize_to_square(bgr: np.ndarray, size: int) -> Tuple[np.ndarray, PreprocessMeta]:
    """
    Simple resize (no letterbox). Useful for some FRCNN exports if you trained that way.
    """
    orig_h, orig_w = bgr.shape[:2]
    inp_h = inp_w = int(size)
    resized = cv2.resize(bgr, (inp_w, inp_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[None, ...]

    meta = PreprocessMeta(
        orig_h=orig_h, orig_w=orig_w,
        inp_h=inp_h, inp_w=inp_w,
        scale=1.0, pad_x=0.0, pad_y=0.0,
        scale_x=orig_w / float(inp_w),
        scale_y=orig_h / float(inp_h),
    )
    return blob, meta


def _clip_bbox_xyxy(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> List[int]:
    x1 = max(0.0, min(float(w - 1), x1))
    y1 = max(0.0, min(float(h - 1), y1))
    x2 = max(0.0, min(float(w - 1), x2))
    y2 = max(0.0, min(float(h - 1), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# NMS (numpy) #####################################
def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_th: float) -> List[int]:
    """
    boxes: (N,4) float xyxy
    scores: (N,) float
    returns kept indices
    """
    if boxes.size == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter + 1e-9
        iou = inter / union

        inds = np.where(iou <= float(iou_th))[0]
        order = order[inds + 1]
    return keep


# Common detection format #####################################

Detection = Dict[str, Any]

# def apply_box_deltas(box, delta):
#     """
#     box: [x1, y1, x2, y2]
#     delta: [dx, dy, dw, dh]
#     """
#     x1, y1, x2, y2 = box
#     w = x2 - x1
#     h = y2 - y1
#     cx = x1 + 0.5 * w
#     cy = y1 + 0.5 * h

#     dx, dy, dw, dh = delta

#     pred_cx = cx + dx * w
#     pred_cy = cy + dy * h
#     pred_w  = w * np.exp(dw)
#     pred_h  = h * np.exp(dh)

#     x1 = pred_cx - 0.5 * pred_w
#     y1 = pred_cy - 0.5 * pred_h
#     x2 = pred_cx + 0.5 * pred_w
#     y2 = pred_cy + 0.5 * pred_h

#     return x1, y1, x2, y2


# def draw_detections(
#     frame_bgr: np.ndarray,
#     dets: List[Detection],
#     title: str = "",
#     box_color: Tuple[int, int, int] = (0, 128, 255),
#     thickness: int = 2,
# ) -> np.ndarray:
#     """
#     Draw boxes with a WHITE background label so text is always readable.
#     Does NOT modify label strings.
#     """
#     img = frame_bgr.copy()
#     for d in dets or []:
#         bbox = d.get("bbox")
#         if not bbox or len(bbox) != 4:
#             continue
#         x1, y1, x2, y2 = map(int, bbox)
#         cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness)

#         label = str(d.get("label", ""))
#         conf = d.get("conf", None)
#         if isinstance(conf, (float, int, np.floating, np.integer)):
#             text = f"{label} {float(conf):.3f}"
#         else:
#             text = f"{label}"

#         # Label background (white)
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 0.55
#         font_th = 1
#         (tw, th), baseline = cv2.getTextSize(text, font, font_scale, font_th)
#         pad = 4
#         y_text = max(0, y1 - 6)
#         x_bg1 = x1
#         y_bg1 = max(0, y_text - th - baseline - pad)
#         x_bg2 = min(img.shape[1] - 1, x1 + tw + 2 * pad)
#         y_bg2 = min(img.shape[0] - 1, y_text + pad)

#         cv2.rectangle(img, (x_bg1, y_bg1), (x_bg2, y_bg2), (255, 255, 255), -1)
#         cv2.putText(img, text, (x1 + pad, y_text), font, font_scale, (0, 0, 0), font_th, cv2.LINE_AA)

#     if title:
#         cv2.putText(img, title, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
#     return img


# Detector base classes #####################################
@dataclass
class DetectorConfig:
    name: str
    det_type: str
    onnx_path: str
    input_size: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    enabled: bool = True
    preprocess: str = "auto"  # "letterbox", "resize", "auto"


class BaseONNXDetector:
    """
    Produces detections in *original frame coordinates*.

    Output detection format:
      {"bbox":[x1,y1,x2,y2], "label":str, "conf":float, "source":detector_name}
    """
    def __init__(self, cfg: DetectorConfig, global_cfg: dict, device: str = "cpu"):
        self.cfg = cfg
        self.name = cfg.name
        self.det_type = cfg.det_type
        self.onnx_path = cfg.onnx_path
        self.input_size = int(cfg.input_size)
        self.conf_thresh = float(cfg.conf_threshold)
        self.iou_thresh = float(cfg.iou_threshold)
        self.preprocess_mode = (cfg.preprocess or "auto").lower()

        self.class_names: List[str] = list(global_cfg.get("classes", []))

        providers = select_providers(device=device)
        LOGGER.info("Loading %s ONNX (%s) providers=%s", self.name, self.onnx_path, providers)

        so = ort.SessionOptions()
        # slightly safer defaults
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(self.onnx_path, providers=providers, sess_options=so)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Help debugging
        try:
            ins = self.session.get_inputs()[0]
            LOGGER.debug("[%s] input=%s shape=%s type=%s", self.name, ins.name, ins.shape, ins.type)
            for o in self.session.get_outputs():
                LOGGER.debug("[%s] output=%s shape=%s type=%s", self.name, o.name, o.shape, o.type)
        except Exception:
            pass

    def set_conf_threshold(self, v: float) -> None:
        self.conf_thresh = float(v)

    def _preprocess(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, PreprocessMeta]:
        """
        Default heuristics:
          - YOLO => letterbox
          - FRCNN => resize (unless you explicitly set preprocess: letterbox)
        """
        if self.preprocess_mode == "letterbox":
            return _letterbox(frame_bgr, (self.input_size, self.input_size))
        if self.preprocess_mode == "resize":
            return _resize_to_square(frame_bgr, self.input_size)

        # auto
        if self.det_type in ("yolo", "yolov7", "yololike"):
            return _letterbox(frame_bgr, (self.input_size, self.input_size))
        return _resize_to_square(frame_bgr, self.input_size)

    def infer(self, frame_bgr: np.ndarray) -> List[Detection]:
        raise NotImplementedError


# YOLO parser (NEED REWORK) #####################################
class YOLODetector(BaseONNXDetector):
    """
    Proper YOLOv7 detect-head decoder.
    Expects output shape: (1, 3, H, W, 5+nc)
    """

    def infer(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        blob, meta = self._preprocess(frame_bgr)
        outs = self.session.run(self.output_names, {self.input_name: blob})
        if not outs:
            return []

        pred = np.asarray(outs[0])  # (1, na, ny, nx, 5+nc)
        if pred.ndim != 5:
            LOGGER.warning("[%s] Unexpected YOLO output shape %s", self.name, pred.shape)
            return []

        _, na, ny, nx, ch = pred.shape
        nc = ch - 5
        if nc <= 0:
            LOGGER.error("[%s] Invalid YOLO channels=%d", self.name, ch)
            return []

        # YOLOv7 stride
        stride = meta.inp_w // nx

        anchors = np.array([
            [12, 16], [19, 36], [40, 28],
            [36, 75], [76, 55], [72, 146],
            [142,110],[192,243],[459,401],
        ], dtype=np.float32)[:na]

        pred = pred[0]  # drop batch
        pred = pred.reshape(na, ny, nx, ch)

        # ---- VECTORIZE EVERYTHING ----
        obj = 1.0 / (1.0 + np.exp(-pred[..., 4]))
        mask = obj >= self.conf_thresh
        if not np.any(mask):
            return []

        cls_scores = 1.0 / (1.0 + np.exp(-pred[..., 5:]))
        cls_id = np.argmax(cls_scores, axis=-1)
        cls_conf = np.take_along_axis(cls_scores, cls_id[..., None], axis=-1)[..., 0]

        conf = obj * cls_conf
        mask &= conf >= self.conf_thresh
        if not np.any(mask):
            return []

        # Grid
        yv, xv = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")

        tx = pred[..., 0]
        ty = pred[..., 1]
        tw = pred[..., 2]
        th = pred[..., 3]

        cx = (1.0 / (1.0 + np.exp(-tx)) * 2.0 - 0.5 + xv) * stride
        cy = (1.0 / (1.0 + np.exp(-ty)) * 2.0 - 0.5 + yv) * stride

        bw = (1.0 / (1.0 + np.exp(-tw)) * 2.0) ** 2 * anchors[:, 0][:, None, None]
        bh = (1.0 / (1.0 + np.exp(-th)) * 2.0) ** 2 * anchors[:, 1][:, None, None]

        x1 = (cx - bw / 2 - meta.pad_x) / meta.scale
        y1 = (cy - bh / 2 - meta.pad_y) / meta.scale
        x2 = (cx + bw / 2 - meta.pad_x) / meta.scale
        y2 = (cy + bh / 2 - meta.pad_y) / meta.scale

        x1 = np.clip(x1, 0, meta.orig_w - 1)
        y1 = np.clip(y1, 0, meta.orig_h - 1)
        x2 = np.clip(x2, 0, meta.orig_w - 1)
        y2 = np.clip(y2, 0, meta.orig_h - 1)

        # Flatten valid detections
        idx = np.where(mask)
        dets: List[Dict[str, Any]] = []

        for a, iy, ix in zip(*idx):
            dets.append({
                "bbox": [
                    int(x1[a, iy, ix]),
                    int(y1[a, iy, ix]),
                    int(x2[a, iy, ix]),
                    int(y2[a, iy, ix]),
                ],
                "label": self.class_names[int(cls_id[a, iy, ix])],
                "conf": float(conf[a, iy, ix]),
                "source": self.name,
            })

        return dets

    def _infer_legacy(self, frame_bgr: np.ndarray) -> List[Detection]:
        blob, meta = self._preprocess(frame_bgr)
        outs = self.session.run(self.output_names, {self.input_name: blob})
        if not outs:
            return []

        pred = np.asarray(outs[0])  # (1,3,H,W,5+nc)
        if pred.ndim != 5:
            LOGGER.warning("[%s] Unexpected YOLO output shape %s", self.name, pred.shape)
            return []

        _, na, ny, nx, ch = pred.shape
        nc = ch - 5
        if nc <= 0:
            LOGGER.error("[%s] Invalid YOLO channels=%d", self.name, ch)
            return []

        # YOLOv7 default strides for 640 input
        stride = meta.inp_w // nx  # e.g. 640 / 80 = 8

        # TODO: MUST MATCH TRAINING CONFIG (REWORK! )
        anchors = np.array([
            [12, 16], [19, 36], [40, 28],   # P3
            [36, 75], [76, 55], [72, 146],  # P4 (not used here)
            [142,110],[192,243],[459,401]  # P5 (not used here)
        ], dtype=np.float32)

        anchors = anchors[:na]  # match head anchors

        pred = pred[0]  # drop batch
        pred = pred.reshape(na, ny, nx, ch)

        dets: List[Detection] = []

        for a in range(na):
            for iy in range(ny):
                for ix in range(nx):
                    p = pred[a, iy, ix]

                    obj = _sigmoid(p[4])
                    if obj < self.conf_thresh:
                        continue

                    cls_scores = _sigmoid(p[5:])
                    cls_id = int(np.argmax(cls_scores))
                    cls_conf = float(cls_scores[cls_id])
                    conf = obj * cls_conf
                    if conf < self.conf_thresh:
                        continue

                    # Decode box
                    tx, ty, tw, th = p[0:4]
                    cx = ( _sigmoid(tx) * 2 - 0.5 + ix ) * stride
                    cy = ( _sigmoid(ty) * 2 - 0.5 + iy ) * stride
                    bw = ( _sigmoid(tw) * 2 ) ** 2 * anchors[a, 0]
                    bh = ( _sigmoid(th) * 2 ) ** 2 * anchors[a, 1]

                    x1 = cx - bw / 2
                    y1 = cy - bh / 2
                    x2 = cx + bw / 2
                    y2 = cy + bh / 2

                    # Undo letterbox
                    x1 = (x1 - meta.pad_x) / meta.scale
                    x2 = (x2 - meta.pad_x) / meta.scale
                    y1 = (y1 - meta.pad_y) / meta.scale
                    y2 = (y2 - meta.pad_y) / meta.scale

                    bbox = _clip_bbox_xyxy(x1, y1, x2, y2, meta.orig_w, meta.orig_h)
                    label = self.class_names[cls_id]

                    dets.append({
                        "bbox": bbox,
                        "label": label,   # 🔒 unchanged
                        "conf": float(conf),
                        "source": self.name,
                    })

        return dets


# FasterRCNN parser #####################################
class FasterRCNNDetector(BaseONNXDetector):
    """
    Decoder for FasterRCNNNoNMS ONNX export.

    ONNX outputs:
      class_logits   [N, C]
      box_regression [N, C*4]
      proposals      [N, 4]   (xyxy, image scale)
    """
    def infer(self, frame_bgr: np.ndarray) -> List[Detection]:
        blob, meta = self._preprocess(frame_bgr)
        outs = self.session.run(self.output_names, {self.input_name: blob})

        if len(outs) != 3:
            LOGGER.error("[%s] Expected 3 outputs (class_logits, box_reg, proposals), got %d", self.name, len(outs))
            return []

        class_logits, box_reg, proposals = map(np.asarray, outs)

        # Remove batch dim if present
        if class_logits.ndim == 3:
            class_logits = class_logits[0]
        if box_reg.ndim == 3 and box_reg.shape[0] == 1 and box_reg.shape[-1] != 4:
            box_reg = box_reg[0]
        if proposals.ndim == 3 and proposals.shape[0] == 1:
            proposals = proposals[0]

        if class_logits.ndim != 2:
            LOGGER.error("[%s] class_logits shape unexpected: %s", self.name, class_logits.shape)
            return []
        N, C = class_logits.shape

        # box_reg => [N, C, 4]
        if box_reg.ndim == 2:
            if box_reg.shape[0] != N or box_reg.shape[1] != C * 4:
                LOGGER.error("[%s] box_reg shape unexpected: %s (expected [N, C*4])", self.name, box_reg.shape)
                return []
            box_reg_view = box_reg.reshape(N, C, 4)
        elif box_reg.ndim == 3:
            if box_reg.shape[0] != N or box_reg.shape[1] != C or box_reg.shape[2] != 4:
                LOGGER.error("[%s] box_reg shape unexpected: %s (expected [N,C,4])", self.name, box_reg.shape)
                return []
            box_reg_view = box_reg
        else:
            LOGGER.error("[%s] box_reg ndim unexpected: %s", self.name, box_reg.shape)
            return []

        if proposals.ndim != 2 or proposals.shape[1] != 4:
            LOGGER.error("[%s] proposals shape unexpected: %s", self.name, proposals.shape)
            return []

        if proposals.shape[0] != N:
            LOGGER.warning("[%s] proposals count mismatch: proposals=%d logits=%d (using min)", self.name, proposals.shape[0], N)
            N = min(N, proposals.shape[0])
            class_logits = class_logits[:N]
            box_reg_view = box_reg_view[:N]
            proposals = proposals[:N]

        # softmax
        logits = class_logits.astype(np.float32)
        logits = logits - np.max(logits, axis=1, keepdims=True)
        expv = np.exp(logits)
        probs = expv / np.sum(expv, axis=1, keepdims=True)

        cls_id = np.argmax(probs, axis=1).astype(np.int32)
        score = probs[np.arange(N), cls_id].astype(np.float32)

        # drop background + threshold
        keep = (cls_id != 0) & (score >= float(self.conf_thresh))
        if not np.any(keep):
            return []

        proposals_k = proposals[keep].astype(np.float32)
        cls_id_k = cls_id[keep]
        score_k = score[keep]
        deltas_k = box_reg_view[keep, cls_id_k].astype(np.float32)

        # --- BoxCoder weights (CRITICAL) ---
        wx, wy, ww, wh = 10.0, 10.0, 5.0, 5.0
        dx = deltas_k[:, 0] / wx
        dy = deltas_k[:, 1] / wy
        dw = deltas_k[:, 2] / ww
        dh = deltas_k[:, 3] / wh

        # exp clip like torchvision
        bbox_xform_clip = float(math.log(1000.0 / 16.0))
        dw = np.clip(dw, -bbox_xform_clip, bbox_xform_clip)
        dh = np.clip(dh, -bbox_xform_clip, bbox_xform_clip)

        px1, py1, px2, py2 = proposals_k[:, 0], proposals_k[:, 1], proposals_k[:, 2], proposals_k[:, 3]
        pw = (px2 - px1)
        ph = (py2 - py1)

        valid = (pw > 1.0) & (ph > 1.0)
        if not np.any(valid):
            return []

        px1, py1, px2, py2 = px1[valid], py1[valid], px2[valid], py2[valid]
        pw, ph = pw[valid], ph[valid]
        dx, dy, dw, dh = dx[valid], dy[valid], dw[valid], dh[valid]
        cls_id_k = cls_id_k[valid]
        score_k = score_k[valid]

        cx = px1 + 0.5 * pw
        cy = py1 + 0.5 * ph
        pred_cx = dx * pw + cx
        pred_cy = dy * ph + cy
        pred_w = np.exp(dw) * pw
        pred_h = np.exp(dh) * ph

        x1 = pred_cx - 0.5 * pred_w
        y1 = pred_cy - 0.5 * pred_h
        x2 = pred_cx + 0.5 * pred_w
        y2 = pred_cy + 0.5 * pred_h

        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

        # --- per-class NMS (since ONNX is NMS-free) ---
        final_idxs = []
        for c in np.unique(cls_id_k):
            inds = np.where(cls_id_k == c)[0]
            keep_local = _nms_xyxy(boxes[inds], score_k[inds], float(self.iou_thresh))
            final_idxs.append(inds[keep_local])

        if not final_idxs:
            return []

        final = np.concatenate(final_idxs, axis=0)
        boxes = boxes[final]
        score_k = score_k[final]
        cls_id_k = cls_id_k[final]

        dets: List[Detection] = []
        for b, conf, cid in zip(boxes, score_k, cls_id_k):
            x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])

            # map back to original
            x1, y1, x2, y2 = _map_xyxy_to_original(meta, x1, y1, x2, y2)
            bbox = _clip_bbox_xyxy(x1, y1, x2, y2, meta.orig_w, meta.orig_h)

            # background is class 0, YAML classes are foreground-only
            label_idx = int(cid)
            label = self.class_names[label_idx] if 0 <= label_idx < len(self.class_names) else str(int(cid))

            dets.append({
                "bbox": bbox,
                "label": label,
                "conf": float(conf),
                "source": self.name,
            })
    
        return dets

# Engine / registry #####################################
DETECTOR_TYPE_REGISTRY = {
    "yolo": YOLODetector,
    "yolov7": YOLODetector,
    "yololike": YOLODetector,
    "frcnn": FasterRCNNDetector,
    "fasterrcnn": FasterRCNNDetector,
}


class MultiDetectorEngine:
    """
    Loads multiple detectors from config["detectors"].

    YAML example:
      device: cuda:0
      detectors:
        yolov7:
          type: yolov7
          onnx_path: ...
          input_size: 640
          conf_threshold: 0.25
          iou_threshold: 0.45
          preprocess: letterbox
        frcnn:
          type: frcnn
          onnx_path: ...
          input_size: 800
          conf_threshold: 0.50
          preprocess: resize
      classes: [...]
    """
    def __init__(self, cfg: dict):
        LOGGER.info("MultiDetectorEngine invoked.")
        self.cfg = cfg or {}
        self.detectors: Dict[str, BaseONNXDetector] = {}

        # Hacky to get GPU inference
        device_cfg = self.cfg.get("device", "cpu")
        if isinstance(device_cfg, dict):
            pref = str(device_cfg.get("preferred", "cpu")).lower()
            gpu_id = int(device_cfg.get("gpu_id", 0))
            device = f"cuda:{gpu_id}" if pref == "cuda" else "cpu"
        else:
            device = str(device_cfg)

        dets = self.cfg.get("detectors", {}) or {}

        for name, det_cfg in dets.items():
            if not (det_cfg or {}).get("enabled", True):
                continue

            det_type = str(det_cfg.get("type", "yolo")).lower().strip()
            cls = DETECTOR_TYPE_REGISTRY.get(det_type)
            if cls is None:
                raise ValueError(f"Unknown detector type '{det_type}' for detector '{name}'")

            onnx_path = str(det_cfg.get("onnx_path", "")).strip()
            if not onnx_path:
                raise ValueError(f"Detector '{name}' missing 'onnx_path'")

            dc = DetectorConfig(
                name=str(name),
                det_type=det_type,
                onnx_path=onnx_path,
                input_size=int(det_cfg.get("input_size", 640)),
                conf_threshold=float(det_cfg.get("conf_threshold", 0.25)),
                iou_threshold=float(det_cfg.get("iou_threshold", 0.45)),
                enabled=bool(det_cfg.get("enabled", True)),
                preprocess=str(det_cfg.get("preprocess", "auto")),
            )

            self.detectors[str(name)] = cls(dc, self.cfg, device=device)

        if not self.detectors:
            raise ValueError("No detectors enabled/loaded from config['detectors'].")

    def set_detector_conf(self, name: str, conf: float) -> None:
        if name in self.detectors:
            self.detectors[name].set_conf_threshold(conf)

    def run(self, frame_bgr: np.ndarray) -> Dict[str, List[Detection]]:
        out: Dict[str, List[Detection]] = {}
        for name, det in self.detectors.items():
            try:
                out[name] = det.infer(frame_bgr)
            except Exception as e:
                LOGGER.exception("Detector '%s' failed: %s", name, e)
                out[name] = []
        return out
