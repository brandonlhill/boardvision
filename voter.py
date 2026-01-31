# voter.py
#
# Purpose:
#   - Generic, stateful voter for EXACTLY TWO detection models (for now)
#   - No framework assumptions (YOLO / FRCNN / etc.)
#   - Uses Pydantic models for inputs, outputs, and params
#   - All debug output is model-name driven
#
# Future-ready:
#   - Can be generalized to N models by replacing pairwise logic
#   - Scoring and fusion logic already isolated
#
# Dependencies:
#   - models.py must define:
#       - DetectionModel
#       - VoterParams

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from basemodels import DetectionModel, VoterParams

LOGGER = logging.getLogger("boardvision.voter")

def iou(box_a: List[int], box_b: List[int]) -> float:
    # Geometry helpers
    """
    Compute IoU for two [x1,y1,x2,y2] boxes.
    Robust, framework-agnostic.
    """
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])

    return inter_area / (area_a + area_b - inter_area + 1e-9)

class TwoModelVoter:
    """
    A stateful voter that merges detections from TWO arbitrary models.

    Design goals:
      - Generic: model names are opaque strings
      - Deterministic and explainable
      - Debug output always names models explicitly
      - No GUI or ONNX dependencies
      - Pydantic everywhere

    This class intentionally supports EXACTLY TWO models.
    """

    def __init__(
        self,
        model_a: str,
        model_b: str,
        f1_scores: Dict[str, Dict[str, float]],
        params: VoterParams,
    ) -> None:
        self.model_a = model_a
        self.model_b = model_b
        self.f1_scores = f1_scores
        self.params = params

        LOGGER.info(
            "TwoModelVoter initialized | models=[%s, %s] | params=%s",
            self.model_a,
            self.model_b,
            self.params.model_dump(),
        )
        
    # Internal helpers
    def _f1_for(self, label: str, model: str) -> float:
        """
        Safe F1 lookup.
        Missing values default to 0.0 (never crash).
        """
        return float(self.f1_scores.get(label, {}).get(model, 0.0))

    def _score(self, det: DetectionModel) -> float:
        """
        Unified scoring function. 

        score = (conf ** gamma) * f1(label, source)
        """
        conf = max(0.0, float(det.conf))
        score = conf ** self.params.gamma # take into consideration both models

        if self.params.use_f1:
            score *= self._f1_for(det.label, det.source)

        return score

    def _fuse_boxes(
        self,
        a: DetectionModel,
        b: DetectionModel,
        wa: float,
        wb: float,
    ) -> List[int]:
        """
        Score-weighted bounding box fusion.
        """
        denom = wa + wb + 1e-9
        return [
            int((a.bbox[i] * wa + b.bbox[i] * wb) / denom)
            for i in range(4)
        ]

    # Core voting algorithm
    def vote(
        self,
        dets_a: List[DetectionModel],
        dets_b: List[DetectionModel],
    ) -> Tuple[List[DetectionModel], List[DetectionModel]]:
        """
        Perform voting between model_a and model_b.

        Returns:
          final_detections: List[DetectionModel] (source="FINAL")
          candidates:       All candidate detections (for visualization)
        """

        final: List[DetectionModel] = []
        candidates: List[DetectionModel] = list(dets_a) + list(dets_b)

        used_a = [False] * len(dets_a)
        used_b = [False] * len(dets_b)

        LOGGER.debug(
            "Vote start | %s=%d dets | %s=%d dets",
            self.model_a, len(dets_a),
            self.model_b, len(dets_b),
        )

        # -------------------------------------------------
        # 1. AGREEMENT PASS
        # -------------------------------------------------
        # Match detections by:
        #   - same class label
        #   - max IoU >= iou_thresh
        # -------------------------------------------------

        for i, da in enumerate(dets_a):
            best_j = -1
            best_iou = 0.0

            for j, db in enumerate(dets_b):
                if used_b[j]:
                    continue
                if da.label != db.label:
                    continue

                ov = iou(da.bbox, db.bbox)
                if ov > best_iou:
                    best_iou = ov
                    best_j = j

            if best_j >= 0 and best_iou >= self.params.iou_thresh:
                db = dets_b[best_j]

                sa = self._score(da) # TODOL: Should have db, da both
                sb = self._score(db)

                LOGGER.debug(
                    "[AGREE] cls=%s IoU=%.3f | %s(score=%.4f) vs %s(score=%.4f)",
                    da.label,
                    best_iou,
                    self.model_a, sa,
                    self.model_b, sb,
                )

                if self.params.fuse_coords:
                    bbox = self._fuse_boxes(da, db, sa, sb)
                    conf = max(da.conf, db.conf)
                else:
                    winner = da if sa >= sb else db
                    bbox = winner.bbox
                    conf = winner.conf

                final.append(
                    DetectionModel(
                        bbox=bbox,
                        label=da.label,
                        conf=float(conf),
                        source="FINAL",
                    )
                )

                used_a[i] = True
                used_b[best_j] = True

        # -------------------------------------------------
        # 2. SOLO PASS
        # -------------------------------------------------
        # Unmatched detections are evaluated using:
        #   - solo_strong
        #   - f1 advantage
        #   - near-tie confidence fallback
        # -------------------------------------------------
        # For legacy boardvision users note this: Steps 2 & 3 are merged into a single loop over both models.
        for dets, used, model, other in (
            (dets_a, used_a, self.model_a, self.model_b),
            (dets_b, used_b, self.model_b, self.model_a),
        ):
            for i, d in enumerate(dets):
                if used[i]:
                    continue

                conf = float(d.conf)

                # NOTE: handle non-F1 mode like the old voter_merge
                if not self.params.use_f1:
                    if conf >= self.params.solo_strong or conf >= self.params.conf_thresh:
                        LOGGER.debug(
                            "[SOLO ACCEPT] model=%s cls=%s conf=%.3f reason=conf_only",
                            model,
                            d.label,
                            conf,
                        )
                        final.append(d.copy(update={"source": "FINAL"}))
                    # Skip F1-based logic entirely
                    continue

                f1_self = self._f1_for(d.label, model)
                f1_other = self._f1_for(d.label, other)

                accept = False
                reason = ""

                if conf >= self.params.solo_strong:
                    accept = True
                    reason = "solo_strong"
                elif f1_self > f1_other and conf >= self.params.conf_thresh:
                    accept = True
                    reason = "f1_advantage"
                elif abs(f1_self - f1_other) <= self.params.f1_margin and conf >= self.params.near_tie_conf:
                    accept = True
                    reason = "near_tie"

                if accept:
                    LOGGER.debug(
                        "[SOLO ACCEPT] model=%s cls=%s conf=%.3f f1=%.3f reason=%s",
                        model,
                        d.label,
                        conf,
                        f1_self,
                        reason,
                    )
                    final.append(d.copy(update={"source": "FINAL"}))


        LOGGER.debug(
            "Vote end | final=%d candidates=%d",
            len(final),
            len(candidates),
        )

        return final, candidates

# # ALGO:
# """
# Voter Algorithm: For each detection, bounding boxes from both models are matched by class and overlap (IoU). 
# When the same class is predicted for the same region, class F1 scores (from the config) are compared. The box
# from the higher-F1 model is selected if its confidence exceeds a threshold, and it is shown as the final output.

# If a box is produced by only one model (no significant overlap), its class F1 is compared with the other models 
# F1. The detection is accepted if the detecting models F1 is higher and its confidence is high. If F1 scores 
# differ by <=5% and confidence is ≥0.95, the detection is also accepted.

# On the VOTER panel, all candidate boxes from YOLOv7 (orange) and Faster R-CNN (blue) are displayed. Boxes 
# chosen by these rules are highlighted in bold yellow and labeled FINAL. An explainable ensemble is thus f
# ormed that favors the most reliable model while allowing very confident outliers.
# """

# import json
# import cv2
# import numpy as np
# import logging
# import numpy as np
# from typing import List, Dict, Tuple, Any

# LOGGER = logging.getLogger("boardvision.voter")

# @DeprecationWarning
# def _iou(boxA, boxB):
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])
#     interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
#     if interArea == 0:
#         return 0.0
#     boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
#     boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
#     iou_val = interArea / float(boxAArea + boxBArea - interArea)
#     return iou_val


# def load_f1_config(config_path: str = "config.json") -> Dict[str, Dict[str, float]]:
#     with open(config_path) as f:
#         cfg = json.load(f)
#     LOGGER.debug("Loaded F1 config with %d classes from %s", len(cfg), config_path)
#     return cfg

# def iou(boxA: List[int], boxB: List[int]) -> float:
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])

#     inter_w = max(0, xB - xA + 1)
#     inter_h = max(0, yB - yA + 1)
#     interArea = inter_w * inter_h
#     if interArea == 0:
#         return 0.0

#     boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
#     boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
#     return interArea / float(boxAArea + boxBArea - interArea)

# def voter_merge(
#     yolo_results: List[Dict[str, Any]],
#     frcnn_results: List[Dict[str, Any]],
#     f1_config: Dict[str, Dict[str, float]],
#     conf_thresh: float = 0.50,      # Base acceptance threshold when no special rule applies
#     solo_strong: float = 0.95,      # Always wins if solo" threshold
#     iou_thresh: float = 0.40,       # Requested overlap for agreement
#     f1_margin: float = 0.05,        # Keep close-F1 exception
#     gamma: float = 1.5,             # Boosts high-confidence a bit when scoring
#     fuse_coords: bool = True,       # If True, do score-weighted box fusion on agreement
#     near_tie_conf=0.95,
#     use_f1: bool = True,
# ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:

#     def score(det, model_name):
#         conf = max(0.0, float(det.get('conf', 0.0)))
#         base = conf ** gamma
#         if use_f1:
#             cls = det['label']
#             f1 = float(f1_config.get(cls, {}).get('YOLOv7' if model_name == 'YOLOv7' else 'FRCNN', 0.0))
#             base *= f1
#         return base

#     def fuse_box(a: List[int], b: List[int], wa: float, wb: float) -> List[int]:
#         denom = max(wa + wb, 1e-9)
#         x1 = int((a[0]*wa + b[0]*wb) / denom)
#         y1 = int((a[1]*wa + b[1]*wb) / denom)
#         x2 = int((a[2]*wa + b[2]*wb) / denom)
#         y2 = int((a[3]*wa + b[3]*wb) / denom)
#         return [x1, y1, x2, y2]

#     def det_str(d: Dict[str, Any]) -> str:
#         return f"{d.get('label')} conf={float(d.get('conf', 0.0)):.3f} bbox={d.get('bbox')}"

#     final_boxes: List[Dict[str, Any]] = []
#     candidates: List[Dict[str, Any]] = []

#     yolo_ann = [dict(d, source='YOLOv7') for d in (yolo_results or [])]
#     frc_ann  = [dict(d, source='FasterRCNN') for d in (frcnn_results or [])]
#     candidates.extend(yolo_ann)
#     candidates.extend(frc_ann)

#     LOGGER.debug(
#         "Voter start: %d YOLO, %d FRCNN | conf_thresh=%.2f solo_strong=%.2f iou_thresh=%.2f f1_margin=%.2f gamma=%.2f fuse=%s",
#         len(yolo_ann), len(frc_ann), conf_thresh, solo_strong, iou_thresh, f1_margin, gamma, fuse_coords
#     )

#     y_used = [False] * len(yolo_ann)
#     f_used = [False] * len(frc_ann)

#     # Step 1: agreement pairing by class + IoU >= iou_thresh
#     for i, yd in enumerate(yolo_ann):
#         best_j, best_iou = -1, 0.0
#         for j, fd in enumerate(frc_ann):
#             if yd['label'] != fd['label'] or f_used[j]:
#                 continue
#             ov = iou(yd['bbox'], fd['bbox'])
#             if ov > best_iou:
#                 best_iou, best_j = ov, j

#         if best_j >= 0 and best_iou >= iou_thresh:
#             fd = frc_ann[best_j]
#             y_s = score(yd, 'YOLOv7')
#             f_s = score(fd, 'FRCNN')
#             LOGGER.debug(
#                 "[AGREE] cls=%s IoU=%.3f | YOLO:(%s) score=%.4f | FRCNN:(%s) score=%.4f",
#                 yd['label'], best_iou, det_str(yd), y_s, det_str(fd), f_s
#             )

#             if fuse_coords:
#                 wbox = fuse_box(yd['bbox'], fd['bbox'], y_s, f_s)
#                 winner = {
#                     'bbox': wbox,
#                     'label': yd['label'],  # same class
#                     'conf': max(float(yd.get('conf', 0.0)), float(fd.get('conf', 0.0))),
#                     'source': 'FINAL'
#                 }
#                 LOGGER.debug("   -> FUSED to bbox=%s conf=%.3f", wbox, winner['conf'])
#             else:
#                 chosen = 'YOLOv7' if y_s >= f_s else 'FasterRCNN'
#                 base = yd if y_s >= f_s else fd
#                 winner = dict(base, source='FINAL')
#                 LOGGER.debug("   -> CHOSE %s as winner", chosen)

#             final_boxes.append(winner)
#             y_used[i] = True
#             f_used[best_j] = True
#         else:
#             LOGGER.debug(
#                 "[NO AGREEMENT] YOLO #%d %s bestIoU=%.3f (thresh=%.3f)",
#                 i, det_str(yd), best_iou, iou_thresh
#             )

#     # Step 2: solo (unmatched) YOLO boxes
#     for i, yd in enumerate(yolo_ann):
#         if y_used[i]: continue
#         conf = float(yd.get('conf', 0.0))
#         if not use_f1:
#             if conf >= solo_strong or conf >= conf_thresh:
#                 final_boxes.append(dict(yd, source='FINAL'))
#             continue
#         # original F1-based branches:
#         cls = yd['label']
#         f1_y = float(f1_config.get(cls, {}).get('YOLOv7', 0.0))
#         f1_f = float(f1_config.get(cls, {}).get('FRCNN', 0.0))
#         if conf >= solo_strong:
#             final_boxes.append(dict(yd, source='FINAL'))
#         elif f1_y > f1_f and conf >= conf_thresh:                      # F1 advantage:contentReference[oaicite:0]{index=0}
#             final_boxes.append(dict(yd, source='FINAL'))
#         elif abs(f1_y - f1_f) <= f1_margin and conf >= near_tie_conf:  # near-tie fallback:contentReference[oaicite:1]{index=1}
#             final_boxes.append(dict(yd, source='FINAL'))

#     # Step 3: solo (unmatched) FRCNN boxes
#     for j, fd in enumerate(frc_ann):
#         if f_used[j]:
#             continue
#         cls = fd['label']
#         f1_y = float(f1_config.get(cls, {}).get('YOLOv7', 0.0))
#         f1_f = float(f1_config.get(cls, {}).get('FRCNN', 0.0))
#         conf = float(fd.get('conf', 0.0))

#         if conf >= solo_strong:
#             final_boxes.append(dict(fd, source='FINAL'))
#             LOGGER.debug("[FRCNN SOLO STRONG] %s (>= %.2f) -> FINAL", det_str(fd), solo_strong)
#         elif f1_f > f1_y and conf >= conf_thresh:
#             final_boxes.append(dict(fd, source='FINAL'))
#             LOGGER.debug("[FRCNN F1 ADVANTAGE] %s (f1_f=%.3f > f1_y=%.3f, conf>=%.2f) -> FINAL",
#                          det_str(fd), f1_f, f1_y, conf_thresh)
#         elif abs(f1_f - f1_y) <= f1_margin and conf >= 0.95:
#             final_boxes.append(dict(fd, source='FINAL'))
#             LOGGER.debug("[FRCNN CLOSE-F1 HI-CONF] %s (|ΔF1|<=%.2f, conf>=0.95) -> FINAL",
#                          det_str(fd), f1_margin)
#         else:
#             LOGGER.debug("[FRCNN SOLO REJECT] %s", det_str(fd))

#     LOGGER.debug("Voter end: %d FINAL, %d candidates", len(final_boxes), len(candidates))
#     return final_boxes, candidates

# def draw_voter_boxes_on_frame(frame, final_boxes, candidates):
#     """
#     Draw voter candidates and FINAL boxes with white label backgrounds.
#     """
#     img = frame.copy()
#     h, w = img.shape[:2]

#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 0.5
#     font_thickness = 1
#     pad = 4

#     def draw_label(text, x, y):
#         (tw, th), baseline = cv2.getTextSize(
#             text, font, font_scale, font_thickness
#         )
#         y_text = max(0, y)
#         y_bg1 = max(0, y_text - th - baseline - pad)
#         y_bg2 = y_text + pad
#         x_bg2 = min(w - 1, x + tw + 2 * pad)

#         cv2.rectangle(
#             img,
#             (x, y_bg1),
#             (x_bg2, y_bg2),
#             (255, 255, 255),
#             -1,
#         )
#         cv2.putText(
#             img,
#             text,
#             (x + pad, y_text),
#             font,
#             font_scale,
#             (0, 0, 0),
#             font_thickness,
#             cv2.LINE_AA,
#         )

#     # Draw candidates (thin)
#     for det in candidates or []:
#         x1, y1, x2, y2 = det["bbox"]
#         x1, y1 = max(0, x1), max(0, y1)
#         x2, y2 = min(w - 1, x2), min(h - 1, y2)

#         src = det.get("source", "")
#         label = det.get("label", "")
#         color = (
#             (0, 128, 255) if src == "YOLOv7"
#             else (255, 128, 0) if src == "FasterRCNN"
#             else (160, 160, 160)
#         )

#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
#         draw_label(f"{label} {src}", x1, y1 - 6)

#     # Draw FINAL boxes (bold yellow)
#     for det in final_boxes or []:
#         x1, y1, x2, y2 = det["bbox"]
#         x1, y1 = max(0, x1), max(0, y1)
#         x2, y2 = min(w - 1, x2), min(h - 1, y2)

#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 3)
#         draw_label(f"{det['label']} FINAL", x1, y1 - 6)

#     return img

# def overlay_label(img: np.ndarray, label: str, color=(0,255,255)) -> np.ndarray:
#     cv2.putText(img, label, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 3)
#     return img
