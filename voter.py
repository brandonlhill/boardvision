# ALGO:
"""
Voter Algorithm: For each detection, bounding boxes from both models are matched by class and overlap (IoU). 
When the same class is predicted for the same region, class F1 scores (from the config) are compared. The box
from the higher-F1 model is selected if its confidence exceeds a threshold, and it is shown as the final output.

If a box is produced by only one model (no significant overlap), its class F1 is compared with the other models 
F1. The detection is accepted if the detecting models F1 is higher and its confidence is high. If F1 scores 
differ by <=5% and confidence is ≥0.95, the detection is also accepted.

On the VOTER panel, all candidate boxes from YOLOv7 (orange) and Faster R-CNN (blue) are displayed. Boxes 
chosen by these rules are highlighted in bold yellow and labeled FINAL. An explainable ensemble is thus f
ormed that favors the most reliable model while allowing very confident outliers.
"""

import json
import cv2
import numpy as np
import json
import logging
import cv2
import numpy as np
from typing import List, Dict, Tuple, Any

LOGGER = logging.getLogger("boardvision.voter")

@DeprecationWarning
def _iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou_val = interArea / float(boxAArea + boxBArea - interArea)
    return iou_val

@DeprecationWarning
def _voter_merge(yolo_results, frcnn_results, f1_config, conf_thresh=0.5, iou_thresh=0.5, f1_margin=0.05):
    yolo_used = [False] * len(yolo_results)
    frcnn_used = [False] * len(frcnn_results)
    final_boxes = []

    # Candidate boxes for debugging/drawing
    candidates = []

    # Step 1: Match overlapping detections by class and IoU
    for i, y_det in enumerate(yolo_results):
        best_iou = 0
        best_j = -1
        for j, f_det in enumerate(frcnn_results):
            if y_det['label'] == f_det['label']:
                curr_iou = iou(y_det['bbox'], f_det['bbox'])
                if curr_iou > best_iou:
                    best_iou = curr_iou
                    best_j = j
        if best_iou > iou_thresh:
            class_name = y_det['label']
            f1_yolo = f1_config[class_name]["YOLOv7"]
            f1_frcnn = f1_config[class_name]["FRCNN"]
            yolo_box = dict(y_det)
            yolo_box['source'] = "YOLOv7"
            frcnn_box = dict(frcnn_results[best_j])
            frcnn_box['source'] = "FasterRCNN"
            candidates.append(yolo_box)
            candidates.append(frcnn_box)
            # Winner based on higher F1
            if f1_yolo >= f1_frcnn:
                if y_det['conf'] > conf_thresh:
                    winner = dict(y_det)
                    winner['source'] = "FINAL"
                    final_boxes.append(winner)
            else:
                if frcnn_results[best_j]['conf'] > conf_thresh:
                    winner = dict(frcnn_results[best_j])
                    winner['source'] = "FINAL"
                    final_boxes.append(winner)
            yolo_used[i] = True
            frcnn_used[best_j] = True

    # Step 2: Handle unmatched YOLO boxes
    for i, y_det in enumerate(yolo_results):
        if not yolo_used[i]:
            class_name = y_det['label']
            f1_yolo = f1_config[class_name]["YOLOv7"]
            f1_frcnn = f1_config[class_name]["FRCNN"]
            # F1 margin logic
            if f1_yolo > f1_frcnn and y_det['conf'] > conf_thresh:
                winner = dict(y_det)
                winner['source'] = "FINAL"
                final_boxes.append(winner)
                candidates.append(dict(y_det, source="YOLOv7"))
            elif abs(f1_yolo - f1_frcnn) < f1_margin and y_det['conf'] > 0.95:
                winner = dict(y_det)
                winner['source'] = "FINAL"
                final_boxes.append(winner)
                candidates.append(dict(y_det, source="YOLOv7"))
            else:
                candidates.append(dict(y_det, source="YOLOv7"))

    # Step 3: Handle unmatched FRCNN boxes
    for j, f_det in enumerate(frcnn_results):
        if not frcnn_used[j]:
            class_name = f_det['label']
            f1_yolo = f1_config[class_name]["YOLOv7"]
            f1_frcnn = f1_config[class_name]["FRCNN"]
            if f1_frcnn > f1_yolo and f_det['conf'] > conf_thresh:
                winner = dict(f_det)
                winner['source'] = "FINAL"
                final_boxes.append(winner)
                candidates.append(dict(f_det, source="FasterRCNN"))
            elif abs(f1_frcnn - f1_yolo) < f1_margin and f_det['conf'] > 0.95:
                winner = dict(f_det)
                winner['source'] = "FINAL"
                final_boxes.append(winner)
                candidates.append(dict(f_det, source="FasterRCNN"))
            else:
                candidates.append(dict(f_det, source="FasterRCNN"))

    return final_boxes, candidates

def load_f1_config(config_path: str = "config.json") -> Dict[str, Dict[str, float]]:
    with open(config_path) as f:
        cfg = json.load(f)
    LOGGER.debug("Loaded F1 config with %d classes from %s", len(cfg), config_path)
    return cfg

def iou(boxA: List[int], boxB: List[int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA + 1)
    inter_h = max(0, yB - yA + 1)
    interArea = inter_w * inter_h
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)

def voter_merge(
    yolo_results: List[Dict[str, Any]],
    frcnn_results: List[Dict[str, Any]],
    f1_config: Dict[str, Dict[str, float]],
    conf_thresh: float = 0.50,      # Base acceptance threshold when no special rule applies
    solo_strong: float = 0.95,      # Always wins if solo" threshold
    iou_thresh: float = 0.40,       # Requested overlap for agreement
    f1_margin: float = 0.05,        # Keep close-F1 exception
    gamma: float = 1.5,             # Boosts high-confidence a bit when scoring
    fuse_coords: bool = True        # If True, do score-weighted box fusion on agreement
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Improved voter with detailed logging.
    """

    def score(det: Dict[str, Any], model_name: str) -> float:
        cls = det['label']
        f1 = float(f1_config.get(cls, {}).get('YOLOv7' if model_name == 'YOLOv7' else 'FRCNN', 0.0))
        conf = max(0.0, float(det.get('conf', 0.0)))
        return (conf ** gamma) * f1

    def fuse_box(a: List[int], b: List[int], wa: float, wb: float) -> List[int]:
        denom = max(wa + wb, 1e-9)
        x1 = int((a[0]*wa + b[0]*wb) / denom)
        y1 = int((a[1]*wa + b[1]*wb) / denom)
        x2 = int((a[2]*wa + b[2]*wb) / denom)
        y2 = int((a[3]*wa + b[3]*wb) / denom)
        return [x1, y1, x2, y2]

    def det_str(d: Dict[str, Any]) -> str:
        return f"{d.get('label')} conf={float(d.get('conf', 0.0)):.3f} bbox={d.get('bbox')}"

    final_boxes: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []

    yolo_ann = [dict(d, source='YOLOv7') for d in (yolo_results or [])]
    frc_ann  = [dict(d, source='FasterRCNN') for d in (frcnn_results or [])]
    candidates.extend(yolo_ann)
    candidates.extend(frc_ann)

    LOGGER.debug(
        "Voter start: %d YOLO, %d FRCNN | conf_thresh=%.2f solo_strong=%.2f iou_thresh=%.2f f1_margin=%.2f gamma=%.2f fuse=%s",
        len(yolo_ann), len(frc_ann), conf_thresh, solo_strong, iou_thresh, f1_margin, gamma, fuse_coords
    )

    y_used = [False] * len(yolo_ann)
    f_used = [False] * len(frc_ann)

    # Step 1: agreement pairing by class + IoU >= iou_thresh
    for i, yd in enumerate(yolo_ann):
        best_j, best_iou = -1, 0.0
        for j, fd in enumerate(frc_ann):
            if yd['label'] != fd['label'] or f_used[j]:
                continue
            ov = iou(yd['bbox'], fd['bbox'])
            if ov > best_iou:
                best_iou, best_j = ov, j

        if best_j >= 0 and best_iou >= iou_thresh:
            fd = frc_ann[best_j]
            y_s = score(yd, 'YOLOv7')
            f_s = score(fd, 'FRCNN')
            LOGGER.debug(
                "[AGREE] cls=%s IoU=%.3f | YOLO:(%s) score=%.4f | FRCNN:(%s) score=%.4f",
                yd['label'], best_iou, det_str(yd), y_s, det_str(fd), f_s
            )

            if fuse_coords:
                wbox = fuse_box(yd['bbox'], fd['bbox'], y_s, f_s)
                winner = {
                    'bbox': wbox,
                    'label': yd['label'],  # same class
                    'conf': max(float(yd.get('conf', 0.0)), float(fd.get('conf', 0.0))),
                    'source': 'FINAL'
                }
                LOGGER.debug("   -> FUSED to bbox=%s conf=%.3f", wbox, winner['conf'])
            else:
                chosen = 'YOLOv7' if y_s >= f_s else 'FasterRCNN'
                base = yd if y_s >= f_s else fd
                winner = dict(base, source='FINAL')
                LOGGER.debug("   -> CHOSE %s as winner", chosen)

            final_boxes.append(winner)
            y_used[i] = True
            f_used[best_j] = True
        else:
            LOGGER.debug(
                "[NO AGREEMENT] YOLO #%d %s bestIoU=%.3f (thresh=%.3f)",
                i, det_str(yd), best_iou, iou_thresh
            )

    # Step 2: solo (unmatched) YOLO boxes
    for i, yd in enumerate(yolo_ann):
        if y_used[i]:
            continue
        cls = yd['label']
        f1_y = float(f1_config.get(cls, {}).get('YOLOv7', 0.0))
        f1_f = float(f1_config.get(cls, {}).get('FRCNN', 0.0))
        conf = float(yd.get('conf', 0.0))

        if conf >= solo_strong:
            final_boxes.append(dict(yd, source='FINAL'))
            LOGGER.debug("[YOLO SOLO STRONG] %s (>= %.2f) -> FINAL", det_str(yd), solo_strong)
        elif f1_y > f1_f and conf >= conf_thresh:
            final_boxes.append(dict(yd, source='FINAL'))
            LOGGER.debug("[YOLO F1 ADVANTAGE] %s (f1_y=%.3f > f1_f=%.3f, conf>=%.2f) -> FINAL",
                         det_str(yd), f1_y, f1_f, conf_thresh)
        elif abs(f1_y - f1_f) <= f1_margin and conf >= 0.95:
            final_boxes.append(dict(yd, source='FINAL'))
            LOGGER.debug("[YOLO CLOSE-F1 HI-CONF] %s (|ΔF1|<=%.2f, conf>=0.95) -> FINAL",
                         det_str(yd), f1_margin)
        else:
            LOGGER.debug("[YOLO SOLO REJECT] %s", det_str(yd))

    # Step 3: solo (unmatched) FRCNN boxes
    for j, fd in enumerate(frc_ann):
        if f_used[j]:
            continue
        cls = fd['label']
        f1_y = float(f1_config.get(cls, {}).get('YOLOv7', 0.0))
        f1_f = float(f1_config.get(cls, {}).get('FRCNN', 0.0))
        conf = float(fd.get('conf', 0.0))

        if conf >= solo_strong:
            final_boxes.append(dict(fd, source='FINAL'))
            LOGGER.debug("[FRCNN SOLO STRONG] %s (>= %.2f) -> FINAL", det_str(fd), solo_strong)
        elif f1_f > f1_y and conf >= conf_thresh:
            final_boxes.append(dict(fd, source='FINAL'))
            LOGGER.debug("[FRCNN F1 ADVANTAGE] %s (f1_f=%.3f > f1_y=%.3f, conf>=%.2f) -> FINAL",
                         det_str(fd), f1_f, f1_y, conf_thresh)
        elif abs(f1_f - f1_y) <= f1_margin and conf >= 0.95:
            final_boxes.append(dict(fd, source='FINAL'))
            LOGGER.debug("[FRCNN CLOSE-F1 HI-CONF] %s (|ΔF1|<=%.2f, conf>=0.95) -> FINAL",
                         det_str(fd), f1_margin)
        else:
            LOGGER.debug("[FRCNN SOLO REJECT] %s", det_str(fd))

    LOGGER.debug("Voter end: %d FINAL, %d candidates", len(final_boxes), len(candidates))
    return final_boxes, candidates


def draw_voter_boxes_on_frame(frame: np.ndarray,
                              final_boxes: List[Dict[str, Any]],
                              candidates: List[Dict[str, Any]]) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]

    # Draw candidates first (thin)
    for det in candidates:
        x1, y1, x2, y2 = det['bbox']
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        label = det['label']
        src = det.get('source', "")
        color = (0, 128, 255) if src == "YOLOv7" else (255, 128, 0) if src == "FasterRCNN" else (160,160,160)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        cv2.putText(img, f"{label} {src}", (x1, max(0, y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw final winners (bold yellow)
    for det in final_boxes:
        x1, y1, x2, y2 = det['bbox']
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        label = det['label']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,255), 3)
        cv2.putText(img, f"{label} FINAL", (x1, max(0, y1-10)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,255,255), 2)

    return img


def overlay_label(img: np.ndarray, label: str, color=(0,255,255)) -> np.ndarray:
    cv2.putText(img, label, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 3)
    return img
