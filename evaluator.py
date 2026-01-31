import os
import cv2
import torch
import yaml
import time
import argparse
import numpy as np
from glob import glob
from typing import Callable, Optional, Tuple, List, Dict
from rich.console import Console

from basemodels import DetectionModel, VoterParams
from voter import TwoModelVoter, iou

console = Console()

try:
    # Weighted Boxes Fusion from ZFTurbo / ensemble-boxes
    from ensemble_boxes import weighted_boxes_fusion
except ImportError:  # graceful fallback if not installed
    weighted_boxes_fusion = None
    console.print(
        "[yellow]Warning:[/yellow] ensemble-boxes is not installed. "
        "WBF ensemble will be disabled."
    )

# Defaults / env overrides ################################################
DEFAULT_CONFIG_PATH = os.environ.get("CONFIG_PATH", "./config.yaml")

# Evaluation knobs (kept from original behavior unless config provides overrides)
DEFAULT_EVAL_IOU_THRESH = 0.5
COCO_IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)

# These are filled after loading config in main()
CLASS_NAMES: List[str] = []
NAME2ID: Dict[str, int] = {}
IOU_THRESH = DEFAULT_EVAL_IOU_THRESH

# Unified runner type
RunnerFn = Callable[[np.ndarray, object, Optional[list], str, float], list]
# runner(frame, model, classes, device, conf_thresh) -> list[pred_dict]


# Config helpers ##########################################################
def load_config_yaml(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _as_float(x, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _as_bool(x, default: bool) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "yes", "y", "on")
    return default


# Prediction helpers #####################################################
def load_yolo_labels(lbl_path, img_shape, class_names):
    """Load YOLO txt labels and convert to absolute pixel [x1,y1,x2,y2] with class NAMES."""
    h, w = img_shape[:2]
    boxes = []
    if not os.path.exists(lbl_path):
        return boxes
    with open(lbl_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, bw, bh = map(float, parts)
            cls = int(cls)
            if cls < 0 or cls >= len(class_names):
                continue
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            label = class_names[cls]
            boxes.append({"bbox": [x1, y1, x2, y2], "label": label, "conf": 1.0})
    return boxes


def _norm_bbox(b):
    if b is None:
        return [0, 0, 0, 0]
    if isinstance(b, (tuple, list)) and len(b) == 4:
        return [int(round(float(x))) for x in b]
    return [0, 0, 0, 0]


def preds_dicts_to_models(preds, source: str):
    """Convert detector dict predictions to DetectionModel, enforcing consistent source names."""
    out = []
    for p in preds or []:
        label = p.get("label") or p.get("class") or p.get("name")
        if label is None:
            continue
        conf = p.get("conf", p.get("confidence", 0.0))
        out.append(
            DetectionModel(
                bbox=_norm_bbox(p.get("bbox")),
                label=str(label),
                conf=float(conf),
                source=source,
            )
        )
    return out


def models_to_preds_dicts(dets):
    """Convert DetectionModel -> dicts expected by evaluator metrics."""
    out = []
    for d in dets or []:
        out.append(
            {
                "bbox": list(d.bbox),
                "label": d.label,
                "conf": float(d.conf),
                "source": d.source,
            }
        )
    return out


def _preds_to_wbf_inputs(preds: list, img_shape) -> Tuple[List[List[float]], List[float], List[int]]:
    """
    Convert a list of prediction dicts into WBF-compatible (boxes, scores, labels).
    boxes are normalized [x1,y1,x2,y2] in 0..1, labels are integer class IDs.
    """
    h, w = img_shape[:2]
    boxes: List[List[float]] = []
    scores: List[float] = []
    labels: List[int] = []

    for p in preds or []:
        label_name = p.get("label") or p.get("class") or p.get("name")
        if label_name not in NAME2ID:
            continue
        bbox = p.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        # normalize
        boxes.append([x1 / w, y1 / h, x2 / w, y2 / h])
        scores.append(float(p.get("conf", p.get("confidence", 0.0))))
        labels.append(NAME2ID[label_name])

    return boxes, scores, labels


def run_wbf_ensemble(
    preds_a: list,
    preds_b: list,
    img_shape,
    weights: Optional[List[float]],
    iou_thr: float,
    skip_box_thr: float,
):
    """
    Run Weighted Boxes Fusion on predictions from model A and B.

    Returns a list of prediction dicts with the same schema used elsewhere:
    { "bbox": [x1,y1,x2,y2], "label": <class_name>, "conf": float, "source": "WBF" }
    """
    if weighted_boxes_fusion is None:
        # Library not available; behave as "no predictions"
        return []

    boxes_a, scores_a, labels_a = _preds_to_wbf_inputs(preds_a, img_shape)
    boxes_b, scores_b, labels_b = _preds_to_wbf_inputs(preds_b, img_shape)

    # If both detectors produced nothing, bail early
    if not boxes_a and not boxes_b:
        return []

    boxes_list = [boxes_a, boxes_b]
    scores_list = [scores_a, scores_b]
    labels_list = [labels_a, labels_b]

    # Ensure weights length matches number of models
    if not weights or len(weights) != len(boxes_list):
        # Default to equal weights
        weights = [1.0] * len(boxes_list)

    # Call WBF from ensemble-boxes
    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=weights,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
    )

    h, w = img_shape[:2]
    merged_preds = []
    for box, score, cls_id in zip(boxes, scores, labels):
        # denormalize
        x1 = int(round(box[0] * w))
        y1 = int(round(box[1] * h))
        x2 = int(round(box[2] * w))
        y2 = int(round(box[3] * h))
        cls_id = int(cls_id)
        if 0 <= cls_id < len(CLASS_NAMES):
            label_name = CLASS_NAMES[cls_id]
        else:
            # Unknown class index; skip
            continue
        merged_preds.append(
            {
                "bbox": [x1, y1, x2, y2],
                "label": label_name,
                "conf": float(score),
                "source": "WBF",
            }
        )

    return merged_preds


# Metric helpers ##########################################################
def compute_metrics(preds, gts, iou_thresh=0.5):
    """
    Per-image micro P/R using greedy matching across all classes.
    Sort predictions by confidence (desc), match each pred to the best unused GT
    of the same class with IoU >= threshold.
    """
    matched_gt = set()
    tp, fp, fn = 0, 0, 0
    preds_sorted = sorted(preds, key=lambda d: float(d.get("conf", 0.0)), reverse=True)
    for p in preds_sorted:
        found = False
        for i, g in enumerate(gts):
            if g["label"] != p["label"] or i in matched_gt:
                continue
            if iou(p["bbox"], g["bbox"]) >= iou_thresh:
                tp += 1
                matched_gt.add(i)
                found = True
                break
        if not found:
            fp += 1
    fn = len(gts) - len(matched_gt)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return precision, recall, tp, fp, fn


def build_pr_data(all_preds, all_gts, class_names, iou_thresh):
    """
    Build TP/FP decisions per prediction for PR curves (dataset-wide),
    using greedy matching per class, per image, sorted by confidence.
    Returns:
      tp_flags: list of 0/1 per prediction (dataset-wide)
      confs:    list of confidences per prediction
      pred_cls: list of predicted class IDs per prediction
      n_gt_per_class: array length C with total GT count per class
    """
    tp_flags, confs, pred_cls = [], [], []
    n_gt_per_class = np.zeros(len(class_names), dtype=int)

    for preds, gts in zip(all_preds, all_gts):
        for g in gts:
            if g["label"] in NAME2ID:
                n_gt_per_class[NAME2ID[g["label"]]] += 1

        # Organize GT by class and usage flags
        gt_by_class = {c: [] for c in class_names}
        gt_used_by_class = {c: [] for c in class_names}
        for g in gts:
            if g["label"] in gt_by_class:
                gt_by_class[g["label"]].append(g["bbox"])
                gt_used_by_class[g["label"]].append(False)

        preds_sorted = sorted(preds, key=lambda d: float(d.get("conf", 0.0)), reverse=True)
        for p in preds_sorted:
            c = p["label"]
            conf = float(p.get("conf", 0.0))
            pred_cls.append(NAME2ID.get(c, -1))
            confs.append(conf)

            if c not in gt_by_class or len(gt_by_class[c]) == 0:
                tp_flags.append(0)
                continue

            best_iou, best_idx = 0.0, -1
            for i, (gbox, used) in enumerate(zip(gt_by_class[c], gt_used_by_class[c])):
                if used:
                    continue
                ov = iou(p["bbox"], gbox)
                if ov > best_iou:
                    best_iou, best_idx = ov, i

            if best_idx >= 0 and best_iou >= iou_thresh:
                tp_flags.append(1)
                gt_used_by_class[c][best_idx] = True
            else:
                tp_flags.append(0)

    return (
        np.array(tp_flags, np.int32),
        np.array(confs, np.float32),
        np.array(pred_cls, np.int32),
        n_gt_per_class,
    )


def per_class_counts_for_image(preds, gts, class_names, iou_thresh):
    """
    Greedy one-to-one matching per CLASS on a single image.
    Returns dicts: tp, fp, fn keyed by class name.
    """
    tp = {c: 0 for c in class_names}
    fp = {c: 0 for c in class_names}
    fn = {c: 0 for c in class_names}

    gt_by_class = {c: [] for c in class_names}
    used_by_class = {c: [] for c in class_names}
    for g in gts:
        if g["label"] in gt_by_class:
            gt_by_class[g["label"]].append(g["bbox"])
            used_by_class[g["label"]].append(False)

    preds_by_class = {c: [] for c in class_names}
    for p in preds:
        if p["label"] in preds_by_class:
            preds_by_class[p["label"]].append(p)
    for c in class_names:
        preds_by_class[c].sort(key=lambda d: float(d.get("conf", 0.0)), reverse=True)

    for c in class_names:
        for p in preds_by_class[c]:
            best_iou, best_idx = 0.0, -1
            for i, (gbox, used) in enumerate(zip(gt_by_class[c], used_by_class[c])):
                if used:
                    continue
                ov = iou(p["bbox"], gbox)
                if ov > best_iou:
                    best_iou, best_idx = ov, i
            if best_idx >= 0 and best_iou >= iou_thresh:
                tp[c] += 1
                used_by_class[c][best_idx] = True
            else:
                fp[c] += 1

        fn[c] += sum(1 for u in used_by_class[c] if not u)

    return tp, fp, fn


def compute_per_class_f1(all_preds, all_gts, class_names, iou_thresh):
    """
    Dataset-level per-class P/R/F1 (sums image-level matches).
    Returns dict: class_name -> dict(P/R/F1/TP/FP/FN)
    """
    TP = {c: 0 for c in class_names}
    FP = {c: 0 for c in class_names}
    FN = {c: 0 for c in class_names}

    for preds, gts in zip(all_preds, all_gts):
        tp, fp, fn = per_class_counts_for_image(preds, gts, class_names, iou_thresh)
        for c in class_names:
            TP[c] += tp[c]
            FP[c] += fp[c]
            FN[c] += fn[c]

    results = {}
    for c in class_names:
        tp, fp, fn = TP[c], FP[c], FN[c]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        results[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "TP": tp,
            "FP": fp,
            "FN": fn,
        }
    return results


def ap_from_pr(tp_flags, confs, pred_cls, n_gt_per_class, num_classes):
    """
    Compute AP per class and mAP with precision envelope (COCO-style 101-pt interp).
    Classes with no GT are ignored in the mAP (nanmean).
    """
    ap = np.full(num_classes, np.nan, dtype=np.float32)
    p_last = np.zeros(num_classes, dtype=np.float32)
    r_last = np.zeros(num_classes, dtype=np.float32)

    order = np.argsort(-confs)
    tp_flags = tp_flags[order]
    pred_cls = pred_cls[order]

    for c in range(num_classes):
        cls_mask = pred_cls == c
        n_p = int(cls_mask.sum())
        n_g = int(n_gt_per_class[c])

        if n_g == 0 or n_p == 0:
            if n_g > 0 and n_p == 0:
                ap[c] = 0.0
            p_last[c] = 0.0
            r_last[c] = 0.0
            continue

        cls_tp = tp_flags[cls_mask].astype(np.int32)
        cls_fp = 1 - cls_tp
        tpc = np.cumsum(cls_tp)
        fpc = np.cumsum(cls_fp)

        recall = tpc / (n_g + 1e-9)
        precision = tpc / (tpc + fpc + 1e-9)

        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
        for i in range(mpre.size - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])

        rc = np.linspace(0, 1, 101)
        prec_interp = np.interp(rc, mrec, mpre)
        ap[c] = np.trapz(prec_interp, rc)

        p_last[c] = precision[-1]
        r_last[c] = recall[-1]

    mAP = float(np.nanmean(ap)) if num_classes > 0 else 0.0
    return ap, mAP, p_last, r_last


def compute_map_at_iou(all_preds, all_gts, class_names, iou_thresh):
    tp_flags, confs, pred_cls, n_gt = build_pr_data(all_preds, all_gts, class_names, iou_thresh)
    ap_per_cls, mAP, _, _ = ap_from_pr(tp_flags, confs, pred_cls, n_gt, len(class_names))
    return mAP, ap_per_cls


def compute_coco_map(all_preds, all_gts, class_names, iou_thresholds):
    aps = []
    for thr in iou_thresholds:
        m, _ = compute_map_at_iou(all_preds, all_gts, class_names, thr)
        aps.append(m)
    return float(np.mean(aps))


# Logging helpers ########################################################
def summarize(tp, fp, fn, name):
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if prec + rec > 0 else 0.0
    console.print(f"\n{name} Averages:")
    console.print(f"  Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")
    console.print(f"  TP={tp}, FP={fp}, FN={fn}")


def print_fps(name: str, total_time: float, num_frames: int):
    if num_frames <= 0 or total_time <= 0:
        console.print(f"{name} Avg FPS: N/A (no frames evaluated or zero time)")
        return
    fps = num_frames / total_time
    console.print(f"{name} Avg FPS over {num_frames} frames: {fps:.2f}")


def compute_and_print_dataset_metrics(
    model_a_name: str,
    model_b_name: str,
    a_all_preds: list,
    b_all_preds: list,
    voter_all_preds: list,
    gt_all: list,
    a_tp: int,
    a_fp: int,
    a_fn: int,
    b_tp: int,
    b_fp: int,
    b_fn: int,
    voter_tp: int,
    voter_fp: int,
    voter_fn: int,
    a_infer_time: float,
    b_infer_time: float,
    voter_time: float,
    a_frame_count: int,
    b_frame_count: int,
    num_eval_frames: int,
    class_names: list,
    iou_thresh: float,
    coco_iou_thresholds,
    # Optional WBF stats
    wbf_all_preds: Optional[list] = None,
    wbf_tp: int = 0,
    wbf_fp: int = 0,
    wbf_fn: int = 0,
    wbf_time: float = 0.0,
    wbf_enabled: bool = False,
):
    """
    Compute and print all dataset-level metrics (P/R/F1, FPS, mAP, per-class AP & F1)
    in one place. This keeps the same evaluations as the original script, and
    optionally adds WBF as a third method.
    """
    console.print("\n====================== FINAL DATASET STATS ======================")
    summarize(a_tp, a_fp, a_fn, model_a_name)
    summarize(b_tp, b_fp, b_fn, model_b_name)
    summarize(voter_tp, voter_fp, voter_fn, "Voter")

    if wbf_enabled and wbf_all_preds is not None:
        summarize(wbf_tp, wbf_fp, wbf_fn, "WBF")

    console.print("\n====================== FPS METRICS ======================")
    # Per-model FPS (detectors only)
    print_fps(model_a_name, a_infer_time, a_frame_count)
    print_fps(model_b_name, b_infer_time, b_frame_count)

    # Voter FPS reflects full pipeline: model A + model B + vote
    voter_total_time = a_infer_time + b_infer_time + voter_time
    print_fps("Voter (A+B+vote)", voter_total_time, num_eval_frames)

    # WBF FPS (optional, full pipeline: A+B+WBF)
    if wbf_enabled and wbf_all_preds is not None:
        wbf_total_time = a_infer_time + b_infer_time + wbf_time
        print_fps("WBF (A+B+WBF)", wbf_total_time, num_eval_frames)

    console.print("\n====================== mAP METRICS ======================")
    mAP50_a, ap_cls_a = compute_map_at_iou(a_all_preds, gt_all, class_names, iou_thresh)
    mAP50_b, ap_cls_b = compute_map_at_iou(b_all_preds, gt_all, class_names, iou_thresh)
    mAP50_voter, ap_cls_voter = compute_map_at_iou(voter_all_preds, gt_all, class_names, iou_thresh)

    mAP_coco_a = compute_coco_map(a_all_preds, gt_all, class_names, coco_iou_thresholds)
    mAP_coco_b = compute_coco_map(b_all_preds, gt_all, class_names, coco_iou_thresholds)
    mAP_coco_voter = compute_coco_map(voter_all_preds, gt_all, class_names, coco_iou_thresholds)

    console.print(f"{model_a_name:12s} mAP@0.5: {mAP50_a:.3f} | mAP@[.5:.95]: {mAP_coco_a:.3f}")
    console.print(f"{model_b_name:12s} mAP@0.5: {mAP50_b:.3f} | mAP@[.5:.95]: {mAP_coco_b:.3f}")
    console.print(f"Voter        mAP@0.5: {mAP50_voter:.3f} | mAP@[.5:.95]: {mAP_coco_voter:.3f}")

    ap_cls_wbf = None
    mAP50_wbf = 0.0
    mAP_coco_wbf = 0.0
    if wbf_enabled and wbf_all_preds is not None:
        mAP50_wbf, ap_cls_wbf = compute_map_at_iou(wbf_all_preds, gt_all, class_names, iou_thresh)
        mAP_coco_wbf = compute_coco_map(wbf_all_preds, gt_all, class_names, coco_iou_thresholds)
        console.print(f"WBF          mAP@0.5: {mAP50_wbf:.3f} | mAP@[.5:.95]: {mAP_coco_wbf:.3f}")

    console.print(f"\nPer-class AP@0.5 ({model_a_name}):")
    for i, ap in enumerate(ap_cls_a):
        console.print(f"  {class_names[i]:25s}: {ap:.3f}")

    console.print(f"Per-class AP@0.5 ({model_b_name}):")
    for i, ap in enumerate(ap_cls_b):
        console.print(f"  {class_names[i]:25s}: {ap:.3f}")

    console.print("Per-class AP@0.5 (Voter):")
    for i, ap in enumerate(ap_cls_voter):
        console.print(f"  {class_names[i]:25s}: {ap:.3f}")

    if wbf_enabled and ap_cls_wbf is not None:
        console.print("Per-class AP@0.5 (WBF):")
        for i, ap in enumerate(ap_cls_wbf):
            console.print(f"  {class_names[i]:25s}: {ap:.3f}")

    console.print(f"\n====================== PER-CLASS F1 (IOU={iou_thresh:.2f}) ======================")

    a_pc = compute_per_class_f1(a_all_preds, gt_all, class_names, iou_thresh)
    b_pc = compute_per_class_f1(b_all_preds, gt_all, class_names, iou_thresh)
    voter_pc = compute_per_class_f1(voter_all_preds, gt_all, class_names, iou_thresh)

    wbf_pc = None
    if wbf_enabled and wbf_all_preds is not None:
        wbf_pc = compute_per_class_f1(wbf_all_preds, gt_all, class_names, iou_thresh)

    console.print(f"\n{model_a_name} per-class:")
    for c in class_names:
        r = a_pc[c]
        console.print(
            f"  {c:25s}  F1={r['f1']:.3f}  P={r['precision']:.3f}  R={r['recall']:.3f}  "
            f"TP={r['TP']} FP={r['FP']} FN={r['FN']}"
        )

    console.print(f"\n{model_b_name} per-class:")
    for c in class_names:
        r = b_pc[c]
        console.print(
            f"  {c:25s}  F1={r['f1']:.3f}  P={r['precision']:.3f}  R={r['recall']:.3f}  "
            f"TP={r['TP']} FP={r['FP']} FN={r['FN']}"
        )

    console.print("\nVoter per-class:")
    for c in class_names:
        r = voter_pc[c]
        console.print(
            f"  {c:25s}  F1={r['f1']:.3f}  P={r['precision']:.3f}  R={r['recall']:.3f}  "
            f"TP={r['TP']} FP={r['FP']} FN={r['FN']}"
        )

    if wbf_enabled and wbf_pc is not None:
        console.print("\nWBF per-class:")
        for c in class_names:
            r = wbf_pc[c]
            console.print(
                f"  {c:25s}  F1={r['f1']:.3f}  P={r['precision']:.3f}  R={r['recall']:.3f}  "
                f"TP={r['TP']} FP={r['FP']} FN={r['FN']}"
            )


# Model loading helpers ##################################################
def resolve_device(cfg_device_section):
    """
    Prefer CUDA if available. Config can still explicitly force 'cpu' or 'mps',
    but default is to rock CUDA.
    """
    if isinstance(cfg_device_section, dict):
        cfg_device = str(cfg_device_section.get("preferred", "")).strip().lower()
    elif isinstance(cfg_device_section, str):
        cfg_device = cfg_device_section.strip().lower()
    else:
        cfg_device = ""

    # Explicit override from config
    if cfg_device in ("cuda", "cpu", "mps"):
        return cfg_device

    # Default: choose CUDA if available
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_detector_model(det_name: str, det_cfg: dict, device: str) -> Tuple[object, Optional[list], str, Optional[RunnerFn]]:
    """
    Generic loader for detector models based on config.
    Supports:
      type: "yolo"   (YOLOv7 or YOLOv8 via version/name/weights)
      type: "frcnn"/"fasterrcnn"/"faster-rcnn"
    Returns: (model, classes_or_None, type_str, runner_fn)

      - type_str is one of: "yolov7", "yolov8", "frcnn"
      - runner_fn: callable(frame, model, classes, device, conf_thresh) -> preds
    """
    if det_cfg is None:
        raise ValueError(f"No detector config found for '{det_name}'")

    det_type = str(det_cfg.get("type", "")).strip().lower()
    enabled = _as_bool(det_cfg.get("enabled", True), True)

    if not enabled:
        console.print(f"[yellow]{det_name} disabled in config[/yellow]")
        return None, None, det_type, None

    # ---------------- YOLO (v7 or v8) ----------------
    if det_type == "yolo":
        weight_path = det_cfg.get("raw_weights_path") or det_cfg.get("weights")
        version = str(det_cfg.get("version", "")).strip().lower()

        console.print(
            f"Loading YOLO-like detector '{det_name}' from: {weight_path} "
            f"(version={version or 'auto'})"
        )

        name_l = det_name.lower()
        path_l = str(weight_path or "").lower()

        is_v8 = (
            version == "v8"
            or "yolov8" in name_l
            or "yolov8" in path_l
        )

        if is_v8:
            # Lazy import YOLOv8
            from frameworks.yolov8.frame_inference import (
                load_yolov8_model,
                detect_frame as detect_frame_yolov8,
            )

            model = load_yolov8_model(weight_path, device=device)

            def runner(frame, model_, classes_, device_, conf_thresh_):
                # classes_ ignored for YOLO
                _, preds = detect_frame_yolov8(
                    frame,
                    model_,
                    device=device_,
                    conf_thresh=conf_thresh_,
                )
                return preds

            return model, None, "yolov8", runner

        else:
            # Lazy import YOLOv7
            from frameworks.yolov7.frame_inference import (
                load_yolov7_model,
                detect_frame as detect_frame_yolov7,
            )

            model = load_yolov7_model(weight_path, device=device)

            def runner(frame, model_, classes_, device_, conf_thresh_):
                # classes_ ignored for YOLO
                _, preds = detect_frame_yolov7(
                    frame,
                    model_,
                    device=device_,
                    conf_thresh=conf_thresh_,
                )
                return preds

            return model, None, "yolov7", runner

    # ---------------- Faster R-CNN ----------------
    if det_type in ("frcnn", "fasterrcnn", "faster-rcnn"):
        from frameworks.fasterrcnn.frame_inference import (
            load_fasterrcnn_model,
            run_fasterrcnn_on_frame,
        )

        weight_path = det_cfg.get("raw_weights_path") or det_cfg.get("weights")
        console.print(f"Loading FasterRCNN-like detector '{det_name}' from: {weight_path}")
        model, classes = load_fasterrcnn_model(weight_path, device=device)

        def runner(frame, model_, classes_, device_, conf_thresh_):
            _, preds = run_fasterrcnn_on_frame(
                frame,
                model_,
                classes_,
                device=device_,
                conf_thresh=conf_thresh_,
            )
            return preds

        return model, classes, "frcnn", runner

    # ---------------- Unsupported ----------------
    raise ValueError(
        f"Detector '{det_name}' has unsupported type '{det_type}'. "
        f"Supported types: 'yolo', 'frcnn'"
    )


# Main evaluation ########################################################
def main():
    global CLASS_NAMES, NAME2ID, IOU_THRESH

    parser = argparse.ArgumentParser(description="Two-model detection evaluator with voting + WBF ensemble.")
    parser.add_argument(
        "-c",
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to config.yaml (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--img-dir",
        required=True,
        help="Directory with test images",
    )
    parser.add_argument(
        "--lbl-dir",
        required=True,
        help="Directory with YOLO-format labels",
    )
    args = parser.parse_args()

    console.print(f"Using config: [bold]{args.config}[/bold]")
    cfg = load_config_yaml(args.config)

    # Required config keys
    CLASS_NAMES = list(cfg.get("classes") or [])
    if not CLASS_NAMES:
        raise ValueError("config.yaml is missing required key: classes")

    NAME2ID = {name: i for i, name in enumerate(CLASS_NAMES)}

    det_cfg = cfg.get("detectors") or {}
    voter_cfg = cfg.get("voter") or {}
    f1_scores = cfg.get("f1_scores") or {}
    device_cfg = cfg.get("device")
    wbf_cfg = cfg.get("wbf") or {}

    if len(det_cfg) < 2:
        raise ValueError(
            "config.detectors must define at least two detectors. "
            "The first two (in order) are used as model A and model B."
        )

    # Choose model A/B from FIRST TWO detectors in config (order matters)
    detector_names = list(det_cfg.keys())
    model_a_name, model_b_name = detector_names[0], detector_names[1]
    console.print(
        f"Model A: [bold]{model_a_name}[/bold], "
        f"Model B: [bold]{model_b_name}[/bold] (from first two config.detectors)"
    )

    # Device (default CUDA if available)
    DEVICE = resolve_device(device_cfg)
    console.print(f"Using device: [bold]{DEVICE}[/bold]")
    is_cuda = str(DEVICE).lower().startswith("cuda")

    # Voter params
    voter_params = VoterParams(
        conf_thresh=_as_float(voter_cfg.get("conf_thresh"), 0.5),
        solo_strong=_as_float(voter_cfg.get("solo_strong"), 0.95),
        iou_thresh=_as_float(voter_cfg.get("iou_thresh"), 0.4),
        f1_margin=_as_float(voter_cfg.get("f1_margin"), 0.05),
        gamma=_as_float(voter_cfg.get("gamma"), 1.5),
        fuse_coords=_as_bool(voter_cfg.get("fuse_coords"), True),
        near_tie_conf=_as_float(voter_cfg.get("near_tie_conf"), 0.95),
        use_f1=_as_bool(voter_cfg.get("use_f1"), True),
    )

    # WBF params
    wbf_enabled = _as_bool(wbf_cfg.get("enabled", False), False) and weighted_boxes_fusion is not None
    wbf_iou_thr = _as_float(wbf_cfg.get("iou_thr"), 0.6)
    wbf_skip_box_thr = _as_float(wbf_cfg.get("skip_box_thr"), 0.001)
    wbf_weights = wbf_cfg.get("weights") or [1.0, 1.0]

    if wbf_enabled:
        console.print(
            f"WBF enabled with iou_thr={wbf_iou_thr}, skip_box_thr={wbf_skip_box_thr}, "
            f"weights={wbf_weights}"
        )
    else:
        console.print("WBF disabled (enable in config under 'wbf.enabled' and install ensemble-boxes).")

    # Evaluator IoU threshold: keep original 0.5 unless config provides eval_iou_thresh
    IOU_THRESH = _as_float(cfg.get("eval_iou_thresh"), DEFAULT_EVAL_IOU_THRESH)

    # Sanity checks and config mismatches
    mismatches = []

    # f1_scores expected structure: {class_name: {detector_name: f1}}
    if f1_scores:
        for c in CLASS_NAMES:
            if c not in f1_scores:
                mismatches.append(f"config.f1_scores missing class '{c}'")
            else:
                per = f1_scores.get(c, {})
                if model_a_name not in per:
                    mismatches.append(f"config.f1_scores['{c}'] missing '{model_a_name}'")
                if model_b_name not in per:
                    mismatches.append(f"config.f1_scores['{c}'] missing '{model_b_name}'")
    else:
        mismatches.append("config.f1_scores is empty; voter will behave as if all F1=0")

    if mismatches:
        console.print("[yellow]Config mismatches detected:[/yellow]")
        for m in mismatches:
            console.print(f"  - {m}")

    # Per-detector runtime conf thresholds ################################
    det_cfg_a = det_cfg.get(model_a_name) or {}
    det_cfg_b = det_cfg.get(model_b_name) or {}

    CONF_THRESH_A = _as_float(det_cfg_a.get("conf_threshold"), 0.6)
    CONF_THRESH_B = _as_float(det_cfg_b.get("conf_threshold"), 0.9)

    ENABLED_A = _as_bool(det_cfg_a.get("enabled", True), True)
    ENABLED_B = _as_bool(det_cfg_b.get("enabled", True), True)

    # Model loading ######################################################
    model_a = model_b = None
    classes_a = classes_b = None
    runner_a = runner_b = None
    type_a = type_b = ""

    if ENABLED_A:
        model_a, classes_a, type_a, runner_a = load_detector_model(model_a_name, det_cfg_a, DEVICE)
    else:
        console.print(f"[yellow]{model_a_name} disabled in config[/yellow]")
        type_a = str(det_cfg_a.get("type", "")).strip().lower()

    if ENABLED_B:
        model_b, classes_b, type_b, runner_b = load_detector_model(model_b_name, det_cfg_b, DEVICE)
    else:
        console.print(f"[yellow]{model_b_name} disabled in config[/yellow]")
        type_b = str(det_cfg_b.get("type", "")).strip().lower()

    # Voter
    voter = TwoModelVoter(
        model_a=model_a_name,
        model_b=model_b_name,
        f1_scores=f1_scores,
        params=voter_params,
    )

    # Dataset loop #######################################################
    img_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"):
        img_files.extend(glob(os.path.join(args.img_dir, ext)))
    img_files = sorted(img_files)
    console.print(f"Found {len(img_files)} test images in '{args.img_dir}'.")

    # Accumulators for micro P/R
    a_tp = a_fp = a_fn = 0
    b_tp = b_fp = b_fn = 0
    voter_tp = voter_fp = voter_fn = 0
    wbf_tp = wbf_fp = wbf_fn = 0

    # For mAP accumulation
    a_all_preds, b_all_preds, voter_all_preds, wbf_all_preds, gt_all = [], [], [], [], []

    # For FPS stats
    a_infer_time = 0.0
    b_infer_time = 0.0
    voter_time = 0.0
    wbf_time = 0.0
    a_frame_count = 0
    b_frame_count = 0
    num_eval_frames = 0  # frames that actually went through evaluation / voting

    for img_path in img_files:
        fname = os.path.basename(img_path)
        lbl_path = os.path.join(args.lbl_dir, os.path.splitext(fname)[0] + ".txt")
        frame = cv2.imread(img_path)

        if frame is None:
            console.print(f"[WARN] Could not read {fname}, skipping.")
            continue

        num_eval_frames += 1

        # Load GT labels (YOLO format -> pixel boxes, names)
        gt_boxes = load_yolo_labels(lbl_path, frame.shape, CLASS_NAMES)
        gt_all.append(gt_boxes)

        # Run detectors
        preds_a = []
        preds_b = []

        if ENABLED_A and model_a is not None and runner_a is not None:
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            preds_a = runner_a(frame.copy(), model_a, classes_a, DEVICE, CONF_THRESH_A)
            if is_cuda:
                torch.cuda.synchronize()
            a_infer_time += time.perf_counter() - t0
            a_frame_count += 1

        if ENABLED_B and model_b is not None and runner_b is not None:
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            preds_b = runner_b(frame.copy(), model_b, classes_b, DEVICE, CONF_THRESH_B)
            if is_cuda:
                torch.cuda.synchronize()
            b_infer_time += time.perf_counter() - t0
            b_frame_count += 1

        # Voter expects DetectionModel inputs
        dets_a = preds_dicts_to_models(preds_a, source=model_a_name)
        dets_b = preds_dicts_to_models(preds_b, source=model_b_name)

        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        final_dets, _candidates = voter.vote(dets_a, dets_b)
        if is_cuda:
            torch.cuda.synchronize()
        voter_time += time.perf_counter() - t0

        final_preds = models_to_preds_dicts(final_dets)

        # Run WBF ensemble (optional)
        wbf_preds = []
        if wbf_enabled:
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            wbf_preds = run_wbf_ensemble(
                preds_a,
                preds_b,
                frame.shape,
                weights=wbf_weights,
                iou_thr=wbf_iou_thr,
                skip_box_thr=wbf_skip_box_thr,
            )
            if is_cuda:
                torch.cuda.synchronize()
            wbf_time += time.perf_counter() - t0

        # Save for mAP
        a_all_preds.append(preds_a)
        b_all_preds.append(preds_b)
        voter_all_preds.append(final_preds)
        if wbf_enabled:
            wbf_all_preds.append(wbf_preds)

        # Per-image P/R (uses IOU_THRESH)
        a_prec, a_rec, tp_a, fp_a, fn_a = compute_metrics(preds_a, gt_boxes, IOU_THRESH)
        b_prec, b_rec, tp_b, fp_b, fn_b = compute_metrics(preds_b, gt_boxes, IOU_THRESH)
        voter_prec, voter_rec, tp_v, fp_v, fn_v = compute_metrics(final_preds, gt_boxes, IOU_THRESH)

        if wbf_enabled:
            wbf_prec, wbf_rec, tp_w, fp_w, fn_w = compute_metrics(wbf_preds, gt_boxes, IOU_THRESH)
        else:
            wbf_prec = wbf_rec = 0.0
            tp_w = fp_w = fn_w = 0

        # Accumulate micro totals
        a_tp += tp_a
        a_fp += fp_a
        a_fn += fn_a
        b_tp += tp_b
        b_fp += fp_b
        b_fn += fn_b
        voter_tp += tp_v
        voter_fp += fp_v
        voter_fn += fn_v
        if wbf_enabled:
            wbf_tp += tp_w
            wbf_fp += fp_w
            wbf_fn += fn_w

        # Optional per-image logging
        console.print(f"\n📷 {fname}")
        #console.print(f"  {model_a_name:12s} -> Precision: {a_prec:.3f}, Recall: {a_rec:.3f}")
        #console.print(f"  {model_b_name:12s} -> Precision: {b_prec:.3f}, Recall: {b_rec:.3f}")
        #console.print(f"  Voter        -> Precision: {voter_prec:.3f}, Recall: {voter_rec:.3f}")
        #if wbf_enabled:
        #    console.print(f"  WBF          -> Precision: {wbf_prec:.3f}, Recall: {wbf_rec:.3f}")

    # Final stats & metrics (all printing centralized here) ###############
    compute_and_print_dataset_metrics(
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        a_all_preds=a_all_preds,
        b_all_preds=b_all_preds,
        voter_all_preds=voter_all_preds,
        gt_all=gt_all,
        a_tp=a_tp,
        a_fp=a_fp,
        a_fn=a_fn,
        b_tp=b_tp,
        b_fp=b_fp,
        b_fn=b_fn,
        voter_tp=voter_tp,
        voter_fp=voter_fp,
        voter_fn=voter_fn,
        a_infer_time=a_infer_time,
        b_infer_time=b_infer_time,
        voter_time=voter_time,
        a_frame_count=a_frame_count,
        b_frame_count=b_frame_count,
        num_eval_frames=num_eval_frames,
        class_names=CLASS_NAMES,
        iou_thresh=IOU_THRESH,
        coco_iou_thresholds=COCO_IOU_THRESHOLDS,
        wbf_all_preds=wbf_all_preds if wbf_enabled else None,
        wbf_tp=wbf_tp,
        wbf_fp=wbf_fp,
        wbf_fn=wbf_fn,
        wbf_time=wbf_time,
        wbf_enabled=wbf_enabled,
    )


if __name__ == "__main__":
    main()

# import os
# import cv2
# import torch
# import yaml
# import time
# import argparse
# import numpy as np
# from glob import glob
# from typing import Callable, Optional, Tuple
# from rich.console import Console

# from basemodels import DetectionModel, VoterParams
# from voter import TwoModelVoter, iou

# console = Console()

# # Defaults / env overrides ################################################
# DEFAULT_CONFIG_PATH = os.environ.get("CONFIG_PATH", "./config.yaml")
# # DEFAULT_IMG_DIR = os.environ.get("IMG_DIR", "./datasets/miraclefactory/test/images")
# # DEFAULT_LBL_DIR = os.environ.get("LBL_DIR", "./datasets/miraclefactory/test/labels")

# # Evaluation knobs (kept from original behavior unless config provides overrides)
# DEFAULT_EVAL_IOU_THRESH = 0.5
# COCO_IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)

# # These are filled after loading config in main()
# CLASS_NAMES = []
# NAME2ID = {}
# IOU_THRESH = DEFAULT_EVAL_IOU_THRESH

# # Unified runner type
# RunnerFn = Callable[[np.ndarray, object, Optional[list], str, float], list]
# # runner(frame, model, classes, device, conf_thresh) -> list[pred_dict]



# # Config helpers ##########################################################
# def load_config_yaml(path: str) -> dict:
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Config not found: {path}")
#     with open(path, "r") as f:
#         return yaml.safe_load(f)


# def _as_float(x, default: float) -> float:
#     try:
#         return float(x)
#     except Exception:
#         return float(default)


# def _as_bool(x, default: bool) -> bool:
#     if isinstance(x, bool):
#         return x
#     if isinstance(x, str):
#         return x.strip().lower() in ("1", "true", "yes", "y", "on")
#     return default

# # Prediction helpers #####################################################
# def load_yolo_labels(lbl_path, img_shape, class_names):
#     """Load YOLO txt labels and convert to absolute pixel [x1,y1,x2,y2] with class NAMES."""
#     h, w = img_shape[:2]
#     boxes = []
#     if not os.path.exists(lbl_path):
#         return boxes
#     with open(lbl_path, "r") as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) != 5:
#                 continue
#             cls, cx, cy, bw, bh = map(float, parts)
#             cls = int(cls)
#             if cls < 0 or cls >= len(class_names):
#                 continue
#             x1 = int((cx - bw / 2) * w)
#             y1 = int((cy - bh / 2) * h)
#             x2 = int((cx + bw / 2) * w)
#             y2 = int((cy + bh / 2) * h)
#             label = class_names[cls]
#             boxes.append({"bbox": [x1, y1, x2, y2], "label": label, "conf": 1.0})
#     return boxes


# def _norm_bbox(b):
#     if b is None:
#         return [0, 0, 0, 0]
#     if isinstance(b, (tuple, list)) and len(b) == 4:
#         return [int(round(float(x))) for x in b]
#     return [0, 0, 0, 0]


# def preds_dicts_to_models(preds, source: str):
#     """Convert detector dict predictions to DetectionModel, enforcing consistent source names."""
#     out = []
#     for p in preds or []:
#         label = p.get("label") or p.get("class") or p.get("name")
#         if label is None:
#             continue
#         conf = p.get("conf", p.get("confidence", 0.0))
#         out.append(
#             DetectionModel(
#                 bbox=_norm_bbox(p.get("bbox")),
#                 label=str(label),
#                 conf=float(conf),
#                 source=source,
#             )
#         )
#     return out


# def models_to_preds_dicts(dets):
#     """Convert DetectionModel -> dicts expected by evaluator metrics."""
#     out = []
#     for d in dets or []:
#         out.append(
#             {
#                 "bbox": list(d.bbox),
#                 "label": d.label,
#                 "conf": float(d.conf),
#                 "source": d.source,
#             }
#         )
#     return out

# # Metric helpers ##########################################################
# def compute_metrics(preds, gts, iou_thresh=0.5):
#     """
#     Per-image micro P/R using greedy matching across all classes.
#     Sort predictions by confidence (desc), match each pred to the best unused GT
#     of the same class with IoU >= threshold.
#     """
#     matched_gt = set()
#     tp, fp, fn = 0, 0, 0
#     preds_sorted = sorted(preds, key=lambda d: float(d.get("conf", 0.0)), reverse=True)
#     for p in preds_sorted:
#         found = False
#         for i, g in enumerate(gts):
#             if g["label"] != p["label"] or i in matched_gt:
#                 continue
#             if iou(p["bbox"], g["bbox"]) >= iou_thresh:
#                 tp += 1
#                 matched_gt.add(i)
#                 found = True
#                 break
#         if not found:
#             fp += 1
#     fn = len(gts) - len(matched_gt)
#     precision = tp / (tp + fp) if tp + fp > 0 else 0.0
#     recall = tp / (tp + fn) if tp + fn > 0 else 0.0
#     return precision, recall, tp, fp, fn


# def build_pr_data(all_preds, all_gts, class_names, iou_thresh):
#     """
#     Build TP/FP decisions per prediction for PR curves (dataset-wide),
#     using greedy matching per class, per image, sorted by confidence.
#     Returns:
#       tp_flags: list of 0/1 per prediction (dataset-wide)
#       confs:    list of confidences per prediction
#       pred_cls: list of predicted class IDs per prediction
#       n_gt_per_class: array length C with total GT count per class
#     """
#     tp_flags, confs, pred_cls = [], [], []
#     n_gt_per_class = np.zeros(len(class_names), dtype=int)

#     for preds, gts in zip(all_preds, all_gts):
#         for g in gts:
#             if g["label"] in NAME2ID:
#                 n_gt_per_class[NAME2ID[g["label"]]] += 1

#         # Organize GT by class and usage flags
#         gt_by_class = {c: [] for c in class_names}
#         gt_used_by_class = {c: [] for c in class_names}
#         for g in gts:
#             if g["label"] in gt_by_class:
#                 gt_by_class[g["label"]].append(g["bbox"])
#                 gt_used_by_class[g["label"]].append(False)

#         preds_sorted = sorted(preds, key=lambda d: float(d.get("conf", 0.0)), reverse=True)
#         for p in preds_sorted:
#             c = p["label"]
#             conf = float(p.get("conf", 0.0))
#             pred_cls.append(NAME2ID.get(c, -1))
#             confs.append(conf)

#             if c not in gt_by_class or len(gt_by_class[c]) == 0:
#                 tp_flags.append(0)
#                 continue

#             best_iou, best_idx = 0.0, -1
#             for i, (gbox, used) in enumerate(zip(gt_by_class[c], gt_used_by_class[c])):
#                 if used:
#                     continue
#                 ov = iou(p["bbox"], gbox)
#                 if ov > best_iou:
#                     best_iou, best_idx = ov, i

#             if best_idx >= 0 and best_iou >= iou_thresh:
#                 tp_flags.append(1)
#                 gt_used_by_class[c][best_idx] = True
#             else:
#                 tp_flags.append(0)

#     return (
#         np.array(tp_flags, np.int32),
#         np.array(confs, np.float32),
#         np.array(pred_cls, np.int32),
#         n_gt_per_class,
#     )


# def per_class_counts_for_image(preds, gts, class_names, iou_thresh):
#     """
#     Greedy one-to-one matching per CLASS on a single image.
#     Returns dicts: tp, fp, fn keyed by class name.
#     """
#     tp = {c: 0 for c in class_names}
#     fp = {c: 0 for c in class_names}
#     fn = {c: 0 for c in class_names}

#     gt_by_class = {c: [] for c in class_names}
#     used_by_class = {c: [] for c in class_names}
#     for g in gts:
#         if g["label"] in gt_by_class:
#             gt_by_class[g["label"]].append(g["bbox"])
#             used_by_class[g["label"]].append(False)

#     preds_by_class = {c: [] for c in class_names}
#     for p in preds:
#         if p["label"] in preds_by_class:
#             preds_by_class[p["label"]].append(p)
#     for c in class_names:
#         preds_by_class[c].sort(key=lambda d: float(d.get("conf", 0.0)), reverse=True)

#     for c in class_names:
#         for p in preds_by_class[c]:
#             best_iou, best_idx = 0.0, -1
#             for i, (gbox, used) in enumerate(zip(gt_by_class[c], used_by_class[c])):
#                 if used:
#                     continue
#                 ov = iou(p["bbox"], gbox)
#                 if ov > best_iou:
#                     best_iou, best_idx = ov, i
#             if best_idx >= 0 and best_iou >= iou_thresh:
#                 tp[c] += 1
#                 used_by_class[c][best_idx] = True
#             else:
#                 fp[c] += 1

#         fn[c] += sum(1 for u in used_by_class[c] if not u)

#     return tp, fp, fn


# def compute_per_class_f1(all_preds, all_gts, class_names, iou_thresh):
#     """
#     Dataset-level per-class P/R/F1 (sums image-level matches).
#     Returns dict: class_name -> dict(P/R/F1/TP/FP/FN)
#     """
#     TP = {c: 0 for c in class_names}
#     FP = {c: 0 for c in class_names}
#     FN = {c: 0 for c in class_names}

#     for preds, gts in zip(all_preds, all_gts):
#         tp, fp, fn = per_class_counts_for_image(preds, gts, class_names, iou_thresh)
#         for c in class_names:
#             TP[c] += tp[c]
#             FP[c] += fp[c]
#             FN[c] += fn[c]

#     results = {}
#     for c in class_names:
#         tp, fp, fn = TP[c], FP[c], FN[c]
#         precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#         recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#         f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
#         results[c] = {
#             "precision": precision,
#             "recall": recall,
#             "f1": f1,
#             "TP": tp,
#             "FP": fp,
#             "FN": fn,
#         }
#     return results


# def ap_from_pr(tp_flags, confs, pred_cls, n_gt_per_class, num_classes):
#     """
#     Compute AP per class and mAP with precision envelope (COCO-style 101-pt interp).
#     Classes with no GT are ignored in the mAP (nanmean).
#     """
#     ap = np.full(num_classes, np.nan, dtype=np.float32)
#     p_last = np.zeros(num_classes, dtype=np.float32)
#     r_last = np.zeros(num_classes, dtype=np.float32)

#     order = np.argsort(-confs)
#     tp_flags = tp_flags[order]
#     pred_cls = pred_cls[order]

#     for c in range(num_classes):
#         cls_mask = pred_cls == c
#         n_p = int(cls_mask.sum())
#         n_g = int(n_gt_per_class[c])

#         if n_g == 0 or n_p == 0:
#             if n_g > 0 and n_p == 0:
#                 ap[c] = 0.0
#             p_last[c] = 0.0
#             r_last[c] = 0.0
#             continue

#         cls_tp = tp_flags[cls_mask].astype(np.int32)
#         cls_fp = 1 - cls_tp
#         tpc = np.cumsum(cls_tp)
#         fpc = np.cumsum(cls_fp)

#         recall = tpc / (n_g + 1e-9)
#         precision = tpc / (tpc + fpc + 1e-9)

#         mrec = np.concatenate(([0.0], recall, [1.0]))
#         mpre = np.concatenate(([1.0], precision, [0.0]))
#         for i in range(mpre.size - 2, -1, -1):
#             mpre[i] = max(mpre[i], mpre[i + 1])

#         rc = np.linspace(0, 1, 101)
#         prec_interp = np.interp(rc, mrec, mpre)
#         ap[c] = np.trapz(prec_interp, rc)

#         p_last[c] = precision[-1]
#         r_last[c] = recall[-1]

#     mAP = float(np.nanmean(ap)) if num_classes > 0 else 0.0
#     return ap, mAP, p_last, r_last


# def compute_map_at_iou(all_preds, all_gts, class_names, iou_thresh):
#     tp_flags, confs, pred_cls, n_gt = build_pr_data(all_preds, all_gts, class_names, iou_thresh)
#     ap_per_cls, mAP, _, _ = ap_from_pr(tp_flags, confs, pred_cls, n_gt, len(class_names))
#     return mAP, ap_per_cls


# def compute_coco_map(all_preds, all_gts, class_names, iou_thresholds):
#     aps = []
#     for thr in iou_thresholds:
#         m, _ = compute_map_at_iou(all_preds, all_gts, class_names, thr)
#         aps.append(m)
#     return float(np.mean(aps))


# # Logging helpers ########################################################

# def summarize(tp, fp, fn, name):
#     prec = tp / (tp + fp) if tp + fp > 0 else 0.0
#     rec = tp / (tp + fn) if tp + fn > 0 else 0.0
#     f1 = (2 * prec * rec) / (prec + rec) if prec + rec > 0 else 0.0
#     console.print(f"\n{name} Averages:")
#     console.print(f"  Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")
#     console.print(f"  TP={tp}, FP={fp}, FN={fn}")


# def print_fps(name: str, total_time: float, num_frames: int):
#     if num_frames <= 0 or total_time <= 0:
#         console.print(f"{name} Avg FPS: N/A (no frames evaluated or zero time)")
#         return
#     fps = num_frames / total_time
#     console.print(f"{name} Avg FPS over {num_frames} frames: {fps:.2f}")


# # Model loading helpers ##################################################
# def resolve_device(cfg_device_section):
#     """
#     Prefer CUDA if available. Config can still explicitly force 'cpu' or 'mps',
#     but default is to rock CUDA.
#     """
#     if isinstance(cfg_device_section, dict):
#         cfg_device = str(cfg_device_section.get("preferred", "")).strip().lower()
#     elif isinstance(cfg_device_section, str):
#         cfg_device = cfg_device_section.strip().lower()
#     else:
#         cfg_device = ""

#     # Explicit override from config
#     if cfg_device in ("cuda", "cpu", "mps"):
#         return cfg_device

#     # Default: choose CUDA if available
#     return "cuda" if torch.cuda.is_available() else "cpu"


# def load_detector_model(det_name: str, det_cfg: dict, device: str) -> Tuple[object, Optional[list], str, Optional[RunnerFn]]:
#     """
#     Generic loader for detector models based on config.
#     Supports:
#       type: "yolo"   (YOLOv7 or YOLOv8 via version/name/weights)
#       type: "frcnn"/"fasterrcnn"/"faster-rcnn"
#     Returns: (model, classes_or_None, type_str, runner_fn)

#       - type_str is one of: "yolov7", "yolov8", "frcnn"
#       - runner_fn: callable(frame, model, classes, device, conf_thresh) -> preds
#     """
#     if det_cfg is None:
#         raise ValueError(f"No detector config found for '{det_name}'")

#     det_type = str(det_cfg.get("type", "")).strip().lower()
#     enabled = _as_bool(det_cfg.get("enabled", True), True)

#     if not enabled:
#         console.print(f"[yellow]{det_name} disabled in config[/yellow]")
#         return None, None, det_type, None

#     # ---------------- YOLO (v7 or v8) ----------------
#     if det_type == "yolo":
#         weight_path = det_cfg.get("raw_weights_path") or det_cfg.get("weights")
#         version = str(det_cfg.get("version", "")).strip().lower()

#         console.print(
#             f"Loading YOLO-like detector '{det_name}' from: {weight_path} "
#             f"(version={version or 'auto'})"
#         )

#         name_l = det_name.lower()
#         path_l = str(weight_path or "").lower()

#         is_v8 = (
#             version == "v8"
#             or "yolov8" in name_l
#             or "yolov8" in path_l
#         )

#         if is_v8:
#             # Lazy import YOLOv8
#             from frameworks.yolov8.frame_inference import (
#                 load_yolov8_model,
#                 detect_frame as detect_frame_yolov8,
#             )

#             model = load_yolov8_model(weight_path, device=device)

#             def runner(frame, model_, classes_, device_, conf_thresh_):
#                 # classes_ ignored for YOLO
#                 _, preds = detect_frame_yolov8(
#                     frame,
#                     model_,
#                     device=device_,
#                     conf_thresh=conf_thresh_,
#                 )
#                 return preds

#             return model, None, "yolov8", runner

#         else:
#             # Lazy import YOLOv7
#             from frameworks.yolov7.frame_inference import (
#                 load_yolov7_model,
#                 detect_frame as detect_frame_yolov7,
#             )

#             model = load_yolov7_model(weight_path, device=device)

#             def runner(frame, model_, classes_, device_, conf_thresh_):
#                 # classes_ ignored for YOLO
#                 _, preds = detect_frame_yolov7(
#                     frame,
#                     model_,
#                     device=device_,
#                     conf_thresh=conf_thresh_,
#                 )
#                 return preds

#             return model, None, "yolov7", runner

#     # ---------------- Faster R-CNN ----------------
#     if det_type in ("frcnn", "fasterrcnn", "faster-rcnn"):
#         from frameworks.fasterrcnn.frame_inference import (
#             load_fasterrcnn_model,
#             run_fasterrcnn_on_frame,
#         )

#         weight_path = det_cfg.get("raw_weights_path") or det_cfg.get("weights")
#         console.print(f"Loading FasterRCNN-like detector '{det_name}' from: {weight_path}")
#         model, classes = load_fasterrcnn_model(weight_path, device=device)

#         def runner(frame, model_, classes_, device_, conf_thresh_):
#             _, preds = run_fasterrcnn_on_frame(
#                 frame,
#                 model_,
#                 classes_,
#                 device=device_,
#                 conf_thresh=conf_thresh_,
#             )
#             return preds

#         return model, classes, "frcnn", runner

#     # ---------------- Unsupported ----------------
#     raise ValueError(
#         f"Detector '{det_name}' has unsupported type '{det_type}'. "
#         f"Supported types: 'yolo', 'frcnn'"
#     )


# # Main evaluation ########################################################
# def main():
#     global CLASS_NAMES, NAME2ID, IOU_THRESH

#     parser = argparse.ArgumentParser(description="Two-model detection evaluator with voting.")
#     parser.add_argument(
#         "-c",
#         "--config",
#         default=DEFAULT_CONFIG_PATH,
#         help=f"Path to config.yaml (default: {DEFAULT_CONFIG_PATH})",
#     )
#     parser.add_argument(
#         "--img-dir",
#         help=f"Directory with test images",
#     )
#     parser.add_argument(
#         "--lbl-dir",
#         help=f"Directory with YOLO-format labels",
#     )
#     args = parser.parse_args()

#     console.print(f"Using config: [bold]{args.config}[/bold]")
#     cfg = load_config_yaml(args.config)

#     # Required config keys
#     CLASS_NAMES = list(cfg.get("classes") or [])
#     if not CLASS_NAMES:
#         raise ValueError("config.yaml is missing required key: classes")

#     NAME2ID = {name: i for i, name in enumerate(CLASS_NAMES)}

#     det_cfg = cfg.get("detectors") or {}
#     voter_cfg = cfg.get("voter") or {}
#     f1_scores = cfg.get("f1_scores") or {}
#     device_cfg = cfg.get("device")

#     if len(det_cfg) < 2:
#         raise ValueError(
#             "config.detectors must define at least two detectors. "
#             "The first two (in order) are used as model A and model B."
#         )

#     # Choose model A/B from FIRST TWO detectors in config (order matters)
#     detector_names = list(det_cfg.keys())
#     model_a_name, model_b_name = detector_names[0], detector_names[1]
#     console.print(
#         f"Model A: [bold]{model_a_name}[/bold], "
#         f"Model B: [bold]{model_b_name}[/bold] (from first two config.detectors)"
#     )

#     # Device (default CUDA if available)
#     DEVICE = resolve_device(device_cfg)
#     console.print(f"Using device: [bold]{DEVICE}[/bold]")
#     is_cuda = str(DEVICE).lower().startswith("cuda")

#     # Voter params
#     voter_params = VoterParams(
#         conf_thresh=_as_float(voter_cfg.get("conf_thresh"), 0.5),
#         solo_strong=_as_float(voter_cfg.get("solo_strong"), 0.95),
#         iou_thresh=_as_float(voter_cfg.get("iou_thresh"), 0.4),
#         f1_margin=_as_float(voter_cfg.get("f1_margin"), 0.05),
#         gamma=_as_float(voter_cfg.get("gamma"), 1.5),
#         fuse_coords=_as_bool(voter_cfg.get("fuse_coords"), True),
#         near_tie_conf=_as_float(voter_cfg.get("near_tie_conf"), 0.95),
#         use_f1=_as_bool(voter_cfg.get("use_f1"), True),
#     )

#     # Evaluator IoU threshold: keep original 0.5 unless config provides eval_iou_thresh
#     IOU_THRESH = _as_float(cfg.get("eval_iou_thresh"), DEFAULT_EVAL_IOU_THRESH)

#     # Sanity checks and config mismatches
#     mismatches = []

#     # f1_scores expected structure: {class_name: {detector_name: f1}}
#     if f1_scores:
#         for c in CLASS_NAMES:
#             if c not in f1_scores:
#                 mismatches.append(f"config.f1_scores missing class '{c}'")
#             else:
#                 per = f1_scores.get(c, {})
#                 if model_a_name not in per:
#                     mismatches.append(f"config.f1_scores['{c}'] missing '{model_a_name}'")
#                 if model_b_name not in per:
#                     mismatches.append(f"config.f1_scores['{c}'] missing '{model_b_name}'")
#     else:
#         mismatches.append("config.f1_scores is empty; voter will behave as if all F1=0")

#     if mismatches:
#         console.print("[yellow]Config mismatches detected:[/yellow]")
#         for m in mismatches:
#             console.print(f"  - {m}")

#     # Per-detector runtime conf thresholds ################################
#     det_cfg_a = det_cfg.get(model_a_name) or {}
#     det_cfg_b = det_cfg.get(model_b_name) or {}

#     CONF_THRESH_A = _as_float(det_cfg_a.get("conf_threshold"), 0.6)
#     CONF_THRESH_B = _as_float(det_cfg_b.get("conf_threshold"), 0.9)

#     ENABLED_A = _as_bool(det_cfg_a.get("enabled", True), True)
#     ENABLED_B = _as_bool(det_cfg_b.get("enabled", True), True)

#     # Model loading ######################################################
#     model_a = model_b = None
#     classes_a = classes_b = None
#     runner_a = runner_b = None
#     type_a = type_b = ""

#     if ENABLED_A:
#         model_a, classes_a, type_a, runner_a = load_detector_model(model_a_name, det_cfg_a, DEVICE)
#     else:
#         console.print(f"[yellow]{model_a_name} disabled in config[/yellow]")
#         type_a = str(det_cfg_a.get("type", "")).strip().lower()

#     if ENABLED_B:
#         model_b, classes_b, type_b, runner_b = load_detector_model(model_b_name, det_cfg_b, DEVICE)
#     else:
#         console.print(f"[yellow]{model_b_name} disabled in config[/yellow]")
#         type_b = str(det_cfg_b.get("type", "")).strip().lower()

#     # Voter
#     voter = TwoModelVoter(
#         model_a=model_a_name,
#         model_b=model_b_name,
#         f1_scores=f1_scores,
#         params=voter_params,
#     )

#     # Dataset loop #######################################################
#     img_files = []
#     for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"):
#         img_files.extend(glob(os.path.join(args.img_dir, ext)))
#     img_files = sorted(img_files)
#     console.print(f"Found {len(img_files)} test images in '{args.img_dir}'.")

#     # Accumulators for micro P/R
#     a_tp = a_fp = a_fn = 0
#     b_tp = b_fp = b_fn = 0
#     voter_tp = voter_fp = voter_fn = 0

#     # For mAP accumulation
#     a_all_preds, b_all_preds, voter_all_preds, gt_all = [], [], [], []

#     # For FPS stats
#     a_infer_time = 0.0
#     b_infer_time = 0.0
#     voter_time = 0.0
#     a_frame_count = 0
#     b_frame_count = 0
#     num_eval_frames = 0  # frames that actually went through evaluation / voting

#     for img_path in img_files:
#         fname = os.path.basename(img_path)
#         lbl_path = os.path.join(args.lbl_dir, os.path.splitext(fname)[0] + ".txt")
#         frame = cv2.imread(img_path)

#         if frame is None:
#             console.print(f"[WARN] Could not read {fname}, skipping.")
#             continue

#         num_eval_frames += 1

#         # Load GT labels (YOLO format -> pixel boxes, names)
#         gt_boxes = load_yolo_labels(lbl_path, frame.shape, CLASS_NAMES)
#         gt_all.append(gt_boxes)

#         # Run detectors
#         preds_a = []
#         preds_b = []

#         if ENABLED_A and model_a is not None and runner_a is not None:
#             if is_cuda:
#                 torch.cuda.synchronize()
#             t0 = time.perf_counter()
#             preds_a = runner_a(frame.copy(), model_a, classes_a, DEVICE, CONF_THRESH_A)
#             if is_cuda:
#                 torch.cuda.synchronize()
#             a_infer_time += time.perf_counter() - t0
#             a_frame_count += 1

#         if ENABLED_B and model_b is not None and runner_b is not None:
#             if is_cuda:
#                 torch.cuda.synchronize()
#             t0 = time.perf_counter()
#             preds_b = runner_b(frame.copy(), model_b, classes_b, DEVICE, CONF_THRESH_B)
#             if is_cuda:
#                 torch.cuda.synchronize()
#             b_infer_time += time.perf_counter() - t0
#             b_frame_count += 1

#         # Voter expects DetectionModel inputs
#         dets_a = preds_dicts_to_models(preds_a, source=model_a_name)
#         dets_b = preds_dicts_to_models(preds_b, source=model_b_name)

#         if is_cuda:
#             torch.cuda.synchronize()
#         t0 = time.perf_counter()
#         final_dets, _candidates = voter.vote(dets_a, dets_b)
#         if is_cuda:
#             torch.cuda.synchronize()
#         voter_time += time.perf_counter() - t0

#         final_preds = models_to_preds_dicts(final_dets)

#         # Save for mAP
#         a_all_preds.append(preds_a)
#         b_all_preds.append(preds_b)
#         voter_all_preds.append(final_preds)

#         # Per-image P/R (uses IOU_THRESH)
#         a_prec, a_rec, tp_a, fp_a, fn_a = compute_metrics(preds_a, gt_boxes, IOU_THRESH)
#         b_prec, b_rec, tp_b, fp_b, fn_b = compute_metrics(preds_b, gt_boxes, IOU_THRESH)
#         voter_prec, voter_rec, tp_v, fp_v, fn_v = compute_metrics(final_preds, gt_boxes, IOU_THRESH)

#         # Accumulate micro totals
#         a_tp += tp_a
#         a_fp += fp_a
#         a_fn += fn_a
#         b_tp += tp_b
#         b_fp += fp_b
#         b_fn += fn_b
#         voter_tp += tp_v
#         voter_fp += fp_v
#         voter_fn += fn_v

#         # Optional per-image logging (commented out)
#         console.print(f"\n📷 {fname}")
#         # console.print(f"  {model_a_name:12s} -> Precision: {a_prec:.3f}, Recall: {a_rec:.3f}")
#         # console.print(f"  {model_b_name:12s} -> Precision: {b_prec:.3f}, Recall: {b_rec:.3f}")
#         # console.print(f"  Voter        -> Precision: {voter_prec:.3f}, Recall: {voter_rec:.3f}")

#     # Final stats #########################################################
#     console.print("\n====================== FINAL DATASET STATS ======================")
#     summarize(a_tp, a_fp, a_fn, model_a_name)
#     summarize(b_tp, b_fp, b_fn, model_b_name)
#     summarize(voter_tp, voter_fp, voter_fn, "Voter")

#     console.print("\n====================== FPS METRICS ======================")
#     print_fps(model_a_name, a_infer_time, a_frame_count)
#     print_fps(model_b_name, b_infer_time, b_frame_count)

#     # Voter FPS should reflect the full pipeline: model A + model B + vote
#     voter_total_time = a_infer_time + b_infer_time + voter_time
#     print_fps("Voter (A+B+vote)", voter_total_time, num_eval_frames)

#     console.print("\n====================== mAP METRICS ======================")
#     mAP50_a, ap_cls_a = compute_map_at_iou(a_all_preds, gt_all, CLASS_NAMES, IOU_THRESH)
#     mAP50_b, ap_cls_b = compute_map_at_iou(b_all_preds, gt_all, CLASS_NAMES, IOU_THRESH)
#     mAP50_voter, ap_cls_voter = compute_map_at_iou(voter_all_preds, gt_all, CLASS_NAMES, IOU_THRESH)

#     mAP_coco_a = compute_coco_map(a_all_preds, gt_all, CLASS_NAMES, COCO_IOU_THRESHOLDS)
#     mAP_coco_b = compute_coco_map(b_all_preds, gt_all, CLASS_NAMES, COCO_IOU_THRESHOLDS)
#     mAP_coco_voter = compute_coco_map(voter_all_preds, gt_all, CLASS_NAMES, COCO_IOU_THRESHOLDS)

#     console.print(f"{model_a_name:12s} mAP@0.5: {mAP50_a:.3f} | mAP@[.5:.95]: {mAP_coco_a:.3f}")
#     console.print(f"{model_b_name:12s} mAP@0.5: {mAP50_b:.3f} | mAP@[.5:.95]: {mAP_coco_b:.3f}")
#     console.print(f"Voter        mAP@0.5: {mAP50_voter:.3f} | mAP@[.5:.95]: {mAP_coco_voter:.3f}")

#     console.print(f"\nPer-class AP@0.5 ({model_a_name}):")
#     for i, ap in enumerate(ap_cls_a):
#         console.print(f"  {CLASS_NAMES[i]:25s}: {ap:.3f}")

#     console.print(f"Per-class AP@0.5 ({model_b_name}):")
#     for i, ap in enumerate(ap_cls_b):
#         console.print(f"  {CLASS_NAMES[i]:25s}: {ap:.3f}")

#     console.print("Per-class AP@0.5 (Voter):")
#     for i, ap in enumerate(ap_cls_voter):
#         console.print(f"  {CLASS_NAMES[i]:25s}: {ap:.3f}")

#     console.print(f"\n====================== PER-CLASS F1 (IOU={IOU_THRESH:.2f}) ======================")

#     a_pc = compute_per_class_f1(a_all_preds, gt_all, CLASS_NAMES, IOU_THRESH)
#     b_pc = compute_per_class_f1(b_all_preds, gt_all, CLASS_NAMES, IOU_THRESH)
#     voter_pc = compute_per_class_f1(voter_all_preds, gt_all, CLASS_NAMES, IOU_THRESH)

#     console.print(f"\n{model_a_name} per-class:")
#     for c in CLASS_NAMES:
#         r = a_pc[c]
#         console.print(
#             f"  {c:25s}  F1={r['f1']:.3f}  P={r['precision']:.3f}  R={r['recall']:.3f}  "
#             f"TP={r['TP']} FP={r['FP']} FN={r['FN']}"
#         )

#     console.print(f"\n{model_b_name} per-class:")
#     for c in CLASS_NAMES:
#         r = b_pc[c]
#         console.print(
#             f"  {c:25s}  F1={r['f1']:.3f}  P={r['precision']:.3f}  R={r['recall']:.3f}  "
#             f"TP={r['TP']} FP={r['FP']} FN={r['FN']}"
#         )

#     console.print("\nVoter per-class:")
#     for c in CLASS_NAMES:
#         r = voter_pc[c]
#         console.print(
#             f"  {c:25s}  F1={r['f1']:.3f}  P={r['precision']:.3f}  R={r['recall']:.3f}  "
#             f"TP={r['TP']} FP={r['FP']} FN={r['FN']}"
#         )


# if __name__ == "__main__":
#     main()