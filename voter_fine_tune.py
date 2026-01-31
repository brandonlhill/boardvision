#!/usr/bin/env python3
"""
Voter hyperparameter tuner focused on very high precision:
- Optimizes recall at a target precision (default: P >= 0.97).
- Runs YOLOv7 and Faster R-CNN once (low thresholds), caches detections.
- Re-runs the voter merge many times with different hyperparameters.
- Keeps per-class F1 priors from config.json FIXED and IMMUTABLE.
- Explores a wide space: gamma, voter_conf_thresh, iou_thresh, solo_strong, f1_margin,
  and an optional post-voter NMS (nms_iou) to de-duplicate finals.

Outputs:
- mAP@0.5, COCO mAP, best-F1 operating point, recall@P>=target, precision@R>=target,
  and a leaderboard of top configs.
"""

"""
Detections are first produced by two independent models (YOLOv7 and Faster R-CNN) using very low confidence thresholds so that most candidate objects are surfaced. For each class, fixed reliability priors (per-class F1 scores) are loaded from a configuration file; these priors are treated as immutable and are used as weights during ensembling. All predictions are kept in memory so the detectors do not have to be rerun during hyperparameter tuning.

Agreement between the two models is then sought. For each YOLO box, the best-overlapping Faster R-CNN box of the same class is identified; if the Intersection-over-Union exceeds an agreement threshold (iou_thresh), the pair is considered a match. A reliability-weighted score is computed for each box as 
(conf)γ×F1prior
(conf)
γ
×F1
prior
	​

, where gamma controls how strongly high confidences are emphasized. The higher-scoring box is selected as the winner, or the two boxes are fused into a single box by score-weighted averaging when coordinate fusion is enabled. The chosen box is marked as a FINAL detection.

Unmatched (solo) boxes are handled by class-aware rules designed to favor precision. A solo detection is accepted outright when its confidence exceeds a stringent auto-accept threshold (solo_strong). Otherwise, the box is accepted when the producing model’s class prior is higher than its counterpart’s and the confidence exceeds a base gate (voter_conf_thresh). An additional “close-prior” exception is permitted: if the two class priors differ by at most f1_margin and the box confidence is very high (≥ 0.95), acceptance is granted. Boxes that do not satisfy any rule are rejected.

After selection, optional non-maximum suppression may be applied per class to the FINAL set (nms_iou). This step is used to remove duplicate finals that overlap too strongly, which typically reduces false positives and improves precision at the top of the precision–recall curve. Throughout, class labels are canonicalized so that minor naming differences do not affect matching.

Performance is evaluated with standard detection metrics. Dataset-wide precision–recall curves are built by greedily matching predictions to ground truth at a fixed IoU (e.g., 0.5). Average precision is computed using the COCO convention (precision envelope and 101-point interpolation), and mAP is reported at IoU = 0.5 as well as averaged over 0.50:0.95. Operating points of interest are extracted, including the best-F1 threshold, recall at a target precision, and precision at a target recall.

Hyperparameters are tuned without rerunning the base detectors. A broad exploration is conducted over gamma, voter_conf_thresh, iou_thresh, solo_strong, f1_margin, and optional nms_iou. Deterministic corner seeds and heavy random sampling are used to cover the space, after which multi-start coordinate ascent is applied (log-space for positive multiplicative parameters and linear space otherwise). The objective can be configured; when high-precision operation is required, recall at or above a specified precision (e.g., P ≥ 0.97) is maximized. The best configuration and a ranked leaderboard are then reported.

In summary, an explainable ensemble is formed in which agreement is rewarded via reliability-weighted scoring or fusion, solos are filtered by confidence and class priors, and duplicates are optionally suppressed. Class priors remain fixed, confidences are modulated by gamma, acceptance of solos is governed by voter_conf_thresh, and strictness of agreement and exceptions is controlled by iou_thresh, solo_strong, and f1_margin. This design allows precision and recall to be traded off predictably while optimizing for the desired operating regime.

"""


import os
import cv2
import time
import json
import math
import random
import numpy as np
from glob import glob
from typing import List, Dict
from rich.console import Console
from rich.table import Table
import torch

# --- Project imports (adjust if needed) ---
from yolov7.frame_inference import load_yolov7_model, detect_frame
from fasterrcnn.frame_inference import load_fasterrcnn_model, run_fasterrcnn_on_frame
from voter import voter_merge, load_f1_config, iou

console = Console()

# ===========================
# Paths / Config
# ===========================
IMG_DIR = "/home/brandon/Desktop/motherboard_model_training/Datasets/motherboard_yolov7/test/images"
LBL_DIR = "/home/brandon/Desktop/motherboard_model_training/Datasets/motherboard_yolov7/test/labels"
YOLO_WEIGHTS = "./weights/yolov7.pt"
FRCNN_WEIGHTS = "./weights/fasterrcnn.pth"
F1_CONFIG = "./config.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Evaluation knobs
IOU_THRESH = 0.5
COCO_IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)

# Run base models ONCE with low thresholds (recall-complete cache)
BASE_CONF_THRESH_YOLO = 0.001
BASE_CONF_THRESH_FRCNN = 0.001

# Class list (must match your dataset)
CLASS_NAMES = [
    'CPU_FAN_NO_Screws',
    'CPU_FAN_Screw_loose',
    'CPU_FAN_Screws',
    'CPU_fan',
    'CPU_fan_port',
    'CPU_fan_port_detached',
    'Incorrect_Screws',
    'Loose_Screws',
    'No_Screws',
    'Scratch',
    'Screws'
]
NUM_CLASSES = len(CLASS_NAMES)
NAME2ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# ===========================
# Multi-objective options (YOUR CHOICES)
# ===========================
OPTIMIZE_MODE = "recall_at_precision"   # "composite" | "map" | "f1" | "recall_at_precision" | "precision_at_recall" | "constrained"

# Composite weights (only used if OPTIMIZE_MODE == "composite")
W_MAP_COCO = 1.0
W_F1_MAX   = 0.3
W_R_AT_P   = 0.5
W_P_AT_R   = 0.0

# PR tradeoff targets
PRECISION_TARGET = 0.97     # optimize recall at this precision
RECALL_TARGET    = 0.90     # (used only if OPTIMIZE_MODE == "precision_at_recall")

# Constraints (if OPTIMIZE_MODE == "constrained")
MIN_RECALL_AT_P  = 0.95
MIN_PREC_AT_R    = 0.95

# Search budget (wider exploration)
WARM_TRIALS     = 300     # heavy random + grid warmup
COORD_ITERS     = 200     # deeper local refinement
TOPK_FOR_LOCAL  = 10      # number of top warm seeds to refine
LOCAL_RESTARTS  = 8       # additional random restarts around best

# ===========================
# Utilities
# ===========================
def set_seeds(seed=123):
    """Set RNG seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def norm_label(s: str) -> str:
    """Normalize label strings to a canonical form (lowercase, underscores)."""
    return s.strip().lower().replace(" ", "_")

CANON = {norm_label(k): k for k in CLASS_NAMES}
def canon_label(s: str) -> str:
    """Map a possibly inconsistent label string to the canonical CLASS_NAMES form."""
    return CANON.get(norm_label(s), s)

def list_images(img_dir: str) -> List[str]:
    """Return a sorted list of image file paths under img_dir with common extensions."""
    img_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"):
        img_files.extend(glob(os.path.join(img_dir, ext)))
    return sorted(img_files)

def load_yolo_labels(lbl_path: str, img_shape, class_names: List[str]):
    """Parse YOLO .txt labels and convert to absolute pixel [x1,y1,x2,y2] + class name."""
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
            x1 = (cx - bw/2) * w
            y1 = (cy - bh/2) * h
            x2 = (cx + bw/2) * w
            y2 = (cy + bh/2) * h
            label = canon_label(class_names[cls])
            boxes.append({'bbox': [float(x1), float(y1), float(x2), float(y2)], 'label': label, 'conf': 1.0})
    return boxes

# ===========================
# PR/AP/mAP (academically standard)
# ===========================
def build_pr_data(all_preds, all_gts, class_names, iou_thresh):
    """
    Convert predictions & GT to dataset-wide PR arrays using greedy matching
    per class and per image. Predictions are matched in descending confidence.

    Returns:
        tp_flags (np.int32), confs (np.float32), pred_cls (np.int32), n_gt_per_class (np.int32)
    """
    tp_flags, confs, pred_cls = [], [], []
    n_gt_per_class = np.zeros(len(class_names), dtype=int)

    for preds, gts in zip(all_preds, all_gts):
        for g in gts:
            n_gt_per_class[NAME2ID[g['label']]] += 1

        gt_by_class = {c: [] for c in class_names}
        gt_used_by_class = {c: [] for c in class_names}
        for g in gts:
            gt_by_class[g['label']].append(g['bbox'])
            gt_used_by_class[g['label']].append(False)

        preds_sorted = sorted(preds, key=lambda d: float(d.get('conf', 0.0)), reverse=True)
        for p in preds_sorted:
            c = canon_label(p['label'])
            conf = float(p.get('conf', 0.0))
            pred_cls.append(NAME2ID.get(c, -1))
            confs.append(conf)

            if c not in gt_by_class or len(gt_by_class[c]) == 0:
                tp_flags.append(0)
                continue

            best_iou, best_idx = 0.0, -1
            for i, (gbox, used) in enumerate(zip(gt_by_class[c], gt_used_by_class[c])):
                if used:
                    continue
                ov = iou(p['bbox'], gbox)
                if ov > best_iou:
                    best_iou, best_idx = ov, i

            if best_idx >= 0 and best_iou >= iou_thresh:
                tp_flags.append(1)
                gt_used_by_class[c][best_idx] = True
            else:
                tp_flags.append(0)

    return (np.array(tp_flags, np.int32),
            np.array(confs, np.float32),
            np.array(pred_cls, np.int32),
            n_gt_per_class)

def ap_from_pr(tp_flags, confs, pred_cls, n_gt_per_class, num_classes):
    """
    Compute per-class AP with precision envelope + 101-point interpolation (COCO-style).
    Classes with no GT remain NaN and are ignored in the mean via nanmean.
    """
    ap = np.full(num_classes, np.nan, dtype=np.float32)
    p_last = np.zeros(num_classes, dtype=np.float32)
    r_last = np.zeros(num_classes, dtype=np.float32)

    order = np.argsort(-confs)
    tp_flags = tp_flags[order]
    pred_cls = pred_cls[order]

    for c in range(num_classes):
        cls_mask = (pred_cls == c)
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
    """Compute mAP at a single IoU threshold."""
    tp_flags, confs, pred_cls, n_gt = build_pr_data(all_preds, all_gts, class_names, iou_thresh)
    ap_per_cls, mAP, _, _ = ap_from_pr(tp_flags, confs, pred_cls, n_gt, len(class_names))
    return mAP, ap_per_cls

def compute_coco_map(all_preds, all_gts, class_names, iou_thresholds):
    """Compute COCO mAP averaged over multiple IoU thresholds (0.50:0.95)."""
    aps = []
    for thr in iou_thresholds:
        m, _ = compute_map_at_iou(all_preds, all_gts, class_names, thr)
        aps.append(m)
    return float(np.mean(aps))

def global_pr_operating_points(all_preds, all_gts, class_names, iou_thresh,
                               precision_target=None, recall_target=None):
    """
    Build global (micro) precision/recall arrays and extract:
      - best-F1 operating point and its score threshold
      - recall at a precision target (>=)
      - precision at a recall target (>=)
    """
    tp_flags, confs, pred_cls, n_gt = build_pr_data(all_preds, all_gts, class_names, iou_thresh)
    n_gt_total = int(np.sum(n_gt))

    order = np.argsort(-confs)
    tp = tp_flags[order].astype(np.int32)
    fp = 1 - tp
    scores_sorted = confs[order]

    tpc = np.cumsum(tp)
    fpc = np.cumsum(fp)

    recall = tpc / (n_gt_total + 1e-9)
    precision = tpc / (tpc + fpc + 1e-9)

    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    idx_best = int(np.argmax(f1)) if f1.size else -1

    best = dict(
        best_f1=float(f1[idx_best]) if idx_best >= 0 else 0.0,
        best_thr=float(scores_sorted[idx_best]) if idx_best >= 0 else 1.0,
        best_p=float(precision[idx_best]) if idx_best >= 0 else 0.0,
        best_r=float(recall[idx_best]) if idx_best >= 0 else 0.0,
    )

    rec_at_p, thr_at_p = None, None
    if precision_target is not None and precision.size:
        mask = precision >= precision_target
        if np.any(mask):
            idx = int(np.argmax(recall * mask))
            rec_at_p = float(recall[idx])
            thr_at_p = float(scores_sorted[idx])

    prec_at_r, thr_at_r = None, None
    if recall_target is not None and recall.size:
        mask = recall >= recall_target
        if np.any(mask):
            idx = int(np.argmax(precision * mask))
            prec_at_r = float(precision[idx])
            thr_at_r = float(scores_sorted[idx])

    return {
        "precision": precision, "recall": recall, "scores": scores_sorted,
        "best": best,
        "recall_at_p_target": rec_at_p, "thr_at_p_target": thr_at_p,
        "precision_at_r_target": prec_at_r, "thr_at_r_target": thr_at_r
    }

# ===========================
# Post-voter NMS (optional, per class)
# ===========================
def nms_per_class(dets, iou_thr=0.6):
    """
    Greedy Non-Maximum Suppression per class on voter FINAL outputs.
    Keeps higher-confidence boxes, suppresses overlapping lower ones.
    """
    if dets is None or len(dets) == 0:
        return dets
    by_cls: Dict[str, list] = {}
    for d in dets:
        by_cls.setdefault(d['label'], []).append(d)
    kept = []
    for cls, arr in by_cls.items():
        arr = sorted(arr, key=lambda x: float(x.get('conf', 0.0)), reverse=True)
        used = [False] * len(arr)
        for i in range(len(arr)):
            if used[i]:
                continue
            keep = arr[i]
            kept.append(keep)
            bi = keep['bbox']
            for j in range(i + 1, len(arr)):
                if used[j]:
                    continue
                bj = arr[j]['bbox']
                if iou(bi, bj) >= iou_thr:
                    used[j] = True
    return kept

# ===========================
# Cache base detections (run once)
# ===========================
def cache_base_detections(img_files, yolo_model, frcnn_model, frcnn_classes):
    """
    Run YOLOv7 and Faster R-CNN once with very low thresholds and cache detections.
    Also load YOLO-format ground truth and return in parallel lists.
    """
    frames, gts = [], []
    yolo_cache, frcnn_cache = [], []

    for img_path in img_files:
        fname = os.path.basename(img_path)
        frame = cv2.imread(img_path)
        if frame is None:
            console.print("[WARN] Could not read {}, skipping.".format(fname))
            continue

        lbl_path = os.path.join(LBL_DIR, os.path.splitext(fname)[0] + ".txt")
        gt_boxes = load_yolo_labels(lbl_path, frame.shape, CLASS_NAMES)
        gts.append(gt_boxes)
        frames.append(frame)

        # Base detectors ONCE (low thresholds)
        _, yolo_preds = detect_frame(frame, yolo_model, device=DEVICE, conf_thresh=BASE_CONF_THRESH_YOLO)
        _, frcnn_preds = run_fasterrcnn_on_frame(frame, frcnn_model, frcnn_classes, device=DEVICE, conf_thresh=BASE_CONF_THRESH_FRCNN)

        for d in yolo_preds:  d['label'] = canon_label(d['label'])
        for d in frcnn_preds: d['label'] = canon_label(d['label'])

        yolo_cache.append(yolo_preds)
        frcnn_cache.append(frcnn_preds)

    return frames, gts, yolo_cache, frcnn_cache

# ===========================
# Evaluate voter (F1 priors fixed)
# ===========================
def evaluate_voter(gts, yolo_cache, frcnn_cache,
                   f1_config,
                   gamma=1.5,
                   voter_conf_thresh=0.05,
                   iou_thresh=0.40,
                   solo_strong=0.95,
                   f1_margin=0.05,
                   fuse_coords=True,
                   nms_iou=None):
    """
    Run voter_merge on cached detections with given hyperparameters and compute metrics.

    Args:
        gamma: float
            Rank sharpness inside voter: score = (conf ** gamma) * F1_prior.
        voter_conf_thresh: float
            Solo acceptance gate for non-agreement boxes.
        iou_thresh: float
            Min IoU to treat YOLO/FRCNN boxes of the same class as an agreement pair.
        solo_strong: float
            Solo detection auto-wins if conf >= solo_strong.
        f1_margin: float
            If |F1_yolo - F1_frcnn| <= f1_margin and conf >= 0.95, accept (exception).
        fuse_coords: bool
            If True, fuse agreed boxes by score-weighting; else choose winner’s box.
        nms_iou: Optional[float]
            If provided, apply per-class NMS at this IoU on FINAL outputs.

    Returns:
        dict of metrics for scoring.
    """
    voter_all = []
    for yolo_preds, frcnn_preds in zip(yolo_cache, frcnn_cache):
        finals, _ = voter_merge(
            yolo_preds, frcnn_preds, f1_config,
            conf_thresh=voter_conf_thresh,
            solo_strong=solo_strong,
            iou_thresh=iou_thresh,
            f1_margin=f1_margin,
            gamma=gamma,
            fuse_coords=fuse_coords
        )
        for d in finals:
            d['label'] = canon_label(d['label'])
        if nms_iou is not None:
            finals = nms_per_class(finals, iou_thr=float(nms_iou))
        voter_all.append(finals)

    mAP50, _ = compute_map_at_iou(voter_all, gts, CLASS_NAMES, IOU_THRESH)
    mAP_coco = compute_coco_map(voter_all, gts, CLASS_NAMES, COCO_IOU_THRESHOLDS)

    pr = global_pr_operating_points(
        voter_all, gts, CLASS_NAMES, IOU_THRESH,
        precision_target=PRECISION_TARGET, recall_target=RECALL_TARGET
    )

    return {
        "mAP50": mAP50,
        "mAPcoco": mAP_coco,
        "bestF1": pr["best"]["best_f1"],
        "bestF1_thr": pr["best"]["best_thr"],
        "bestF1_P": pr["best"]["best_p"],
        "bestF1_R": pr["best"]["best_r"],
        "recall_at_Pt": pr["recall_at_p_target"],
        "thr_at_Pt": pr["thr_at_p_target"],
        "precision_at_Rt": pr["precision_at_r_target"],
        "thr_at_Rt": pr["thr_at_r_target"],
    }

# ===========================
# Search space + optimizer
# ===========================
def sample_params():
    """
    Randomly sample a hyperparameter set with wide exploration:
      - gamma: log-uniform [0.2, 10.0]
      - voter_conf_thresh: 70% log-uniform [1e-5, 0.3], 30% uniform [0.3, 0.9]
      - iou_thresh: uniform [0.35, 0.70]
      - solo_strong: uniform [0.93, 0.995]
      - f1_margin: uniform [0.00, 0.06]
      - nms_iou: 50% chance of None (off), else uniform [0.40, 0.70]
    """
    gamma = float(np.exp(np.random.uniform(np.log(0.2), np.log(10.0))))
    if np.random.rand() < 0.70:
        voter_conf = float(10 ** np.random.uniform(np.log10(1e-5), np.log10(0.3)))
    else:
        voter_conf = float(np.random.uniform(0.3, 0.9))
    iou_t = float(np.random.uniform(0.35, 0.70))
    solo_s = float(np.random.uniform(0.93, 0.995))
    f1_m  = float(np.random.uniform(0.00, 0.06))
    if np.random.rand() < 0.5:
        nms_iou = None
    else:
        nms_iou = float(np.random.uniform(0.40, 0.70))

    return dict(
        gamma=gamma,
        voter_conf_thresh=voter_conf,
        iou_thresh=iou_t,
        solo_strong=solo_s,
        f1_margin=f1_m,
        fuse_coords=True,
        nms_iou=nms_iou
    )

def extreme_seeds():
    """
    Deterministic seeds to cover the corners; thinned to keep runtime reasonable.
    """
    gammas = [0.2, 1.0, 2.0, 4.0, 8.0, 10.0]
    confs  = [1e-5, 0.001, 0.05, 0.20, 0.40, 0.60, 0.90]
    ious   = [0.35, 0.45, 0.55, 0.70]
    solos  = [0.93, 0.96, 0.98, 0.995]
    margins= [0.00, 0.02, 0.05, 0.06]
    nmses  = [None, 0.50, 0.60]

    seeds = []
    for g in gammas:
        for c in confs:
            for it in (0.35, 0.55, 0.70):
                seeds.append(dict(
                    gamma=float(g),
                    voter_conf_thresh=float(c),
                    iou_thresh=float(it),
                    solo_strong=0.96,
                    f1_margin=0.02,
                    fuse_coords=True,
                    nms_iou=None
                ))
    for s in solos:
        for m in margins:
            for n in nmses:
                seeds.append(dict(
                    gamma=2.0, voter_conf_thresh=0.2,
                    iou_thresh=0.55, solo_strong=float(s),
                    f1_margin=float(m), fuse_coords=True,
                    nms_iou=n
                ))
    return seeds

def coordinate_search(score_fn, start_params, steps_log, bounds, iters=30):
    """
    Derivative-free coordinate ascent.
    - gamma, voter_conf_thresh in log-space (positive multiplicative).
    - iou_thresh, solo_strong, f1_margin, nms_iou in linear space.
    """
    def clamp(v, lo, hi): return max(lo, min(hi, v))

    p = start_params.copy()
    best = score_fn(p)
    improved = True

    keys_log = ('gamma', 'voter_conf_thresh')
    keys_lin = ('iou_thresh', 'solo_strong', 'f1_margin', 'nms_iou')

    for _ in range(iters):
        if not improved:
            for k in steps_log:
                steps_log[k] *= 0.5
        improved = False

        for key in list(keys_log) + list(keys_lin):
            base = p[key]
            lo, hi = bounds[key]
            tried = []

            if key in keys_log:
                log_base = math.log(max(base if base is not None else 1e-6, 1e-12))
                step = steps_log[key]
                for direction in (+1, -1):
                    cand = math.exp(log_base + direction * step)
                    cand = clamp(cand, lo, hi)
                    q = p.copy(); q[key] = cand
                    tried.append(score_fn(q))
            else:
                step = steps_log[key]
                for direction in (+1, -1):
                    cand = base
                    if cand is None:
                        cand = (lo + hi) / 2.0  # park at middle if starting None
                    cand = clamp(cand + direction * step, lo, hi)
                    q = p.copy(); q[key] = cand
                    tried.append(score_fn(q))

            candidates = tried + [best]
            candidates.sort(key=lambda r: r['score'], reverse=True)
            if candidates[0]['score'] > best['score'] + 1e-9:
                best = candidates[0]
                p = best['params'].copy()
                improved = True

        if max(steps_log.values()) < 1e-3:
            break

    return best

def make_scalar_objective(metrics):
    """
    Convert metrics to a single 'score' depending on OPTIMIZE_MODE.
    """
    m = metrics
    score = None
    reason = ""

    if OPTIMIZE_MODE == "map":
        score = m["mAPcoco"]; reason = "mAPcoco"

    elif OPTIMIZE_MODE == "f1":
        score = m["bestF1"]; reason = "bestF1"

    elif OPTIMIZE_MODE == "recall_at_precision":
        score = -1.0
        if m["recall_at_Pt"] is not None:
            score = m["recall_at_Pt"]
        reason = "recall@precision_target"

    elif OPTIMIZE_MODE == "precision_at_recall":
        score = -1.0
        if m["precision_at_Rt"] is not None:
            score = m["precision_at_Rt"]
        reason = "precision@recall_target"

    elif OPTIMIZE_MODE == "constrained":
        feas = True
        if MIN_RECALL_AT_P is not None:
            feas = feas and (m["recall_at_Pt"] is not None and m["recall_at_Pt"] >= MIN_RECALL_AT_P)
        if MIN_PREC_AT_R is not None:
            feas = feas and (m["precision_at_Rt"] is not None and m["precision_at_Rt"] >= MIN_PREC_AT_R)
        score = m["mAPcoco"] if feas else -1.0
        reason = "mAPcoco (constrained)"

    else:  # "composite"
        r_at_p = m["recall_at_Pt"] if m["recall_at_Pt"] is not None else 0.0
        p_at_r = m["precision_at_Rt"] if m["precision_at_Rt"] is not None else 0.0
        score = (W_MAP_COCO * m["mAPcoco"]
                 + W_F1_MAX   * m["bestF1"]
                 + W_R_AT_P   * r_at_p
                 + W_P_AT_R   * p_at_r)
        reason = "composite"

    out = dict(score=score, reason=reason)
    out.update(m)
    return out

def tune(gts, yolo_cache, frcnn_cache, f1_config, seed=123, warm_trials=WARM_TRIALS):
    """
    Full tuning loop:
      1) Evaluate deterministic extreme seeds + random samples.
      2) Pick top-K seeds and run coordinate ascent from each.
      3) Gaussian restarts around the best and refine again.
      4) Return best result and a leaderboard.
    """
    set_seeds(seed)

    cache = {}
    def key_of(params):
        return json.dumps({k: (None if v is None else round(float(v), 10)) for k, v in sorted(params.items())})

    def evaluate_params(p):
        k = key_of(p)
        if k in cache:
            return cache[k]
        metrics = evaluate_voter(gts, yolo_cache, frcnn_cache, f1_config, **p)
        result = make_scalar_objective(metrics)
        result['params'] = p.copy()
        cache[k] = result
        console.print(
            "trial score={:.4f}  mAP@0.5={:.4f}  COCO={:.4f}  F1={:.4f} (thr={:.4f}, P={:.3f}, R={:.3f})  "
            "R@P≥{:.2f}={}  P@R≥{:.2f}={}  params={}".format(
                result['score'], metrics['mAP50'], metrics['mAPcoco'],
                metrics['bestF1'], metrics['bestF1_thr'], metrics['bestF1_P'], metrics['bestF1_R'],
                PRECISION_TARGET, metrics['recall_at_Pt'],
                RECALL_TARGET, metrics['precision_at_Rt'],
                p
            )
        )
        return result

    # Warm seeds
    seeds = extreme_seeds()
    for _ in range(max(0, warm_trials - len(seeds))):
        seeds.append(sample_params())

    results = [evaluate_params(p) for p in seeds]

    # Top-K local refinement
    results_sorted = sorted(results, key=lambda r: r['score'], reverse=True)
    top_seeds = [r['params'] for r in results_sorted[:TOPK_FOR_LOCAL]]

    bounds = {
        'gamma': (0.2, 10.0),
        'voter_conf_thresh': (1e-5, 0.9),
        'iou_thresh': (0.35, 0.70),
        'solo_strong': (0.93, 0.995),
        'f1_margin': (0.0, 0.06),
        'nms_iou': (0.40, 0.70),  # when not None
    }
    steps_log = {
        'gamma': 0.8,              # log step
        'voter_conf_thresh': 1.0,  # log step
        'iou_thresh': 0.06,        # linear step
        'solo_strong': 0.01,       # linear step
        'f1_margin': 0.01,         # linear step
        'nms_iou': 0.06,           # linear step
    }

    best_overall = results_sorted[0]
    all_refined = []

    for si, seed_params in enumerate(top_seeds, 1):
        # Ensure nms_iou is not None for refinement moves; keep as-is if None
        sp = seed_params.copy()
        refined = coordinate_search(evaluate_params, sp, steps_log.copy(), bounds, iters=COORD_ITERS)
        all_refined.append(refined)
        if refined['score'] > best_overall['score']:
            best_overall = refined
        console.print("refine[{}/{}] seed={} -> score={:.4f} params={}".format(
            si, len(top_seeds), seed_params, refined['score'], refined['params'])
        )

    # Random restarts around current best
    for r_idx in range(LOCAL_RESTARTS):
        bp = best_overall['params']
        jitter = {
            'gamma': float(np.exp(np.random.normal(loc=np.log(max(bp['gamma'], 1e-6)), scale=0.25))),
            'voter_conf_thresh': float(np.exp(np.random.normal(
                loc=np.log(max(bp['voter_conf_thresh'], 1e-5)), scale=0.5))),
            'iou_thresh': float(np.random.normal(loc=bp['iou_thresh'], scale=0.05)),
            'solo_strong': float(np.random.normal(loc=bp['solo_strong'], scale=0.01)),
            'f1_margin': float(np.random.normal(loc=bp['f1_margin'], scale=0.01)),
            'fuse_coords': True,
            'nms_iou': bp['nms_iou'] if bp.get('nms_iou', None) is not None else float(np.random.uniform(0.40, 0.70)),
        }
        # clamp linear params
        for k in ('iou_thresh','solo_strong','f1_margin','nms_iou'):
            lo, hi = bounds[k]
            val = jitter[k]
            if val is not None:
                jitter[k] = min(max(val, lo), hi)
        refined = coordinate_search(evaluate_params, jitter, steps_log.copy(), bounds, iters=int(COORD_ITERS*0.6))
        all_refined.append(refined)
        if refined['score'] > best_overall['score']:
            best_overall = refined
        console.print("restart[{}/{}] -> score={:.4f} params={}".format(
            r_idx + 1, LOCAL_RESTARTS, refined['score'], refined['params'])
        )

    leaderboard = sorted(results + all_refined, key=lambda r: r['score'], reverse=True)
    return best_overall, leaderboard

# ===========================
# Pretty print leaderboard
# ===========================
def show_top(results, k=10):
    """
    Render the top-k results in a Rich table. Includes extra voter knobs and NMS.
    """
    rows = results[:k]
    table = Table(title="Top {} Voter Hyperparams (mode: {})".format(min(k, len(results)), OPTIMIZE_MODE))
    cols = ["Rank","Score","mAP@0.5","mAP@[.5:.95]","F1","F1_thr","P","R","R@P*","P@R*",
            "gamma","voter_conf","iou_thr","solo_strong","f1_margin","nms_iou"]
    for c in cols:
        table.add_column(c, justify="right")

    def fmt(val, spec=".4f"):
        try:
            return format(float(val), spec)
        except (TypeError, ValueError):
            return "None"

    for i, r in enumerate(rows, 1):
        p = r.get("params", {}) or {}
        recall_at_pt_str = "None" if r.get("recall_at_Pt")   is None else fmt(r["recall_at_Pt"], ".3f")
        prec_at_rt_str   = "None" if r.get("precision_at_Rt") is None else fmt(r["precision_at_Rt"], ".3f")
        nms_str = "off" if p.get("nms_iou", None) is None else fmt(p.get("nms_iou"), ".2f")

        table.add_row(
            str(i),
            fmt(r.get("score")),
            fmt(r.get("mAP50")),
            fmt(r.get("mAPcoco")),
            fmt(r.get("bestF1")),
            fmt(r.get("bestF1_thr")),
            fmt(r.get("bestF1_P"), ".3f"),
            fmt(r.get("bestF1_R"), ".3f"),
            recall_at_pt_str,
            prec_at_rt_str,
            fmt(p.get("gamma"), ".3f"),
            fmt(p.get("voter_conf_thresh")),
            fmt(p.get("iou_thresh"), ".2f"),
            fmt(p.get("solo_strong"), ".3f"),
            fmt(p.get("f1_margin"), ".3f"),
            nms_str,
        )
    console.print(table)

# ===========================
# Main
# ===========================
def main():
    """Entry point: load models, cache detections, run tuner, and print results."""
    console.print("[bold]Loading models...[/bold]")
    yolo_model = load_yolov7_model(YOLO_WEIGHTS, device=DEVICE)
    frcnn_model, frcnn_classes = load_fasterrcnn_model(FRCNN_WEIGHTS, device=DEVICE)
    f1_config = load_f1_config(F1_CONFIG)  # F1 priors are FIXED and never changed

    console.print("[bold]Indexing images...[/bold]")
    img_files = list_images(IMG_DIR)
    console.print("Found {} images.".format(len(img_files)))

    console.print("[bold]Caching base detections (one pass)...[/bold]")
    t0 = time.time()
    frames, gts, yolo_cache, frcnn_cache = cache_base_detections(img_files, yolo_model, frcnn_model, frcnn_classes)
    console.print("Cached {} images in {:.1f}s".format(len(frames), time.time()-t0))

    # Optional: IoU self-check
    if yolo_cache and yolo_cache[0]:
        b = yolo_cache[0][0]['bbox']
        console.print("IoU self-test (should be 1.0): {:.3f}".format(iou(b, b)))

    console.print("[bold]Tuning (mode: {})...[/bold]".format(OPTIMIZE_MODE))
    best, results = tune(gts, yolo_cache, frcnn_cache, f1_config, seed=123, warm_trials=WARM_TRIALS)

    console.print("\n[bold green]Best config[/bold green]")
    best_clean = {
        "score": round(best["score"], 6),
        "mAP@0.5": round(best["mAP50"], 6),
        "mAP@[.5:.95]": round(best["mAPcoco"], 6),
        "bestF1": round(best["bestF1"], 6),
        "bestF1_thr": best["bestF1_thr"],
        "bestF1_P": best["bestF1_P"],
        "bestF1_R": best["bestF1_R"],
        "recall@P≥{:.2f}".format(PRECISION_TARGET): best["recall_at_Pt"],
        "precision@R≥{:.2f}".format(RECALL_TARGET): best["precision_at_Rt"],
        "params": best["params"]
    }
    console.print(json.dumps(best_clean, indent=2))

    console.print("\n[bold]Leaderboard[/bold]")
    show_top(results, k=min(10, len(results)))

    # Threshold guidance for deployment
    msg = None
    if OPTIMIZE_MODE == "recall_at_precision" and best.get("thr_at_Pt", None) is not None:
        msg = "Recommended threshold for P≥{:.2f}: {}".format(PRECISION_TARGET, best.get("thr_at_Pt"))
    elif OPTIMIZE_MODE == "precision_at_recall" and best.get("thr_at_Rt", None) is not None:
        msg = "Recommended threshold for R≥{:.2f}: {}".format(RECALL_TARGET, best.get("thr_at_Rt"))
    elif OPTIMIZE_MODE in ("f1", "composite"):
        msg = "Recommended deployment threshold (best-F1): {:.4f}".format(best_clean["bestF1_thr"])
    if msg:
        console.print("\n[bold cyan]{}[/bold cyan]".format(msg))

if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# """
# Voter hyperparameter tuner (strict mode: F1 priors fixed)

# - Runs base detectors once with very low thresholds to cache detections.
# - Re-runs the voter many times while tuning:
#     * gamma               : rank sharpness in (conf ** gamma) * F1
#     * voter_conf_thresh   : solo-acceptance gate inside voter_merge
# - Explores a wide search space with:
#     * deterministic extreme seeds (grid corners)
#     * heavy random warmup (biased to also explore high thresholds)
#     * multi-start coordinate ascent (log-space) with shrinking steps
# - Reports mAP (0.5, 0.50:0.95), best-F1 operating point, recall@precision target,
#   precision@recall target, plus a leaderboard.

# This script NEVER alters your per-class F1 priors (from config.json).
# """

# import os
# import cv2
# import time
# import json
# import math
# import random
# import numpy as np
# from glob import glob
# from rich.console import Console
# from rich.table import Table
# import torch

# # --- Project imports (adjust if needed) ---
# from yolov7.frame_inference import load_yolov7_model, detect_frame
# from fasterrcnn.frame_inference import load_fasterrcnn_model, run_fasterrcnn_on_frame
# from voter import voter_merge, load_f1_config, iou

# console = Console()

# # ===========================
# # Paths / Config
# # ===========================
# IMG_DIR = "/home/brandon/Desktop/motherboard_model_training/Datasets/motherboard_yolov7/test/images"
# LBL_DIR = "/home/brandon/Desktop/motherboard_model_training/Datasets/motherboard_yolov7/test/labels"
# YOLO_WEIGHTS = "./weights/yolov7.pt"
# FRCNN_WEIGHTS = "./weights/fasterrcnn.pth"
# F1_CONFIG = "./config.json"

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # Evaluation knobs
# IOU_THRESH = 0.5
# COCO_IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)

# # Run base models ONCE with low thresholds (recall-complete)
# BASE_CONF_THRESH_YOLO = 0.001
# BASE_CONF_THRESH_FRCNN = 0.001

# # Class list (must match your dataset)
# CLASS_NAMES = [
#     'CPU_FAN_NO_Screws',
#     'CPU_FAN_Screw_loose',
#     'CPU_FAN_Screws',
#     'CPU_fan',
#     'CPU_fan_port',
#     'CPU_fan_port_detached',
#     'Incorrect_Screws',
#     'Loose_Screws',
#     'No_Screws',
#     'Scratch',
#     'Screws'
# ]
# NUM_CLASSES = len(CLASS_NAMES)
# NAME2ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# # ===========================
# # Multi-objective options
# # ===========================
# # OPTIMIZE_MODE = "recall_at_precision"   # "composite" | "map" | "f1" | "recall_at_precision" | "precision_at_recall" | "constrained"

# # # Composite weights (only used if OPTIMIZE_MODE == "composite")
# # W_MAP_COCO = 1.0
# # W_F1_MAX   = 0.3
# # W_R_AT_P   = 0.5
# # W_P_AT_R   = 0.0

# # # PR tradeoff targets
# # PRECISION_TARGET = 0.97     # for recall@precision
# # RECALL_TARGET    = 0.90     # for precision@recall

# # # Constraints (if OPTIMIZE_MODE == "constrained")
# # MIN_RECALL_AT_P  = 0.95
# # MIN_PREC_AT_R    = 0.95

# # # Search budget (wider exploration)
# # WARM_TRIALS     = 300     # heavy random warmup
# # COORD_ITERS     = 200     # deeper local refinement
# # TOPK_FOR_LOCAL  = 10       # number of top warm seeds to refine
# # LOCAL_RESTARTS  = 8       # additional random restarts around best

# OPTIMIZE_MODE = "recall_at_precision"   # "composite" | "map" | "f1" | "recall_at_precision" | "precision_at_recall" | "constrained"

# # Composite weights (only used if OPTIMIZE_MODE == "composite")
# W_MAP_COCO = 1.0
# W_F1_MAX   = 0.3
# W_R_AT_P   = 0.5
# W_P_AT_R   = 0.0

# # PR tradeoff targets
# PRECISION_TARGET = 0.97     # optimize recall at this precision
# RECALL_TARGET    = 0.90     # (used only if OPTIMIZE_MODE == "precision_at_recall")

# # Constraints (if OPTIMIZE_MODE == "constrained")
# MIN_RECALL_AT_P  = 0.95
# MIN_PREC_AT_R    = 0.95

# # Search budget (wider exploration)
# WARM_TRIALS     = 300     # heavy random + grid warmup
# COORD_ITERS     = 200     # deeper local refinement
# TOPK_FOR_LOCAL  = 10      # number of top warm seeds to refine
# LOCAL_RESTARTS  = 8       # additional random restarts around best

# # ===========================
# # Utils
# # ===========================
# def set_seeds(seed=123):
#     """Set RNG seeds for reproducibility across Python, NumPy, and PyTorch."""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# def norm_label(s: str) -> str:
#     """Normalize label strings to a canonical form (lowercase, underscores)."""
#     return s.strip().lower().replace(" ", "_")

# CANON = {norm_label(k): k for k in CLASS_NAMES}
# def canon_label(s: str) -> str:
#     """Map a potentially inconsistent label string to the canonical CLASS_NAMES form."""
#     return CANON.get(norm_label(s), s)

# def list_images(img_dir):
#     """Return a sorted list of image file paths under img_dir with common extensions."""
#     img_files = []
#     for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"):
#         img_files.extend(glob(os.path.join(img_dir, ext)))
#     return sorted(img_files)

# def load_yolo_labels(lbl_path, img_shape, class_names):
#     """Parse YOLO .txt labels and convert to absolute pixel [x1,y1,x2,y2] + class name."""
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
#             x1 = (cx - bw/2) * w
#             y1 = (cy - bh/2) * h
#             x2 = (cx + bw/2) * w
#             y2 = (cy + bh/2) * h
#             label = canon_label(class_names[cls])
#             boxes.append({'bbox': [float(x1), float(y1), float(x2), float(y2)], 'label': label, 'conf': 1.0})
#     return boxes

# # ===========================
# # PR/AP/mAP (academically correct)
# # ===========================
# def build_pr_data(all_preds, all_gts, class_names, iou_thresh):
#     """
#     Convert predictions & GT to dataset-wide PR-building arrays using greedy matching
#     per class and per image. Sort predictions by descending confidence for matching.

#     Args:
#         all_preds: list(list(det)) - detections per image (dicts with 'bbox','label','conf')
#         all_gts:   list(list(gt))  - ground-truth per image (dicts with 'bbox','label')
#         class_names: list[str]
#         iou_thresh: float IoU threshold for a TP match

#     Returns:
#         tp_flags: np.ndarray[int] - 1 for TP, 0 for FP per prediction (global order)
#         confs:    np.ndarray[float] - confidence per prediction (same order)
#         pred_cls: np.ndarray[int] - class index per prediction
#         n_gt_per_class: np.ndarray[int] - total GT count per class
#     """
#     tp_flags, confs, pred_cls = [], [], []
#     n_gt_per_class = np.zeros(len(class_names), dtype=int)

#     for preds, gts in zip(all_preds, all_gts):
#         for g in gts:
#             n_gt_per_class[NAME2ID[g['label']]] += 1

#         gt_by_class = {c: [] for c in class_names}
#         gt_used_by_class = {c: [] for c in class_names}
#         for g in gts:
#             gt_by_class[g['label']].append(g['bbox'])
#             gt_used_by_class[g['label']].append(False)

#         preds_sorted = sorted(preds, key=lambda d: float(d.get('conf', 0.0)), reverse=True)
#         for p in preds_sorted:
#             c = canon_label(p['label'])
#             conf = float(p.get('conf', 0.0))
#             pred_cls.append(NAME2ID.get(c, -1))
#             confs.append(conf)

#             if c not in gt_by_class or len(gt_by_class[c]) == 0:
#                 tp_flags.append(0)
#                 continue

#             best_iou, best_idx = 0.0, -1
#             for i, (gbox, used) in enumerate(zip(gt_by_class[c], gt_used_by_class[c])):
#                 if used:
#                     continue
#                 ov = iou(p['bbox'], gbox)
#                 if ov > best_iou:
#                     best_iou, best_idx = ov, i

#             if best_idx >= 0 and best_iou >= iou_thresh:
#                 tp_flags.append(1)
#                 gt_used_by_class[c][best_idx] = True
#             else:
#                 tp_flags.append(0)

#     return (np.array(tp_flags, np.int32),
#             np.array(confs, np.float32),
#             np.array(pred_cls, np.int32),
#             n_gt_per_class)

# def ap_from_pr(tp_flags, confs, pred_cls, n_gt_per_class, num_classes):
#     """
#     Compute per-class AP with precision envelope + 101-point interpolation (COCO-style).
#     Classes with no GT remain NaN and are ignored in the mean via nanmean.

#     Returns:
#         ap: np.ndarray[float] - AP per class (NaN where undefined)
#         mAP: float            - mean of AP over classes with GT (nanmean)
#         p_last, r_last: np.ndarray[float] - precision/recall at last point in sweep
#     """
#     ap = np.full(num_classes, np.nan, dtype=np.float32)
#     p_last = np.zeros(num_classes, dtype=np.float32)
#     r_last = np.zeros(num_classes, dtype=np.float32)

#     order = np.argsort(-confs)
#     tp_flags = tp_flags[order]
#     pred_cls = pred_cls[order]

#     for c in range(num_classes):
#         cls_mask = (pred_cls == c)
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
#     """Compute mAP at a single IoU threshold (e.g., 0.5) using build_pr_data + ap_from_pr."""
#     tp_flags, confs, pred_cls, n_gt = build_pr_data(all_preds, all_gts, class_names, iou_thresh)
#     ap_per_cls, mAP, _, _ = ap_from_pr(tp_flags, confs, pred_cls, n_gt, len(class_names))
#     return mAP, ap_per_cls

# def compute_coco_map(all_preds, all_gts, class_names, iou_thresholds):
#     """Compute COCO mAP by averaging mAP across multiple IoU thresholds (0.50:0.95)."""
#     aps = []
#     for thr in iou_thresholds:
#         m, _ = compute_map_at_iou(all_preds, all_gts, class_names, thr)
#         aps.append(m)
#     return float(np.mean(aps))

# def global_pr_operating_points(all_preds, all_gts, class_names, iou_thresh,
#                                precision_target=None, recall_target=None):
#     """
#     Build global (micro) precision/recall arrays and extract:
#       - best-F1 operating point and its score threshold
#       - recall at a precision target
#       - precision at a recall target
#     """
#     tp_flags, confs, pred_cls, n_gt = build_pr_data(all_preds, all_gts, class_names, iou_thresh)
#     n_gt_total = int(np.sum(n_gt))

#     order = np.argsort(-confs)
#     tp = tp_flags[order].astype(np.int32)
#     fp = 1 - tp
#     scores_sorted = confs[order]

#     tpc = np.cumsum(tp)
#     fpc = np.cumsum(fp)

#     recall = tpc / (n_gt_total + 1e-9)
#     precision = tpc / (tpc + fpc + 1e-9)

#     f1 = 2 * precision * recall / (precision + recall + 1e-9)
#     idx_best = int(np.argmax(f1)) if f1.size else -1

#     best = dict(
#         best_f1=float(f1[idx_best]) if idx_best >= 0 else 0.0,
#         best_thr=float(scores_sorted[idx_best]) if idx_best >= 0 else 1.0,
#         best_p=float(precision[idx_best]) if idx_best >= 0 else 0.0,
#         best_r=float(recall[idx_best]) if idx_best >= 0 else 0.0,
#     )

#     rec_at_p, thr_at_p = None, None
#     if precision_target is not None and precision.size:
#         mask = precision >= precision_target
#         if np.any(mask):
#             idx = int(np.argmax(recall * mask))
#             rec_at_p = float(recall[idx])
#             thr_at_p = float(scores_sorted[idx])

#     prec_at_r, thr_at_r = None, None
#     if recall_target is not None and recall.size:
#         mask = recall >= recall_target
#         if np.any(mask):
#             idx = int(np.argmax(precision * mask))
#             prec_at_r = float(precision[idx])
#             thr_at_r = float(scores_sorted[idx])

#     return {
#         "precision": precision, "recall": recall, "scores": scores_sorted,
#         "best": best,
#         "recall_at_p_target": rec_at_p, "thr_at_p_target": thr_at_p,
#         "precision_at_r_target": prec_at_r, "thr_at_r_target": thr_at_r
#     }

# # ===========================
# # Cache base detections (run once)
# # ===========================
# def cache_base_detections(img_files, yolo_model, frcnn_model, frcnn_classes):
#     """
#     Run YOLOv7 and Faster R-CNN once with very low thresholds and cache detections.
#     Also load YOLO-format ground truth and return in parallel lists.

#     Returns:
#         frames: list[np.ndarray] (not used in tuning loop but handy for debug)
#         gts: list[list[dict]]    - GT boxes per image
#         yolo_cache, frcnn_cache: list[list[dict]] - detections per image
#     """
#     frames, gts = [], []
#     yolo_cache, frcnn_cache = [], []

#     for img_path in img_files:
#         fname = os.path.basename(img_path)
#         frame = cv2.imread(img_path)
#         if frame is None:
#             console.print("[WARN] Could not read {}, skipping.".format(fname))
#             continue

#         lbl_path = os.path.join(LBL_DIR, os.path.splitext(fname)[0] + ".txt")
#         gt_boxes = load_yolo_labels(lbl_path, frame.shape, CLASS_NAMES)
#         gts.append(gt_boxes)
#         frames.append(frame)

#         # Base detectors ONCE (low thresholds)
#         _, yolo_preds = detect_frame(frame, yolo_model, device=DEVICE, conf_thresh=BASE_CONF_THRESH_YOLO)
#         _, frcnn_preds = run_fasterrcnn_on_frame(frame, frcnn_model, frcnn_classes, device=DEVICE, conf_thresh=BASE_CONF_THRESH_FRCNN)

#         for d in yolo_preds:  d['label'] = canon_label(d['label'])
#         for d in frcnn_preds: d['label'] = canon_label(d['label'])

#         yolo_cache.append(yolo_preds)
#         frcnn_cache.append(frcnn_preds)

#     return frames, gts, yolo_cache, frcnn_cache

# # ===========================
# # Voter evaluation on cached detections (STRICT: F1 priors fixed, no score scaling)
# # ===========================
# def evaluate_voter(gts, yolo_cache, frcnn_cache,
#                    f1_config,
#                    gamma=1.5,
#                    voter_conf_thresh=0.05):
#     """
#     Run voter_merge on cached detections with given hyperparameters and compute:
#       - mAP@0.5 and COCO mAP
#       - best-F1 operating point (threshold, P, R)
#       - recall at precision target and precision at recall target

#     Args:
#         gts: list[list[dict]] - ground-truth boxes per image
#         yolo_cache, frcnn_cache: cached detections per image
#         f1_config: dict - per-class F1 priors (immutable)
#         gamma: float - sharpness in score = (conf ** gamma) * F1_prior
#         voter_conf_thresh: float - solo acceptance gate inside voter_merge

#     Returns:
#         dict of metrics suitable for scalar objective + leaderboard.
#     """
#     voter_all = []
#     for yolo_preds, frcnn_preds in zip(yolo_cache, frcnn_cache):
#         final_preds, _aux = voter_merge(
#             yolo_preds, frcnn_preds, f1_config,
#             conf_thresh=voter_conf_thresh,  # solo acceptance gate
#             gamma=gamma                      # rank sharpness in (conf^gamma)*F1
#             # other voter params use their defaults (iou_thresh, f1_margin, solo_strong, fuse_coords)
#         )
#         for d in final_preds:
#             d['label'] = canon_label(d['label'])
#         voter_all.append(final_preds)

#     # mAP metrics
#     mAP50, _ = compute_map_at_iou(voter_all, gts, CLASS_NAMES, IOU_THRESH)
#     mAP_coco = compute_coco_map(voter_all, gts, CLASS_NAMES, COCO_IOU_THRESHOLDS)

#     # Global PR operating points (micro)
#     pr = global_pr_operating_points(
#         voter_all, gts, CLASS_NAMES, IOU_THRESH,
#         precision_target=PRECISION_TARGET, recall_target=RECALL_TARGET
#     )

#     return {
#         "mAP50": mAP50,
#         "mAPcoco": mAP_coco,
#         "bestF1": pr["best"]["best_f1"],
#         "bestF1_thr": pr["best"]["best_thr"],
#         "bestF1_P": pr["best"]["best_p"],
#         "bestF1_R": pr["best"]["best_r"],
#         "recall_at_Pt": pr["recall_at_p_target"],
#         "thr_at_Pt": pr["thr_at_p_target"],
#         "precision_at_Rt": pr["precision_at_r_target"],
#         "thr_at_Rt": pr["thr_at_r_target"],
#     }

# # ===========================
# # Search space + optimizer
# # ===========================
# def sample_params():
#     """
#     Randomly sample a hyperparameter set with wide exploration:
#       - gamma: log-uniform in [0.2, 10]
#       - voter_conf_thresh: 70% log-uniform in [1e-5, 0.3], 30% uniform in [0.3, 0.9]
#     """
#     gamma = float(np.exp(np.random.uniform(np.log(0.2), np.log(10.0))))
#     if np.random.rand() < 0.70:
#         voter_conf = float(10 ** np.random.uniform(np.log10(1e-5), np.log10(0.3)))
#     else:
#         voter_conf = float(np.random.uniform(0.3, 0.9))
#     return dict(gamma=gamma, voter_conf_thresh=voter_conf)

# def extreme_seeds():
#     """
#     Deterministic set of 'corner' seeds to ensure coverage of low/high gammas
#     and low/mid/high voter thresholds. Keeps runtime moderate by thinning grid.
#     """
#     gammas = [0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
#     confs  = [1e-5, 1e-4, 0.001, 0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90]
#     seeds = []
#     for g in gammas:
#         for c in confs:
#             if g in (0.2, 1.0, 2.0, 4.0, 8.0, 10.0) or c in (1e-5, 0.001, 0.05, 0.2, 0.6, 0.9):
#                 seeds.append(dict(gamma=float(g), voter_conf_thresh=float(c)))
#     return seeds

# def coordinate_search(score_fn, start_params, steps_log, bounds, iters=30):
#     """
#     Derivative-free coordinate ascent in log-space with shrinking steps.
#     Good for non-smooth objectives like mAP/F1.

#     Args:
#         score_fn: callable(params)->dict with keys 'score' and 'params'
#         start_params: dict initial point
#         steps_log: dict of initial step sizes in log-space (per key)
#         bounds: dict of (min, max) per key in real space
#         iters: int number of outer iterations

#     Returns:
#         dict: best result from score_fn (includes 'score' and 'params')
#     """
#     def clamp(v, lo, hi):
#         return max(lo, min(hi, v))

#     p = start_params.copy()
#     best = score_fn(p)
#     improved = True

#     for _ in range(iters):
#         if not improved:
#             for k in steps_log:
#                 steps_log[k] *= 0.5
#         improved = False

#         for key in ['gamma', 'voter_conf_thresh']:
#             base = p[key]
#             lo, hi = bounds[key]
#             log_base = math.log(max(base, 1e-12))
#             step = steps_log[key]

#             tried = []
#             for direction in (+1, -1):
#                 cand = math.exp(log_base + direction * step)
#                 cand = clamp(cand, lo, hi)
#                 q = p.copy(); q[key] = cand
#                 tried.append(score_fn(q))

#             candidates = tried + [best]
#             candidates.sort(key=lambda r: r['score'], reverse=True)
#             if candidates[0]['score'] > best['score'] + 1e-9:
#                 best = candidates[0]
#                 p = best['params'].copy()
#                 improved = True

#         if max(steps_log.values()) < 1e-3:
#             break

#     return best

# def make_scalar_objective(metrics):
#     """
#     Convert the metrics dict to a single scalar 'score' depending on OPTIMIZE_MODE.
#     Returns a copy of metrics with 'score' and 'reason' fields added.
#     """
#     m = metrics
#     score = None
#     reason = ""

#     if OPTIMIZE_MODE == "map":
#         score = m["mAPcoco"]; reason = "mAPcoco"

#     elif OPTIMIZE_MODE == "f1":
#         score = m["bestF1"]; reason = "bestF1"

#     elif OPTIMIZE_MODE == "recall_at_precision":
#         score = -1.0
#         if m["recall_at_Pt"] is not None:
#             score = m["recall_at_Pt"]
#         reason = "recall@precision_target"

#     elif OPTIMIZE_MODE == "precision_at_recall":
#         score = -1.0
#         if m["precision_at_Rt"] is not None:
#             score = m["precision_at_Rt"]
#         reason = "precision@recall_target"

#     elif OPTIMIZE_MODE == "constrained":
#         feas = True
#         if MIN_RECALL_AT_P is not None:
#             feas = feas and (m["recall_at_Pt"] is not None and m["recall_at_Pt"] >= MIN_RECALL_AT_P)
#         if MIN_PREC_AT_R is not None:
#             feas = feas and (m["precision_at_Rt"] is not None and m["precision_at_Rt"] >= MIN_PREC_AT_R)
#         score = m["mAPcoco"] if feas else -1.0
#         reason = "mAPcoco (constrained)"

#     else:  # "composite"
#         r_at_p = m["recall_at_Pt"] if m["recall_at_Pt"] is not None else 0.0
#         p_at_r = m["precision_at_Rt"] if m["precision_at_Rt"] is not None else 0.0
#         score = (W_MAP_COCO * m["mAPcoco"]
#                  + W_F1_MAX   * m["bestF1"]
#                  + W_R_AT_P   * r_at_p
#                  + W_P_AT_R   * p_at_r)
#         reason = "composite"

#     out = dict(score=score, reason=reason)
#     out.update(m)
#     return out

# def tune(gts, yolo_cache, frcnn_cache, f1_config, seed=123, warm_trials=WARM_TRIALS):
#     """
#     Full tuning loop:
#       1) Evaluate deterministic extreme seeds + random samples.
#       2) Pick top-K seeds and run coordinate ascent (log-space) from each.
#       3) Do a few Gaussian restarts around the best and refine again.
#       4) Return best result and a leaderboard (sorted by score).
#     """
#     set_seeds(seed)

#     # caching evals so repeated params are cheap
#     cache = {}
#     def key_of(params):
#         return json.dumps({k: round(float(v), 10) for k, v in sorted(params.items())})

#     def evaluate_params(p):
#         k = key_of(p)
#         if k in cache:
#             return cache[k]
#         metrics = evaluate_voter(gts, yolo_cache, frcnn_cache, f1_config, **p)
#         result = make_scalar_objective(metrics)
#         result['params'] = p.copy()
#         cache[k] = result
#         console.print(
#             "trial score={:.4f}  mAP@0.5={:.4f}  COCO={:.4f}  F1={:.4f} (thr={:.4f}, P={:.3f}, R={:.3f})  "
#             "R@P≥{:.2f}={}  P@R≥{:.2f}={}  params={}".format(
#                 result['score'], metrics['mAP50'], metrics['mAPcoco'],
#                 metrics['bestF1'], metrics['bestF1_thr'], metrics['bestF1_P'], metrics['bestF1_R'],
#                 PRECISION_TARGET, metrics['recall_at_Pt'],
#                 RECALL_TARGET, metrics['precision_at_Rt'],
#                 p
#             )
#         )
#         return result

#     # ---------- Warm seeds: deterministic extremes + random ----------
#     seeds = extreme_seeds()
#     for _ in range(max(0, warm_trials - len(seeds))):
#         seeds.append(sample_params())

#     results = []
#     for p in seeds:
#         results.append(evaluate_params(p))

#     # ---------- Select top-K seeds for local refinement ----------
#     results_sorted = sorted(results, key=lambda r: r['score'], reverse=True)
#     top_seeds = [r['params'] for r in results_sorted[:TOPK_FOR_LOCAL]]

#     bounds = {
#         'gamma': (0.2, 10.0),
#         'voter_conf_thresh': (1e-5, 0.9),
#     }

#     best_overall = results_sorted[0]
#     all_refined = []

#     for si, seed_params in enumerate(top_seeds, 1):
#         steps_log = {'gamma': 0.8, 'voter_conf_thresh': 1.0}
#         refined = coordinate_search(evaluate_params, seed_params, steps_log, bounds, iters=COORD_ITERS)
#         all_refined.append(refined)
#         if refined['score'] > best_overall['score']:
#             best_overall = refined
#         console.print("refine[{}/{}] seed={} -> score={:.4f} params={}".format(
#             si, len(top_seeds), seed_params, refined['score'], refined['params'])
#         )

#     # ---------- Additional random restarts around the current best ----------
#     for r in range(LOCAL_RESTARTS):
#         jitter = {
#             'gamma': float(np.exp(np.random.normal(loc=np.log(best_overall['params']['gamma']), scale=0.25))),
#             'voter_conf_thresh': float(np.exp(np.random.normal(
#                 loc=np.log(max(best_overall['params']['voter_conf_thresh'], 1e-5)), scale=0.5)))
#         }
#         # clamp to bounds
#         jitter['gamma'] = min(max(jitter['gamma'], bounds['gamma'][0]), bounds['gamma'][1])
#         jitter['voter_conf_thresh'] = min(max(jitter['voter_conf_thresh'], bounds['voter_conf_thresh'][0]),
#                                           bounds['voter_conf_thresh'][1])

#         steps_log = {'gamma': 0.5, 'voter_conf_thresh': 0.7}
#         refined = coordinate_search(evaluate_params, jitter, steps_log, bounds, iters=int(COORD_ITERS * 0.6))
#         all_refined.append(refined)
#         if refined['score'] > best_overall['score']:
#             best_overall = refined
#         console.print("restart[{}/{}] -> score={:.4f} params={}".format(
#             r + 1, LOCAL_RESTARTS, refined['score'], refined['params'])
#         )

#     leaderboard = sorted(results + all_refined, key=lambda r: r['score'], reverse=True)
#     return best_overall, leaderboard

# # ===========================
# # Pretty print leaderboard (safe: no nested f-strings)
# # ===========================
# def show_top(results, k=10):
#     """
#     Render the top-k results in a Rich table.
#     Avoids nested f-strings; handles None gracefully.
#     """
#     rows = results[:k]
#     table = Table(title="Top {} Voter Hyperparams (mode: {})".format(min(k, len(results)), OPTIMIZE_MODE))
#     cols = ["Rank","Score","mAP@0.5","mAP@[.5:.95]","F1","F1_thr","P","R","R@P*","P@R*","gamma","voter_conf"]
#     for c in cols:
#         table.add_column(c, justify="right")

#     def fmt(val, spec=".4f"):
#         try:
#             return format(float(val), spec)
#         except (TypeError, ValueError):
#             return "None"

#     for i, r in enumerate(rows, 1):
#         recall_at_pt_val = r.get("recall_at_Pt", None)
#         prec_at_rt_val   = r.get("precision_at_Rt", None)
#         recall_at_pt_str = "None" if recall_at_pt_val is None else fmt(recall_at_pt_val, ".3f")
#         prec_at_rt_str   = "None" if prec_at_rt_val   is None else fmt(prec_at_rt_val,   ".3f")

#         p = r.get("params", {}) or {}

#         table.add_row(
#             str(i),
#             fmt(r.get("score")),
#             fmt(r.get("mAP50")),
#             fmt(r.get("mAPcoco")),
#             fmt(r.get("bestF1")),
#             fmt(r.get("bestF1_thr")),
#             fmt(r.get("bestF1_P"), ".3f"),
#             fmt(r.get("bestF1_R"), ".3f"),
#             recall_at_pt_str,
#             prec_at_rt_str,
#             fmt(p.get("gamma"), ".3f"),
#             fmt(p.get("voter_conf_thresh")),
#         )

#     console.print(table)

# # ===========================
# # Main
# # ===========================
# def main():
#     """Entry point: load models, cache detections, run tuner, and print results."""
#     console.print("[bold]Loading models...[/bold]")
#     yolo_model = load_yolov7_model(YOLO_WEIGHTS, device=DEVICE)
#     frcnn_model, frcnn_classes = load_fasterrcnn_model(FRCNN_WEIGHTS, device=DEVICE)
#     f1_config = load_f1_config(F1_CONFIG)  # F1 priors are FIXED and never changed

#     console.print("[bold]Indexing images...[/bold]")
#     img_files = list_images(IMG_DIR)
#     console.print("Found {} images.".format(len(img_files)))

#     console.print("[bold]Caching base detections (one pass)...[/bold]")
#     t0 = time.time()
#     frames, gts, yolo_cache, frcnn_cache = cache_base_detections(img_files, yolo_model, frcnn_model, frcnn_classes)
#     console.print("Cached {} images in {:.1f}s".format(len(frames), time.time()-t0))

#     # Optional: IoU self-check
#     if yolo_cache and yolo_cache[0]:
#         b = yolo_cache[0][0]['bbox']
#         console.print("IoU self-test (should be 1.0): {:.3f}".format(iou(b, b)))

#     console.print("[bold]Tuning (mode: {})...[/bold]".format(OPTIMIZE_MODE))
#     best, results = tune(gts, yolo_cache, frcnn_cache, f1_config, seed=123, warm_trials=WARM_TRIALS)

#     console.print("\n[bold green]Best config[/bold green]")
#     best_clean = {
#         "score": round(best["score"], 6),
#         "mAP@0.5": round(best["mAP50"], 6),
#         "mAP@[.5:.95]": round(best["mAPcoco"], 6),
#         "bestF1": round(best["bestF1"], 6),
#         "bestF1_thr": best["bestF1_thr"],
#         "bestF1_P": best["bestF1_P"],
#         "bestF1_R": best["bestF1_R"],
#         "recall@P≥{:.2f}".format(PRECISION_TARGET): best["recall_at_Pt"],
#         "precision@R≥{:.2f}".format(RECALL_TARGET): best["precision_at_Rt"],
#         "params": best["params"]
#     }
#     console.print(json.dumps(best_clean, indent=2))

#     console.print("\n[bold]Leaderboard[/bold]")
#     show_top(results, k=min(10, len(results)))

#     # Recommend a deployment threshold depending on mode
#     thr_msg = None
#     if OPTIMIZE_MODE in ("f1", "composite"):
#         thr_msg = "Recommended deployment threshold (best-F1): {:.4f}".format(best_clean["bestF1_thr"])
#     elif OPTIMIZE_MODE == "recall_at_precision" and best_clean.get("recall@P≥{:.2f}".format(PRECISION_TARGET)) is not None:
#         thr_msg = "Recommended threshold achieving P≥{:.2f}: {}".format(PRECISION_TARGET, best.get("thr_at_Pt", None))
#     elif OPTIMIZE_MODE == "precision_at_recall" and best_clean.get("precision@R≥{:.2f}".format(RECALL_TARGET)) is not None:
#         thr_msg = "Recommended threshold achieving R≥{:.2f}: {}".format(RECALL_TARGET, best.get("thr_at_Rt", None))
#     if thr_msg:
#         console.print("\n[bold cyan]{}[/bold cyan]".format(thr_msg))

# if __name__ == "__main__":
#     main()

