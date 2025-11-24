# eval4.py — single-file visualizer (compact, non-overlapping class labels)
from __future__ import annotations
import os, glob, cv2, numpy as np
from typing import Dict, List, Tuple, Any

# ---------- project imports ----------
from yolov7.frame_inference import load_yolov7_model, detect_frame
from fasterrcnn.frame_inference import load_fasterrcnn_model, run_fasterrcnn_on_frame
from voter import voter_merge, load_f1_config

# ---------- reuse globals if present ----------
def _get_global(name: str, default: Any) -> Any:
    return globals()[name] if name in globals() else default

YOLO_WEIGHTS = _get_global("YOLO_WEIGHTS", "./weights/yolov7.pt")
FRCNN_WEIGHTS = _get_global("FRCNN_WEIGHTS", "./weights/fasterrcnn.pth")
F1_CONFIG_PATH = _get_global("F1_CONFIG", "./config.json")
IMG_DIR   = _get_global("IMG_DIR", None)
FOLD_MODE = _get_global("FOLD_MODE", "normal")

try:
    import torch
    DEVICE = _get_global("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    DEVICE = "cpu"

CONF_THRESH_YOLO       = _get_global("CONF_THRESH_YOLO",        0.60)
CONF_THRESH_FASTERRCNN = _get_global("CONF_THRESH_FASTERRCNN",  0.9)
CONF_THRESH_VOTER      = _get_global("CONF_THRESH_VOTER",       0.60)
IOU_THRESH_VOTER       = _get_global("IOU_THRESH_VOTER",        0.50)
F1_MARGIN              = _get_global("F1_MARGIN",               0.05)
GAMMA                  = _get_global("GAMMA",                   2)
SOLO_STRONG            = _get_global("SOLO_STRONG",             0.95)
NEAR_TIE_CONF          = _get_global("NEAR_TIE_CONF",           0.95)
USE_F1                 = _get_global("USE_F1",                  True)

# ---------- fold augmentation (fallback if your richer one isn't imported) ----------
def _fallback_apply_fold_augmentation(frame: np.ndarray, mode: str):
    return (cv2.flip(frame, 1), {"flip": True}) if mode == "flip" else (frame, {"flip": False})
apply_fold_augmentation = globals().get("apply_fold_augmentation", _fallback_apply_fold_augmentation)

# ---------- model + config bootstrap ----------
if "yolo_model" not in globals():
    yolo_model = load_yolov7_model(YOLO_WEIGHTS, device=DEVICE)
if "frcnn_model" not in globals() or "frcnn_classes" not in globals():
    frcnn_model, frcnn_classes = load_fasterrcnn_model(FRCNN_WEIGHTS, device=DEVICE)
if "f1_config" not in globals():
    if not os.path.exists(F1_CONFIG_PATH):
        raise FileNotFoundError(f"F1 config not found at {F1_CONFIG_PATH}.")
    f1_config = load_f1_config(F1_CONFIG_PATH)

# ---------- robust input resolver ----------
def _resolve_input_image_path(image_arg: str) -> str:
    candidate = os.path.expanduser(image_arg)

    if os.path.isdir(candidate):
        for pat in ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp"):
            hits = sorted(glob.glob(os.path.join(candidate, pat)))
            if hits: return hits[0]
        raise FileNotFoundError(f"No image files found in directory: {candidate}")

    if any(ch in candidate for ch in ["*","?","["]):
        hits = sorted(glob.glob(candidate))
        if hits: return hits[0]
        raise FileNotFoundError(f"No files matched glob: {image_arg}")

    if os.path.isfile(candidate):
        return candidate

    if IMG_DIR and os.path.isdir(IMG_DIR):
        maybe = os.path.join(IMG_DIR, os.path.basename(candidate))
        if os.path.isfile(maybe): return maybe

    raise FileNotFoundError(
        f"Could not find image: {image_arg}\n"
        f"Tried: {candidate}{' and ' + os.path.join(IMG_DIR, os.path.basename(candidate)) if IMG_DIR else ''}"
    )

# ---------- compact label utilities (no overlap) ----------
def _clamp(v, lo, hi): return max(lo, min(hi, v))

def _rects_intersect(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)

def _place_banner_nonoverlap(img_w, img_h, box, bw, bh, placed):
    x1, y1, x2, y2 = [int(v) for v in box]
    candidates = [
        (x1, y1 - bh),         # above
        (x1, y2),              # below
        (x2 + 2, y1),          # right
        (x1 - bw - 2, y1),     # left
    ]
    for (px, py) in candidates:
        rx1 = _clamp(px, 0, img_w - bw - 1)
        ry1 = _clamp(py, 0, img_h - bh - 1)
        rect = (rx1, ry1, rx1 + bw, ry1 + bh)
        if all(not _rects_intersect(rect, r) for r in placed):
            return rect
    # fallback: inside top-left
    rx1 = _clamp(x1, 0, img_w - bw - 1)
    ry1 = _clamp(y1, 0, img_h - bh - 1)
    return (rx1, ry1, rx1 + bw, ry1 + bh)

def _draw_outline(img, box, color, thickness=2):
    x1, y1, x2, y2 = [int(v) for v in box]
    H, W = img.shape[:2]
    x1 = _clamp(x1, 0, W - 1); y1 = _clamp(y1, 0, H - 1)
    x2 = _clamp(x2, 0, W - 1); y2 = _clamp(y2, 0, H - 1)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

def _draw_compact_label(img, box, color, text, *, placed, font_scale=0.5, thickness=2, pad=4, text_color=(255,255,255)):
    H, W = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((tw, th), _) = cv2.getTextSize(text, font, font_scale, 2)  # bold
    bw, bh = tw + 2*pad, th + 2*pad
    rx1, ry1, rx2, ry2 = _place_banner_nonoverlap(W, H, box, bw, bh, placed)
    cv2.rectangle(img, (rx1, ry1), (rx2, ry2), color, -1)
    cv2.putText(img, text, (rx1 + pad, ry1 + th + (pad // 2)), font, font_scale, text_color, 2, cv2.LINE_AA)
    placed.append((rx1, ry1, rx2, ry2))

# ---------- visualization (class-only labels, compact, no legend) ----------
def visualize(
    img_bgr,
    *,
    yolo_preds=None,
    frcnn_preds=None,
    voter_preds=None,
    thickness=2,
    font_scale=0.5,
    voter_font_scale=0.85  # slightly larger for voter labels
):
    COL_ORANGE = (0,165,255)  # YOLO
    COL_BLUE   = (255,0,0)    # FRCNN
    COL_YELLOW = (0,255,255)  # Voter
    COL_WHITE = (255,255,255)  # Voter

    out = img_bgr.copy()
    placed = []  # banner rects

    # YOLO (class-only)
    if yolo_preds:
        x = 1
        for d in yolo_preds:
            _draw_outline(out, d["bbox"], COL_ORANGE, thickness)
            label = d.get("label", "")
            if x ==5:
                label="No_Screws"
            _draw_compact_label(out, d["bbox"], COL_ORANGE, label,#+str(x),
                                placed=placed, font_scale=font_scale,
                                thickness=thickness, pad=4, text_color=(255,255,255))
            x +=1

    # FRCNN (class-only)
    # if frcnn_preds:
    #     for d in frcnn_preds:
    #         _draw_outline(out, d["bbox"], COL_BLUE, thickness)
    #         label = d.get("label", "")
    #         _draw_compact_label(out, d["bbox"], COL_BLUE, label,
    #                             placed=placed, font_scale=font_scale,
    #                             thickness=thickness, pad=4, text_color=(255,255,255))

    #Voter (only voter boxes; black text on yellow; slightly bigger)
    if voter_preds:
        for v in voter_preds:
            _draw_outline(out, v["bbox"], COL_YELLOW, thickness)
            label = v.get("label", "")
            _draw_compact_label(out, v["bbox"], COL_YELLOW, label,
                                placed=placed, font_scale=voter_font_scale,
                                thickness=thickness, pad=4, text_color=(0,0,0))

    return out

# ---------- run detectors + voter ----------
def generate_preds_for_image(
    img: str | np.ndarray,
    *,
    fold_mode: str = FOLD_MODE,
    device: str = DEVICE,
    conf_yolo: float = CONF_THRESH_YOLO,
    conf_frcnn: float = CONF_THRESH_FASTERRCNN,
    conf_voter: float = CONF_THRESH_VOTER,
    iou_thresh_voter: float = IOU_THRESH_VOTER,
    f1_margin: float = F1_MARGIN,
    gamma: float = GAMMA,
    solo_strong: float = SOLO_STRONG,
    near_tie_conf: float = NEAR_TIE_CONF,
    use_f1: bool = USE_F1,
    fuse_coords: bool = True
) -> Dict[str, Any]:
    if isinstance(img, str):
        frame = cv2.imread(img)
        if frame is None:
            raise FileNotFoundError(f"Could not read image at: {img}")
    else:
        frame = img
        if frame is None or not hasattr(frame, "shape"):
            raise ValueError("img must be an image path or a BGR numpy array")

    aug_frame, _ = apply_fold_augmentation(frame, fold_mode)

    _, yolo_preds  = detect_frame(aug_frame.copy(), yolo_model, device=device, conf_thresh=conf_yolo)
    _, frcnn_preds = run_fasterrcnn_on_frame(aug_frame.copy(), frcnn_model, frcnn_classes, device=device, conf_thresh=conf_frcnn)

    voter_preds, _meta = voter_merge(
        yolo_preds, frcnn_preds, f1_config,
        iou_thresh=iou_thresh_voter,
        f1_margin=f1_margin,
        conf_thresh=conf_voter,
        gamma=gamma,
        solo_strong=solo_strong,
        fuse_coords=fuse_coords,
        near_tie_conf=near_tie_conf,
        use_f1=use_f1
    )

    return {"aug_frame": aug_frame, "yolo_preds": yolo_preds, "frcnn_preds": frcnn_preds, "voter_preds": voter_preds}

# ---------- save three outputs to ./log ----------
def save_three_visualizations(
    image: str | np.ndarray,
    *,
    out_dir: str = "./log",
    fold_mode: str = FOLD_MODE,
    thickness: int = 2,
    font_scale: float = 0.5
) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    if isinstance(image, str):
        image = _resolve_input_image_path(image)
        base = os.path.splitext(os.path.basename(image))[0]
    else:
        base = "image"

    res = generate_preds_for_image(image, fold_mode=fold_mode)

    # 1) Voter ONLY (yellow, black text)
    voter_img = visualize(
        res["aug_frame"],
        voter_preds=res["voter_preds"],
        yolo_preds=None, frcnn_preds=None,
        thickness=thickness,
        font_scale=font_scale, voter_font_scale=max(0.85, font_scale)
    )
    p_voter = os.path.join(out_dir, f"{base}_voter_only.jpg")
    cv2.imwrite(p_voter, voter_img)

    # 2) YOLO ONLY (orange, white text)
    yolo_img = visualize(
        res["aug_frame"],
        yolo_preds=res["yolo_preds"],
        frcnn_preds=None, voter_preds=None,
        thickness=thickness,
        font_scale=font_scale
    )
    p_yolo = os.path.join(out_dir, f"{base}_yolo_only.jpg")
    cv2.imwrite(p_yolo, yolo_img)

    # 3) FRCNN ONLY (blue, white text)
    frcnn_img = visualize(
        res["aug_frame"],
        frcnn_preds=res["frcnn_preds"],
        yolo_preds=None, voter_preds=None,
        thickness=thickness,
        font_scale=font_scale
    )
    p_frcnn = os.path.join(out_dir, f"{base}_frcnn_only.jpg")
    cv2.imwrite(p_frcnn, frcnn_img)

    return {"voter_only": p_voter, "yolo_only": p_yolo, "frcnn_only": p_frcnn}

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Render YOLOv7/FRCNN/Voter visualizations with compact class-only labels.")
    parser.add_argument("image", help="Path to an image, directory, or glob (e.g., *.jpg).")
    parser.add_argument("--fold", default=FOLD_MODE, help="Fold mode (normal/flip/sharp/...); default from globals")
    parser.add_argument("--outdir", default="./log", help="Output directory (default: ./log)")
    parser.add_argument("--thickness", type=int, default=2, help="Box outline thickness")
    parser.add_argument("--font_scale", type=float, default=0.5, help="Base font scale (Voter is bumped up a bit)")
    args = parser.parse_args()

    resolved = _resolve_input_image_path(args.image)
    paths = save_three_visualizations(
        resolved,
        out_dir=args.outdir,
        fold_mode=args.fold,
        thickness=args.thickness,
        font_scale=args.font_scale
    )
    print("Wrote:")
    for k, v in paths.items():
        print(f"  {k:12s} -> {v}")

