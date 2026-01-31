# frameworks/yolov8/frame_inference.py

import torch
import cv2
from ultralytics import YOLO


def load_yolov8_model(weights, device="cpu"):
    """
    Load YOLOv8 model using Ultralytics API.
    Mirrors load_yolov7_model signature.
    """
    model = YOLO(weights)

    # Force device
    if device.startswith("cuda"):
        model.to("cuda")
    elif device == "mps":
        model.to("mps")
    else:
        model.to("cpu")

    model.fuse()  # slightly faster inference
    return model


def detect_frame(
    frame,
    model,
    device="cpu",
    img_size=640,
    conf_thresh=0.6,
    iou_thres=0.5,
    classes=None,
    agnostic_nms=False,
):
    """
    Run YOLOv8 inference on a single frame.
    Output format intentionally matches yolov7.detect_frame().
    """

    im0 = frame.copy()  # BGR

    # Ultralytics handles preprocessing internally
    results = model.predict(
        source=im0,
        conf=conf_thresh,
        iou=iou_thres,
        imgsz=img_size,
        device=device,
        classes=classes,
        agnostic_nms=agnostic_nms,
        verbose=False,
    )

    detections = []

    if not results or len(results) == 0:
        return im0, detections

    r = results[0]

    if r.boxes is None or len(r.boxes) == 0:
        return im0, detections

    boxes = r.boxes
    names = model.names

    for box in boxes:
        xyxy = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])

        label = names.get(cls_id, str(cls_id))
        bbox = [int(x) for x in xyxy]

        detections.append(
            {
                "bbox": bbox,
                "label": label,
                "conf": conf,
            }
        )

        # Optional visualization (comment out if not needed)
        cv2.rectangle(im0, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(
            im0,
            f"{label} {conf:.2f}",
            (bbox[0], max(0, bbox[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return im0, detections
