import sys
import frameworks.yolov7.models as yolov7_models
sys.modules['models'] = yolov7_models

import torch
import cv2
import random
from frameworks.yolov7.models.experimental import attempt_load
from frameworks.yolov7.utils.datasets import letterbox
from frameworks.yolov7.utils.general import non_max_suppression, scale_coords
from frameworks.yolov7.utils.plots import plot_one_box
from frameworks.yolov7.utils.torch_utils import select_device

def load_yolov7_model(weights, device='cpu', img_size=640, trace=True):
    model = attempt_load(weights, map_location=device)
    if trace:
        from frameworks.yolov7.utils.torch_utils import TracedModel
        model = TracedModel(model, device, img_size)
    model.eval()
    return model

def detect_frame(frame, model, device='cpu', img_size=640, conf_thresh=0.6, iou_thres=0.5, classes=None, agnostic_nms=False):
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0,255) for _ in range(3)] for _ in range(len(names))]  # ensure length matches

    im0 = frame.copy()  # original image (BGR)

    # Proper preprocess: letterbox (preserves aspect ratio)
    stride = int(getattr(model, 'stride', torch.tensor([32])).max()) if hasattr(model, 'stride') else 32
    lb = letterbox(im0, new_shape=img_size, stride=stride, auto=True)
    
    # Letterbox returns (img, ratio, (dw, dh)) in YOLOv7
    if isinstance(lb, (list, tuple)) and len(lb) == 3:
        img_letterboxed, ratio, pad = lb
        ratio_pad = (ratio, pad)
    else:
        # Fallback: assume square resize without distortion (rare in v7, but safe)
        img_letterboxed = lb
        h0, w0 = im0.shape[:2]
        r = min(img_size / h0, img_size / w0)
        new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
        dw = (img_size - new_unpad[0]) / 2
        dh = (img_size - new_unpad[1]) / 2
        ratio_pad = ((r, r), (dw, dh))

    # Convert to model input tensor: BGR->RGB, HWC->CHW, [0,1]
    img = cv2.cvtColor(img_letterboxed, cv2.COLOR_BGR2RGB)
    img_resized = img.transpose(2, 0, 1)  # CHW
    img_tensor = torch.from_numpy(img_resized).to(device)
    img_tensor = img_tensor.float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    detections = []
    with torch.no_grad():
        pred = model(img_tensor)[0]
        pred = non_max_suppression(pred, conf_thresh, iou_thres, classes=classes, agnostic=agnostic_nms)
    det = pred[0]

    if det is not None and len(det):
        # Map boxes back to ORIGINAL frame using ratio/pad from letterbox
        det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], im0.shape, ratio_pad=ratio_pad).round()
        for *xyxy, conf, cls in det:
            cls = int(cls)
            label = names[cls] if isinstance(names, (list, tuple)) else names.get(cls, str(cls))
            bbox = [int(x.item()) for x in xyxy]
            detections.append({'bbox': bbox, 'label': label, 'conf': float(conf)})
            plot_one_box(bbox, im0, label=f'{label} {conf:.2f}', color=colors[cls], line_thickness=2)

    return im0, detections
