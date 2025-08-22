import sys
import yolov7.models as yolov7_models
sys.modules['models'] = yolov7_models

import torch
import cv2
import random
from yolov7.models.experimental import attempt_load
from yolov7.utils.general import non_max_suppression, scale_coords
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device

def load_yolov7_model(weights, device='cpu', img_size=640, trace=True):
    model = attempt_load(weights, map_location=device)
    if trace:
        from yolov7.utils.torch_utils import TracedModel
        model = TracedModel(model, device, img_size)
    model.eval()
    return model

def detect_frame(frame, model, device='cpu', img_size=640, conf_thresh=0.6, iou_thres=0.5, classes=None, agnostic_nms=False):
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0,255) for _ in range(3)] for _ in names]

    im0 = frame.copy()  # original image, for output and scaling
    # Prepare for inference: resize to img_size, RGB, CHW, float32, normalize
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (img_size, img_size))
    img_resized = img_resized.transpose(2, 0, 1)
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
        # Scale coords back to original frame size!
        det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], im0.shape).round()
        for *xyxy, conf, cls in det:
            label = names[int(cls)]
            bbox = [int(x.item()) for x in xyxy]
            detections.append({'bbox': bbox, 'label': label, 'conf': float(conf)})
            plot_one_box(xyxy, im0, label=f'{label} {conf:.2f}', color=colors[int(cls)], line_thickness=2)
    return im0, detections
