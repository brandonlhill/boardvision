# ALGO:
"""
The improved voter algorithm is designed to intelligently combine the results of two object 
detection models (YOLOv7 and Faster R-CNN) for each frame of a video. For every detection, it
 first attempts to match bounding boxes from both models by class name and by measuring their 
 overlap using the Intersection over Union (IoU) metric. When both models predict the same 
 class for the same region (i.e., their bounding boxes overlap significantly), the algorithm 
 compares the F1 scores of each model for that class (from a configuration file) and selects 
 the detection from the model with the higher F1 score, provided the model's confidence for 
 that detection is above a certain threshold. The "winning" detection is then displayed as 
 the final decision in the voter output.

If a bounding box is detected by only one model (meaning there is no significant overlap with
 any box from the other model), the algorithm checks whether the F1 score for the detecting 
 model is greater than that of the other model for the predicted class and whether the 
 detection's confidence is sufficiently high. Additionally, to allow for close contests, 
 if the F1 scores of both models are within 5% of each other, and the detection's confidence
  is above 0.95, the algorithm will accept and display this detection as well.

Visually, all candidate detections from YOLOv7 and Faster R-CNN are shown on the “VOTER” output
 panel with color-coding (orange for YOLOv7, blue for Faster R-CNN), while the final chosen 
 bounding boxes—those that “win” according to the above rules—are highlighted in bold yellow 
 and labeled as "FINAL." This ensures that, for every region and class, the most reliable model
  is trusted, but also allows very confident outlier predictions to be included, resulting in 
  an ensemble that combines both models' strengths with explainable logic.
"""


import json
import cv2
import numpy as np

def load_f1_config(config_path="config.json"):
    with open(config_path) as f:
        return json.load(f)

def iou(boxA, boxB):
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

def voter_merge(yolo_results, frcnn_results, f1_config, conf_thresh=0.5, iou_thresh=0.5, f1_margin=0.05):
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

def draw_voter_boxes_on_frame(frame, final_boxes, candidates):
    img = frame.copy()
    h, w = img.shape[:2]
    # Draw candidates first (faint)
    for det in candidates:
        x1, y1, x2, y2 = det['bbox']
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        conf = det.get('conf', 0)
        label = det['label']
        src = det.get('source', "")
        if src == "YOLOv7":
            color = (0, 128, 255)  # orange
        elif src == "FasterRCNN":
            color = (255, 128, 0)  # blue
        else:
            color = (160,160,160)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        cv2.putText(img, f"{label} {src}", (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # Draw final winner boxes (bold yellow)
    for det in final_boxes:
        x1, y1, x2, y2 = det['bbox']
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        label = det['label']
        conf = det.get('conf', 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,255), 3)
        cv2.putText(img, f"{label} FINAL", (x1, max(0, y1-10)), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,255,255), 2)
    return img

def overlay_label(img, label, color=(0,255,255)):
    cv2.putText(img, label, (20, 40),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 3)
    return img

# import json
# import cv2
# import numpy as np

# def load_f1_config(config_path="config.json"):
#     with open(config_path) as f:
#         return json.load(f)

# def get_preferred_model_for_class(class_name, f1_config):
#     entry = f1_config[class_name]
#     return "YOLOv7" if entry["YOLOv7"] >= entry["FRCNN"] else "FRCNN"

# def iou(boxA, boxB):
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])
#     interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
#     if interArea == 0:
#         return 0.0
#     boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
#     boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
#     iou = interArea / float(boxAArea + boxBArea - interArea)
#     return iou

# def voter_merge(yolo_results, frcnn_results, f1_config, conf_thresh=0.5, iou_thresh=0.5):
#     yolo_used = [False] * len(yolo_results)
#     frcnn_used = [False] * len(frcnn_results)
#     merged = []

#     for i, y_det in enumerate(yolo_results):
#         best_iou = 0
#         best_j = -1
#         for j, f_det in enumerate(frcnn_results):
#             if y_det['label'] == f_det['label']:
#                 curr_iou = iou(y_det['bbox'], f_det['bbox'])
#                 if curr_iou > best_iou:
#                     best_iou = curr_iou
#                     best_j = j
#         if best_iou > iou_thresh:
#             class_name = y_det['label']
#             f1_yolo = f1_config[class_name]["YOLOv7"]
#             f1_frcnn = f1_config[class_name]["FRCNN"]
#             if f1_yolo >= f1_frcnn:
#                 if y_det['conf'] > conf_thresh:
#                     merged.append(y_det)
#             else:
#                 if frcnn_results[best_j]['conf'] > conf_thresh:
#                     merged.append(frcnn_results[best_j])
#             yolo_used[i] = True
#             frcnn_used[best_j] = True

#     for i, y_det in enumerate(yolo_results):
#         if not yolo_used[i]:
#             class_name = y_det['label']
#             f1_yolo = f1_config[class_name]["YOLOv7"]
#             f1_frcnn = f1_config[class_name]["FRCNN"]
#             if f1_yolo > f1_frcnn and y_det['conf'] > conf_thresh:
#                 merged.append(y_det)

#     for j, f_det in enumerate(frcnn_results):
#         if not frcnn_used[j]:
#             class_name = f_det['label']
#             f1_yolo = f1_config[class_name]["YOLOv7"]
#             f1_frcnn = f1_config[class_name]["FRCNN"]
#             if f1_frcnn > f1_yolo and f_det['conf'] > conf_thresh:
#                 merged.append(f_det)
#     return merged

# def draw_detections_on_frame(frame, detections, color=(0,255,255)):
#     img = frame.copy()
#     h, w = img.shape[:2]
#     for det in detections:
#         x1, y1, x2, y2 = det['bbox']
#         x1, y1 = max(0, x1), max(0, y1)
#         x2, y2 = min(w-1, x2), min(h-1, y2)
#         label = det['label']
#         conf = det.get('conf', 0)
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(img, f"{label} {conf:.2f}", (x1, max(0, y1-5)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#     return img

# def overlay_label(img, label, color=(0,255,255)):
#     cv2.putText(img, label, (20, 40),
#                 cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 3)
#     return img
