import torch
import numpy as np
import cv2
from frameworks.fasterrcnn.models.create_fasterrcnn_model import create_model
from frameworks.fasterrcnn.utils.transforms import infer_transforms, resize
from frameworks.fasterrcnn.utils.annotations import convert_detections, inference_annotations

def load_fasterrcnn_model(weights_path, device, model_name='fasterrcnn_resnet50_fpn'):
    checkpoint = torch.load(weights_path, map_location=device)

    data_configs = checkpoint['data']
    NUM_CLASSES = data_configs['NC']
    CLASSES = data_configs['CLASSES']

    model = create_model[model_name](num_classes=NUM_CLASSES, coco_model=False)

    state_dict = checkpoint['model_state_dict']

    # First, try to load as-is (this keeps your legacy working model working)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        # If it fails, check for "module." prefix (DataParallel checkpoints)
        has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
        if not has_module_prefix:
            # Different problem; re-raise the original error
            raise e

        # Build a new state dict without the "module." prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_k = k[len("module."):]
            else:
                new_k = k
            new_state_dict[new_k] = v

        # Try loading again with cleaned keys
        model.load_state_dict(new_state_dict)

    model.to(device).eval()
    return model, CLASSES
# def load_fasterrcnn_model(weights_path, device, model_name='fasterrcnn_resnet50_fpn'):
#     checkpoint = torch.load(weights_path, map_location=device)
#     data_configs = checkpoint['data']
#     NUM_CLASSES = data_configs['NC']
#     CLASSES = data_configs['CLASSES']
#     model = create_model[model_name](num_classes=NUM_CLASSES, coco_model=False)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.to(device).eval()
#     return model, CLASSES

def run_fasterrcnn_on_frame(frame, model, CLASSES, device='cpu', conf_thresh=0.8, image_size=None, square_img=False): # TODO: Rename to detect_frame
    orig_frame = frame.copy()
    frame_height, frame_width = orig_frame.shape[:2]
    RESIZE_TO = image_size if image_size else frame_width

    # Get resize scale if resizing for inference
    frame_resized = resize(orig_frame, RESIZE_TO, square=square_img)
    scale_x = orig_frame.shape[1] / frame_resized.shape[1]
    scale_y = orig_frame.shape[0] / frame_resized.shape[0]

    image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    image = infer_transforms(image)
    image = torch.unsqueeze(image, 0).to(device)

    with torch.no_grad():
        outputs = model(image)
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

    detections = []
    if len(outputs[0]['boxes']) != 0:
        draw_boxes, pred_classes, scores = convert_detections(
            outputs, conf_thresh, CLASSES, {'no_labels': False, 'classes': None, 'track': False}
        )
        # Scale boxes back to original frame size if resized
        draw_boxes_scaled = []
        for box in draw_boxes:
            x1 = int(box[0] * scale_x)
            y1 = int(box[1] * scale_y)
            x2 = int(box[2] * scale_x)
            y2 = int(box[3] * scale_y)
            draw_boxes_scaled.append([x1, y1, x2, y2])
        COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
        out_frame = inference_annotations(
            draw_boxes_scaled, pred_classes, scores, CLASSES, COLORS, orig_frame, orig_frame, {'no_labels': False, 'classes': None, 'track': False}
        )
        for box, cls, score in zip(draw_boxes_scaled, pred_classes, scores):
            label = str(cls)
            detections.append({'bbox': box, 'label': label, 'conf': float(score)})
    else:
        out_frame = orig_frame

    return out_frame, detections
