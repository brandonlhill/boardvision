import numpy as np
import cv2
import torch
import glob
import os
import time
import yaml
import matplotlib.pyplot as plt
from torchvision import transforms as transforms
from deep_sort_realtime.deepsort_tracker import DeepSort

from fasterrcnn.models.create_fasterrcnn_model import create_model
from fasterrcnn.utils.annotations import (
    inference_annotations, convert_detections,
    annotate_fps, convert_pre_track, convert_post_track
)
from fasterrcnn.utils.general import set_infer_dir
from fasterrcnn.utils.transforms import infer_transforms, resize
from fasterrcnn.utils.logging import log_to_json

def collect_all_images(dir_test):
    test_images = []
    if os.path.isdir(dir_test):
        image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
        for file_type in image_file_types:
            test_images.extend(glob.glob(f"{dir_test}/{file_type}"))
    else:
        test_images.append(dir_test)
    return test_images

def run_fasterrcnn_inference(
    input_path,
    data_config=None,
    model_name=None,
    weights_path=None,
    threshold=0.3,
    show=False,
    mpl_show=False,
    device=None,
    image_size=None,
    no_labels=False,
    square_img=False,
    filter_classes=None,
    log_json=False,
    track=False
):
    print("[INFO] Device used: ", torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    np.random.seed(42)

    if isinstance(device, str):
        device = torch.device(device)
    elif device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    settings_dict = {
        'no_labels': no_labels,
        'classes': filter_classes,
        'track': track  # This is the fix: so the annotations function has access to track flag
    }

    if track:
        tracker = DeepSort(max_age=30)

    data_configs = None
    if data_config is not None:
        with open(data_config) as file:
            data_configs = yaml.safe_load(file)
        NUM_CLASSES = data_configs['NC']
        CLASSES = data_configs['CLASSES']

    OUT_DIR = set_infer_dir()

    if weights_path is None:
        if data_configs is None:
            with open(os.path.join('data_configs', 'test_video_config.yaml')) as file:
                data_configs = yaml.safe_load(file)
            NUM_CLASSES = data_configs['NC']
            CLASSES = data_configs['CLASSES']
        build_model = create_model.get(model_name, create_model['fasterrcnn_resnet50_fpn_v2'])
        model, _ = build_model(num_classes=NUM_CLASSES, coco_model=True)
    else:
        checkpoint = torch.load(weights_path, map_location=device)
        if data_configs is None:
            data_configs = checkpoint['data']
            NUM_CLASSES = data_configs['NC']
            CLASSES = data_configs['CLASSES']
        model_name = model_name or checkpoint.get('model_name', 'fasterrcnn_resnet50_fpn_v2')
        build_model = create_model.get(model_name, create_model['fasterrcnn_resnet50_fpn_v2'])
        model = build_model(num_classes=NUM_CLASSES, coco_model=False)

        model_state_dict = checkpoint['model_state_dict']
        try:
            model.load_state_dict(model_state_dict)
        except RuntimeError as e:
            print("\n[Warning] Model load_state_dict() failed with error:")
            print(e)
            print("\nAttempting non-strict load_state_dict()...")
            model.load_state_dict(model_state_dict, strict=False)

    model.to(device).eval()
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    if input_path.lower().endswith(('.mp4', '.avi', '.mov')):
        cap = cv2.VideoCapture(input_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        assert frame_width and frame_height, f"Invalid video file or path: {input_path}"

        save_name = os.path.splitext(os.path.basename(input_path))[0]
        out = cv2.VideoWriter(f"{OUT_DIR}/{save_name}.mp4",
                              cv2.VideoWriter_fourcc(*'mp4v'), 30,
                              (frame_width, frame_height))

        RESIZE_TO = image_size if image_size else frame_width

        frame_count = 0
        total_fps = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            orig_frame = frame.copy()
            frame = resize(frame, RESIZE_TO, square=square_img)
            image = frame.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = infer_transforms(image)
            image = torch.unsqueeze(image, 0)

            start_time = time.time()
            with torch.no_grad():
                outputs = model(image.to(device))
            forward_end_time = time.time()

            forward_pass_time = forward_end_time - start_time
            fps = 1 / forward_pass_time
            total_fps += fps
            frame_count += 1

            outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

            if log_json:
                log_to_json(frame, os.path.join(OUT_DIR, 'log.json'), outputs)

            if len(outputs[0]['boxes']) != 0:
                draw_boxes, pred_classes, scores = convert_detections(
                    outputs, threshold, CLASSES, settings_dict
                )
                if track:
                    tracker_inputs = convert_pre_track(draw_boxes, pred_classes, scores)
                    tracks = tracker.update_tracks(tracker_inputs, frame=frame)
                    draw_boxes, pred_classes, scores = convert_post_track(tracks)

                frame = inference_annotations(
                    draw_boxes, pred_classes, scores, CLASSES,
                    COLORS, orig_frame, frame, settings_dict
                )
            else:
                frame = orig_frame

            frame = annotate_fps(frame, fps)
            out.write(frame)

            print(f"Frame: {frame_count}, FPS: {fps:.2f}, Time per frame: {forward_pass_time:.3f}s")

            if show:
                cv2.imshow('Prediction', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        if frame_count > 0:
            avg_fps = total_fps / frame_count
            print(f"Average FPS: {avg_fps:.3f}")
            return avg_fps
        else:
            print("No valid video frames were processed.")
            return 0

    else:
        test_images = collect_all_images(input_path)
        print(f"Test instances: {len(test_images)}")

        detection_threshold = threshold
        frame_count = 0
        total_fps = 0

        for i, image_path in enumerate(test_images):
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            orig_image = cv2.imread(image_path)
            frame_height, frame_width, _ = orig_image.shape
            RESIZE_TO = image_size if image_size else frame_width

            image_resized = resize(orig_image, RESIZE_TO, square=square_img)
            image = image_resized.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = infer_transforms(image)
            image = torch.unsqueeze(image, 0)

            start_time = time.time()
            with torch.no_grad():
                outputs = model(image.to(device))
            end_time = time.time()

            fps = 1 / (end_time - start_time)
            total_fps += fps
            frame_count += 1

            outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

            if log_json:
                log_to_json(orig_image, os.path.join(OUT_DIR, 'log.json'), outputs)

            if len(outputs[0]['boxes']) != 0:
                draw_boxes, pred_classes, scores = convert_detections(
                    outputs, detection_threshold, CLASSES, settings_dict
                )
                orig_image = inference_annotations(
                    draw_boxes, pred_classes, scores, CLASSES,
                    COLORS, orig_image, image_resized, settings_dict
                )

                if show:
                    cv2.imshow('Prediction', orig_image)
                    cv2.waitKey(1)
                if mpl_show:
                    plt.imshow(orig_image[:, :, ::-1])
                    plt.axis('off')
                    plt.show()

            cv2.imwrite(f"{OUT_DIR}/{image_name}.jpg", orig_image)
            print(f"Image {i+1} done...")
            print('-'*50)

        print('TEST PREDICTIONS COMPLETE')
        cv2.destroyAllWindows()
        avg_fps = total_fps / frame_count
        print(f"Average FPS: {avg_fps:.3f}")
        return avg_fps
