import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import torch

from yolov7.frame_inference import load_yolov7_model, detect_frame
from fasterrcnn.frame_inference import load_fasterrcnn_model, run_fasterrcnn_on_frame
from voter import load_f1_config, voter_merge, draw_voter_boxes_on_frame, overlay_label

class VideoInferenceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv7 vs Faster R-CNN vs VOTER Inference (Side by Side)")

        self.video_path = ""
        self.cap = None
        self.is_running = False

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolov7_model = load_yolov7_model("weights/yolov7_best.pt", device=self.device)
        self.frcnn_model, self.frcnn_classes = load_fasterrcnn_model("weights/fasterrcnn.pth", device=self.device)
        self.f1_config = load_f1_config("config.json")

        self.add_widgets()

    def add_widgets(self):
        tk.Button(self.root, text="Select Video File", command=self.select_video).pack(pady=10)
        tk.Button(self.root, text="Start Inference", command=self.start_inference).pack(pady=10)
        self.display_label = tk.Label(self.root)
        self.display_label.pack(pady=10)

    def select_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if not self.video_path:
            messagebox.showwarning("No File", "Please select a video file.")

    def start_inference(self):
        if not self.video_path:
            messagebox.showwarning("No Video", "Please select a video file.")
            return

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open video file.")
            return

        self.is_running = True
        self.root.after(1, self.process_next_frame)

    def process_next_frame(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            messagebox.showinfo("Info", "End of video.")
            return

        try:
            yolov7_img, yolo_dets = detect_frame(frame.copy(), model=self.yolov7_model, device=self.device)
            frcnn_img, frcnn_dets = run_fasterrcnn_on_frame(
                frame.copy(), model=self.frcnn_model, CLASSES=self.frcnn_classes, device=self.device)

            yolov7_img = overlay_label(yolov7_img, "YOLOv7", color=(0, 128, 255))
            frcnn_img = overlay_label(frcnn_img, "Faster R-CNN", color=(255, 128, 0))

            final_boxes, candidates = voter_merge(yolo_dets, frcnn_dets, self.f1_config)
            voter_img = draw_voter_boxes_on_frame(frame.copy(), final_boxes, candidates)
            voter_img = overlay_label(voter_img, "VOTER", color=(0, 255, 255))

            height = 480
            yolov7_img = cv2.resize(yolov7_img, (int(yolov7_img.shape[1] * height / yolov7_img.shape[0]), height))
            frcnn_img = cv2.resize(frcnn_img, (int(frcnn_img.shape[1] * height / frcnn_img.shape[0]), height))
            voter_img = cv2.resize(voter_img, (int(voter_img.shape[1] * height / voter_img.shape[0]), height))

            combined = cv2.hconcat([yolov7_img, frcnn_img, voter_img])
            rgb_img = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_img)
            imgtk = ImageTk.PhotoImage(image=img)

            self.display_label.imgtk = imgtk
            self.display_label.config(image=imgtk)
        except Exception as e:
            messagebox.showerror("Inference Error", f"Error during inference: {e}")
            self.cap.release()
            self.is_running = False
            return

        self.root.after(1, self.process_next_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoInferenceApp(root)
    root.mainloop()
