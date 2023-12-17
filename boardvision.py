import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from time import sleep
from PIL import Image, ImageTk
import cv2
import subprocess
import os
import re

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BoardVision App - Alpha v1.0")
        self.analysis_video_file = None
        
        # Create GUI components
        self.label = tk.Label(root, text="Select an mp4 video file:")
        self.label.pack(pady=10)

        self.browse_button = tk.Button(root, text="Browse", command=self.browse_file)
        self.browse_button.pack()

        self.canvas = None

        # Create a StringVar to store the selected model
        self.model_var = tk.StringVar(root)
        self.model_var.set("yolov7")  # Set the default value

        # Create a label
        label = tk.Label(root, text="Select a model:")
        label.pack(pady=10)

        # Create an OptionMenu with the available models
        model_menu = tk.OptionMenu(root, self.model_var, "yolov7", "faster-rcnn")
        model_menu.pack(pady=10)

        self.progressbar = ttk.Progressbar(root, orient="horizontal", length=200, mode="determinate")
        self.progressbar.pack()

        self.detect_button = tk.Button(root, text="Detect Objects", command=self.detect_objects)
        self.detect_button.pack()

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        if file_path:
            #self.display_video(file_path)
            self.analysis_video_file = file_path
            print("[INFO] Selected Video File: ",self.analysis_video_file)

    def create_canvas(self):
        if self.canvas == None:
            self.canvas = tk.Canvas(self.root)
            self.canvas.pack()

    def display_video(self, file_path):
        self.create_canvas()

        cap = cv2.VideoCapture(file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_width, target_height = 480, 360  # Set the target width and height

        for i in range(total_frames):
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize the frame to the target width and height
            frame = cv2.resize(frame, (target_width, target_height))

            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(img)

            self.canvas.config(width=img.width(), height=img.height())
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
            self.canvas.image = img

            # Update progressbar
            #self.progressbar["value"] = i + 1
            self.root.update_idletasks()

            # Delay to control the playback speed (adjust as needed)
            self.root.after(50)  # 50 milliseconds delay
        
        cap.release()
        self.canvas.destroy()
        self.canvas = None
   
    def update_progress(self, output):
        # Example line: "stdout: video 1/1 (7/368) /mnt/c/Users/nnn/Desktop/MotherboardProject/test_video.mp4: Done. (20.7ms) Inference, (0.3ms) NMS"
        match = re.search(r'\((\d+)/(\d+)\)', output)
        if match:
            current_frame, total_frames = map(int, match.groups())
            progress_value = int((current_frame / total_frames) * 100)
            self.progressbar["value"] = progress_value
            self.root.update_idletasks()
    
    def get_latest_exp_directory(self, parent_path, pattern="exp\d+"):
        # Get a list of all directories in the specified path
        directories = [d for d in os.listdir(parent_path) if os.path.isdir(os.path.join(parent_path, d))]

        # Filter directories that match the specified pattern
        exp_directories = [d for d in directories if re.match(pattern, d)]
        
        if len(directories) == 1:
            return directories[0]
        
        # Sort the directories by creation time
        sorted_directories = sorted(exp_directories, key=lambda d: os.path.getctime(os.path.join(parent_path, d)), reverse=True)

        # Return the latest directory name
        if sorted_directories:
            return sorted_directories[0]
        else:
            return None
    
    def detect_objects(self):
        print("[INFO] Detect button pressed.")
        self.progressbar["value"] = 0
        #gets the selected model 
        selected_model = self.model_var.get()
        cdir = os.getcwd()
        
        # Perform your object detection on 'frame' here
        if selected_model == "yolov7":
            print("[INFO] Spawning Subprocess for image detection (yolov7).")
            command = f"python3 {cdir}/Yolov7/detect.py --device 0 --weights {cdir}/Trained_Yolov7/best.pt --source {self.analysis_video_file} --project {cdir}/outputs/project"
            print("[CMD][EXEC] ", command)
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            while True:
                output = process.stdout.readline()
                error = process.stderr.readline()

                if output == '' and error == '' and process.poll() is not None:
                    break
                if output:
                    print(f"stdout: {output.strip()}", flush=True)
                    self.update_progress(output)
                if error:
                    print(f"stderr: {error.strip()}", flush=True)
                sleep(0.01)

            # Wait for the process to complete
            process.wait()

            print("[INFO] Finished running video against model.")
            
            # After object detection is complete, display the output video in a separate window
            x = self.get_latest_exp_directory(f"{cdir}/outputs/project/")
            model_video = f"{cdir}/outputs/project/{x}/test_video.mp4"
           
            self.display_video(model_video)
        
        # Perform your object detection on 'frame' here
        elif selected_model == "faster-rcnn":
            print("[INFO] Spawning Subprocess for image detection (faster-rcnn).")
            # python inference_video.py --input test_video.mp4 --weights outputs/training/output/last_model_state.pth
            command = f"python3 {cdir}/Faster_rcnn/inference_video.py --weights {cdir}/Trained_Faster_rcnn/best_model.pth --input {self.analysis_video_file} "
            print("[CMD][EXEC] ", command)

            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            while True:
                output = process.stdout.readline()
                error = process.stderr.readline()

                if output == '' and error == '' and process.poll() is not None:
                    break
                if output:
                    print(f"stdout: {output.strip()}", flush=True)
                    self.update_progress(output)
                if error:
                    print(f"stderr: {error.strip()}", flush=True)
                sleep(0.01)

            # Wait for the process to complete
            process.wait()

            print("[INFO] Finished running video against model.")
            
            # After object detection is complete, display the output video in a separate window
            
            x = self.get_latest_exp_directory(f"{cdir}/outputs/inference", "res_\d+")
            model_video = f"{cdir}/outputs/inference/{x}/test_video.mp4"
            self.display_video(model_video)

        else:
            print ("[INFO] Unknown model selected.")
        return 0

    def display_output_video(self, output_path):
        cap = cv2.VideoCapture(output_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("Output Video", frame)
            if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
