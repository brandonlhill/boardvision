from __future__ import annotations
import cv2
import numpy as np
import torch
import sys
import os
import logging

from collections import Counter
from typing import Optional, List, Any
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QTextCursor, QTextOption
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
    QStatusBar,
    QComboBox,
    QFrame,
    QTextEdit,
    QSizePolicy,
)



from yolov7.frame_inference import load_yolov7_model, detect_frame
from fasterrcnn.frame_inference import load_fasterrcnn_model, run_fasterrcnn_on_frame
from voter import (
    load_f1_config,
    voter_merge,
    draw_voter_boxes_on_frame,
    overlay_label,
)
from PySide6.QtGui import QTextOption
from PySide6.QtWidgets import QSizePolicy
from rich.logging import RichHandler

################### NOTE ###################
# TODO: Multi-Threading support (thread inference) 
# TODO: Implement stream handler: given a video stream, buffer the stream (use a python lib to determine the best frames to inference on [reduce infernece costs], then send the frames to the inference engine)
# TODO: Clean up the hacky drop down
# TODO: Update the requirements.txt to reflect the new packages imported by this GUI


################### Logging ###################
LOGGER = logging.getLogger("boardvision")
logging.basicConfig(
    level=logging.DEBUG, # TODO: Disable in final release 
    format="%(message)s",  # Let RichHandler format the rest
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)


################### UI Widgets ###################
class Panel(QFrame):
    """A titled panel with a video area and a scrolling text log."""

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.setStyleSheet(
            """
            QFrame#Panel { background-color: #1a1a1a; border: 1px solid #2b2b2b; border-radius: 10px; }
            QLabel#Title { color: #e5e5e5; font-weight: 700; padding: 8px 0; font-size: 16px; }
            QLabel#Video { background: #000; border: 1px solid #2b2b2b; border-radius: 12px; }
            QTextEdit { background: #0f0f0f; color: #cfcfcf; border-top: 1px solid #2b2b2b; font-size: 13px; }
            """
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(8)

        # Title
        self.title = QLabel(title)
        self.title.setObjectName("Title")
        self.title.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        # Video: never push layout; we scale pixmaps ourselves
        self.video = QLabel()
        self.video.setObjectName("Video")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(0, 0)
        self.video.setScaledContents(False)  # we handle scaling
        self.video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)  # <<< key: ignore sizeHint

        # Log: fixed height + never push layout vertically or horizontally
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(140)
        self.log.setLineWrapMode(QTextEdit.NoWrap)
        self.log.setWordWrapMode(QTextOption.NoWrap)
        self.log.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.log.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)     # <<< key: no horizontal pressure

        outer.addWidget(self.title)
        outer.addWidget(self.video, 1)  # expands only within available space
        outer.addWidget(self.log, 0)    # stays fixed

    def append_log(self, text: str) -> None:
        self.log.append(text)
        self.log.moveCursor(QTextCursor.End)





################### Main Window ###################
class VideoInferenceWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BoardVision – Live Inference")
        self.resize(1350, 800)

        # State
        self.video_path: str = ""
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running: bool = False
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.frame_index: int = 0

        # Models (eager load to mirror original behavior)
        LOGGER.info("Loading models (device=%s) …", self.device)
        self.yolov7_model = load_yolov7_model("weights/yolov7.pt", device=self.device)
        self.frcnn_model, self.frcnn_classes = load_fasterrcnn_model(
            "weights/fasterrcnn.pth", device=self.device
        )
        self.f1_config = load_f1_config("config.json")
        LOGGER.info("Models loaded.")

        # Build UI
        self._build_ui()
        self._connect_signals()

        # Timer for frame processing
        self.timer = QTimer(self)
        self.timer.setInterval(0)  # process next frame ASAP
        self.timer.timeout.connect(self._process_next_frame)

        # Status bar 
        self.statusBar().showMessage(f"Device: {self.device}")

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(8)

        # Top bar (Source + Start/Stop)
        top = QHBoxLayout()
        lbl_source = QLabel("Source:")
        self.source_combo = QComboBox()
        self.source_combo.addItem("Open File…")
        self.source_combo.addItem("Stream")
        self.source_combo.setMinimumHeight(34)   # larger dropdown
        self.source_combo.setStyleSheet(
            """
            QComboBox {
                background: #2a2a2a; 
                color: #e0e0e0; 
                border: 1px solid #3a3a3a; 
                padding: 6px 10px; 
                border-radius: 6px;
                font-size: 14px;
            }
            QComboBox QAbstractItemView {
                background: #2a2a2a; 
                color: #e0e0e0;
                selection-background-color: #444444;
                selection-color: #ffffff;
                border: 1px solid #3a3a3a;
            }
            """
        )

        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        for b, clr in ((self.btn_start, "#2e7dff"), (self.btn_stop, "#e23c3c")):
            b.setMinimumHeight(34)   # larger buttons
            b.setStyleSheet(
                f"""
                QPushButton {{
                    background: {clr}; 
                    color: white; 
                    border: none; 
                    padding: 8px 16px; 
                    border-radius: 8px;
                    font-size: 14px;
                }}
                QPushButton:disabled {{ background: #555; }}
                """
            )

        top.addWidget(lbl_source)
        top.addWidget(self.source_combo)
        top.addStretch(1)
        top.addWidget(self.btn_start)
        top.addWidget(self.btn_stop)

        # Three columns
        cols = QHBoxLayout()
        cols.setSpacing(12)
        self.panel_yolo = Panel("YOLOv7")
        self.panel_voter = Panel("Voter / Ensemble")
        self.panel_frcnn = Panel("Faster R-CNN")
        cols.addWidget(self.panel_yolo, 1)
        cols.addWidget(self.panel_voter, 1)
        cols.addWidget(self.panel_frcnn, 1)

        root.addLayout(top)
        root.addLayout(cols, 1)

        # Dark window style
        self.setStyleSheet(
            """
            QMainWindow { background: #101010; }
            QLabel { color: #cfcfcf; font-size: 14px; }
            QStatusBar { background: #0e0e0e; color: #9a9a9a; }
            """
        )


    def _connect_signals(self) -> None:
        self.source_combo.activated.connect(self._select_video)
        self.btn_start.clicked.connect(self._start_inference)
        self.btn_stop.clicked.connect(self._stop_inference)

    def _select_video(self) -> None: #TODO: I think there is a framerate issue with the video loader
        """ Video controller """
        idx = self.source_combo.currentIndex()
        if idx == 0:  # Open File…
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Video",
                "",
                "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
            )
            if path:
                self.video_path = path
                LOGGER.info("Selected video: %s", path)
                self.statusBar().showMessage(
                    f"Selected: {os.path.basename(path)} — Device: {self.device}"
                )
            else:
                LOGGER.warning("Open File canceled.")
        elif idx == 1:  # Stream option (not implemented yet)
            QMessageBox.information(
                self, "Stream", "Stream option selected (not yet implemented)."
            )
            LOGGER.info("Stream option chosen (not yet implemented).")

    def _start_inference(self) -> None:
        if self.is_running:
            return
        if not self.video_path:
            QMessageBox.warning(
                self, "No Video", "Please select a video file from Source → Open File…"
            )
            return

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            if self.cap:
                self.cap.release()
            self.cap = None
            QMessageBox.critical(self, "Error", "Failed to open video file.")
            LOGGER.error("Failed to open video: %s", self.video_path)
            return

        self.is_running = True
        self.frame_index = 0
        LOGGER.info("Starting inference on %s", self.video_path)
        self._clear_panel_logs()
        self.timer.start()

    def _stop_inference(self) -> None:
        if not self.is_running:
            return
        self.is_running = False
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.statusBar().showMessage(f"Stopped — Device: {self.device}")
        LOGGER.info("Stopped inference.")

    def _clear_panel_logs(self) -> None:
        self.panel_yolo.log.clear()
        self.panel_voter.log.clear()
        self.panel_frcnn.log.clear()

    def _process_next_frame(self) -> None:
        """ Frame Processing: GUI timer refresh rate"""
        if not self.is_running or self.cap is None:
            self.timer.stop()
            return

        ok, frame = self.cap.read()
        if not ok:
            self._stop_inference()
            QMessageBox.information(self, "Info", "End of video.")
            LOGGER.info("Reached end of video after %d frames.", self.frame_index)
            return

        idx = self.frame_index
        self.frame_index += 1

        try:
            # Run YOLOv7
            yolo_img, yolo_dets = detect_frame(
                frame.copy(), model=self.yolov7_model, device=self.device
            )
            yolo_summary = self._summarize_dets(yolo_dets)
            self.panel_yolo.append_log(f"#{idx:05d} {yolo_summary}")

            # Run Faster R-CNN
            frcnn_img, frcnn_dets = run_fasterrcnn_on_frame(
                frame.copy(),
                model=self.frcnn_model,
                CLASSES=self.frcnn_classes,
                device=self.device,
            )
            frcnn_summary = self._summarize_dets(
                frcnn_dets, class_names=self.frcnn_classes
            )
            self.panel_frcnn.append_log(f"#{idx:05d} {frcnn_summary}")

            # VOTER merge
            final_boxes, candidates = voter_merge(
                yolo_dets, frcnn_dets, self.f1_config
            )
            voter_img = draw_voter_boxes_on_frame(
                frame.copy(), final_boxes, candidates
            )
            voter_summary = self._summarize_voter(final_boxes, candidates)
            self.panel_voter.append_log(f"#{idx:05d} {voter_summary}")

            # Overlay small labels
            yolo_img = overlay_label(yolo_img, "YOLOv7", color=(0, 128, 255))
            frcnn_img = overlay_label(
                frcnn_img, "Faster R-CNN", color=(255, 128, 0)
            )
            voter_img = overlay_label(voter_img, "VOTER", color=(0, 255, 255))

            # Display in each panel
            self._display_image(self.panel_yolo.video, yolo_img)
            self._display_image(self.panel_voter.video, voter_img)
            self._display_image(self.panel_frcnn.video, frcnn_img)

            # Terminal logging (strong)
            LOGGER.info(
                "F%05d | yolo: %s | frcnn: %s | voter: %s",
                idx,
                yolo_summary,
                frcnn_summary,
                voter_summary,
            )

        except Exception as e:  # noqa: BLE001
            self._stop_inference()
            QMessageBox.critical(self, "Inference Error", f"Error during inference:\n{e}")
            LOGGER.exception("Inference error on frame %d: %s", idx, e)

    @staticmethod
    def _resize_keep_h(img: np.ndarray, target_h: int) -> np.ndarray:
        h, w = img.shape[:2]
        new_w = int(w * (target_h / float(h)))
        return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

    def _display_image(self, label: QLabel, bgr: np.ndarray) -> None:
        # Convert BGR -> RGB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        bytes_per_line = 3 * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Create pixmap and scale it to the label's CURRENT size
        pm = QPixmap.fromImage(qimg)
        # Important: scale to label.size() so we don't change QLabel's sizeHint
        scaled = pm.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled)

    @staticmethod
    def _summarize_dets(
        dets: Any, class_names: Optional[List[str]] = None
    ) -> str:
        """Return a compact string with counts by class and basic stats.

        `dets` can be:
        - list[dict] with keys like ('label' or 'class' or 'cls', 'conf' or 'score')
        - tuple(np.ndarray, ...) or other shapes — we will try to infer counts
        """
        n = 0
        labels: List[str] = []
        confs: List[float] = []

        if dets is None:
            return "(no detections)"

        # Try common structures robustly
        try:
            if isinstance(dets, (list, tuple)):
                for d in dets:
                    if isinstance(d, dict):
                        n += 1
                        lbl = d.get("label") or d.get("class") or d.get("cls")
                        if (
                            isinstance(lbl, (int, np.integer))
                            and class_names
                            and 0 <= int(lbl) < len(class_names)
                        ):
                            labels.append(class_names[int(lbl)])
                        elif isinstance(lbl, str):
                            labels.append(lbl)
                        conf = (
                            d.get("conf") or d.get("score") or d.get("confidence")
                        )
                        if isinstance(conf, (int, float, np.floating)):
                            confs.append(float(conf))
                    else:
                        # assume iterable box; count only
                        n += 1
            else:
                # Could be numpy array [N, …]
                if hasattr(dets, "shape") and len(getattr(dets, "shape", [])) >= 1:
                    n = int(dets.shape[0])
                else:
                    n = len(dets)  # type: ignore[arg-type]
        except Exception:
            try:
                n = len(dets)  # type: ignore[arg-type]
            except Exception:
                n = 0

        from collections import Counter as _Counter

        by_cls = _Counter(labels)
        cls_str = (
            ", ".join(f"{k}:{v}" for k, v in sorted(by_cls.items())) if by_cls else "no det list"
        )
        if confs:
            mean_conf = sum(confs) / max(len(confs), 1)
            conf_str = f"avg_conf={mean_conf:.3f} max={max(confs):.3f}"
        else:
            conf_str = "avg_conf=NA"
        return f"{n} dets ({cls_str}); {conf_str}"

    @staticmethod
    def _summarize_voter(final_boxes: Any, candidates: Any) -> str:
        def _safe_len(x: Any) -> int:
            try:
                return len(x) if x is not None else 0
            except Exception:
                return 0

        return f"fused={_safe_len(final_boxes)} (candidates={_safe_len(candidates)})"

    # Ensure we clean up resources on window close
    def closeEvent(self, event) -> None:  # noqa: N802
        try:
            self._stop_inference()
        finally:
            super().closeEvent(event)


################### Main ###################
def main() -> int:
    app = QApplication(sys.argv)
    win = VideoInferenceWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
