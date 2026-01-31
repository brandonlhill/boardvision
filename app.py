# app.py
# Goal:
#   - Support exactly TWO detectors (generic) loaded from config.yaml["detectors"]
#   - No hard-coded "yolov7"/"frcnn" assumptions anywhere
#   - Use Pydantic models to manage config + voter params + detections
#   - GUI edits voter params + per-model conf thresholds + F1 table
#   - Persist changes back to the SAME config.yaml on disk (round-trip)
#   - Debug/log output always includes *model names* (not framework names)
#
# Dependencies expected in project:
#   - frame_inference.py provides MultiDetectorEngine with .run(frame)-> dict[name]->list[dict]
#   - voter.py provides TwoModelVoter class (stateful) operating on DetectionModel / VoterParams
#   - models.py provides Pydantic models: DetectionModel, VoterParams
#
# NOTE:
#   - This refactor intentionally keeps the UX similar to your existing GUI.
#   - It dynamically generates UI labels/columns based on the TWO selected detector names.

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from rich.logging import RichHandler
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QImage, QPalette, QColor, QTextCursor, QTextOption
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

# use generic engine + voter + pydantic models
from frame_inference import MultiDetectorEngine, available_providers
from basemodels import DetectionModel, VoterParams  # pydantic models
from voter import TwoModelVoter  # stateful two-model voter

LOGGER = logging.getLogger("boardvision")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

# ----------------------------
# Small utils
# ----------------------------

def human_ms(seconds: float) -> str:
    return f"{seconds * 1000.0:.1f} ms"


def safe_cap_set(cap: cv2.VideoCapture, prop, value) -> None:
    try:
        cap.set(prop, value)
    except Exception:
        pass


def _ensure_dir(p: str) -> None:
    d = os.path.dirname(os.path.abspath(p))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _deep_get(dct: dict, path: List[str], default=None):
    cur = dct
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _deep_set(dct: dict, path: List[str], value):
    cur = dct
    for k in path[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[path[-1]] = value


# ----------------------------
# Drawing helpers
# ----------------------------
def draw_boxes(
    frame_bgr: np.ndarray,
    dets: List[Dict[str, Any]],
    color=(0, 255, 0),
) -> np.ndarray:
    """
    Draw bounding boxes with HIGH-VISIBILITY thickness and
    auto-scaled labels based on image resolution.
    """
    img = frame_bgr.copy()
    h, w = img.shape[:2]

    # ---- VISIBILITY TUNING (KEY CHANGE) ----
    base = min(w, h)

    # Make boxes clearly visible even on 4K images
    box_thickness = max(3, int(base / 400))     # ⬅️ THICKER BOXES
    text_thickness = max(2, int(base / 700))
    font_scale = max(0.6, base / 1000.0)
    pad = max(6, int(base / 250))

    font = cv2.FONT_HERSHEY_SIMPLEX

    for d in dets or []:
        try:
            x1, y1, x2, y2 = map(int, d["bbox"])
            label = str(d.get("label", "?"))
            conf = float(d.get("conf", 0.0))
            src = str(d.get("source", ""))

            text = f"{label} {conf:.2f}"
            if src:
                text += f" [{src}]"

            # Clamp bbox
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            # ---- DRAW BOX (THICK & CLEAR) ----
            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                color,
                box_thickness,
            )

            # ---- LABEL ----
            (tw, th), baseline = cv2.getTextSize(
                text,
                font,
                font_scale,
                text_thickness,
            )

            y_text = max(th + pad, y1 - 8)
            x_bg2 = min(w - 1, x1 + tw + pad * 2)
            y_bg1 = y_text - th - baseline - pad
            y_bg2 = y_text + pad

            # White background
            cv2.rectangle(
                img,
                (x1, y_bg1),
                (x_bg2, y_bg2),
                (255, 255, 255),
                -1,
            )

            # Black text
            cv2.putText(
                img,
                text,
                (x1 + pad, y_text),
                font,
                font_scale,
                (0, 0, 0),
                text_thickness,
                cv2.LINE_AA,
            )

        except Exception:
            continue

    return img

# def draw_boxes(
#     frame_bgr: np.ndarray,
#     dets: List[Dict[str, Any]],
#     color=(0, 255, 0),
# ) -> np.ndarray:
#     """
#     Draw bounding boxes with label text that AUTO-SCALES
#     based on image resolution.
#     """
#     img = frame_bgr.copy()
#     h, w = img.shape[:2]

#     # Scale text relative to image size
#     base = min(w, h)
#     font_scale = max(0.4, base / 1200.0)
#     thickness = max(1, int(base / 800))
#     pad = max(4, int(base / 300))

#     font = cv2.FONT_HERSHEY_SIMPLEX

#     for d in dets or []:
#         try:
#             x1, y1, x2, y2 = map(int, d["bbox"])
#             label = str(d.get("label", "?"))
#             conf = float(d.get("conf", 0.0))
#             src = str(d.get("source", ""))

#             text = f"{label} {conf:.2f}"
#             if src:
#                 text += f" [{src}]"

#             # Clamp box
#             x1 = max(0, min(w - 1, x1))
#             y1 = max(0, min(h - 1, y1))
#             x2 = max(0, min(w - 1, x2))
#             y2 = max(0, min(h - 1, y2))

#             cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

#             (tw, th), baseline = cv2.getTextSize(
#                 text, font, font_scale, thickness
#             )

#             y_text = max(th + pad, y1 - 6)
#             x_bg2 = min(w - 1, x1 + tw + pad * 2)
#             y_bg1 = y_text - th - baseline - pad
#             y_bg2 = y_text + pad

#             # White label background
#             cv2.rectangle(
#                 img,
#                 (x1, y_bg1),
#                 (x_bg2, y_bg2),
#                 (255, 255, 255),
#                 -1,
#             )

#             # Black text
#             cv2.putText(
#                 img,
#                 text,
#                 (x1 + pad, y_text),
#                 font,
#                 font_scale,
#                 (0, 0, 0),
#                 thickness,
#                 cv2.LINE_AA,
#             )
#         except Exception:
#             continue

#     return img

def overlay_label(
    img_bgr: np.ndarray,
    text: str,
    color=(255, 255, 255),
) -> np.ndarray:
    """
    Overlay top-left title that scales with image resolution.
    """
    out = img_bgr.copy()
    h, w = out.shape[:2]

    base = min(w, h)
    font_scale = max(0.6, base / 900.0)
    thickness = max(2, int(base / 700))

    cv2.putText(
        out,
        text,
        (20, int(40 * font_scale)),
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return out



# ----------------------------
# Rolling stats
# ----------------------------

class RollingStats:
    def __init__(self, window: int = 60) -> None:
        self.times: Deque[float] = deque(maxlen=window)

    def add(self, seconds: float) -> None:
        self.times.append(max(1e-6, float(seconds)))

    @property
    def avg_ms(self) -> float:
        return (sum(self.times) / len(self.times)) * 1000.0 if self.times else 0.0

    @property
    def fps(self) -> float:
        if not self.times:
            return 0.0
        total = sum(self.times)
        return (len(self.times) / total) if total > 0 else 0.0


# ----------------------------
# UI Components
# ----------------------------

class Panel(QFrame):
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.setStyleSheet(
            """
            QFrame#Panel { background-color: #171717; border: 1px solid #2b2b2b; border-radius: 10px; }
            QLabel#Title { color: #f0f0f0; font-weight: 700; padding: 8px 0; font-size: 16px; }
            QLabel#Stats { color: #d2d2d2; font-weight: 600; padding-bottom: 4px; font-size: 12px; }
            QLabel#Video { background: #000; border: 1px solid #2b2b2b; border-radius: 12px; }
            QTextEdit { background: #0f0f0f; color: #ededed; border-top: 1px solid #2b2b2b; font-size: 13px; }
            """
        )
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(6)

        self.title = QLabel(title)
        self.title.setObjectName("Title")
        self.title.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        self.stats = QLabel("-")
        self.stats.setObjectName("Stats")
        self.stats.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        self.video = QLabel()
        self.video.setObjectName("Video")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setScaledContents(False)
        self.video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(140)
        self.log.setLineWrapMode(QTextEdit.NoWrap)
        self.log.setWordWrapMode(QTextOption.NoWrap)

        outer.addWidget(self.title)
        outer.addWidget(self.stats)
        outer.addWidget(self.video, 1)
        outer.addWidget(self.log, 0)

    def set_stats(self, ms: float, fps: float) -> None:
        self.stats.setText("-" if (ms <= 0 and fps <= 0) else f"{ms:.1f} ms  |  {fps:.1f} FPS")

    def append_log(self, text: str) -> None:
        self.log.append(text)
        self.log.moveCursor(QTextCursor.End)


class SettingsDock(QDockWidget):
    """
      Builds F1 table and model threshold controls dynamically for EXACTLY two models.
      Exposes generic widgets keyed by model name.
    """
    def __init__(self, parent: VideoInferenceWindow, model_a: str, model_b: str) -> None:
        super().__init__("", parent)
        self.setObjectName("SettingsDock")
        self.setAllowedAreas(Qt.LeftDockWidgetArea)
        self.setFeatures(QDockWidget.NoDockWidgetFeatures)

        self.model_a = model_a
        self.model_b = model_b

        w = QWidget(self)
        self.setWidget(w)
        root = QVBoxLayout(w)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        # Controls
        controls_grp = QGroupBox("Controls")
        cl = QHBoxLayout(controls_grp)
        self.btn_start_left = QPushButton("Start")
        self.btn_stop_left = QPushButton("Stop")
        for b, clr in ((self.btn_start_left, "#2e7dff"), (self.btn_stop_left, "#e23c3c")):
            b.setMinimumHeight(34)
            b.setStyleSheet(
                f"QPushButton {{ background: {clr}; color: white; border: none; padding: 8px 16px; border-radius: 8px; font-size: 14px; }}"
                f" QPushButton:disabled {{ background: #555; }}"
            )
        cl.addWidget(self.btn_start_left)
        cl.addWidget(self.btn_stop_left)
        root.addWidget(controls_grp)

        # Source
        src_grp = QGroupBox("Source")
        gl = QGridLayout(src_grp)
        self.rb_file = QRadioButton("File")
        self.rb_stream = QRadioButton("Stream")
        self.bg_source = QButtonGroup(self)
        self.bg_source.setExclusive(True)
        self.bg_source.addButton(self.rb_file)
        self.bg_source.addButton(self.rb_stream)
        gl.addWidget(self.rb_file, 0, 0)
        gl.addWidget(self.rb_stream, 0, 1, 1, 2)
        self.le_file = QLineEdit()
        self.le_file.setPlaceholderText("Video file path…")
        self.btn_browse = QToolButton()
        self.btn_browse.setText("Pick")
        self.le_stream = QLineEdit()
        self.le_stream.setPlaceholderText("Camera index (e.g., 0) or RTSP/HTTP URL")
        gl.addWidget(QLabel("File:"), 1, 0)
        gl.addWidget(self.le_file, 1, 1)
        gl.addWidget(self.btn_browse, 1, 2)
        gl.addWidget(QLabel("Stream:"), 2, 0)
        gl.addWidget(self.le_stream, 2, 1, 1, 2)

        # Performance
        perf_grp = QGroupBox("Performance")
        pl = QGridLayout(perf_grp)
        self.sb_stride = QSpinBox()
        self.sb_stride.setRange(1, 16)
        self.sb_stride.setValue(1)
        pl.addWidget(QLabel("Stride (process every Nth frame):"), 0, 0)
        pl.addWidget(self.sb_stride, 0, 1)

        # Config file
        cfg_path_grp = QGroupBox("Config File (YAML)")
        cl2 = QHBoxLayout(cfg_path_grp)
        self.le_cfg_path = QLineEdit()
        self.le_cfg_path.setPlaceholderText("config.yaml")
        self.btn_cfg_browse = QToolButton()
        self.btn_cfg_browse.setText("Pick")
        cl2.addWidget(self.le_cfg_path, 1)
        cl2.addWidget(self.btn_cfg_browse, 0)

        # F1 table (dynamic columns)
        cfg_grp = QGroupBox("Per-class F1 Scores")
        tbl_layout = QVBoxLayout(cfg_grp)
        self.tbl_cfg = QTableWidget(0, 3)
        self.tbl_cfg.setHorizontalHeaderLabels(["Label", self.model_a, self.model_b])
        self.tbl_cfg.horizontalHeader().setStretchLastSection(True)
        self.tbl_cfg.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_cfg.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.SelectedClicked | QAbstractItemView.EditKeyPressed
        )
        self.tbl_cfg.verticalHeader().setVisible(False)

        row_btns = QHBoxLayout()
        self.btn_add = QPushButton("Add Row")
        self.btn_del = QPushButton("Delete Selected")
        self.btn_reload = QPushButton("Reload")
        row_btns.addWidget(self.btn_add)
        row_btns.addWidget(self.btn_del)
        row_btns.addStretch(1)
        row_btns.addWidget(self.btn_reload)

        tbl_layout.addWidget(self.tbl_cfg)
        tbl_layout.addLayout(row_btns)

        # Voter parameters (pydantic-backed)
        vparams_grp = QGroupBox("Voter Parameters")
        vp = QGridLayout(vparams_grp)

        def mkdbl(minv, maxv, step, decimals=3):
            w = QDoubleSpinBox()
            w.setRange(minv, maxv)
            w.setSingleStep(step)
            w.setDecimals(decimals)
            w.setAlignment(Qt.AlignRight)
            return w

        self.ds_conf_thresh = mkdbl(0.0, 1.0, 0.01)
        self.ds_solo_strong = mkdbl(0.0, 1.0, 0.01)
        self.ds_iou_thresh = mkdbl(0.0, 1.0, 0.01)
        self.ds_f1_margin = mkdbl(0.0, 1.0, 0.01)
        self.ds_gamma = mkdbl(0.1, 10.0, 0.1, decimals=2)
        self.ds_near_tie_conf = mkdbl(0.0, 1.0, 0.01)
        self.cb_fuse_coords = QCheckBox("Fuse coordinates")
        self.cb_use_f1 = QCheckBox("Use F1 weights")
        self.btn_voter_defaults = QToolButton()
        self.btn_voter_defaults.setText("Reset")

        row = 0
        vp.addWidget(QLabel("Base conf (conf_thresh):"), row, 0); vp.addWidget(self.ds_conf_thresh, row, 1); row += 1
        vp.addWidget(QLabel("Solo strong (solo_strong):"), row, 0); vp.addWidget(self.ds_solo_strong, row, 1); row += 1
        vp.addWidget(QLabel("IoU threshold (iou_thresh):"), row, 0); vp.addWidget(self.ds_iou_thresh, row, 1); row += 1
        vp.addWidget(QLabel("F1 margin (f1_margin):"), row, 0); vp.addWidget(self.ds_f1_margin, row, 1); row += 1
        vp.addWidget(QLabel("Gamma (gamma):"), row, 0); vp.addWidget(self.ds_gamma, row, 1); row += 1
        vp.addWidget(QLabel("Near-tie conf (near_tie_conf):"), row, 0); vp.addWidget(self.ds_near_tie_conf, row, 1); row += 1
        vp.addWidget(self.cb_fuse_coords, row, 1); vp.addWidget(self.btn_voter_defaults, row, 2); row += 1
        vp.addWidget(self.cb_use_f1, row, 1); row += 1

        # Model thresholds (dynamic)
        model_grp = QGroupBox("Model Thresholds")
        mg = QGridLayout(model_grp)
        self.model_conf_spins: Dict[str, QDoubleSpinBox] = {}

        self.model_conf_spins[self.model_a] = mkdbl(0.0, 1.0, 0.01)
        self.model_conf_spins[self.model_b] = mkdbl(0.0, 1.0, 0.01)

        mg.addWidget(QLabel(f"{self.model_a} conf_threshold:"), 0, 0)
        mg.addWidget(self.model_conf_spins[self.model_a], 0, 1)

        mg.addWidget(QLabel(f"{self.model_b} conf_threshold:"), 1, 0)
        mg.addWidget(self.model_conf_spins[self.model_b], 1, 1)

        # assemble
        root.addWidget(src_grp)
        root.addWidget(perf_grp)
        root.addWidget(cfg_path_grp)
        root.addWidget(cfg_grp)
        root.addWidget(model_grp)
        root.addWidget(vparams_grp)

        # ----------------------------
        # Voter Info Box
        # ----------------------------
        info = QTextEdit()
        info.setReadOnly(True)
        info.setFixedHeight(110)
        info.setStyleSheet(
            """
            QTextEdit {
                background: #1a1a1a;
                border: 1px solid #333;
                border-radius: 8px;
                color: #e0e0e0;
                font-size: 12px;
                padding: 8px;
            }
            """
        )

        info.setText(
            "Voter Visualization:\n\n"
            "• Grey boxes: All candidate detections from both models.\n"
            "  These are shown for transparency and debugging.\n\n"
            "• Yellow boxes: Final detections selected by the voter.\n"
            "  These passed IoU, confidence, and F1-based rules."
        )

        root.addWidget(info)

        # ----------------------------
        # Save Configs button (moved)
        # ----------------------------
        self.btn_save = QPushButton("Save Configs")
        self.btn_save.setMinimumHeight(38)
        self.btn_save.setStyleSheet(
            """
            QPushButton {
                background: #4caf50;
                color: white;
                font-weight: 700;
                border-radius: 8px;
                font-size: 14px;
                padding: 10px;
            }
            QPushButton:hover {
                background: #43a047;
            }
            """
        )
        root.addWidget(self.btn_save)
        root.addStretch(1)

        self.setStyleSheet(
            """
            QDockWidget#SettingsDock { background: #121212; color: #f2f2f2; }
            QGroupBox { border: 1px solid #2e2e2e; border-radius: 8px; margin-top: 12px; color: #eaeaea; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #cfcfcf; }
            QLabel { color: #e8e8e8; }
            QLineEdit { background: #1d1d1d; color: #f0f0f0; border: 1px solid #3a3a3a; border-radius: 6px; padding: 6px; }
            QSpinBox { background: #1d1d1d; color: #f0f0f0; border: 1px solid #3a3a3a; border-radius: 6px; padding: 4px; }
            QDoubleSpinBox { background: #1d1d1d; color: #f0f0f0; border: 1px solid #3a3a3a; border-radius: 6px; padding: 4px; }
            QCheckBox { color: #e8e8e8; }
            QTableWidget { background: #161616; color: #f0f0f0; gridline-color: #333; }
            QHeaderView::section { background: #202020; color: #f0f0f0; border: none; padding: 6px; }
            QPushButton, QToolButton { background: #2e7dff; color: white; border: none; padding: 6px 10px; border-radius: 6px; }
            QRadioButton { color: #e8e8e8; }
            QRadioButton::indicator { width: 16px; height: 16px; border-radius: 8px; border: 2px solid #6b7a86; background: #1d1d1d; }
            QRadioButton::indicator:checked { background: #2e7dff; border: 2px solid #9fb8ff; }
            """
        )

        # connect UI -> parent slots
        self.btn_browse.clicked.connect(parent._browse_file)
        self.btn_cfg_browse.clicked.connect(parent._browse_config)
        self.btn_add.clicked.connect(parent._cfg_add_row)
        self.btn_del.clicked.connect(parent._cfg_delete_rows)
        self.btn_reload.clicked.connect(parent._cfg_reload_from_disk)
        self.btn_save.clicked.connect(parent._cfg_save_to_disk)

        self.le_file.textChanged.connect(lambda s: parent._on_file_changed(s))
        self.le_stream.textChanged.connect(lambda s: parent._on_stream_changed(s))
        self.sb_stride.valueChanged.connect(lambda v: parent._on_stride_changed(v))
        self.le_cfg_path.textChanged.connect(lambda s: parent._on_cfg_path_changed(s))
        self.rb_file.toggled.connect(lambda v: (parent._on_source_kind_changed("file") if v else None))
        self.rb_stream.toggled.connect(lambda v: (parent._on_source_kind_changed("stream") if v else None))

        # voter params
        self.ds_conf_thresh.valueChanged.connect(lambda v: parent._on_voter_param_changed("conf_thresh", float(v)))
        self.ds_solo_strong.valueChanged.connect(lambda v: parent._on_voter_param_changed("solo_strong", float(v)))
        self.ds_iou_thresh.valueChanged.connect(lambda v: parent._on_voter_param_changed("iou_thresh", float(v)))
        self.ds_f1_margin.valueChanged.connect(lambda v: parent._on_voter_param_changed("f1_margin", float(v)))
        self.ds_gamma.valueChanged.connect(lambda v: parent._on_voter_param_changed("gamma", float(v)))
        self.ds_near_tie_conf.valueChanged.connect(lambda v: parent._on_voter_param_changed("near_tie_conf", float(v)))
        self.cb_fuse_coords.toggled.connect(lambda v: parent._on_voter_param_changed("fuse_coords", bool(v)))
        self.cb_use_f1.toggled.connect(lambda v: parent._on_voter_param_changed("use_f1", bool(v)))
        self.btn_voter_defaults.clicked.connect(parent._reset_voter_params)

        # model conf thresholds (generic)
        self.model_conf_spins[self.model_a].valueChanged.connect(lambda v: parent._on_model_conf_changed(self.model_a, float(v)))
        self.model_conf_spins[self.model_b].valueChanged.connect(lambda v: parent._on_model_conf_changed(self.model_b, float(v)))

    def set_source_mode(self, mode: str) -> None:
        if mode == "file":
            self.rb_file.setChecked(True)
            self.le_file.setEnabled(True)
            self.btn_browse.setEnabled(True)
            self.le_stream.setEnabled(False)
        else:
            self.rb_stream.setChecked(True)
            self.le_file.setEnabled(False)
            self.btn_browse.setEnabled(False)
            self.le_stream.setEnabled(True)


# ----------------------------
# Main window
# ----------------------------

class VideoInferenceWindow(QMainWindow):
    """
       Class Goals:
       - Two detector names discovered from config dynamically
       - No yolov7/frcnn naming anywhere
       - Config is represented by Pydantic voter params (VoterParams)
       - GUI updates voter + model thresholds + f1 scores, then persists to YAML
       - Uses TwoModelVoter (stateful) that remembers model names + params
    """
    def __init__(self, config_path: str) -> None:
        super().__init__()
        self.setWindowTitle("BoardVision – Live Inference (Generic Two-Model)")
        self.resize(1350, 800)

        # video state
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running: bool = False
        self.frame_index: int = 0

        # ---- Cached images for resize re-render ----
        self._last_img_a: Optional[np.ndarray] = None
        self._last_img_b: Optional[np.ndarray] = None
        self._last_img_voter: Optional[np.ndarray] = None

        # source / perf
        self.video_path: str = ""
        self.stream_spec: str = "0"
        self.frame_stride: int = 1
        self.config_path: str = config_path
        self.source_mode: str = "stream"

        # load yaml
        self.cfg_raw: Dict[str, Any] = self._load_yaml_config(self.config_path)

        # device / provider summary (kept)
        pref = str((self.cfg_raw.get("device") or {}).get("preferred", "cuda")).lower()
        gpu_id = int((self.cfg_raw.get("device") or {}).get("gpu_id", 0))
        provs = available_providers()
        if pref == "cuda" and "CUDAExecutionProvider" in provs:
            self.device_str = f"cuda:{gpu_id}"
        else:
            self.device_str = "cpu"

        # engine
        LOGGER.info("Loading detectors (device=%s) …", self.device_str)
        self.detector_engine = MultiDetectorEngine(self.cfg_raw)  # provides .detectors and .run()

        # pick EXACTLY TWO detectors (first two enabled in YAML order)
        self.detector_names: List[str] = list(self.detector_engine.detectors.keys())[:2]
        if len(self.detector_names) != 2:
            raise RuntimeError("Config must enable exactly two detectors for this GUI refactor.")
        self.model_a: str = self.detector_names[0]
        self.model_b: str = self.detector_names[1]

        # pydantic voter params from YAML
        self.voter_params: VoterParams = self._load_voter_params(self.cfg_raw)

        # f1 scores from YAML (keys must match model names)
        self.f1_scores: Dict[str, Dict[str, float]] = self._load_f1_scores(self.cfg_raw)

        # model conf thresholds from YAML (generic)
        self.model_conf: Dict[str, float] = {
            self.model_a: float(_deep_get(self.cfg_raw, ["detectors", self.model_a, "conf_threshold"], 0.25)),
            self.model_b: float(_deep_get(self.cfg_raw, ["detectors", self.model_b, "conf_threshold"], 0.25)),
        }

        # stateful voter
        self.voter = TwoModelVoter(
            model_a=self.model_a,
            model_b=self.model_b,
            f1_scores=self.f1_scores,
            params=self.voter_params,
        )

        # stats (generic)
        self.stats_loop = RollingStats(120)
        self.stats_model = {self.model_a: RollingStats(120), self.model_b: RollingStats(120)}
        self.stats_decision = RollingStats(120)

        # build UI (dynamic model labels)
        self._build_ui()

        # timer
        self.timer = QTimer(self)
        self.timer.setInterval(0)
        self.timer.timeout.connect(self._process_next_frame)

        self._install_contrast_palette()
        self.statusBar().showMessage(f"Device: {self.device_str} | Providers: {', '.join(provs)}")

        # initialize UI from config
        self._sync_settings_to_ui()

    # ----------------------------
    # YAML + config helpers
    # ----------------------------
    def resizeEvent(self, event) -> None:
        """
        Qt resize hook.
        Re-render images at the new widget size for crisp visuals.
        """
        super().resizeEvent(event)
        self._rerender_cached_images()


    def _load_yaml_config(self, path: str) -> dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            LOGGER.error("Failed to read YAML config '%s': %s", path, e)
            return {}

    def _save_yaml_config(self, path: str, cfg: dict) -> None:
        _ensure_dir(path)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=True)

    def _load_f1_scores(self, cfg: dict) -> Dict[str, Dict[str, float]]:
        """
          Expects cfg["f1_scores"][label][<model_name>] floats
          - We only care about the TWO active models, but we preserve other keys if present.
        """
        f1 = cfg.get("f1_scores")
        out: Dict[str, Dict[str, float]] = {}
        if isinstance(f1, dict):
            for label, v in f1.items():
                if isinstance(v, dict):
                    out[str(label)] = {str(k): float(vv) for k, vv in v.items()}
        return out

    def _load_voter_params(self, cfg: dict) -> VoterParams:
        """
        Pydantic controls defaults + validation.
        """
        voter_cfg = cfg.get("voter", {}) or {}
        try:
            return VoterParams(**voter_cfg)
        except Exception as e:
            LOGGER.warning("Invalid voter config in YAML; using defaults. Error=%s", e)
            return VoterParams()

    # ----------------------------
    # UI
    # ----------------------------

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(8)

        cols = QHBoxLayout()
        cols.setSpacing(12)

        # dynamic panels
        self.panel_a = Panel(f"Detector: {self.model_a}")
        self.panel_voter = Panel("Decision (Voter)")
        self.panel_b = Panel(f"Detector: {self.model_b}")

        cols.addWidget(self.panel_a, 1)
        cols.addWidget(self.panel_voter, 1)
        cols.addWidget(self.panel_b, 1)
        root.addLayout(cols, 1)

        # settings dock
        self.settings_dock = SettingsDock(self, self.model_a, self.model_b)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.settings_dock)

        # bind start/stop from dock
        self.btn_start = self.settings_dock.btn_start_left
        self.btn_stop = self.settings_dock.btn_stop_left
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)

        # status bar
        self.setStatusBar(QStatusBar(self))

    def _install_contrast_palette(self) -> None:
        pal = QPalette()
        pal.setColor(QPalette.Window, QColor("#121212"))
        pal.setColor(QPalette.WindowText, QColor("#f2f2f2"))
        pal.setColor(QPalette.Base, QColor("#121212"))
        pal.setColor(QPalette.Text, QColor("#f2f2f2"))
        pal.setColor(QPalette.Button, QColor("#2e7dff"))
        pal.setColor(QPalette.ButtonText, QColor("#f2f2f2"))
        self.setPalette(pal)

    def _sync_settings_to_ui(self) -> None:
        # source + perf
        self.settings_dock.le_cfg_path.setText(self.config_path)
        self.settings_dock.sb_stride.setValue(int(self.frame_stride))
        self.settings_dock.set_source_mode(self.source_mode)
        self.settings_dock.le_file.setText(self.video_path)
        self.settings_dock.le_stream.setText(self.stream_spec)

        # voter params
        vp = self.voter_params
        self.settings_dock.ds_conf_thresh.setValue(float(vp.conf_thresh))
        self.settings_dock.ds_solo_strong.setValue(float(vp.solo_strong))
        self.settings_dock.ds_iou_thresh.setValue(float(vp.iou_thresh))
        self.settings_dock.ds_f1_margin.setValue(float(vp.f1_margin))
        self.settings_dock.ds_gamma.setValue(float(vp.gamma))
        self.settings_dock.ds_near_tie_conf.setValue(float(vp.near_tie_conf))
        self.settings_dock.cb_fuse_coords.setChecked(bool(vp.fuse_coords))
        self.settings_dock.cb_use_f1.setChecked(bool(vp.use_f1))

        # model thresholds
        self.settings_dock.model_conf_spins[self.model_a].setValue(float(self.model_conf[self.model_a]))
        self.settings_dock.model_conf_spins[self.model_b].setValue(float(self.model_conf[self.model_b]))

        # f1 table
        self._cfg_populate_table(self.f1_scores)

        # start/stop enabled state
        self.btn_start.setEnabled(not self.is_running)
        self.btn_stop.setEnabled(self.is_running)

    # ----------------------------
    # File dialogs
    # ----------------------------

    def _file_dialog_options(self):
        return QFileDialog.Options()

    def _browse_file(self) -> None:
        media_filter = "Media (*.mp4 *.mov *.avi *.mkv *.webm *.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"
        dlg = QFileDialog(self, "Select Video or Image")
        dlg.setOptions(self._file_dialog_options())
        dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setNameFilters([media_filter, "All Files (*)"])
        dlg.selectNameFilter(media_filter)

        if dlg.exec():
            path = dlg.selectedFiles()[0]
            if path:
                self.settings_dock.le_file.setText(path)
                self.statusBar().showMessage(f"Selected: {os.path.basename(path)} | Device: {self.device_str}")

    def _browse_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Config YAML",
            "",
            "YAML (*.yaml *.yml);All Files (*)",
            options=self._file_dialog_options(),
        )
        if not path:
            return

        # reload config
        try:
            self.config_path = path
            self.cfg_raw = self._load_yaml_config(path)
            self.detector_engine = MultiDetectorEngine(self.cfg_raw)

            self.detector_names = list(self.detector_engine.detectors.keys())[:2]
            if len(self.detector_names) != 2:
                raise RuntimeError("Config must enable exactly two detectors for this GUI.")

            # if model names changed, easiest is recreate the whole window
            new_a, new_b = self.detector_names[0], self.detector_names[1]
            if (new_a != self.model_a) or (new_b != self.model_b):
                QMessageBox.information(
                    self,
                    "Config Loaded",
                    "Detector names changed. Restarting window to rebuild dynamic UI.",
                )
                self.close()
                # recreate
                w = VideoInferenceWindow(path)
                w.show()
                return

            self.voter_params = self._load_voter_params(self.cfg_raw)
            self.f1_scores = self._load_f1_scores(self.cfg_raw)

            self.model_conf[self.model_a] = float(_deep_get(self.cfg_raw, ["detectors", self.model_a, "conf_threshold"], self.model_conf[self.model_a]))
            self.model_conf[self.model_b] = float(_deep_get(self.cfg_raw, ["detectors", self.model_b, "conf_threshold"], self.model_conf[self.model_b]))

            # update voter instance with new config
            self.voter = TwoModelVoter(self.model_a, self.model_b, self.f1_scores, self.voter_params)

            self._sync_settings_to_ui()
            self.statusBar().showMessage(f"Loaded config: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Config Error", f"Failed to load config:\n{e}")

    # ----------------------------
    # F1 table (yaml)
    # ----------------------------

    def _cfg_populate_table(self, f1_scores: dict) -> None:
        """
          - Uses columns [Label, model_a, model_b]
          - Reads per-label f1_scores[label][model_a/model_b]
        """
        tbl = self.settings_dock.tbl_cfg
        tbl.setRowCount(0)
        for label, model_dict in (f1_scores or {}).items():
            row = tbl.rowCount()
            tbl.insertRow(row)
            tbl.setItem(row, 0, QTableWidgetItem(str(label)))
            tbl.setItem(row, 1, QTableWidgetItem(str((model_dict or {}).get(self.model_a, ""))))
            tbl.setItem(row, 2, QTableWidgetItem(str((model_dict or {}).get(self.model_b, ""))))

    def _cfg_extract_from_table(self) -> dict:
        """
          - Extracts into cfg["f1_scores"][label][model_name]
        """
        tbl = self.settings_dock.tbl_cfg
        out: Dict[str, Dict[str, float]] = {}

        for r in range(tbl.rowCount()):
            label_item = tbl.item(r, 0)
            a_item = tbl.item(r, 1)
            b_item = tbl.item(r, 2)

            if not label_item:
                continue
            label = label_item.text().strip()
            if not label:
                continue

            def parse_cell(item, col_name: str) -> Optional[float]:
                if item is None:
                    return None
                txt = (item.text() or "").strip()
                if txt == "":
                    return None
                try:
                    return float(txt)
                except Exception:
                    raise ValueError(f"Row {r+1}: {col_name} must be a number")

            av = parse_cell(a_item, self.model_a)
            bv = parse_cell(b_item, self.model_b)

            out[label] = {}
            if av is not None:
                out[label][self.model_a] = av
            if bv is not None:
                out[label][self.model_b] = bv

        return out

    def _cfg_add_row(self) -> None:
        tbl = self.settings_dock.tbl_cfg
        row = tbl.rowCount()
        tbl.insertRow(row)
        tbl.setItem(row, 0, QTableWidgetItem("New_Label"))
        tbl.setItem(row, 1, QTableWidgetItem("0.500"))
        tbl.setItem(row, 2, QTableWidgetItem("0.500"))

    def _cfg_delete_rows(self) -> None:
        tbl = self.settings_dock.tbl_cfg
        rows = sorted({i.row() for i in tbl.selectedIndexes()}, reverse=True)
        for r in rows:
            tbl.removeRow(r)

    def _cfg_reload_from_disk(self) -> None:
        try:
            self.cfg_raw = self._load_yaml_config(self.config_path)
            self.f1_scores = self._load_f1_scores(self.cfg_raw)
            self.voter_params = self._load_voter_params(self.cfg_raw)
            # refresh voter object
            self.voter = TwoModelVoter(self.model_a, self.model_b, self.f1_scores, self.voter_params)
        except Exception as e:
            QMessageBox.critical(self, "Config Error", f"Failed to read config:\n{e}")
            return
        self._sync_settings_to_ui()
        self.statusBar().showMessage("Reloaded config from disk.")

    def _cfg_save_to_disk(self) -> None:
        try:
            f1_new = self._cfg_extract_from_table()
        except Exception as e:
            QMessageBox.critical(self, "Validation Error", str(e))
            return

        # update in-memory yaml from pydantic + UI values
        self.f1_scores = f1_new
        self.voter.f1_scores = self.f1_scores  # keep voter in sync

        self._write_back_config_to_yaml()

        try:
            self._save_yaml_config(self.config_path, self.cfg_raw)
            self.statusBar().showMessage("Saved config.yaml to disk.")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save YAML:\n{e}")

    # ----------------------------
    # Settings changed handlers
    # ----------------------------

    def _on_file_changed(self, s: str) -> None:
        self.video_path = s

    def _on_stream_changed(self, s: str) -> None:
        self.stream_spec = s

    def _on_stride_changed(self, v: int) -> None:
        self.frame_stride = int(v)

    def _on_cfg_path_changed(self, s: str) -> None:
        # keep but do not auto-load; browse button loads
        self.config_path = s

    def _on_source_kind_changed(self, mode: str) -> None:
        self.source_mode = mode
        self.settings_dock.set_source_mode(mode)

    def _on_voter_param_changed(self, key: str, value: Any) -> None:
        """
           Updates pydantic voter params + voter instance
           No direct primitive member variables
        """
        try:
            data = self.voter_params.model_dump()
            data[key] = value
            self.voter_params = VoterParams(**data)  # validate
            self.voter.params = self.voter_params
        except Exception as e:
            LOGGER.warning("Rejected voter param update: %s=%r (%s)", key, value, e)

    def _reset_voter_params(self) -> None:
        self.voter_params = VoterParams()
        self.voter.params = self.voter_params
        self._sync_settings_to_ui()

    def _on_model_conf_changed(self, model_name: str, value: float) -> None:
        """
          Generic model threshold update
          Updates engine runtime threshold + local state
        """
        self.model_conf[model_name] = float(value)
        self.detector_engine.set_detector_conf(model_name, float(value))

    # ----------------------------
    # Persist changes back to YAML
    # ----------------------------

    def _write_back_config_to_yaml(self) -> None:
        """
          Writes current GUI state back into self.cfg_raw (original YAML dict)
          - Preserves unknown keys/sections (does not overwrite whole file)
        """
        # voter section (pydantic → dict)
        self.cfg_raw["voter"] = self.voter_params.model_dump()

        # model conf thresholds
        _deep_set(self.cfg_raw, ["detectors", self.model_a, "conf_threshold"], float(self.model_conf[self.model_a]))
        _deep_set(self.cfg_raw, ["detectors", self.model_b, "conf_threshold"], float(self.model_conf[self.model_b]))

        # f1 scores
        self.cfg_raw["f1_scores"] = self.f1_scores

    # ----------------------------
    # Start/Stop
    # ----------------------------

    def start(self) -> None:
        if self.is_running:
            return

        # FORCE source mode from UI (important)
        if self.settings_dock.rb_file.isChecked():
            self.source_mode = "file"
        elif self.settings_dock.rb_stream.isChecked():
            self.source_mode = "stream"

        # FIX: IMAGE FILES SHOULD NOT OPEN VideoCapture
        if self.source_mode == "file" and self._is_image_path(self.video_path):
            if not os.path.isfile(self.video_path):
                QMessageBox.critical(self, "File Error", "Image file does not exist.")
                return

            self.is_running = True
            self.frame_index = 0
            self.timer.start()

            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)

            self.statusBar().showMessage(
                f"Image mode | Models: {self.model_a}, {self.model_b}"
            )
            return  # 🚨 CRITICAL: do NOT open VideoCapture

        # ---------- normal video / stream ----------
        self._open_capture()
        if self.cap is None or not self.cap.isOpened():
            QMessageBox.critical(self, "Capture Error", "Failed to open video source.")
            return

        self.is_running = True
        self.frame_index = 0
        self.stats_loop = RollingStats(120)
        self.stats_model = {self.model_a: RollingStats(120), self.model_b: RollingStats(120)}
        self.stats_decision = RollingStats(120)

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.timer.start()

        self.statusBar().showMessage(
            f"Running | Models: {self.model_a}, {self.model_b} | Device: {self.device_str}"
        )


    def stop(self) -> None:
        if not self.is_running:
            return
        self.is_running = False
        self.timer.stop()
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.statusBar().showMessage("Stopped.")

    def closeEvent(self, event) -> None:
        self.stop()
        super().closeEvent(event)

    # ----------------------------
    # Capture open logic
    # ----------------------------

    def _is_image_path(self, p: str) -> bool:
        ext = os.path.splitext(p)[1].lower()
        return ext in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    def _open_capture(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        if self.source_mode == "file":
            p = (self.video_path or "").strip()
            if not p:
                self.cap = None
                return
            if self._is_image_path(p):
                # image "capture" handled by one-shot in process loop
                self.cap = None
                return
            self.cap = cv2.VideoCapture(p)
        else:
            spec = (self.stream_spec or "0").strip()
            # integer camera index?
            if spec.isdigit():
                self.cap = cv2.VideoCapture(int(spec))
            else:
                self.cap = cv2.VideoCapture(spec)

        if self.cap is not None and self.cap.isOpened():
            safe_cap_set(self.cap, cv2.CAP_PROP_BUFFERSIZE, 1)

    # ----------------------------
    # Frame processing
    # ----------------------------

    def _display_image(self, label: QLabel, img_bgr: np.ndarray) -> None:
        """
        Correct, single-pass image scaling.
        Proper QImage -> QPixmap conversion.
        """
        if img_bgr is None or img_bgr.size == 0:
            return

        # Convert BGR → RGB
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        # Create QImage (backed by numpy memory)
        qimg = QImage(
            rgb.data,
            w,
            h,
            rgb.strides[0],
            QImage.Format_RGB888,
        )

        # Scale ONCE to QLabel size
        qimg_scaled = qimg.scaled(
            label.width(),
            label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        # ✅ REQUIRED: convert to QPixmap
        pixmap = QPixmap.fromImage(qimg_scaled)

        label.setPixmap(pixmap)




    def _summarize_dets(self, dets: List[Dict[str, Any]]) -> str:
        n = len(dets or [])
        return f"{n} dets"

    def _summarize_voter(self, final_boxes: Any, candidates: Any) -> str:
        def _safe_len(x: Any) -> int:
            try:
                return len(x) if x is not None else 0
            except Exception:
                return 0
        return f"final={_safe_len(final_boxes)} (candidates={_safe_len(candidates)})"

    def _convert_to_pydantic(self, dets: List[Dict[str, Any]], source_default: str) -> List[DetectionModel]:
        out: List[DetectionModel] = []
        for d in dets or []:
            dd = dict(d)
            dd.setdefault("source", source_default)
            try:
                out.append(DetectionModel(**dd))
            except Exception:
                # tolerate bad dets
                continue
        return out

    def _process_next_frame(self) -> None:
        if not self.is_running:
            return

        # image file one-shot
        if self.source_mode == "file" and self._is_image_path(self.video_path):
            try:
                frame_bgr = cv2.imread(self.video_path)
                if frame_bgr is None:
                    raise RuntimeError("Failed to read image.")
                self._run_inference_on_frame(frame_bgr, one_shot=True)
                self.stop()
            except Exception as e:
                QMessageBox.critical(self, "Image Error", str(e))
                self.stop()
            return

        if self.cap is None or not self.cap.isOpened():
            self._open_capture()
            if self.cap is None or not self.cap.isOpened():
                self.stop()
                return

        # stride skip
        self.frame_index += 1
        if (self.frame_index % self.frame_stride) != 0:
            return

        ok, frame_bgr = self.cap.read()
        if not ok or frame_bgr is None:
            self.stop()
            return

        self._run_inference_on_frame(frame_bgr, one_shot=False)

    def _run_inference_on_frame(self, frame_bgr: np.ndarray, one_shot: bool) -> None:
        t0 = time.perf_counter()

        # run engine (dict model->list[dict])
        # ---- per-model timing ----
        det_out = {}
        for name, detector in self.detector_engine.detectors.items():
            t0m = time.perf_counter()
            det_out[name] = detector.infer(frame_bgr.copy())
            dtm = time.perf_counter() - t0m

            if name in self.stats_model:
                self.stats_model[name].add(dtm)


        # extract model outputs (generic)
        a_dets = det_out.get(self.model_a, [])
        b_dets = det_out.get(self.model_b, [])

        # update per-model stats (we can't measure inside engine per-model without timing; keep overall)
        # You can add per-model timers later inside MultiDetectorEngine (future-proof).
        t1 = time.perf_counter()

        # voter expects pydantic dets
        a_pd = self._convert_to_pydantic(a_dets, self.model_a)
        b_pd = self._convert_to_pydantic(b_dets, self.model_b)

        # vote
        t2 = time.perf_counter()
        final_pd, candidates_pd = self.voter.vote(a_pd, b_pd)
        t3 = time.perf_counter()

        # convert final back to dict for drawing
        final_boxes = [d.model_dump() for d in final_pd]
        candidates = [d.model_dump() for d in candidates_pd]

        # render panels
        a_img = draw_boxes(frame_bgr.copy(), a_dets, color=(0, 128, 255))
        b_img = draw_boxes(frame_bgr.copy(), b_dets, color=(255, 128, 0))

        # voter visualization: draw candidates lightly + finals strongly
        voter_img = frame_bgr.copy()
        voter_img = draw_boxes(voter_img, candidates, color=(80, 80, 80))
        voter_img = draw_boxes(voter_img, final_boxes, color=(0, 255, 255))

        #a_img = overlay_label(a_img, f"{self.model_a} (ONNX)", color=(0, 128, 255))
        #b_img = overlay_label(b_img, f"{self.model_b} (ONNX)", color=(255, 128, 0))
        #voter_img = overlay_label(voter_img, "Decision (Voter)", color=(0, 255, 255))
        a_img = overlay_label(a_img, "", color=(0, 128, 255))
        b_img = overlay_label(b_img, "", color=(255, 128, 0))
        voter_img = overlay_label(voter_img, "", color=(0, 255, 255))


        # ---- cache last rendered images (for resize re-render) ----
        self._last_img_a = a_img
        self._last_img_b = b_img
        self._last_img_voter = voter_img

        # display
        self._display_image(self.panel_a.video, self._last_img_a)
        self._display_image(self.panel_b.video, self._last_img_b)
        self._display_image(self.panel_voter.video, self._last_img_voter)


        # logs (include model names)
        idx = self.frame_index
        self.panel_a.append_log(f"#{idx:05d} [{self.model_a}] {self._summarize_dets(a_dets)}")
        self.panel_b.append_log(f"#{idx:05d} [{self.model_b}] {self._summarize_dets(b_dets)}")
        self.panel_voter.append_log(f"#{idx:05d} {self._summarize_voter(final_boxes, candidates)}")

        # stats
        loop_dt = time.perf_counter() - t0
        self.stats_loop.add(loop_dt)
        self.stats_decision.add(t3 - t2)

        self.panel_a.set_stats(
            self.stats_model[self.model_a].avg_ms,
            self.stats_model[self.model_a].fps,
        )

        self.panel_b.set_stats(
            self.stats_model[self.model_b].avg_ms,
            self.stats_model[self.model_b].fps,
        )

        self.panel_voter.set_stats(
            self.stats_decision.avg_ms,
            self.stats_loop.fps,   # voter FPS tied to loop
        )


        # status bar debug
        self.statusBar().showMessage(
            f"Models: {self.model_a}, {self.model_b} | "
            f"Loop: {human_ms(loop_dt)} | Vote: {human_ms(t3-t2)} | "
            f"Stride: {self.frame_stride} | Device: {self.device_str}"
        )

        if one_shot:
            LOGGER.info(
                "Image inference done | models=[%s,%s] vote=%s",
                self.model_a, self.model_b, self._summarize_voter(final_boxes, candidates)
            )

    def _rerender_cached_images(self) -> None:
        """
        Re-render last cached images when GUI resizes.
        Does NOT re-run inference.
        """
        if self._last_img_a is not None:
            self._display_image(self.panel_a.video, self._last_img_a)

        if self._last_img_b is not None:
            self._display_image(self.panel_b.video, self._last_img_b)

        if self._last_img_voter is not None:
            self._display_image(self.panel_voter.video, self._last_img_voter)

# ----------------------------
# CLI
# ----------------------------

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BoardVision ONNX GUI (Generic Two-Model)")
    p.add_argument("--config", required=True, help="Path to config.yaml")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args(sys.argv[1:])
    app = QApplication(sys.argv)
    w = VideoInferenceWindow(args.config)
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
