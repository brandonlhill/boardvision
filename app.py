from __future__ import annotations
import cv2
import numpy as np
import torch
import sys
import os
import json
import time
import logging
import re
from collections import Counter, deque
from typing import Optional, List, Any, Deque, Tuple, Dict

from PySide6.QtCore import Qt, QTimer, QByteArray
from PySide6.QtGui import (
    QImage, QPixmap, QTextCursor, QTextOption, QAction,
    QIcon, QPalette, QColor
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QStatusBar, QFrame,
    QTextEdit, QSizePolicy, QLineEdit, QGroupBox, QGridLayout, QSpinBox,
    QTableWidget, QTableWidgetItem, QAbstractItemView, QToolButton,
    QDockWidget, QRadioButton, QButtonGroup, QDoubleSpinBox, QCheckBox
)

from yolov7.frame_inference import load_yolov7_model, detect_frame
from fasterrcnn.frame_inference import load_fasterrcnn_model, run_fasterrcnn_on_frame
from voter import (
    load_f1_config as default_load_f1_config,
    voter_merge as default_voter_merge,
    draw_voter_boxes_on_frame as default_draw_voter_boxes_on_frame,
    overlay_label as default_overlay_label,
)

from rich.logging import RichHandler

LOGGER = logging.getLogger("boardvision")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

def human_ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f} ms"

def safe_cap_set(cap: cv2.VideoCapture, prop, value) -> None:
    try:
        cap.set(prop, value)
    except Exception:
        pass

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
        outer = QVBoxLayout(self); outer.setContentsMargins(12, 10, 12, 10); outer.setSpacing(6)
        self.title = QLabel(title); self.title.setObjectName("Title"); self.title.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.stats = QLabel("-"); self.stats.setObjectName("Stats"); self.stats.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.video = QLabel(); self.video.setObjectName("Video"); self.video.setAlignment(Qt.AlignCenter)
        self.video.setScaledContents(False); self.video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setFixedHeight(140)
        self.log.setLineWrapMode(QTextEdit.NoWrap); self.log.setWordWrapMode(QTextOption.NoWrap); self.log.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.log.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        outer.addWidget(self.title); outer.addWidget(self.stats); outer.addWidget(self.video, 1); outer.addWidget(self.log, 0)
    def set_stats(self, ms: float, fps: float) -> None:
        self.stats.setText("-" if (ms <= 0 and fps <= 0) else f"{ms:.1f} ms  |  {fps:.1f} FPS")
    def append_log(self, text: str) -> None:
        self.log.append(text); self.log.moveCursor(QTextCursor.End)

class SettingsDock(QDockWidget):
    def __init__(self, parent: "VideoInferenceWindow") -> None:
        super().__init__("", parent)  # dock without title bar
        self.setObjectName("SettingsDock")
        self.setAllowedAreas(Qt.LeftDockWidgetArea)
        self.setFeatures(QDockWidget.NoDockWidgetFeatures)

        w = QWidget(self); self.setWidget(w)
        root = QVBoxLayout(w); root.setContentsMargins(8, 8, 8, 8); root.setSpacing(10)

        # Controls
        controls_grp = QGroupBox("Controls"); cl = QHBoxLayout(controls_grp)
        self.btn_start_left = QPushButton("Start"); self.btn_stop_left = QPushButton("Stop")
        for b, clr in ((self.btn_start_left, "#2e7dff"), (self.btn_stop_left, "#e23c3c")):
            b.setMinimumHeight(34)
            b.setStyleSheet(f"QPushButton {{ background: {clr}; color: white; border: none; padding: 8px 16px; border-radius: 8px; font-size: 14px; }} QPushButton:disabled {{ background: #555; }}")
        cl.addWidget(self.btn_start_left); cl.addWidget(self.btn_stop_left)
        root.addWidget(controls_grp)

        # Source
        src_grp = QGroupBox("Source"); gl = QGridLayout(src_grp)
        self.rb_file = QRadioButton("File")
        self.rb_stream = QRadioButton("Stream")
        self.bg_source = QButtonGroup(self); self.bg_source.setExclusive(True)
        self.bg_source.addButton(self.rb_file); self.bg_source.addButton(self.rb_stream)
        gl.addWidget(self.rb_file, 0, 0); gl.addWidget(self.rb_stream, 0, 1, 1, 2)
        self.le_file = QLineEdit(); self.le_file.setPlaceholderText("Video file path…")
        self.btn_browse = QToolButton(); self.btn_browse.setText("Pick")
        self.le_stream = QLineEdit(); self.le_stream.setPlaceholderText("Camera index (e.g., 0) or RTSP/HTTP URL")
        gl.addWidget(QLabel("File:"),    1, 0); gl.addWidget(self.le_file, 1, 1); gl.addWidget(self.btn_browse, 1, 2)
        gl.addWidget(QLabel("Stream:"),  2, 0); gl.addWidget(self.le_stream, 2, 1, 1, 2)

        # Performance
        perf_grp = QGroupBox("Performance"); pl = QGridLayout(perf_grp)
        self.sb_stride = QSpinBox(); self.sb_stride.setRange(1, 8); self.sb_stride.setValue(1)
        pl.addWidget(QLabel("Stride (process every Nth frame):"), 0, 0); pl.addWidget(self.sb_stride, 0, 1)

        # Config path
        cfg_path_grp = QGroupBox("Config File"); cl2 = QHBoxLayout(cfg_path_grp)
        self.le_cfg_path = QLineEdit(); self.le_cfg_path.setPlaceholderText("config.json")
        self.btn_cfg_browse = QToolButton(); self.btn_cfg_browse.setText("Pick")
        cl2.addWidget(self.le_cfg_path, 1); cl2.addWidget(self.btn_cfg_browse, 0)

        # Class Thresholds
        cfg_grp = QGroupBox("Class Thresholds"); tbl_layout = QVBoxLayout(cfg_grp)
        self.tbl_cfg = QTableWidget(0, 3)
        self.tbl_cfg.setHorizontalHeaderLabels(["Label", "YOLOv7", "FRCNN"])
        self.tbl_cfg.horizontalHeader().setStretchLastSection(True)
        self.tbl_cfg.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_cfg.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.SelectedClicked | QAbstractItemView.EditKeyPressed)
        self.tbl_cfg.verticalHeader().setVisible(False)
        row_btns = QHBoxLayout()
        self.btn_add = QPushButton("Add Row")
        self.btn_del = QPushButton("Delete Selected")
        self.btn_reload = QPushButton("Reload")
        self.btn_save = QPushButton("Save")
        row_btns.addWidget(self.btn_add); row_btns.addWidget(self.btn_del); row_btns.addStretch(1); row_btns.addWidget(self.btn_reload); row_btns.addWidget(self.btn_save)
        tbl_layout.addWidget(self.tbl_cfg); tbl_layout.addLayout(row_btns)

        # Voter Parameters
        vparams_grp = QGroupBox("Voter Parameters"); vp = QGridLayout(vparams_grp)
        def mkdbl(minv, maxv, step, decimals=3):
            w = QDoubleSpinBox(); w.setRange(minv, maxv); w.setSingleStep(step); w.setDecimals(decimals)
            w.setAlignment(Qt.AlignRight)
            return w
        self.ds_conf_thresh  = mkdbl(0.0, 1.0, 0.01)
        self.ds_solo_strong  = mkdbl(0.0, 1.0, 0.01)
        self.ds_iou_thresh   = mkdbl(0.0, 1.0, 0.01)
        self.ds_f1_margin    = mkdbl(0.0, 1.0, 0.01)
        self.ds_gamma        = mkdbl(0.1, 5.0, 0.1, decimals=2)
        self.cb_fuse_coords  = QCheckBox("Fuse coordinates")
        self.btn_voter_defaults = QToolButton(); self.btn_voter_defaults.setText("Reset")
        row = 0
        vp.addWidget(QLabel("Base conf (conf_thresh):"), row, 0); vp.addWidget(self.ds_conf_thresh, row, 1); row += 1
        vp.addWidget(QLabel("Solo strong (solo_strong):"), row, 0); vp.addWidget(self.ds_solo_strong, row, 1); row += 1
        vp.addWidget(QLabel("IoU threshold (iou_thresh):"), row, 0); vp.addWidget(self.ds_iou_thresh, row, 1); row += 1
        vp.addWidget(QLabel("F1 margin (f1_margin):"), row, 0); vp.addWidget(self.ds_f1_margin, row, 1); row += 1
        vp.addWidget(QLabel("Gamma (gamma):"), row, 0); vp.addWidget(self.ds_gamma, row, 1); row += 1
        vp.addWidget(self.cb_fuse_coords, row, 1); vp.addWidget(self.btn_voter_defaults, row, 2); row += 1

        # Model Thresholds
        model_grp = QGroupBox("Model Thresholds"); mg = QGridLayout(model_grp)
        self.ds_yolo_conf  = mkdbl(0.0, 1.0, 0.01)
        self.ds_frcnn_conf = mkdbl(0.0, 1.0, 0.01)
        mg.addWidget(QLabel("YOLOv7 conf_thresh:"), 0, 0); mg.addWidget(self.ds_yolo_conf,  0, 1)
        mg.addWidget(QLabel("Faster R-CNN conf_thresh:"), 1, 0); mg.addWidget(self.ds_frcnn_conf, 1, 1)

        # Assemble dock
        root.addWidget(src_grp)
        root.addWidget(perf_grp)
        root.addWidget(cfg_path_grp)
        root.addWidget(cfg_grp)
        root.addWidget(model_grp)
        root.addWidget(vparams_grp)
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
            QPushButton#danger { background: #e23c3c; }
            QRadioButton { color: #e8e8e8; }
            QRadioButton::indicator { width: 16px; height: 16px; border-radius: 8px; border: 2px solid #6b7a86; background: #1d1d1d; }
            QRadioButton::indicator:checked { background: #2e7dff; border: 2px solid #9fb8ff; }
            """
        )

        # Connections
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

        # Voter parameter signals
        self.ds_conf_thresh.valueChanged.connect(lambda v: parent._on_voter_param_changed("conf_thresh", float(v)))
        self.ds_solo_strong.valueChanged.connect(lambda v: parent._on_voter_param_changed("solo_strong", float(v)))
        self.ds_iou_thresh.valueChanged.connect(lambda v: parent._on_voter_param_changed("iou_thresh", float(v)))
        self.ds_f1_margin.valueChanged.connect(lambda v: parent._on_voter_param_changed("f1_margin", float(v)))
        self.ds_gamma.valueChanged.connect(lambda v: parent._on_voter_param_changed("gamma", float(v)))
        self.cb_fuse_coords.toggled.connect(lambda v: parent._on_voter_param_changed("fuse_coords", bool(v)))
        self.btn_voter_defaults.clicked.connect(parent._reset_voter_params)

        # Model conf signals
        self.ds_yolo_conf.valueChanged.connect(lambda v: parent._on_model_conf_changed("yolov7", float(v)))
        self.ds_frcnn_conf.valueChanged.connect(lambda v: parent._on_model_conf_changed("frcnn", float(v)))

    def set_source_mode(self, mode: str) -> None:
        if mode == "file":
            self.rb_file.setChecked(True)
            self.le_file.setEnabled(True); self.btn_browse.setEnabled(True)
            self.le_stream.setEnabled(False)
        else:
            self.rb_stream.setChecked(True)
            self.le_file.setEnabled(False); self.btn_browse.setEnabled(False)
            self.le_stream.setEnabled(True)


class VideoInferenceWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BoardVision – Live Inference")
        self.resize(1350, 800)

        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running: bool = False
        self.frame_index: int = 0

        self.video_path: str = ""
        self.stream_spec: str = "0"
        self.frame_stride: int = 1
        self.config_path: str = "config.json"
        self.source_mode: str = "stream"

        # Voter parameters (GUI-adjustable)
        self.v_conf_thresh = 0.50
        self.v_solo_strong = 0.95
        self.v_iou_thresh  = 0.40
        self.v_f1_margin   = 0.05
        self.v_gamma       = 1.50
        self.v_fuse_coords = True

        # Model-level thresholds (GUI-adjustable)
        self.m_yolo_conf  = 0.25
        self.m_frcnn_conf = 0.50

        self.stats_loop = RollingStats(120)
        self.stats_yolo = RollingStats(120)
        self.stats_frcnn = RollingStats(120)
        self.stats_decision = RollingStats(120)

        LOGGER.info("Loading models (device=%s) …", self.device)
        self.yolov7_model = load_yolov7_model("weights/yolov7.pt", device=self.device)
        self.frcnn_model, self.frcnn_classes = load_fasterrcnn_model("weights/fasterrcnn.pth", device=self.device)

        self.cfg_raw: Dict[str, Any] = self._load_config_file(self.config_path)
        self.cfg_thresholds, _ = self._split_thresholds_and_meta(self.cfg_raw)

        # Bind voter functions (no external import option)
        self.fn_voter_merge = default_voter_merge
        self.fn_draw_voter_boxes_on_frame = default_draw_voter_boxes_on_frame
        self.fn_overlay_label = default_overlay_label

        self._load_qsettings()
        self._build_ui()
        self._connect_signals()
        self._sync_settings_to_ui()

        self.timer = QTimer(self); self.timer.setInterval(0); self.timer.timeout.connect(self._process_next_frame)
        self._install_contrast_palette()
        self.statusBar().showMessage(f"Device: {self.device}")

    def _build_ui(self) -> None:
        central = QWidget(self); self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setContentsMargins(10, 8, 10, 8); root.setSpacing(8)

        cols = QHBoxLayout(); cols.setSpacing(12)
        self.panel_yolo = Panel("YOLOv7")
        self.panel_voter = Panel("Decision (YOLO+FRCNN+Voter)")
        self.panel_frcnn = Panel("Faster R-CNN")
        cols.addWidget(self.panel_yolo, 1)
        cols.addWidget(self.panel_voter, 1)
        cols.addWidget(self.panel_frcnn, 1)

        root.addLayout(cols, 1)

        self.settings_dock = SettingsDock(self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.settings_dock)

        self.btn_start = self.settings_dock.btn_start_left
        self.btn_stop = self.settings_dock.btn_stop_left

        act_open_cfg = QAction("Open Config…", self); act_open_cfg.triggered.connect(self._browse_config)
        act_save_cfg = QAction("Save Config", self);  act_save_cfg.triggered.connect(self._cfg_save_to_disk)
        self.menuBar().addMenu("&Config").addActions([act_open_cfg, act_save_cfg])
        self.menuBar().setStyleSheet(
            "QMenuBar { background: #2a3439; color: #e6e6e6; border: none; }"
            "QMenuBar::item { padding: 6px 10px; }"
            "QMenuBar::item:selected { background: #3a454b; }"
            "QMenu { background: #2a3439; color: #e6e6e6; border: 1px solid #3a3a3a; }"
            "QMenu::item:selected { background: #3a454b; }"
        )

        self.setStyleSheet("QMainWindow { background: #101010; } QLabel { color: #f0f0f0; font-size: 14px; } QStatusBar { background: #0e0e0e; color: #e0e0e0; }")

    def _connect_signals(self) -> None:
        self.btn_start.clicked.connect(self._start_inference)
        self.btn_stop.clicked.connect(self._stop_inference)

    def _install_contrast_palette(self) -> None:
        pal = self.palette()
        pal.setColor(QPalette.PlaceholderText, QColor("#b0b0b0"))
        pal.setColor(QPalette.Text, QColor("#f0f0f0"))
        pal.setColor(QPalette.WindowText, QColor("#f0f0f0"))
        self.setPalette(pal)

    def _file_dialog_options(self) -> QFileDialog.Options:
        o = QFileDialog.Options()
        o |= QFileDialog.DontUseNativeDialog
        return o

    def _load_qsettings(self) -> None:
        from PySide6.QtCore import QSettings
        s = QSettings("BoardVision", "GUI")
        self.video_path  = s.value("source/file_path", self.video_path)
        self.stream_spec = s.value("source/stream_spec", self.stream_spec)
        try:
            self.frame_stride = int(s.value("perf/stride", self.frame_stride))
        except Exception:
            pass
        self.config_path = s.value("config/path", self.config_path)
        self.source_mode = s.value("source/mode", self.source_mode)
        geo = s.value("window/geometry"); st = s.value("window/state")
        if isinstance(geo, QByteArray): self.restoreGeometry(geo)
        if isinstance(st, QByteArray):  self.restoreState(st)

        # Voter params
        def _f(key, default):
            try: return float(s.value(key, default))
            except Exception: return default
        self.v_conf_thresh = _f("voter/params/conf_thresh", self.v_conf_thresh)
        self.v_solo_strong = _f("voter/params/solo_strong", self.v_solo_strong)
        self.v_iou_thresh  = _f("voter/params/iou_thresh",  self.v_iou_thresh)
        self.v_f1_margin   = _f("voter/params/f1_margin",   self.v_f1_margin)
        self.v_gamma       = _f("voter/params/gamma",       self.v_gamma)
        try:
            self.v_fuse_coords = bool(s.value("voter/params/fuse_coords", self.v_fuse_coords, type=bool))
        except Exception:
            self.v_fuse_coords = str(s.value("voter/params/fuse_coords", self.v_fuse_coords)).lower() in ("1","true","yes","on")

        # Model-level thresholds (persist)  ⬅️ new
        self.m_yolo_conf  = _f("model/yolov7/conf_thresh", self.m_yolo_conf)
        self.m_frcnn_conf = _f("model/frcnn/conf_thresh",  self.m_frcnn_conf)

    def _save_qsettings(self) -> None:
        from PySide6.QtCore import QSettings
        s = QSettings("BoardVision", "GUI")
        s.setValue("source/file_path", self.video_path)
        s.setValue("source/stream_spec", self.stream_spec)
        s.setValue("perf/stride", self.frame_stride)
        s.setValue("config/path", self.config_path)
        s.setValue("source/mode", self.source_mode)
        s.setValue("window/geometry", self.saveGeometry())
        s.setValue("window/state", self.saveState())
        # Voter params
        s.setValue("voter/params/conf_thresh", self.v_conf_thresh)
        s.setValue("voter/params/solo_strong", self.v_solo_strong)
        s.setValue("voter/params/iou_thresh",  self.v_iou_thresh)
        s.setValue("voter/params/f1_margin",   self.v_f1_margin)
        s.setValue("voter/params/gamma",       self.v_gamma)
        s.setValue("voter/params/fuse_coords", self.v_fuse_coords)
        # Model thresholds
        s.setValue("model/yolov7/conf_thresh", self.m_yolo_conf)
        s.setValue("model/frcnn/conf_thresh",  self.m_frcnn_conf)

    def closeEvent(self, event) -> None:
        try:
            self._save_qsettings()
            self._stop_inference()
        finally:
            super().closeEvent(event)

    def _sync_settings_to_ui(self) -> None:
        d = self.settings_dock
        d.le_file.setText(self.video_path)
        d.le_stream.setText(self.stream_spec)
        d.sb_stride.setValue(self.frame_stride)
        d.le_cfg_path.setText(self.config_path)
        d.set_source_mode(self.source_mode if self.source_mode in ("file", "stream") else ("file" if self.video_path else "stream"))
        self._cfg_populate_table(self.cfg_thresholds)

        # Push voter params to the UI (without retrigger)
        def _set(spin, val): spin.blockSignals(True); spin.setValue(val); spin.blockSignals(False)
        _set(d.ds_conf_thresh, self.v_conf_thresh)
        _set(d.ds_solo_strong, self.v_solo_strong)
        _set(d.ds_iou_thresh,  self.v_iou_thresh)
        _set(d.ds_f1_margin,   self.v_f1_margin)
        _set(d.ds_gamma,       self.v_gamma)
        d.cb_fuse_coords.blockSignals(True); d.cb_fuse_coords.setChecked(self.v_fuse_coords); d.cb_fuse_coords.blockSignals(False)

        # Push model thresholds to UI
        _set(d.ds_yolo_conf,  self.m_yolo_conf)
        _set(d.ds_frcnn_conf, self.m_frcnn_conf)

    def _on_source_kind_changed(self, kind: Optional[str]) -> None:
        if kind in ("file", "stream"):
            self.source_mode = kind
            self._save_qsettings()
            self.settings_dock.set_source_mode(kind)

    def _on_file_changed(self, path: str) -> None:
        self.video_path = path.strip(); self._save_qsettings()
    def _on_stream_changed(self, spec: str) -> None:
        self.stream_spec = spec.strip(); self._save_qsettings()
    def _on_stride_changed(self, v: int) -> None:
        self.frame_stride = int(v); self._save_qsettings()
    def _on_cfg_path_changed(self, p: str) -> None:
        self.config_path = p.strip(); self._save_qsettings()

    def _on_voter_param_changed(self, name: str, value: Any) -> None:
        # name in {"conf_thresh","solo_strong","iou_thresh","f1_margin","gamma","fuse_coords"}
        setattr(self, f"v_{name}", value)
        self._save_qsettings()
        self.statusBar().showMessage(f"Voter param '{name}' set to {value}")

    # Unified handler for per-model conf_thresh
    def _on_model_conf_changed(self, model: str, value: float) -> None:
        if model == "yolov7":
            self.m_yolo_conf = float(value)
        elif model == "frcnn":
            self.m_frcnn_conf = float(value)
        else:
            return
        self._save_qsettings()
        self.statusBar().showMessage(f"{model} conf_thresh set to {value:.2f}")

    def _reset_voter_params(self) -> None:
        self.v_conf_thresh = 0.50
        self.v_solo_strong = 0.95
        self.v_iou_thresh  = 0.40
        self.v_f1_margin   = 0.05
        self.v_gamma       = 1.50
        self.v_fuse_coords = True
        self._sync_settings_to_ui()
        self._save_qsettings()
        self.statusBar().showMessage("Voter parameters reset to defaults.")

    def _browse_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)", options=self._file_dialog_options())
        if path:
            self.settings_dock.le_file.setText(path)
            self.statusBar().showMessage(f"Selected file: {os.path.basename(path)} - Device: {self.device}")

    def _browse_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Config JSON", "", "JSON (*.json);;All Files (*)", options=self._file_dialog_options())
        if path:
            self.settings_dock.le_cfg_path.setText(path)
            self.cfg_raw = self._load_config_file(path)
            self.cfg_thresholds, _ = self._split_thresholds_and_meta(self.cfg_raw)
            self._cfg_populate_table(self.cfg_thresholds)
            self.statusBar().showMessage(f"Loaded config: {os.path.basename(path)}")

    def _load_config_file(self, path: str) -> dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            LOGGER.warning("Failed to read config '%s': %s. Trying default loader.", path, e)
            try:
                return default_load_f1_config("config.json")
            except Exception:
                return {}

    def _split_thresholds_and_meta(self, cfg: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
        thresholds: Dict[str, Dict[str, float]] = {}
        meta: Dict[str, Any] = {}
        for k, v in cfg.items():
            if isinstance(v, dict) and any(key in v for key in ("YOLOv7", "FRCNN")):
                thresholds[k] = v
            else:
                meta[k] = v
        return thresholds, meta

    def _cfg_populate_table(self, thresholds: dict) -> None:
        tbl = self.settings_dock.tbl_cfg
        tbl.setRowCount(0)
        for label, model_dict in thresholds.items():
            row = tbl.rowCount(); tbl.insertRow(row)
            tbl.setItem(row, 0, QTableWidgetItem(str(label)))
            tbl.setItem(row, 1, QTableWidgetItem(str(model_dict.get("YOLOv7", ""))))
            tbl.setItem(row, 2, QTableWidgetItem(str(model_dict.get("FRCNN", ""))))

    def _cfg_extract_from_table(self) -> dict:
        tbl = self.settings_dock.tbl_cfg
        out = {}
        for r in range(tbl.rowCount()):
            label_item = tbl.item(r, 0); y_item = tbl.item(r, 1); f_item = tbl.item(r, 2)
            if not label_item:
                continue
            label = label_item.text().strip()
            if not label:
                continue
            try:
                yv = float(y_item.text()) if y_item and y_item.text() != "" else None
            except Exception:
                raise ValueError(f"Row {r+1}: YOLOv7 must be a number")
            try:
                fv = float(f_item.text()) if f_item and f_item.text() != "" else None
            except Exception:
                raise ValueError(f"Row {r+1}: FRCNN must be a number")
            out[label] = {}
            if yv is not None: out[label]["YOLOv7"] = yv
            if fv is not None: out[label]["FRCNN"] = fv
        return out

    def _cfg_add_row(self) -> None:
        tbl = self.settings_dock.tbl_cfg
        row = tbl.rowCount(); tbl.insertRow(row)
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
            self.cfg_raw = self._load_config_file(self.config_path)
        except Exception as e:
            QMessageBox.critical(self, "Config Error", f"Failed to read config:\n{e}")
            return
        self.cfg_thresholds, _ = self._split_thresholds_and_meta(self.cfg_raw)
        self._cfg_populate_table(self.cfg_thresholds)
        self.statusBar().showMessage("Reloaded config from disk.")

    def _cfg_save_to_disk(self) -> None:
        try:
            thresholds = self._cfg_extract_from_table()
        except Exception as e:
            QMessageBox.critical(self, "Validation Error", str(e))
            return
        # Preserve any non-threshold metadata from the current cfg
        _, meta = self._split_thresholds_and_meta(self.cfg_raw)
        new_cfg = dict(meta)
        new_cfg.update(thresholds)
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(new_cfg, f, indent=2, sort_keys=True)
        except Exception as e:
            QMessageBox.critical(self, "Write Error", f"Failed to write {self.config_path}:\n{e}")
            return
        self.cfg_raw = new_cfg
        self.cfg_thresholds = thresholds
        self.statusBar().showMessage(f"Saved config to {os.path.basename(self.config_path)}")

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        cap: Optional[cv2.VideoCapture] = None
        if self.source_mode == "file":
            if not self.video_path:
                return None
            cap = cv2.VideoCapture(self.video_path)
        else:
            if not self.stream_spec:
                return None
            try:
                cam_idx = int(self.stream_spec)
                cap = cv2.VideoCapture(cam_idx)
            except Exception:
                cap = cv2.VideoCapture(self.stream_spec)
            safe_cap_set(cap, cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap or not cap.isOpened():
            return None
        return cap

    def _start_inference(self) -> None:
        if self.is_running:
            return
        if self.source_mode == "file" and not self.video_path:
            QMessageBox.warning(self, "No Source", "Pick a file in the Settings panel.")
            return
        if self.source_mode == "stream" and not self.stream_spec:
            QMessageBox.warning(self, "No Source", "Enter a stream in the Settings panel.")
            return

        self.cap = self._open_capture()
        if not self.cap:
            QMessageBox.critical(self, "Error", "Failed to open source.")
            return

        self.is_running = True
        self.frame_index = 0
        self._clear_panel_logs()

        self.stats_loop = RollingStats(120)
        self.stats_yolo = RollingStats(120)
        self.stats_frcnn = RollingStats(120)
        self.stats_decision = RollingStats(120)

        src_desc = self.video_path if self.source_mode == "file" else f"stream:{self.stream_spec}"
        LOGGER.info("Starting inference on %s", src_desc)
        self.timer.start()

    def _stop_inference(self) -> None:
        if not self.is_running:
            return
        self.is_running = False
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.statusBar().showMessage(f"Stopped - Device: {self.device}")
        LOGGER.info("Stopped inference.")

    def _clear_panel_logs(self) -> None:
        self.panel_yolo.log.clear(); self.panel_voter.log.clear(); self.panel_frcnn.log.clear()
        self.panel_yolo.set_stats(0, 0); self.panel_frcnn.set_stats(0, 0); self.panel_voter.set_stats(0, 0)

    def _process_next_frame(self) -> None:
        if not self.is_running or self.cap is None:
            self.timer.stop(); return

        for _ in range(max(0, self.frame_stride - 1)):
            _ = self.cap.grab()

        t_loop_start = time.perf_counter()
        ok, frame = self.cap.read()
        if not ok or frame is None:
            if self.source_mode == "file":
                self._stop_inference()
                QMessageBox.information(self, "Info", "End of video.")
                LOGGER.info("Reached end of video after.")
            else:
                LOGGER.warning("Stream read() failed; will retry…")
            return

        idx = self.frame_index; self.frame_index += 1
        try:
            t0 = time.perf_counter()
            yolo_img, yolo_dets = detect_frame(
                frame.copy(),
                model=self.yolov7_model,
                device=self.device,
                conf_thresh=self.m_yolo_conf,
            )
            t1 = time.perf_counter(); yolo_dt = t1 - t0
            self.stats_yolo.add(yolo_dt)
            yolo_summary = self._summarize_dets(yolo_dets)
            self.panel_yolo.append_log(f"#{idx:05d} {yolo_summary}  [{human_ms(yolo_dt)}]")
            self.panel_yolo.set_stats(self.stats_yolo.avg_ms, self.stats_yolo.fps)

            t0 = time.perf_counter()
            frcnn_img, frcnn_dets = run_fasterrcnn_on_frame(
                frame.copy(),
                model=self.frcnn_model,
                CLASSES=self.frcnn_classes,
                device=self.device,
                conf_thresh=self.m_frcnn_conf,
            )
            t1 = time.perf_counter(); frcnn_dt = t1 - t0
            self.stats_frcnn.add(frcnn_dt)
            frcnn_summary = self._summarize_dets(frcnn_dets, class_names=self.frcnn_classes)
            self.panel_frcnn.append_log(f"#{idx:05d} {frcnn_summary}  [{human_ms(frcnn_dt)}]")
            self.panel_frcnn.set_stats(self.stats_frcnn.avg_ms, self.stats_frcnn.fps)

            t0 = time.perf_counter()
            final_boxes, candidates = self.fn_voter_merge(
                yolo_dets,
                frcnn_dets,
                self.cfg_thresholds,
                conf_thresh=self.v_conf_thresh,
                solo_strong=self.v_solo_strong,
                iou_thresh=self.v_iou_thresh,
                f1_margin=self.v_f1_margin,
                gamma=self.v_gamma,
                fuse_coords=self.v_fuse_coords,
            )
            voter_img = self.fn_draw_voter_boxes_on_frame(frame.copy(), final_boxes, candidates)
            t1 = time.perf_counter(); voter_dt = t1 - t0

            decision_dt = yolo_dt + frcnn_dt + voter_dt
            self.stats_decision.add(decision_dt)

            voter_summary = self._summarize_voter(final_boxes, candidates)
            self.panel_voter.append_log(f"#{idx:05d} {voter_summary}  [voter={human_ms(voter_dt)}; decision={human_ms(decision_dt)}]")
            self.panel_voter.set_stats(self.stats_decision.avg_ms, self.stats_decision.fps)

            yolo_img = self.fn_overlay_label(yolo_img, "YOLOv7", color=(0, 128, 255))
            frcnn_img = self.fn_overlay_label(frcnn_img, "Faster R-CNN", color=(255, 128, 0))
            voter_img = self.fn_overlay_label(voter_img, "Decision", color=(0, 255, 255))
            self._display_image(self.panel_yolo.video, yolo_img)
            self._display_image(self.panel_voter.video, voter_img)
            self._display_image(self.panel_frcnn.video, frcnn_img)

            loop_dt = time.perf_counter() - t_loop_start
            self.stats_loop.add(loop_dt)
            src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
            msg = (
                f"Device: {self.device} | Loop: {human_ms(self.stats_loop.avg_ms)} ({self.stats_loop.fps:.1f} FPS)"
                f" | YOLO: {self.stats_yolo.avg_ms:.1f} ms ({self.stats_yolo.fps:.1f} FPS)"
                f" | FRCNN: {self.stats_frcnn.avg_ms:.1f} ms ({self.stats_frcnn.fps:.1f} FPS)"
                f" | Decision: {self.stats_decision.avg_ms:.1f} ms ({self.stats_decision.fps:.1f} FPS)"
                + (f" | Source FPS: {src_fps:.1f}" if src_fps > 0 else "")
                + (f" | Stride: {self.frame_stride}" if self.frame_stride > 1 else "")
                + (
                    f" | Voter: conf={self.v_conf_thresh:.2f}, solo={self.v_solo_strong:.2f}, "
                    f"iou={self.v_iou_thresh:.2f}, f1={self.v_f1_margin:.2f}, "
                    f"gamma={self.v_gamma:.2f}, fuse={self.v_fuse_coords}"
                )
                + (f" | Model conf: yolo={self.m_yolo_conf:.2f}, frcnn={self.m_frcnn_conf:.2f}")
            )

            self.statusBar().showMessage(msg)

            LOGGER.info(
                "F%05d | yolo=%s | frcnn=%s | voter=%s | decision=%s | loop=%s",
                idx, human_ms(yolo_dt), human_ms(frcnn_dt), human_ms(voter_dt),
                human_ms(decision_dt), human_ms(loop_dt)
            )

        except Exception as e:
            self._stop_inference()
            QMessageBox.critical(self, "Inference Error", f"Error during inference:\n{e}")
            LOGGER.exception("Inference error on frame %d: %s", idx, e)

    def _display_image(self, label: QLabel, bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]; bytes_per_line = 3 * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pm = QPixmap.fromImage(qimg)
        scaled = pm.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled)

    @staticmethod
    def _summarize_dets(dets: Any, class_names: Optional[List[str]] = None) -> str:
        n = 0; labels: List[str] = []; confs: List[float] = []
        if dets is None: return "(no detections)"
        try:
            if isinstance(dets, (list, tuple)):
                for d in dets:
                    if isinstance(d, dict):
                        n += 1
                        lbl = d.get("label") or d.get("class") or d.get("cls")
                        if isinstance(lbl, (int, np.integer)) and class_names and 0 <= int(lbl) < len(class_names):
                            labels.append(class_names[int(lbl)])
                        elif isinstance(lbl, str):
                            labels.append(lbl)
                        conf = d.get("conf") or d.get("score") or d.get("confidence")
                        if isinstance(conf, (int, float, np.floating)): confs.append(float(conf))
                    else:
                        n += 1
            else:
                if hasattr(dets, "shape") and len(getattr(dets, "shape", [])) >= 1:
                    n = int(dets.shape[0])
                else:
                    n = len(dets)
        except Exception:
            try: n = len(dets)
            except Exception: n = 0
        by_cls = Counter(labels)
        cls_str = ", ".join(f"{k}:{v}" for k, v in sorted(by_cls.items())) if by_cls else "no det list"
        if confs:
            mean_conf = sum(confs) / max(len(confs), 1)
            conf_str = f"avg_conf={mean_conf:.3f} max={max(confs):.3f}"
        else:
            conf_str = "avg_conf=NA"
        return f"{n} dets ({cls_str}); {conf_str}"

    @staticmethod
    def _summarize_voter(final_boxes: Any, candidates: Any) -> str:
        def _safe_len(x: Any) -> int:
            try: return len(x) if x is not None else 0
            except Exception: return 0
        return f"fused={_safe_len(final_boxes)} (candidates={_safe_len(candidates)})"


def main() -> int:
    app = QApplication(sys.argv)

    pal = app.palette()
    pal.setColor(QPalette.PlaceholderText, QColor("#b0b0b0"))
    pal.setColor(QPalette.Text, QColor("#f0f0f0"))
    pal.setColor(QPalette.WindowText, QColor("#f0f0f0"))
    app.setPalette(pal)

    win = VideoInferenceWindow()
    win.show()

    if sys.platform.startswith("win"): # Helps fix DWMAPI issues
        try:
            import ctypes
            hwnd = int(win.winId())
            DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, ctypes.byref(ctypes.c_int(1)), ctypes.sizeof(ctypes.c_int))
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 19, ctypes.byref(ctypes.c_int(1)), ctypes.sizeof(ctypes.c_int))
        except Exception:
            pass
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())
