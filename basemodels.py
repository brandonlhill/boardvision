from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class DetectionModel(BaseModel):
    bbox: List[int]                  # [x1,y1,x2,y2]
    label: str
    conf: float
    source: str                      # detector.name


class DetectorRuntimeConfig(BaseModel):
    name: str                        # e.g. "yolov7", "retinanet"
    conf_threshold: float = 0.25
    enabled: bool = True


class VoterParams(BaseModel):
    conf_thresh: float = 0.5
    solo_strong: float = 0.95
    iou_thresh: float = 0.4
    f1_margin: float = 0.05
    gamma: float = 1.5
    fuse_coords: bool = True
    near_tie_conf: float = 0.95
    use_f1: bool = True


class AppConfig(BaseModel):
    detectors: Dict[str, DetectorRuntimeConfig]
    voter: VoterParams
    f1_scores: Dict[str, Dict[str, float]] = {}
