# References:
Fasterrcnn submodule sourced from: https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline
Yolov7 submodule sourced from: https://github.com/WongKinYiu/yolov7

## GUI Notes:
Voter Parameters
- yolo_results: List[Dict[str, Any]],
- frcnn_results: List[Dict[str, Any]],
- f1_config: Dict[str, Dict[str, float]],
- conf_thresh: float = 0.50,      # base acceptance threshold when no special rule applies
- solo_strong: float = 0.95,      # "always wins if solo" threshold
- iou_thresh: float = 0.40,       # requested overlap for agreement
- f1_margin: float = 0.05,        # keep close-F1 exception
- gamma: float = 1.5,             # boosts high-confidence a bit when scoring
- fuse_coords: bool = True        # if True, do score-weighted box fusion on agreement

### Gamma
- Higher gamma (e.g. 2.0+): the voter heavily trusts detections with very high confidence, tends to discard weaker overlaps.
- Lower gamma (e.g. 0.5–1.0): the voter treats lower confidence detections more leniently, leading to more fused boxes overall.
- Default (1.5 in your code): a moderate bias towards stronger detections, while still considering weaker ones if they align well.

################### TODOS ###################
- Refactor: Split the codebase, it's a monolith
- Multi-Threading support (thread inference) 
- Update the requirements.txt to reflect the new packages imported by this GUI
- Cache the video feed for playback and the boundboxes for analysis, perhaps add and analysis pane
- Make the inference render boxes be the same
- Add toggle to disable the box render for Dscision frame, and voter class
- And abiltiy to "pause" inference (the image video upload can become a video stream into the inference engine)
