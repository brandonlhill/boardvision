## ONNX GPU Cuda
Please change the version of `onnxruntime-gpu` within `requirements.txt` if you need more up-to-date version of CUDA! 

## ONNX Weight Conversion
Please read: note that we never want to export NMS metadata!
Why? 
- Injects NMS logic into the ONNX graph
- Forces fixed IoU + confidence thresholds
- Breaks threshold sweeps
- Makes per-class F1 analysis painful

### Yolov7 Weights
Download the `WongKinYiu/yolov7` github repo and open the `requirements.txt` in the root part of the repo... uncomment the installs for `exporting`. 

Use the export.py script with the following flags below:
```python
python export.py \
  --weights yolov7.pt \
  --img 640 \
  --batch 1 \  
  --dynamic \
  --simplify \ ## Graph cleanup only! Please use AI to ask what this does... before using this flag
```

The script will generate the yolov7.onnx file

### 

# References:
[FasterRCNN] Fasterrcnn submodule sourced from: https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline
[Yolov7] Yolov7 submodule sourced from: https://github.com/WongKinYiu/yolov7
