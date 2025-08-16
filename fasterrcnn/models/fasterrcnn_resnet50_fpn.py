import torch
import torchvision
import random
import numpy as np
import time

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Set a time-based random seed for torch and related libs
# def set_random_seed(seed=None):
#     if seed is None:
#         seed = 123123
#     print(f"[INFO] Torch seed set to: {seed}")
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = False
#     torch.backends.cudnn.benchmark = True

def create_model(num_classes, pretrained=True, coco_model=False):
    #set_random_seed()  # Set the seed when this module is loaded
    # Load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None
    )
    if coco_model:  # Return the COCO pretrained model for COCO classes.
        return model, coco_model

    # Get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    return model

if __name__ == '__main__':
    from model_summary import summary
    model = create_model(num_classes=81, pretrained=True, coco_model=True)
    summary(model)
