from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def fasterrcnn(num_classes, pretrained=False):
    
    # load the existing object detection model
    model = fasterrcnn_resnet50_fpn(pretrained=pretrained, pretrained_backbone=False)
    # change the head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model