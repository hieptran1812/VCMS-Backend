import torch

def model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='/data/hieptq/deep-sight-image/checkpoint/detect_block.pt')
    return model