import torch
import torch.nn as nn
import torch.nn.functional as F
from det_module.model.resnet import resnet18, resnet50
from det_module.model.head import DBHead
from det_module.model.body import FPN


class DBNet(nn.Module):
    def __init__(self):
        super().__init__()
        pretrained = False
        backbone_out = [256, 512, 1024, 2048]
        # backbone_out = [64, 128, 256, 512]
        self.backbone = resnet50(pretrained=pretrained)
        self.segmentation_body = FPN(backbone_out, inner_channels=256)
        self.segmentation_head = DBHead(self.segmentation_body.conv_out, adaptive=True, serial=False)

    def forward(self, x):
        """
        :return: Train mode: prob_map, threshold_map, appro_binary_map
        :return: Eval mode: prob_map, threshold_map
        """
        _, _, H, W = x.size()
        backbone_out = self.backbone(x)
        segmentation_body_out = self.segmentation_body(backbone_out)
        segmentation_head_out = self.segmentation_head(segmentation_body_out)
        y = F.interpolate(segmentation_head_out, size=(H, W),
                          mode='bilinear', align_corners=True)
        return y


if __name__ == '__main__':
    model = DBNet()
    img = torch.randn(1, 3, 640, 640)
    output = model(img)
    print('output', output.size())
