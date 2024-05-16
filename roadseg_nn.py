import torch
import torch.nn as nn
from resnet import ResNet50, ResNet101
from swin_transformer import swin_t
from modules import FPN, PredictionModule

class RoadSegNN(nn.Module):
    def __init__(self, backbone_type):
        super().__init__()
        if backbone_type == 'ResNet-50':

            self.backbone = ResNet50(img_channels=1) # For ResNet
        
            self.fpn = FPN(in_channels=[512, 1024, 2048]) # For ResNet
                   
        elif backbone_type == 'ResNet-101':
            self.backbone = ResNet101(img_channels=1) # For ResNet
        
            self.fpn = FPN(in_channels=[512, 1024, 2048]) # For ResNet
        
        elif backbone_type == 'Swin-T':
            self.backbone = swin_t(channels=1, window_size=8) # For Swin-T
            self.fpn = FPN(in_channels=[192, 384, 768]) # For Swin-T      

        self.pred = PredictionModule(input_channels=256*3)

    def load_weights(self, weight):
        state_dict = torch.load(weight, map_location='cpu')
        self.load_state_dict(state_dict, strict=True)
        print("Model loaded with {}.\n".format(weight))
        
    def forward(self, x):
        y = self.backbone(x)
        feat = self.fpn(y)
        out = self.pred(feat)

        return out

# backbone_type = 'Swin-T'
# s = OSRD(backbone_type=backbone_type).cuda()
# a = torch.rand(16,1,512,512).cuda()
# out = s(a)
# print(out.shape)