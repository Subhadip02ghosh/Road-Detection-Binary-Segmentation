import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet
import swin_transformer

class PredictionModule(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.input_channels = input_channels
        self.upfeature = nn.Sequential(nn.Conv2d(self.input_channels, 256, kernel_size=3, padding=1),
                                       nn.ReLU(inplace=True))
        self.conf_layer = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1),
                                        nn.ReLU(),
                                        nn.Conv2d(256, 1, kernel_size=3, padding=1),
                                        nn.Sigmoid())

    def forward(self, x):
        x = self.upfeature(x)
        conf = self.conf_layer(x)
        return conf

class FPN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.lat_layers = nn.ModuleList(
            [nn.Conv2d(x, 256, kernel_size=1) for x in self.in_channels])
        self.pred_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                                        nn.ReLU(inplace=True)) for _ in self.in_channels])

        self.upsample_module = nn.ModuleList([nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                              nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)])

        self.p3_upscale = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 256, kernel_size=1, stride=1)
        )
        self.p4_upscale = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 256, kernel_size=1, stride=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 256, kernel_size=1, stride=1)
        )
        self.p5_upscale = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 256, kernel_size=1, stride=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 256, kernel_size=1, stride=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 256, kernel_size=1, stride=1)
        )

    def forward(self, backbone_outs):
        p5_1 = self.lat_layers[2](backbone_outs[2])
        p5_upsample = self.upsample_module[1](p5_1)

        p4_1 = self.lat_layers[1](backbone_outs[1]) + p5_upsample
        p4_upsample = self.upsample_module[0](p4_1)

        p3_1 = self.lat_layers[0](backbone_outs[0]) + p4_upsample

        p3 = self.pred_layers[0](p3_1)
        p4 = self.pred_layers[1](p4_1)
        p5 = self.pred_layers[2](p5_1)

        p3_upscale = self.p3_upscale(p3)
        p4_upscale = self.p4_upscale(p4)
        p5_upscale = self.p5_upscale(p5)

        final_feat_map = torch.cat([p3_upscale, p4_upscale, p5_upscale], dim=1)

        return final_feat_map

def test():
    # backbone = resnet.ResNet50(img_channels=1) # Resnet-50
    backbone = swin_transformer.swin_t(channels=1, window_size=8) # Swin-T
    x = torch.rand(1, 1, 512, 512)
    y = backbone(x)
    # fpn = FPN(in_channels=[512, 1024, 2048]) # For Resnet
    fpn = FPN(in_channels=[192, 384, 768]) # For Swin-T
    feat = fpn(y)
    pred = PredictionModule(input_channels=feat.shape[1])
    out = pred(feat)

test()
