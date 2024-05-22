import torch
import torch.nn as nn


class SegNet(nn.Module):
    def __init__(self, in_channels=1):
        super(SegNet, self).__init__()

        self.stage1_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.stage2_encoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.stage3_encoder = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.stage4_encoder = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.stage5_encoder = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, return_indices=True)

        # Decoder
        self.stage1_decoder = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.stage2_decoder = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.stage3_decoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.stage4_decoder = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.stage5_decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )

        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)

    def load_weights(self, weight):
        state_dict = torch.load(weight, map_location='cpu')
        self.load_state_dict(state_dict, strict=True)
        print("Model loaded with {}.\n".format(weight))

    def forward(self, x):
        # Encoder
        # print('img:', x.shape)
        x = self.stage1_encoder(x)
        # print('s1_enc:', x.shape)
        x1_size = x.size()
        x, indices1 = self.pool(x)
        # print('s1_pool:', x.shape)

        x = self.stage2_encoder(x)
        # print('s2_enc:', x.shape)
        x2_size = x.size()
        x, indices2 = self.pool(x)
        # print('s2_pool:', x.shape)

        x = self.stage3_encoder(x)
        # print('s3_enc:', x.shape)
        x3_size = x.size()
        x, indices3 = self.pool(x)
        # print('s3_pool:', x.shape)

        x = self.stage4_encoder(x)
        # print('s4_enc:', x.shape)
        x4_size = x.size()
        x, indices4 = self.pool(x)
        # print('s4_pool:', x.shape)

        x = self.stage5_encoder(x)
        # print('s5_enc:', x.shape)
        x5_size = x.size()
        x, indices5 = self.pool(x)
        # print('s5_pool:', x.shape)

        # Decoder
        x = self.unpool(x, indices=indices5, output_size=x5_size)
        # print('s1_unpool:', x.shape)
        x = self.stage1_decoder(x)
        # print('s1_dec:', x.shape)

        x = self.unpool(x, indices=indices4, output_size=x4_size)
        # print('s2_unpool:', x.shape)
        x = self.stage2_decoder(x)
        # print('s2_dec:', x.shape)

        x = self.unpool(x, indices=indices3, output_size=x3_size)
        # print('s3_unpool:', x.shape)
        x = self.stage3_decoder(x)
        # print('s3_dec:', x.shape)

        x = self.unpool(x, indices=indices2, output_size=x2_size)
        # print('s4_unpool:', x.shape)
        x = self.stage4_decoder(x)
        # print('s4_dec:', x.shape)

        x = self.unpool(x, indices=indices1, output_size=x1_size)
        # print('s5_unpool:', x.shape)
        x = self.stage5_decoder(x)
        # print('s5_dec:', x.shape)
        x = torch.sigmoid(x)
        # print()

        return x
    
# s = SegNet().cuda()
# a = torch.rand((2,1,256,256)).cuda()
# o = s(a)
