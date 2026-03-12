import torch
import torch.nn as nn
import torchvision.models as models


class MultiTaskResNetUNet(nn.Module):
    def __init__(self):
        super(MultiTaskResNetUNet, self).__init__()

        resnet = models.resnet18(weights=None)

        self.encoder0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )  # 64 channels

        self.pool0 = resnet.maxpool
        self.encoder1 = resnet.layer1   # 64
        self.encoder2 = resnet.layer2   # 128
        self.encoder3 = resnet.layer3   # 256
        self.encoder4 = resnet.layer4   # 512

        # Segmentation decoder
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.segmentation_head = nn.Conv2d(64, 1, kernel_size=1)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        e0 = self.encoder0(x)              # [B,64,H/2,W/2]
        e1 = self.encoder1(self.pool0(e0)) # [B,64,H/4,W/4]
        e2 = self.encoder2(e1)             # [B,128,H/8,W/8]
        e3 = self.encoder3(e2)             # [B,256,H/16,W/16]
        e4 = self.encoder4(e3)             # [B,512,H/32,W/32]

        # Classification branch
        cls_out = self.avgpool(e4)
        cls_out = self.classifier(cls_out)

        # Segmentation branch
        d1 = self.up1(e4)
        d1 = torch.cat([d1, e3], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        d3 = self.dec3(d3)

        d4 = self.up4(d3)
        d4 = torch.cat([d4, e0], dim=1)
        d4 = self.dec4(d4)

        seg_out = self.segmentation_head(d4)
        seg_out = torch.sigmoid(seg_out)

        return seg_out, cls_out

















