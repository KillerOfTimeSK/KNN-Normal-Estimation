import torchvision.models as tvmodels
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        resnet50 = tvmodels.resnet50(pretrained=True)
        self.encoder = nn.Sequential(
            nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.relu),
            nn.Sequential(resnet50.maxpool, resnet50.layer1),
            nn.Sequential(resnet50.layer2),
            nn.Sequential(resnet50.layer3),
            nn.Sequential(resnet50.layer4)
        )
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.bottleneck_channels = resnet50.layer4[-1].bn2.num_features
        N = self.bottleneck_channels

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(N * 4, N * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(N * 2, N * 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.finisher = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(N, 3, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        y = self.encoder(x)
        y = self.decoder(y)
        y = self.finisher(y)
        return y