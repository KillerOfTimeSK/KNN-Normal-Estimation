from torchvision.models import vgg16, resnet50
import torch.nn as nn



class VGGNormal(nn.Module):
    def __init__(self):
        super(VGGNormal, self).__init__()
        # Use the VGG16 feature extractor
        self.vgg = vgg16(weights='DEFAULT').features
        self.layer_ids = [4, 9, 16, 23]
        # Initialize the decoder
        self.decoder0 = nn.Sequential(
            # Initial deconvolution layer: Upsample to (14, 14, 512) - same as 23rd layer of vgg16
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.decoder1 = nn.Sequential(
            # First deconvolution layer: Upsample to (28, 28, 256) - same as 16th layer of vgg16
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.decoder2 = nn.Sequential(
            # Second deconvolution layer: Upsample to (56, 56, 128) - same as 9th layer of vgg16
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.decoder3 = nn.Sequential(
            # Third deconvolution layer: Upsample to (112, 112, 64) same as 4th layer of vgg16
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.decoder4 = nn.Sequential(
            # Fourth deconvolution layer: Upsample to (224, 224, 32) - same as input image HxW
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.decoder5 = nn.Sequential(
            # Final deconvolution layer: Upsample to (224, 224, 3) - same dimensions and channel number as input
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),
        )
        self.final = nn.Tanh()

    def forward(self, x):
        outputs = {}
        for i, layer in enumerate(self.vgg):

            x = layer(x)
            if i in self.layer_ids:
                outputs[f'layer_{i}'] = x

        # Reconstruct the image using the decoder
        x = self.decoder0(x) + outputs['layer_23']
        x = self.decoder1(x) + outputs['layer_16']
        x = self.decoder2(x) + outputs['layer_9']
        x = self.decoder3(x) + outputs['layer_4']
        x = self.decoder4(x)
        x = self.decoder5(x)
        # Tanh for output in range <-1,1>
        return self.final(x)


class VGGDepthNormal(nn.Module):
    def __init__(self):
        super(VGGDepthNormal, self).__init__()
        # Use the VGG16 feature extractor
        self.vgg = vgg16(weights='DEFAULT').features
        self.layer_ids = [4, 9, 16, 23]
        self.resnet = resnet50(weights='DEFAULT')

        self.conv256_128 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv512_256 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv1024_512 = nn.Conv2d(1024, 512, kernel_size=1)
        self.conv2048_512 = nn.Conv2d(2048, 512, kernel_size=1)
        # Initialize the decoder
        self.decoder0 = nn.Sequential(
            # Initial deconvolution layer: Upsample to (14, 14, 512) - same as 23rd layer of vgg16
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.decoder1 = nn.Sequential(
            # First deconvolution layer: Upsample to (28, 28, 256) - same as 16th layer of vgg16
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.decoder2 = nn.Sequential(
            # Second deconvolution layer: Upsample to (56, 56, 128) - same as 9th layer of vgg16
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.decoder3 = nn.Sequential(
            # Third deconvolution layer: Upsample to (112, 112, 64) same as 4th layer of vgg16
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.decoder4 = nn.Sequential(
            # Fourth deconvolution layer: Upsample to (224, 224, 32) - same as input image HxW
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.decoder5 = nn.Sequential(
            # Final deconvolution layer: Upsample to (224, 224, 3) - same dimensions and channel number as input
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),
        )
        self.final = nn.Tanh()

    def forward(self, x, y):
        y = self.resnet.conv1(y)
        y = self.resnet.bn1(y)
        y = self.resnet.relu(y)
        y1 = y
        y = self.resnet.maxpool(y)
        y = self.resnet.layer1(y)
        y2 = self.conv256_128(y)
        y = self.resnet.layer2(y)
        y3 = self.conv512_256(y)
        y = self.resnet.layer3(y)
        y4 = self.conv1024_512(y)
        y = self.resnet.layer4(y)
        y = self.conv2048_512(y)

        outputs = {}
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layer_ids:
                outputs[f'layer_{i}'] = x

        x = x + y
        # Reconstruct the image using the decoder
        x = self.decoder0(x) + outputs['layer_23'] + y4
        x = self.decoder1(x) + outputs['layer_16'] + y3
        x = self.decoder2(x) + outputs['layer_9'] + y2
        x = self.decoder3(x) + outputs['layer_4'] + y1
        x = self.decoder4(x)
        x = self.decoder5(x)
        # Tanh for output in range <-1,1>
        return self.final(x)

