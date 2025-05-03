import torchvision.models as tvmodels
import torch.nn.functional as Functional
import torch.nn as nn
import torch
from main import angular_error
from UNetProfiler import UNetProfiler

class UNetBase(nn.Module):
    def __init__(self):
        super(UNetBase, self).__init__()
        self.LearnableLayers = {}
        self.FreezeLayers = {}
        self.Profiler = UNetProfiler()
        self.PrintSizes = True
        print("Preparing UNetBase")
        resnet50 = tvmodels.resnet50(pretrained=True)
        self.encoder1 = nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.relu)
        self.encoder2 = nn.Sequential(resnet50.maxpool, resnet50.layer1)
        self.encoder3 = nn.Sequential(resnet50.layer2)
        self.encoder4 = nn.Sequential(resnet50.layer3)
        self.encoder5 = nn.Sequential(resnet50.layer4)

        self.FreezeLayers['encoder1'] = self.encoder1
        self.FreezeLayers['encoder2'] = self.encoder2
        self.FreezeLayers['encoder3'] = self.encoder3
        self.FreezeLayers['encoder4'] = self.encoder4
        self.FreezeLayers['encoder5'] = self.encoder5
        
        self.bottleneck_channels = self.encoder5[-1][-1].bn2.num_features
        self.expSizes = []

        self.encoderN = 5
        N = self.bottleneck_channels
        self.bottleneck = nn.Sequential(
            #nn.Conv2d(N * 4, N * 4, kernel_size=3, dilation=2, padding=2), # 2048x8x8
            #nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8, stride=8), # 2048x1x1
            nn.Conv2d(N * 4, N * 8, kernel_size=1), # 2048x1x1
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True), # 2048x8x8
            nn.Conv2d(N * 8, N * 4, kernel_size=3, dilation=2, padding=2), # 2048x2x2
            nn.ReLU(inplace=True),
        )
        self.expSizes.append(('Bottleneck', N * 4, N * 4))
        
        self.decoder1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(N * 4, N * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.expSizes.append(('Decoder0', N * 4, N * 4))
        self.out_size = N * 4

        self.LearnableLayers['bottleneck'] = self.bottleneck
        self.LearnableLayers['decoder1'] = self.decoder1

        for freezeLayer in self.FreezeLayers.values():
            for param in freezeLayer.parameters():
                param.requires_grad = False
    
    def MoveToGPU(self):
        self.to('cuda')
        for layer in self.LearnableLayers.values():
            layer.to('cuda')
            for i in range(layer.__len__()): layer[i].to('cuda')
        for layer in self.FreezeLayers.values():
            layer.to('cuda')
            for i in range(layer.__len__()): layer[i].to('cuda')
            for param in layer.parameters(): param.requires_grad = False

    def forward_base(self, x):
        self.Profiler("UNetBase Input", x.size(), self.PrintSizes)
        x0 = Functional.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
        self.Profiler("UNetBase Input transformed", x0.size(), self.PrintSizes)
        e1 = self.encoder1(x0)
        self.Profiler("UNetBase Encoder1", e1.size(), self.PrintSizes)
        e2 = self.encoder2(e1)
        self.Profiler("UNetBase Encoder2", e2.size(), self.PrintSizes)
        e3 = self.encoder3(e2)
        self.Profiler("UNetBase Encoder3", e3.size(), self.PrintSizes)
        e4 = self.encoder4(e3)
        self.Profiler("UNetBase Encoder4", e4.size(), self.PrintSizes)
        e5 = self.encoder5(e4)
        self.Profiler("UNetBase Encoder5", e5.size(), self.PrintSizes)

        b1 = self.bottleneck(e5)
        self.Profiler("UNetBase Bottleneck", b1.size(), self.PrintSizes)

        d1 = self.decoder1(b1)
        self.Profiler("UNetBase Decoder1", d1.size(), self.PrintSizes)

        return x.size()[2:], e1, e2, e3, e4, e5, d1

class UNet(UNetBase):
    def __init__(self):
        super(UNet, self).__init__()
        N = self.out_size // 2
        
        self.decoders = []
        input_size_dividers = [1, 1, 1, 2]

        for i in range(len(input_size_dividers)):
            extras = input_size_dividers[i]
            extras = 0 if extras == 0 else N // extras
            self.expSizes.append(('Decoder' + str(i + 1), 2 * N + extras, N))

            decoder = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(2 * N + extras, N, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            self.LearnableLayers[f'decoder{i + 1}'] = decoder
            self.decoders.append(decoder)
            N = N // 2

        self.expSizes.append(('Finisher', 2 * N + 3, 3))
        self.finisher = nn.Sequential(
            #Scale from 256x256 to 1024x768, so 1 pixel expand by 4x3
            ##nn.Upsample([768, 1024], mode='bilinear', align_corners=True),
            nn.Conv2d(2 * N + 3, 3, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        orig_size, e1, e2, e3, e4, e5, d1 = self.forward_base(x)

        y = d1
        outputs = [e4, e3, e2, e1]
        #self.PrintSizes = True
        for i in range(len(outputs)):
            # if self.PrintSizes:
            #     if outputs[i] is not None: print(f"Full-UNet Decoder{i+1} input sizes:", y.size(), outputs[i].size(), 'expected:', self.expSizes[2 + i])
            #     else: print(f"Full-UNet Decoder{i+1} input sizes:", y.size(), 'None background', 'expected:', self.expSizes[2 + i])
            if outputs[i] is not None: y = self.decoders[i](torch.cat([y, outputs[i]], 1))
            self.Profiler(f"Full-UNet Decoder{i+1}", y.size(), self.PrintSizes)

        y = Functional.interpolate(y, size=orig_size, mode='bilinear', align_corners=True)
        y = self.finisher(torch.cat([y, x], 1))
        self.Profiler("Full-UNet Finisher", y.size(), self.PrintSizes)
        self.PrintSizes = False
        return y
    
    def ApplyMiniParameters(self, mini):
        # my_bottleneck = self.LearnableLayers['bottleneck']
        # my_decoder1 = self.LearnableLayers['decoder1']
        # self.LearnableLayers.pop('bottleneck')
        # self.LearnableLayers.pop('decoder1')
        # self.FreezeLayers['bottleneck'] = my_bottleneck
        # self.FreezeLayers['decoder1'] = my_decoder1

        mini_bottleneck = mini.bottleneck.state_dict()
        mini_decoder1 = mini.decoder1.state_dict()

        self.bottleneck.load_state_dict(mini_bottleneck)
        self.decoder1.load_state_dict(mini_decoder1)

class UNetMini(UNetBase):
    # Use this to train the inner part of the network, apply the wights to the full network later
    def __init__(self):
        super(UNetMini, self).__init__()
        N = self.bottleneck_channels

        self.finisher = nn.Sequential(
            nn.Conv2d(self.out_size + 3, 3, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        _, _, _, _, _, _, d1 = self.forward_base(x)
        
        xT = Functional.interpolate(x, size=(16, 16), mode='bilinear', align_corners=True)
        y = self.finisher(torch.cat([d1, xT], 1))
        #if self.PrintSizes: print("UNetMini Finisher size:", y.size())
        self.Profiler("UNetMini Finisher", y.size(), self.PrintSizes)
        self.PrintSizes = False
        return y

#Input size: torch.Size([16, 3, 256, 256])
#Encoder1 size: torch.Size([16, 64, 128, 128])
#Encoder2 size: torch.Size([16, 256, 64, 64])
#Encoder3 size: torch.Size([16, 512, 32, 32])
#Encoder4 size: torch.Size([16, 1024, 16, 16])
#Encoder5 size: torch.Size([16, 2048, 8, 8])
#Decoder1 size: torch.Size([16, 1024, 16, 16])
#Decoder2 size: torch.Size([16, 512, 32, 32])
#Decoder3 size: torch.Size([16, 512, 64, 64])
#Decoder4 size: torch.Size([16, 256, 128, 128])
#Decoder5 size: torch.Size([16, 128, 256, 256])
#Finisher size: torch.Size([16, 3, 256, 256])

class AngularLoss(nn.Module):
    def __init__(self):
        super(AngularLoss, self).__init__()

    def forward(self, output, target):
        dot = torch.sum(output * target, dim=1)
        dot = torch.clamp(dot, -0.99, 0.99)
        ang = torch.acos(dot)
        err = ang * 180 / torch.pi
        loss = torch.mean(err)
        return loss
        # dot = Functional.cosine_similarity(output, target, dim=1)
        # return (1 - dot).mean()