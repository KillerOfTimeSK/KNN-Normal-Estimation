import torchvision.models as tvmodels
import torch.nn.functional as Functional
import torch.nn as nn
import torch
import sys
from main import angular_error
from UNetProfiler import UNetProfiler

def SetActive(module : nn.Module, isLive=True):
    module.requires_grad = isLive
    for param in module.parameters():
        param.requires_grad = isLive
    if isinstance(module, nn.Sequential):
        for i in range(module.__len__()): module[i].requires_grad = isLive

def PrintGradients(module : nn.Module, file, moduleName=None):
    for name, param in module.named_parameters():
        if moduleName is not None: name = moduleName + '.' + name
        if param.grad is not None: file(f'Gradient for {name}: mean={param.grad.mean():.4f}, std={param.grad.std():.4f}, min={param.grad.min():.4f}, max={param.grad.max():.4f}, size={param.grad.size()}, requires_grad={param.requires_grad}, device={param.device}')
        #else: print(f'No gradient for {name}')

def PrintTensorStats(tensor, name, file):
    if tensor is not None: file(f'{name} - mean: {tensor.mean():.4f}, std: {tensor.std():.4f}, min: {tensor.min():.4f}, max: {tensor.max():.4f}, size: {tensor.size()}')
    else: file(f'{name} - No tensor')

class DecoderBlock(nn.Module):
    def __init__(self, inN, outN, innerN, extras, GlFeatures, file, moduleName=None, useLarge=False):
        self.file = file
        self.moduleName = moduleName
        self.firstPass = True
        super(DecoderBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.inN = inN + extras + GlFeatures
        self.outN = outN
        self.file(f"Constructing DecoderBlock {moduleName} with innerN != outN, using inN={inN}, outN={outN}, innerN={innerN}, extras={extras}, GlFeatures={GlFeatures}")
        if useLarge:
            self.inner = nn.Sequential(
                nn.Conv2d(self.inN, innerN, kernel_size=1),
                nn.PReLU(),
                nn.Conv2d(innerN, innerN, kernel_size=5, dilation=2, padding='same', padding_mode='reflect'),
                nn.ReLU(inplace=True),
                nn.Conv2d(innerN, innerN, kernel_size=3, padding='same', padding_mode='reflect'),
                nn.ReLU(inplace=True),
                nn.Conv2d(innerN, innerN, kernel_size=3, padding='same', padding_mode='reflect'),
                nn.ReLU(inplace=True),
                nn.Conv2d(innerN, self.outN, kernel_size=1),
            )
        else:
            self.inner = nn.Sequential(
                nn.Conv2d(self.inN, innerN, kernel_size=1),
                nn.PReLU(),
                nn.Conv2d(innerN, innerN, kernel_size=3, dilation=2, padding='same', padding_mode='reflect'),
                nn.ReLU(inplace=True),
                nn.Conv2d(innerN, innerN, kernel_size=3, padding='same', padding_mode='reflect'),
                nn.ReLU(inplace=True),
                nn.Conv2d(innerN, self.outN, kernel_size=1),
            )
        self.transform = nn.Conv2d(self.inN, self.outN, kernel_size=1)
        self.norm = nn.BatchNorm2d(self.outN)
        self.act = nn.Tanh()
    
    def InitWeights(self):
        self.file(f"Initializing weights for DecoderBlock {self.moduleName}")
        for m in self.inner.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                #nn.init.constant_(m.weight, 0.001)
                if m.bias is not None: nn.init.zeros_(m.bias)
    
    def forward(self, x, extras, GlFeatures):
        if self.firstPass: self.file(f"DecoderBlock {self.moduleName} input size: {x.size()}, extras size: {extras.size() if extras is not None else "No extras"}, GlFeatures size: {GlFeatures.size() if GlFeatures is not None else "No GlFeatures"} || {self.inN}, {self.outN}")
        if GlFeatures is not None:
            broadcasted = GlFeatures.expand(-1, -1, x.size()[2], x.size()[3])
            if extras is not None: x = torch.cat([x, broadcasted, extras], 1)
            else: x = torch.cat([x, broadcasted], 1)
        elif extras is not None: x = torch.cat([x, extras], 1)
        if self.firstPass: self.file("DecoderBlock concatenated size:", x.size())
        x = self.upsample(x)
        if self.firstPass: self.file("DecoderBlock upsampled size:", x.size())
        simple = self.transform(x)
        processed = self.inner(x)
        if self.PrintSizes: PrintTensorStats(x, "DecoderBlock before inner", self.file)
        if self.PrintSizes: PrintTensorStats(processed, "DecoderBlock after inner", self.file)
        if self.PrintSizes: PrintTensorStats(simple, "DecoderBlock after transform", self.file)
        self.firstPass = False
        y = 0.2 * simple + 0.8 * processed
        y = self.norm(y)
        return self.act (y)

    def ToDevice(self, device):
        self.file(f"Moving DecoderBlock {self.moduleName} to device:", device)
        self.upsample.to(device)
        self.inner.to(device)
        self.act.to(device)
        for i in range(self.inner.__len__()): self.inner[i].to(device)
        self.transform.to(device)
    
    def SetActive(self, isLive=True):
        self.file("Setting decoder block to activity:", isLive)
        for module in [self, self.upsample, self.inner, self.transform, self.act]: SetActive(module, isLive)

    def PrintGrads(self): PrintGradients(self, self.file, self.moduleName)

class UNet(nn.Module):
    def __init__(self, file=sys.stdout, useLarge=False, useGlobalFeatures=True):
        super(UNet, self).__init__()
        self.file = file
        self.Profiler = UNetProfiler(self.file)
        self.PrintSizes = True
        self.useGlobalFeatures = useGlobalFeatures
        self.file("Preparing UNetBase")
        resnet50 = tvmodels.resnet50(weights='DEFAULT')
        self.encoder1 = nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.relu)
        self.encoder2 = nn.Sequential(resnet50.maxpool, resnet50.layer1)
        self.encoder3 = nn.Sequential(resnet50.layer2)
        self.encoder4 = nn.Sequential(resnet50.layer3)
        self.encoder5 = nn.Sequential(resnet50.layer4)
        
        self.bottleneck_channels = self.encoder5[-1][-1].bn2.num_features
        self.expSizes = []

        self.encoderN = 5
        N = self.bottleneck_channels

        self.normalizer = nn.Sequential(
            nn.BatchNorm2d(4 * N),
            nn.Tanh()
        )

        if useGlobalFeatures:
            self.globalFeatures = N // 4
            self.globalExtractor = nn.Sequential(
                nn.Conv2d(N * 4, 2 * self.globalFeatures, kernel_size=2, padding=0, stride=2),
                nn.PReLU(),
                nn.Conv2d(2 * self.globalFeatures, self.globalFeatures, kernel_size=4, padding=0),
                nn.PReLU(),
                nn.Conv2d(self.globalFeatures, self.globalFeatures, kernel_size=1),
                nn.Tanh(),
            )
        else:
            self.globalFeatures = 0
            self.globalExtractor = None
        
        self.expSizes.append(('Decoder0', N * 4, N * 4))

        N = N * 2
        self.decoder1 = DecoderBlock(2 * N, N, N // 2, 2 * N, self.globalFeatures, self.file, 'Decoder1', useLarge)
        N = N // 2
        self.decoder2 = DecoderBlock(2 * N, N, N // 2, 2 * N, self.globalFeatures, self.file, 'Decoder2', useLarge)
        N = N // 2
        self.decoder3 = DecoderBlock(2 * N, N, N // 2, 2 * N, self.globalFeatures, self.file, 'Decoder3', useLarge)
        N = N // 2
        self.decoder4 = DecoderBlock(2 * N, N, N // 2, 2 * N, self.globalFeatures, self.file, 'Decoder4', useLarge)
        N = N // 2
        
        #self.decoders = []
        # self.decoders = nn.ModuleList()
        # input_size_dividers = [1, 1, 1, 1]

        # for i in range(len(input_size_dividers)):
        #     extras = input_size_dividers[i]
        #     if extras > 0: extras = 2 * N // extras
        #     elif extras < 0: extras = 2 * N * (-extras)
        #     self.expSizes.append(('Decoder' + str(i + 1), 2 * N + extras, N))
        #     output_size = N

        #     decoder = DecoderBlock(2 * N, output_size, N // 2, extras, self.globalFeatures, self.file, f'Decoder{i+1}')
        #     self.decoders.append(decoder)
        #     #self.decoderModules.append(decoder)
        #     N = output_size // 2

        self.expSizes.append(('Finisher', 2 * N + 3, 3))
        self.finisher = nn.Sequential(
            nn.Conv2d(2 * N + 3, 3, kernel_size=1),
            nn.Tanh()
        )

    def InitWeights(self):
        self.file("Initializing weights for UNetBase")
        #for decoder in self.decoders: decoder.InitWeights()
        for d in [self.decoder1, self.decoder2, self.decoder3, self.decoder4]: d.InitWeights()
        for m in self.finisher.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
    
    def ToDevice(self, device):
        pass
        self.file(f"Moving UNetBase to device:", device)
        #self.to(device)
        # blocks = [self.encoder1, self.encoder2, self.encoder3, self.encoder4, self.encoder5, self.globalExtractor, self.finisher]
        # for block in blocks:
        #     block.to(device)
        #     for i in range(block.__len__()): block[i].to(device)

        # for decoder in self.decoders: decoder.ToDevice(device)
    
    def AddGlobalFeatures(self, x, g):
        broadcasted = g.expand(-1, -1, x.size()[2], x.size()[3])
        x = torch.cat([x, broadcasted], 1)
        return x

    def forward(self, x):
        for d in [self.decoder1, self.decoder2, self.decoder3, self.decoder4]: d.PrintSizes = self.PrintSizes
        self.Profiler("UNetBase Input", x.size(), self.PrintSizes)
        x0 = Functional.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
        e1 = self.encoder1(x0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        if self.PrintSizes: PrintTensorStats(e5, "ResNet", self.file)
        b1 = self.normalizer(e5)
        b1 = e5
        if self.PrintSizes: PrintTensorStats(b1, "Normalized", self.file)

        if self.useGlobalFeatures: g = self.globalExtractor(b1)
        else: g = None
        if self.PrintSizes: PrintTensorStats(g, "GlobalFeatures", self.file)
        if g is not None: self.Profiler("UNetBase GlobalExtractor", g.size(), self.PrintSizes)

        #orig_size = x.size()[2:]

        y = b1
        outputs = [e5, e4, e3, e2]
        # for i in range(len(self.decoders)):
        #     y = self.decoders[i](y, outputs[i], g)
        #     self.Profiler(f"Full-UNet Decoder{i+1}", y.size(), self.PrintSizes)
        #     if self.PrintSizes: PrintTensorStats(y, f"Decoder{i+1}", self.file)
        # for i, decoder in enumerate(self.decoders):
        #     y = decoder(y, outputs[i], g)
        #     self.Profiler(f"Full-UNet Decoder{i+1}", y.size(), self.PrintSizes)
        #     if self.PrintSizes: PrintTensorStats(y, f"Decoder{i+1}", self.file)
        y = self.decoder1(y, outputs[0], g)
        self.Profiler("Full-UNet Decoder1", y.size(), self.PrintSizes)
        if self.PrintSizes: PrintTensorStats(y, "Decoder1", self.file)
        y = self.decoder2(y, outputs[1], g)
        self.Profiler("Full-UNet Decoder2", y.size(), self.PrintSizes)
        if self.PrintSizes: PrintTensorStats(y, "Decoder2", self.file)
        y = self.decoder3(y, outputs[2], g)
        self.Profiler("Full-UNet Decoder3", y.size(), self.PrintSizes)
        if self.PrintSizes: PrintTensorStats(y, "Decoder3", self.file)
        y = self.decoder4(y, outputs[3], g)
        self.Profiler("Full-UNet Decoder4", y.size(), self.PrintSizes)
        if self.PrintSizes: PrintTensorStats(y, "Decoder4", self.file)

        #if self.PrintSizes: print("Expanding current size:", y.size(), "to original size:", orig_size)
        #y = Functional.interpolate(y, size=orig_size, mode='bilinear', align_corners=True)
        x = Functional.interpolate(x, size=(y.size()[2:]), mode='bilinear', align_corners=True)
        y = self.finisher(torch.cat([y, x], 1))
        if self.PrintSizes: PrintTensorStats(y, "Finisher", self.file)
        self.Profiler("Full-UNet Finisher", y.size(), self.PrintSizes)
        self.PrintSizes = False
        for d in [self.decoder1, self.decoder2, self.decoder3, self.decoder4]: d.PrintSizes = False
        return y
    
    def PrintGrads(self):
        PrintGradients(self, self.file)
        #for block in self.decoders: block.PrintGrads()
    
    def SetActive(self, isLive=True):
        self.file("Setting UNetBase to activity:", isLive)
        #SetActive(self.globalExtractor, isLive)
        for layer in [self.encoder1, self.encoder2, self.encoder3, self.encoder4, self.encoder5]: SetActive(layer, False)
        # SetActive(self.decoders, isLive)
        #for decoder in self.decoders: decoder.SetActive(False)
        # SetActive(self.finisher, isLive)


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()

    def forward(self, output, target):
        dot = torch.sum(output * target, dim=1)
        dot = torch.clamp(dot, -0.999, 0.999)
        ang = torch.acos(dot) / torch.pi
        pixel_loss = target - output
        length = torch.linalg.norm(output, dim=1)
        length_loss = torch.mean((length - 1)**2)
        loss = (6 * torch.mean(ang**2) + 3 * torch.mean(pixel_loss**2) + length_loss) / 10
        return loss

class AngularLoss(nn.Module):
    def __init__(self):
        super(AngularLoss, self).__init__()

    def forward(self, output, target):
        dot = torch.sum(output * target, dim=1)
        dot = torch.clamp(dot, -0.999, 0.999)
        ang = torch.acos(dot)
        err = ang * 180 / torch.pi
        return torch.mean(err)