# from .spherenet import SphereAdjust
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from .selftransnet.TransNet import TransNet


class ConvModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1,
                 bn=False,
                 maxpool=False, pool_kernel=3, pool_stride=2, pool_pad=1):
        super(ConvModule, self).__init__()
        conv2d = nn.Conv2d(inplanes,planes,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation)
        layers = []
        if bn:
            layers += [nn.BatchNorm2d(planes), nn.ReLU(inplace=True)]
        else:
            layers += [nn.ReLU(inplace=True)]
        if maxpool:
            layers += [nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride,padding=pool_pad)]

        self.layers = nn.Sequential(*([conv2d]+layers))
    def forward(self, x):
        # x = self.conv2d(x)
        x = self.layers(x)
        return x

class UNet_simple(nn.Module):
    def __init__(self):
        super(UNet_simple, self).__init__()
        self.conv1 = ConvModule(3, 32, stride=1)
        self.conv2 = ConvModule(32, 64, stride=1)
        self.conv3 = ConvModule(64, 32, stride=1)
        self.conv4 = ConvModule(32, 1, stride=1)

    def forward(self, x):
        f1 = F.relu((self.conv1(x)))
        f2 = F.relu((self.conv2(f1)))
        f3 = F.relu((self.conv3(f2)))
        f4 = F.relu((self.conv4(f3)))
        
        
        return f4


class TransformerNet(nn.Module):
    def __init__(self, image_height=256, image_width=512):
        super(TransformerNet, self).__init__()
        self.transformer_module = TransNet(image_height=image_height, image_width=image_width)

    def forward(self, x):
        result, f1, f2, f3, f4 = self.transformer_module(x)
        return result, f1, f2, f3, f4


if __name__ == '__main__':
    #main()
    device = torch.device("cuda:0")
    sphere_model = TransformerNet().to(device)
    print(sphere_model)
    # data = torch.ones(2, 3, 256, 128).to(device)
    data = torch.ones(2, 3, 256, 512).to(device)
    output = sphere_model(data)
    print(output.shape)
    print("Ending test!")