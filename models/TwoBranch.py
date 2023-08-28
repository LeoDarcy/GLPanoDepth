
from .TransformerNet import TransformerNet
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import sys
import os
from collections import OrderedDict
def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(inplace=False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )

class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)

        # return out + x


class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
            # output += res

        output = self.resConfUnit2(output)

        output = self.out_conv(output)

        return output


def conv3x3(in_planes, out_planes, padding=1, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class FeatureFusionBlock(nn.Module):
    def __init__(self, planes, stride=1):
        super(FeatureFusionBlock, self).__init__()
        norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(2*planes, planes)
        self.conv2 = conv1x1(planes, planes)
        self.conv3 = conv3x3(2*planes, planes)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        attn = self.sigmoid(x)
        x1 = x1 * attn
        x2 = x2 * (1-attn)
        out = torch.cat((x1, x2), 1)
        out = self.conv3(out)
        out = self.relu(out)

        return out


class FeatureFusionBlock_ADD(nn.Module):
    def __init__(self, planes, stride=1):
        super(FeatureFusionBlock_ADD, self).__init__()
        norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(planes, planes)
        self.conv2 = conv1x1(planes, planes)
        self.conv3 = conv3x3(planes, planes)
        self.conv4 = ResBlock(planes, planes)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x = x1+x2
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        attn = self.sigmoid(x)
        x1 = x1 * attn
        x2 = x2 * (1-attn)
        out = x1+x2
        out = self.conv3(out)
        out = self.relu(out)
        out = self.conv4(out)

        return out



class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(ResBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.ifshift = False
        if inplanes != planes:
            self.ifshift = True
            self.shift = conv1x1(inplanes, planes, 1)
        self.conv1 = conv3x3(inplanes, planes, 1, stride)

        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes, 1)

        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)
        if self.ifshift:
            identity = self.shift(identity)
        out += identity
        out = self.relu(out)

        return out



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

class Depth_module(nn.Module):
    def __init__(self, batch_norm=False):
        super(Depth_module, self).__init__()
        use_bn = False
        self.fus1 = FeatureFusionBlock_ADD(64)
        self.fus2 = FeatureFusionBlock_ADD(128)
        self.fus3 = FeatureFusionBlock_ADD(128)

        self.combine0 = FeatureFusionBlock_ADD(64)
        self.combine1 = FeatureFusionBlock_ADD(64)
        self.combine2 = FeatureFusionBlock_ADD(128)
        self.combine3 = FeatureFusionBlock_ADD(128)

        self.shift0 = ConvModule(256, 128, stride=1)
        self.shift1 = ConvModule(256, 128, stride=1)
        self.shift2 = ConvModule(128, 64, stride=1)
        self.shift3 = ConvModule(128, 64, stride=1)
        self.shift4 = ConvModule(128, 64, stride=1)

        self.outlayer2 = ResBlock(64, 64, stride=1)
        self.outlayer1 = ConvModule(64, 1, stride=1)

    def forward(self, f0, f1, f2, f3, rgb1, rgb2, rgb3, rgb4):
        pre4 = nn.functional.interpolate(
            rgb4, scale_factor=2, mode="bilinear", align_corners=True)
        pre4 = self.shift0(pre4)  # torch.Size([18, 128, 32, 64])
        rgb3 = self.shift1(rgb3)
        pre3 = self.fus3(f3, rgb3)  # torch.Size([18, 128, 64, 128])
        pre3 = self.combine3(pre4, pre3)

        pre3 = nn.functional.interpolate(
            pre3, scale_factor=2, mode="bilinear", align_corners=True)
        pre2 = self.fus2(f2, rgb2)# torch.Size([18, 128, 64, 128])
        pre2 = self.combine2(pre3, pre2)# torch.Size([18, 128, 64, 128])

        pre2 = nn.functional.interpolate(
            pre2, scale_factor=2, mode="bilinear", align_corners=True)
        f1 = self.shift2(f1)
        pre1 = self.fus1(f1, rgb1)# torch.Size([18, 64, 128, 256])
        pre2 = self.shift3(pre2)# torch.Size([18, 64, 128, 256])
        pre1 = self.combine1(pre2, pre1)# torch.Size([18, 64, 128, 256])
        pre1 = nn.functional.interpolate(
            pre1, scale_factor=2, mode="bilinear", align_corners=True)
        f0 = self.shift4(f0)
        pre1 = self.combine0(pre1, f0)

        x = self.outlayer2(pre1)
        x = self.outlayer1(x)
        return x


class RGB_module(nn.Module):
    def __init__(self, batch_norm=False):
        super(RGB_module, self).__init__()
        in_channels = 3
        
        
        self.conv1_1 = ConvModule(3, 64, kernel_size=3,stride=1, padding=1)
        self.conv1_2 = ResBlock(64, 64)
        self.pool1 = ConvModule(64, 64, kernel_size=4,stride=2, padding=1)

        self.conv2_1 = ResBlock(64, 64)
        self.conv2_2 = ResBlock(64, 64)
        self.conv2_3 = ResBlock(64, 64)
        self.pool2 = ConvModule(64, 64, kernel_size=4,stride=2, padding=1)

        self.conv3_1 = ResBlock(64, 128)
        self.conv3_2 = ResBlock(128, 128)
        self.conv3_3 = ResBlock(128, 128)
        self.pool3 = ConvModule(128, 128, kernel_size=4,stride=2, padding=1)

        self.conv4_1 = ResBlock(128, 256)
        self.conv4_2 = ResBlock(256, 256)
        self.conv4_3 = ResBlock(256, 256)
        self.pool4 = ConvModule(256, 256, kernel_size=4,stride=2, padding=1)

        self.conv5_1 = ResBlock(256, 256)
        self.conv5_2 = ResBlock(256, 256)
        self.conv5_3 = ResBlock(256, 256)
        # self.pool5 = nn.MaxPool2d(kernel_size=3,stride=2, padding=1)
        
    def forward(self, x):
        x = self.conv1_1(x)
        
        x = self.conv1_2(x)
        x = self.pool1(x)
        
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x1 = self.conv2_3(x)
        x = self.pool2(x1)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x2 = self.conv3_3(x)
        x = self.pool3(x2)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x3 = self.conv4_3(x)
        x = self.pool4(x3)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x4 = self.conv5_3(x)

        return x1, x2, x3, x4

class TwoBranch(nn.Module):
    '''
    use global and local feature
    '''
    def __init__(self, image_height, image_width):
        super(TwoBranch, self).__init__()
        self.Transformer_branch = TransformerNet(image_height=image_height, image_width=image_width)
        self.RGB_branch = RGB_module()
        self.depth_branch = Depth_module()

    def forward(self, img, cubemap):
        trans_depth, f1, f2, f3, f4 = self.Transformer_branch(cubemap)
        rgb1, rgb2, rgb3, rgb4 = self.RGB_branch(img)
        result = self.depth_branch(f1, f2, f3, f4, rgb1, rgb2, rgb3, rgb4)
        return trans_depth, result
    
def ModelLoadTransformer(model, transformer_path):
        sg_load_weights_dir = transformer_path
        sg_load_weights_dir = os.path.expanduser(sg_load_weights_dir)

        assert os.path.isdir(sg_load_weights_dir), \
            "Cannot find Transformer weight folder {}".format(sg_load_weights_dir)
        print("loading sg estimator model from folder {}".format(sg_load_weights_dir))


        path = os.path.join(sg_load_weights_dir, "{}.pth".format("model"))
        model_dict = model.module.Transformer_branch.state_dict()
        pretrained_dict = torch.load(path)

        
        clear_dicts = OrderedDict()
        for k, value in pretrained_dict.items():
            if "module" in k: 
                k = k.split(".")[1:]
                k = ".".join(k)
            clear_dicts[k] = value
        model.module.Transformer_branch.load_state_dict({k: v for k, v in clear_dicts.items() if k in model_dict})
        for k, v in model_dict.items():
            assert k in clear_dicts, "No match model weight, please check"


