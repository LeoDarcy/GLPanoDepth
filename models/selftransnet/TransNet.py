import torch
import torch.nn as nn
import types
import math
import torch.nn.functional as F
import einops

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .model_utils import Transformer, Transpose, Interpolate, FeatureFusionBlock_custom

def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )

class TransNet(nn.Module):
    def __init__(self, depth=12, heads=16, mlp_dim=2048, dim=1024, image_height=256, image_width=512, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super(TransNet, self).__init__()
        ngf = 128
        self.patch_height = 16
        self.patch_width = 16
        cube_height = image_height //2
        cube_width = image_height //2
        num_patches = (image_height// self.patch_height) * (image_width // self.patch_width)  #16*32=512
        cube_num_patches = 6 * (cube_height // self.patch_height) * (cube_width // self.patch_width)  #64*6=384
        patch_dim = channels * self.patch_height * self.patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b p c (h p1) (w p2) -> b (p h w) (p1 p2 c)', p1=self.patch_height, p2=self.patch_width),
            nn.Linear(patch_dim, dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, cube_num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        features = [ngf, ngf//2, ngf//4, 1]

        self.postprocess1 = nn.Sequential(
            Transpose(1, 2),#B, dim, cube_num_patches
            nn.Linear(cube_num_patches, num_patches),
            Rearrange('b c (h w) -> b c h w', h=(image_height // self.patch_height), w=(image_width // self.patch_width)),
            nn.Conv2d(
                in_channels=dim,
                out_channels=features[0]*4,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.PixelShuffle(2),
            nn.Conv2d(
                in_channels=features[0],
                out_channels=features[0]*4,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.PixelShuffle(2),
            nn.Conv2d(
                in_channels=features[0],
                out_channels=features[0]*4,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.PixelShuffle(2),
        )
        self.postprocess2 = nn.Sequential(
            Transpose(1, 2),
            nn.Linear(cube_num_patches, num_patches),
            Rearrange('b c (h w) -> b c h w', h=(image_height // self.patch_height),
                      w=(image_width // self.patch_width)),
            nn.Conv2d(
                in_channels=dim,
                out_channels=features[0] * 4,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.PixelShuffle(2),
            nn.Conv2d(
                in_channels=features[0],
                out_channels=features[0] * 4,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.PixelShuffle(2),
        )
        self.postprocess3 = nn.Sequential(
            Transpose(1, 2),
            nn.Linear(cube_num_patches, num_patches),
            Rearrange('b c (h w) -> b c h w', h=(image_height // self.patch_height),
                      w=(image_width // self.patch_width)),
            nn.Conv2d(
                in_channels=dim,
                out_channels=features[0] * 4,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.PixelShuffle(2),
            nn.Conv2d(
                in_channels=features[0],
                out_channels=features[0],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )
        self.postprocess4 = nn.Sequential(
            Transpose(1, 2),
            nn.Linear(cube_num_patches, num_patches),
            Rearrange('b c (h w) -> b c h w', h=(image_height // self.patch_height),
                      w=(image_width // self.patch_width)),
            nn.Conv2d(
                in_channels=dim,
                out_channels=features[0],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )
        use_bn = False
        self.fus1 = _make_fusion_block(features[0], use_bn)
        self.fus2 = _make_fusion_block(features[0], use_bn)
        self.fus3 = _make_fusion_block(features[0], use_bn)
        self.fus4 = _make_fusion_block(features[0], use_bn)

        self.output_conv = nn.Sequential(
            nn.Conv2d(features[0], features[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features[1], features[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features[2], 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True)
        )
        self.hook = [3, 6, 9, 11]

    def forward(self, x):
        x = self.to_patch_embedding(x)
        x += self.pos_embedding
        x = self.dropout(x)

        features = self.transformer(x)
        x1 = self.postprocess1(features[self.hook[0]])
        x2 = self.postprocess2(features[self.hook[1]])
        x3 = self.postprocess3(features[self.hook[2]])
        x4 = self.postprocess4(features[self.hook[3]])

        path3 = self.fus4(x4)
        path2 = self.fus3(path3, x3)
        path1 = self.fus2(path2, x2)
        path0 = self.fus1(path1, x1)
        x = self.output_conv(path0)
        return x, path0, path1, path2, path3
