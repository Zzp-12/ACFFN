import torch
import torch.nn as nn
class FLB(nn.Module):
    def __init__(self, in_channels, num_membership):
        super(FLB, self).__init__()
        self.M = num_membership
        self.mu = nn.Parameter(torch.Tensor(num_membership, in_channels))
        self.t = nn.Parameter(torch.ones(1))
        self.sigma = nn.Parameter(torch.Tensor(num_membership, in_channels))
        self._init_parameters()
        self.cov=nn.Sequential(nn.Conv2d(in_channels*2,in_channels,3,1,1),
        nn.ReLU())

    def _init_parameters(self):
        nn.init.normal_(self.mu, mean=0.0, std=1.0)
        nn.init.constant_(self.sigma, 1.0)
    def forward(self, x):
        x1=x
        x = x.unsqueeze(1)  # (B, 1, C, H, W)
        mu = self.mu.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, M, C, 1, 1)
        sigma = self.sigma.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        z = torch.exp(-((x - mu) / sigma).pow(2))  # (B, M, C, H, W)
        z_and = torch.exp(torch.mean(torch.log(z + 1e-8), dim=1))   # (B, C, H, W)
        z_and=torch.softmax(-z_and / 0.05, dim=1)
        z_or = torch.max(z, dim=1)[0]
        z_or = torch.softmax(-z_or / 0.05, dim=1)
        output = torch.cat([z_and, z_or], dim=1)
        return self.cov(output)

class NSPLevel(nn.Module):
    def __init__(self, in_channels, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        self.low_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding=kernel_size // 2, stride=1,
            groups=in_channels,
            bias=False
        )
        self.high_adapt = nn.Conv2d(
            in_channels, in_channels, 1,
            padding=0, bias=True
        )
        self._init_gaussian_weights()

    def _init_gaussian_weights(self):

        x = torch.arange(self.kernel_size, dtype=torch.float32) - (self.kernel_size - 1) // 2
        y = torch.arange(self.kernel_size, dtype=torch.float32) - (self.kernel_size - 1) // 2
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        sigma = 0.3 * ((self.kernel_size - 1) * 0.5 - 1) + 0.8
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, self.kernel_size, self.kernel_size)
        kernel = kernel.repeat(self.low_conv.weight.size(0), 1, 1, 1)
        with torch.no_grad():
            self.low_conv.weight.data = kernel

    def forward(self, x):
        low_freq = self.low_conv(x)
        high_freq = x - low_freq + self.high_adapt(x)
        return 0.5 * low_freq, high_freq
import torch.nn.functional as F

import torch.fft
class GDDM(nn.Module):
    def __init__(self, directions=8,inc=1):
        super().__init__()
        self.gradient_conv = nn.Conv2d(inc, 2, 3, padding=1, bias=False)
        self._init_sobel()
        self.gradient_conv.weight.requires_grad_(False)
        self.freq= nn.Sequential(
            nn.Conv2d(2, directions, 1),
            nn.GELU()
        )
        self.sp= nn.Sequential(
            Spatialbranch(2, directions),
            nn.Sigmoid()
        )
    def _init_sobel(self):
        with torch.no_grad():
            self.gradient_conv.weight.data[0, 0] = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            self.gradient_conv.weight.data[1, 0] = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    def forward(self, x_lidar):
        grad = self.gradient_conv(x_lidar)  # [B,2,H,W]
        fft_feat = torch.fft.fft2(grad)
        magnitude = torch.log(1 + torch.abs(fft_feat))
        phase = torch.angle(fft_feat)
        freq_feat = self.freq(magnitude*phase)
        return self.sp(grad)*freq_feat
class Spatialbranch(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.s1= nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1),nn.Mish())
        self.s2 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1),nn.Mish())
    def forward(self, x):
        a1 = self.s1(x)
        a2 = self.s2(x)
        return a1 * a2
class GAPDynamicNSDFB(nn.Module):
    def __init__(self, in_channels, directions=8, kernel_size=3,inc=1):
        super().__init__()
        self.directions = directions
        self.kernel_size=kernel_size
        self.in_channels=in_channels

        self.projection = nn.Sequential(
            nn.Conv2d(inc, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, in_channels, 1)
        )
        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   padding=kernel_size // 2, groups=in_channels, bias=False)
        self.attention=GDDM(inc=inc)
    def rotate_weights(self, weights, rotation):
        """Rotate weights by 90 degrees multiples, considering directions."""
        if rotation == 0:
            return weights
        elif rotation == 1:
            return torch.rot90(weights, k=1, dims=[2, 3])
        elif rotation == 2:
            return torch.rot90(weights, k=2, dims=[2, 3])
        elif rotation == 3:
            return torch.rot90(weights, k=3, dims=[2, 3])
        elif rotation >= 4:
            flipped = torch.flip(weights, [3])
            return torch.rot90(flipped, k=rotation - 4, dims=[2, 3])
    def forward(self, x_hsi, x_lidar):
        lidar_proj = self.projection(x_lidar)
        x_hsi = x_hsi * torch.sigmoid(lidar_proj)
        base_weight = self.base_conv.weight
        directional_features = []
        for i in range(self.directions):
            rotated_weight = self.rotate_weights(base_weight, i)
            feature = F.conv2d(x_hsi, rotated_weight,
                               padding=self.kernel_size // 2, groups=self.in_channels)
            directional_features.append(feature)

        weights = self.attention(x_lidar)
        weighted_sum = sum([weights[:, i].unsqueeze(1) * feat for i, feat in enumerate(directional_features)])
        return x_hsi + weighted_sum



class AdaptiveNSCT(nn.Module):
    def __init__(self, num_scales=3, num_directions=8, in_channels=30, base_channels=32,inc=1):
        super().__init__()
        self.num_scales = num_scales
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU()
        )
        self.nsp_levels = nn.ModuleList([
            NSPLevel(base_channels) for _ in range(num_scales)
        ])
        self.nsdfb_levels = nn.ModuleList([
            GAPDynamicNSDFB(base_channels, num_directions,inc=inc) for _ in range(num_scales)
        ])

        total_channels = base_channels + num_scales * base_channels
        self.final_conv = nn.Sequential(
            nn.Conv2d(total_channels, in_channels, 1),
            nn.Tanh()
        )

    def forward(self, x,x2):
        features = self.initial_conv(x)
        subbands = [features]
        current_low = features
        for i in range(self.num_scales):
            low, high = self.nsp_levels[i](current_low)
            directional = self.nsdfb_levels[i](high,x2)
            subbands.append(directional)
            current_low = low
        combined = torch.cat(subbands, dim=1)
        return self.final_conv(combined)
class SpectralAttention(nn.Module):
    def __init__(self, bands):
        super().__init__()
        # 光谱
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_pool2 = nn.AdaptiveMaxPool1d(1)
        self.attention = nn.Sequential(
            nn.Linear(bands, bands // 8),
            nn.ReLU(),
            nn.Linear(bands // 8, bands),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W)

        # 光谱注意力生成
        x1 = self.global_pool(x_flat).squeeze(-1)
        x2 = self.global_pool2(x_flat).squeeze(-1)
        att = self.attention(x1 * x2)  # [B,C]
        return x * att.view(B, C, 1, 1)
class SpectralSpatialNSCT(nn.Module):
    def __init__(self, bands=30, base_channels=32,bands2=1):
        super().__init__()
        self.spectral_branch=nn.Sequential(SpectralAttention(bands),nn.BatchNorm2d(bands),)
        self.spatial_branch = AdaptiveNSCT(
            in_channels=bands,
            base_channels=base_channels,
            inc=bands2
        )
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(2*bands, base_channels // 2, 1),
            nn.ReLU(),
            nn.Conv2d(base_channels // 2, 2, 1),
            nn.Softmax(dim=1)
        )
    def forward(self, x,x2):
        spectral_feat = self.spectral_branch(x)
        spatial_feat = self.spatial_branch(x,x2)
        combined = torch.cat([spectral_feat, spatial_feat], dim=1)
        gate = self.fusion_gate(combined)  # [B,2,H,W]
        return gate[:, 0:1] * spectral_feat + gate[:, 1:2] * spatial_feat+x



class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积"""

    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding=kernel_size // 2, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, bias=False
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))
class MorphoQKVAttention(nn.Module):
    def __init__(self, in_channels=1,morpho_kernel_size=5):
        super().__init__()
        self.morpho_kernel_size = morpho_kernel_size
        self.dilation_kernel = nn.Parameter(
            torch.randn(in_channels, in_channels, morpho_kernel_size, morpho_kernel_size)
        )
        self.erosion_kernel = nn.Parameter(
            torch.randn(in_channels, in_channels, morpho_kernel_size, morpho_kernel_size)
        )

        self.morpho_out_channels = 2

        self.q_proj = nn.Conv2d(in_channels, self.morpho_out_channels , 1)
        self.k_proj = nn.Conv2d(self.morpho_out_channels, self.morpho_out_channels, 1)
        self.v_proj = nn.Conv2d(self.morpho_out_channels, self.morpho_out_channels , 1)

        self.fc = nn.Sequential(
            nn.Conv2d(
                self.morpho_out_channels,
                in_channels,
                1
            ),
        )
    def forward(self, x):
        B, C, H, W = x.shape

        pad = self.morpho_kernel_size // 2
        padded_x = F.pad(x, (pad,) * 4, mode='reflect')
        dilation = F.conv2d(
            padded_x,
            torch.sigmoid(self.dilation_kernel),
            stride=1,
            padding=0
        )
        dilation = dilation.max(dim=1, keepdim=True)[0]
        erosion = F.conv2d(padded_x, 1 - torch.sigmoid(self.erosion_kernel))
        erosion = erosion.min(dim=1, keepdim=True)[0]
        morpho_feat = torch.cat([dilation, erosion], dim=1)
        q = self.q_proj(x).view(B, 2, H * W)  # [B, h, C//h, N]
        k = self.k_proj(morpho_feat).view(B, 2, H * W)
        v = self.v_proj(morpho_feat).view(B,  2, H * W)

        attn = q*k / (q.shape[2] ** 0.5)
        attn = F.softmax(attn, dim=-1)
        global_feat =  attn* v
        global_feat = global_feat.reshape(B, self.morpho_out_channels , H, W)

        return self.fc(global_feat)+x
class SimpleCNN(nn.Module):
    def __init__(self,  num_classes=6,dim=248,band1=30,band2=1):
        super(SimpleCNN, self).__init__()
        self.name='Tri'
        self.band2=band2
        self.c1=nn.Sequential(nn.Conv2d(band1,32,3,1,1),nn.BatchNorm2d(32),nn.ReLU())
        self.c2 = nn.Sequential(nn.Conv2d(32, 64, 3,1,1), nn.BatchNorm2d(64), nn.ReLU())
        self.c3 = nn.Sequential(nn.Conv2d(64, 128, 3,1,1), nn.BatchNorm2d(128), nn.ReLU())
        self.c01 = nn.Sequential(nn.Conv2d(band2, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU())
        self.c02 = nn.Sequential(nn.Conv2d(32, 64, 3,1,1), nn.BatchNorm2d(64), nn.ReLU())
        self.c03 = nn.Sequential(nn.Conv2d(64, 128, 3,1,1), nn.BatchNorm2d(128), nn.ReLU())
        self.fz=FLB(dim,7)
        # self.fz1 = FLB(dim, 3)
        # self.fz2 = FLB(64, 3)muu hh no

        self.cf1=nn.Sequential(nn.Sigmoid())
        self.cf2 = nn.Sequential( nn.Sigmoid())
        self.pf0=SpectralSpatialNSCT(bands=band1,bands2=band2)

        self.p2=MorphoQKVAttention(in_channels=band2)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.full_connection = nn.Sequential(
            nn.Linear(128, num_classes),)
    def forward(self, x,x2):
        x=self.pf0(x,x2)
        x2=self.p2(x2)
        x1 = self.c1(x)
        x11 = self.c2(x1)
        x111 = self.c3(x11)
        x12 = self.c01(x2)
        x112 = self.c02(x12)
        x1112 = self.c03(x112)

        ou1 = self.fz(x111)
        ou1=self.cf1(ou1)
        ou2 = self.fz(x1112)
        ou2 = self.cf2(ou2)
        ou = ou1 * x111 + ou2 * x1112
        out = self.global_pooling(ou).squeeze().squeeze()
        out1 = self.full_connection(out)
        return out1

