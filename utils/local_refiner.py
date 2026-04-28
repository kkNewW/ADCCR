import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMFusion(nn.Module):
    def __init__(self, feat_dim, text_dim):
        super().__init__()
        self.gamma = nn.Linear(text_dim, feat_dim)
        self.beta = nn.Linear(text_dim, feat_dim)

    def forward(self, feat, text_feat):
        gamma = self.gamma(text_feat).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(text_feat).unsqueeze(-1).unsqueeze(-1)
        return feat * (1.0 + gamma) + beta


class LocalBackbone(nn.Module):
    def __init__(self, in_ch=3, feat_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, feat_dim, 3, 2, 1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class LocalRefiner(nn.Module):
    def __init__(self, text_dim=768, feat_dim=256, hm_size=32):
        super().__init__()
        self.backbone = LocalBackbone(feat_dim=feat_dim)
        self.fusion = FiLMFusion(feat_dim, text_dim)
        self.head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, 1, 1)
        )
        self.hm_size = hm_size

    def forward(self, patch, text_feat):
        feat = self.backbone(patch)
        feat = self.fusion(feat, text_feat)
        heatmap = self.head(feat)
        heatmap = F.interpolate(
            heatmap,
            size=(self.hm_size, self.hm_size),
            mode="bilinear",
            align_corners=False
        )
        return heatmap
