# models/heads.py
"""多任务头 - 适配 ViT-S 小维度"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class AttentionPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.attn(x), dim=1)
        return (x * weights).sum(dim=1)


class MultiScalePooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn_pool = AttentionPooling(dim)
        self.proj = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor, anomaly_scores: torch.Tensor) -> torch.Tensor:
        attn_pooled = self.attn_pool(features)
        avg_pooled = features.mean(dim=1)
        max_pooled = features.max(dim=1)[0]
        anom_weights = F.softmax(anomaly_scores * 5.0, dim=-1).unsqueeze(-1)
        anom_pooled = (features * anom_weights).sum(dim=1)
        return self.proj(torch.cat([attn_pooled, avg_pooled, max_pooled, anom_pooled], dim=-1))


class ClassificationHead(nn.Module):
    """
    强化分类头: 多尺度池化 + 多层encoder特征
    """

    def __init__(
        self,
        in_dim: int = 256,
        encoder_dim: int = 384,
        num_layers: int = 3,
        num_classes: int = 9,
        drop: float = 0.3,
    ):
        super().__init__()

        self.layer_projs = nn.ModuleList([
            nn.Linear(encoder_dim, in_dim) for _ in range(num_layers)
        ])
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

        self.pool_fused = MultiScalePooling(in_dim)
        self.pool_encoder = MultiScalePooling(in_dim)

        cls_in_dim = in_dim * 2 + 1

        self.classifier = nn.Sequential(
            nn.Linear(cls_in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(drop * 0.5),
            nn.Linear(64, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        fused_features: torch.Tensor,
        encoder_features: List[torch.Tensor],
        anomaly_scores: torch.Tensor,
    ) -> torch.Tensor:
        fused_pooled = self.pool_fused(fused_features, anomaly_scores)

        weights = F.softmax(self.layer_weights, dim=0)
        enc_combined = sum(
            self.layer_projs[i](encoder_features[i].float()) * weights[i]
            for i in range(len(encoder_features))
        )
        enc_pooled = self.pool_encoder(enc_combined, anomaly_scores)

        max_score = anomaly_scores.max(dim=1, keepdim=True)[0]
        combined = torch.cat([fused_pooled, enc_pooled, max_score], dim=-1)
        return self.classifier(combined)


class MaskHead(nn.Module):
    def __init__(self, in_dim: int = 256, img_size: int = 518, patch_size: int = 14):
        super().__init__()
        self.h = img_size // patch_size
        self.w = img_size // patch_size
        self.img_size = img_size

        self.head = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        B = fused_features.shape[0]
        mask = self.head(fused_features).transpose(1, 2).reshape(B, 1, self.h, self.w)
        mask = F.interpolate(mask, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        return self.refine(mask)


class AnomalyScoreHead(nn.Module):
    def __init__(self, in_dim: int = 256, drop: float = 0.2):
        super().__init__()
        self.pool = AttentionPooling(in_dim)
        self.head = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(fused_features))