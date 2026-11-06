# models/decoder.py
"""轻量异常感知模块"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class LightweightAnomalyDetector(nn.Module):
    """
    轻量异常检测: 多层聚合 → 瓶颈MLP → 余弦差异
    适配 ViT-S encoder_dim=384
    """

    def __init__(
        self,
        encoder_dim: int = 384,
        bottleneck_dim: int = 96,
        num_layers: int = 3,
        noise_std: float = 0.1,
    ):
        super().__init__()
        self.noise_std = noise_std

        self.layer_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoder_dim, bottleneck_dim),
                nn.LayerNorm(bottleneck_dim),
            )
            for _ in range(num_layers)
        ])

        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

        self.bottleneck = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
        )

        self.reconstruct = nn.Sequential(
            nn.Linear(bottleneck_dim, encoder_dim),
            nn.LayerNorm(encoder_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, encoder_features: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = [f.float() for f in encoder_features]

        weights = F.softmax(self.layer_weights, dim=0)
        aggregated = sum(
            self.layer_projs[i](feats[i]) * weights[i]
            for i in range(len(feats))
        )

        if self.training and self.noise_std > 0:
            aggregated = aggregated + torch.randn_like(aggregated) * self.noise_std

        bottleneck_out = self.bottleneck(aggregated)

        reconstructed = self.reconstruct(bottleneck_out)
        target = feats[-1].detach()

        recon_norm = F.normalize(reconstructed, p=2, dim=-1, eps=1e-6)
        target_norm = F.normalize(target, p=2, dim=-1, eps=1e-6)
        cos_sim = (recon_norm * target_norm).sum(dim=-1).clamp(-1, 1)
        anomaly_scores = 1.0 - cos_sim

        recon_loss = anomaly_scores.mean()

        return bottleneck_out, anomaly_scores, recon_loss