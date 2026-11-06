# models/fusion.py
"""轻量特征融合"""
import torch
import torch.nn as nn


class LightFusion(nn.Module):
    """
    拼接 + MLP 融合, 适配 ViT-S
    """

    def __init__(
        self,
        encoder_dim: int = 384,
        anomaly_dim: int = 96,
        out_dim: int = 256,
    ):
        super().__init__()

        in_dim = encoder_dim + anomaly_dim

        self.fuse = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
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
        orig_features: torch.Tensor,
        anomaly_features: torch.Tensor,
        anomaly_scores: torch.Tensor,
    ) -> torch.Tensor:
        score_weight = anomaly_scores.detach().unsqueeze(-1)
        enhanced_anom = anomaly_features * (1.0 + score_weight)
        concat = torch.cat([orig_features, enhanced_anom], dim=-1)
        return self.fuse(concat)