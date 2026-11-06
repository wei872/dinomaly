# models/encoder.py
"""DINOv2 冻结编码器 - 使用最小模型 ViT-S/14"""
import torch
import torch.nn as nn
from typing import List, Tuple


class DINOv2Encoder(nn.Module):
    """
    冻结的 DINOv2 ViT-S/14 特征提取器
    embed_dim=384, 参数量 21M (冻结不计入可训练)
    """

    VALID_MODELS = [
        'dinov2_vits14', 'dinov2_vitb14',
        'dinov2_vitl14', 'dinov2_vitg14',
    ]

    def __init__(
        self,
        model_name: str = 'dinov2_vits14',
        selected_layers: List[int] = None,
    ):
        super().__init__()

        if model_name not in self.VALID_MODELS:
            raise ValueError(f"Invalid model: {model_name}")

        self.encoder = torch.hub.load(
            'facebookresearch/dinov2', model_name, pretrained=True
        )
        self.embed_dim = self.encoder.embed_dim  # vits14=384, vitb14=768
        self.patch_size = self.encoder.patch_size
        self.num_layers = len(self.encoder.blocks)

        if selected_layers is None:
            n = self.num_layers  # vits14 有 12 层
            selected_layers = [n - 3, n - 2, n - 1]

        self.selected_layers = sorted(selected_layers)

        # 完全冻结
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        print(f"[Encoder] {model_name}: dim={self.embed_dim}, "
              f"patch_size={self.patch_size}, layers={self.num_layers}, "
              f"selected={self.selected_layers}")

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor, Tuple[int, int]]:
        """
        Returns:
            multi_feats: list of (B, N, D) 选中层patch特征
            cls_token: (B, D) 最后一层CLS token
            spatial_shape: (h, w)
        """
        B, _, H, W = x.shape
        h = H // self.patch_size
        w = W // self.patch_size

        n_layers = max(self.selected_layers) + 1

        try:
            all_outputs = self.encoder.get_intermediate_layers(
                x, n=n_layers, reshape=False, return_class_token=True,
            )
        except TypeError:
            all_outputs = self.encoder.get_intermediate_layers(x, n=n_layers)

        multi_feats = []
        cls_token = None

        for idx in self.selected_layers:
            actual_idx = min(idx, len(all_outputs) - 1)
            feat = all_outputs[actual_idx]

            if isinstance(feat, tuple):
                patch_tok, cls_tok = feat
                multi_feats.append(patch_tok.float())
                if idx == self.selected_layers[-1]:
                    cls_token = cls_tok.float()
            else:
                feat = feat.float()
                if feat.shape[1] == h * w + 1:
                    multi_feats.append(feat[:, 1:])
                    if idx == self.selected_layers[-1]:
                        cls_token = feat[:, 0]
                else:
                    multi_feats.append(feat)
                    if idx == self.selected_layers[-1]:
                        cls_token = feat.mean(dim=1)

        if cls_token is None:
            cls_token = multi_feats[-1].mean(dim=1)

        return multi_feats, cls_token, (h, w)