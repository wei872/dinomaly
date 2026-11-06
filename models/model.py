"""完整模型 - ViT-S/14 最小配置"""
import torch
import torch.nn as nn
from typing import Dict, List

from .encoder import DINOv2Encoder
from .decoder import LightweightAnomalyDetector
from .fusion import LightFusion
from .heads import ClassificationHead, MaskHead, AnomalyScoreHead


class DinoDefectClassifier(nn.Module):
    """
    DinoDefectClassifier v2-Small

    Encoder: DINOv2 ViT-S/14 (dim=384, 冻结 21M)
    可训练参数: ~1.3M
    """

    def __init__(
        self,
        num_classes: int = 9,
        encoder_name: str = 'dinov2_vits14',
        selected_layers: List[int] = None,
        bottleneck_dim: int = 96,
        fused_dim: int = 256,
        noise_std: float = 0.1,
        drop_rate: float = 0.1,
        img_size: int = 518,
    ):
        super().__init__()

        if selected_layers is None:
            selected_layers = [9, 10, 11]

        self.img_size = img_size
        self.num_classes = num_classes
        num_enc_layers = len(selected_layers)

        # Encoder (Frozen)
        self.encoder = DINOv2Encoder(encoder_name, selected_layers)
        encoder_dim = self.encoder.embed_dim
        patch_size = self.encoder.patch_size

        # 轻量异常检测（保留别名兼容旧trainer）
        self.anomaly_detector = LightweightAnomalyDetector(
            encoder_dim=encoder_dim,
            bottleneck_dim=bottleneck_dim,
            num_layers=num_enc_layers,
            noise_std=noise_std,
        )
        # ★ 别名，兼容旧版 trainer 调用 model.recon_branch
        self.recon_branch = self.anomaly_detector

        # 轻量融合
        self.fusion = LightFusion(
            encoder_dim=encoder_dim,
            anomaly_dim=bottleneck_dim,
            out_dim=fused_dim,
        )

        # Task Heads
        self.cls_head = ClassificationHead(
            in_dim=fused_dim,
            encoder_dim=encoder_dim,
            num_layers=num_enc_layers,
            num_classes=num_classes,
            drop=0.3,
        )
        self.mask_head = MaskHead(
            in_dim=fused_dim,
            img_size=img_size,
            patch_size=patch_size,
        )
        self.anomaly_head = AnomalyScoreHead(
            in_dim=fused_dim,
            drop=0.2,
        )

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        multi_feats, cls_token, (h, w) = self.encoder(images)

        anomaly_features, anomaly_scores, recon_loss = self.anomaly_detector(multi_feats)

        orig_features = multi_feats[-1]
        fused = self.fusion(orig_features, anomaly_features, anomaly_scores)

        class_logits = self.cls_head(fused, multi_feats, anomaly_scores)
        mask_pred = self.mask_head(fused)
        anomaly_prob = self.anomaly_head(fused)

        return {
            'class_logits': class_logits,
            'mask_pred': mask_pred,
            'anomaly_prob': anomaly_prob,
            'anomaly_map': anomaly_scores,
            'recon_loss': recon_loss,
            'spatial_shape': (h, w),
        }

    def get_trainable_params(self, stage: int = 2, base_lr: float = 5e-4) -> list:
        """
        兼容旧版两阶段 trainer 接口

        stage=1: 只训练异常检测模块
        stage=2: 训练全部可训练模块
        """
        if stage == 1:
            return [
                {
                    'params': self.anomaly_detector.parameters(),
                    'lr': base_lr,
                    'name': 'anomaly_detector',
                },
            ]
        else:
            return [
                {
                    'params': self.anomaly_detector.parameters(),
                    'lr': base_lr * 0.1,
                    'name': 'anomaly_detector',
                },
                {
                    'params': self.fusion.parameters(),
                    'lr': base_lr,
                    'name': 'fusion',
                },
                {
                    'params': self.cls_head.parameters(),
                    'lr': base_lr,
                    'name': 'cls_head',
                },
                {
                    'params': self.mask_head.parameters(),
                    'lr': base_lr,
                    'name': 'mask_head',
                },
                {
                    'params': self.anomaly_head.parameters(),
                    'lr': base_lr,
                    'name': 'anomaly_head',
                },
            ]

    def get_param_count(self) -> dict:
        def count(module):
            t = sum(p.numel() for p in module.parameters())
            tr = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return t, tr

        info = {}
        for name, mod in [
            ('encoder', self.encoder),
            ('anomaly_detector', self.anomaly_detector),
            ('fusion', self.fusion),
            ('cls_head', self.cls_head),
            ('mask_head', self.mask_head),
            ('anomaly_head', self.anomaly_head),
        ]:
            t, tr = count(mod)
            info[name] = {'total_M': t / 1e6, 'trainable_M': tr / 1e6}

        all_t = sum(v['total_M'] for v in info.values())
        all_tr = sum(v['trainable_M'] for v in info.values())
        info['summary'] = {'total_M': all_t, 'trainable_M': all_tr}
        return info