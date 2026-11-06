# losses/loss.py
"""多任务损失函数 - 修复类别不平衡"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Focal Loss - 专治类别不平衡"""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(pred, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def compute_class_weights(class_counts: list, method: str = 'inverse_sqrt') -> torch.Tensor:
    """
    根据类别样本数计算权重

    Args:
        class_counts: 每个类的样本数
        method:
            'inverse': 1/count
            'inverse_sqrt': 1/sqrt(count)  ← 推荐，不会太极端
            'effective': effective number of samples
    """
    counts = torch.tensor(class_counts, dtype=torch.float32)

    if method == 'inverse':
        weights = 1.0 / counts
    elif method == 'inverse_sqrt':
        weights = 1.0 / torch.sqrt(counts)
    elif method == 'effective':
        beta = 0.9999
        effective_num = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / effective_num
    else:
        weights = torch.ones_like(counts)

    # 归一化：使权重均值为1
    weights = weights / weights.mean()

    return weights


class DinoDefectLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 9,
        lambda_cls: float = 1.0,
        lambda_mask: float = 1.0,
        lambda_recon: float = 0.5,
        lambda_anomaly: float = 0.3,
        label_smoothing: float = 0.1,
        focal_gamma: float = 2.0,
        class_weights: torch.Tensor = None,   # ★ 只接收计算好的权重
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_mask = lambda_mask
        self.lambda_recon = lambda_recon
        self.lambda_anomaly = lambda_anomaly

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
        self.focal_loss = FocalLoss(gamma=focal_gamma, weight=class_weights)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.anomaly_bce = nn.BCELoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        stage: int = 2,
    ) -> Dict[str, torch.Tensor]:

        losses = {}

        if stage == 1:
            losses['recon'] = outputs['recon_loss']
            losses['total'] = losses['recon']
            return losses

        # === Stage 2 ===

        # 1. 分类损失
        class_logits = outputs['class_logits']
        class_labels = targets['class_labels']

        # ★ 确保权重在同一设备
        if self.class_weights is not None:
            self.ce_loss.weight = self.class_weights.to(class_logits.device)
            self.focal_loss.weight = self.class_weights.to(class_logits.device)

        loss_ce = self.ce_loss(class_logits, class_labels)
        loss_focal = self.focal_loss(class_logits, class_labels)
        loss_cls = 0.5 * loss_ce + 0.5 * loss_focal
        losses['cls'] = loss_cls

        # 2. 分割损失
        mask_pred = outputs['mask_pred']
        mask_gt = targets['masks']

        if mask_pred.shape[-2:] != mask_gt.shape[-2:]:
            mask_gt = F.interpolate(mask_gt, size=mask_pred.shape[-2:], mode='nearest')

        loss_bce = self.bce_loss(mask_pred, mask_gt)
        loss_dice = self.dice_loss(mask_pred, mask_gt)
        loss_mask = loss_bce + loss_dice
        losses['mask'] = loss_mask

        # 3. 重建损失
        losses['recon'] = outputs['recon_loss']

        # 4. 异常检测损失
        anomaly_prob = outputs['anomaly_prob']
        is_anomaly = targets['is_anomaly']
        loss_anomaly = self.anomaly_bce(anomaly_prob, is_anomaly)
        losses['anomaly'] = loss_anomaly

        # 总损失
        total = (
            self.lambda_cls * loss_cls +
            self.lambda_mask * loss_mask +
            self.lambda_recon * losses['recon'] +
            self.lambda_anomaly * loss_anomaly
        )
        losses['total'] = total

        return losses