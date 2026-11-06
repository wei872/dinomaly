# utils/metrics.py
"""评估指标"""
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
)
from typing import Dict, List, Optional


class MetricTracker:
    """指标追踪器"""

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        self.reset()

    def reset(self):
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
        self.all_anomaly_scores = []
        self.all_is_anomaly = []
        self.mask_ious = []

    def update(
            self,
            class_logits: torch.Tensor,
            class_labels: torch.Tensor,
            anomaly_prob: Optional[torch.Tensor] = None,
            is_anomaly: Optional[torch.Tensor] = None,
            mask_pred: Optional[torch.Tensor] = None,
            mask_gt: Optional[torch.Tensor] = None,
    ):
        """更新指标"""
        # 分类
        probs = torch.softmax(class_logits, dim=1)
        preds = probs.argmax(dim=1)

        self.all_preds.extend(preds.cpu().numpy().tolist())
        self.all_labels.extend(class_labels.cpu().numpy().tolist())
        self.all_probs.extend(probs.cpu().numpy().tolist())

        # 异常检测
        if anomaly_prob is not None and is_anomaly is not None:
            self.all_anomaly_scores.extend(anomaly_prob.cpu().numpy().flatten().tolist())
            self.all_is_anomaly.extend(is_anomaly.cpu().numpy().flatten().tolist())

        # 分割 IoU
        if mask_pred is not None and mask_gt is not None:
            mask_pred_bin = (torch.sigmoid(mask_pred) > 0.5).float()
            for i in range(mask_pred_bin.shape[0]):
                pred_m = mask_pred_bin[i].cpu().numpy().flatten()
                gt_m = mask_gt[i].cpu().numpy().flatten()
                if gt_m.sum() > 0:  # 只对有缺陷的样本计算 IoU
                    intersection = (pred_m * gt_m).sum()
                    union = pred_m.sum() + gt_m.sum() - intersection
                    iou = intersection / (union + 1e-8)
                    self.mask_ious.append(iou)

    def compute(self) -> Dict[str, float]:
        """计算所有指标"""
        results = {}

        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)

        # 分类指标
        results['accuracy'] = accuracy_score(labels, preds) * 100
        results['precision_macro'] = precision_score(
            labels, preds, average='macro', zero_division=0
        ) * 100
        results['recall_macro'] = recall_score(
            labels, preds, average='macro', zero_division=0
        ) * 100
        results['f1_macro'] = f1_score(
            labels, preds, average='macro', zero_division=0
        ) * 100

        # 每类指标
        per_class_f1 = f1_score(
            labels, preds, average=None, zero_division=0
        )
        for i, name in enumerate(self.class_names):
            if i < len(per_class_f1):
                results[f'f1_{name}'] = per_class_f1[i] * 100

        # 异常检测 AUC
        if len(self.all_anomaly_scores) > 0:
            try:
                results['anomaly_auc'] = roc_auc_score(
                    self.all_is_anomaly, self.all_anomaly_scores
                ) * 100
            except ValueError:
                results['anomaly_auc'] = 0.0

        # 分割 IoU
        if len(self.mask_ious) > 0:
            results['mask_iou'] = np.mean(self.mask_ious) * 100

        return results

    def get_classification_report(self) -> str:
        """获取详细分类报告"""
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)

        return classification_report(
            labels, preds,
            target_names=self.class_names,
            digits=4,
            zero_division=0,
        )

    def get_confusion_matrix(self) -> np.ndarray:
        """获取混淆矩阵"""
        return confusion_matrix(
            np.array(self.all_labels),
            np.array(self.all_preds),
        )