# utils/visualization.py
"""可视化工具"""
import torch
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from PIL import Image
from typing import List, Optional
import os


class Visualizer:
    """可视化工具集"""

    def __init__(self, class_names: List[str], save_dir: str = './vis'):
        self.class_names = class_names
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 反归一化
        self.mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """反归一化图像 tensor"""
        return tensor.cpu() * self.std + self.mean

    def tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """将 tensor 转为 numpy 图像 (H, W, 3) uint8"""
        img = self.denormalize(tensor)
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()
        return (img * 255).astype(np.uint8)

    def visualize_prediction(
            self,
            image: torch.Tensor,
            mask_gt: torch.Tensor,
            mask_pred: torch.Tensor,
            anomaly_map: torch.Tensor,
            class_name: str,
            pred_class: str,
            confidence: float,
            anomaly_score: float,
            save_name: str = None,
    ):
        """
        可视化单张预测结果
        """
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))

        img_np = self.tensor_to_numpy(image)

        # 1. 原图
        axes[0].imshow(img_np)
        axes[0].set_title(f'GT: {class_name}', fontsize=12)

        # 2. 预测结果
        axes[1].imshow(img_np)
        color = 'green' if class_name == pred_class else 'red'
        axes[1].set_title(
            f'Pred: {pred_class}\nConf: {confidence:.3f}',
            fontsize=12, color=color,
        )

        # 3. 真实掩膜
        mask_gt_np = mask_gt.cpu().squeeze().numpy()
        axes[2].imshow(mask_gt_np, cmap='gray')
        axes[2].set_title('GT Mask', fontsize=12)

        # 4. 预测掩膜
        mask_pred_np = torch.sigmoid(mask_pred).cpu().squeeze().numpy()
        axes[3].imshow(mask_pred_np, cmap='hot')
        axes[3].set_title(f'Pred Mask\nAnomaly: {anomaly_score:.3f}', fontsize=12)

        # 5. 异常热力图叠加
        anomaly_np = anomaly_map.cpu().squeeze().numpy()
        h = w = int(np.sqrt(anomaly_np.size))
        anomaly_np = anomaly_np.reshape(h, w)
        anomaly_np = np.array(Image.fromarray(anomaly_np).resize(
            (img_np.shape[1], img_np.shape[0]), resample=Image.BILINEAR
        ))

        axes[4].imshow(img_np)
        axes[4].imshow(anomaly_np, alpha=0.5, cmap='jet')
        axes[4].set_title('Anomaly Heatmap', fontsize=12)

        for ax in axes:
            ax.axis('off')

        plt.tight_layout()

        if save_name:
            save_path = os.path.join(self.save_dir, save_name)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.close()

    def plot_confusion_matrix(
            self,
            cm: np.ndarray,
            save_name: str = 'confusion_matrix.png',
    ):
        """绘制混淆矩阵"""
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
        )
        ax.set_xlabel('Predicted', fontsize=14)
        ax.set_ylabel('True', fontsize=14)
        ax.set_title('Confusion Matrix', fontsize=16)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150)
        plt.close()

    def plot_training_curves(
            self,
            train_losses: List[float],
            val_losses: List[float],
            train_accs: List[float],
            val_accs: List[float],
            save_name: str = 'training_curves.png',
    ):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(train_losses) + 1)

        ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
        if val_losses:
            ax1.plot(epochs[:len(val_losses)], val_losses, 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curves')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(epochs, train_accs, 'b-', label='Train Acc')
        if val_accs:
            ax2.plot(epochs[:len(val_accs)], val_accs, 'r-', label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy Curves')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150)
        plt.close()