# engine/evaluator.py
"""评估器"""
import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm
from typing import Dict, List, Optional

from models.model import DinoDefectClassifier
from utils.metrics import MetricTracker
from utils.visualization import Visualizer


class Evaluator:
    """模型评估器"""

    def __init__(
            self,
            model: DinoDefectClassifier,
            dataloader,
            num_classes: int,
            class_names: List[str],
            device: str = 'cuda',
            save_dir: str = './eval_results',
    ):
        self.model = model
        self.dataloader = dataloader
        self.num_classes = num_classes
        self.class_names = class_names
        self.device = device
        self.save_dir = save_dir

        os.makedirs(save_dir, exist_ok=True)

        self.visualizer = Visualizer(class_names, save_dir)
        self.metric_tracker = MetricTracker(num_classes, class_names)

    @torch.no_grad()
    def evaluate(self, visualize_num: int = 20) -> Dict:
        """
        完整评估

        Args:
            visualize_num: 可视化的样本数量

        Returns:
            metrics: 所有评估指标
        """
        self.model.eval()
        self.metric_tracker.reset()

        vis_count = 0

        for batch in tqdm(self.dataloader, desc="Evaluating"):
            images = batch['image'].to(self.device)
            targets = {
                'class_labels': batch['label'].to(self.device),
                'masks': batch['mask'].to(self.device),
                'is_anomaly': batch['is_anomaly'].to(self.device),
            }

            with autocast():
                outputs = self.model(images)

            self.metric_tracker.update(
                class_logits=outputs['class_logits'],
                class_labels=targets['class_labels'],
                anomaly_prob=outputs['anomaly_prob'],
                is_anomaly=targets['is_anomaly'],
                mask_pred=outputs['mask_pred'],
                mask_gt=targets['masks'],
            )

            # 可视化
            if vis_count < visualize_num:
                probs = F.softmax(outputs['class_logits'], dim=1)
                preds = probs.argmax(dim=1)

                for i in range(min(images.size(0), visualize_num - vis_count)):
                    self.visualizer.visualize_prediction(
                        image=images[i],
                        mask_gt=targets['masks'][i],
                        mask_pred=outputs['mask_pred'][i],
                        anomaly_map=outputs['anomaly_map'][i],
                        class_name=self.class_names[targets['class_labels'][i].item()],
                        pred_class=self.class_names[preds[i].item()],
                        confidence=probs[i, preds[i]].item(),
                        anomaly_score=outputs['anomaly_prob'][i, 0].item(),
                        save_name=f'pred_{vis_count:04d}.png',
                    )
                    vis_count += 1

        # 计算指标
        metrics = self.metric_tracker.compute()

        # 打印报告
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        print(self.metric_tracker.get_classification_report())

        # 混淆矩阵
        cm = self.metric_tracker.get_confusion_matrix()
        self.visualizer.plot_confusion_matrix(cm)

        print(f"\nOverall Accuracy: {metrics['accuracy']:.2f}%")
        print(f"Macro F1: {metrics['f1_macro']:.2f}%")
        if 'anomaly_auc' in metrics:
            print(f"Anomaly AUC: {metrics['anomaly_auc']:.2f}%")
        if 'mask_iou' in metrics:
            print(f"Mask IoU: {metrics['mask_iou']:.2f}%")

        return metrics