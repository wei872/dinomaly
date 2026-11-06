"""训练器 - 适配轻量 ViT-S/14 模型"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict
from tqdm import tqdm

from models.model import DinoDefectClassifier


class Trainer:
    def __init__(self, model: DinoDefectClassifier, dataloaders: Dict[str, DataLoader],
                 config, logger):
        self.model = model
        self.dataloaders = dataloaders
        self.config = config
        self.logger = logger
        self.device = config.device

        self.criterion_cls = nn.CrossEntropyLoss(
            label_smoothing=config.train.label_smoothing
        )
        self.criterion_mask = nn.BCEWithLogitsLoss()

        self.optimizer = None
        self.scheduler = None
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0

        # 打印可用的 dataloader keys
        available = [k for k, v in self.dataloaders.items() if v is not None and not k.endswith('_dataset')]
        self.logger.info(f"Available dataloaders: {available}")

    def _get_optimizer(self, stage: int = 2):
        lr = self.config.train.stage1_lr if stage == 1 else self.config.train.stage2_lr
        params = self.model.get_trainable_params(base_lr=lr)
        return torch.optim.AdamW(params, weight_decay=self.config.train.weight_decay)

    def _get_scheduler(self, optimizer, total_epochs):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=1e-6
        )

    def _get_loader(self, *keys) -> DataLoader:
        """按优先级查找 dataloader，找到第一个存在的就返回"""
        for key in keys:
            loader = self.dataloaders.get(key)
            if loader is not None:
                self.logger.info(f"Using dataloader: '{key}'")
                return loader

        # 兜底：返回任意可用的
        for key, loader in self.dataloaders.items():
            if loader is not None and not key.endswith('_dataset'):
                self.logger.info(f"Fallback to dataloader: '{key}'")
                return loader

        raise RuntimeError(f"No dataloader found! Tried: {keys}")

    def train(self):
        self.logger.info("=" * 60)
        self.logger.info("Stage 1: Reconstruction Pretraining (Normal samples)")
        self.logger.info("=" * 60)
        self.train_stage1()

        self.logger.info("=" * 60)
        self.logger.info("Stage 2: Full Multi-Task Training")
        self.logger.info("=" * 60)
        self.train_stage2()

        self.logger.info("Training completed!")

    # ===================== Stage 1 =====================
    def train_stage1(self):
        self.optimizer = self._get_optimizer(stage=1)
        scheduler = self._get_scheduler(self.optimizer, self.config.train.stage1_epochs)

        # 兼容多种 key 命名
        loader = self._get_loader('train_normal', 'normal_train', 'train_full', 'train')

        avg_loss = 0.0
        for epoch in range(self.config.train.stage1_epochs):
            self.model.train()
            total_loss = 0.0
            num_batches = 0

            pbar = tqdm(loader,
                        desc=f"S1 [{epoch+1}/{self.config.train.stage1_epochs}]",
                        leave=False)

            for batch in pbar:
                images = batch['image'].to(self.device)

                outputs = self.model(images)
                recon_loss = outputs['recon_loss']

                self.optimizer.zero_grad()
                recon_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.train.max_grad_norm
                )
                self.optimizer.step()

                total_loss += recon_loss.item()
                num_batches += 1
                pbar.set_postfix({'recon': f"{recon_loss.item():.4f}"})

            scheduler.step()
            avg_loss = total_loss / max(num_batches, 1)
            lr = self.optimizer.param_groups[0]['lr']

            self.logger.info(
                f"Stage1 Epoch {epoch+1}/{self.config.train.stage1_epochs} - "
                f"Recon Loss: {avg_loss:.4f}, LR: {lr:.6f}"
            )

            if (epoch + 1) % self.config.train.save_interval == 0:
                self._save_checkpoint(f'stage1_ep{epoch+1}', epoch, avg_loss, stage=1)

        self._save_checkpoint('stage1_final', self.config.train.stage1_epochs, avg_loss, stage=1)
        self.logger.info("Stage 1 completed.")

    # ===================== Stage 2 =====================
    def train_stage2(self):
        self.optimizer = self._get_optimizer(stage=2)
        scheduler = self._get_scheduler(self.optimizer, self.config.train.stage2_epochs)

        # 兼容多种 key 命名
        train_loader = self._get_loader('train_full', 'train', 'train_normal')
        val_loader = self.dataloaders.get('val')

        avg_loss = 0.0
        for epoch in range(self.config.train.stage2_epochs):
            # --- Train ---
            train_metrics = self._train_one_epoch(train_loader, epoch)
            scheduler.step()

            lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(
                f"Stage2 Epoch {epoch+1}/{self.config.train.stage2_epochs} - "
                f"Loss: {train_metrics['loss']:.4f}, "
                f"Cls: {train_metrics['cls']:.4f}, "
                f"Recon: {train_metrics['recon']:.4f}, "
                f"Acc: {train_metrics['acc']:.2f}%, "
                f"LR: {lr:.6f}"
            )

            # --- Validate ---
            if val_loader is not None and (epoch + 1) % self.config.train.eval_interval == 0:
                val_metrics = self._validate(val_loader)
                self.logger.info(
                    f"  Val - Loss: {val_metrics['loss']:.4f}, "
                    f"Acc: {val_metrics['acc']:.2f}%"
                )

                if val_metrics['acc'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['acc']
                    self._save_checkpoint('best', epoch, val_metrics['loss'], stage=2)
                    self.logger.info(f"  >> New best! Acc: {self.best_val_acc:.2f}%")

            # --- Save ---
            if (epoch + 1) % self.config.train.save_interval == 0:
                self._save_checkpoint(
                    f'stage2_ep{epoch+1}', epoch, train_metrics['loss'], stage=2
                )

        self._save_checkpoint('stage2_final', self.config.train.stage2_epochs,
                              train_metrics['loss'], stage=2)
        self.logger.info(f"Stage 2 completed. Best Val Acc: {self.best_val_acc:.2f}%")

    def _train_one_epoch(self, loader: DataLoader, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        total_cls = 0.0
        total_recon = 0.0
        correct = 0
        total_samples = 0
        num_batches = 0

        pbar = tqdm(loader,
                    desc=f"S2 [{epoch+1}/{self.config.train.stage2_epochs}]",
                    leave=False)

        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            masks = batch.get('mask')
            if masks is not None:
                masks = masks.to(self.device)

            outputs = self.model(images)

            # === Losses ===
            loss_cls = self.criterion_cls(outputs['class_logits'], labels)
            loss_recon = outputs['recon_loss']

            # Anomaly binary loss
            is_anomaly = (labels != 0).float()
            anomaly_prob = outputs['anomaly_prob'].squeeze(-1)
            # 确保维度匹配
            if anomaly_prob.dim() == 0:
                anomaly_prob = anomaly_prob.unsqueeze(0)
            loss_anomaly = F.binary_cross_entropy(anomaly_prob, is_anomaly)

            # Mask loss
            loss_mask = torch.tensor(0.0, device=self.device)
            if masks is not None:
                mask_pred = outputs['mask_pred']
                # 确保 mask 维度匹配 mask_pred
                if masks.dim() == 3:
                    masks = masks.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
                if masks.shape != mask_pred.shape:
                    masks = F.interpolate(
                        masks.float(), size=mask_pred.shape[2:],
                        mode='nearest'
                    )
                loss_mask = self.criterion_mask(mask_pred, masks)

            # Weighted total
            loss = (
                self.config.train.lambda_cls * loss_cls +
                self.config.train.lambda_recon * loss_recon +
                self.config.train.lambda_anomaly * loss_anomaly +
                self.config.train.lambda_mask * loss_mask
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.train.max_grad_norm
            )
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            total_cls += loss_cls.item()
            total_recon += loss_recon.item()
            preds = outputs['class_logits'].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.3f}",
                'cls': f"{loss_cls.item():.3f}",
                'acc': f"{100.0 * correct / total_samples:.1f}%"
            })

        n = max(num_batches, 1)
        return {
            'loss': total_loss / n,
            'cls': total_cls / n,
            'recon': total_recon / n,
            'acc': 100.0 * correct / max(total_samples, 1),
        }

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> dict:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        num_batches = 0

        for batch in loader:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            outputs = self.model(images)
            loss_cls = self.criterion_cls(outputs['class_logits'], labels)

            total_loss += loss_cls.item()
            preds = outputs['class_logits'].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            num_batches += 1

        n = max(num_batches, 1)
        return {
            'loss': total_loss / n,
            'acc': 100.0 * correct / max(total_samples, 1),
        }

    def _save_checkpoint(self, name: str, epoch: int, loss: float, stage: int = 2):
        checkpoint = {
            'epoch': epoch,
            'stage': stage,
            'loss': loss,
            'best_val_acc': self.best_val_acc,
            'model_state_dict': {
                k: v for k, v in self.model.state_dict().items()
                if not k.startswith('encoder.')
            },
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'encoder_name': self.config.model.encoder_name,
                'selected_layers': self.config.model.selected_layers,
                'bottleneck_dim': self.config.model.bottleneck_dim,
                'fused_dim': self.config.model.fused_dim,
                'noise_std': self.config.model.noise_std,
                'drop_rate': self.config.model.drop_rate,
                'img_size': self.config.data.img_size,
                'num_classes': self.config.data.num_classes,
            },
        }

        save_path = os.path.join(self.config.train.save_dir, f'{name}.pth')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(checkpoint, save_path)
        self.logger.info(f"Saved: {save_path}")