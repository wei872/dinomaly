"""主训练脚本 - 适配 ViT-S/14 轻量模型"""
import os
import random
import numpy as np
import torch

from config import get_config
from models.model import DinoDefectClassifier
from data.dataset import build_dataloaders
from engine.trainer import Trainer
from utils.logger import Logger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    config = get_config()
    set_seed(config.seed)

    os.makedirs(config.train.save_dir, exist_ok=True)
    os.makedirs(config.train.log_dir, exist_ok=True)
    config.save(os.path.join(config.train.save_dir, 'config.yaml'))

    logger = Logger(config.train.log_dir)
    logger.info(f"Config: {config}")
    logger.info(f"Device: {config.device}")

    # 构建数据加载器
    logger.info("Building dataloaders...")
    dataloaders = build_dataloaders(
        data_root=config.data.data_root,
        img_size=config.data.img_size,
        batch_size_s1=config.train.stage1_batch_size,
        batch_size_s2=config.train.stage2_batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    # 构建模型 — 参数与 DinoDefectClassifier.__init__ 完全对齐
    logger.info("Building model...")
    model = DinoDefectClassifier(
        num_classes=config.data.num_classes,
        encoder_name=config.model.encoder_name,          # dinov2_vits14
        selected_layers=config.model.selected_layers,    # [9, 10, 11]
        bottleneck_dim=config.model.bottleneck_dim,      # 96
        fused_dim=config.model.fused_dim,                # 256
        noise_std=config.model.noise_std,                # 0.1
        drop_rate=config.model.drop_rate,                # 0.1
        img_size=config.data.img_size,                   # 518
    ).to(config.device)

    # 打印参数统计
    param_info = model.get_param_count()
    logger.info("=" * 50)
    logger.info("Model Parameter Summary:")
    for name, info in param_info.items():
        logger.info(
            f"  {name:20s}: {info['total_M']:8.2f}M total, "
            f"{info['trainable_M']:8.2f}M trainable"
        )
    logger.info("=" * 50)

    # 训练
    trainer = Trainer(model, dataloaders, config, logger)
    trainer.train()

    logger.close()
    print("\n Training completed successfully!")


if __name__ == '__main__':
    main()