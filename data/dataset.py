# data/dataset.py
"""缺陷检测数据集"""
import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable, Dict

from .transforms import get_train_transform, get_val_transform, get_mask_transform


class DefectDataset(Dataset):
    """
    缺陷检测数据集

    数据格式（兼容 YOLO 风格）：
    root/
    ├── images/
    │   ├── 000001.jpg
    │   └── ...
    ├── masks/
    │   ├── 000001.png      # 正常图全黑，缺陷图标注区域为255
    │   └── ...
    └── labels.csv
        image_name,class_id,class_name
        000001.jpg,0,normal
        000002.jpg,1,scratch
        ...

    也支持无 CSV，从文件夹结构推断：
    root/
    ├── normal/
    │   ├── images/
    │   └── masks/
    ├── scratch/
    │   ├── images/
    │   └── masks/
    └── ...
    """

    def __init__(
            self,
            root: str,
            img_size: int = 518,
            is_train: bool = True,
            normal_only: bool = False,
            transform: Optional[Callable] = None,
            mask_transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.root = root
        self.img_size = img_size
        self.is_train = is_train
        self.normal_only = normal_only

        # 设置变换
        if transform is None:
            self.transform = get_train_transform(img_size) if is_train else get_val_transform(img_size)
        else:
            self.transform = transform

        if mask_transform is None:
            self.mask_transform = get_mask_transform(img_size)
        else:
            self.mask_transform = mask_transform

        # 加载数据列表
        self.samples = self._load_samples()

        if normal_only:
            self.samples = [s for s in self.samples if s['class_id'] == 0]

        print(f"[Dataset] Loaded {len(self.samples)} samples from {root}")
        self._print_class_distribution()

    def _load_samples(self):
        """加载样本列表"""
        samples = []

        csv_path = os.path.join(self.root, 'labels.csv')
        if os.path.exists(csv_path):
            samples = self._load_from_csv(csv_path)
        else:
            samples = self._load_from_folders()

        return samples

    def _load_from_csv(self, csv_path: str):
        """从 CSV 加载"""
        df = pd.read_csv(csv_path)
        samples = []

        for _, row in df.iterrows():
            img_name = row['image_name']
            class_id = int(row['class_id'])
            class_name = str(row.get('class_name', f'class_{class_id}'))

            img_path = os.path.join(self.root, 'images', img_name)

            # 掩膜路径
            mask_name = os.path.splitext(img_name)[0] + '.png'
            mask_path = os.path.join(self.root, 'masks', mask_name)

            if not os.path.exists(img_path):
                continue

            samples.append({
                'img_path': img_path,
                'mask_path': mask_path if os.path.exists(mask_path) else None,
                'class_id': class_id,
                'class_name': class_name,
                'image_name': img_name,
            })

        return samples

    def _load_from_folders(self):
        """从文件夹结构加载"""
        samples = []
        class_dirs = sorted([
            d for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d))
        ])

        for class_id, class_name in enumerate(class_dirs):
            class_dir = os.path.join(self.root, class_name)
            img_dir = os.path.join(class_dir, 'images')
            mask_dir = os.path.join(class_dir, 'masks')

            if not os.path.isdir(img_dir):
                # 直接在 class_dir 下寻找图片
                img_dir = class_dir
                mask_dir = None

            for img_file in sorted(os.listdir(img_dir)):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue

                img_path = os.path.join(img_dir, img_file)
                mask_path = None

                if mask_dir:
                    mask_name = os.path.splitext(img_file)[0] + '.png'
                    mask_candidate = os.path.join(mask_dir, mask_name)
                    if os.path.exists(mask_candidate):
                        mask_path = mask_candidate

                samples.append({
                    'img_path': img_path,
                    'mask_path': mask_path,
                    'class_id': class_id,
                    'class_name': class_name,
                    'image_name': img_file,
                })

        return samples

    def _print_class_distribution(self):
        """打印类别分布"""
        from collections import Counter
        counter = Counter(s['class_name'] for s in self.samples)
        print(f"  Class distribution:")
        for cls_name, count in sorted(counter.items()):
            print(f"    {cls_name}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # 加载图像
        image = Image.open(sample['img_path']).convert('RGB')

        # 加载掩膜
        if sample['mask_path'] and os.path.exists(sample['mask_path']):
            mask = Image.open(sample['mask_path']).convert('L')
        else:
            # 无掩膜时创建全黑掩膜（正常图像）
            mask = Image.new('L', image.size, 0)

        # 应用变换
        # 注意：训练时需要对图像和掩膜做一致的几何变换
        if self.is_train:
            image, mask = self._sync_transform(image, mask)
        else:
            image = self.transform(image)
            mask = self.mask_transform(mask)

        # 二值化掩膜
        mask = (mask > 0.5).float()

        # 类别标签
        class_id = sample['class_id']

        # 是否异常
        is_anomaly = 1.0 if class_id > 0 else 0.0

        return {
            'image': image,  # (3, H, W)
            'mask': mask,  # (1, H, W)
            'label': torch.tensor(class_id, dtype=torch.long),
            'is_anomaly': torch.tensor([is_anomaly], dtype=torch.float32),
            'image_name': sample['image_name'],
        }

    def _sync_transform(self, image, mask):
        """同步图像和掩膜的几何变换"""
        import torchvision.transforms.functional as TF
        import random

        # Resize
        image = TF.resize(image, (self.img_size, self.img_size))
        mask = TF.resize(mask, (self.img_size, self.img_size),
                         interpolation=T.InterpolationMode.NEAREST)

        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        # Color jitter (只对图像)
        if random.random() > 0.5:
            image = T.ColorJitter(0.2, 0.2, 0.1, 0.05)(image)

        # To tensor + normalize
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        mask = TF.to_tensor(mask)

        return image, mask

    def get_class_counts(self) -> list:
        """
        自动统计每个类别的样本数量
        Returns:
            counts: list[int], 长度=num_classes, counts[i]=class_id为i的样本数
        """
        from collections import Counter
        counter = Counter(s['class_id'] for s in self.samples)

        # 找到最大 class_id
        max_id = max(counter.keys()) if counter else 0

        # 按 class_id 顺序排列，缺失的类填1（避免除零）
        counts = []
        for i in range(max_id + 1):
            counts.append(counter.get(i, 1))

        return counts

    def get_class_weights(self, method: str = 'inverse_sqrt') -> torch.Tensor:
        """
        自动计算类别权重
        Args:
            method: 'inverse_sqrt' | 'inverse' | 'effective'
        """
        counts = self.get_class_counts()
        counts_tensor = torch.tensor(counts, dtype=torch.float32)

        if method == 'inverse':
            weights = 1.0 / counts_tensor
        elif method == 'inverse_sqrt':
            weights = 1.0 / torch.sqrt(counts_tensor)
        elif method == 'effective':
            beta = 0.9999
            effective_num = 1.0 - torch.pow(beta, counts_tensor)
            weights = (1.0 - beta) / effective_num
        else:
            weights = torch.ones_like(counts_tensor)

        weights = weights / weights.mean()
        return weights


import torchvision.transforms as T


# data/dataset.py — 修改 build_dataloaders 函数

def build_dataloaders(
        data_root: str,
        img_size: int = 518,
        batch_size_s1: int = 16,
        batch_size_s2: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
):
    """构建所有数据加载器"""

    train_normal = DefectDataset(
        os.path.join(data_root, 'train'),
        img_size=img_size, is_train=True, normal_only=True,
    )

    train_full = DefectDataset(
        os.path.join(data_root, 'train'),
        img_size=img_size, is_train=True, normal_only=False,
    )

    val_set = DefectDataset(
        os.path.join(data_root, 'val'),
        img_size=img_size, is_train=False,
    )

    test_path = os.path.join(data_root, 'test')
    test_set = None
    if os.path.exists(test_path):
        test_set = DefectDataset(test_path, img_size=img_size, is_train=False)

    loader_normal = DataLoader(
        train_normal, batch_size=batch_size_s1,
        shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory, drop_last=True,
    )
    loader_full = DataLoader(
        train_full, batch_size=batch_size_s2,
        shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory, drop_last=True,
    )
    loader_val = DataLoader(
        val_set, batch_size=batch_size_s2,
        shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory,
    )
    loader_test = None
    if test_set:
        loader_test = DataLoader(
            test_set, batch_size=batch_size_s2,
            shuffle=False, num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return {
        'train_normal': loader_normal,
        'train_full': loader_full,
        'val': loader_val,
        'test': loader_test,
        # ★ 返回数据集对象，供后续自动计算权重
        'train_full_dataset': train_full,
    }