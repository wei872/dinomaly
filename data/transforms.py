# data/transforms.py
"""数据增强"""
import torchvision.transforms as T


def get_train_transform(img_size: int = 518):
    """训练集数据增强"""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
        ),
        T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05,
        ),
        T.RandomGrayscale(p=0.05),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_val_transform(img_size: int = 518):
    """验证/测试集变换"""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_mask_transform(img_size: int = 518):
    """掩膜变换"""
    return T.Compose([
        T.Resize(
            (img_size, img_size),
            interpolation=T.InterpolationMode.NEAREST,
        ),
        T.ToTensor(),
    ])


def get_inverse_normalize():
    """反归一化，用于可视化"""
    return T.Compose([
        T.Normalize(
            mean=[0., 0., 0.],
            std=[1/0.229, 1/0.224, 1/0.225],
        ),
        T.Normalize(
            mean=[-0.485, -0.456, -0.406],
            std=[1., 1., 1.],
        ),
    ])