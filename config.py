"""配置解析 - 适配 ViT-S/14 轻量模型，支持命令行参数覆盖"""
import yaml
import os
import argparse
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    data_root: str = "./dataset/dino"
    img_size: int = 518
    num_classes: int = 9
    class_names: List[str] = field(default_factory=lambda: [
        "normal", "small-black-dot", "non-overlapping-double-latitude",
        "folded-edge-not-included", "feathering", "small-tail",
        "meridian-anomaly", "latitude-pollution", "meridian-pollution",
    ])
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    encoder_name: str = "dinov2_vits14"
    selected_layers: List[int] = field(default_factory=lambda: [9, 10, 11])
    bottleneck_dim: int = 96
    fused_dim: int = 256
    noise_std: float = 0.1
    drop_rate: float = 0.1


@dataclass
class TrainConfig:
    stage1_epochs: int = 50
    stage1_lr: float = 1e-4
    stage1_batch_size: int = 16

    stage2_epochs: int = 50
    stage2_lr: float = 5e-4
    stage2_recon_lr: float = 1e-5
    stage2_batch_size: int = 16

    weight_decay: float = 0.05
    max_grad_norm: float = 1.0
    warmup_epochs: int = 5
    label_smoothing: float = 0.1

    lambda_cls: float = 1.0
    lambda_mask: float = 1.0
    lambda_recon: float = 0.5
    lambda_anomaly: float = 0.3

    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    save_interval: int = 5
    eval_interval: int = 5


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    device: str = "cuda"
    seed: int = 42

    def save(self, path: str):
        import dataclasses

        def to_dict(obj):
            if dataclasses.is_dataclass(obj):
                return {k: to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            return obj

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(to_dict(self), f, default_flow_style=False, allow_unicode=True)

    def __repr__(self):
        return (
            f"Config(\n"
            f"  encoder={self.model.encoder_name},\n"
            f"  layers={self.model.selected_layers},\n"
            f"  bottleneck={self.model.bottleneck_dim},\n"
            f"  fused={self.model.fused_dim},\n"
            f"  classes={self.data.num_classes},\n"
            f"  img_size={self.data.img_size},\n"
            f"  device={self.device}\n"
            f")"
        )


def _parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="DinoDefectClassifier Training")

    # 配置文件路径
    parser.add_argument('--config', type=str, default='config_dino_legacy.yaml',
                        help='Path to config YAML file')

    # 可覆盖的常用参数
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override both stage1 and stage2 batch size')
    parser.add_argument('--stage1_batch_size', type=int, default=None)
    parser.add_argument('--stage2_batch_size', type=int, default=None)
    parser.add_argument('--stage1_epochs', type=int, default=None)
    parser.add_argument('--stage2_epochs', type=int, default=None)
    parser.add_argument('--stage1_lr', type=float, default=None)
    parser.add_argument('--stage2_lr', type=float, default=None)
    parser.add_argument('--img_size', type=int, default=None)
    parser.add_argument('--encoder', type=str, default=None,
                        help='Encoder model name, e.g. dinov2_vits14')
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--noise_std', type=float, default=None)
    parser.add_argument('--drop_rate', type=float, default=None)
    parser.add_argument('--label_smoothing', type=float, default=None)

    return parser.parse_args()


def _load_yaml(path: str) -> dict:
    """安全加载 YAML，处理编码问题"""
    if not os.path.exists(path):
        print(f"[Config] {path} not found, using defaults")
        return {}

    # 依次尝试 utf-8, utf-8-sig (带BOM), gbk
    for encoding in ['utf-8', 'utf-8-sig', 'gbk']:
        try:
            with open(path, 'r', encoding=encoding) as f:
                raw = yaml.safe_load(f)
            print(f"[Config] Loaded {path} with encoding={encoding}")
            return raw if raw is not None else {}
        except (UnicodeDecodeError, UnicodeError):
            continue
        except yaml.YAMLError as e:
            print(f"[Config] YAML parse error in {path}: {e}")
            return {}

    # 最后用 errors='ignore' 兜底
    print(f"[Config] Warning: Could not detect encoding for {path}, using utf-8 with ignore")
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = yaml.safe_load(f)
    return raw if raw is not None else {}


def get_config() -> Config:
    """从 YAML + 命令行参数构建配置"""
    args = _parse_args()
    config = Config()

    # === Step 1: 从 YAML 加载 ===
    raw = _load_yaml(args.config)
    if raw.get('pipeline') == 'fabric_mobilenet':
        raise ValueError('这是MobileNet配置，请运行 fabric_pipeline.py train；旧DINO入口使用config_dino_legacy.yaml')

    if 'data' in raw:
        for k, v in raw['data'].items():
            if hasattr(config.data, k):
                setattr(config.data, k, v)

    if 'model' in raw:
        for k, v in raw['model'].items():
            if hasattr(config.model, k):
                setattr(config.model, k, v)

    if 'train' in raw:
        for k, v in raw['train'].items():
            if hasattr(config.train, k):
                setattr(config.train, k, v)

    if 'device' in raw:
        config.device = raw['device']
    if 'seed' in raw:
        config.seed = raw['seed']

    # === Step 2: 命令行参数覆盖 ===
    if args.batch_size is not None:
        config.train.stage1_batch_size = args.batch_size
        config.train.stage2_batch_size = args.batch_size
    if args.stage1_batch_size is not None:
        config.train.stage1_batch_size = args.stage1_batch_size
    if args.stage2_batch_size is not None:
        config.train.stage2_batch_size = args.stage2_batch_size
    if args.stage1_epochs is not None:
        config.train.stage1_epochs = args.stage1_epochs
    if args.stage2_epochs is not None:
        config.train.stage2_epochs = args.stage2_epochs
    if args.stage1_lr is not None:
        config.train.stage1_lr = args.stage1_lr
    if args.stage2_lr is not None:
        config.train.stage2_lr = args.stage2_lr
    if args.img_size is not None:
        config.data.img_size = args.img_size
    if args.encoder is not None:
        config.model.encoder_name = args.encoder
    if args.data_root is not None:
        config.data.data_root = args.data_root
    if args.device is not None:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    if args.save_dir is not None:
        config.train.save_dir = args.save_dir
    if args.log_dir is not None:
        config.train.log_dir = args.log_dir
    if args.noise_std is not None:
        config.model.noise_std = args.noise_std
    if args.drop_rate is not None:
        config.model.drop_rate = args.drop_rate
    if args.label_smoothing is not None:
        config.train.label_smoothing = args.label_smoothing

    return config