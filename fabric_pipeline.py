"""Lightweight normal-reference anomaly detection followed by YOLO-region classification.

Separate from the original DINO training pipeline. See FABRIC_GUIDE.md.
"""
import argparse
import json
import time
from pathlib import Path, PureWindowsPath

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def load_spec(filename, root_override=None):
    filename = Path(filename).resolve()
    spec = yaml.safe_load(filename.read_text(encoding='utf-8'))
    names = spec['names']
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    if len(names) != spec['nc']:
        raise ValueError('nc 与 names 长度不一致')
    root = Path(root_override or spec.get('path', '.')).expanduser()
    if not root.is_absolute() and not PureWindowsPath(str(root)).is_absolute():
        root = filename.parent / root
    for key in ('train', 'val', 'normaltrain', 'normalval'):
        raw = str(spec[key])
        if root_override:
            # YAML contains absolute Windows paths: retain the portion under path.
            old = str(spec.get('path', '')).replace('\\', '/').rstrip('/')
            normalized = raw.replace('\\', '/')
            if old and normalized.startswith(old + '/'):
                raw = normalized[len(old) + 1:]
            elif Path(raw).is_absolute() or PureWindowsPath(raw).is_absolute():
                raise ValueError(f'{key} 不在 YAML path 下，不能自动重定位')
        p = Path(raw)
        if not p.is_absolute() and not PureWindowsPath(raw).is_absolute():
            p = root / p
        if not p.is_dir():
            raise FileNotFoundError(f'{key}: {p}；请设置 --data-root 或修改 YAML')
        spec[key] = p
    spec['names'] = names
    return spec


def images(directory):
    result = sorted(p for p in Path(directory).rglob('*') if p.suffix.lower() in EXTENSIONS)
    if not result:
        raise ValueError(f'没有图片: {directory}')
    return result


def label_path(image):
    parts = list(image.parts)
    if 'images' not in parts:
        raise ValueError(f'YOLO 图片路径必须含 images 目录: {image}')
    parts[len(parts) - 1 - parts[::-1].index('images')] = 'labels'
    return Path(*parts).with_suffix('.txt')


def read_boxes(path, num_classes):
    if not path.exists():
        raise FileNotFoundError(f'缺少标签（正常图请提供空 txt）: {path}')
    boxes = []
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        values = [float(v) for v in line.split()]
        if len(values) != 5:
            raise ValueError(f'{path}:{line_number}: 仅支持 cls cx cy w h 检测框')
        cls, cx, cy, w, h = values
        if not all(np.isfinite(values)) or cls != int(cls) or not 0 <= cls < num_classes:
            raise ValueError(f'{path}:{line_number}: 非法类别/数值')
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f'{path}:{line_number}: 非法归一化框')
        boxes.append((int(cls), max(0, cx-w/2), max(0, cy-h/2), min(1, cx+w/2), min(1, cy+h/2)))
    return boxes


class FabricModel(nn.Module):
    def __init__(self, classes, pretrained=False):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        # stride 16, 48 channels; preserves more spatial detail than final stride 32.
        self.features = mobilenet_v3_small(weights=weights).features[:9]
        self.classifier = nn.Linear(48, classes)
        self.register_buffer('mean', torch.zeros(48))
        self.register_buffer('variance', torch.ones(48))
        self.register_buffer('reference', torch.zeros(1))
        for p in self.features.parameters():
            p.requires_grad_(False)

    def encode(self, x):
        return self.features(x)

    def distance(self, features):
        return ((features - self.mean[None, :, None, None]).square() /
                self.variance[None, :, None, None]).mean(1).sqrt()


def read_image(path, size, device):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'无法读取图片: {path}')
    rgb = cv2.cvtColor(cv2.resize(image, (size, size)), cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div_(255).to(device)
    mean = x.new_tensor([.485, .456, .406])[:, None, None]
    std = x.new_tensor([.229, .224, .225])[:, None, None]
    return image, ((x - mean) / std).unsqueeze(0)


def pool_region(feature, box):
    h, w = feature.shape[-2:]
    x1, y1, x2, y2 = box
    left, top = min(w-1, int(x1*w)), min(h-1, int(y1*h))
    right, bottom = min(w, max(left+1, int(np.ceil(x2*w)))), min(h, max(top+1, int(np.ceil(y2*h))))
    return feature[0, :, top:bottom, left:right].mean((1, 2))


def percentile_scores(values, reference):
    """Empirical normal-validation patch percentile, NOT P(defect|image)."""
    return np.searchsorted(reference, values, side='right') / len(reference)



def check_dataset(spec):
    """Fail before weight downloads/training on bad labels or split leakage."""
    import hashlib
    seen = {}
    report = {}
    for split in ('normaltrain', 'normalval', 'train', 'val'):
        paths = images(spec[split])
        counts = np.zeros(len(spec['names']), dtype=int)
        for path in paths:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen and seen[digest] != split:
                raise ValueError(f'跨集合重复图片: {path} ({seen[digest]} / {split})')
            seen[digest] = split
            if split in ('train', 'val'):
                for cls, *_ in read_boxes(label_path(path), len(spec['names'])):
                    counts[cls] += 1
        report[split] = {'images': len(paths), 'boxes_per_class': counts.tolist()}
        if split == 'train' and np.any(counts == 0):
            raise ValueError(f'训练框缺少类别: {np.where(counts == 0)[0].tolist()}')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


@torch.no_grad()
def fit_normal(model, spec, size, device):
    """Fit normal train moments, then calibrate on normal validation only."""
    total = torch.zeros(48, device=device, dtype=torch.float64)
    square = total.clone()
    count = 0
    with torch.no_grad():
        for path in images(spec['normaltrain']):
            _, x = read_image(path, size, device)
            f = model.encode(x).permute(0, 2, 3, 1).reshape(-1, 48).double()
            total += f.sum(0)
            square += f.square().sum(0)
            count += len(f)
        model.mean.copy_((total / count).float())
        model.variance.copy_((square/count - (total/count).square()).clamp_min(1e-4).float())
        reference = []
        for path in images(spec['normalval']):
            _, x = read_image(path, size, device)
            reference.append(model.distance(model.encode(x)).flatten().cpu().numpy())
        reference = np.sort(np.concatenate(reference))
        # Bound checkpoint size while retaining quantiles across the full reference.
        reference = np.quantile(reference, np.linspace(0, 1, min(100000, len(reference)))).astype('float32')
        model.reference = torch.from_numpy(reference).to(device)
    print(f'正常特征拟合完成: {count} patches；标定参考: {len(reference)} patches')


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    spec = load_spec(args.data, args.data_root)
    check_dataset(spec)
    model = FabricModel(len(spec['names']), pretrained=True).to(args.device).eval()
    print('阶段1：拟合正常训练分布，并用正常验证集标定异常阈值')
    fit_normal(model, spec, args.size, args.device)
    print('阶段2：提取YOLO框特征并训练缺陷分类器')
    with torch.no_grad():
        vectors, targets = [], []
        for path in images(spec['train']):
            boxes = read_boxes(label_path(path), len(spec['names']))
            if not boxes:
                continue
            _, x = read_image(path, args.size, args.device)
            feature = model.encode(x)
            for cls, *box in boxes:
                vectors.append(pool_region(feature, box).cpu())
                targets.append(cls)
    if not targets:
        raise ValueError('训练集没有有效缺陷框')
    counts = np.bincount(targets, minlength=len(spec['names']))
    if np.any(counts == 0):
        raise ValueError(f'以下类别没有训练框: {[spec["names"][i] for i in np.where(counts == 0)[0]]}')
    vectors, targets = torch.stack(vectors), torch.tensor(targets)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(vectors, targets),
                                         batch_size=256, shuffle=True)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr)
    model.classifier.train()
    for epoch in range(args.epochs):
        loss_sum = 0.
        for x, y in loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(model.classifier(x.to(args.device)), y.to(args.device))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        print(f'epoch {epoch+1}/{args.epochs} loss={loss_sum/len(loader):.4f}')
    model.eval()
    correct = total_boxes = 0
    with torch.no_grad():
        for path in images(spec['val']):
            boxes = read_boxes(label_path(path), len(spec['names']))
            if not boxes:
                continue
            _, x = read_image(path, args.size, args.device)
            feature = model.encode(x)
            for cls, *box in boxes:
                correct += int(model.classifier(pool_region(feature, box)).argmax().item() == cls)
                total_boxes += 1
    print(f'验证集 GT 框分类准确率（非端到端检测指标）: {correct}/{total_boxes}')
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.cpu().state_dict(), 'names': spec['names'],
                'size': args.size, 'quantile': args.quantile, 'min_area': args.min_area,
                'class_threshold': args.class_threshold, 'margin': args.margin}, args.checkpoint)


class FabricInference:
    def __init__(self, checkpoint, device='cuda'):
        self.device = device
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.config = state
        self.names = state['names']
        self.model = FabricModel(len(self.names))
        self.model.reference = torch.empty_like(state['state_dict']['reference'])
        self.model.load_state_dict(state['state_dict'])
        self.model.to(device).eval()
        self.reference = self.model.reference.cpu().numpy()
        self.threshold = float(np.quantile(self.reference, state['quantile']))

    @torch.inference_mode()
    def predict(self, path):
        image, x = read_image(path, self.config['size'], self.device)
        feature = self.model.encode(x)
        distance = self.model.distance(feature)[0].cpu().numpy()
        mask = (distance > self.threshold).astype('uint8')
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        height, width = distance.shape
        regions = []
        candidates = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= self.config['min_area']]
        # All candidate regions classified together, with a single backbone forward.
        if candidates:
            boxes = []
            for i in candidates:
                left, top, w, h, _ = stats[i]
                boxes.append((left/width, top/height, (left+w)/width, (top+h)/height))
            logits = self.model.classifier(torch.stack([pool_region(feature, b) for b in boxes]))
            probs = logits.softmax(-1).cpu().numpy()
            for i, box, prob in zip(candidates, boxes, probs):
                order = np.argsort(prob)[::-1]
                cls = int(order[0])
                confidence = float(prob[cls])
                gap = confidence - float(prob[order[1]]) if len(order) > 1 else confidence
                unknown = confidence < self.config['class_threshold'] or gap < self.config['margin']
                raw = float(distance[labels == i].mean())
                ih, iw = image.shape[:2]
                regions.append({'bbox_xyxy': [round(box[0]*iw), round(box[1]*ih), round(box[2]*iw), round(box[3]*ih)],
                                'class_id': len(self.names) if unknown else cls,
                                'class_name': '未知类型' if unknown else self.names[cls],
                                'class_confidence': confidence, 'class_margin': gap,
                                'anomaly_distance': raw,
                                'normal_deviation_confidence': float(percentile_scores(raw, self.reference))})
        return {'image': str(path), 'is_defect': bool(regions), 'regions': regions,
                'anomaly_distance': float(distance.max()),
                'normal_deviation_confidence': float(percentile_scores(distance.max(), self.reference)),
                'score_definition': '正常验证集 patch 距离的经验百分位，非缺陷后验概率'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['check', 'train', 'predict', 'benchmark'])
    parser.add_argument('--data', default='dataset/fabric19.yaml')
    parser.add_argument('--data-root')
    parser.add_argument('--checkpoint', default='checkpoints/fabric_light.pt')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--size', type=int, default=320)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quantile', type=float, default=.995)
    parser.add_argument('--min-area', type=int, default=2, help='最小连通区域，单位为特征格')
    parser.add_argument('--class-threshold', type=float, default=.65)
    parser.add_argument('--margin', type=float, default=.15)
    parser.add_argument('--source', help='图片或图片目录')
    parser.add_argument('--output', default='results/fabric_predictions.json')
    parser.add_argument('--iterations', type=int, default=100)
    args = parser.parse_args()
    if args.size < 32 or args.size % 16 or not 0 < args.quantile < 1:
        parser.error('size 必须为 >=32 的16倍数，quantile 必须在 (0,1)')
    if args.epochs < 1 or args.iterations < 1 or args.min_area < 1:
        parser.error('epochs、iterations、min-area 必须大于0')
    if not 0 <= args.class_threshold <= 1 or not 0 <= args.margin <= 1:
        parser.error('分类阈值与 margin 必须在 [0,1]')
    if args.mode == 'check':
        check_dataset(load_spec(args.data, args.data_root))
        return
    if args.mode == 'train':
        train(args)
        return
    if not args.source:
        parser.error('predict/benchmark 需要 --source')
    engine = FabricInference(args.checkpoint, args.device)
    source = Path(args.source)
    paths = images(source) if source.is_dir() else [source]
    if args.mode == 'predict':
        results = [engine.predict(p) for p in paths]
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'结果保存至 {args.output}')
    else:
        for _ in range(10):
            engine.predict(paths[0])
        timings = []
        for i in range(args.iterations):
            if str(args.device).startswith('cuda'):
                torch.cuda.synchronize()
            start = time.perf_counter()
            engine.predict(paths[i % len(paths)])
            if str(args.device).startswith('cuda'):
                torch.cuda.synchronize()
            timings.append((time.perf_counter()-start)*1000)
        print(json.dumps({'device': args.device, 'size': engine.config['size'],
                          'mean_ms': float(np.mean(timings)), 'p95_ms': float(np.percentile(timings, 95)),
                          'fps': float(1000/np.mean(timings)),
                          'scope': 'batch=1 含读图/预处理/传输/异常检测/区域分类，不含JSON写盘'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
