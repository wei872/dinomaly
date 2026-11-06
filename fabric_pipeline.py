"""Lightweight normal-reference anomaly detection followed by YOLO-region classification.

Separate from the original DINO training pipeline. See FABRIC_GUIDE.md.
"""
import argparse
import json
import os
import time
from pathlib import Path, PureWindowsPath

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw, ImageFont
from dataset_audit import audit_duplicates, filter_cross_split_duplicates
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
        self.register_buffer('known_mean', torch.zeros(classes, 48))
        self.register_buffer('known_variance', torch.ones(classes, 48))
        self.register_buffer('known_radius', torch.full((classes,), float('inf')))
        for p in self.features.parameters():
            p.requires_grad_(False)

    def encode(self, x):
        return self.features(x)

    def distance(self, features):
        return ((features - self.mean[None, :, None, None]).square() /
                self.variance[None, :, None, None]).mean(1).sqrt()


def decode_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'无法读取图片: {path}')
    return image


def preprocess_image(image, size):
    rgb = cv2.cvtColor(cv2.resize(image, (size, size)), cv2.COLOR_BGR2RGB)
    # Normalize on CPU, transfer one contiguous tensor instead of many CUDA ops.
    array = rgb.astype(np.float32) / 255.
    array = (array - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def read_image(path, size, device):
    image = decode_image(path)
    return image, preprocess_image(image, size).unsqueeze(0).to(device)


class FeatureImages(torch.utils.data.Dataset):
    def __init__(self, paths, size):
        self.paths, self.size = list(paths), size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return preprocess_image(decode_image(self.paths[index]), self.size), index


@torch.no_grad()
def feature_batches(model, paths, size, device, batch_size=16, workers=0):
    paths = list(paths)
    loader = torch.utils.data.DataLoader(FeatureImages(paths, size), batch_size=batch_size,
                                        num_workers=workers, pin_memory=str(device).startswith('cuda'))
    for x, indices in loader:
        features = model.encode(x.to(device, non_blocking=True))
        yield [paths[i] for i in indices.tolist()], features


def fuse_backbone(module):
    """In-place eval Conv+BN fusion; deployment only, no checkpoint mutation."""
    for child in module.children():
        fuse_backbone(child)
    if isinstance(module, nn.Sequential):
        for index in range(len(module)-1):
            if isinstance(module[index], nn.Conv2d) and isinstance(module[index+1], nn.BatchNorm2d):
                module[index] = torch.nn.utils.fuse_conv_bn_eval(module[index], module[index+1])
                module[index+1] = nn.Identity()


def pool_region(feature, box):
    h, w = feature.shape[-2:]
    x1, y1, x2, y2 = box
    left, top = min(w-1, int(x1*w)), min(h-1, int(y1*h))
    right, bottom = min(w, max(left+1, int(np.ceil(x2*w)))), min(h, max(top+1, int(np.ceil(y2*h))))
    return feature[0, :, top:bottom, left:right].mean((1, 2))


def percentile_scores(values, reference):
    """Empirical normal-validation patch percentile, NOT P(defect|image)."""
    return np.searchsorted(reference, values, side='right') / len(reference)



@torch.no_grad()
def fit_known_distribution(model, vectors, targets, quantile=.99):
    """Class-conditional diagonal distances. Training radii need held-out tuning."""
    floor = vectors.var(0, unbiased=False).clamp_min(1e-4) * .05
    for cls in range(model.classifier.out_features):
        samples = vectors[targets == cls]
        if len(samples) < 2:
            raise ValueError(f'类别 {cls} 至少需要2个框来拟合已知类分布')
        mean = samples.mean(0)
        variance = samples.var(0, unbiased=False).maximum(floor).clamp_min(1e-4)
        distances = ((samples - mean).square() / variance).mean(-1).sqrt()
        model.known_mean[cls].copy_(mean)
        model.known_variance[cls].copy_(variance)
        model.known_radius[cls] = torch.quantile(distances, quantile).clamp_min(1.)


def rejection_reasons(confidence, gap, known_distance, known_radius, threshold, margin):
    reasons = []
    if known_distance > known_radius:
        reasons.append('outside_known_distribution')
    if confidence < threshold:
        reasons.append('low_class_confidence')
    if gap < margin:
        reasons.append('ambiguous_classes')
    return reasons


def split_images(spec, split):
    # Never rescan raw folders after checking: skipped files must stay excluded.
    if '_filtered_paths' in spec:
        return spec['_filtered_paths'][split]
    return images(spec[split])


def check_dataset(spec, duplicate_report='results/dataset_duplicates.json', strict_duplicates=False):
    """Filter exact duplicate validation files without modifying the dataset."""
    spec.pop('_filtered_paths', None)
    split_paths = {split: images(spec[split]) for split in ('normaltrain', 'normalval', 'train', 'val')}
    duplicates = audit_duplicates(split_paths, duplicate_report)
    if duplicates['duplicates'] and strict_duplicates:
        raise ValueError(f"发现 {duplicates['duplicate_group_count']} 组跨集合重复，完整报告: {duplicate_report}")
    split_paths, skipped = filter_cross_split_duplicates(split_paths, duplicates)
    duplicates['skipped'] = skipped
    duplicates['skipped_count'] = len(skipped)
    duplicates['policy'] = '保留训练副本，跳过重叠验证副本；仅验证集合之间重复时优先normalval；不修改文件'
    Path(duplicate_report).write_text(json.dumps(duplicates, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"发现 {duplicates['duplicate_group_count']} 组跨集合重复，跳过 {len(skipped)} 张验证图片；报告: {duplicate_report}")
    for split, paths in split_paths.items():
        if not paths:
            raise ValueError(f'{split} 过滤重复后没有可用图片，请补充独立样本；不能用训练图片代替验证集')
    spec['_filtered_paths'] = split_paths
    report = {}
    for split in ('normaltrain', 'normalval', 'train', 'val'):
        paths = split_paths[split]
        counts = np.zeros(len(spec['names']), dtype=int)
        for path in paths:
            if split in ('train', 'val'):
                for cls, *_ in read_boxes(label_path(path), len(spec['names'])):
                    counts[cls] += 1
        report[split] = {'images': len(paths), 'skipped_duplicates': sum(item['split'] == split for item in skipped), 'boxes_per_class': counts.tolist()}
        if split == 'train' and np.any(counts == 0):
            raise ValueError(f'训练框缺少类别: {np.where(counts == 0)[0].tolist()}')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


@torch.no_grad()
def fit_normal(model, spec, size, device, batch_size=16, workers=0):
    """Fit normal train moments, then calibrate on normal validation only."""
    total = torch.zeros(48, device=device, dtype=torch.float64)
    square = total.clone()
    count = 0
    with torch.no_grad():
        for _, features in feature_batches(model, split_images(spec, 'normaltrain'), size, device, batch_size, workers):
            f = features.permute(0, 2, 3, 1).reshape(-1, 48).float()
            total += f.sum(0).double()
            square += f.square().sum(0).double()
            count += len(f)
        model.mean.copy_((total / count).float())
        model.variance.copy_((square/count - (total/count).square()).clamp_min(1e-4).float())
        reference = []
        for _, features in feature_batches(model, split_images(spec, 'normalval'), size, device, batch_size, workers):
            reference.append(model.distance(features).flatten().cpu().numpy())
        reference = np.sort(np.concatenate(reference))
        # Bound checkpoint size while retaining quantiles across the full reference.
        reference = np.quantile(reference, np.linspace(0, 1, min(100000, len(reference)))).astype('float32')
        model.reference = torch.from_numpy(reference).to(device)
    print(f'正常特征拟合完成: {count} patches；标定参考: {len(reference)} patches')


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    spec = load_spec(args.data, args.data_root)
    check_dataset(spec, args.duplicate_report, args.strict_duplicates)
    model = FabricModel(len(spec['names']), pretrained=True).to(args.device).eval()
    print('阶段1：拟合正常训练分布，并用正常验证集标定异常阈值')
    fit_normal(model, spec, args.size, args.device, args.feature_batch_size, args.workers)
    print('阶段2：提取YOLO框特征并训练缺陷分类器')
    with torch.no_grad():
        vectors, targets = [], []
        for paths, features in feature_batches(model, split_images(spec, 'train'), args.size, args.device,
                                                args.feature_batch_size, args.workers):
            batch_vectors = []
            for index, path in enumerate(paths):
                for cls, *box in read_boxes(label_path(path), len(spec['names'])):
                    batch_vectors.append(pool_region(features[index:index+1], box))
                    targets.append(cls)
            if batch_vectors:
                vectors.extend(torch.stack(batch_vectors).cpu().unbind())
    if not targets:
        raise ValueError('训练集没有有效缺陷框')
    counts = np.bincount(targets, minlength=len(spec['names']))
    if np.any(counts == 0):
        raise ValueError(f'以下类别没有训练框: {[spec["names"][i] for i in np.where(counts == 0)[0]]}')
    vectors, targets = torch.stack(vectors), torch.tensor(targets)
    fit_known_distribution(model, vectors.to(args.device), targets.to(args.device), args.known_quantile)
    if not args.cpu_feature_cache:
        vectors, targets = vectors.to(args.device), targets.to(args.device)
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
        for paths, features in feature_batches(model, split_images(spec, 'val'), args.size, args.device,
                                                args.feature_batch_size, args.workers):
            pooled, labels = [], []
            for index, path in enumerate(paths):
                for cls, *box in read_boxes(label_path(path), len(spec['names'])):
                    pooled.append(pool_region(features[index:index+1], box))
                    labels.append(cls)
            if pooled:
                predicted = model.classifier(torch.stack(pooled)).argmax(-1)
                correct += int((predicted == torch.tensor(labels, device=args.device)).sum().item())
                total_boxes += len(labels)
    print(f'验证集 GT 框分类准确率（非端到端检测指标）: {correct}/{total_boxes}')
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.cpu().state_dict(), 'names': spec['names'],
                'size': args.size, 'quantile': args.quantile, 'min_area': args.min_area,
                'class_threshold': args.class_threshold, 'margin': args.margin}, args.checkpoint)


class FabricInference:
    def __init__(self, checkpoint, device='cuda', fuse=True):
        self.device = device
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.config = state
        self.names = state['names']
        self.model = FabricModel(len(self.names))
        self.model.reference = torch.empty_like(state['state_dict']['reference'])
        # Old checkpoints have no known-class statistics: retain uncertainty rejection.
        missing = {'known_mean', 'known_variance', 'known_radius'} - state['state_dict'].keys()
        merged = self.model.state_dict()
        for key in missing:
            state['state_dict'][key] = merged[key]
        self.model.load_state_dict(state['state_dict'])
        self.known_distribution_available = not missing
        self.model.to(device).eval()
        if fuse:
            fuse_backbone(self.model.features)
        self.reference = self.model.reference.cpu().numpy()
        self.threshold = float(np.quantile(self.reference, state['quantile']))

    @torch.inference_mode()
    def predict(self, path):
        return self.predict_frame(decode_image(path), str(path))

    @torch.inference_mode()
    def predict_frame(self, image, image_id='frame'):
        """BGR uint8 frame API: no disk IO, no visualization."""
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('image 必须为 H×W×3 BGR uint8 数组')
        x = preprocess_image(image, self.config['size']).unsqueeze(0).to(self.device)
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
            region_features = torch.stack([pool_region(feature, b) for b in boxes])
            logits = self.model.classifier(region_features)
            known_distances = ((region_features[:, None, :] - self.model.known_mean[None]).square() /
                               self.model.known_variance[None]).mean(-1).sqrt().cpu().numpy()
            known_radii = self.model.known_radius.cpu().numpy()
            probs = logits.softmax(-1).cpu().numpy()
            for index, (i, box, prob) in enumerate(zip(candidates, boxes, probs)):
                order = np.argsort(prob)[::-1]
                cls = int(order[0])
                confidence = float(prob[cls])
                gap = confidence - float(prob[order[1]]) if len(order) > 1 else confidence
                known_distance = float(known_distances[index, cls])
                reasons = rejection_reasons(confidence, gap, known_distance, float(known_radii[cls]),
                                            self.config['class_threshold'], self.config['margin'])
                unknown = bool(reasons)
                raw = float(distance[labels == i].mean())
                ih, iw = image.shape[:2]
                regions.append({'bbox_xyxy': [round(box[0]*iw), round(box[1]*ih), round(box[2]*iw), round(box[3]*ih)],
                                'class_id': len(self.names) if unknown else cls,
                                'class_name': '未知类型' if unknown else self.names[cls],
                                'class_confidence': confidence, 'class_margin': gap,
                                'unknown_reasons': reasons,
                                'candidate_known_class': self.names[cls],
                                'known_class_distance': known_distance if self.known_distribution_available else None,
                                'known_class_radius': float(known_radii[cls]) if self.known_distribution_available else None,
                                'anomaly_distance': raw,
                                'normal_deviation_confidence': float(percentile_scores(raw, self.reference))})
        return {'image': str(image_id), 'is_defect': bool(regions), 'regions': regions,
                'anomaly_distance': float(distance.max()),
                'normal_deviation_confidence': float(percentile_scores(distance.max(), self.reference)),
                'known_distribution_available': self.known_distribution_available,
                'score_definition': '正常验证集 patch 距离的经验百分位，非缺陷后验概率'}


def load_label_font(font_path=None, size=22):
    candidates = [font_path] if font_path else [
        os.environ.get('FABRIC_FONT'),
        'C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf',
        '/System/Library/Fonts/PingFang.ttc',
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return ImageFont.truetype(str(candidate), size=size)
    raise FileNotFoundError('未找到中文字体；请安装 Noto Sans CJK 或通过 --font 指定微软雅黑/思源黑体字体文件')


def save_annotated_image(result, output_path, font):
    """Keep original resolution; write a new image, never overwrite the input."""
    source = Path(result['image'])
    output_path = Path(output_path)
    if output_path.resolve() == source.resolve():
        raise ValueError('可视化输出不能覆盖原始图片')
    with Image.open(source) as original:
        canvas = original.convert('RGB')
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    line_width = max(2, round(min(width, height) / 400))
    for region in result['regions']:
        x1, y1, x2, y2 = region['bbox_xyxy']
        x1, x2 = sorted((max(0, min(width-1, x1)), max(0, min(width-1, x2))))
        y1, y2 = sorted((max(0, min(height-1, y1)), max(0, min(height-1, y2))))
        unknown = bool(region.get('unknown_reasons')) or region['class_name'] == '未知类型'
        color = (255, 140, 0) if unknown else (255, 40, 40)
        if unknown:
            text = f"未知类型 | 偏离正常: {region['normal_deviation_confidence']:.1%}"
        else:
            text = (f"{region['class_name']} | 分类: {region['class_confidence']:.1%}"
                    f" | 偏离正常: {region['normal_deviation_confidence']:.1%}")
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        # Shrink long labels to fit the original image, without resizing the image.
        label_font = font
        while draw.textbbox((0, 0), text, font=label_font)[2] + 8 > width and label_font.size > 8:
            label_font = label_font.font_variant(size=label_font.size-1)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=label_font)
        tw, th = right-left+8, bottom-top+8
        tx = max(0, min(x1, width-tw))
        ty = max(0, y1-th) if y1 >= th else max(0, min(y1, height-th))
        draw.rectangle((tx, ty, min(width-1, tx+tw), min(height-1, ty+th)), fill=color)
        draw.text((tx+4-left, ty+4-top), text, font=label_font, fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format='PNG')
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['check', 'train', 'predict', 'benchmark'])
    parser.add_argument('--data', default='dataset/fabric19.yaml')
    parser.add_argument('--data-root')
    parser.add_argument('--checkpoint', default='checkpoints/fabric_light.pt')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--size', type=int, default=448, help='训练输入尺寸，默认448；推理使用checkpoint保存的尺寸')
    parser.add_argument('--feature-batch-size', type=int, default=16, help='冻结骨干特征提取batch，显存不足时调小')
    parser.add_argument('--workers', type=int, default=0, help='Windows可尝试2或4个读图进程')
    parser.add_argument('--cpu-feature-cache', action='store_true', help='框特征留CPU，节约GPU显存')
    parser.add_argument('--benchmark-vis', action='store_true', help='测速计入中文画框和PNG写盘')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quantile', type=float, default=.995)
    parser.add_argument('--min-area', type=int, default=2, help='最小连通区域，单位为特征格')
    parser.add_argument('--class-threshold', type=float, default=.65)
    parser.add_argument('--known-quantile', type=float, default=.99, help='已知类训练距离半径分位数，须验证集调参')
    parser.add_argument('--margin', type=float, default=.15)
    parser.add_argument('--source', help='图片或图片目录')
    parser.add_argument('--strict-duplicates', action='store_true', help='发现跨集合重复即停止；默认跳过重叠验证图片')
    parser.add_argument('--duplicate-report', default='results/dataset_duplicates.json', help='跨集合重复图片完整路径报告')
    parser.add_argument('--output', default='results/fabric_predictions.json')
    parser.add_argument('--vis-dir', default='results/annotated', help='原始分辨率带框图片输出目录')
    parser.add_argument('--font', help='支持中文的 ttf/ttc/otf 字体路径')
    parser.add_argument('--no-vis', action='store_true', help='仅保存JSON，不生成带框图片')
    parser.add_argument('--iterations', type=int, default=100)
    args = parser.parse_args()
    if args.size < 32 or args.size % 16 or not 0 < args.quantile < 1:
        parser.error('size 必须为 >=32 的16倍数，quantile 必须在 (0,1)')
    if not 0 < args.known_quantile <= 1:
        parser.error('known-quantile 必须在 (0,1]')
    if args.feature_batch_size < 1 or args.workers < 0:
        parser.error('feature-batch-size必须大于0，workers不能小于0')
    if args.epochs < 1 or args.iterations < 1 or args.min_area < 1:
        parser.error('epochs、iterations、min-area 必须大于0')
    if not 0 <= args.class_threshold <= 1 or not 0 <= args.margin <= 1:
        parser.error('分类阈值与 margin 必须在 [0,1]')
    if args.mode == 'check':
        check_dataset(load_spec(args.data, args.data_root), args.duplicate_report, args.strict_duplicates)
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
        font = None if args.no_vis else load_label_font(args.font)
        results = []
        for path in paths:
            result = engine.predict(path)
            if font is not None:
                relative = path.relative_to(source) if source.is_dir() else Path(path.name)
                # Preserve subdirectories and extensions in the filename to avoid collisions.
                destination = Path(args.vis_dir) / relative.parent / (relative.name + '.png')
                result['annotated_image'] = save_annotated_image(result, destination, font)
            results.append(result)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'结果保存至 {args.output}')
    else:
        for _ in range(10):
            engine.predict(paths[0])
        font = load_label_font(args.font) if args.benchmark_vis else None
        timings = []
        for i in range(args.iterations):
            if str(args.device).startswith('cuda'):
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = engine.predict(paths[i % len(paths)])
            if font is not None:
                save_annotated_image(result, Path(args.vis_dir) / '_benchmark.png', font)
            if str(args.device).startswith('cuda'):
                torch.cuda.synchronize()
            timings.append((time.perf_counter()-start)*1000)
        print(json.dumps({'device': args.device, 'size': engine.config['size'],
                          'mean_ms': float(np.mean(timings)), 'p95_ms': float(np.percentile(timings, 95)),
                          'fps': float(1000/np.mean(timings)),
                          'includes_visualization': args.benchmark_vis,
                          'scope': 'batch=1 含读图/预处理/传输/检测/分类，可选画框PNG写盘，不含JSON写盘'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
