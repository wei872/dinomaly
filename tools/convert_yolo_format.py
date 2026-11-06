# tools/convert_yolo_format.py
"""
将 YOLO 目标检测/分割格式数据转换为本项目需要的格式

支持的 YOLO 格式:
1. YOLO Detection: class_id cx cy w h
2. YOLO Segmentation: class_id x1 y1 x2 y2 ...
3. YOLO 目录结构 (按类别文件夹组织)
"""
import os
import cv2
import csv
import numpy as np
import argparse
from tqdm import tqdm


def yolo_bbox_to_mask(label_path, img_h, img_w, expand_ratio=0.1):
    """
    将 YOLO 检测框转换为掩膜

    Args:
        label_path: YOLO label 文件路径
        img_h, img_w: 图像尺寸
        expand_ratio: 边框扩展比例
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    class_id = 0

    if not os.path.exists(label_path):
        return mask, class_id

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls = int(parts[0])
            class_id = cls + 1  # 0=normal

            if len(parts) == 5:
                # Detection format: class cx cy w h
                cx, cy, w, h = map(float, parts[1:5])

                # 扩展
                w *= (1 + expand_ratio)
                h *= (1 + expand_ratio)

                x1 = max(0, int((cx - w / 2) * img_w))
                y1 = max(0, int((cy - h / 2) * img_h))
                x2 = min(img_w, int((cx + w / 2) * img_w))
                y2 = min(img_h, int((cy + h / 2) * img_h))

                mask[y1:y2, x1:x2] = 255
            else:
                # Segmentation format: class x1 y1 x2 y2 ...
                coords = list(map(float, parts[1:]))
                points = []
                for j in range(0, len(coords), 2):
                    if j + 1 < len(coords):
                        px = int(coords[j] * img_w)
                        py = int(coords[j + 1] * img_h)
                        points.append([px, py])

                if len(points) >= 3:
                    pts = np.array(points, dtype=np.int32)
                    cv2.fillPoly(mask, [pts], 255)

    return mask, class_id


def convert(
        input_dir: str,
        output_dir: str,
        class_map: dict = None,
        splits: list = None,
):
    """
    主转换函数

    Args:
        input_dir: YOLO 数据集根目录
        output_dir: 输出目录
        class_map: {yolo_class_id: (our_class_id, class_name)}
        splits: ['train', 'val', 'test']
    """
    if splits is None:
        splits = ['train', 'val']

    if class_map is None:
        class_map = {
            0: (1, 'defect_0'),
            1: (2, 'defect_1'),
            2: (3, 'defect_2'),
            3: (4, 'defect_3'),
        }

    for split in splits:
        img_dir = os.path.join(input_dir, 'images', split)
        label_dir = os.path.join(input_dir, 'labels', split)

        if not os.path.exists(img_dir):
            print(f"Skip {split}: {img_dir} not found")
            continue

        out_img_dir = os.path.join(output_dir, split, 'images')
        out_mask_dir = os.path.join(output_dir, split, 'masks')
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_mask_dir, exist_ok=True)

        rows = []

        img_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])

        for img_file in tqdm(img_files, desc=f"Converting {split}"):
            img_path = os.path.join(img_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]

            # 标签文件
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(label_dir, label_file)

            # 转换掩膜
            mask, yolo_cls = yolo_bbox_to_mask(label_path, h, w)

            # 映射类别
            if yolo_cls == 0 or yolo_cls - 1 not in class_map:
                our_cls_id = 0
                cls_name = 'normal'
            else:
                our_cls_id, cls_name = class_map[yolo_cls - 1]

            # 保存
            cv2.imwrite(os.path.join(out_img_dir, img_file), img)
            mask_name = os.path.splitext(img_file)[0] + '.png'
            cv2.imwrite(os.path.join(out_mask_dir, mask_name), mask)

            rows.append([img_file, our_cls_id, cls_name])

        # CSV
        csv_path = os.path.join(output_dir, split, 'labels.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_name', 'class_id', 'class_name'])
            writer.writerows(rows)

        print(f"  {split}: {len(rows)} images converted")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert YOLO format to project format')
    parser.add_argument('--input_dir', type=str, required=True, help='YOLO dataset root')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--classes', nargs='+', default=['small-black-dot', 'non-overlapping-double-latitude', 'folded-edge-not-included', 'feathering', 'small-tail', 'meridian-anomaly', 'latitude-pollution', 'meridian-pollution']
)
    args = parser.parse_args()

    class_map = {}
    for i, name in enumerate(args.classes):
        class_map[i] = (i + 1, name)

    convert(args.input_dir, args.output_dir, class_map)
    print("Done!")