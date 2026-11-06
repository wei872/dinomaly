"""推理脚本 - 适配轻量 ViT-S/14 版本 (完整输出)"""
import os
import sys
import csv
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from typing import Dict

from models.model import DinoDefectClassifier
from data.transforms import get_val_transform


class DefectInference:
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = device

        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )

        config = checkpoint.get('config', {})

        self.num_classes = config.get('num_classes', 9)
        self.img_size = config.get('img_size', 518)

        self.class_names = config.get('class_names', [
            "normal", "small-black-dot", "non-overlapping-double-latitude",
            "folded-edge-not-included", "feathering", "small-tail",
            "meridian-anomaly", "latitude-pollution", "meridian-pollution"
        ])

        model_args = {
            'num_classes': self.num_classes,
            'encoder_name': config.get('encoder_name', 'dinov2_vits14'),
            'selected_layers': config.get('selected_layers', [9, 10, 11]),
            'bottleneck_dim': config.get('bottleneck_dim', 96),
            'fused_dim': config.get('fused_dim', 256),
            'noise_std': config.get('noise_std', 0.1),
            'drop_rate': config.get('drop_rate', 0.1),
            'img_size': self.img_size,
        }

        print(f"Building model with: {model_args}")
        self.model = DinoDefectClassifier(**model_args).to(device)

        state_dict = checkpoint['model_state_dict']
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        self.transform = get_val_transform(self.img_size)

        print(f"Model loaded! Classes: {self.num_classes} | Size: {self.img_size}")

    # ================================================================
    #  核心推理
    # ================================================================
    @torch.no_grad()
    def predict(self, image_path: str) -> Dict:
        img = Image.open(image_path).convert('RGB')
        orig_w, orig_h = img.size
        orig_img = cv2.imread(image_path)

        x = self.transform(img).unsqueeze(0).to(self.device)
        outputs = self.model(x)

        # --- 分类 ---
        probs = F.softmax(outputs['class_logits'], dim=1)[0]
        pred_class_id = probs.argmax().item()
        confidence = probs[pred_class_id].item()
        all_probs = {self.class_names[i]: p.item() for i, p in enumerate(probs)}

        # --- 异常分数 ---
        anomaly_score = outputs['anomaly_prob'][0, 0].item() \
            if outputs['anomaly_prob'].numel() > 0 else 0.0

        # --- 异常热力图 (重建差异) ---
        anomaly_map = outputs['anomaly_map'][0].cpu().numpy()
        h, w = outputs.get('spatial_shape', (37, 37))
        anomaly_map_2d = anomaly_map.reshape(h, w)
        a_min, a_max = anomaly_map_2d.min(), anomaly_map_2d.max()
        anomaly_map_norm = (anomaly_map_2d - a_min) / (a_max - a_min + 1e-8)
        anomaly_map_full = cv2.resize(
            anomaly_map_norm, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
        )

        # --- Mask Head 掩膜 ---
        mask_logits = outputs['mask_pred'][0, 0].cpu().numpy()
        mask_prob = 1.0 / (1.0 + np.exp(-mask_logits))
        mask_prob_full = cv2.resize(
            mask_prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
        )
        mask_binary = (mask_prob_full > 0.5).astype(np.uint8) * 255

        # --- 融合掩膜 ---
        combined_prob = 0.5 * anomaly_map_full + 0.5 * mask_prob_full
        combined_binary = (combined_prob > 0.35).astype(np.uint8) * 255

        return {
            'class_name': self.class_names[pred_class_id],
            'class_id': pred_class_id,
            'confidence': confidence,
            'all_probs': all_probs,
            'is_defect': pred_class_id > 0,
            'anomaly_score': anomaly_score,
            'anomaly_map': anomaly_map_full,        # float32 [0,1]
            'mask_prob': mask_prob_full,             # float32 [0,1]
            'mask_binary': mask_binary,              # uint8   0/255
            'combined_prob': combined_prob,           # float32 [0,1]
            'combined_binary': combined_binary,       # uint8   0/255
            'orig_size': (orig_w, orig_h),
            'orig_img': orig_img,
        }

    # ================================================================
    #  保存全部结果
    # ================================================================
    def predict_and_save(
        self,
        image_path: str,
        output_dir: str,
        save_mask: bool = True,
        save_heatmap: bool = True,
        save_visualization: bool = True,
    ) -> Dict:
        result = self.predict(image_path)
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        orig_img = result['orig_img']

        # ========== 1. 二值掩膜 + 概率图 ==========
        if save_mask:
            mask_dir = os.path.join(output_dir, 'masks')
            os.makedirs(mask_dir, exist_ok=True)

            # MaskHead 二值掩膜
            cv2.imwrite(
                os.path.join(mask_dir, f'{img_name}_mask.png'),
                result['mask_binary']
            )
            # 融合二值掩膜（推荐）
            cv2.imwrite(
                os.path.join(mask_dir, f'{img_name}_combined_mask.png'),
                result['combined_binary']
            )

            # 概率灰度图
            prob_dir = os.path.join(output_dir, 'prob_maps')
            os.makedirs(prob_dir, exist_ok=True)
            prob_uint8 = (result['combined_prob'] * 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(prob_dir, f'{img_name}_prob.png'),
                prob_uint8
            )

        # ========== 2. 异常热力图 ==========
        if save_heatmap and orig_img is not None:
            heatmap_dir = os.path.join(output_dir, 'heatmaps')
            os.makedirs(heatmap_dir, exist_ok=True)

            heatmap_uint8 = (result['anomaly_map'] * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(orig_img, 0.6, heatmap_color, 0.4, 0)

            cv2.imwrite(
                os.path.join(heatmap_dir, f'{img_name}_heatmap.png'),
                overlay
            )

        # ========== 3. 综合可视化 ==========
        if save_visualization and orig_img is not None:
            vis_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)

            vis_img = self._create_visualization(orig_img, result, img_name)
            cv2.imwrite(
                os.path.join(vis_dir, f'{img_name}_vis.png'),
                vis_img
            )

        return result

    # ================================================================
    #  综合可视化 (2x2 拼接)
    # ================================================================
    def _create_visualization(
        self, img: np.ndarray, result: Dict, img_name: str
    ) -> np.ndarray:
        target_h, target_w = 300, 300
        img_resized = cv2.resize(img, (target_w, target_h))

        # ------ Panel 1: 原图 + 分类结果 ------
        panel1 = img_resized.copy()
        color = (0, 0, 255) if result['is_defect'] else (0, 255, 0)
        label_text = f"{result['class_name']}"
        conf_text = f"Conf: {result['confidence']:.3f}"
        anom_text = f"Anomaly: {result['anomaly_score']:.3f}"

        cv2.putText(panel1, label_text, (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(panel1, conf_text, (5, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(panel1, anom_text, (5, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 分类结果边框
        cv2.rectangle(panel1, (0, 0), (target_w - 1, target_h - 1), color, 3)

        # ------ Panel 2: 异常热力图叠加 ------
        heatmap = (result['anomaly_map'] * 255).astype(np.uint8)
        heatmap = cv2.resize(heatmap, (target_w, target_h))
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        panel2 = cv2.addWeighted(img_resized, 0.5, heatmap_color, 0.5, 0)
        cv2.putText(panel2, "Anomaly Heatmap", (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # ------ Panel 3: MaskHead 掩膜 ------
        mask_vis = cv2.resize(result['mask_binary'], (target_w, target_h))
        panel3 = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
        cv2.putText(panel3, "Mask (Segmentation)", (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ------ Panel 4: 融合掩膜叠加到原图 ------
        combined = cv2.resize(result['combined_binary'], (target_w, target_h))
        panel4 = img_resized.copy()
        # 红色半透明覆盖缺陷区域
        red_overlay = np.zeros_like(panel4)
        red_overlay[:, :, 2] = combined
        panel4 = cv2.addWeighted(panel4, 0.7, red_overlay, 0.3, 0)
        # 绘制缺陷轮廓（绿色）
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel4, contours, -1, (0, 255, 0), 2)
        cv2.putText(panel4, "Defect Overlay", (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 拼接 2x2
        top_row = np.hstack([panel1, panel2])
        bottom_row = np.hstack([panel3, panel4])
        vis = np.vstack([top_row, bottom_row])

        # 底部信息条
        info_bar = np.zeros((40, vis.shape[1], 3), dtype=np.uint8)
        info_text = (
            f"{img_name} | "
            f"Pred: {result['class_name']} ({result['confidence']:.3f}) | "
            f"Defect: {result['is_defect']} | "
            f"Anomaly: {result['anomaly_score']:.3f}"
        )
        cv2.putText(info_bar, info_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        vis = np.vstack([vis, info_bar])
        return vis

    # ================================================================
    #  批量推理
    # ================================================================
    def predict_directory(
        self,
        input_dir: str,
        output_dir: str = './results',
        save_mask: bool = True,
        save_heatmap: bool = True,
        save_visualization: bool = True,
    ):
        os.makedirs(output_dir, exist_ok=True)

        img_files = sorted([
            f for f in os.listdir(input_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
        ])

        print(f"\n{'=' * 60}")
        print(f"Inference Settings:")
        print(f"  Input:  {input_dir} ({len(img_files)} images)")
        print(f"  Output: {output_dir}")
        print(f"  Save masks:          {save_mask}")
        print(f"  Save heatmaps:       {save_heatmap}")
        print(f"  Save visualizations: {save_visualization}")
        print(f"{'=' * 60}\n")

        results = []
        for img_file in tqdm(img_files, desc="Inference"):
            img_path = os.path.join(input_dir, img_file)
            try:
                result = self.predict_and_save(
                    img_path, output_dir,
                    save_mask=save_mask,
                    save_heatmap=save_heatmap,
                    save_visualization=save_visualization,
                )
                result['image_name'] = img_file
                results.append(result)
            except Exception as e:
                print(f"\n  Error on {img_file}: {e}")
                results.append({'image_name': img_file, 'error': str(e)})

        # ========== 保存 CSV ==========
        csv_path = os.path.join(output_dir, 'predictions.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'image_name', 'predicted_class', 'class_id',
                'confidence', 'is_defect', 'anomaly_score'
            ])
            for r in results:
                if 'error' not in r:
                    writer.writerow([
                        r['image_name'],
                        r['class_name'],
                        r['class_id'],
                        f"{r['confidence']:.4f}",
                        r['is_defect'],
                        f"{r['anomaly_score']:.4f}"
                    ])

        # ========== 打印统计 ==========
        valid = [r for r in results if 'error' not in r]
        self._print_summary(valid, len(img_files), output_dir,
                            save_mask, save_heatmap, save_visualization)

        return results

    def _print_summary(self, results, total, output_dir,
                       save_mask, save_heatmap, save_vis):
        from collections import Counter

        print(f"\n{'=' * 60}")
        print(f"Inference Complete: {len(results)}/{total} images processed")
        print(f"{'=' * 60}")

        print(f"\nOutput files:")
        print(f"  {output_dir}/predictions.csv")
        if save_mask:
            print(f"  {output_dir}/masks/              (binary masks)")
            print(f"  {output_dir}/prob_maps/           (probability maps)")
        if save_heatmap:
            print(f"  {output_dir}/heatmaps/            (anomaly heatmaps)")
        if save_vis:
            print(f"  {output_dir}/visualizations/      (2x2 panels)")

        # 分类分布
        counter = Counter(r['class_name'] for r in results)
        print(f"\nClass distribution:")
        for cls, cnt in counter.most_common():
            pct = cnt / len(results) * 100
            marker = "  " if cls == "normal" else "!!"
            print(f"  {marker} {cls:40s}: {cnt:4d} ({pct:5.1f}%)")

        # 缺陷统计
        defect = sum(1 for r in results if r['is_defect'])
        normal = len(results) - defect
        print(f"\nSummary:")
        print(f"  Normal:  {normal}")
        print(f"  Defect:  {defect}")

        if results:
            scores = [r['anomaly_score'] for r in results]
            confs = [r['confidence'] for r in results]
            print(f"  Avg anomaly score: {np.mean(scores):.4f} "
                  f"(min={np.min(scores):.4f}, max={np.max(scores):.4f})")
            print(f"  Avg confidence:    {np.mean(confs):.4f} "
                  f"(min={np.min(confs):.4f}, max={np.max(confs):.4f})")

        print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description='Defect Inference (Full Output)')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--image', type=str, default=None,
                        help='Single image path')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Directory of images for batch inference')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda')

    # 输出控制
    parser.add_argument('--save_mask', action='store_true', default=True,
                        help='Save binary masks and probability maps')
    parser.add_argument('--save_heatmap', action='store_true', default=True,
                        help='Save anomaly heatmaps overlaid on original')
    parser.add_argument('--save_vis', action='store_true', default=True,
                        help='Save 2x2 visualization panels')
    parser.add_argument('--no_mask', action='store_true',
                        help='Disable mask saving')
    parser.add_argument('--no_heatmap', action='store_true',
                        help='Disable heatmap saving')
    parser.add_argument('--no_vis', action='store_true',
                        help='Disable visualization saving')

    args = parser.parse_args()

    # 处理 flag
    do_mask = args.save_mask and not args.no_mask
    do_heatmap = args.save_heatmap and not args.no_heatmap
    do_vis = args.save_vis and not args.no_vis

    engine = DefectInference(args.checkpoint, args.device)

    if args.image:
        # ===== 单张推理 =====
        os.makedirs(args.output_dir, exist_ok=True)
        result = engine.predict_and_save(
            args.image, args.output_dir,
            save_mask=do_mask,
            save_heatmap=do_heatmap,
            save_visualization=do_vis,
        )

        print(f"\nSingle Image Result:")
        print(f"  Image:        {args.image}")
        print(f"  Prediction:   {result['class_name']}")
        print(f"  Confidence:   {result['confidence']:.4f}")
        print(f"  Is Defect:    {result['is_defect']}")
        print(f"  Anomaly Score:{result['anomaly_score']:.4f}")
        print(f"\n  All probabilities:")
        for cls, prob in sorted(result['all_probs'].items(), key=lambda x: -x[1]):
            bar = "#" * int(prob * 30)
            print(f"    {cls:40s}: {prob:.4f} {bar}")

    elif args.input_dir:
        # ===== 批量推理 =====
        engine.predict_directory(
            args.input_dir, args.output_dir,
            save_mask=do_mask,
            save_heatmap=do_heatmap,
            save_visualization=do_vis,
        )

    else:
        print("Error: Please specify --image or --input_dir")
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()