                        输入图片 (B, 3, 518, 518)
                              │
                ┌─────────────▼──────────────┐
                │   DINOv2 ViT-B/14 (冻结)    │
                │   86.58M 参数，全部不训练     │
                │                             │
                │   提取3层特征:                │
                │   Layer 3:  (B, 1369, 768)  │
                │   Layer 7:  (B, 1369, 768)  │
                │   Layer 11: (B, 1369, 768)  │
                └────────┬───────────────┬────┘
                         │               │
            ┌────────────▼────┐    ┌─────▼──────────────────┐
            │   Branch A:      │    │   Branch B:             │
            │   原始特征        │    │   重建差异流             │
            │   直接取最后一层  │    │   (Dinomaly 核心)       │
            │   (B, 1369, 768) │    │   4.34M 可训练参数       │
            │                  │    │                         │
            │                  │    │ 1. 多层聚合              │
            │                  │    │ 2. 加噪瓶颈              │
            │                  │    │ 3. Linear Attn Decoder  │
            │                  │    │ 4. 计算余弦距离          │
            │                  │    │                         │
            │                  │    │ 输出:                    │
            │                  │    │  异常特征 (B,1369,256)   │
            │                  │    │  异常分数 (B,1369)       │
            │                  │    │  重建损失 (scalar)       │
            └────────┬────────┘    └──────────┬──────────────┘
                     │                        │
            ┌────────▼────────────────────────▼────┐
            │        双流融合模块 (4.47M)            │
            │                                       │
            │  原始特征(768d) ──→ 投影(512d) ──┐     │
            │                                  ├─→ 融合  │
            │  异常特征(256d) ──→ 投影(512d) ──┘     │
            │                                       │
            │  交叉注意力 + 门控融合                   │
            │  输出: (B, 1369, 512)                  │
            └───────────┬───────────────────────────┘
                        │
          ┌─────────────┼─────────────┐
          │             │             │
    ┌─────▼─────┐ ┌────▼─────┐ ┌────▼──────┐
    │ 分类头     │ │ 分割头    │ │ 异常评分头 │
    │ 0.30M     │ │ 0.16M    │ │ 0.07M     │
    │           │ │          │ │           │
    │ 全局池化   │ │ patch→   │ │ 全局池化   │
    │ +异常加权  │ │ pixel    │ │ → sigmoid │
    │ → 9类分类  │ │ 上采样    │ │           │
    └─────┬─────┘ └────┬─────┘ └────┬──────┘
          │            │            │
    class_logits  mask_pred   anomaly_prob
    (B, 9)       (B,1,518,518) (B, 1)


模型的优势:
├── 小数据友好 (冻结 DINOv2，只训练 9.34M)
├── 可解释性强 (热力图直观展示异常区域)
├── 多输出实用 (分类+掩膜+评分 一次推理全搞定)
└── 核心思想优雅 ("故意限制能力" 来检测异常)

模型的局限:
├── 对小数据仍然参数偏多 (9.34M vs 915张)
├── 重建差异不能直接区分缺陷类型
├── Linear Attention 的异常检测优势在有监督场景下减弱
└── 分割头在数据少时效果有限


[//]: # (Step 1: 生成演示数据集（验证代码可运行）)

# 生成随机演示数据
python tools/prepare_data.py --mode demo --output_dir ./dataset --num_per_class 50
  

[//]: # (Step 2: 或者从 YOLO 格式转换你的真实数据)

# 从 YOLO 分割格式转换

python tools/convert_yolo_format.py --input_dir ./dataset/yolo --output_dir ./dataset/dino --classes small-black-dot non-overlapping-double-latitude folded-edge-not-included feathering small-tail meridian-anomaly latitude-pollution meridian-pollution
  

[//]: # (Step 3: 训练)


# 指定自定义配置
python train.py --config config.yaml --batch_size 8 --stage2_epochs 200


[//]: # (Step 4: 推理)

# 单张图推理
python inference.py \
    --checkpoint checkpoints/best_acc.pth \
    --image test.jpg \
    --visualize

# 批量推理
python inference.py --checkpoint checkpoints/best_acc.pth --input_dir ./dataset/20260304 --output_dir ./results-20260304