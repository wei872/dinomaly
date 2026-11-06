# 布匹：先异常检测，再区域分类（GTX 1660 轻量入口）

新增 `fabric_pipeline.py`，不改变原来的 DINOv2 训练/推理脚本及其旧 checkpoint 格式。
此入口采用轻量 CNN 正常特征统计，不是 DINOv2 模型压缩版。两者 checkpoint 不能混用。

## 原理与输出

1. 共享冻结的 ImageNet MobileNetV3-Small 前半部，输入默认 320×320，输出 stride=16 的 48 通道特征；每张图只运行一次骨干。
2. 仅 `normaltrain` 更新每通道正常均值/方差，计算各位置标准化特征距离。不使用缺陷图拟合正常分布。
3. 用独立 `normalval` 的 patch 距离建立固定参考分布，默认超过其 99.5% 分位数的 patch 为异常。不存在逐图片 min-max 归一化造成正常图片也被强行拉满的问题。
4. 在异常特征图提取连通区域，仅对区域做分类；支持同一图片多区域、多类别。正常区域不强制归入任何缺陷类。
5. 分类器只用 YOLO 训练框覆盖的特征训练，不把全图多类别标注压成单类别；验证集报告的是 **GT 框分类准确率**，不是检测 mAP 或端到端召回率。
6. 最大分类 softmax 小于 `class_threshold` 或前两类差值小于 `margin`，输出 `未知类型`，ID=19。原 19 类 ID 保持 0–18，不额外插入 normal。未知是拒识结果，不是有样本训练的第20类，也不保证识别所有未见缺陷。

JSON 每个区域包含原图坐标 `bbox_xyxy`、类别、`class_confidence`、`class_margin`、`anomaly_distance` 和 `normal_deviation_confidence`。
最后一个字段是相对于正常验证集 **patch 距离**的经验百分位（0–1），例如 0.999 表示该距离高于约 99.9% 的正常参考 patch；**不是“99.9%概率为缺陷”**。图级分数使用最大 patch 距离，不能解释为图级误报率。`is_defect` 取决于是否有满足最小面积的异常区域，因而可能出现分数高但小区域被过滤的情况。

## 安装与数据

```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-fabric.txt
```

MobileNet 训练初始化会下载 torchvision ImageNet 权重，需要机器能访问其权重服务器，或事先将权重放入 torch hub 缓存；推理加载本地 checkpoint 不下载权重。

读取 `dataset/fabric19.yaml` 的类别和四个路径。当前 YAML 是 Windows 绝对地址，本仓库没有实际布匹数据。支持在原机器直接运行，或把数据挂载后用 `--data-root` 重定位：

```text
/data/source5_augmented/
  images/train/xxx.jpg
  images/val/xxx.jpg
  labels/train/xxx.txt
  labels/val/xxx.txt
  normal/images/train/...
  normal/images/val/...
```

YOLO 行为 `cls cx cy w h`，坐标归一化。支持子目录和一张图片多个框；正常的 YOLO 图片须有空 txt，缺失 txt 报错避免静默漏标。这里只支持检测框，不支持 YOLO polygon 分割标签。`normaltrain/normalval` 直接读图片，不需要 txt；请确保完全正常且无训练/验证图重复、增强副本跨集合泄漏。缺陷训练集须覆盖全部19类。

## 训练前检查

```bash
python fabric_pipeline.py check --data-root /data/source5_augmented
```

检查四个目录、YOLO标签格式、训练集类别覆盖，以及跨集合完全相同文件的 SHA256 重复；输出每个集合图片数和每类框数量。训练自动先执行相同检查，在下载权重前发现数据问题。哈希无法发现重新编码或增强过的近重复图，仍需自行按原始布匹/采集批次划分数据。

## 训练

```bash
python fabric_pipeline.py train --data dataset/fabric19.yaml \
  --data-root /data/source5_augmented --device cuda \
  --size 320 --epochs 100 --checkpoint checkpoints/fabric_light.pt
```

在原 Windows 路径无需 `--data-root`。区域特征暂存在 CPU 内存后训练线性分类头；数据量很大时需要留意内存。骨干完全冻结，不进行监督微调。训练、推理严格使用一致的缩放和归一化。

可调整训练参数并重新生成 checkpoint：
- `--quantile 0.995`：异常 patch 阈值分位数；提高减少误报，但可能漏检。
- `--min-area 2`：最少2个相连特征格；细小缺陷可试1。
- `--class-threshold 0.65 --margin 0.15`：未知拒识阈值。

这些默认值只是起点，应使用独立验证集调整；最终还需要未参与调参的测试集。阈值保存在 checkpoint 中，predict/benchmark 使用 checkpoint 的设置。

## 推理和测速

```bash
python fabric_pipeline.py predict --checkpoint checkpoints/fabric_light.pt \
  --source /data/test --output results/fabric_predictions.json --device cuda

python fabric_pipeline.py benchmark --checkpoint checkpoints/fabric_light.pt \
  --source /data/test --device cuda --iterations 200

python -m unittest discover -s tests -v
```

`--source` 可以是单张图片或目录（递归读取）。输出 JSON，不默认写可视化大图。可在应用中直接复用 `FabricInference(...).predict(path)`。

测速 batch=1，预热10次，CUDA同步计时，包含读图、缩放、CPU/GPU传输、骨干、异常连通域和区域分类，不含 JSON 写盘或相机采集；报告均值、P95、FPS。请使用有代表性的正常/缺陷混合测试图（区域数量也影响耗时），并记录显卡、分辨率、CPU和磁盘条件。当前是 FP32 路径，不假定 GTX 1660 具有 Tensor Core 加速。

## 实时性与精度限制

这是面向 GTX 1660 的轻量基线，**没有在 GTX 1660 实测，不承诺30 FPS或任何帧率**。本开发环境已安装 PyTorch 2.5.1 / torchvision 0.20.1，6项CPU测试通过，包含正常分布拟合、checkpoint保存加载和推理联通测试（使用随机初始化骨干和合成图，仅验证代码）。CUDA不可用，真实数据尚未提供，未完成实际布匹训练与 GPU 测试。若目标30 FPS，应在部署机检查平均及P95是否低于33.3ms；采集、显示和业务逻辑开销需额外计入。

320输入的 stride16 特征对极小黑点、细线等可能不够，框边界也为特征格级别而非精细分割。增大到480/640或分块可改善小缺陷但降低速度，应比较实际数据上的端到端召回、每类准确率、未知拒识率和误报率。全局对角高斯也不能表示复杂多模态正常纹理，布种/光照改变需重新建立参考；若分类分不开或误报高，需进一步做骨干微调、蒸馏或更强正常分布模型，而不能仅靠降低拒识阈值。
