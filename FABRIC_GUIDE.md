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

`--source` 可以是单张图片或目录（递归读取）。默认同时输出 JSON 和原始分辨率带框 PNG 图片（不覆盖原图）。可在应用中直接复用 `FabricInference(...).predict(path)`。

测速 batch=1，预热10次，CUDA同步计时，包含读图、缩放、CPU/GPU传输、骨干、异常连通域和区域分类，不含 JSON 写盘或相机采集；报告均值、P95、FPS。请使用有代表性的正常/缺陷混合测试图（区域数量也影响耗时），并记录显卡、分辨率、CPU和磁盘条件。当前是 FP32 路径，不假定 GTX 1660 具有 Tensor Core 加速。

## 实时性与精度限制

这是面向 GTX 1660 的轻量基线，**没有在 GTX 1660 实测，不承诺30 FPS或任何帧率**。本开发环境已安装 PyTorch 2.5.1 / torchvision 0.20.1，6项CPU测试通过，包含正常分布拟合、checkpoint保存加载和推理联通测试（使用随机初始化骨干和合成图，仅验证代码）。CUDA不可用，真实数据尚未提供，未完成实际布匹训练与 GPU 测试。若目标30 FPS，应在部署机检查平均及P95是否低于33.3ms；采集、显示和业务逻辑开销需额外计入。

320输入的 stride16 特征对极小黑点、细线等可能不够，框边界也为特征格级别而非精细分割。增大到480/640或分块可改善小缺陷但降低速度，应比较实际数据上的端到端召回、每类准确率、未知拒识率和误报率。全局对角高斯也不能表示复杂多模态正常纹理，布种/光照改变需重新建立参考；若分类分不开或误报高，需进一步做骨干微调、蒸馏或更强正常分布模型，而不能仅靠降低拒识阈值。

## 未知异常与类别分不清（新增拒识）

检测完全由正常参考距离决定，不依赖19类 softmax：符合异常阈值及面积要求的区域都会保留检测框。
对已检出区域，满足以下任意条件即分到 ID=19「未知类型」：

1. 区域特征与预测已知类的标准化距离超出该类范围，即使 softmax 很高也拒识；
2. 最大 softmax 低于分类阈值；
3. 前两类分数差太小，类别分不清。

训练时从各类别 YOLO 框特征拟合均值、对角方差、距离半径，每类至少需要2个框；`--known-quantile .99` 控制训练距离半径分位数。使用方差下限避免常量特征产生除零。此半径是训练集拟合的启发式起点，不是经过未知测试集验证的开放集保证；应在独立已知/未见类验证数据上调整，特别注意 GT 框与自动异常框的分布差异。

输出新增 `unknown_reasons`（`outside_known_distribution`、`low_class_confidence`、`ambiguous_classes`）、`candidate_known_class`、`known_class_distance`、`known_class_radius`。`class_confidence` 仍表示候选已知类的 softmax，**不是未知类别概率**。

**已有旧权重需要重新训练**才能获得已知类分布。兼容旧权重推理时 `known_distribution_available=false`，仍可低置信度/模糊拒识，但没有分布距离拒识。

本次8项CPU测试通过，其中使用受控特征验证了：异常区域即使被分类器以>99%置信度分到已知类，只要超出已知分布，也会保留框并输出未知类型。此测试验证代码逻辑，不代表真实未知缺陷召回率。所有未知检出仍以正常参考模型能区分该缺陷为前提，不保证与正常特征相近的未知缺陷被检出。


## 最终预测图片输出

`predict` 默认把检测框画回原始分辨率图片，保存到 `results/annotated/`，并在 JSON 的 `annotated_image` 中记录路径。递归目录保留子目录；输出文件为 `原文件名.png`（例如 `a.jpg.png`），避免不同扩展名同名文件覆盖。无异常时保存不加框的原图像素副本。

- 红框：已知类别，显示「类别 | 分类: xx% | 偏离正常: xx%」。
- 橙框：未知类型，显示「未知类型 | 偏离正常: xx%」。未知不是有监督概率类，不能把候选已知类 softmax 标成未知置信度；候选已知类分数仍保留在 JSON 中。
- 不生成热力图或拼图，图片尺寸不变，不覆盖输入文件。

```bash
python fabric_pipeline.py predict --source /data/test --device cuda \
  --vis-dir results/annotated --font /path/to/NotoSansCJK-Regular.ttc
```

Windows 自动尝试微软雅黑/黑体，Linux 自动尝试 Noto CJK，macOS 自动尝试苹方；也可设置 `FABRIC_FONT`。找不到中文字体会明确报错，避免无声生成乱码。`--font` 必须指向支持中文的字体。`--no-vis` 可恢复只输出 JSON；`benchmark` 不画框写图片，部署时可视化开销需要另外计入。

本次单独执行了渲染冒烟测试，验证原始尺寸、已知/未知框颜色、正常图不画框及禁止覆盖输入；当前环境缺少 PyTorch，未重跑完整模型测试。中文实际字形取决于部署时提供的字体。

### 重复数据报告

`check` 和训练前检查现在会先扫描所有跨集合完全重复文件，一次列出每组的所有完整路径（包括对应的训练集图片），并写入 `results/dataset_duplicates.json`。可用 `--duplicate-report other/report.json` 修改位置。发现重复时仍停止训练，**不自动移动或删除文件**。报告无重复后才继续检查标签和类别覆盖；无重复时也会更新报告以避免读取上次的旧结果。

```bat
python fabric_pipeline.py check --data-root D:/tyhn/yolo/dataset/source5_augmented/
```

若 normaltrain/normalval 完全重复，需从真实正常样本重新独立划分，不能把验证集全部移空。先备份数据，按报告处理重复，再重新运行检查；近重复增强图需另外按原始样本/采集批次分组。
