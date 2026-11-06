# 布匹：先异常检测，再区域分类（GTX 1660 轻量入口）

新增 `fabric_pipeline.py`，不改变原来的 DINOv2 训练/推理脚本及其旧 checkpoint 格式。
此入口采用轻量 CNN 正常特征统计，不是 DINOv2 模型压缩版。两者 checkpoint 不能混用。

## 原理与输出

1. 共享冻结的 ImageNet MobileNetV3-Small 前半部，输入默认 448×448，输出 stride=16 的 48 通道特征；每张图只运行一次骨干。
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
  --size 448 --epochs 100 --checkpoint checkpoints/fabric_light.pt
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

448输入的 stride16 特征对极小黑点、细线等可能不够，框边界也为特征格级别而非精细分割。增大到480/640或分块可改善小缺陷但降低速度，应比较实际数据上的端到端召回、每类准确率、未知拒识率和误报率。全局对角高斯也不能表示复杂多模态正常纹理，布种/光照改变需重新建立参考；若分类分不开或误报高，需进一步做骨干微调、蒸馏或更强正常分布模型，而不能仅靠降低拒识阈值。

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

`check` 和训练前检查现在会先扫描所有跨集合完全重复文件，一次列出每组的所有完整路径（包括对应的训练集图片），并写入 `results/dataset_duplicates.json`。可用 `--duplicate-report other/report.json` 修改位置。默认跳过与训练集重复的验证图片，`--strict-duplicates` 可恢复报错停止；**不自动移动或删除文件**。报告无重复后才继续检查标签和类别覆盖；无重复时也会更新报告以避免读取上次的旧结果。

```bat
python fabric_pipeline.py check --data-root D:/tyhn/yolo/dataset/source5_augmented/
```

若 normaltrain/normalval 完全重复，过滤后验证集为空时仍会报错，需从真实正常样本重新独立划分；近重复增强图需另外按原始样本/采集批次分组。

## GTX 1660 轻量化优化

原始图片尺寸不固定，不必按原始分辨率输入骨干：模型默认448输入，输出28×28的特征图。可额外训练一个256或320配置做精度/速度对比。256输入的像素数量约为448的33%，不是保证耗时按相同比例降低；细线/小黑点可能损失召回率。推理必须使用权重保存的输入尺寸，不能临时改小而沿用旧正常阈值。

本次优化：
- 正常拟合、正常标定、缺陷训练框提取、验证框提取都支持批量骨干前向，`--feature-batch-size 16`，读取支持 `--workers`。
- 冻结骨干只做特征提取一次，不在100个分类训练epoch里重复运行；默认把48维训练框特征一次放到GPU，避免每轮重复传输。框很多时加 `--cpu-feature-cache` 或减小特征提取batch。
- 正常统计在GPU用FP32做大张量归约，仅48维累计用FP64，避免1660低FP64吞吐拖慢整个特征图计算。
- CPU完成缩放/归一化后一次传输输入，推理自动融合Conv+BN；不改变已存checkpoint，不开启在1660上未必加速的FP16。
- `FabricInference.predict_frame(bgr_uint8_frame)` 接收相机/内存图像，不读盘、不画图、不写盘。仍然每帧仅一次骨干，异常区域批量分类。

当前推荐448配置（输出保持每张图片的实际原始宽高；1660显存不足时将特征提取batch从8降至4）：
```bat
python fabric_pipeline.py train --data-root D:/tyhn/yolo/dataset/source5_augmented/ --device cuda --size 448 --feature-batch-size 8 --workers 2 --checkpoint checkpoints/fabric_448.pt
python fabric_pipeline.py predict --checkpoint checkpoints/fabric_448.pt --source D:/path/to/test --device cuda
python fabric_pipeline.py benchmark --checkpoint checkpoints/fabric_448.pt --source D:/path/to/test --device cuda --iterations 200
```

对比包含画框和PNG写盘的速度：
```bat
python fabric_pipeline.py benchmark --checkpoint checkpoints/fabric_448.pt --source D:/path/to/test --device cuda --iterations 200 --benchmark-vis
```

该模式将每次结果写入 `results/annotated/_benchmark.png`（反复覆盖测速文件），计入画框、重新读原图及写盘时间，不含JSON序列化。实时产线不建议同步为每帧写PNG；建议相机帧直接调用 `predict_frame`，显示/存图由上层按需安排。正常 `predict` 仍默认逐图保存原始尺寸带框结果。

优化后12项CPU单元测试通过（含融合数值等价及批量/单图特征测试），另使用512原图合成数据完成训练→保存权重→加载→内存帧推理冒烟测试。测试随机初始化骨干，不代表精度或GPU性能；未实测GTX1660。目标30FPS时均值和P95建议都小于33.3ms，且需加上相机采集和显示等业务耗时。数据SHA256检查仍会扫描全量图片，这是训练前数据完整性工作，不是每个epoch重复运行。


### 当前选定配置：MobileNet + 448

新增轻量入口默认输入为448；根配置现在对应MobileNet，旧DINOv2配置已单独保存。已有256/320权重仍按自身保存的尺寸推理；要用448必须重新训练并标定正常参考与已知类分布。448比320输入像素数量约多1.96倍，不代表延迟一定增加1.96倍，也不能保证1660实时性。先以batch=1测速并验证细小缺陷召回；实时预测batch仍为1，`--feature-batch-size`只影响离线特征提取。

原图可以是非正方形或不同尺寸：当前预处理直接缩放为448×448（不是等比补边），检测框分别按原图实际宽、高映射回去；保存时保持原图尺寸，不强制输出512×512。大幅面图片缩小后可能丢失微小缺陷，原图读取和绘图写盘耗时也随分辨率变化。


### 默认跳过重复数据（当前行为）

`train` 和 `check` 默认过滤跨集合完全重复图片，不再因存在重复直接中断。保留训练副本，跳过与其重复的正常验证/缺陷验证副本；仅两个验证集合重叠时优先保留normalval。训练角色 normaltrain 与 train 共享的副本保留（不影响训练/验证独立性，但仍应自行核实正常标签）。所有正常建模、阈值标定、分类训练和验证均使用过滤后的列表，不会重新读取被排除的图片；磁盘图片和标签不修改。

报告 `results/dataset_duplicates.json` 新增 `skipped` 完整路径、原因及 `skipped_count`。每个集合打印实际使用图片数和跳过数。若任何必要集合过滤后为空，或训练类别缺失，仍停止并提示补充数据，不会生成假验证结果。`--strict-duplicates` 恢复发现重复即停止。当前自动跳过针对完全重复图片；缺失/非法标签等其他错误仍明确报错，不会静默当作正常图。增强、重新编码近重复不在此过滤范围，仍需按原始样本分组划分。

```bat
python fabric_pipeline.py train --data-root D:/tyhn/yolo/dataset/source5_augmented/ --device cuda --size 448 --feature-batch-size 8
```

本次依赖无关的4项重复审计测试通过；新增过滤列表贯穿检查及空验证集保护的集成测试，当前环境缺少PyTorch未执行该集成测试。


## 配置入口统一（当前版本）

- `config.yaml`：当前 MobileNet 流程，`size: 448`，`data: dataset/fabric19.yaml`，默认特征提取batch=8。
- `dataset/fabric19.yaml`：唯一类别来源，`nc: 19`及19个中文名称，ID为0–18；不在根配置重复写类别列表。
- 「未知类型」为推理拒识ID=19，不是额外训练的softmax输出；正常样本只建立正常参考，不占缺陷分类ID。
- `config_dino_legacy.yaml`：保留原来9类/518的历史DINO配置，仅供旧 `train.py`，不用于当前布匹MobileNet训练。

`fabric_pipeline.py` 现在默认加载根 `config.yaml`，可用 `--config` 指定另一份MobileNet配置；命令行优先于YAML。修改根配置的类别数量不会生效，应修改数据集YAML及实际标签，配置中不接受重复的num_classes/class_names项。推理仍使用checkpoint保存的类别和输入尺寸，改YAML不会改变旧权重。

```bat
python fabric_pipeline.py train --data-root D:/tyhn/yolo/dataset/source5_augmented/ --device cuda
```

上述命令即使用448输入、数据集19类。训练开始会打印实际输入尺寸及已知类别数量。旧DINO的 `train.py` 默认改为读取 `config_dino_legacy.yaml`，误传MobileNet根配置会明确提示正确入口。3项配置测试通过：根配置448/19类、命令行覆盖、拒绝混用旧配置；未重跑模型训练。
