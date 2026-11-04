# angio-train 冠脉造影投照体位分类训练（离线、本机 GPU）

投照体位分类训练流程的**第 2 步**：用 [angio-frames](https://github.com/Acalaxiaka/angio-frames) 的输出（`labels.csv` + `images/`）训练体位分类模型（可选同时判断左/右冠），并导出 ONNX 供后续软件集成。
**训练全程离线**，数据不离开本机（只有 `setup.bat` 安装时需联网下载 PyTorch 等依赖）。本仓库不含任何患者数据。
推荐硬件：Windows 10/11 64 位 + NVIDIA 显卡（已按 RTX 2060 Super 8GB / R7 5700X / 64GB 内存调参）。

## 下载
在 [Releases](../../releases) 下载 `angio-train-v1.0.0-windows.zip`，解压到任意文件夹（建议路径不含空格）。
压缩包已**内置 EfficientNet-B0 ImageNet 预训练权重**（`weights/efficientnet_b0.pth`，16MB）。如需 ResNet-50，另下载 `resnet50.pth` 放入 `weights/`。

## 使用步骤
1. 安装 [Python 3.11](https://www.python.org/downloads/)（安装时勾选 **Add python.exe to PATH**），并确认 NVIDIA 驱动较新（≥ 531）。
2. 双击 `setup.bat`（只需一次，约 10–30 分钟，下载约 3GB）：创建 `venv` 虚拟环境 → 安装 CUDA 版 PyTorch 2.5.1（官方 cu121 → 阿里云镜像 → cu118 依次尝试）→ 清华 pip 镜像安装其它依赖 → 检查权重。最后应显示 `CUDA 可用: True  NVIDIA GeForce RTX 2060 SUPER`。
3. （可选）双击 `自测.bat`，用合成数据检查（最后显示 `RESULT: ALL PASS`）。
4. 双击 `训练.bat`，把 angio-frames 输出文件夹拖进窗口回车，再输入结果文件夹（回车默认 `runs\exp1`）。黑色窗口每轮显示损失/准确率/宏 F1。
5. 中途关闭没关系：再次运行 `训练.bat` 选同一结果文件夹，选 Y 即**续训**。
6. 训练结束自动打开 `misclassified.html`，请复核误分类帧（常见原因：角度接近容差边界导致标签本身有误）。
7. 双击 `预测.bat`，选择模型文件夹和新数据文件夹（PNG 帧，或已脱敏 DICOM），得到逐帧与**序列级投票**结果。

高级（命令行，在本文件夹打开 cmd）：
```
venv\Scripts\python train.py -d D:\frames -o runs\exp1 --cache --multitask
venv\Scripts\python train.py -d D:\frames -o runs\r50 --arch resnet50 --batch 16 --cache
venv\Scripts\python predict.py -m runs\exp1 -i D:\new_frames -o runs\pred
```
| 参数 | 默认 | 说明 |
|---|---|---|
| `--arch` | efficientnet_b0 | 或 resnet50（8GB 显存建议 `--batch 16`）/ 任意 timm 名称 |
| `--weights` | weights\\<arch>.pth | 离线预训练权重路径；`--no-pretrained` 不用 |
| `--img` / `--batch` | 384 / 32 | 显存不足（CUDA out of memory）改 `--batch 16` 或 `--img 320` |
| `--epochs` / `--patience` | 40 / 8 | 最大轮数 / 验证宏 F1 连续 N 轮不升则早停 |
| `--split` / `--seed` | 0.7,0.15,0.15 / 42 | **按患者**划分训练/验证/测试，固定种子可复现 |
| `--keep-unknown` | 关 | 默认丢弃 UNKNOWN（角度缺失） |
| `--min-count` | 20 | 序列数少于此值的体位类别丢弃（记录在 metrics.json） |
| `--multitask` | 关（训练.bat 开） | 增加左/右冠第二输出头；side=unknown 的样本不计该头损失 |
| `--balance` | sampler | 类别均衡采样；`loss` 加权损失；`none` 不处理 |
| `--cache` | 关（训练.bat 开） | 图片预载入内存（384² 约 0.15MB/张，1.5 万帧约 2.2GB） |
| `--workers` | Windows 4（缓存时 2） | DataLoader 进程数 |
| `--resume` | — | 从结果文件夹 `last.pt` 续训 |
| `--no-amp` / `--cpu` | — | 关闭混合精度 / 强制 CPU |

## 预训练权重（只需一次，之后离线）
- 发布包已内置 `efficientnet_b0.pth`。若缺失，`setup.bat` 会运行 `download_weights.py`，**先试 hf-mirror.com（国内镜像）再试 huggingface.co**。
- 手动下载：浏览器打开 `https://hf-mirror.com/timm/efficientnet_b0.ra_in1k/tree/main`（ResNet-50：`timm/resnet50.a1_in1k`）下载 `model.safetensors`，然后
  `venv\Scripts\python convert_weights.py model.safetensors weights\efficientnet_b0.pth`，或训练时 `--weights 路径`。

## 输出内容
```
结果文件夹/
  best.pt / last.pt              最佳模型（按验证集宏 F1） / 续训检查点
  model.onnx                     ONNX（opset 17，输入 input: N×3×384×384，输出 view_logits [, side_logits]）
  config.json                    类别顺序、输入尺寸、归一化参数（集成时必读）
  metrics.json                   测试集帧级/序列级准确率与宏 F1、每类指标、每轮训练曲线、参数
  classification_report.txt      每类精确率/召回率/F1（帧级）+ 序列级准确率
  per_class_report.csv           同上（Excel 可打开）
  confusion_matrix.png           帧级混淆矩阵；confusion_matrix_series.png 序列级
  misclassified.html             测试集误分类画廊（图片 + 真/预测 + 置信度 + 角度），浏览器打开
  split.csv                      每帧所属 train/val/test（可审计无患者泄漏）
预测输出/
  predictions_frames.csv         逐帧预测与各类概率
  predictions_series.csv         序列级投票（各帧概率平均）、置信度、帧间一致率、needs_review（低置信度需复核）
```
ONNX 预处理：灰度图等比缩放到 384×384（黑边补齐）→ `x/255` → `(x-0.449)/0.226` → 复制成 3 通道。

## 方法说明
- 骨干：timm EfficientNet-B0（默认）或 ResNet-50，ImageNet 预训练；AdamW + OneCycle 学习率，标签平滑 0.05，AMP 混合精度，channels_last。
- 划分：**按 patient_id 划分**，同一患者的所有序列/帧只在一个集合（自测验证无泄漏）。早停与最佳模型依据验证集宏 F1，测试集只在最后评估一次。
- 增强（适合 X 线）：±10° 旋转、±6% 平移、0.9–1.1 缩放、亮度/对比度/伽马、高斯噪声、轻度模糊。
- **⚠ 不做水平/垂直翻转**：造影图像左右翻转后，LAO 视图看起来像 RAO（解剖左右互换），会**改变体位语义、污染标签**。自测会检查代码中没有翻转。
- 序列级投票：同一序列（angio-frames 文件名 `..._f006` 前缀相同）各帧 softmax 概率平均后取最大。

## 预计训练时间（RTX 2060 Super 8GB, AMP, 384）
| 数据量 | EfficientNet-B0 batch 32 | ResNet-50 batch 16–24 |
|---|---|---|
| ~2000 序列 × 5 帧 ≈ 1 万帧 | 每轮约 1–1.5 分钟；早停通常 20–30 轮 → **约 30–45 分钟** | 每轮约 3–4 分钟 → 约 1.5–2 小时 |
| ~5000 序列 ≈ 2.5 万帧 | 每轮约 2.5–3.5 分钟 → **约 1–1.5 小时** | 约 3–5 小时 |
首次开启 `--cache` 需额外 1–3 分钟读图。以上为估算，实际以窗口中每轮 `sec` 为准。

## 局限
- 标签来自机架角度（angio-frames），角度边界附近的标签本身可能“错误”；模型学到的是角度规则的视觉近似。请用误分类画廊复核并修正 labels.csv 后重训。
- 类别不均衡明显时（如 AP_CAU、单纯 AP 很少），少数类召回率可能较低；`--min-count` 会丢弃过少的类。
- side 头依赖序列描述关键词，unknown 多时该头效果有限。
- 不同医院/设备（Siemens vs Philips、不同窗宽窗位）差异大时，跨中心泛化需外部验证；测试集仅来自同一数据源。
- 本工具为科研用途，未经临床验证，不可用于诊断。
- 未提供 exe（PyTorch 体积约 3GB，打包不划算）；需按上方步骤安装 Python 环境。

## 开发
`python selftest.py`：合成数据 → 患者级划分无泄漏/可复现 → CPU 极短训练（多任务）→ 报告文件 → 续训 → ONNX 导出并与 PyTorch 输出一致 → 加权损失+缓存 → 预测+序列投票 → 无翻转检查。
