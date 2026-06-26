# 🌾 谷物颗粒品质分级检测数据集 (Grain Quality Grading Detection Dataset)

> 基于 MMDetection + RTMDet 的谷物颗粒目标检测与品质分级项目

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Framework: MMDetection](https://img.shields.io/badge/Framework-MMDetection-blue.svg)](https://github.com/open-mmlab/mmdetection)
[![Model: RTMDet](https://img.shields.io/badge/Model-RTMDet--tiny-green.svg)]()

---

## 📋 目录

- [1. 研究背景与意义](#1-研究背景与意义)
- [2. 研究现状](#2-研究现状)
- [3. 数据集概览](#3-数据集概览)
- [4. 模型设计](#4-模型设计)
- [5. 目录结构](#5-目录结构)
- [6. 环境配置](#6-环境配置)
- [7. 训练流程](#7-训练流程)
- [8. 检测结果](#8-检测结果)
- [9. 模型部署](#9-模型部署)
- [10. Bug记录与调试方案](#10-bug记录与调试方案)
- [11. 开源发布](#11-开源发布)
- [12. 许可证](#12-许可证)

---

## 1. 研究背景与意义

### 1.1 研究背景

谷物（如大米、小麦、玉米、大豆等）是人类最主要的食物来源，其品质直接关系到食品安全和经济效益。传统谷物品质分级主要依赖人工目测，存在以下痛点：

- 👁️ **主观性强**：不同质检员判断标准不一致
- 🐌 **效率低下**：人工检测速度有限，无法满足大规模生产需求
- 😫 **疲劳误判**：长时间工作后准确率大幅下降
- 💰 **人力成本高**：需要培训专业质检人员

计算机视觉技术，特别是深度学习目标检测算法，为解决上述问题提供了有效的技术手段。

### 1.2 研究意义

1. **保障粮食安全**：自动化的品质分级可有效筛选不合格谷物，防止霉变谷物进入食品供应链
2. **提升产业效率**：替代人工检测，实现24/7不间断自动化品质管控
3. **标准化评价**：建立统一的品质检测标准，消除主观因素影响
4. **降低人力成本**：减少对专业质检人员的依赖
5. **可追溯管理**：检测结果数字化存储，支持品质追溯和数据分析

---

## 2. 研究现状

### 2.1 传统方法

- **图像处理法**：基于颜色阈值、形态学操作提取谷物特征（如长短轴比、颜色矩等），再通过SVM/随机森林进行分类
- **高光谱成像**：利用不同波长光谱分析谷物内部品质，但设备昂贵、速度慢
- **X射线检测**：可检测内部虫蛀，但安全性和成本是问题

### 2.2 深度学习方案

近年来的主流方法均基于深度学习：

| 方法 | 特点 | 局限性 |
|------|------|--------|
| YOLO系列 | 单阶段检测，速度快 | 小目标检测能力弱 |
| Faster R-CNN | 两阶段检测，精度高 | 推理速度慢 |
| RTMDet (本项目) | 兼顾精度与速度 | 需要足够训练数据 |
| DETR系列 | 端到端检测 | 训练收敛慢 |

### 2.3 本项目的技术选型

本项目采用 **RTMDet-tiny**（Real-Time Models for Object Detection），基于 MMDetection 框架实现。RTMDet 是 OpenMMLab 在 2023 年提出的实时目标检测模型，具有以下优势：

- **CSPNeXt 骨干网络**：改进的跨阶段局部网络，计算效率高
- **软标签动态分配**：Dynamic Soft Label Assigner，正负样本分配更精准
- **QFL (Quality Focal Loss)**：分类与定位质量联合学习
- **GIoU Loss**：更优的边界框回归损失
- **Mosaic + MixUp 数据增强**：提升小目标和难样本的检测能力
- **两阶段训练策略**：前280 epoch使用强增强，后20 epoch使用弱增强微调

---

## 3. 数据集概览

### 3.1 数据集来源

数据通过高清工业相机采集谷物颗粒图像，使用 Label Studio 进行 COCO 格式标注。

### 3.2 数据采集方案

| 参数 | 配置 |
|------|------|
| 采集设备 | 工业相机 (建议分辨率 ≥ 1920×1080) |
| 光源 | LED环形光源，均匀照明 |
| 背景 | 黑色/白色绒布，减少反光 |
| 拍摄距离 | 30-50cm |
| 每张颗粒数 | 5-15粒 |

### 3.3 类别定义

| ID | 类别名 | 英文 | 说明 | 标注颜色 |
|----|--------|------|------|----------|
| 0 | 完整粒 | `wanzheng` | 颗粒完整、无破损、无霉变、色泽正常 | 🟢 绿色 |
| 1 | 破损粒 | `posun` | 颗粒有明显裂纹、断裂、缺损 | 🔴 红色 |
| 2 | 霉变粒 | `meibian` | 表面有霉斑、菌丝、变色霉变区域 | ⚫ 灰色 |
| 3 | 异色粒 | `yise` | 颜色异常（发黄、发黑、发白等），但未霉变 | 🟡 黄色 |

### 3.4 数据集划分

| 集合 | 图像数 | 标注数 | 占比 |
|------|--------|--------|------|
| Train (训练集) | 14 | 45 | ~70% |
| Val (验证集) | 3 | 12 | ~15% |
| Test (测试集) | 3 | 10 | ~15% |

> ⚠️ **注意**：以上为模板数据量。实际使用时应采集 **≥100张训练/≥30张验证/≥30张测试** 以确保模型泛化能力。

---

## 4. 模型设计

### 4.1 整体架构

```
输入图像 (640×640)
    │
    ▼
┌─────────────────────┐
│   CSPNeXt Backbone  │  ← CSPNeXt-tiny (ImageNet预训练)
│  ┌───┬───┬───┬───┐  │
│  │C1 │C2 │C3 │C4 │  │    输出: [96, 192, 384] channels
│  └───┴───┴───┴───┘  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   CSPNeXtPAFPN Neck │  ← 特征金字塔 (PAFPN)
│   ┌───┐   ┌───┐     │
│   │P3 │←→│P4 │←→│P5││    多尺度特征融合
│   └───┘   └───┘     │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   RTMDetSepBN Head  │  ← 解耦检测头
│  ┌──────┬──────┐     │
│  │Cls   │Reg   │     │    4 类别 × 1 anchor
│  │Branch│Branch│     │
│  └──────┴──────┘     │
└─────────┬───────────┘
          │
          ▼
    检测输出: [cls, bbox]
```

### 4.2 关键创新点

1. **Dynamic Soft Label Assigner**：动态软标签分配策略，根据预测质量自适应分配正负样本
2. **CSPNeXt + Channel Attention**：引入通道注意力机制，增强特征表达能力
3. **EMA (Exponential Moving Average)**：模型参数指数移动平均，提升推理稳定性
4. **Pipeline Switch Hook**：训练后期切换数据增强策略，从强增强平滑过渡到弱增强

### 4.3 损失函数

| 损失 | 类型 | 权重 | 说明 |
|------|------|------|------|
| 分类损失 | QualityFocalLoss | 1.0 | 质量感知的焦点损失 |
| 回归损失 | GIoULoss | 2.0 | 广义IoU损失 |

---

## 5. 目录结构

```
谷物颗粒品质分级检测数据集/
│
├── README.md                           # 本文档
├── 实验报告.md                          # 完整实验报告
│
├── configs/
│   └── rtmdet/
│       └── my_rtmdet_tiny_8xb32-300e_coco.py  # ★ 训练配置文件
│
├── date/
│   └── coco/
│       ├── annotations/
│       │   ├── instances_train.json     # 训练集标注 (COCO格式)
│       │   ├── instances_val.json       # 验证集标注 (COCO格式)
│       │   └── instances_test.json      # 测试集标注 (COCO格式)
│       └── images/
│           ├── train/                   # 训练集图片 (*.jpg)
│           ├── val/                     # 验证集图片 (*.jpg)
│           └── test/                    # 测试集图片 (*.jpg)
│
├── work_dirs/
│   └── rtmdet_tiny_8xb32-300e_coco/    # 训练输出目录 (checkpoints/logs)
│
├── scripts/
│   ├── train.py                         # 训练脚本
│   ├── infer.py                         # 推理脚本
│   ├── check_dataset.py                 # 数据集校验脚本
│   └── convert_labelstudio.py           # Label Studio 导出转换脚本
│
├── output/                              # 推理结果输出
│
└── tools/                               # MMDetection 工具脚本
    ├── train.py                         # 训练入口
    ├── test.py                          # 测试入口
    └── analysis_tools/                  # 分析工具
```

---

## 6. 环境配置

### 6.1 依赖安装

```bash
# 1. 安装 PyTorch (根据CUDA版本选择)
# CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# 2. 安装 MMEngine, MMCV, MMDetection
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.0.0"

# 3. 验证安装
python -c "import mmdet; print(mmdet.__version__)"
```

### 6.2 预训练权重下载

```bash
# 下载 CSPNeXt-tiny ImageNet 预训练权重
wget https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth -P date/
```

---

## 7. 训练流程（详细指令流程）

### 7.1 数据准备

```bash
# 步骤1: 将采集的图片按比例放入对应文件夹
# 训练集图片 → date/coco/images/train/
# 验证集图片 → date/coco/images/val/
# 测试集图片 → date/coco/images/test/

# 步骤2: 使用 Label Studio 标注数据，导出为 COCO JSON 格式
# 或使用 scripts/convert_labelstudio.py 转换
python scripts/convert_labelstudio.py \
    --input output/project-xxx.json \
    --output date/coco/annotations/

# 步骤3: 检查数据集完整性
python scripts/check_dataset.py
```

### 7.2 配置文件修改

编辑 `configs/rtmdet/my_rtmdet_tiny_8xb32-300e_coco.py`：

```python
# 关键修改项:
# 1. num_classes: 改为你的类别数 (当前: 4)
# 2. data_root: 数据集根目录
# 3. metainfo.classes: 类别名称元组

# 当前配置已预设为:
# - 4个类别: wanzheng, posun, meibian, yise
# - 使用相对路径 date/coco/
```

### 7.3 开始训练

```bash
# 单 GPU 训练 (推荐)
python tools/train.py configs/rtmdet/my_rtmdet_tiny_8xb32-300e_coco.py

# 多 GPU 训练 (2卡)
bash tools/dist_train.sh configs/rtmdet/my_rtmdet_tiny_8xb32-300e_coco.py 2

# 从 checkpoint 恢复训练
python tools/train.py configs/rtmdet/my_rtmdet_tiny_8xb32-300e_coco.py --resume
```

### 7.4 模型评估

```bash
# 验证集评估
python tools/test.py configs/rtmdet/my_rtmdet_tiny_8xb32-300e_coco.py \
    work_dirs/rtmdet_tiny_8xb32-300e_coco/epoch_300.pth

# 测试集评估
python tools/test.py configs/rtmdet/my_rtmdet_tiny_8xb32-300e_coco.py \
    work_dirs/rtmdet_tiny_8xb32-300e_coco/epoch_300.pth \
    --cfg-options test_dataloader.dataset.ann_file=date/coco/annotations/instances_test.json
```

### 7.5 可视化推理

```bash
# 单张图片推理
python scripts/infer.py \
    --img date/coco/images/test/grain_test_001.jpg \
    --config configs/rtmdet/my_rtmdet_tiny_8xb32-300e_coco.py \
    --checkpoint work_dirs/rtmdet_tiny_8xb32-300e_coco/epoch_300.pth \
    --out output/result.jpg \
    --score-thr 0.3

# 批量推理
python tools/test.py configs/rtmdet/my_rtmdet_tiny_8xb32-300e_coco.py \
    work_dirs/rtmdet_tiny_8xb32-300e_coco/epoch_300.pth \
    --show-dir output/vis/
```

---

## 8. 检测结果

### 8.1 评价指标

| 指标 | 说明 |
|------|------|
| mAP@0.5 | IoU=0.5 时的平均精度 |
| mAP@0.5:0.95 | IoU从0.5到0.95的平均精度 |
| AP_wanzheng | 完整粒类别精度 |
| AP_posun | 破损粒类别精度 |
| AP_meibian | 霉变粒类别精度 |
| AP_yise | 异色粒类别精度 |

### 8.2 结果记录表

| Epoch | mAP@0.5 | mAP@0.5:0.95 | AP_wanzheng | AP_posun | AP_meibian | AP_yise | 备注 |
|-------|---------|--------------|-------------|----------|------------|---------|------|
| 10 | -- | -- | -- | -- | -- | -- | 初始训练 |
| 50 | -- | -- | -- | -- | -- | -- | |
| 100 | -- | -- | -- | -- | -- | -- | |
| 150 | -- | -- | -- | -- | -- | -- | |
| 200 | -- | -- | -- | -- | -- | -- | |
| 250 | -- | -- | -- | -- | -- | -- | |
| 280 | -- | -- | -- | -- | -- | -- | 切换弱增强 |
| 300 | -- | -- | -- | -- | -- | -- | 最终结果 |

> 📝 **请在实际训练后填写上表**

### 8.3 检测效果示例

```
┌──────────────────────────────────────────────┐
│  Input: grain_test_001.jpg                    │
│  ┌──────────────────────────────────┐         │
│  │  🟢 wanzheng  0.95               │         │
│  │  ┌──────┐                        │         │
│  │  │ 完整  │  🔴 posun  0.87      │         │
│  │  │ 颗粒  │  ┌──┐                │         │
│  │  └──────┘  │破│                │         │
│  │            └──┘                │         │
│  │   ⚫ meibian 0.78               │         │
│  │   ┌─────┐    🟡 yise  0.82     │         │
│  │   │霉变 │    ┌────┐            │         │
│  │   └─────┘    │异色│            │         │
│  │              └────┘            │         │
│  └──────────────────────────────────┘         │
│  Summary: 4 grains detected                   │
│    完整粒×1, 破损粒×1, 霉变粒×1, 异色粒×1      │
└──────────────────────────────────────────────┘
```

---

## 9. 模型部署

### 9.1 ONNX 导出

```bash
# 导出为 ONNX 格式
python tools/deployment/pytorch2onnx.py \
    configs/rtmdet/my_rtmdet_tiny_8xb32-300e_coco.py \
    work_dirs/rtmdet_tiny_8xb32-300e_coco/epoch_300.pth \
    --output-file work_dirs/rtmdet.onnx \
    --input-img date/coco/images/test/grain_test_001.jpg \
    --dynamic-export
```

### 9.2 TensorRT 部署 (FP16/INT8)

```bash
# FP16 精度 - 推理速度提升 ~2x
python tools/deployment/onnx2tensorrt.py \
    work_dirs/rtmdet.onnx \
    --fp16 \
    --output-file work_dirs/rtmdet_fp16.trt

# INT8 精度 - 推理速度提升 ~4x (需校准数据)
python tools/deployment/onnx2tensorrt.py \
    work_dirs/rtmdet.onnx \
    --int8 \
    --calib-dataset date/coco/images/train/ \
    --output-file work_dirs/rtmdet_int8.trt
```

### 9.3 NCNN 部署 (移动端)

```bash
# ONNX → NCNN
python tools/deployment/pytorch2onnx.py \
    configs/rtmdet/my_rtmdet_tiny_8xb32-300e_coco.py \
    work_dirs/rtmdet_tiny_8xb32-300e_coco/epoch_300.pth \
    --output-file work_dirs/rtmdet.onnx \
    --simplify

onnx2ncnn work_dirs/rtmdet.onnx work_dirs/rtmdet.param work_dirs/rtmdet.bin
```

---

## 10. Bug记录与调试方案

### Bug #1: CUDA Out of Memory (OOM)

**现象**：
```
RuntimeError: CUDA out of memory. Tried to allocate XX MiB
```

**原因**：batch_size 过大或输入图片尺寸过大

**解决方案**：
```python
# 在配置文件中减小 batch_size
train_dataloader = dict(batch_size=8)  # 原16 → 8

# 或减小输入尺寸
img_scale=(480, 480)  # 原640 → 480
```

### Bug #2: 类别数量不匹配

**现象**：
```
AssertionError: The `num_classes` (4) in bbox_head should match the length of `metainfo.classes` (3)
```

**解决方案**：确保配置文件中 METAINFO.classes、bbox_head.num_classes 数量一致

### Bug #3: 预训练权重加载失败

**现象**：
```
WARNING: checkpoint is not loaded: backbone.xxx missing keys
```

**原因**：预训练权重路径错误或网络连接问题

**解决方案**：
```bash
# 手动下载权重
wget https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth -P date/

# 或在配置中修改路径
init_cfg=dict(checkpoint='date/cspnext-tiny_imagenet_600e.pth', ...)
```

### Bug #4: 标注文件路径问题

**现象**：
```
FileNotFoundError: date/coco/annotations/instances_train.json
```

**原因**：data_root 相对路径解析不正确

**解决方案**：
```python
# 使用绝对路径
data_root = 'D:/xxx/谷物颗粒品质分级检测数据集/date/coco/'
```

### Bug #5: Mosaic增强与小数据集不兼容

**现象**：训练时 loss 不下降或为 NaN

**原因**：Mosaic 增强在少量数据时可能产生无意义的拼接图

**解决方案**：
```python
# 数据量 < 50 张时，注释掉 CachedMosaic 和 CachedMixUp
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='CachedMosaic', ...),  # 暂时禁用
    dict(type='RandomResize', ...),
    dict(type='RandomCrop', ...),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', ...),
    # dict(type='CachedMixUp', ...),   # 暂时禁用
    dict(type='PackDetInputs'),
]
```

### Bug #6: Windows 下分布式训练报错

**现象**：
```
RuntimeError: Nccl error: unhandled system error
```

**解决方案**：
```bash
# Windows 下使用单 GPU 训练
python tools/train.py configs/rtmdet/my_rtmdet_tiny_8xb32-300e_coco.py
```

---

## 11. 开源发布

### 11.1 GitHub 上传

```bash
# 1. 初始化 Git 仓库
git init
git add README.md .gitignore configs/ scripts/ date/coco/annotations/
git commit -m "feat: 谷物颗粒品质分级检测数据集 - 初始版本"

# 2. 创建 GitHub 仓库并推送
git remote add origin https://github.com/YOUR_USERNAME/grain-quality-detection.git
git branch -M main
git push -u origin main

# 3. 打标签发布
git tag -a v1.0 -m "v1.0: 初始数据集发布，包含4类谷物品质标注"
git push origin v1.0
```

### 11.2 Hugging Face 数据集上传

```bash
# 1. 安装 huggingface_hub
pip install huggingface_hub datasets

# 2. 登录
huggingface-cli login

# 3. 上传数据集
python scripts/upload_to_huggingface.py
```

---

## 12. 许可证

- 数据集：CC BY 4.0
- 代码：Apache License 2.0 (基于 MMDetection)

---

## 📚 参考文献

1. Lyu C, Zhang W, Huang H, et al. RTMDet: An Empirical Study of Designing Real-Time Object Detectors. arXiv, 2022.
2. Liu Z, Mao H, Wu C Y, et al. A ConvNet for the 2020s. CVPR, 2022.
3. Li X, Wang W, Wu L, et al. Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection. NeurIPS, 2020.
4. MMDetection Contributors. OpenMMLab Detection Toolbox and Benchmark. 2018.

---

> 📧 如有问题，欢迎提 Issue 或 Pull Request！
