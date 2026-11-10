# SEG-BONE — 骨缺损（牙周缺损）CT 分割

基于深度学习对 CT 影像中的骨缺损（Periodontal Defect / Bone Defect）进行二分类分割的代码库。项目集成并对比了多类分割模型：MONAI 3D 模型、`segmentation-models-pytorch`（SMP）2D / 2.5D 模型，以及 nnU-Net v2，并提供了统一的训练、评估、集成与汇总流程。

## 目录结构

```
SEG-BONE/
├── configs/                  # 各模型的 JSON 配置文件
│   ├── unetpp_2d.json
│   ├── unetpp_25d.json
│   ├── deeplabv3p_2d.json
│   ├── segresnet_d1.json
│   ├── attention_unet_3d.json
│   ├── dynunet.json
│   ├── unetr.json
│   ├── swunetr.json
│   ├── mednext_small.json
│   └── nnunet.json
├── models/                   # 模型构建
│   ├── __init__.py           # build_model 入口，区分 2D/2.5D/3D
│   ├── smp2d.py              # SMP 2D / 2.5D 模型
│   └── monai3d.py            # MONAI 3D 模型
├── nnunet/
│   └── runner.py             # nnU-Net v2 全流程封装
├── dataset.py                # 数据加载、病例管理、patch 采样
├── train.py                  # 训练脚本
├── evaluate.py               # 评估脚本（Dice、体积、ICC、滑窗推理）
├── ensemble.py               # 多模型集成（多数投票）
├── summarize.py              # 汇总各模型指标
└── requirements.txt          # 依赖
```

## 依赖安装

```bash
pip install -r requirements.txt
```

依赖包括：`torch`、`monai`、`segmentation-models-pytorch`、`SimpleITK`、`numpy`、`nnunetv2`。

## 数据格式

数据集根目录需按 `train / val / test` 三个子目录组织，每个病例为一个子目录：

```
cases-root/
├── train/
│   └── <case_id>/
│       ├── *_ct*.nii*      # 或 *_image*.nii*：CT 影像
│       └── *_mask*.nii*    # 或 *_label*.nii* / *_gt*.nii*：二值掩码（前景=骨缺损）
├── val/
└── test/
```

> 说明：影像与掩码通过文件名自动匹配（见 `dataset.find_pair`）。掩码读取后会被二值化为 `> 0`。像素间距（spacing）从影像头信息读取，用于体积（ml）计算。

## 支持的模型

| 模型名 | 类型 | 实现 | 说明 |
| --- | --- | --- | --- |
| `unetpp_2d` | 2D | SMP `UnetPlusPlus` (resnet34) | 单通道 2D 切片 |
| `unetpp_25d` | 2.5D | SMP `UnetPlusPlus` (resnet34) | 3 通道（相邻 3 层切片） |
| `deeplabv3p_2d` | 2D | SMP `DeepLabV3Plus` (resnet34) | 单通道 2D 切片 |
| `segresnet_d1` | 3D | MONAI `SegResNet` | 轻量版（init_filters=16） |
| `attention_unet_3d` | 3D | MONAI `AttentionUNet` | |
| `dynunet` | 3D | MONAI `DynUNet` | |
| `unetr` | 3D | MONAI `UNETR` | |
| `swunetr` | 3D | MONAI `SwinUNETR` | |
| `mednext_small` | 3D | MONAI `MedNeXt` | 小规模版 |
| `nnunet` | 3D | nnU-Net v2 | 由 `nnunet/runner.py` 单独驱动 |

2D / 2.5D 模型由 `models/smp2d.py` 构建；3D 模型由 `models/monai3d.py` 构建；`models/__init__.py` 中的 `build_model` 根据模型名自动选择分支。

## 使用方式

### 1. 训练（MONAI / SMP 模型）

```bash
python train.py \
  --config configs/unetr.json \
  --cases-root /path/to/cases-root \
  --output-dir /path/to/output
```

- 训练完成后会在 `output-dir` 下保存模型权重 `<model>.pt`。
- 训练结束后会自动调用 `evaluate`，对 val / test 集进行评估。
- 训练期间每个 epoch 采样 `max(64, 病例数 × 8)` 个 batch，3D 模型做随机前景/随机中心 patch 采样，2D/2.5D 模型做随机切片采样。

### 2. 单独评估

```bash
python evaluate.py \
  --config configs/unetr.json \
  --checkpoint /path/to/model.pt \
  --cases-root /path/to/cases-root \
  --output-dir /path/to/output
```

- 2D/2.5D 模型按切片逐层推理；3D 模型使用 MONAI 滑窗推理（overlap=0.5）。
- 输出：每个病例的预测 `.npy`（保存在 `predictions/<model>/<split>/`），以及指标表 `eval_<model>.csv`（含 Dice、预测体积、真值体积）。

### 3. 多模型集成（多数投票）

```bash
python ensemble.py \
  --models unetr,swunetr,segresnet_d1 \
  --cases-root /path/to/cases-root \
  --output-dir /path/to/output
```

- 读取各模型已保存的 `.npy` 预测结果做逐体素多数投票，输出 `eval_<模型名拼接>.csv`。

### 4. 汇总

```bash
python summarize.py --output-dir /path/to/output
```

- 汇总 `output-dir` 下所有 `eval_*.csv`，输出 `summary.csv` 与 `summary.json`（按模型 × val/test 给出样本数、平均 Dice、体积 ICC(3,1)）。

### 5. nnU-Net v2 全流程

```bash
python nnunet/runner.py \
  --cases-root /path/to/cases-root \
  --output-dir /path/to/output \
  --seed 0 \
  --epochs 200 \
  --dataset-id 701 \
  --fold 0
```

- 自动完成：数据转换（写入 nnUNet_raw，CT 通道名为 `_0000`）、`nnUNetv2_plan_and_preprocess` 预处理、`nnUNetv2_train` 训练、`nnUNetv2_predict` 预测，并计算 Dice，输出 `eval_val.csv` / `eval_test.csv` 与 `DONE` 标记文件。
- 相关日志写入 `preprocess_stdout.log` / `preprocess_stderr.log` / `train_stdout.log` / `train_stderr.log`。

## 配置文件

每个模型对应 `configs/` 下的一个 JSON 文件，常见字段：

| 字段 | 含义 |
| --- | --- |
| `model` | 模型名（对应上表） |
| `patch` | 3D patch 尺寸（如 `[96, 96, 96]`） |
| `epochs` | 训练轮数 |
| `batch` | 批大小 |
| `lr` | 学习率 |

`configs/nnunet.json` 为 nnU-Net 专用配置：`dataset_id`、`epochs`、`fold`。

## 评估指标

- **Dice 系数**：分割区域的重叠度。
- **体积（ml）**：由预测/真值体素数与像素间距换算。
- **ICC(3,1)**（组内相关系数）：用于评估预测体积与真值体积的一致性。

## 说明

- 默认设备为 `cuda`（不可用时回退 `cpu`），也可通过配置文件中的 `device` 字段指定。
- 本项目为骨缺损分割的研究性代码，可直接复现多模型对比实验流程。
