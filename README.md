# Transformer (EN→DE) - PyTorch

一个从零实现的 Encoder–Decoder Transformer，用于英语→德语机器翻译。代码精简、可直接复现训练与消融实验。

## 硬件要求

- **CPU**: 支持（训练速度较慢，建议用于快速验证）
- **GPU**: 推荐（NVIDIA GPU，支持 CUDA，显存 ≥ 4GB）
- **内存**: ≥ 8GB RAM
- **磁盘空间**: ≥ 2GB（用于模型检查点和结果）

**测试环境**:
- CPU: Intel/AMD x86_64
- GPU: NVIDIA GPU with CUDA 11.0+ (可选)
- Python: 3.8+
- PyTorch: 1.9.0+

## 1) 安装

```bash
pip install -r requirements.txt
```

## 2) 数据准备

快速生成一个小样例数据集（用于跑通流程）：
```bash
python src/data_utils/prepare_data.py --data_dir data --source simple
```
也可在 `data/` 放入自定义文件：`train.en, train.de, valid.en, valid.de, test.en, test.de`。

## 3) 训练（Baseline）

### 方法一：直接使用 Python 命令（推荐，含完整参数）

**复现 Baseline 实验的 exact 命令行（含随机种子）**：

```bash
python train_transformer.py \
  --data_dir data \
  --data_source simple \
  --level word \
  --d_model 256 \
  --num_encoder_layers 3 \
  --num_decoder_layers 3 \
  --num_heads 4 \
  --d_ff 1024 \
  --batch_size 64 \
  --epochs 10 \
  --lr 1e-4 \
  --seed 42 \
  --save_dir checkpoints
```

**关键参数说明**：
- `--seed 42`: 固定随机种子，确保结果可复现
- `--d_model 256`: 模型嵌入维度
- `--num_encoder_layers 3 --num_decoder_layers 3`: 编码器和解码器各 3 层
- `--num_heads 4`: 多头注意力头数
- `--d_ff 1024`: 前馈网络维度
- `--batch_size 64`: 批次大小
- `--epochs 10`: 训练轮数

结果会自动保存到：
- `results/training_curves.png`, `results/training_losses.csv`, `results/model_stats.txt`
- `checkpoints/best_model.pt`

### 方法二：使用脚本

```bash
bash scripts/run.sh --small  # 快速测试（小模型，10 epochs）
bash scripts/run.sh           # 标准配置（大模型，50 epochs）
```

脚本默认随机种子为 `42`，可通过 `--seed` 参数修改。

## 4) 消融实验

运行所有消融实验（9 个配置）：
```bash
python run_ablation_experiments.py
```

或仅运行未完成的实验（自动检测已有检查点）：
```bash
python run_remaining_ablation.py
```

**消融实验配置**（所有实验使用 `--seed 42`）：
- `baseline`: 默认配置（3层，4头，d_ff=1024，dropout=0.1）
- `heads_1`: 1个注意力头
- `heads_8`: 8个注意力头
- `dff_512`: 前馈维度 512
- `dff_2048`: 前馈维度 2048
- `dropout_0.0`: Dropout=0.0
- `dropout_0.3`: Dropout=0.3
- `layers_2`: 2层编码器/解码器
- `layers_4`: 4层编码器/解码器

对比图与汇总会保存到 `results/ablation/`：
- `ablation_summary.csv`, `ablation_results.json`
- `ablation_comparison.png`, `dff_comparison.png`
- 每个实验的 `training_curves.png`, `training_losses.csv`

## 5) 推理（翻译）

```bash
python translate.py --checkpoint checkpoints/best_model.pt --text "Hello, how are you?"
```

或交互式/批量模式见脚本参数。

## 6) 目录结构

```
.
├── src/                          # 源代码目录
│   ├── model/                    # 模型实现
│   │   ├── attention.py          # 多头注意力机制
│   │   ├── blocks.py             # Encoder/Decoder 块
│   │   └── transformer.py        # Transformer 主模型
│   └── data_utils/               # 数据处理工具
│       ├── tokenizer.py          # 分词器
│       ├── datasets.py           # 数据集类
│       └── prepare_data.py       # 数据准备脚本
├── scripts/
│   └── run.sh                    # 一键复现实验脚本
├── train_transformer.py          # 训练入口
├── translate.py                  # 推理入口
├── run_ablation_experiments.py   # 消融实验脚本
├── run_remaining_ablation.py     # 增量消融实验脚本
├── visualize_ablation_results.py # 结果可视化脚本
├── requirements.txt              # Python 依赖
├── README.md                     # 本文件
├── data/                         # 数据目录（需自行准备）
├── results/                      # 训练结果（训练后生成）
│   ├── training_curves.png       # 训练曲线图
│   ├── training_losses.csv       # 训练损失数据
│   ├── model_stats.txt           # 模型统计信息
│   └── ablation/                 # 消融实验结果
└── checkpoints/                  # 模型检查点（训练后生成）
```

## 7) 复现说明

### 完整复现流程

1. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

2. **准备数据**：
   ```bash
   python src/data_utils/prepare_data.py --data_dir data --source simple
   ```

3. **训练 Baseline**（使用 exact 命令行，随机种子 42）：
   ```bash
   python train_transformer.py \
     --data_dir data \
     --data_source simple \
     --level word \
     --d_model 256 \
     --num_encoder_layers 3 \
     --num_decoder_layers 3 \
     --num_heads 4 \
     --d_ff 1024 \
     --batch_size 64 \
     --epochs 10 \
     --lr 1e-4 \
     --seed 42 \
     --save_dir checkpoints
   ```

4. **运行消融实验**：
   ```bash
   python run_ablation_experiments.py
   ```

5. **查看结果**：
   - 训练曲线：`results/training_curves.png`
   - 消融对比：`results/ablation/ablation_comparison.png`
   - 结果表格：`results/ablation/ablation_summary.csv`

### 可复现性保证

- **随机种子**: 所有实验默认使用 `--seed 42`，确保结果可复现
- **环境变量**: 脚本中设置 `PYTHONHASHSEED=42` 进一步保证可复现性
- **固定超参数**: 所有超参数在命令行中明确指定，避免配置文件差异

### 预期运行时间（参考）

- **Baseline 训练**（10 epochs）:
  - CPU: ~2-4 小时
  - GPU (4GB): ~15-30 分钟
- **完整消融实验**（9 个配置，每个 10 epochs）:
  - CPU: ~18-36 小时
  - GPU (4GB): ~2-5 小时

## 8) 说明

- 默认随机种子：`--seed 42`（可复现）
- CPU 可跑通，小显存/无 GPU 建议用 `--small` 快速验证
- 所有实验结果保存在 `results/` 目录，可直接用于报告生成



