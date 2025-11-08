#!/bin/bash
# Transformer 训练和评估脚本
# 用于复现实验结果

set -e  # 遇到错误立即退出

# 设置随机种子以确保可复现性
export PYTHONHASHSEED=42

# 默认参数
DATA_DIR="data"
DATA_SOURCE="simple"
LEVEL="word"
MAX_LEN=128
D_MODEL=512
NUM_ENCODER_LAYERS=6
NUM_DECODER_LAYERS=6
NUM_HEADS=8
D_FF=2048
DROPOUT=0.1
BATCH_SIZE=32
EPOCHS=50
LR=1e-4
WARMUP_STEPS=4000
CLIP_GRAD=1.0
SEED=42
SAVE_DIR="checkpoints"
RESULTS_DIR="results"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --data_source)
            DATA_SOURCE="$2"
            shift 2
            ;;
        --level)
            LEVEL="$2"
            shift 2
            ;;
        --d_model)
            D_MODEL="$2"
            shift 2
            ;;
        --num_encoder_layers)
            NUM_ENCODER_LAYERS="$2"
            shift 2
            ;;
        --num_decoder_layers)
            NUM_DECODER_LAYERS="$2"
            shift 2
            ;;
        --num_heads)
            NUM_HEADS="$2"
            shift 2
            ;;
        --d_ff)
            D_FF="$2"
            shift 2
            ;;
        --dropout)
            DROPOUT="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --save_dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --results_dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --small|--quick)
            # 快速测试模式：小模型
            D_MODEL=256
            NUM_ENCODER_LAYERS=3
            NUM_DECODER_LAYERS=3
            NUM_HEADS=4
            D_FF=1024
            BATCH_SIZE=64
            EPOCHS=10
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --data_dir DIR          数据目录 (默认: $DATA_DIR)"
            echo "  --data_source SOURCE    数据源 simple/custom (默认: $DATA_SOURCE)"
            echo "  --level LEVEL           分词级别 char/word (默认: $LEVEL)"
            echo "  --d_model DIM           模型维度 (默认: $D_MODEL)"
            echo "  --num_encoder_layers N  Encoder层数 (默认: $NUM_ENCODER_LAYERS)"
            echo "  --num_decoder_layers N  Decoder层数 (默认: $NUM_DECODER_LAYERS)"
            echo "  --num_heads N           注意力头数 (默认: $NUM_HEADS)"
            echo "  --d_ff DIM               前馈网络维度 (默认: $D_FF)"
            echo "  --dropout RATE          Dropout率 (默认: $DROPOUT)"
            echo "  --batch_size N          批次大小 (默认: $BATCH_SIZE)"
            echo "  --epochs N              训练轮数 (默认: $EPOCHS)"
            echo "  --lr RATE               学习率 (默认: $LR)"
            echo "  --seed N                随机种子 (默认: $SEED)"
            echo "  --save_dir DIR          模型保存目录 (默认: $SAVE_DIR)"
            echo "  --results_dir DIR       结果保存目录 (默认: $RESULTS_DIR)"
            echo "  --small|--quick          快速测试模式（小模型）"
            exit 1
            ;;
    esac
done

# 创建必要的目录
mkdir -p "$SAVE_DIR"
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "Transformer 训练脚本"
echo "=========================================="
echo "数据目录: $DATA_DIR"
echo "数据源: $DATA_SOURCE"
echo "分词级别: $LEVEL"
echo "模型维度: $D_MODEL"
echo "Encoder层数: $NUM_ENCODER_LAYERS"
echo "Decoder层数: $NUM_DECODER_LAYERS"
echo "注意力头数: $NUM_HEADS"
echo "前馈网络维度: $D_FF"
echo "Dropout率: $DROPOUT"
echo "批次大小: $BATCH_SIZE"
echo "训练轮数: $EPOCHS"
echo "学习率: $LR"
echo "随机种子: $SEED"
echo "模型保存目录: $SAVE_DIR"
echo "结果保存目录: $RESULTS_DIR"
echo "=========================================="

# 训练模型
echo "开始训练..."
python train_transformer.py \
    --data_dir "$DATA_DIR" \
    --data_source "$DATA_SOURCE" \
    --level "$LEVEL" \
    --max_len "$MAX_LEN" \
    --d_model "$D_MODEL" \
    --num_encoder_layers "$NUM_ENCODER_LAYERS" \
    --num_decoder_layers "$NUM_DECODER_LAYERS" \
    --num_heads "$NUM_HEADS" \
    --d_ff "$D_FF" \
    --dropout "$DROPOUT" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --warmup_steps "$WARMUP_STEPS" \
    --clip_grad "$CLIP_GRAD" \
    --save_dir "$SAVE_DIR" \
    --seed "$SEED"

# 绘制训练曲线
echo "绘制训练曲线..."
python -c "
import torch
import matplotlib.pyplot as plt
import os

checkpoint_path = os.path.join('$SAVE_DIR', 'best_model.pt')
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失', marker='o', markersize=3)
    plt.plot(val_losses, label='验证损失', marker='s', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join('$RESULTS_DIR', 'training_curves.png'), dpi=200, bbox_inches='tight')
    print(f'训练曲线已保存到: {os.path.join(\"$RESULTS_DIR\", \"training_curves.png\")}')
else:
    print('警告: 未找到模型检查点')
"

echo "训练完成！"
echo "模型保存在: $SAVE_DIR"
echo "结果保存在: $RESULTS_DIR"







