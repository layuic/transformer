"""
完整的 Transformer 训练脚本 - 英语到德语机器翻译
"""
import argparse
import math
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_utils.tokenizer import SimpleTokenizer
from src.data_utils.datasets import Seq2SeqDataset
from src.data_utils.prepare_data import load_custom_data, create_simple_dataset
from src.model.transformer import TransformerEncoderDecoder


def create_padding_mask(seq, pad_id=0):
    """
    创建 padding mask
    Args:
        seq: [batch_size, seq_len] token ids
        pad_id: padding token id
    Returns:
        mask: [batch_size, seq_len] True 表示 padding，False 表示有效 token
    """
    return seq == pad_id


def train_epoch(model, dataloader, optimizer, criterion, device, clip_grad=1.0):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (src_ids, tgt_in_ids, tgt_out_ids) in enumerate(pbar):
        src_ids = src_ids.to(device)
        tgt_in_ids = tgt_in_ids.to(device)
        tgt_out_ids = tgt_out_ids.to(device)
        
        # 创建 padding masks
        src_pad_mask = create_padding_mask(src_ids, pad_id=0)
        tgt_pad_mask = create_padding_mask(tgt_in_ids, pad_id=0)
        
        # 前向传播
        logits = model(src_ids, tgt_in_ids, src_pad_mask, tgt_pad_mask)
        
        # 计算损失（忽略 padding）
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out_ids.view(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / num_batches:.4f}'
        })
    
    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for src_ids, tgt_in_ids, tgt_out_ids in tqdm(dataloader, desc="Evaluating"):
            src_ids = src_ids.to(device)
            tgt_in_ids = tgt_in_ids.to(device)
            tgt_out_ids = tgt_out_ids.to(device)
            
            # 创建 padding masks
            src_pad_mask = create_padding_mask(src_ids, pad_id=0)
            tgt_pad_mask = create_padding_mask(tgt_in_ids, pad_id=0)
            
            # 前向传播
            logits = model(src_ids, tgt_in_ids, src_pad_mask, tgt_pad_mask)
            
            # 计算损失
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out_ids.view(-1))
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def translate_sample(model, src_text, src_tokenizer, tgt_tokenizer, device, max_len=100):
    """翻译单个样本"""
    model.eval()
    
    # 编码源序列
    src_ids = src_tokenizer.encode(src_text, max_len=128).unsqueeze(0).to(device)
    src_pad_mask = create_padding_mask(src_ids, pad_id=0)
    
    # 生成翻译
    bos_id = tgt_tokenizer.stoi.get("<s>", 2)
    eos_id = tgt_tokenizer.stoi.get("</s>", 3)
    
    with torch.no_grad():
        generated_ids = model.generate(
            src_ids, 
            src_pad_mask=src_pad_mask,
            max_len=max_len,
            bos_id=bos_id,
            eos_id=eos_id,
            temperature=1.0,
            top_k=50
        )
    
    # 解码
    generated_ids = generated_ids[0].cpu().tolist()
    # 移除 BOS 和 EOS，转换为文本
    tokens = []
    for idx in generated_ids:
        if idx == eos_id:
            break
        if idx != bos_id and idx != 0:  # 跳过 BOS 和 PAD
            token = tgt_tokenizer.itos.get(idx, "<unk>")
            tokens.append(token)
    
    return "".join(tokens) if src_tokenizer.level == "char" else " ".join(tokens)


def main():
    parser = argparse.ArgumentParser(description="训练 Transformer 模型进行英语到德语翻译")
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, default="data", help="数据目录")
    parser.add_argument("--data_source", type=str, default="simple", 
                       choices=["simple", "custom"],
                       help="数据源：simple (示例数据) 或 custom (自定义文件)")
    parser.add_argument("--level", type=str, default="word", choices=["char", "word"],
                       help="分词级别：char (字符级) 或 word (词级)")
    parser.add_argument("--max_len", type=int, default=128, help="最大序列长度")
    
    # 模型参数
    parser.add_argument("--d_model", type=int, default=512, help="模型维度")
    parser.add_argument("--num_encoder_layers", type=int, default=6, help="Encoder 层数")
    parser.add_argument("--num_decoder_layers", type=int, default=6, help="Decoder 层数")
    parser.add_argument("--num_heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--d_ff", type=int, default=2048, help="前馈网络维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 率")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--warmup_steps", type=int, default=4000, help="Warmup 步数")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="梯度裁剪")
    
    # 其他参数
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="模型保存目录")
    parser.add_argument("--save_every", type=int, default=5, help="每 N 个 epoch 保存一次")
    parser.add_argument("--device", type=str, default=None, help="设备 (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 设置设备
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    print(f"PyTorch 版本: {torch.__version__}")
    
    # 加载数据
    print("\n加载数据...")
    if args.data_source == "simple":
        data = create_simple_dataset(args.data_dir)
    else:
        data = load_custom_data(args.data_dir)
    
    train_src, train_tgt = data["train"]
    valid_src, valid_tgt = data["valid"]
    test_src, test_tgt = data.get("test", ([], []))
    
    print(f"训练集: {len(train_src)} 对")
    print(f"验证集: {len(valid_src)} 对")
    if test_src:
        print(f"测试集: {len(test_src)} 对")
    
    # 创建分词器
    print("\n创建分词器...")
    src_tokenizer = SimpleTokenizer(train_src, level=args.level, min_freq=1)
    tgt_tokenizer = SimpleTokenizer(train_tgt, level=args.level, min_freq=1)
    
    print(f"源语言词表大小: {src_tokenizer.vocab_size()}")
    print(f"目标语言词表大小: {tgt_tokenizer.vocab_size()}")
    
    # 创建数据集
    train_dataset = Seq2SeqDataset(train_src, train_tgt, src_tokenizer, tgt_tokenizer, 
                                   max_len=args.max_len)
    valid_dataset = Seq2SeqDataset(valid_src, valid_tgt, src_tokenizer, tgt_tokenizer, 
                                   max_len=args.max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=0, pin_memory=True if device.type == "cuda" else False)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True if device.type == "cuda" else False)
    
    # 创建模型
    print("\n创建模型...")
    model = TransformerEncoderDecoder(
        src_vocab_size=src_tokenizer.vocab_size(),
        tgt_vocab_size=tgt_tokenizer.vocab_size(),
        d_model=args.d_model,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_len=args.max_len,
        dropout=args.dropout,
        tie_embeddings=True
    ).to(device)
    
    # 计算参数量
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {num_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 保存参数统计
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'model_stats.txt'), 'w', encoding='utf-8') as f:
        f.write("模型参数统计\n")
        f.write("=" * 50 + "\n")
        f.write(f"总参数量: {num_params:,}\n")
        f.write(f"可训练参数量: {trainable_params:,}\n")
        f.write(f"模型维度 (d_model): {args.d_model}\n")
        f.write(f"Encoder 层数: {args.num_encoder_layers}\n")
        f.write(f"Decoder 层数: {args.num_decoder_layers}\n")
        f.write(f"注意力头数: {args.num_heads}\n")
        f.write(f"前馈网络维度 (d_ff): {args.d_ff}\n")
        f.write(f"源语言词表大小: {src_tokenizer.vocab_size()}\n")
        f.write(f"目标语言词表大小: {tgt_tokenizer.vocab_size()}\n")
    print(f"[OK] 参数统计已保存到: {os.path.join(results_dir, 'model_stats.txt')}")
    
    # 优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 padding
    
    # 学习率调度器（可选：warmup）
    def get_lr(step):
        if step < args.warmup_steps:
            return args.lr * step / args.warmup_steps
        return args.lr
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 训练循环
    print("\n开始训练...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, args.clip_grad)
        train_losses.append(train_loss)
        
        # 验证
        val_loss = evaluate(model, valid_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f"\n训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "src_vocab": src_tokenizer.stoi,
                "tgt_vocab": tgt_tokenizer.stoi,
                "src_itos": src_tokenizer.itos,
                "tgt_itos": tgt_tokenizer.itos,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "args": vars(args),
                "best_val_loss": best_val_loss
            }
            torch.save(checkpoint, os.path.join(args.save_dir, "best_model.pt"))
            print(f"[OK] 保存最佳模型 (验证损失: {val_loss:.4f})")
        
        # 定期保存
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "src_vocab": src_tokenizer.stoi,
                "tgt_vocab": tgt_tokenizer.stoi,
                "src_itos": src_tokenizer.itos,
                "tgt_itos": tgt_tokenizer.itos,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "args": vars(args),
                "val_loss": val_loss
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"[OK] 保存检查点: {checkpoint_path}")
        
        # 翻译示例
        if (epoch + 1) % 5 == 0:
            print("\n翻译示例:")
            sample_src = train_src[0] if train_src else "Hello"
            translation = translate_sample(model, sample_src, src_tokenizer, tgt_tokenizer, device)
            print(f"源: {sample_src}")
            print(f"翻译: {translation}")
            if train_tgt:
                print(f"参考: {train_tgt[0]}")
    
    print("\n训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"模型已保存到: {os.path.join(args.save_dir, 'best_model.pt')}")
    
    # 保存训练曲线到 results 目录
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
        # 配置中文字体（在常见中文字体中就地选择可用的一个）
        try:
            from matplotlib import font_manager
            _available_fonts = {f.name for f in font_manager.fontManager.ttflist}
            for _font in ["Microsoft YaHei", "SimHei", "Source Han Sans CN", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Sarasa UI SC"]:
                if _font in _available_fonts:
                    matplotlib.rcParams['font.family'] = 'sans-serif'
                    matplotlib.rcParams['font.sans-serif'] = [_font]
                    matplotlib.rcParams['axes.unicode_minus'] = False
                    break
        except Exception:
            pass
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='训练损失', marker='o', markersize=3)
        plt.plot(val_losses, label='验证损失', marker='s', markersize=3)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('训练曲线')
        plt.legend()
        plt.grid(True)
        curve_path = os.path.join(results_dir, 'training_curves.png')
        plt.savefig(curve_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"[OK] 训练曲线已保存到: {curve_path}")
        
        # 保存损失数据为CSV
        import csv
        csv_path = os.path.join(results_dir, 'training_losses.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Loss', 'Val Loss'])
            for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
                writer.writerow([i+1, f'{train_loss:.6f}', f'{val_loss:.6f}'])
        print(f"[OK] 损失数据已保存到: {csv_path}")
    except ImportError:
        print("警告: matplotlib未安装，跳过训练曲线绘制")


if __name__ == "__main__":
    main()


