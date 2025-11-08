"""
使用训练好的 Transformer 模型进行英语到德语翻译
"""
import argparse
import torch
from src.data_utils.tokenizer import SimpleTokenizer
from src.model.transformer import TransformerEncoderDecoder


def load_model(checkpoint_path, device):
    """加载训练好的模型"""
    print(f"加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    args = checkpoint.get("args", {})
    
    # 创建分词器
    src_stoi = checkpoint["src_vocab"]
    tgt_stoi = checkpoint["tgt_vocab"]
    src_itos = checkpoint.get("src_itos", {i: s for s, i in src_stoi.items()})
    tgt_itos = checkpoint.get("tgt_itos", {i: s for s, i in tgt_stoi.items()})
    
    # 创建 tokenizer 对象（简化版，仅用于推理）
    class LoadedTokenizer:
        def __init__(self, stoi, itos, level="word"):
            self.stoi = stoi
            self.itos = itos
            self.level = level
            self.pad = "<pad>"
            self.unk = "<unk>"
            self.bos = "<s>"
            self.eos = "</s>"
        
        def encode(self, text, max_len):
            toks = text.split() if self.level == "word" else list(text)
            ids = [self.stoi.get(t, self.stoi.get(self.unk, 0)) for t in toks][:max_len]
            ids = ids + [self.stoi[self.pad]] * (max_len - len(ids))
            return torch.tensor(ids, dtype=torch.long)
    
    level = args.get("level", "word")
    src_tokenizer = LoadedTokenizer(src_stoi, src_itos, level)
    tgt_tokenizer = LoadedTokenizer(tgt_stoi, tgt_itos, level)
    
    # 创建模型
    model = TransformerEncoderDecoder(
        src_vocab_size=len(src_stoi),
        tgt_vocab_size=len(tgt_stoi),
        d_model=args.get("d_model", 512),
        num_encoder_layers=args.get("num_encoder_layers", 6),
        num_decoder_layers=args.get("num_decoder_layers", 6),
        num_heads=args.get("num_heads", 8),
        d_ff=args.get("d_ff", 2048),
        max_len=args.get("max_len", 128),
        dropout=0.0,  # 推理时不需要 dropout
        tie_embeddings=True
    ).to(device)
    
    # 加载权重
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"模型加载完成 (Epoch {checkpoint.get('epoch', '?')})")
    print(f"最佳验证损失: {checkpoint.get('best_val_loss', checkpoint.get('val_loss', '?')):.4f}")
    
    return model, src_tokenizer, tgt_tokenizer


def create_padding_mask(seq, pad_id=0):
    """创建 padding mask"""
    return seq == pad_id


def translate(model, src_text, src_tokenizer, tgt_tokenizer, device, max_len=100, 
              temperature=1.0, top_k=50):
    """翻译单个句子"""
    model.eval()
    
    # 编码源序列
    src_ids = src_tokenizer.encode(src_text, max_len=128).unsqueeze(0).to(device)
    src_pad_mask = create_padding_mask(src_ids, pad_id=src_tokenizer.stoi.get("<pad>", 0))
    
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
            temperature=temperature,
            top_k=top_k
        )
    
    # 解码
    generated_ids = generated_ids[0].cpu().tolist()
    tokens = []
    for idx in generated_ids:
        if idx == eos_id:
            break
        if idx != bos_id and idx != src_tokenizer.stoi.get("<pad>", 0):
            token = tgt_tokenizer.itos.get(idx, "<unk>")
            tokens.append(token)
    
    return " ".join(tokens) if src_tokenizer.level == "word" else "".join(tokens)


def main():
    parser = argparse.ArgumentParser(description="使用训练好的 Transformer 模型进行翻译")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--input", type=str, default=None, help="输入文件路径（每行一个句子）")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    parser.add_argument("--text", type=str, default=None, help="直接翻译的文本")
    parser.add_argument("--max_len", type=int, default=100, help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k 采样")
    parser.add_argument("--device", type=str, default=None, help="设备 (cuda/cpu)")
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    
    # 加载模型
    model, src_tokenizer, tgt_tokenizer = load_model(args.checkpoint, device)
    
    # 翻译
    if args.text:
        # 直接翻译输入的文本
        print(f"\n源文本: {args.text}")
        translation = translate(model, args.text, src_tokenizer, tgt_tokenizer, device,
                              args.max_len, args.temperature, args.top_k)
        print(f"翻译结果: {translation}")
    
    elif args.input:
        # 从文件读取并翻译
        with open(args.input, "r", encoding="utf-8") as f:
            src_texts = [line.strip() for line in f if line.strip()]
        
        translations = []
        for src_text in src_texts:
            translation = translate(model, src_text, src_tokenizer, tgt_tokenizer, device,
                                  args.max_len, args.temperature, args.top_k)
            translations.append(translation)
            print(f"源: {src_text}")
            print(f"翻译: {translation}\n")
        
        # 保存结果
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                for trans in translations:
                    f.write(trans + "\n")
            print(f"翻译结果已保存到: {args.output}")
    
    else:
        # 交互式翻译
        print("\n交互式翻译模式（输入 'quit' 或 'exit' 退出）")
        while True:
            try:
                src_text = input("\n请输入英语句子: ").strip()
                if src_text.lower() in ["quit", "exit", "q"]:
                    break
                if not src_text:
                    continue
                
                translation = translate(model, src_text, src_tokenizer, tgt_tokenizer, device,
                                      args.max_len, args.temperature, args.top_k)
                print(f"德语翻译: {translation}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"错误: {e}")


if __name__ == "__main__":
    main()




