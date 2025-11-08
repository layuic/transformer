import math, argparse, torch, torch.nn as nn
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data_utils.tokenizer import SimpleTokenizer
from src.data_utils.datasets import LMDataset
from src.model.transformer import TinyDecoderOnlyLM

def main(args):
    texts = open(args.corpus, "r", encoding="utf-8").read().splitlines()
    tok = SimpleTokenizer(texts, level=args.level)
    ds = LMDataset(texts, tok, seq_len=args.seq_len)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, drop_last=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyDecoderOnlyLM(tok.vocab_size(), d_model=args.d_model, num_layers=args.layers, num_heads=args.heads, d_ff=args.d_ff, max_len=args.seq_len, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(dl))
    loss_fn = nn.CrossEntropyLoss()
    logs = []
    for ep in range(args.epochs):
        pbar = tqdm(dl, desc=f"ep{ep}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            ppl = math.exp(min(20, loss.item()))
            pbar.set_postfix(loss=f"{loss.item():.3f}", ppl=f"{ppl:.2f}")
            if len(logs) == 0 or len(logs) % 50 == 0:
                logs.append((len(logs), float(loss.item()), float(ppl)))
    torch.save({"model": model.state_dict(), "vocab": tok.stoi, "logs": logs}, args.out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, required=True)
    ap.add_argument("--level", type=str, default="char")
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--out", type=str, default="lm_ckpt.pt")
    args = ap.parse_args()
    main(args)