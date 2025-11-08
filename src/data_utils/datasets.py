import torch
from torch.utils.data import Dataset

class LMDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=128):
        self.tok = tokenizer
        self.seq_len = seq_len
        self.ids = []
        for t in texts:
            x = tokenizer.encode(t, max_len=seq_len + 1)
            self.ids.append(x)
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx):
        ids = self.ids[idx]
        return ids[:-1], ids[1:]

class Seq2SeqDataset(Dataset):
    """
    序列到序列数据集，用于机器翻译任务
    返回: (src_ids, tgt_in_ids, tgt_out_ids)
    - src_ids: 源序列 token ids
    - tgt_in_ids: 目标序列输入（用于 teacher forcing），以 BOS 开头，去掉 EOS
    - tgt_out_ids: 目标序列输出（用于计算损失），以 EOS 结尾，去掉 BOS
    """
    def __init__(self, src_texts, tgt_texts, src_tok, tgt_tok, max_len=128):
        self.src_tok, self.tgt_tok = src_tok, tgt_tok
        self.max_len = max_len
        self.bos_id = tgt_tok.stoi.get("<s>", 2)
        self.eos_id = tgt_tok.stoi.get("</s>", 3)
        self.pad_id = tgt_tok.stoi.get("<pad>", 0)
        self.src = []
        self.tgt_in_list = []
        self.tgt_out_list = []
        for src_text, tgt_text in zip(src_texts, tgt_texts):
            src_ids = src_tok.encode(src_text, max_len)
            tgt_tokens = tgt_text.split() if tgt_tok.level == "word" else list(tgt_text)
            tgt_token_ids = [tgt_tok.stoi.get(t, tgt_tok.stoi.get("<unk>", 1)) for t in tgt_tokens]
            tgt_in = [self.bos_id] + tgt_token_ids
            tgt_out = tgt_token_ids + [self.eos_id]
            tgt_in = tgt_in[:max_len]
            tgt_out = tgt_out[:max_len]
            tgt_in = tgt_in + [self.pad_id] * (max_len - len(tgt_in))
            tgt_out = tgt_out + [self.pad_id] * (max_len - len(tgt_out))
            self.src.append(src_ids)
            self.tgt_in_list.append(torch.tensor(tgt_in, dtype=torch.long))
            self.tgt_out_list.append(torch.tensor(tgt_out, dtype=torch.long))
    def __len__(self):
        return len(self.src)
    def __getitem__(self, i):
        return self.src[i], self.tgt_in_list[i], self.tgt_out_list[i]




