import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, is_causal=False):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.is_causal = is_causal
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_kv, attn_mask=None, key_padding_mask=None):
        B, Tq, C = x_q.size()
        Tk = x_kv.size(1)
        q = self.Wq(x_q).view(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.Wk(x_kv).view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.Wv(x_kv).view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.is_causal:
            causal = torch.triu(torch.ones(Tq, Tk, device=attn_scores.device), diagonal=1).bool()
            attn_scores = attn_scores.masked_fill(causal, float("-inf"))
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float("-inf"))
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].expand(-1, self.num_heads, Tq, -1)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, Tq, C)
        return self.out(out)




