import torch.nn as nn
from .attention import MultiHeadAttention

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop2 = nn.Dropout(dropout)
    def forward(self, x):
        x = self.fc2(self.drop1(self.act(self.fc1(x))))
        return self.drop2(x)

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout, is_causal=False)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.ln1(x), self.ln1(x), key_padding_mask=key_padding_mask)
        x = x + self.ffn(self.ln2(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, is_causal=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, is_causal=False)
        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
    def forward(self, x, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = x + self.self_attn(self.ln1(x), self.ln1(x), key_padding_mask=tgt_key_padding_mask)
        x = x + self.cross_attn(self.ln2(x), memory, key_padding_mask=memory_key_padding_mask)
        x = x + self.ffn(self.ln3(x))
        return x




