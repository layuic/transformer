import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import EncoderBlock, DecoderBlock

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)

class TinyDecoderOnlyLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_layers=6, num_heads=4, d_ff=1024, max_len=512, dropout=0.1, tie_embeddings=True):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.tok.weight
    def forward(self, x_ids, pad_mask=None):
        x = self.pos(self.tok(x_ids))
        memory = x  # cross_attn 不使用时也可传 x，本实现复用 DecoderBlock
        for blk in self.blocks:
            x = blk(x, memory, tgt_key_padding_mask=pad_mask, memory_key_padding_mask=pad_mask)
        x = self.ln(x)
        return self.lm_head(x)

class TransformerEncoderDecoder(nn.Module):
    """
    完整的 Encoder-Decoder Transformer 模型，用于机器翻译任务
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_encoder_layers=6,
                 num_decoder_layers=6, num_heads=8, d_ff=2048, max_len=512, dropout=0.1,
                 tie_embeddings=True):
        super().__init__()
        self.d_model = d_model
        
        # 词嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        
        # Encoder 和 Decoder 层
        self.encoder = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Layer Normalization
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        
        # 输出层
        self.lm_head = nn.Linear(d_model, tgt_vocab_size, bias=False)
        
        # 权重共享：输出层和嵌入层共享权重
        if tie_embeddings:
            self.lm_head.weight = self.tgt_embedding.weight
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self, src_ids, src_pad_mask=None):
        # 词嵌入 + 位置编码
        x = self.src_embedding(src_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        # 通过 Encoder 层
        for encoder_layer in self.encoder:
            x = encoder_layer(x, key_padding_mask=src_pad_mask)
        return self.encoder_norm(x)
    
    def decode(self, tgt_ids, memory, tgt_pad_mask=None, memory_pad_mask=None):
        # 词嵌入 + 位置编码
        y = self.tgt_embedding(tgt_ids) * math.sqrt(self.d_model)
        y = self.pos_encoding(y)
        # 通过 Decoder 层
        for decoder_layer in self.decoder:
            y = decoder_layer(y, memory,
                            tgt_key_padding_mask=tgt_pad_mask,
                            memory_key_padding_mask=memory_pad_mask)
        return self.decoder_norm(y)
    
    def forward(self, src_ids, tgt_ids, src_pad_mask=None, tgt_pad_mask=None):
        memory = self.encode(src_ids, src_pad_mask)
        decoder_output = self.decode(tgt_ids, memory, tgt_pad_mask, src_pad_mask)
        logits = self.lm_head(decoder_output)
        return logits
    
    def generate(self, src_ids, src_pad_mask=None, max_len=100, bos_id=2, eos_id=3,
                 temperature=1.0, top_k=50):
        self.eval()
        batch_size = src_ids.size(0)
        device = src_ids.device
        memory = self.encode(src_ids, src_pad_mask)
        tgt_ids = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        with torch.no_grad():
            for _ in range(max_len):
                decoder_output = self.decode(tgt_ids, memory,
                                            tgt_pad_mask=None,
                                            memory_pad_mask=src_pad_mask)
                logits = self.lm_head(decoder_output[:, -1:, :])
                logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                    logits = torch.where(logits < v[..., [-1]],
                                       torch.full_like(logits, float('-inf')),
                                       logits)
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs.squeeze(1), num_samples=1)
                tgt_ids = torch.cat([tgt_ids, next_token], dim=1)
                if (next_token == eos_id).all():
                    break
        return tgt_ids

# 为了保持向后兼容，保留原类名作为别名
TinyTransformerSeq2Seq = TransformerEncoderDecoder




