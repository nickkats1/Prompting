import torch
import torch.nn as nn

from transformer.gelu import GELU
from transformer.attention import SelfAttention
from transformer.config import CONFIG
from transformer.feed_forward import FeedForward
from transformer.layer_norm import LayerNorm



class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.attn = SelfAttention(
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            context_length=cfg['context_length'],
            num_heads=cfg['n_heads'],
            dropout=cfg['drop_rate'],
            qkv_bias=cfg['qkv_bias']
        )
        
        
        # first feed forward layer
        
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        
        self.drop = nn.Dropout(cfg['drop_rate'])
        
    def forward(self, x):
        short = x
        
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop(x)
        x = x + short
        
        
        short = x
        
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x)
        x = x + short
        
        return x
    
            