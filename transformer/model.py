
import torch
from torch import nn

from transformer.positional_encoding import PositionalEncoding
from transformer.encoder import TransformerEncoderLayer

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, output_dim, max_len: int):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)


        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for layer in self.layers:
            x = layer(x)

        return self.output_proj(x[:, 0]) 
