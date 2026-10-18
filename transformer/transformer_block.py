import torch
import torch.nn as nn

from transformer.attention import MultiHeadAttention
from transformer.mlp import MLP


class Block(nn.Module):
    def __init__(self, n_head: int, n_embed: int, context_length: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = MultiHeadAttention(n_head, n_embed, context_length)
        self.ln2 = nn.LayerNorm(n_embed)
        self.mlp = MLP(n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    def forward_embedding(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x + self.attn(self.ln1(x))
        hidden = self.mlp.forward_embedding(self.ln2(residual))
        return hidden, residual
