import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    tril: torch.Tensor

    def __init__(self, head_size: int, n_embed: int, context_length: int) -> None:
        super().__init__()
        assert (
            head_size > 0 and n_embed > 0 and context_length > 0
        ), "head_size, n_embed, and context_length must be positive"

        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(context_length, context_length)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, _ = x.shape
        assert (
            0 < T <= self.tril.size(0)
        ), "sequence length must be between 1 and context_length"

        head_size = self.key.out_features
        k = self.key(x)
        q = self.query(x)
        scale_factor = 1 / math.sqrt(head_size)
        attn_weights = q @ k.transpose(-2, -1) * scale_factor
        attn_weights = attn_weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        v = self.value(x)
        return attn_weights @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, n_embed: int, context_length: int) -> None:
        super().__init__()
        assert n_head > 0 and n_embed > 0, "n_head and n_embed must be positive"
        assert n_embed % n_head == 0, "n_embed must be divisible by n_head"

        head_size = n_embed // n_head
        self.heads = nn.ModuleList(
            [Head(head_size, n_embed, context_length) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(x)
