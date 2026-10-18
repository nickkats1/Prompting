from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from transformer.sampling import filter_logits
from transformer.transformer_block import Block


class Transformer(nn.Module):

    def __init__(
        self,
        n_head: int,
        n_embed: int,
        context_length: int,
        vocab_size: int,
        N_BLOCKS: int,
    ) -> None:
        super().__init__()
        assert (
            context_length > 0 and vocab_size > 0 and N_BLOCKS > 0
        ), "context_length, vocab_size, and N_BLOCKS must be positive"

        self.context_length = context_length
        self.N_BLOCKS = N_BLOCKS
        self.gradient_checkpointing = False
        self.token_embed = nn.Embedding(vocab_size, n_embed)
        self.position_embed = nn.Embedding(context_length, n_embed)
        self.attn_blocks = nn.ModuleList(
            [Block(n_head, n_embed, context_length) for _ in range(N_BLOCKS)]
        )
        self.layer_norm = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.register_buffer("pos_idxs", torch.arange(context_length))

    def _pre_attn_pass(self, idx: torch.Tensor) -> torch.Tensor:
        assert idx.ndim == 2, "idx must have shape (batch, sequence)"
        _, T = idx.shape
        assert (
            0 < T <= self.context_length
        ), "sequence length must be between 1 and context_length"

        tok_embedding = self.token_embed(idx)
        pos_embedding = self.position_embed(self.pos_idxs[:T])
        return tok_embedding + pos_embedding

    def forward_hidden(self, idx: torch.Tensor) -> torch.Tensor:
        x = self._pre_attn_pass(idx)
        for block in self.attn_blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.layer_norm(x)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.forward_hidden(idx)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            assert (
                targets.shape == idx.shape
            ), "targets must match idx; supply next-token targets"
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1).long()
            )
        return logits, loss

    def forward_embedding(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._pre_attn_pass(idx)
        for block in self.attn_blocks[:-1]:
            x = block(x)
        last_block = cast(Block, self.attn_blocks[-1])
        return last_block.forward_embedding(x)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        context_window: int | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
    ) -> torch.Tensor:
        assert (
            idx.ndim == 2 and idx.size(1) > 0
        ), "idx must have shape (batch, nonempty sequence)"
        assert max_new_tokens >= 0, "max_new_tokens must be nonnegative"
        assert (
            context_window is None or context_window > 0
        ), "context_window must be positive"

        window = min(context_window or self.context_length, self.context_length)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -window:]
            logits, _ = self(idx_cond)
            logits = filter_logits(logits[:, -1, :], temperature, top_k, top_p, min_p)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
