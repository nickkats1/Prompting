from __future__ import annotations

import math

import torch
from torch import Tensor


def filter_logits(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
) -> Tensor:
    assert (
        math.isfinite(temperature) and temperature >= 0
    ), "temperature must be finite and nonnegative"
    assert top_k is None or (
        isinstance(top_k, int) and top_k >= 0
    ), "top_k must be a nonnegative integer"
    assert top_p is None or 0 < top_p <= 1, "top_p must be greater than 0 and at most 1"
    assert min_p is None or 0 <= min_p <= 1, "min_p must be between 0 and 1"

    if temperature == 0:
        greedy_logits = torch.full_like(logits, float("-inf"))
        indices = logits.argmax(dim=-1, keepdim=True)
        return greedy_logits.scatter(-1, indices, 0.0)
    if temperature != 1.0:
        logits = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cumprobs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        remove = remove.scatter(-1, sorted_idx, remove)
        logits = logits.masked_fill(remove, float("-inf"))

    if min_p is not None and 0.0 < min_p <= 1.0:
        probs = logits.softmax(dim=-1)
        threshold = min_p * probs.max(dim=-1, keepdim=True).values
        logits = logits.masked_fill(probs < threshold, float("-inf"))

    return logits
