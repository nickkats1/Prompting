import torch
import torch.nn as nn
import math


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_head: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        assert d_model % num_head == 0, 'd_model must be divisible by num_head'

        self.d_k = d_model // num_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_score.masked_fill_(mask == 0, -1e9)

        attention_score = attention_score.softmax(dim=-1)

        if dropout is not None:
            attention_score = dropout(attention_score)

        return (attention_score @ value), attention_score

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.num_head, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_head, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_head, self.d_k).transpose(1, 2)

        x, self.attention_score = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )

        x = x.transpose(1, 2).contiguous().view(
            x.shape[0], -1, self.num_head * self.d_k
        )

        return self.w_o(x)
        
        
        