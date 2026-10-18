import torch.nn as nn
from torch import Tensor


class MLP(nn.Module):
    def __init__(self, n_embed: int) -> None:
        super().__init__()
        assert n_embed > 0, "n_embed must be positive"

        self.hidden = nn.Linear(n_embed, 4 * n_embed)
        self.relu = nn.ReLU()
        self.proj = nn.Linear(4 * n_embed, n_embed)

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_embedding(x)
        return self.project_embedding(x)

    def forward_embedding(self, x: Tensor) -> Tensor:
        return self.relu(self.hidden(x))

    def project_embedding(self, x: Tensor) -> Tensor:
        return self.proj(x)
