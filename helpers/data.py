import torch
from torch import Tensor
from torch.utils.data import Dataset


class NextTokenDataset(Dataset):
    """Each item is `block_size` token ids and the single token that follows them."""

    def __init__(self, token_ids: list[int], block_size: int) -> None:
        self.ids = torch.tensor(token_ids)
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.ids) - self.block_size

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        window = self.ids[i : i + self.block_size]
        target = self.ids[i + self.block_size]
        return window, target
