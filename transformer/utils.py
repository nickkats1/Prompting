import torch
from torch.utils.data import Dataset


class CharTokenizer:
    def __init__(self, text: str) -> None:
        self.chars = sorted(set(text))
        self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def __len__(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids) -> str:
        return "".join(self.id_to_char[int(i)] for i in ids)


class NextTokenDataset(Dataset):
    def __init__(self, ids: list[int], block_size: int) -> None:
        if not isinstance(block_size, int) or block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        if len(ids) <= block_size:
            raise ValueError("ids must contain at least block_size + 1 tokens")
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.ids) - self.block_size

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError("dataset index out of range")
        chunk = self.ids[i : i + self.block_size + 1]
        return chunk[:-1], chunk[1:]
