import torch

from helpers.data import NextTokenDataset


def test_length_is_ids_minus_block(dataset: NextTokenDataset, token_ids, block_size):
    assert len(dataset) == len(token_ids) - block_size


def test_item_is_window_and_next_token(dataset: NextTokenDataset, block_size):
    x, y = dataset[3]
    assert torch.equal(x, torch.tensor([3, 4, 5, 6]))
    assert y.item() == 3 + block_size


def test_item_shapes(dataset: NextTokenDataset, block_size):
    x, y = dataset[0]
    assert x.shape == (block_size,)
    assert y.shape == ()
