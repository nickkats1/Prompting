"""Shared fixtures for the helpers tests."""

import pytest
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from helpers.data import NextTokenDataset
from transformer.model import TransformerModel


@pytest.fixture
def token_ids() -> list[int]:
    return list(range(20))


@pytest.fixture
def block_size() -> int:
    return 4


@pytest.fixture
def dataset(token_ids: list[int], block_size: int) -> NextTokenDataset:
    return NextTokenDataset(token_ids, block_size)


@pytest.fixture
def loader(dataset: NextTokenDataset) -> DataLoader:
    return DataLoader(dataset, batch_size=4)


@pytest.fixture
def tiny_model() -> TransformerModel:
    return TransformerModel(
        vocab_size=20, embed_dim=8, num_heads=2, num_layers=1, ff_dim=16, output_dim=20
    )


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture
def gpt2_sized_model(tokenizer) -> TransformerModel:
    return TransformerModel(
        vocab_size=len(tokenizer),
        embed_dim=8,
        num_heads=2,
        num_layers=1,
        ff_dim=16,
        output_dim=len(tokenizer),
    )
