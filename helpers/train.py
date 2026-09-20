import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def train(
    model: nn.Module, loader: DataLoader, epochs: int, lr: float, device: str
) -> list[float]:
    """Train with AdamW and cross-entropy, returning the loss at every step."""
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    losses = []

    for epoch in range(epochs):
        for x, y in tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    return losses


@torch.no_grad()
def generate(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    block_size: int,
    device: str,
) -> str:
    """Append `max_new_tokens` sampled tokens to `prompt`, feeding the last `block_size` ids each step."""
    model.to(device).eval()
    ids = tokenizer.encode(prompt)

    for _ in range(max_new_tokens):
        window = torch.tensor([ids[-block_size:]], device=device)
        probs = torch.softmax(model(window), dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        ids.append(next_id)

    return tokenizer.decode(ids)
