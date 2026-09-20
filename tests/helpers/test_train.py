import math

from helpers.train import generate, train


def test_train_returns_one_loss_per_batch(tiny_model, loader):
    losses = train(tiny_model, loader, epochs=1, lr=1e-3, device="cpu")
    assert len(losses) == len(loader)
    assert all(math.isfinite(loss) for loss in losses)


def test_train_two_epochs_doubles_steps(tiny_model, loader):
    losses = train(tiny_model, loader, epochs=2, lr=1e-3, device="cpu")
    assert len(losses) == 2 * len(loader)


def test_generate_extends_prompt(gpt2_sized_model, tokenizer):
    prompt = "Winston"
    text = generate(
        gpt2_sized_model, tokenizer, prompt, max_new_tokens=5, block_size=4, device="cpu"
    )
    assert isinstance(text, str)
    assert text.startswith(prompt)
    assert len(tokenizer.encode(text)) == len(tokenizer.encode(prompt)) + 5
