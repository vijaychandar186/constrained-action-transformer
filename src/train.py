"""
Ch05 equivalent: Training loop for ChessGPT.

Replaces gpt_train.py. Same structure as the book:
  - calc_loss_batch / calc_loss_loader
  - evaluate_model
  - train_model_simple

Key differences from book:
  - No tiktoken / text — uses ChessTokenizer
  - No URL download — expects PGN files in data/
  - Checkpoint saving every N epochs
"""

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.vocab import ChessTokenizer
from src.dataset import load_data, create_chess_dataloader
from src.model import GPTModel
from src.generate import generate_sample_game


# ---------------------------------------------------------------------------
# Ch05: Loss utilities
# ---------------------------------------------------------------------------

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch  = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss   = nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    num_batches = min(num_batches or len(data_loader), len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        total_loss += calc_loss_batch(input_batch, target_batch, model, device).item()
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss   = calc_loss_loader(val_loader,   model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


# ---------------------------------------------------------------------------
# Ch05: Training loop
# ---------------------------------------------------------------------------

def train_model(
    model, train_loader, val_loader, optimizer, device,
    num_epochs, eval_freq, eval_iter, tokenizer,
    checkpoint_dir="checkpoints", checkpoint_every=1,
):
    train_losses, val_losses, tokens_seen_list = [], [], []
    tokens_seen = 0
    global_step = -1

    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                tokens_seen_list.append(tokens_seen)
                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        # Print a sample game snippet after each epoch
        sample = generate_sample_game(model, tokenizer, device, num_moves=10)
        print(f"  Sample moves: {' '.join(sample)}")

        if (epoch + 1) % checkpoint_every == 0:
            ckpt = os.path.join(checkpoint_dir, f"chess_gpt_ep{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"  Checkpoint saved: {ckpt}")

    return train_losses, val_losses, tokens_seen_list


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, save_path="loss.pdf"):
    fig, ax1 = plt.subplots()
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    plt.savefig(save_path)
    print(f"Loss plot saved to {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    from src.supervised import main as run
    run()


if __name__ == "__main__":
    main()
