"""Supervised training with game-level splits and resumable checkpoints."""
from __future__ import annotations
import argparse
import json
import math
import random
from pathlib import Path
import torch
from config import CHESS_GPT_CONFIG
from src.dataset import create_chess_dataloader
from src.model import GPTModel
from src.pipeline import (choose_device, file_hash, read_games, save_checkpoint,
                          seed_all, tokens_for_games, write_games, write_json)
from src.vocab import ChessTokenizer

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pgn", nargs="+")
    p.add_argument("--output", default="checkpoints/pretrain.pth")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--min-elo", type=int, default=0)
    p.add_argument("--max-elo", type=int)
    p.add_argument("--max-games", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--emb-dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--board-state", action="store_true")
    p.add_argument("--validation-pgn", help="Explicit disjoint validation games for controlled comparisons")
    a = p.parse_args()
    if min(a.epochs, a.batch_size, a.save_every) < 1:
        p.error("epochs, batch-size and save-every must be positive.")
    seed_all(a.seed)
    device = choose_device(a.require_cuda)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    output = Path(a.output)
    directory = output.parent / output.stem
    directory.mkdir(parents=True, exist_ok=True)
    latest = output.with_name(output.stem + "_last.pth")
    tokenizer = ChessTokenizer()
    games = read_games(a.pgn, a.min_elo, a.max_elo, a.max_games)
    if len(games) < 2:
        raise ValueError("At least two complete games are needed for a train/validation split.")
    random.Random(a.seed).shuffle(games)
    split = max(1, min(len(games) - 1, int(.9 * len(games))))
    training, validation = games[:split], games[split:]
    if a.validation_pgn:
        training = games
        validation = read_games([a.validation_pgn])
        if not validation:
            raise ValueError("The explicit validation PGN is empty.")
        training_games = {str(game) for game in training}
        if any(str(game) in training_games for game in validation):
            raise ValueError("Training and validation share an identical game.")
    write_games(directory / "validation.pgn", validation)
    write_games(directory / "training.pgn", training)
    config = dict(CHESS_GPT_CONFIG, context_length=a.context_length,
                  emb_dim=a.emb_dim, n_layers=a.layers, n_heads=a.heads)
    if a.board_state:
        from src.boardstate import BoardTokenizer, create_board_loader
        tokenizer = BoardTokenizer()
        config["vocab_size"] = tokenizer.vocab_size
        train_loader = create_board_loader(training, tokenizer, a.context_length, a.batch_size, True)
        val_loader = create_board_loader(validation, tokenizer, a.context_length, a.batch_size, False)
    else:
        train_ids, val_ids = [tokens_for_games(part, tokenizer) for part in (training, validation)]
        train_loader = create_chess_dataloader(train_ids, a.context_length, a.context_length // 2, a.batch_size, True, False)
        val_loader = create_chess_dataloader(val_ids, a.context_length, a.context_length // 2, a.batch_size, False, False)
    if not len(train_loader) or not len(val_loader):
        raise ValueError("Dataset split is too short for the configured context window.")
    manifest = {"sources": [{"path": path, "sha256": file_hash(path)} for path in a.pgn],
                "train_games": len(training), "validation_games": len(validation),
                "validation_sha256": file_hash(directory / "validation.pgn"),
                "seed": a.seed, "min_elo": a.min_elo, "max_elo": a.max_elo,
                "representation": "board-state" if a.board_state else "moves"}
    write_json(directory / "dataset.json", manifest)
    write_json(directory / "config.json", {"model": config, "training": vars(a)})
    tokenizer.save(str(directory / "vocab.txt"))
    model = GPTModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs * len(train_loader))
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    epoch_start, batch_start, step = 0, 0, 0
    if a.resume and latest.exists():
        saved = torch.load(latest, map_location=device, weights_only=True)
        if saved["config"] != config or saved["dataset"] != manifest:
            raise ValueError("Resume checkpoint configuration or dataset changed.")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        epoch_start, batch_start, step = saved["epoch"], saved["batch"], saved["step"]
        torch.set_rng_state(saved["rng"].cpu())
        if amp:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        print("Resuming step", step, flush=True)

    def checkpoint(path, epoch, batch, metrics=None):
        save_checkpoint(path, model, config, tokenizer=tokenizer, optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                        epoch=epoch, batch=batch, step=step, dataset=manifest,
                        rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if amp else [],
                        metrics=metrics or {})

    def validation_loss():
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100, reduction="sum")
                total += loss.item()
                count += (targets != -100).sum().item()
        model.train()
        if not count:
            raise ValueError("Validation contains no move targets.")
        return total / count

    print("Device:", device, "| Parameters:", sum(x.numel() for x in model.parameters()), flush=True)
    last_metrics = {}
    for epoch in range(epoch_start, a.epochs):
        # A fixed per-epoch permutation makes skipping saved batches reproducible.
        train_loader.generator = torch.Generator().manual_seed(a.seed + epoch)
        train_loader.sampler.generator = train_loader.generator
        model.train()
        total_loss, count = 0.0, 0
        for batch, (inputs, targets) in enumerate(train_loader):
            if epoch == epoch_start and batch < batch_start:
                continue
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(inputs)
                loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            total_loss += loss.item()
            count += 1
            if step % 100 == 0:
                print(f"epoch={epoch+1} step={step} loss={loss.item():.4f}", flush=True)
            if step % a.save_every == 0:
                checkpoint(latest, epoch, batch + 1)
            if a.max_steps and step >= a.max_steps:
                break
        val_loss = validation_loss()
        last_metrics = {"epoch": epoch + 1, "step": step, "train_loss": total_loss / max(count, 1),
                        "validation_loss": val_loss, "validation_perplexity": math.exp(min(val_loss, 80))}
        with (directory / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(last_metrics) + "\n")
        checkpoint(latest, epoch + 1, 0, last_metrics)
        write_json(directory / "metrics.json", last_metrics)
        print(last_metrics, flush=True)
        batch_start = 0
        if a.max_steps and step >= a.max_steps:
            break
    if not last_metrics:
        last_metrics = json.loads((directory / "metrics.json").read_text())
    checkpoint(output, last_metrics["epoch"], 0, last_metrics)
    print("Final checkpoint:", output, flush=True)

if __name__ == "__main__":
    main()
