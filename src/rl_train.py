"""Stockfish-reward policy updates constrained to a frozen reference."""
from __future__ import annotations
import argparse
import copy
import json
import random
from pathlib import Path
import chess
import chess.engine
import torch
from src.evaluate import centipawn_loss
from src.pipeline import (choose_device, evaluation_positions, file_hash, legal_distribution,
                          load_checkpoint, save_checkpoint, seed_all, stockfish_path, write_json)

def policy_loss(log_probs, reference_log_probs, index, advantage, kl_weight):
    probabilities = log_probs.exp()
    kl = torch.sum(probabilities * (log_probs - reference_log_probs)).clamp_min(0)
    return -advantage * log_probs[index] + kl_weight * kl, kl

def group_policy_loss(log_probs, reference_log_probs, indices, rewards, kl_weight):
    if len(indices) < 2 or len(indices) != len(rewards):
        raise ValueError('A reward group must contain at least two sampled moves.')
    values=log_probs.new_tensor(rewards)
    advantages=(values-values.mean())*len(rewards)/(len(rewards)-1)
    kl=torch.sum(log_probs.exp()*(log_probs-reference_log_probs)).clamp_min(0)
    return -(advantages*log_probs[indices]).mean()+kl_weight*kl,kl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--positions", required=True, help="Training PGN, never the held-out evaluation split")
    p.add_argument("--positions-json", action="store_true", help="Verified positions sampled from packed training tokens")
    p.add_argument("--group-size", type=int, default=1)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--kl-weight", type=float, default=.1)
    p.add_argument("--max-kl", type=float, default=.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stockfish")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--require-cuda", action="store_true")
    a = p.parse_args()
    if a.batch_size < 2 or min(a.steps, a.depth, a.group_size) < 1 or a.kl_weight <= 0 or a.max_kl <= 0:
        p.error("counts and policy-stability weights must be positive.")
    seed_all(a.seed)
    device = choose_device(a.require_cuda)
    model, tokenizer, original = load_checkpoint(a.checkpoint, device)
    reference = copy.deepcopy(model).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    # Disable dropout for both policies; gradients still flow through model.eval().
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.learning_rate)
    dataset = {"source": a.positions, "sha256": file_hash(a.positions)}
    heldout = Path(a.checkpoint).parent / Path(a.checkpoint).stem / "validation.pgn"
    heldout_hash = file_hash(heldout) if heldout.exists() else None
    # The queue explicitly supplies the supervised training split.
    if heldout_hash == dataset["sha256"]:
        raise ValueError("RL must not train on held-out evaluation games.")
    if a.positions_json:
        sampled=json.loads(Path(a.positions).read_text())
        provenance=original['dataset']
        if sampled['source_sha256'] != provenance['sha256']['train.bin'] or sampled['excluded_validation_sha256'] != provenance['selection']['validation_sha256']:
            raise ValueError("RL samples must originate from this model's verified training-only token stream.")
        positions=sampled['positions']
        if len(positions)<a.batch_size:
            raise ValueError('Too few training positions.')
    else:
        positions = evaluation_positions(a.positions, max(256, a.batch_size * 16), a.seed)
    directory = Path(a.output).parent / Path(a.output).stem
    directory.mkdir(parents=True, exist_ok=True)
    latest = Path(a.output).with_name(Path(a.output).stem + "_last.pth")
    start = 0
    reference_hash = file_hash(a.checkpoint)
    if a.resume and latest.exists():
        _, _, saved = load_checkpoint(latest, device)
        if saved["reference_sha256"] != reference_hash or saved["dataset"] != dataset:
            raise ValueError("RL resume reference or dataset changed.")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start = saved["step"]
        torch.set_rng_state(saved["rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    settings = {"reward": "-centipawn_loss / (100 + centipawn_loss)", "advantage": "reward minus leave-one-out batch mean",
                "kl": "exact KL(current || frozen supervised reference) over legal moves",
                "kl_weight": a.kl_weight, "max_kl": a.max_kl, "depth": a.depth,
                "batch_size": a.batch_size, "steps": a.steps, "seed": a.seed}
    settings['learning_rate']=a.learning_rate
    settings['group_size']=a.group_size
    if a.group_size>1:
        settings['advantage']='reward minus leave-one-out mean of moves sampled at the same position'
    if a.resume and latest.exists() and saved['settings'] != settings:
        raise ValueError('RL resume hyperparameters changed.')
    write_json(directory / "config.json", settings)
    final = {}
    with chess.engine.SimpleEngine.popen_uci(stockfish_path(a.stockfish)) as engine:
        engine.configure({"Threads": 1, "Hash": 64})
        for step in range(start, a.steps):
            rng = random.Random(a.seed + step)
            records = []
            for _ in range(a.batch_size):
                position = rng.choice(positions)
                board = chess.Board()
                for move in position["moves"]:
                    board.push_uci(move)
                moves, log_probs = legal_distribution(model, board, position["moves"], tokenizer, device)
                with torch.no_grad():
                    _, ref_log_probs = legal_distribution(reference, board, position["moves"], tokenizer, device)
                if board.fen()!=position['fen']:
                    raise ValueError('RL position history differs from its FEN.')
                indices=torch.multinomial(log_probs.detach().exp(), a.group_size, replacement=True).tolist()
                rewards=[]; cached={}
                for index in indices:
                    if index not in cached:
                        cpl, _, _ = centipawn_loss(engine, board, moves[index], a.depth)
                        cached[index]=-cpl/(100+cpl)
                    rewards.append(cached[index])
                records.append((log_probs,ref_log_probs,indices,rewards))
            all_rewards=[r for record in records for r in record[3]]
            mean_reward = sum(all_rewards) / len(all_rewards)
            optimizer.zero_grad(set_to_none=True)
            if a.group_size>1:
                terms=[group_policy_loss(logs,refs,indices,rewards,a.kl_weight) for logs,refs,indices,rewards in records]
            else:
                terms=[policy_loss(logs,refs,indices[0],rewards[0]-(a.batch_size*mean_reward-rewards[0])/(a.batch_size-1),a.kl_weight) for logs,refs,indices,rewards in records]
            loss = torch.stack([term[0] for term in terms]).mean()
            kl = torch.stack([term[1] for term in terms]).mean()
            if not torch.isfinite(loss) or not torch.isfinite(kl):
                raise RuntimeError("Nonfinite RL loss or KL divergence.")
            # Stop before another update when the policy departs from its reference.
            stopped = kl.item() > a.max_kl
            if not stopped:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            final = {"step": step + 1, "reward_mean": mean_reward, "loss": loss.item(),
                     "kl": kl.item(), "reward_variance": sum((reward - mean_reward) ** 2 for reward in all_rewards) / len(all_rewards), "stopped_for_kl": stopped}
            with (directory / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(final) + "\n")
            if (step + 1) % 25 == 0 or stopped or step + 1 == a.steps:
                save_checkpoint(latest, model, original["config"], tokenizer=tokenizer,
                                optimizer=optimizer.state_dict(), step=step + 1, dataset=dataset,
                                reference_sha256=reference_hash, settings=settings, metrics=final,
                                rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [])
                print(final, flush=True)
            if stopped:
                break
    if not final and latest.exists():
        final = torch.load(latest, map_location="cpu", weights_only=True)["metrics"]
    save_checkpoint(a.output, model, original["config"], tokenizer=tokenizer,
                    reference_sha256=reference_hash, dataset=dataset, settings=settings, metrics=final)
    write_json(directory / "metrics.json", final)
    print("RL checkpoint:", a.output, flush=True)

if __name__ == "__main__":
    main()
