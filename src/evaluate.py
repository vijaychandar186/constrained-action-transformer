"""Evaluate all policies on a shared held-out position protocol."""
from __future__ import annotations
import argparse
import math
import statistics
import chess
import chess.engine
from src.pipeline import (choose_device, evaluation_positions, file_hash, load_checkpoint,
                          pick_move, seed_all, stockfish_path, write_json)

def mover_score(engine, board, turn, depth):
    result = board.outcome(claim_draw=True)
    if result:
        return 0 if result.winner is None else (10000 if result.winner == turn else -10000)
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    return info["score"].pov(turn).score(mate_score=10000)

def centipawn_loss(engine, board, move, depth):
    turn = board.turn
    # Clear transposition state so ordering across checkpoints cannot bias results.
    engine.configure({"Clear Hash": None})
    best = mover_score(engine, board, turn, depth)
    after = board.copy(stack=True)
    after.push(move)
    engine.configure({"Clear Hash": None})
    chosen = mover_score(engine, after, turn, depth)
    return max(0, best - chosen), best, chosen

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--positions", required=True, help="Shared held-out PGN")
    p.add_argument("--output", required=True)
    p.add_argument("--count", type=int, default=128)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stockfish")
    p.add_argument("--require-cuda", action="store_true")
    a = p.parse_args()
    if min(a.count, a.depth) < 1:
        p.error("count and depth must be positive.")
    seed_all(a.seed)
    device = choose_device(a.require_cuda)
    model, tokenizer, checkpoint = load_checkpoint(a.checkpoint, device)
    positions = evaluation_positions(a.positions, a.count, a.seed)
    rows = []
    with chess.engine.SimpleEngine.popen_uci(stockfish_path(a.stockfish)) as engine:
        engine.configure({"Threads": 1, "Hash": 64})
        identity = dict(engine.id)
        for index, position in enumerate(positions):
            board = chess.Board()
            for uci in position["moves"]:
                board.push_uci(uci)
            if board.fen() != position["fen"]:
                raise ValueError("Position history does not match its FEN.")
            move = pick_move(model, board, position["moves"], tokenizer, device)
            loss, best, chosen = centipawn_loss(engine, board, move, a.depth)
            rows.append({**position, "move": move.uci(), "centipawn_loss": loss,
                         "best_score": best, "chosen_score": chosen})
            if (index + 1) % 25 == 0:
                print("Evaluated positions:", index + 1, flush=True)
    losses = [r["centipawn_loss"] for r in rows]
    write_json(a.output, {"checkpoint": a.checkpoint, "checkpoint_sha256": file_hash(a.checkpoint),
                         "positions_sha256": file_hash(a.positions), "seed": a.seed,
                         "depth": a.depth, "engine": identity, "decoding": "greedy with legal mask",
                         "mate_score": 10000, "count": len(rows), "acpl": statistics.mean(losses),
                         "acpl_standard_error": statistics.stdev(losses) / math.sqrt(len(losses)) if len(losses) > 1 else 0,
                         "rows": rows})
    print("ACPL:", statistics.mean(losses), flush=True)

if __name__ == "__main__":
    main()
