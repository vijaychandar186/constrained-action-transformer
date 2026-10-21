"""Color-swapped paired matches between two chess policies."""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
import chess
import chess.pgn
from src.pipeline import (choose_device, evaluation_positions, file_hash, load_checkpoint,
                          pick_move, seed_all, write_games, write_json)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", "--rl", dest="rl", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate-name", default="RL")
    p.add_argument("--baseline-name", default="Stockfish baseline")
    p.add_argument("--positions", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--pairs", type=int, default=10)
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--acpl", nargs="*", default=[])
    p.add_argument("--require-cuda", action="store_true")
    a = p.parse_args()
    if min(a.pairs, a.max_plies) < 1:
        p.error("pairs and max-plies must be positive.")
    seed_all(a.seed)
    device = choose_device(a.require_cuda)
    rl, rt, _ = load_checkpoint(a.rl, device)
    baseline, bt, _ = load_checkpoint(a.baseline, device)
    positions = evaluation_positions(a.positions, a.pairs, a.seed)
    results, games = [], []
    for pair, position in enumerate(positions):
        for rl_white in (True, False):
            board = chess.Board()
            history = list(position["moves"])
            for uci in history:
                board.push_uci(uci)
            game = chess.pgn.Game()
            game.setup(board)
            game.headers.update(Event="Queen's GAMBIT paired comparison",
                                White=a.candidate_name if rl_white else a.baseline_name,
                                Black=a.baseline_name if rl_white else a.candidate_name)
            node = game
            plies = 0
            while not board.is_game_over(claim_draw=True) and plies < a.max_plies:
                is_rl = board.turn == (chess.WHITE if rl_white else chess.BLACK)
                move = pick_move(rl if is_rl else baseline, board, history, rt if is_rl else bt, device)
                board.push(move)
                history.append(move.uci())
                node = node.add_variation(move)
                plies += 1
            outcome = board.outcome(claim_draw=True)
            result = board.result(claim_draw=True) if outcome else "1/2-1/2"
            points = .5 if not outcome or outcome.winner is None else float(outcome.winner == (chess.WHITE if rl_white else chess.BLACK))
            game.headers["Result"] = result
            if not outcome:
                game.headers["Termination"] = "ply-cap adjudication as draw"
            rows = {"pair": pair, "rl_white": rl_white, "initial_fen": position["fen"],
                    "result": result, "rl_points": points, "plies": plies,
                    "truncated": outcome is None}
            results.append(rows)
            games.append(game)
            print(rows, flush=True)
    pair_scores = [sum(row["rl_points"] for row in results if row["pair"] == pair) / 2 for pair in range(len(positions))]
    rng = random.Random(a.seed)
    bootstrap = sorted(sum(rng.choice(pair_scores) for _ in pair_scores) / len(pair_scores) for _ in range(1000))
    acpl = {}
    reference = None
    for path in a.acpl:
        report = json.loads(Path(path).read_text())
        protocol = (report["positions_sha256"], report["seed"], report["depth"], report["engine"], report["decoding"], report["count"],
                    [(row["fen"], row["moves"]) for row in report["rows"]])
        if reference is not None and protocol != reference:
            raise ValueError("ACPL reports use different evaluation protocols.")
        reference = protocol
        acpl[Path(path).stem] = report["acpl"]
    write_games(Path(a.output).with_suffix(".pgn"), games)
    write_json(a.output, {"seed": a.seed, "pairs": len(positions), "max_plies": a.max_plies,
                         "rl_sha256": file_hash(a.rl), "baseline_sha256": file_hash(a.baseline),
                         "positions_sha256": file_hash(a.positions), "decoding": "greedy legal moves",
                         "candidate_name": a.candidate_name, "baseline_name": a.baseline_name,
                         "candidate_score_fraction": sum(pair_scores) / len(pair_scores),
                         "candidate_wins": sum(row["rl_points"] == 1 for row in results),
                         "candidate_losses": sum(row["rl_points"] == 0 for row in results),
                         "draws": sum(row["rl_points"] == .5 for row in results),
                         "rl_score_fraction": sum(pair_scores) / len(pair_scores),
                         "paired_bootstrap_95_ci": [bootstrap[25], bootstrap[974]],
                         "truncated_games": sum(row["truncated"] for row in results),
                         "acpl": acpl, "games": results})

if __name__ == "__main__":
    main()
