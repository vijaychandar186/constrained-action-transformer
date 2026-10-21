"""Generate games using Stockfish suggestions for both sides."""
from __future__ import annotations
import argparse
import random
from pathlib import Path
import chess
import chess.engine
import chess.pgn
from src.pipeline import file_hash, seed_all, stockfish_path, write_games, write_json

def generate(output, games, depth, max_plies, seed, executable=None):
    if min(games, depth, max_plies) < 1:
        raise ValueError("games, depth and max-plies must be positive.")
    rng = random.Random(seed)
    generated = []
    with chess.engine.SimpleEngine.popen_uci(stockfish_path(executable)) as engine:
        engine.configure({"Threads": 1, "Hash": 64})
        identity = dict(engine.id)
        for index in range(games):
            board = chess.Board()
            game = chess.pgn.Game()
            game.headers.update(Event="Stockfish self-play", White="Stockfish", Black="Stockfish")
            node = game
            for ply in range(max_plies):
                if board.is_game_over(claim_draw=True):
                    break
                if ply < 12:
                    choices = engine.analyse(board, chess.engine.Limit(depth=depth),
                                             multipv=min(3, board.legal_moves.count()), game=game)
                    move = rng.choice(choices)["pv"][0]
                else:
                    move = engine.play(board, chess.engine.Limit(depth=depth), game=game).move
                board.push(move)
                node = node.add_variation(move)
            game.headers["Result"] = board.result(claim_draw=True)
            if not board.is_game_over(claim_draw=True):
                game.headers["Termination"] = "ply limit reached"
            generated.append(game)
            if (index + 1) % 25 == 0:
                # Persist work without presenting an incomplete dataset as complete.
                write_games(Path(output).with_suffix(".partial.pgn"), generated)
                print("Stockfish games:", index + 1, flush=True)
    write_games(output, generated)
    write_json(Path(output).with_suffix(".manifest.json"),
               {"source": "Stockfish self-play", "engine": identity, "games": games,
                "depth": depth, "max_plies": max_plies, "seed": seed,
                "opening_sampling": "uniform over top three Stockfish PVs for first 12 plies",
                "sha256": file_hash(output)})
    print("Saved Stockfish games:", output, flush=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--games", type=int, default=512)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stockfish")
    a = p.parse_args()
    generate(a.output, a.games, a.depth, a.max_plies, a.seed, a.stockfish)

if __name__ == "__main__":
    main()
