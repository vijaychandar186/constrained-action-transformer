"""Stream a bounded sample of rated standard Lichess games."""
from __future__ import annotations
import argparse
import io
import json
import urllib.request
from pathlib import Path
import chess.pgn
import zstandard
from src.pipeline import write_games, write_json, file_hash, read_games

def prepare(source, output, max_games, min_elo, max_elo, exclude_pgn=None):
    output = Path(output)
    manifest_path = output.with_suffix(".manifest.json")
    selection = dict(source=source, max_games=max_games, min_elo=min_elo, max_elo=max_elo,
                     exclude_sha256=file_hash(exclude_pgn) if exclude_pgn else None)
    excluded = {str(game) for game in read_games([exclude_pgn])} if exclude_pgn else set()
    if output.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if all(manifest.get(k) == v for k, v in selection.items()) and manifest["sha256"] == file_hash(output):
            print("Using verified dataset:", output)
            return
    if max_games <= 0 or not 0 <= min_elo <= max_elo:
        raise ValueError("Choose a positive game count and an ordered rating range.")
    is_url = source.startswith(("https://", "http://"))
    raw = urllib.request.urlopen(urllib.request.Request(source, headers={"User-Agent": "QueensGambit/1.0"}), timeout=60) if is_url else open(source, "rb")
    accepted = []
    scanned = 0
    try:
        stream = zstandard.ZstdDecompressor().stream_reader(raw) if source.endswith(".zst") else raw
        with io.TextIOWrapper(stream, encoding="utf-8", errors="replace") as handle:
            while len(accepted) < max_games:
                game = chess.pgn.read_game(handle)
                if game is None:
                    break
                scanned += 1
                try:
                    ratings = [int(game.headers[k]) for k in ("WhiteElo", "BlackElo")]
                except (KeyError, ValueError):
                    continue
                if game.errors or game.headers.get("Variant", "Standard") not in ("Standard", "Chess"):
                    continue
                if min(ratings) < min_elo or max(ratings) > max_elo or not list(game.mainline_moves()):
                    continue
                if str(game) in excluded:
                    continue
                accepted.append(game)
                if len(accepted) % 1000 == 0:
                    print(f"Accepted {len(accepted)} / scanned {scanned}", flush=True)
    finally:
        raw.close()
    if len(accepted) < 10:
        raise ValueError("Fewer than ten eligible games; use a larger source or check the ratings.")
    temporary = output.with_suffix(".partial.pgn")
    write_games(temporary, accepted)
    temporary.replace(output)
    write_json(manifest_path, {**selection, "accepted_games": len(accepted), "scanned_games": scanned, "sha256": file_hash(output)})
    print("Dataset saved:", output, len(accepted), "games")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-games", type=int, default=20000)
    p.add_argument("--min-elo", type=int, default=1800)
    p.add_argument("--max-elo", type=int, default=2500)
    p.add_argument("--exclude-pgn", help="Exclude shared held-out games from the training sample")
    a = p.parse_args()
    prepare(a.source, a.output, a.max_games, a.min_elo, a.max_elo, a.exclude_pgn)

if __name__ == "__main__":
    main()
