"""Shared artifacts, reproducible positions, and checkpoint loading."""
from __future__ import annotations
import hashlib
import json
import os
import random
import shutil
import subprocess
from pathlib import Path
import chess
import chess.pgn
import torch
from src.model import GPTModel
from src.vocab import ChessTokenizer

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)

def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def choose_device(require_cuda=False):
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("This run requires CUDA; refusing to start a full run on CPU.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(path, model, config, tokenizer=None, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save({"format_version": 1, "model": model.state_dict(),
                "config": config, "vocab": list((tokenizer or ChessTokenizer()).move_to_id),
                **metadata}, temp)
    temp.replace(path)

def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if "config" not in checkpoint or "model" not in checkpoint:
        raise ValueError("A pipeline checkpoint with configuration is required.")
    tokenizer = ChessTokenizer()
    if checkpoint["config"]["vocab_size"] > tokenizer.vocab_size:
        from src.boardstate import BoardTokenizer
        tokenizer = BoardTokenizer()
    if checkpoint["vocab"] != list(tokenizer.move_to_id):
        raise ValueError("Checkpoint vocabulary does not match this tokenizer.")
    model = GPTModel(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, tokenizer, checkpoint

def stockfish_path(value=None):
    path = value or os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish")
    if not path:
        raise RuntimeError("Install Stockfish or set STOCKFISH_PATH to its executable.")
    return path

def read_games(paths, min_elo=0, max_elo=None, max_games=None):
    games = []
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as handle:
            while max_games is None or len(games) < max_games:
                game = chess.pgn.read_game(handle)
                if game is None:
                    break
                if game.errors or game.headers.get("Variant", "Standard") not in ("Standard", "Chess"):
                    continue
                if min_elo or max_elo is not None:
                    try:
                        ratings = [int(game.headers[k]) for k in ("WhiteElo", "BlackElo")]
                    except (KeyError, ValueError):
                        continue
                    if min(ratings) < min_elo or max_elo is not None and max(ratings) > max_elo:
                        continue
                if list(game.mainline_moves()):
                    games.append(game)
        if max_games is not None and len(games) >= max_games:
            break
    return games

def write_games(path, games):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for game in games:
            handle.write(str(game) + "\n\n")

def tokens_for_games(games, tokenizer):
    ids = []
    for game in games:
        ids.append(tokenizer.game_start_id)
        ids.extend(tokenizer.move_to_id[move.uci()] for move in game.mainline_moves())
        ids.append(tokenizer.game_end_id)
    return ids

def evaluation_positions(pgn_path, count, seed=42):
    """One reservoir sample over positions, including full move history."""
    rng = random.Random(seed)
    selected = []
    seen = 0
    with open(pgn_path, encoding="utf-8") as handle:
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            if game.errors or game.board().fen() != chess.Board().fen():
                continue
            board = game.board()
            history = []
            for move in game.mainline_moves():
                if not board.is_game_over(claim_draw=True):
                    item = {"fen": board.fen(), "moves": list(history)}
                    seen += 1
                    if len(selected) < count:
                        selected.append(item)
                    else:
                        index = rng.randrange(seen)
                        if index < count:
                            selected[index] = item
                board.push(move)
                history.append(move.uci())
    if not selected:
        raise ValueError("No evaluable positions in the held-out PGN.")
    return selected

def legal_distribution(model, board, context, tokenizer, device, temperature=1.0):
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    moves = list(board.legal_moves)
    if not moves:
        raise ValueError("No legal moves.")
    ids = [tokenizer.move_to_id[m.uci()] for m in moves]
    tokens = tokenizer.context_ids(context) if hasattr(tokenizer, "context_ids") else [tokenizer.game_start_id] + [tokenizer.move_to_id[m] for m in context]
    tokens = tokens[-model.pos_emb.num_embeddings:]
    inputs = torch.tensor([tokens], dtype=torch.long, device=device)
    logits = model(inputs)[0, -1, ids] / temperature
    return moves, torch.log_softmax(logits, dim=-1)

def pick_move(model, board, context, tokenizer, device, sample=False):
    with torch.no_grad():
        moves, log_probs = legal_distribution(model, board, context, tokenizer, device)
    index = torch.multinomial(log_probs.exp(), 1).item() if sample else log_probs.argmax().item()
    return moves[index]
