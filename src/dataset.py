"""
Ch02 equivalent: Chess dataset and dataloader.

Replaces GPTDatasetV1 + create_dataloader_v1.
Input: PGN file(s)  ->  sequences of UCI move token ids.

Each game is wrapped: [GAME_START, move1, move2, ..., GAME_END]
Games are concatenated into one long token stream, then sliced with a
sliding window (same as the book's text approach).

Training data format: https://database.lichess.org/ (download a .pgn.zst,
decompress, place in data/). Any standard PGN file works.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import chess.pgn
except ImportError:
    raise ImportError("pip install chess")

from src.vocab import ChessTokenizer


def parse_pgn(
    path: str,
    max_games: int | None = None,
    min_elo: int = 0,
) -> list[list[str]]:
    """
    Read a PGN file and return a list of games.
    Each game is a list of UCI move strings (no special tokens yet).

    min_elo: skip games where either player is below this rating.
             Set to 0 (default) to accept all games.
             Recommended: 1800 for decent quality, 2000 for strong play.
    """
    games   = []
    skipped = 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            if min_elo > 0:
                def _elo(v):
                    try: return int(v or 0)
                    except (ValueError, TypeError): return 0
                white_elo = _elo(game.headers.get("WhiteElo", 0))
                black_elo = _elo(game.headers.get("BlackElo", 0))
                if min(white_elo, black_elo) < min_elo:
                    skipped += 1
                    continue
            moves = [m.uci() for m in game.mainline_moves()]
            if moves:
                games.append(moves)
            if max_games and len(games) >= max_games:
                break
    print(f"Loaded {len(games)} games from {path}" +
          (f" (skipped {skipped} below ELO {min_elo})" if min_elo > 0 else ""))
    return games


def games_to_token_ids(games: list[list[str]], tokenizer: ChessTokenizer) -> list[int]:
    """
    Flatten games into one token stream with GAME_START / GAME_END boundaries.
    Unknown moves (shouldn't happen with full vocab) are silently skipped.
    """
    stream = []
    for moves in games:
        stream.append(tokenizer.game_start_id)
        for m in moves:
            if m in tokenizer.move_to_id:
                stream.append(tokenizer.move_to_id[m])
        stream.append(tokenizer.game_end_id)
    return stream


class ChessDataset(Dataset):
    """
    Ch02 GPTDatasetV1 equivalent for chess.

    Sliding window over the flattened token stream.
    input[i]  = token_ids[i : i + context_length]
    target[i] = token_ids[i+1 : i + context_length + 1]
    """

    def __init__(self, token_ids: list[int], context_length: int, stride: int):
        self.input_ids  = []
        self.target_ids = []

        for i in range(0, len(token_ids) - context_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i : i + context_length]))
            self.target_ids.append(torch.tensor(token_ids[i + 1 : i + context_length + 1]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_chess_dataloader(
    token_ids: list[int],
    context_length: int = 256,
    stride: int = 128,
    batch_size: int = 32,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Ch02 create_dataloader_v1 equivalent."""
    dataset = ChessDataset(token_ids, context_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


def load_data(
    pgn_paths: list[str],
    tokenizer: ChessTokenizer,
    max_games: int | None = None,
    min_elo: int = 0,
):
    """
    Load one or more PGN files and return a flat token id list.
    """
    all_games = []
    for path in pgn_paths:
        all_games.extend(parse_pgn(path, max_games=max_games, min_elo=min_elo))
    return games_to_token_ids(all_games, tokenizer)


if __name__ == "__main__":
    import sys

    tok = ChessTokenizer()
    if len(sys.argv) < 2:
        print("Usage: python -m src.dataset <path/to/games.pgn>")
        sys.exit(1)

    ids = load_data([sys.argv[1]], tok, max_games=1000)
    print(f"Total tokens: {len(ids)}")

    loader = create_chess_dataloader(ids, context_length=256, stride=128, batch_size=4)
    x, y = next(iter(loader))
    print(f"Batch shapes: x={x.shape}, y={y.shape}")
