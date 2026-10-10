"""
Ch02 equivalent: Chess tokenizer.

Replaces tiktoken. Vocab = all possible UCI move strings + 2 special tokens.
No subword splitting needed — each chess move is one token.

Vocab breakdown:
  - Non-promotion moves: all (from_sq, to_sq) pairs where from != to  -> 64*63 = 4032
  - White promotions: rank7->rank8, any file pair, piece in qrbn      -> 8*8*4 = 256
  - Black promotions: rank2->rank1, any file pair, piece in qrbn      -> 8*8*4 = 256
  - Special tokens: <|game_start|>, <|game_end|>                      -> 2
  Total: 4546
"""

import os

GAME_START = "<|game_start|>"
GAME_END   = "<|game_end|>"
SPECIAL_TOKENS = [GAME_START, GAME_END]


def _enumerate_all_moves():
    files = "abcdefgh"
    ranks = "12345678"
    pieces = "qrbn"
    moves = []

    # All non-promotion moves (from_sq != to_sq)
    for f1 in files:
        for r1 in ranks:
            for f2 in files:
                for r2 in ranks:
                    if f1 + r1 != f2 + r2:
                        moves.append(f1 + r1 + f2 + r2)

    # Promotion moves: rank 7->8 (white) and rank 2->1 (black)
    # Include all file combos — impossible ones never appear in training data
    # and are masked at inference time, so they harmlessly consume vocab slots.
    for f1 in files:
        for f2 in files:
            for p in pieces:
                moves.append(f1 + "7" + f2 + "8" + p)
                moves.append(f1 + "2" + f2 + "1" + p)

    return moves


class ChessTokenizer:
    """
    Minimal tokenizer: chess move string -> integer id, and back.

    Special token ids:
      GAME_START = 0
      GAME_END   = 1
    Move ids start at 2.
    """

    def __init__(self):
        all_moves = _enumerate_all_moves()
        vocab = SPECIAL_TOKENS + sorted(set(all_moves))
        self.move_to_id = {m: i for i, m in enumerate(vocab)}
        self.id_to_move = {i: m for m, i in self.move_to_id.items()}
        self.vocab_size  = len(vocab)
        self.game_start_id = self.move_to_id[GAME_START]
        self.game_end_id   = self.move_to_id[GAME_END]

    def encode(self, moves: list[str]) -> list[int]:
        """Encode a list of move strings (including special tokens) to ids."""
        return [self.move_to_id[m] for m in moves]

    def decode(self, ids: list[int]) -> list[str]:
        """Decode a list of ids back to move strings."""
        return [self.id_to_move[i] for i in ids]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            for i in range(self.vocab_size):
                f.write(self.id_to_move[i] + "\n")
        print(f"Vocab ({self.vocab_size} tokens) saved to {path}")

    @classmethod
    def load(cls, path: str) -> "ChessTokenizer":
        """Load a tokenizer from a saved vocab file (order must match)."""
        tok = cls.__new__(cls)
        with open(path) as f:
            vocab = [line.rstrip("\n") for line in f]
        tok.move_to_id = {m: i for i, m in enumerate(vocab)}
        tok.id_to_move = {i: m for m, i in tok.move_to_id.items()}
        tok.vocab_size  = len(vocab)
        tok.game_start_id = tok.move_to_id[GAME_START]
        tok.game_end_id   = tok.move_to_id[GAME_END]
        return tok


if __name__ == "__main__":
    tok = ChessTokenizer()
    print(f"Vocab size: {tok.vocab_size}")
    print(f"game_start id: {tok.game_start_id}")
    print(f"game_end   id: {tok.game_end_id}")
    sample = [GAME_START, "e2e4", "e7e5", "g1f3", GAME_END]
    ids = tok.encode(sample)
    print(f"encode {sample} -> {ids}")
    print(f"decode {ids} -> {tok.decode(ids)}")
    tok.save("data/vocab.txt")
