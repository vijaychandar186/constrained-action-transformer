"""
Chess GPT configuration.

CHESS_GPT_CONFIG mirrors the book's GPT_CONFIG_124M but chess-sized:
  - vocab_size   ~4546 (exact count set at runtime from ChessTokenizer)
  - context_length 256 — covers ~128 full moves, enough for most games
  - emb_dim       256 — chess is a simple domain; no need for 768
  - n_heads         8 — head_dim = 256/8 = 32
  - n_layers        6 — ~8M params total
"""

from src.vocab import ChessTokenizer

_tok = ChessTokenizer()

CHESS_GPT_CONFIG = {
    "vocab_size":     _tok.vocab_size,   # ~4546
    "context_length": 256,
    "emb_dim":        256,
    "n_heads":        8,
    "n_layers":       6,
    "drop_rate":      0.1,
    "qkv_bias":       False,
}

TRAIN_SETTINGS = {
    "learning_rate": 3e-4,
    "weight_decay":  0.1,
    "num_epochs":    10,
    "batch_size":    64,
    "eval_freq":     100,   # steps between loss evals
    "eval_iter":     10,    # batches used per eval
    "max_games":     None,  # set to e.g. 50000 to limit training data
    "min_elo":       1800,  # skip games where either player is below this ELO
}
