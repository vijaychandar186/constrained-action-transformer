# Queen's GAMBIT — Game-Aware Move-Based Inference Transformer

**GAMBIT = Game-Aware Move-Based Inference Transformer**

A GPT transformer trained purely on chess games, following the *LLMs from Scratch* architecture adapted for chess. No language model or fine-tuning: the vocabulary is every possible chess move.

**Reference implementation:** [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch). Queen's GAMBIT adapts its dataset, model, training loop, and generation code by replacing text/tiktoken with UCI chess moves.

---

## What This Is

Standard approaches fine-tune a language model such as GPT-2 or LLaMA on chess PGN. This project instead builds a dedicated chess model from scratch:

- Vocabulary: all ~4,546 possible UCI moves, not English tokens
- Architecture: a chess-sized GPT transformer
- Training: Lichess games, optionally filtered by ELO
- Inference: legal-move masking

The core hypothesis is that a model trained on high-ELO games should produce stronger play than one trained on all ELOs, measurable with Average Centipawn Loss (ACPL).

---

## Notebooks

| Notebook | Description |
|---|---|
| `notebooks/chess_gpt_colab.ipynb` | POC: 2018 Lichess data, all ELOs, no legality training signal |
| `notebooks/chess_gpt_1800elo_colab.ipynb` | 2026-04 data, 2000+ ELO, illegal-move penalty during training |
| `notebooks/chess_gpt_2500elo_colab.ipynb` | 2026-04 data, 2500+ ELO, illegal-move penalty during training |
| `notebooks/chess_gpt_boardstate_colab.ipynb` | Board-state tokens interleaved, intrinsic legality, AMP, and ACPL evaluation |

---

## Architecture

Standard GPT, based on the attention, layer-normalization, feed-forward, and transformer-block pattern:

```
MultiHeadAttention → LayerNorm → FeedForward → LayerNorm
```

| | POC | ELO notebooks | Board-state |
|---|---:|---:|---:|
| Vocab | 4,546 | 4,546 | **4,559** (+ 13 piece tokens) |
| Context | 256 | 256 | **512** |
| `emb_dim` | 256 | 256 | **384** |
| Layers | 6 | 6 | **8** |
| Parameters | ~7M | ~7M | **~14M** |

---

## Tokenization

### Move tokens

Moves use UCI: origin square + destination square + optional promotion piece.

- `e2e4`: pawn push
- `e1g1`: kingside castling
- `e7e8q`: promotion to queen

The vocabulary exhaustively enumerates:

- 4,032 regular moves (64 origin squares × 63 destinations)
- 512 promotion moves
- `<|game_start|>` and `<|game_end|>`
- **4,546 tokens total**

Games are represented as `[GAME_START, e2e4, d7d5, ..., GAME_END]`, concatenated into a flat token stream, then split into sliding windows.

### Why UCI instead of SAN?

SAN (`e4`, `Nf3`) identifies the moving piece, while UCI (`e2e4`, `g1f3`) only identifies squares. The model must infer board state from the game history: it only sees legal moves in training, so it learns which moves are plausible from a position.

### Board-state tokens

The advanced notebook makes state explicit:

```
[GAME_START, board0 × 64, move1, board1 × 64, move2, ..., GAME_END]
```

It adds 13 piece-token types: `. P N B R Q K p n b r q k`. Cross-entropy is calculated only at move-token targets; board tokens provide context.

---

## Legal Move Masking

### Inference

All non-legal move logits are set to `-inf` before softmax, preserving the learned distribution among legal moves:

```python
mask = torch.full_like(logits, float("-inf"))
mask[legal_ids] = logits[legal_ids]
probs = torch.softmax(mask / temperature, dim=-1)
```

### ELO notebook training signal

The ELO notebooks also penalize probability mass assigned to illegal moves:

```python
loss = ce_loss + lambda * illegal_prob_mass
```

This requires tracking board state while building the dataset and storing legal move IDs for each token position.

### Board-state learning

With board-state tokens, the model sees a complete board before each move and learns legal moves directly from the data. Inference-time masking remains a useful safety net for rare positions.

---

## Training

### Data

Lichess rated games are available at [database.lichess.org](https://database.lichess.org/). Files are `.pgn.zst`; decompress them before training. ELO filtering skips games when either player is below the configured threshold.

| Notebook | ELO filter | Game cap |
|---|---:|---:|
| POC | none | 121k |
| 1800 ELO | 2000+ | 500k |
| 2500 ELO | 2500+ | 100k |
| Board-state | 2000+ | 100k |

### Improvements explored in notebooks

- Overlapping windows with `stride = context_length // 2`
- Gradient clipping at 1.0
- Cosine learning-rate annealing
- Mixed precision (AMP) in the board-state notebook
- Move-token-only loss for board-state training

### POC results

On 2018 all-ELO data over 10 epochs:

```
Epoch 1:  loss 8.48
Epoch 4:  loss 4.21 — recognizable openings appearing
Epoch 10: loss 3.05 — coherent positional play, frequent 50-move draws
```

---

## Evaluation: ACPL

Average Centipawn Loss measures how much worse a model move is than Stockfish's best move. Lower is stronger.

```python
CPL = max(0, eval(Stockfish best move) - eval(model move))
```

Approximate reference points:

| Player | ACPL |
|---|---:|
| Random legal moves | ~300 |
| Beginner | ~200 |
| Club player (~1200) | ~100 |
| Intermediate (~1800) | ~50 |
| Strong (~2500) | ~20 |
| Stockfish vs. itself | ~0 |

---

## Why Not Fine-Tune a Language Model?

Advantages of a dedicated chess vocabulary:

- No capacity spent on unrelated language tokens
- Atomic move tokens (`e2e4` is one token)
- Natural board-state extension
- A smaller, faster model for the chess domain

Trade-offs:

- No pre-trained knowledge transfer
- Less established tooling and community support

---

## Project Structure

```
src/
  vocab.py      — ChessTokenizer and UCI move enumeration
  dataset.py    — PGN parser, sliding-window dataset, DataLoader
  model.py      — GPTModel and transformer components
  train.py      — training loop and loss utilities
  generate.py   — legal-move masking and game loop
notebooks/
  chess_gpt_colab.ipynb              — POC
  chess_gpt_1800elo_colab.ipynb      — 2000+ ELO + training signal
  chess_gpt_2500elo_colab.ipynb      — 2500+ ELO + training signal
  chess_gpt_boardstate_colab.ipynb   — board-state tokens + ACPL
config.py       — model and training settings
TODO.md         — planned and unverified work
```
