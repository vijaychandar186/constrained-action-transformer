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


## Sequential experiment pipeline

experiments/full_pipeline.json runs one stage at a time: prerequisite checks,
Lichess sampling, supervised training and evaluation, Stockfish self-play,
baseline training and evaluation, RL, RL evaluation, paired matches, and a
PyTorch benchmark.

The initial run uses 20,000 games with both players rated 1800?2500, three
supervised epochs, a 512-game Stockfish baseline, and 128 RL updates. These are
bounded initial settings, not completed research results. Sampling streams
the April 2026 Lichess archive and stops at the game limit. The dataset manifest
records the source, filter, count, and SHA256.

Training splits complete games before making token windows. Checkpoints include
model configuration, vocabulary, optimizer/scheduler state, random state,
dataset manifest, and progress. Validation PGNs, configuration, and metrics
are saved beside each checkpoint. Queue status and per-stage logs live in
the run directory.

ACPL uses the same held-out PGN, position sample, engine version, depth, and
decoding rule for every policy. Matches pair starting positions and exchange
colors. The report identifies games adjudicated as draws at the ply limit.
These limited protocols do not provide a calibrated playing ELO.

RL uses a bounded Stockfish-regret reward, -CPL / (100 + CPL), with a
leave-one-out batch baseline. It penalizes exact KL divergence over legal moves
against the frozen supervised policy, clips gradients, and stops updates at
the KL threshold. RL receives the training split; evaluation uses held-out games.

### Lightning AI

Provider initialization, packaging, launch, and status inspection live in
lightningai/. The API key stays in the root .env and is excluded from uploads.
HTTP debug logging is disabled before authentication.

Copy `lightningai/settings.example.json` to `lightningai/settings.json` and
fill in your Studio/teamspace details; the local settings file is ignored by Git.

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r lightningai\requirements.txt
.\.venv\Scripts\python.exe -m lightningai.config
~~~

If there are multiple teamspaces, set teamspace to owner/teamspace in
lightningai/settings.json. For a bare name, specify org or user alongside it.
The key must permit teamspace access and Studio/job operations.

~~~powershell
.\.venv\Scripts\python.exe -m lightningai.launch --dry-run
.\.venv\Scripts\python.exe -m lightningai.launch
~~~

The default launcher prepares a dedicated CPU Studio and switches it to T4_SMALL
for the sequential queue. GNU timeout bounds the queue process to three hours.
The Studio-native runner stops compute on success, failure, or timeout; this
shutdown path was verified on CPU before GPU launch. Artifacts remain in the
Studio filesystem. Machine, runtime, plan, and output directory are configurable
in settings.json. A failed stage stops the queue, and saved stages support resume.
An optional execution=job mode is available for accounts permitting the Jobs API.
For separate Jobs, prior outputs must be restored before resume.

A 403 when accessing a teamspace must be resolved before launch. This account
permits Studio operations but rejects Jobs API submission, so the configured
execution mode is studio. Local
lightningai/launch.json records either the blocked reason or the submitted Studio/job details.
The initial T4_SMALL queue completed all 11 stages and shut down automatically.
Checkpoints and reports are retained locally under runs/full/; see [RESULTS.md](RESULTS.md). The same
organization rejected L4 with a payment-verification error; that error does
not apply to the successful T4 request.

References: [Lightning SDK](https://lightning.ai/docs/platform/developers/sdk),
[Jobs](https://lightning.ai/docs/cli/stable/command-details/job),
[Lichess database](https://database.lichess.org/).

### Local checks and follow-up experiments

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\run_experiment_queue.py --plan experiments\full_pipeline.json --run-dir runs\full --dry-run
~~~

Install Stockfish and set STOCKFISH_PATH if it is not on PATH. The smoke plan
uses a small fixture at runs/smoke/fixture.pgn, tiny models, shallow engine
analysis, and a few updates. Its results verify functionality, not strength.

experiments/research_pipeline.json defines the ELO comparison, board-state
model, and larger-model follow-up after the full pipeline. It shares held-out
games and excludes those games from downloaded training samples. Board-state
training computes loss only on moves and uses AMP when CUDA is available.

### Serving

python -m src.serve --checkpoint <checkpoint> --port 8000 exposes a local
POST /move endpoint. Send {"moves": ["e2e4", "e7e5"]}; it returns a legal UCI move.
The request must supply move history because it is part of the model input.

experiments/serving_pipeline.json exports the checkpoint to Hugging Face GPT-2,
verifies its logits against the original, and benchmarks PyTorch and an optional
TensorRT-LLM AutoDeploy adapter on identical positions. The adapter masks illegal
tokens before generating exactly one move. Failures have no silent fallback.

Install requirements-serving.txt and a compatible TensorRT-LLM build in a
supported Linux/CUDA environment before using this plan. Current TensorRT-LLM
releases support Ampere-or-newer NVIDIA architectures; the laptop GTX 1650 Ti
cannot perform this validation. GPU compatibility,
latency, throughput, and memory require validation on the target hardware.
The current native TensorRT-LLM support list does not include GPT-2. AutoDeploy
is a candidate path, not a verified speedup. The legacy engine-build CLI is no
longer the current workflow.

References: [supported models](https://nvidia.github.io/TensorRT-LLM/latest/models/supported-models.html),
[AutoDeploy](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/auto_deploy/auto-deploy.md),
[backend migration](https://nvidia.github.io/TensorRT-LLM/latest/legacy/tensorrt-backend-removal.html).

### Larger token-budgeted run

The larger model has 24,870,144 parameters (12 layers, width 384, 8 heads).
Its one-pass target is 497,402,880 fresh training tokens, approximately 20 per
parameter. The CPU preparation step filters 1800-2500 rated standard games,
deduplicates move sequences, and excludes the existing 2,000 validation games.
Little-endian uint16 token files use about 1 GB for the training token budget;
memory-mapped nonoverlapping target windows avoid keeping PGN trees in RAM.

The completed run is retained locally as a checkpoint and report. The 20:1
token-to-parameter ratio was a planning heuristic, not a validated chess
optimum.

The controlled large-versus-small comparison is documented in
[COMPARISON.md](COMPARISON.md). The large model scored 47 wins, 53 draws,
and 0 losses in 100 color-swapped continuation games, with 33.2% lower
ACPL on identical sampled positions. These results do not measure absolute Elo.

### Large-model Stockfish feedback

`experiments/large_rl_pipeline.json` samples 4,096 positions from the verified
packed training stream, whose complete games exclude the held-out validation
games. The initial post-training run uses up to 512 updates, four positions
per update, and four independently sampled legal moves per position. Rewards
are `-CPL/(100+CPL)` from Stockfish 16 at depth 8. Within-position
leave-one-out reward baselines and a frozen-reference KL penalty constrain
the updates; mean KL above 0.1 stops training before the next update.

The queue evaluated the RL and supervised checkpoints on identical
512-position samples and played 100 continuation games with colors swapped.
This is engine-feedback RL, not human-preference RLHF. The completed reports
are retained locally.

### Play in a browser

The chess.js 1.4.0 UI and model API are served together by `src.chess_web`.
The app supports White or Black, original or Stockfish-RL opponents, legal
move highlighting, castling, promotion choices, undo, board flipping, and
checkmate/draw detection. chess.js is vendored with its MIT license so the
page does not depend on a browser CDN. The backend validates every history
and masks model outputs to legal moves. Checkpoint loading disables dropout
for inference.

Run locally:

```powershell
.\.venv\Scripts\python.exe -m src.chess_web --original runs/large/checkpoints/pretrain.pth --rl runs/large_rl/checkpoints/rl.pth
```

Open http://127.0.0.1:8000. The Lightning app uses a separate CPU Studio,
`queens-gambit-play`, so training shutdown does not stop the chess UI.
`python -m lightningai.serve_chess` prepares the app and exposes port 8000;
its verified URL is saved in ignored `lightningai/serving.json`. The models
are read from the training Studio's shared teamspace filesystem. The API key
stays in the local .env and is excluded from uploads. To stop hosting, stop
`queens-gambit-play` in Lightning. Model availability depends on that hosting
Studio running; this is a Studio-hosted app, not an autoscaling deployment.
