# Initial T4 experiment results

The 11-stage queue completed on October 10, 2026 at 7:11 p.m. America/New_York in 27.7 minutes. The Studio stopped automatically; artifacts are saved locally in runs/full/.

All three models were evaluated on the same 128 held-out positions with Stockfish 16 at depth 8 and greedy legal decoding. Lower average centipawn loss (ACPL) is better.

| Model | ACPL |
| --- | ---: |
| Lichess pretraining | 233.16 |
| Stockfish-game baseline | 374.36 |
| RL after Lichess pretraining | 236.17 |

RL did not improve ACPL in this initial run. Its 55% match score against the Stockfish-game baseline came from only 20 games (10 paired starting positions); the paired bootstrap 95% interval was 50% to 62.5%, and one game reached the ply cap. These results do not establish a reliable match advantage. The baseline used only 512 training games, whereas Lichess pretraining used 20,000, so this comparison does not isolate training method from dataset size.

RL completed all 128 updates without triggering the KL stop (final KL 0.0323). T4 PyTorch serving measured 8.47 ms median and 9.72 ms p95 latency across 64 positions, with serial throughput 115.8 moves/s. These are the small initial model's measurements, not TensorRT results.

Next: expand the held-out evaluation, add paired matches between the pretrained and RL models, and check uncertainty before tuning RL or scaling training. Keep the pretrained checkpoint as the provisional reference because it has the lowest measured ACPL.

## Larger-model throughput check

The 24,870,144-parameter model passed a real T4 benchmark. Batch size 64
measured 30,300 tokens/s and
9.23 GB peak Torch allocation.
The projected 497,402,880-token training pass is 5.70 hours
including a 25% allowance, excluding CPU preparation. This is a projection
from synthetic batches, not a completed training measurement. The controller
has advanced to CPU data preparation; model-quality results remain pending.

## Completed larger-model run

The 24,870,144-parameter model completed 30,360 optimizer steps and processed
497,402,880 tokens: exactly 20 tokens per parameter. Final validation loss
was 2.05155 and perplexity 7.77994. Training plus evaluation took 4.66 hours
and the Studio shut down automatically. The checkpoint is retained locally
at runs/large/checkpoints/pretrain.pth; its SHA256 matches the evaluation report.

Stockfish 16 depth-8 evaluation on 512 held-out positions measured ACPL
212.998 with standard error 47.462. The initial model was evaluated on only
128 positions, so the numerical difference alone does not establish an
improvement. Reevaluate both checkpoints on identical sampled positions
and run matches before drawing strength conclusions. No Elo is measured.

## Completed controlled comparison

Both pretrained models were reevaluated on identical 512-position samples.
ACPL was 318.92 for the small model and 213.00 for the large model (33.2% lower).
Median error fell from 77.0 to 36.5 centipawns. In 100 continuation games from
50 held-out positions with colors swapped, the large model had 47 wins,
53 draws, and no losses: a 73.5% score (paired bootstrap 95% interval 69-78%).
No game hit the ply cap. These results support improvement relative to the
small model, not an absolute Elo or an isolated architecture effect.
See [COMPARISON.md](COMPARISON.md) for the full protocol and limitations.

## Serving verification

Checkpoint loading explicitly disables dropout for inference. Verification
against the saved initial source hash confirmed this behavior was already
present in the earlier runs; the initial serving-time diagnosis of active
dropout was mistaken. Earlier measurements remain valid. A repeat
RL-versus-reference run is retained in runs/large_rl_inference/ as verification.

## Large-model Stockfish-feedback RL

The bounded group-relative RL run stopped at attempted step 403 when the
mean sampled KL exceeded 0.1 (observed 0.1153); 402 updates had been applied.
The frozen supervised checkpoint was preserved. Training-only positions were
sampled from the verified packed stream that excludes complete held-out games.

On identical 512 positions, RL measured 193.58 ACPL versus 213.00 for the
supervised reference. The paired difference was -19.42 centipawns with a
position-bootstrap 95% interval [-59.33, 3.67], including no improvement.
Across 100 color-swapped continuation games, RL had 11 wins, 74 draws, and
15 losses: a 48% score with paired bootstrap 95% interval 43.5-52.5%. No
game hit the move cap. This pilot does not establish a playing-strength
improvement, so the playable app defaults to the supervised original while
allowing the RL model to be selected. The RL checkpoint is retained locally
and its SHA256 matches the evaluation report.
