# TODO

## Implemented and locally checked

- [x] UCI vocabulary and legal-move decoding.
- [x] Stream and version a bounded 1800?2500 Lichess dataset.
- [x] Supervised CLI with game-level splits, AMP, clipping, and resumable checkpoints.
- [x] Save configuration, dataset hashes, vocabulary, logs, and validation metrics.
- [x] Generate Stockfish-only games and train a separate baseline.
- [x] Stockfish-reward updates with frozen-reference KL and early stopping.
- [x] Shared held-out ACPL evaluation and paired matches.
- [x] Sequential queue with logs, artifact checks, timeouts, locking, and resume.
- [x] Isolated Lightning configuration and a secret-free bounded Studio queue launcher.
- [x] Board-state tokens and move-target-only loss.
- [x] Legal-move HTTP serving, PyTorch benchmarks, and verified model export.
- [x] Small end-to-end queue and behavioral tests.

## Completed experiments

These results have retained run artifacts, checkpoints, logs, and reports.

- [x] Launch the T4_SMALL GPU queue; CUDA and Stockfish preflight passed.
- [x] Complete all 11 queue stages and retain datasets, checkpoints, logs, and reports locally.
- [x] Complete 1800?2500 pretraining and retain cloud artifacts.
- [x] Complete Stockfish baseline and RL post-training.
- [x] Report shared-protocol ACPL and paired-match results.
- [x] Train and evaluate a larger model (24.9M parameters, exactly 20 tokens per parameter).
- [x] Expand held-out evaluation and compare RL directly against its pretrained reference on 512 positions and 100 paired games; the pilot did not establish a reliable improvement.

- [x] Add streaming deduplicated token packing and nonoverlapping memory-mapped training.
- [x] Add the 24.9M model, measured throughput gate, and 20-hour GPU cap.
- [x] Benchmark the 24.9M model on T4: 30.3k tokens/s; projected 5.7 hours with allowance.
- [x] Complete CPU preparation: 6,649,336 games and 497,402,969 packed tokens.
- [x] Complete the 497M-token T4 training run and 512-position held-out evaluation; checkpoint retained locally.

- [x] Compare large and small pretrained checkpoints on identical 512 positions and 100 color-swapped continuation matches; retain reports and games.

- [x] Complete Stockfish-feedback RL on the 24.9M model, identical-position evaluation, and 100 matches against its reference; no clear match improvement.

- [x] Host a chess.js board and original/RL move APIs in a separate Lightning CPU Studio; verify both through the real browser URL.
- [x] Verify castling, promotions, undo, checkmate, and responsive layout in a browser.
- [x] Repeat RL-versus-reference verification; identical results confirmed that inference was already in eval mode.

## Optional research extensions

These are not required to train, evaluate, serve, or deploy the current project.

- [ ] Run the all-ELO vs. 2000+ vs. 2500+ comparison.
- [ ] Train and evaluate the board-state model on GPU.

## External serving validation

- [ ] Validate TensorRT-LLM AutoDeploy compatibility on Linux with an Ampere-or-newer GPU.
- [ ] Benchmark TensorRT-LLM latency, throughput, memory, agreement, and legality on that GPU.
