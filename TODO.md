# TODO

## Completed baseline

- [x] Represent moves as direct UCI tokens.
- [x] Apply a legality mask during decoding so invalid moves receive zero probability.

## Training and evaluation

- [ ]  Pretraining run on Lightning AI using Lichess games from players rated 1800-2500.
- [ ] Save the resulting checkpoint, configuration, dataset version, training logs, and final validation metrics.
- [ ] Add reinforcement-learning post-training using Stockfish evaluation scores as rewards.
- [ ] Define the reward formulation and safeguards, including a KL penalty or equivalent policy-stability constraint.
- [ ] Train a baseline exclusively on Stockfish-generated games.
- [ ] Evaluate the RL-post-trained model against the Stockfish-only baseline using a reproducible match and ACPL protocol.

## Serving

- [ ] Add TensorRT-LLM inference serving and benchmark latency, throughput, and GPU memory against the PyTorch implementation.
- [ ] Preserve legal-move masking in the serving path.

## Existing research roadmap

- [ ] Run the supervised ELO comparison experiment (all games vs. 2000+ vs. 2500+) and report ACPL.
- [ ] Move the board-state, AMP, and Stockfish evaluation notebook work into the `src/` package where appropriate.
- [ ] Train and evaluate a larger model.
