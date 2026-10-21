# Large-versus-small model comparison

The comparison completed on October 11, 2026. The Studio stopped automatically, and all reports and PGN games are retained locally in runs/comparison/.

The small model is the 7,127,552-parameter Lichess-pretrained checkpoint. The large model is the 24,870,144-parameter checkpoint trained on 497,402,880 tokens. Both evaluated checkpoint hashes match the retained local files.

## Identical held-out position evaluation

Both models used the same 512 sampled positions, the same move histories, Stockfish 16 at depth 8, and greedy decoding restricted to legal moves. The validation PGN, seed, engine, and sampled positions were verified to match.

| Metric | Small model | Large model |
| --- | ---: | ---: |
| Mean centipawn loss (ACPL) | 318.92 | 213.00 |
| Median centipawn loss | 77.0 | 36.5 |
| Positions with loss over 100 centipawns | 43.6% | 24.6% |
| Positions with zero estimated centipawn loss | 9.4% | 18.4% |

Mean move error fell by 33.2%. The paired mean difference (large minus small) was -105.93 centipawns, with a position-bootstrap 95% interval of [-208.89, -6.35]. This bootstrap resamples positions; correlations between positions from the same source game are a limitation. The engine scores are depth-limited estimates, with mate scores mapped to 10,000 centipawns.

## Color-swapped continuation matches

We sampled 50 held-out starting positions with seed 123 and played each position twice, swapping model colors. Each game had a 200-ply limit and greedy legal-move decoding.

The large model scored 47 wins, 53 draws, and 0 losses across 100 games: 73.5 points out of 100. The paired bootstrap 95% score interval was 69% to 78%. No game reached the ply limit; all draws arose under the normal game-over/draw-claim rules.

## Interpretation

The larger training run improved move prediction and playing performance relative to the small model. This comparison changed both model size and training data, so it does not isolate the effect of parameter count. These are continuations from held-out positions, not a rated tournament or an absolute Elo measurement. Beating the small model does not establish strength against Stockfish or human players. Tactical testing and matches against calibrated opponents remain necessary to characterize absolute chess strength.

Artifacts: runs/comparison/summary.json, small_acpl.json, large_acpl.json, matches.json, and matches.pgn.

## Source verification

The saved initial source hash confirms the checkpoint loader already used
eval mode, disabling dropout. The serving-time diagnosis of active dropout
was mistaken; the comparison figures above remain valid.
