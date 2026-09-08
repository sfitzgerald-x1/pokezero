# Full-path encoder timing readout

Date: 2026-09-08. **Decision: park the identifier-normalization fast path as a
full-search performance candidate.** It retains its narrow encoder microbenchmark
and output-coverage value, but this development-only read does not support a
reproducible end-to-end latency improvement or reinvesting time into extra search.

## Frozen scope

- Baseline source: `db09c56f59991846f04526dc16c004adef8ba893`.
- Candidate source: `cd1fbd5750e6b46921dfa1e44bfce4da96b891a0`.
  The inspected source diff is confined to 54 added lines in
  `rust/pokezero-search/src/encoder.rs`.
- Checkpoint: final-enthalf iteration 9375, SHA-256
  `0fd095923b4ac7e05d6e2b3ccab9c1e6869dff4893c2dae456caff10dce690be`.
- Development corpus: 16 decisions, canonical SHA-256
  `6d4be46153251e8a615275e600a9e557fb109609c9fd3111e7d087c27a6d8d11`
  (raw-file SHA-256
  `a1930e513149d39166fc8fe5e0ddbbfd509cd0d4e50f0319f8f20f29d39a6203`).
- Fixed work: native CPU model leaf, depth 2, 256 simulations, batch 16,
  four worlds, serial execution. Every accepted run reports 16,384 total
  iterations, 19,117 model evaluations, zero fallback, zero prior fallback,
  zero invalid action, and the same 16 root argmax actions.

These are warm-search timing runs: they exclude prefix replay and process startup.
They are not a serving-latency measurement or a fixed-wall comparison.

## Results

| Block order | Baseline median (s) | Candidate median (s) | Candidate minus baseline (s) | Reading |
| --- | ---: | ---: | ---: | --- |
| candidate then baseline | 9.599 | 16.996 | +7.397 | Candidate ran first. |
| candidate then baseline repeat | 6.961 | 10.712 | +3.751 | Candidate again ran first. |
| baseline then candidate order flip | 6.174 | 6.131 | -0.044 | Candidate ran second. |

The apparent candidate penalty reverses when the order reverses. The runs also
become substantially faster over the sequence, so order and warm-state effects
dominate the between-source difference. The final order-flip pair is effectively
tied: candidate median is 0.7% lower while mean and p95 are slightly higher
(7.094 vs 7.072 seconds and 11.286 vs 11.235 seconds).

The immutable terminal roots are:

- `/shared/scott-experiment/mcts-fullpath-timing-current-cd1fbd57-20260908-d2s256`
  and `/shared/scott-experiment/mcts-fullpath-timing-baseline-db09c56f-20260908-d2s256`;
- `/shared/scott-experiment/mcts-fullpath-timing-candidate-repeat-cd1fbd57-20260908-d2s256`
  and `/shared/scott-experiment/mcts-fullpath-timing-baseline-repeat-db09c56f-20260908-d2s256`;
- `/shared/scott-experiment/mcts-fullpath-timing-baseline-orderflip-db09c56f-20260908-d2s256-r1`
  and `/shared/scott-experiment/mcts-fullpath-timing-candidate-orderflip-cd1fbd57-20260908-d2s256-r1`.

An earlier order-flip root without the frozen corpus failed closed before timing
and is preserved as a diagnostic; it is not a result.

## Limits and next action

The timing artifacts prove matching work counts, root actions, fallback behavior,
and source/corpus/checkpoint provenance. They do not record a full numerical
completed-tree-value or encoded-array equality proof, so this read is not a new
semantic-parity certificate.

No follow-up cache, batch, depth, or fixed-wall experiment is justified by this
result. A future performance claim would require a newly declared randomized
order study with explicit cold/warm treatment and the missing full numerical
parity capture. That study is lower priority than the source-bound backup-repair
strength comparison.
