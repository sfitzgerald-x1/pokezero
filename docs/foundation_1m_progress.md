# Foundation 1M continuation progress

Status: live progress record for the recipe-fidelity continuation from the completed
**500,800-game** foundation checkpoint toward **1,000,000 self-play games**.

This note records evaluation evidence only. It intentionally omits private operational details.

## Evaluation cadence

- The 500k anchor row comes from [`foundation_500k_results.md`](foundation_500k_results.md).
- Continuation rows use the active 1M run's scheduled readouts.
- Continuation non-foul-play opponents are mirrored **400-game** aggregate reads.
- Continuation foul-play is a **100-game** async read at the same scheduled milestone.
- Larger independent 1,000-game reads are reserved for 50k milestones and should be added below as
  they complete.

## Standard 10k Progress

| Total self-play games | Checkpoint | Random-legal | Simple-legal | Max-damage | Foul-play |
|---:|---|---:|---:|---:|---:|
| 500,800 | 500k anchor, iteration 313 | 592 / 600 (98.7%) | 554 / 600 (92.3%) | 309 / 600 (51.5%) | 37 / 1000 (3.7%) |
| 502,400 | continuation iteration 1 | 396 / 400 (99.0%) | 352 / 400 (88.0%) | 213 / 400 (53.2%) | 5 / 100 (5.0%) |
| 510,400 | continuation iteration 6 | 390 / 400 (97.5%) | 364 / 400 (91.0%) | 220 / 400 (55.0%) | 2 / 100 (2.0%) |

The 502,400 row is the first checkpoint after resuming from the 500,800-game model. It is useful as
an initial continuation baseline, but it is closer to a startup read than a regular 10k interval.
The foul-play anchor is a higher-fidelity 1,000-game read; continuation foul-play rows are the
scheduled 100-game milestone reads.

## Current Readout

The early continuation rows are consistent with the 500k interpretation: max-damage remains a
healthy non-collapsed signal, while foul-play remains a harder downstream benchmark with expected
low early win rates. The next meaningful check is whether max-damage continues to hold or improve as
the run crosses later 10k thresholds, and whether the higher-fidelity 50k reads show the same trend.

## Next Updates

- Add the next scheduled 10k read once the run crosses the next threshold.
- At the first continuation 50k boundary, generate the current 10k trend plot and launch the
  independent 1,000-game high-fidelity reads for random-legal, simple-legal, max-damage, and
  foul-play.
