# No-belief foundation baseline

This document records the foundation checkpoint family trained before the belief-input bug was
identified. Keep these results: they are still useful baseline data for comparing a fixed
belief-input run against the same recipe.

## What no-belief means

The `pokezero-no-belief-*` checkpoints were trained with the normal public battle state, but the
belief-derived opponent set features were unavailable to the model. In practice, the policy was blind
to:

- candidate opponent movesets from the Gen 3 randbat universe
- possible opponent items
- hidden-set branches that should remain possible from public evidence

This does not invalidate the logged evals. It changes the interpretation: these rows measure the
recipe with no effective belief over opponent sets/items, so future fixed-belief runs should be
compared against this family rather than mixed into the same curve.

## Curated checkpoints

| Checkpoint | Run | Games | Parent | Notes |
|---|---|---:|---|---|
| `pokezero-no-belief-gen3-500k` | `foundation-500k-20260629192858` | 500,800 | — | First recipe-faithful no-belief foundation checkpoint. |
| `pokezero-no-belief-gen3-1m` | `foundation-1m-20260630020847` | 1,000,000 | `pokezero-no-belief-gen3-500k` | Continuation of the 500k no-belief line. |
| `pokezero-no-belief-gen3-1-5m` | `foundation-2m-20260630171151` | 1,500,800 | `pokezero-no-belief-gen3-1m` | Continuation paused after the input bug was identified. |

## High-fidelity evals

Standard high-fidelity opponents use 2,000 mirrored games per matchup. Foul-play uses 1,000 direct
games. The 500k row has a high-fidelity foul-play read plus the final standard mirrored yardstick
snapshot for the other opponents; 1M and 1.5M have complete high-fidelity rows for all four
opponents.

| Checkpoint | Random-legal | Simple-legal | Max-damage | Foul-play | Source |
|---|---:|---:|---:|---:|---|
| `pokezero-no-belief-gen3-500k` | 592 / 600 (98.7%) | 554 / 600 (92.3%) | 309 / 600 (51.5%) | 37 / 1000 (3.7%) | Final standard yardstick plus high-fidelity foul-play |
| `pokezero-no-belief-gen3-1m` | 1979 / 2000 (99.0%) | 1901 / 2000 (95.0%) | 1245 / 2000 (62.2%) | 56 / 1000 (5.6%) | High-fidelity |
| `pokezero-no-belief-gen3-1-5m` | 1990 / 2000 (99.5%) | 1925 / 2000 (96.2%) | 1458 / 2000 (72.9%) | 87 / 1000 (8.7%) | High-fidelity |

## Comparison guidance

Use this family as the no-belief baseline when evaluating the fixed belief-input recipe. A fixed run
should be judged by whether it improves the same high-fidelity curves, especially max-damage and
foul-play, without regressing the saturated random/simple baselines.

Do not merge fixed-belief checkpoints into this series. Start a separately named family so plots and
tables can compare `no-belief` versus `belief-fixed` directly.
