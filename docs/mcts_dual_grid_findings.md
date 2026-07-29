# The n=400 dual-purpose grid: both depth ladders are flat

**Status:** depth ladder COMPLETE and reported below. Sims ladder PARTIAL at
`s2048`/`s4096` — preliminary numbers are marked as such and must not be cited
until the cells finish.

**Date:** 2026-07-29.
**Build era (per the #960 control-reuse standard):** every shard in this campaign
was produced by one image — main @ `f88c8e9`, engine fingerprint
`814b2bd28d3983813b972ba3fd0af7fcc46871085fdea6f0e654c767d076b577`, **30
vendored patches**, vocabulary digest `b80edf4be9932c04` (1216 tokens, root ==
checkpoint == leaf, asserted in-pod per shard). One distinct fingerprint across
all shards, verified. Engine patches merged after launch (#959 explosion
residuals, #961 Kecleon world seeding, #962 stat-modifier flooring) are **not**
in these recordings and do not affect them: the arms are search-vs-raw-policy and
the build was pinned by digest at launch. Any future cell added to this grid must
be produced by the same image or reported as a separate era.

**Checkpoints.** `k0` = `v3hist-k0-enthalf-5m-20260724` @ iteration-2618
(`transition_token_budget` **0**); `k64` = `v3hist-k64-enthalf-5m-20260723` @
iteration-2657 (`transition_token_budget` **64**). Both carry vocab 1216 and
1233 trained embedding rows.

**Design.** 200 within-seed mirrored pairs per cell (seeds 8100000–8100199, one
reserved band shared by every cell so all comparisons — across depth, across
sims, and across the two checkpoints — are paired on identical battles). Each
checkpoint has **its own** raw-vs-raw control on the same seeds.

---

## 1. Headline: k64's depth slope is now FLAT, exactly like k0's

Paired per-seed pair-score delta against each checkpoint's **own** `d1`, same
seeds throughout:

| vs own d1 | k0 | k64 |
|---|---|---|
| `d2` − `d1` | +0.043 [−0.011, +0.096] | −0.048 [−0.100, +0.005] |
| `d4` − `d1` | +0.046 [−0.006, +0.099] | +0.040 [−0.013, +0.092] |
| `d6` − `d1` | +0.037 [−0.015, +0.090] | +0.031 [−0.022, +0.086] |

**Every interval contains zero, on both checkpoints.** Neither ladder slopes
down. Neither slopes up.

**This kills off-distribution synthesized history as a decay mechanism.** The
hypothesis was that k64, trained with a 64-token history budget, would degrade
with depth because deep rollouts synthesize history it never saw in training,
while k0 (budget 0) would not. On the fixed build k64 shows no downward slope at
all, and its slope is statistically indistinguishable from k0's at every depth.

Directly on the history axis — k64 minus k0, paired on seeds:

| depth | k64 − k0 |
|---|---|
| `d1` | +0.049 [−0.006, +0.102] |
| `d2` | −0.041 [−0.099, +0.018] |
| `d4` | +0.043 [−0.016, +0.104] |
| `d6` | +0.043 [−0.019, +0.105] |

All four contain zero. The history budget makes no measurable difference to
search strength at any depth tested.

## 2. Per-cell detail, depth ladder @ s1024

| cell | p1 seat | p2 seat | gap | pooled pair mean | DiD vs own control | fallback p1/p2 |
|---|---|---|---|---|---|---|
| **k0** control | 0.445 [0.378, 0.514] | 0.555 [0.486, 0.622] | −0.110 | 0.500 [0.500, 0.500] | — | 0 / 0 |
| k0 `d1` | 0.435 [0.368, 0.504] | 0.502 [0.434, 0.571] | −0.067 | 0.469 [0.429, 0.510] | +0.043 [−0.083, +0.168] | 1.83% / 0.75% |
| k0 `d2` | 0.495 [0.426, 0.564] | 0.527 [0.458, 0.596] | −0.032 | 0.511 [0.469, 0.555] | +0.077 [−0.052, +0.210] | 1.17% / 0.74% |
| k0 `d4` | 0.517 [0.449, 0.586] | 0.512 [0.444, 0.581] | +0.005 | 0.515 [0.472, 0.556] | +0.115 [−0.007, +0.237] | 0.98% / 0.78% |
| k0 `d6` | 0.510 [0.441, 0.578] | 0.502 [0.434, 0.571] | +0.008 | 0.506 [0.465, 0.547] | +0.117 [−0.005, +0.240] | 0.93% / 0.77% |
| **k64** control | 0.527 [0.458, 0.596] | 0.472 [0.404, 0.542] | +0.055 | 0.500 [0.500, 0.500] | — | 0 / 0 |
| k64 `d1` | 0.552 [0.483, 0.620] | 0.482 [0.414, 0.551] | +0.070 | 0.517 [0.480, 0.555] | +0.015 [−0.092, +0.125] | 1.08% / 1.75% |
| k64 `d2` | 0.465 [0.397, 0.534] | 0.475 [0.407, 0.544] | −0.010 | 0.470 [0.430, 0.510] | −0.065 [−0.180, +0.055] | 0.88% / 1.60% |
| k64 `d4` | 0.573 [0.503, 0.639] | 0.542 [0.473, 0.610] | +0.030 | 0.557 [0.516, 0.600] | −0.025 [−0.152, +0.102] | 0.63% / 1.57% |
| k64 `d6` | 0.565 [0.496, 0.632] | 0.532 [0.463, 0.600] | +0.032 | 0.549 [0.506, 0.593] | −0.022 [−0.150, +0.107] | 1.13% / 1.56% |

Three things worth naming:

- **Both controls are exactly 0.500 with zero-width intervals.** The within-seed
  pairing identity holds independently for each checkpoint — the same strict
  equality test as §13.3, now passed twice more.
- **The two controls tilt in OPPOSITE directions** (k0 favours the p2-slot team
  by 11 points, k64 favours p1 by 5.5) on the *same seeds*. Seat tilt is a
  property of the checkpoint's play, not of the seed block, which is why each
  checkpoint needs its own control and why a shared control would have been wrong.
- **No cell shows a significant seat residual.** Every DiD interval in the depth
  ladder contains zero, on both checkpoints. This corroborates §14's verdict that
  §13.4's +0.152 was most likely noise — now at 10 more cells.

## 3. Levels are lower than §13/§14, and that is seeds

k64 `d4-s1024` reads **0.557** [0.516, 0.600] here against **0.615** [0.572,
0.658] in §13 — same checkpoint, same cell, same build family, **different seed
block** (8100000+ vs 7800000+) and n=200 vs 220. The intervals overlap
substantially, so this is seed-block variation, not a discrepancy. It is a
reminder that absolute levels move ±6 points between 200-pair seed blocks and
that only *paired* comparisons within a block should be read finely.

## 4. Sims ladder @ d4 — PARTIAL, do not cite yet

Paired against each checkpoint's own `s512`:

| vs own s512 | k0 | k64 |
|---|---|---|
| `s1024` | +0.034 [−0.006, +0.072] (n=200) | +0.039 [−0.003, +0.081] (n=200) |
| `s2048` | +0.062 [+0.011, +0.115] (n=176 ⚠) | +0.018 [−0.056, +0.092] (n=112 ⚠) |
| `s4096` | +0.073 [+0.014, +0.130] (n=165 ⚠) | +0.025 [−0.071, +0.125] (n=70 ⚠) |

⚠ = cell incomplete. `s512` and `s1024` are complete on both checkpoints and
both are flat. The two k0 cells that exclude zero are **35% and 18% short of
their pair count**; they hint that more simulations may be weakly *positive* on
k0 (+0.06 to +0.07), which would differ from §14's flat result on k64 — but that
must not be claimed until the cells finish. §4.2's *downward* slope is already
absent on every complete cell.

**One partial cell to watch:** k64 `d4-s4096` has DiD **−0.286 [−0.514, −0.057]**
at n=70 — the only **DiD** interval in the grid excluding zero, in the direction
of p2 being *better* than p1. (The two k0 sims-slope intervals in the table above
also exclude zero; they belong to a different family. See §6.) It is the
least-complete cell in the campaign. It is recorded so it is not discovered
later, not asserted.

## 5. What this settles

- **§4.1's monotone depth decay is not a property of search.** Re-run properly on
  the fixed build, at n=400 per cell, on two checkpoints, with per-checkpoint
  controls and within-seed pairing: both depth ladders are flat.
- **Off-distribution synthesized history is not a decay mechanism.** k64 does not
  slope down where k0 doesn't; the two are indistinguishable at every depth.
- **The seat residual does not reappear.** Ten more cells, every DiD interval
  containing zero.
- **Search's gain over its own prior is real but small and does not scale with
  depth.** The best complete cells sit at 0.506–0.557 pooled against a 0.500
  control on the same seeds.

---

## 6. Pre-registered multiple-comparisons rule

**Committed 2026-07-29, while `s2048`/`s4096` were still draining and the outcome
was unknown.** At the time of writing: k0 `s2048` 352/400 games, k0 `s4096`
380/400, k64 `s2048` 240/400, k64 `s4096` 150/400 — so criterion (B) below is
decided by data that did not yet exist when the rule was fixed.

The reason for writing this before the data: the grid currently holds two
candidate "findings" pointing in opposite emotional directions. The two k0 sims
slopes point somewhere **welcome** (more search helps). The one k64 DiD cell
points somewhere **unwelcome** (a seat residual, which §14 argued was noise).
Post-hoc discretion would be tempted to bank the first and explain away the
second. This rule binds both identically.

### 6.1 Test families, enumerated

| family | tests | why that count | expected exclusions at α = 0.05 |
|---|---|---|---|
| per-cell DiD | **14** | 7 search cells × 2 checkpoints. The 2 controls are excluded: their DiD is identically 0 by construction, not an estimate | 0.70 |
| paired slopes | **12** | (3 depth + 3 sims) deltas vs the cell's own baseline, × 2 checkpoints | 0.60 |
| cross-checkpoint | **4** | k64 − k0 at d1/d2/d4/d6 | 0.20 |
| **total** | **30** | | **≈ 1.5** |

A note on the arithmetic: the campaign has 16 *cells*, but only **14** carry a
testable DiD, so the DiD family is 14 × 0.05 = **0.70** expected, not 0.80.

**Observed so far: three zero-excluding intervals against ≈1.5 expected by
chance** — and the two k0 sims slopes are not independent of each other (both are
measured against the same `s512` baseline over nested data), so the effective
count of independent exclusions is closer to two. **Exactly the order chance
predicts.** No candidate is presently distinguishable from noise.

### 6.2 The criterion, committed

`p` is obtained from the same percentile bootstrap already used throughout, as
the two-sided `p = 2 × min(P[resample ≤ 0], P[resample ≥ 0])`, on the same
resample seed (20260729) and 20 000 resamples. No other test may be substituted
after the fact.

A candidate is **ACCEPTED as an effect** only if at least one of:

- **(A) Holm.** Its `p` survives Holm–Bonferroni within its own family, at the
  family sizes fixed in §6.1 — not at a size recomputed after seeing which cells
  finished.
- **(B) Replication.** The same `(depth, sims)` cell on the **other** checkpoint
  shows the **same sign** with its own unadjusted 95% interval also excluding
  zero, both cells complete at 400 games.

Anything else is reported as **chance-compatible** and must not be called an
effect, a trend, a hint, or a signal — in either direction.

### 6.3 Applied in advance to the two live candidates

- **k0 `d4-s2048` and `d4-s4096` sims slopes** (+0.062, +0.073; welcome
  direction). Route (B) requires k64's `s2048`/`s4096` slopes to come in positive
  with intervals excluding zero. As of §4 they are +0.018 and +0.025 with
  intervals spanning zero at n=112 and n=70. **If they complete without
  excluding zero, the k0 sims slope is chance-compatible and the sims axis is
  reported as flat** — matching §14 — regardless of how clean the k0 numbers look
  on their own.
- **k64 `d4-s4096` DiD** (−0.286; unwelcome direction). Route (B) requires k0's
  `d4-s4096` DiD to be negative and exclude zero; as of §4 it is **+0.079**,
  wrong sign. Route (A) requires surviving Holm across 14 — a single marginal
  interval at the campaign's least-complete cell will not. **On present evidence
  it is chance-compatible**, and the completed cell must be re-tested against
  this same rule rather than dropped.

Whichever way the remaining shards fall, the §4 refresh adjudicates against this
block as written, and any departure from it must be argued explicitly and dated.
