# Checkpoint trait-tracking — findings

Implementation of `docs/checkpoint_trait_tracking_plan.md`. Behavioral traits measured per
checkpoint over cumulative-games milestones across the five lineages, from self-play (every
milestone) and foul-play (500k + each lineage's frontier). Every metric is derived from the
omniscient Showdown protocol log; the machinery, gates, and per-metric definitions live in
`scripts/trait_*.py`. Regenerate any time with `scripts/trait_extract_all.sh`.

**Data.** 67 metric sets. Self-play at every 100k milestone per lineage (2000 games/milestone,
5000 at 500k) — 57 checkpoints, v22-lr3m spanning 100k→1900k. Foul-play (~950–1000 games,
FoulPlay search at 1000 ms/move) at 500k and each frontier — 10 checkpoints. Self-play and
foul-play are kept separate and never merged. The observation unit is the behavioral-seat-game
(self-play has two behavioral seats, foul-play one), so rates are comparable across the two.

## What changes over training (self-play)

Each point is one checkpoint; no aggregation.

- **Conditional move use is *learned*, not innate.** Early checkpoints fire conditional moves
  blindly; later ones gate them on the condition that makes them good:
  - *Solar Beam in sun*: **~3–20% → 96–99.7%** for lineages that train long enough (m50-seq
    4.7→96.0, v22-lr3m →99.7, m50-ep7 17.1→99.6). The short runs never develop it — this trait
    needs training length, and it is the sharpest signal in the dataset.
  - *Phazing when justified* (enemy boosted or behind a Substitute): **~0–26% → 40–59%** across
    all five lineages. Early Roar/Whirlwind is indiscriminate.
- **Early toxic-spam collapses.** v22-lr3m opens at 5.41 Toxic/seat-game at 100k and settles to
  ~2.3. Very early checkpoints lean on status as a crutch.
- **The setup/utility toolkit is picked up over training.** stat-boost, Substitute, Spikes, and
  healing (excl. Rest) all rise from near-zero at 100k and plateau.
- **The weakest checkpoints can't close games.** v22-lr3m@100k times out (stalls to the turn cap)
  in ~50% of its self-play games; its *decided* games average ~57 turns. Timeout rate falls to ~0
  by 300k. avg-turns is reported over decided games only, with timeout rate as its own trajectory —
  a checkpoint that cannot win is a distinct failure mode from one that wins slowly.

## Strength vs FoulPlay improves with training

| lineage | 500k | frontier | |
|---|---|---|---|
| m50-ep7 | 0.198 | **0.353** @1000k | |
| v22-lr3m | 0.307 | **0.420** @1900k | best at frontier |
| m50-seq | 0.339 | **0.392** @1000k | |
| l200-seq | 0.317 | **0.347** @1000k | |
| l200-ep7-wu75 | 0.319 | 0.295 @800k | **not significant** (z=1.13) |

l200-ep7-wu75 is the one lineage that appears to get *worse*. It does not survive a test: a
two-proportion z-test on 1000 vs 978 games gives **z=1.13** (needs |z|>1.96 at α=0.05). Do not
report it as a regression — it is noise.

## Trait ↔ win correlation — read the per-game charts, not the aggregate

Two correlations are rendered, and they disagree. **The per-game one is the trustworthy one.**

### Per-game (n = games) — the real answer

For each game: x = the seat's count of the trait *in that game*, y = 1 if that seat won.
Point-biserial across games, computed *within* each checkpoint, then aggregated by reporting the
mean r and the min..max range across checkpoints. Self-play is a **paired design** — both seats
are the same policy in the same game, so comparing winner against loser holds policy strength and
game length fixed by construction (a game-level quantity has no within-game variance and correctly
falls out at r≈0). 57 self-play checkpoints / 254,678 decided seat-games; 10 foul-play / 9,736.

**Headline: per-game behavior barely predicts winning.** Every effect is |r| ≤ 0.10 — under ~1% of
outcome variance. Only these are sign-consistent across *every* checkpoint (consistency across
independent checkpoints is the evidence, not any single r):

| trait | self-play (57 ckpts) | vs FoulPlay (10 ckpts) |
|---|---|---|
| Substitute | **−0.103** (all 57) | **−0.087** (all 10) |
| healing (excl Rest) | **−0.070** (all 57) | −0.050 (not consistent) |
| immunity switch-in | +0.042 (not consistent) | **+0.068** (all 10) |
| phaze when justified | — | **−0.026** (all 10) |

Substitute and healing tracking *losses* is almost certainly **reverse causality** — you Substitute
and heal when you are behind — not evidence that they lose games. Read these as association within
game context. The clean version would be a per-decision propensity analysis, which is a much
bigger lift.

**A trap worth recording:** the first cut of this chart ranked `forced_switch` first at r≈−0.65,
sign-consistent everywhere. It is circular: a forced switch happens *because your mon fainted*, so
its count is essentially "mons lost" and correlating it with losing restates the outcome. It is now
excluded, with a test (`test_outcome_definitional_traits_excluded`) so it cannot creep back. Any
trait added to `PER_GAME_TRAITS` must be a *chosen* behavior.

### Aggregate (n = checkpoints) — kept, but weak

One point per checkpoint: the trait and win rate are both aggregated over that checkpoint's
foul-play games (same population). n=10 across a 0.198–0.420 win-rate spread. It is **confounded by
overall checkpoint strength** — better models do more of everything effective *and* win more — so
it largely measures "this trait tracks being a stronger model", not any contribution to winning.
Its instability is documented: earlier cuts of this same chart (n=5, then n=4 with m50-ep7 held
out, then switching the trait source from self-play to foul-play games) produced *sign flips* on
`focus punch success`, `sleeping mon out`, and `spikes`. Treat it as descriptive only; where it and
the per-game chart disagree, believe the per-game chart.

`CORR_EXCLUDE_LINEAGES` is now empty: m50-ep7 was held out while foul-play existed only at 500k
(one point at 0.198 against a 0.31–0.34 cluster = high leverage at n=5). With foul-play at each
frontier its two points sit on a continuum, so excluding it would only discard win-rate spread.

## Per-trait conditional definitions

- **explosion/self-destruct**: one combined "boom" category (not split by move).
- **focus punch success rate**: landed (not disrupted) / total bot attempts.
- **opp focus-punch disrupted**: opponent Focus Punches the bot broke / opponent attempts — vs
  FoulPlay the bot lands ~90% of its own Focus Punches but disrupts only ~12% of FoulPlay's.
- **BP w/ stat or sub**: Baton Passes that actually carry a stat boost or Substitute (checked on
  the outgoing mon at pass time), not every BP switch.
- **win rate** is only reported where there is a real opponent. In self-play the bot drives both
  seats, so p1's rate is ~0.5 by construction and is omitted.

## Caveats

- **Foul-play covers 500k + frontier, not every milestone.** Trajectories over training are
  self-play. Extending foul-play to every milestone is gated on compute (~40 s/game).
- **Rare conditionals are noisy at low volume.** rapid-spin-when-spikes-down and solar-beam-in-sun
  ride on small denominators at early checkpoints. Denominators are surfaced (`carrier_rate`,
  category totals).
- **A per-game memory climb** (~28 MB/game, framework-side, independent of env lifetime) caps
  shards at 500 games / 24 GiB; production is sharded accordingly. Documented in `trait_eval.py`.
- **gen3-randbats coverage.** Screen-conditioned traits are absent from the pool by construction
  and are not tracked.
