# Checkpoint trait-tracking — findings

Implementation of `docs/checkpoint_trait_tracking_plan.md`. Behavioral traits measured per
checkpoint over cumulative-games milestones across the five lineages, from self-play (every
milestone) and foul-play (selected checkpoints). Every metric is derived from the omniscient
Showdown protocol log; the machinery, gates, and per-metric definitions live in `scripts/trait_*.py`.

> **Status: the numbers below are a snapshot and are being refreshed.** The lineages are still
> training, and a foul-play run on each lineage's frontier checkpoint is in flight. The *machinery*
> is final; the *figures* here trail it. Regenerate any time with `scripts/trait_extract_all.sh`,
> which re-extracts every (lineage, milestone, opponent) present and rebuilds the report. Treat
> specific values as illustrative until this banner is removed.

**Data.** Self-play trajectories at every 100k milestone per lineage (2000 games/milestone; 5000 at
500k), and foul-play (~1000 games, FoulPlay search at 1000 ms/move) at 500k plus each lineage's
frontier checkpoint. Self-play and foul-play stats are kept separate and never merged. The
observation unit is the behavioral-seat-game (self-play has two behavioral seats, foul-play one),
so rates are directly comparable across the two.

## What changes over training (self-play)

These are the clearest per-checkpoint trajectories; each point is one checkpoint, no aggregation.

- **Conditional move use is *learned*, not innate.** Early checkpoints fire conditional moves
  blindly; later ones gate them on the condition that makes them good:
  - *Solar Beam in sun*: **~3–20% → 96–99.7%** for the lineages that train long enough
    (m50-seq 4.7→96.0, v22-lr3m →99.7, m50-ep7 17.1→99.6). The two short runs never develop it
    (l200-ep7-wu75 20→25 at 500k; l200-seq only reaches 40.6 by 1M) — this trait needs training
    length, and it is the sharpest signal in the dataset.
  - *Phazing when justified* (enemy boosted or behind a Substitute): **~0–26% → 40–59%** across
    all five lineages. Early Roar/Whirlwind is indiscriminate; the fraction aimed at a boosted /
    subbed enemy roughly doubles-to-triples over training.
- **Early toxic-spam collapses.** v22-lr3m opens at **5.41 Toxic/seat-game at 100k** and settles
  to **2.28**; the effect is milder elsewhere. Very early checkpoints lean on status as a crutch.
- **The setup/utility toolkit is picked up over training.** stat-boost, Substitute, Spikes, and
  healing (excl. Rest) all rise from near-zero at 100k and plateau — early policies barely use
  them, later ones fold them in.
- **v22-lr3m is the frontier outlier.** The 3M-LR lineage keeps climbing on Sleep-move and Spikes
  usage past 1M games where the others have plateaued or stopped.
- **The weakest checkpoints can't close games.** v22-lr3m@100k times out (stalls to the turn cap)
  in **~50%** of its self-play games; its *decided* games average ~57 turns. Timeout rate falls to
  ~0 by 300k. avg-turns is reported over decided games only, with the timeout rate as its own
  trajectory (a checkpoint that can't win is a distinct failure mode from one that wins slowly).

## Trait ↔ foul-play win-rate correlation — currently underpowered, do not interpret

Pearson r of each checkpoint's trait against its foul-play win rate, **both measured on the same
foul-play games**. One point per *checkpoint*: each checkpoint's ~1000 foul-play games collapse to
a single (trait, win-rate) pair, so n is the number of checkpoints with foul-play data — game
volume buys precision within a point, not more points. It is an aggregate correlation confounded
by overall checkpoint strength (better models do more of everything effective *and* win more), so
it reads as association, never cause.

**This correlation does not currently carry signal, and we can demonstrate it.** With all five
lineages, m50-ep7 (foul-play win rate 0.198) sits far below the other four (0.308–0.339), so it is
a single high-leverage point and the fit largely reduces to "m50-ep7 vs the rest". Holding it out
leaves n=4 over a **0.031** win-rate spread — and the r values *invert*:

| trait | n=5 | n=4 (m50-ep7 held out) |
|---|---|---|
| focus punch success % | +0.68 | **−0.79** |
| sleeping mon out | +0.60 | **−0.75** |
| spikes | −0.97 | **+0.22** |

Sign flips of that size from removing one point mean the estimator is fitting sampling noise, not
a behavioral relationship. Neither ranking should be quoted. The report renders the chart with
`m50-ep7` held out (`CORR_EXCLUDE_LINEAGES`) and a low-power warning; the machinery is correct and
the numbers are real, there is simply not enough variance on the win-rate axis to correlate
against.

**What would fix it:** foul-play at every milestone. That turns each lineage's single point into a
trajectory (≈49 checkpoints spanning weak→strong, so a wide win-rate axis), making this a
per-checkpoint correlation with actual power. That run is gated on compute (FoulPlay search is
~40 s/game). Until then, treat this section as machinery-ready and evidence-empty.

## New per-trait conditional definitions (as of the review pass)

- **explosion/self-destruct**: one combined "boom" category (not split by move).
- **focus punch success rate**: landed (not disrupted) / total bot attempts.
- **opp focus-punch disrupted**: opponent Focus Punches the bot broke / opponent attempts — in
  foul-play the bot lands ~90% of its own Focus Punches but disrupts only ~12% of FoulPlay's.
- **BP w/ stat or sub**: Baton Passes that actually carry a stat boost or Substitute (checked on
  the outgoing mon at pass time), not every BP switch.

## Foul-play (500k)

Against FoulPlay search (1000 ms/move) the 500k checkpoints win **20–34%** (m50-ep7 0.198,
v22-lr3m 0.308, l200-ep7-wu75 0.319, l200-seq 0.317, m50-seq 0.339). This is the expected gap: a
depth-searching opponent beats a raw 500k policy. The value here is behavioral, not the win rate —
the foul-play panel shows how each checkpoint's move/switch/resource profile shifts under pressure
from a strong searcher versus against itself (kept in a separate column, never merged).

## Caveats

- **Foul-play covers selected checkpoints, not every milestone.** Trajectories over training are
  self-play; foul-play runs at 500k and each lineage's frontier. Extending it to *every* milestone
  — which is what would give the correlation real power — is gated on compute (FoulPlay search is
  ~40 s/game).
- **Rare conditionals are noisy at low volume.** rapid-spin-when-spikes-down and solar-beam-in-sun
  ride on small denominators at early checkpoints (few carriers dealt the move); read the early end
  of those trajectories as noisy. Denominators are surfaced (`carrier_rate`, category totals).
- **A per-game memory climb** (~28 MB/game, framework-side, independent of env lifetime) caps
  shards at 500 games / 24 GiB; production is sharded accordingly. Documented in `trait_eval.py`.
- **gen3-randbats coverage.** Screen-conditioned traits are absent from the pool by construction
  and are not tracked.
