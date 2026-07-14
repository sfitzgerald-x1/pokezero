# Checkpoint trait-tracking — findings

Implementation of `docs/checkpoint_trait_tracking_plan.md`. Behavioral traits measured per
checkpoint over cumulative-games milestones across the five lineages, from self-play (all
milestones) and foul-play (500k). Every metric is derived from the omniscient Showdown protocol
log; the machinery, gates, and per-metric definitions live in `scripts/trait_*.py`.

**Data.** 54 metric sets. Self-play trajectories at every 100k milestone per lineage — 2000
games/milestone (v22-lr3m spans all 17 from 100k→1700k; m50-seq / l200-seq to 1000k; m50-ep7 to
700k; l200-ep7-wu75 to 500k). Foul-play at 500k for all five lineages, ~946–1000 games each,
FoulPlay search at 1000 ms/move. Self-play and foul-play stats are kept separate. Observation
unit is the behavioral-seat-game (self-play has two behavioral seats, foul-play one), so rates
are directly comparable across the two.

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

Pearson r of each lineage's 500k self-play trait against its 500k foul-play win rate. Foul-play
exists only at 500k, so the correlation has at most **one point per lineage** — n=5.

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

- **Foul-play is 500k-only.** Trajectories over training are self-play; the foul-play column is a
  single flagship checkpoint per lineage. Extending foul-play across milestones is gated on
  compute (FoulPlay search is ~40 s/game).
- **Rare conditionals are noisy at low volume.** rapid-spin-when-spikes-down and solar-beam-in-sun
  ride on small denominators at early checkpoints (few carriers dealt the move); read the early end
  of those trajectories as noisy. Denominators are surfaced (`carrier_rate`, category totals).
- **A per-game memory climb** (~28 MB/game, framework-side, independent of env lifetime) caps
  shards at 500 games / 24 GiB; production is sharded accordingly. Documented in `trait_eval.py`.
- **gen3-randbats coverage.** Screen-conditioned traits are absent from the pool by construction
  and are not tracked.
