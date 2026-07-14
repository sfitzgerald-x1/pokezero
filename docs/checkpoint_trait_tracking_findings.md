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

## Trait ↔ foul-play win-rate correlation

Pearson r of each lineage's 500k self-play trait against its 500k foul-play win rate (n=5
lineages, win rates 0.20–0.34). The checkpoints that beat FoulPlay lean on **tempo** — Substitute
(r=+0.80), landing Focus Punches (+0.68), immunity switch-ins (+0.65), pivoting sleepers (+0.60),
meaningful Baton Pass (+0.55). Those that fare worse lean on **stall** — Rest (−0.99), Spikes
(−0.97), grinding opponent PP (−0.97), and long games (avg-turns −0.96). This matches the search
opponent: FoulPlay punishes passive stall and rewards proactive tempo. **Caveat: n=5 with a narrow
win-rate spread — directional/hypothesis-generating, not significant.** Extending foul-play across
milestones would turn this into a per-checkpoint correlation with real power.

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
