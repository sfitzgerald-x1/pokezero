# Strategy-diversity fingerprints: do equal-strength checkpoints play the same game?

Status: plan for review, 2026-07-11. Agent-executable; every stage names its inputs,
outputs, validation gates, and pre-registered statistics before any compute is spent.

## The question

Five-plus checkpoints now sit in the same strength band (84–90% vs max-damage at
their 500k reads) with meaningfully different histories: recipe twins, scale rungs,
and two 200M runs that reached similar strength via radically different
trajectories. Do models that self-play into similar power levels
converge to the **same strategy**, or are these **diverse strategies of equal
strength**? The answer routes real decisions: monoculture makes refutation mining
(G4) the critical path and ensembles worthless; diversity makes population play,
pool anchors, and ensemble value heads immediately valuable.

## What "same strategy" means operationally — four layers, which can dissociate

| layer | object | same-strategy signature | diversity signature |
|---|---|---|---|
| action | policy distribution on a fixed state | high top-1 agreement, low JS divergence | structured divergence localized to state classes |
| value | position pricing on a fixed state | high Pearson between value vectors | disagreement concentrated in rare/contested states |
| matchup | pairwise game outcomes | Bradley-Terry (one strength axis) explains the win matrix; per-seed outcomes uncorrelated noise | BT residual cycles (rock-paper-scissors); systematic seed partitioning |
| style | aggregate behavior statistics | style vectors within reference spread | separable clusters (pivot rate, hazard usage, game length, conversion speed) |

Equal strength + same strategy predicts: ~50% head-to-heads AND random per-seed
outcomes AND near-coincident fingerprints. Equal strength + diverse strategies
violates at least one layer measurably.

## Checkpoint roster (pin by run id + iteration + checkpoint sha at execution time)

**Owner scope decision (2026-07-11): v2.2 schema family only, pure self-play only.**
Behavior-cloned/distilled lineages (`fpbc-*`) and other-schema checkpoints
(`emeta-obsv2-*`, v2/121 census) are excluded — this analysis asks whether *the
self-play recipe itself* converges to one strategy, so every entrant shares the
observation schema and the pure self-play reward path. A single schema family also
means all four layers apply to every pair with no carve-outs.

**Roster (5):**
1. `emetamon-v2-2-lr3m-500k-belief` — final (10M-class recipe reference, 90.3 md)
2. `metamon-m-50m-500k-seq-20260710` — final (clean 50M baseline)
3. `metamon-l-200m-500k-20260710` — final (200M, incident-annotated history)
4. `metamon-l-200m-500k-seq-20260710` — strength-matched snapshot (see below)
5. `metamon-l-200m-500k-lr15-20260710` — strength-matched snapshot

**Strength matching**: for the seq-L vs lr15 natural experiment (same architecture,
same schedule family, spiral-recovery vs clean-warmup histories — the purest test of
whether trajectory history imprints strategy), select each run's checkpoint at the
low-fi milestone nearest md = 0.85 rather than final, and record both milestones in
provenance. Everywhere else strength differences are handled by the BT fit, not by
pretending they don't exist.

**Null control (defines the verdict scale — no invented thresholds):** for each run
in the fingerprint set, also fingerprint the checkpoint **30k games earlier** (the
nearest stored iteration). Within-run adjacent pairs define the "same strategy"
reference distribution for every metric. Pre-registered verdict rule: a cross-run
pair reads **diverse** on a layer iff its distance exceeds the 95th percentile of
the within-run reference pairs on that layer; the pair verdict is *diverse* iff ≥2
layers read diverse; the fleet verdict reports the full pair × layer matrix, never
a single bit.

## Stage A — fixed-corpus fingerprints (hours; no new training)

**Corpora (fixed, neutral, provenance-stamped; never a model's own self-play
games — state-distribution shift confounds every metric):**
- the public-decision corpus (`pokezero.public_decision_corpus` schema, the
  2k+-decision band captured for the search-plan audits), and
- the Step-0 v2.2 evaluation band (120 games of external-opponent states).

Both already exist; record `data_sha256` + seed bands in the analysis manifest.

**Per checkpoint, per state**: policy distribution over legal actions, top-1 action,
value estimate. **Per pair**: top-1 agreement rate (report separately on all states
and on *contested* states — normalized policy entropy above the Step-2 profile's
median — since easy states agree trivially); mean JS divergence between full
distributions (report alongside each model's mean entropy so temperature differences
are not misread as strategy differences); Pearson between value vectors plus the
95th-percentile absolute value disagreement (localizes latent divergence).

**Style vectors, per checkpoint** (from existing machinery — the hi-fi behavior
probe and `hazard_metrics`): move-class usage (attack/status/switch), pivot rate,
hazard set rate and clear rate in hazard-legal states, mean game length, and
conversion speed (turns from first ≥80%-value position to terminal) over a fixed
1,000-game self-play probe per checkpoint with a shared seed band. Z-score across
the roster; report the full vector table and pairwise euclidean distances.

**Validation gates**: self-pair sanity (checkpoint vs itself: agreement 1.0, JS 0,
Pearson 1.0) runs first and hard-fails the stage on violation; a label-shuffled
control (states permuted between models) must read as maximally distant; the census
guard must refuse any schema-mismatched checkpoint/corpus pairing loudly (this is
the existing guard, exercised, not bypassed).

**Outputs**: `diversity-fingerprints/<analysis-id>/fingerprints.json` +
`pairwise.json` (schema `pokezero.diversity_fingerprint.v1`), one file per matrix,
plus a manifest binding checkpoint shas, corpus shas, and code version.

## Stage B — mirrored-seed round-robin (a day of cluster time)

All roster pairs (10 pairs at 5 entrants): **1,000 games per pair, shared fresh
seed band, both seats per seed** (the mirrored-seat machinery and
per-seed outcome persistence from the capstone work are reused as-is). Reserve a new
disjoint seed band via the existing seed-range reservation files; no overlap with
capstone, Step-0, or audit bands.

**Pre-registered analyses:**
1. **Bradley-Terry fit** to the pairwise win matrix. Report per-pair residuals and
   the **cycle statistic** (sum of signed residual products over all 3-cycles);
   significance by permutation of per-seed outcomes within pairs (10k permutations).
   BT explains everything → one strength axis; significant cycles → non-transitive
   strategic structure.
2. **Per-seed agreement**: for each pair, over the shared seed set played against
   each common baseline opponent AND in their mirror games, the φ coefficient of
   per-seed win/loss vectors with a McNemar check on discordant counts. High φ =
   same seeds are hard for both = behavioral similarity even at 50% aggregate; φ
   near the within-run reference = monoculture; φ near zero with equal strength =
   systematic seed partitioning = diversity at the matchup layer.
3. Ties/capped games count 0.5 and are reported separately (house rule).

**Validation gates**: seat balance exact (every seed both seats); zero fallback
rows; the mirror-match aliasing fix (the gauntlet bug) verified by asserting
self-play pairs of the same checkpoint land at 50% ± binomial CI.

## Stage C — exploiter transfer (conditional; the definitive layer)

Trigger only if Stages A and B disagree, or on explicit request: strategies differ
*meaningfully* iff something beats one but not the others. Reuse the diversity-tier
exploiter planner: one cheap exploiter arm per roster member (fixed small budget,
identical recipe), then the full exploit-transfer matrix (each exploiter vs every
roster member, 500 games mirrored). Near-diagonal matrix = diversity; uniform
transfer = monoculture. Pre-register the same within-run null control as the scale.

## Web report — one self-contained page per analysis run

`scripts/diversity_report.py` renders everything above from the artifact directory
alone (re-runnable without recomputation; no network, **no CDN** — inline CSS and
inline-SVG charts only, both light and dark schemes):

- header: roster table with run ids, milestones, checkpoint shas, corpus shas;
- four **similarity heatmaps** (action / value / matchup-φ / style), shared color
  scale anchored by the within-run null band, with the reference percentile drawn
  on the color bar;
- **dendrogram** per layer (average-linkage over the distance matrices);
- **BT panel**: fitted strengths with CIs, residual heatmap, cycle statistic with
  permutation p-value;
- **per-seed partition view**: for the most-diverse and least-diverse pairs, the
  seed-outcome contingency tables;
- **style table**: z-scored vectors with per-cell shading;
- **verdict matrix**: pair × layer, colored same/diverse/insufficient, with the
  pre-registered rule quoted verbatim in the caption.

`scripts/diversity_report.py --index` additionally writes an `index.html` linking
every analysis run found under the experiment root (id, date, roster size, verdict
summary), so successive analysis runs accumulate into a browsable history. The
private deployment may link or serve this directory from its run dashboard; that
wiring stays out of this repo.

## Execution order and budget

1. Stage A fingerprints + null controls + report: ~2–3 h wall (inference-only reads
   of ~2.2k states × 10 checkpoints (5 roster + 5 within-run nulls) + 10 ×
   1,000-game style probes).
2. Stage B round-robin: 10 pairs × 1,000 mirrored games ≈ 20k games ≈ half a day
   at modest parallelism; BT/φ analysis and report refresh: minutes.
3. Stage C only on trigger.

## Don't-do list

- No new training in Stages A/B (Stage C's exploiters are the only training, and
  only on trigger).
- Never evaluate fingerprints on a model's own self-play corpus.
- Never bypass the observation-census guard; any future cross-schema or
  behavior-cloned entrant requires a dated owner amendment to the roster.
- Any preflight/plumbing run must carry `smoke` in its id (dashboard filter
  convention).
- No threshold tuning after unblinding: the within-run null percentile rule is
  frozen at first execution; changes require a dated owner amendment.

## Verification evidence required per stage

- Stage A: self-pair and shuffled-control gate outputs embedded in the manifest;
  per-metric within-run reference distributions plotted in the report.
- Stage B: seat-balance and fallback-zero assertions in the run log; permutation
  seed recorded; raw per-seed outcome files retained.
- Report: renders byte-identically from artifacts alone (hash the HTML twice);
  opens with no network access.
