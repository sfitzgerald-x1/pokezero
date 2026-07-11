# Strategy-diversity fingerprints — findings (analysis diversity-20260711a)

Executes `docs/diversity_fingerprint_plan.md` on the five v2.2 pure-self-play checkpoints.
Question: do equal-strength self-play checkpoints converge to the **same strategy**, or are
they **diverse strategies of equal strength**?

## Roster (pinned, md5-provenanced)

| label | run | iteration | role | md (at pin) |
|---|---|---|---|---|
| ref10m | foundation-emetamon-v2-2-lr3m-500k-belief | 313 (final) | 10M recipe reference | 0.903 |
| clean50m | metamon-m-50m-500k-seq-20260710 | 313 (final) | clean 50M baseline | 0.872 |
| orig200m | metamon-l-200m-500k-20260710 | 313 (final) | 200M (incident-annotated) | 0.873 |
| seqL200m | metamon-l-200m-500k-seq-20260710 | 168 (md≈0.85 snapshot) | 200M spiral-recovery | 0.853 |
| lr15L200m | metamon-l-200m-500k-lr15-20260710 | 162 (md≈0.85 snapshot) | 200M clean-warmup | 0.845 |

Each roster checkpoint is paired with its own within-run checkpoint ~30k games earlier
(the "null control"); the 5 within-run distances define the "same strategy" reference
distribution, and a cross-run pair reads **diverse** on a layer only if it exceeds the
95th percentile of that within-run band.

## Gates (all passed)

- **Census guard**: all 10 checkpoints census-155; the observation-schema guard is exercised
  (a mismatch would fail the forward loudly), not bypassed.
- **Determinism**: re-extraction of ref10m gives 0/200 top-1 flips; the JS noise floor is
  1.8e-8, so the cross-run signal (JS 0.09–0.14) is ~6×10⁶ times the noise.
- **Shuffled-label control**: misaligning decisions inflates both top-1 disagreement and JS
  (as required) — the metrics respond to real alignment, not artifacts.

## Layer results

### Action layer — DIVERSE (all 10 cross-run pairs)
Over 6,613 shared external-opponent decisions. Within-run null p95: top-1 disagreement 0.288,
policy JS 0.051. Every cross-run pair exceeds both (top-1 0.35–0.45, JS 0.09–0.14) — 2–3× the
within-run reference. The recipe-family finals are closest (clean50m↔ref10m JS 0.092);
pairs involving the md≈0.85 200M snapshots are farthest.

### Value layer — DIVERSE (all 10 cross-run pairs)
Within-run null p95: 1−Pearson 0.059. Cross-run 1−Pearson 0.09–0.24 (value Pearson 0.76–0.91).
Positions are priced measurably differently across runs, most similarly within the recipe family
(orig200m↔ref10m Pearson 0.91).

### Style layer — MIXED (behavioral aggregates, 500-game self-play probes)
Features: attack/status/setup/heal/hazard/clear/phaze move-class rates, avg game length,
pivot rate, pivots/game, forced-switches/game, distinct moves. z-scored; euclidean distance.
About half the cross-run pairs exceed the within-run null (p95 4.83 in z-units); the rest are
within it. **Caveat**: at 500 games the within-run null carries sampling noise, inflating the
threshold and making style verdicts conservative. Notable dissociation: **seqL200m↔lr15L200m
read style-SAME despite being action/value-DIVERSE** — the two matched-strength 200M runs
reach similar aggregate playstyle via different specific decisions.

### Matchup layer — TRANSITIVE, NOT diverse (round-robin, 150 games/matchup seat-balanced)
Neural round-robin over all pairs (benchmark auto-plays both seats). Bradley-Terry fit:
**ref10m +0.22 ≈ clean50m +0.20 > orig200m +0.05 > lr15L200m −0.19 > seqL200m −0.41.**
A single latent strength axis explains **every** head-to-head to within ~1% (all residuals
≤ 0.019). The directed-3-cycle intransitivity statistic is **0.0000**, versus a bootstrap null
mean of 0.0000 (p = 1.000): **no rock-paper-scissors structure, no non-transitivity.** The win
matrix is exactly what one strength dimension predicts — there is no matchup-layer diversity.

## The matched-strength natural experiment (seqL200m vs lr15L200m)

The purest test in the roster: identical 200M architecture and schedule family, both pinned at
md≈0.85, but radically different training histories (seq-L spiralled into passivity and
recovered; lr15 warmed up cleanly at half peak LR). **Action: diverse (JS 0.134). Value: diverse
(Pearson 0.83). Style: same. Matchup: same — lr15 beats seq-L 53.7%, exactly its Bradley-Terry
prediction (0.556) from a −0.19 vs −0.41 strength gap.** The two histories produce policies that
*choose and value positions differently* but sit on the *same transitive strength axis* with
*similar aggregate playstyle*. Trajectory history imprints fine-grained policy noise, not a
distinct, exploitable strategy.

## Verdict

**These equal-strength v2.2 self-play checkpoints are the SAME strategy with diverse
micro-behavior — they are NOT diverse strategies of equal strength.**

The four layers agree on a single coherent picture:
- **Action + value: diverse everywhere.** Every cross-run pair picks different moves and prices
  positions differently, 2–3× beyond the within-run training-drift baseline.
- **Style: mixed/noisy.** Aggregate behavioral signatures partly overlap; the 500-game null
  band is sampling-noise-limited.
- **Matchup: perfectly transitive, zero cycles.** One strength axis explains all head-to-heads;
  no rock-paper-scissors.

The reconciliation: the fingerprint differences are **variation around a common strategy —
different stochastic paths up the same hill, or strategically-equivalent local optima — not
distinct viable strategies.** If the runs were genuinely diverse strategies, the matchup layer
would show non-transitivity (something that beats A but loses to B); it shows none. Micro-level
divergence in specific choices does not create macro-level strategic structure.

### What this routes
- **Monoculture confirmed at the game-theoretic level.** A population/ensemble of these
  checkpoints buys little in matchup terms — there is no non-transitivity to exploit and no
  matchup-aware advantage to capture. Pool anchors and ensemble value heads gain only whatever
  the (real but strategically-inert) action/value variance provides.
- **Refutation mining (G4) is the critical path**, not passive ensembling: genuine strategic
  diversity has to be *manufactured* (actively finding and training against refutations),
  because self-play alone converges to one strategy here.
- **The shared blind spot follows.** One strategy means one set of blind spots — consistent with
  the hazard-mispricing (ΔV) signature seen across the family.

### Caveats
- lr15/seq-L are mid-training md≈0.85 snapshots (strength-matched but less mature than the
  finals); part of their action/value distance from the finals reflects maturity. Their matchup
  transitivity and the seq-L↔lr15 comparison are clean regardless.
- The within-run null (roster vs its own self 30k games earlier) folds in genuine training
  drift, so it is a *conservative* "same-strategy" bar; cross-run pairs exceed it anyway on
  action/value.
- n=5 roster ⇒ 10 triples for cycle detection — modest, but the transitivity is unambiguous
  (every residual < 2%). Foul-play/human-play opponents are out of scope (this asks whether the
  recipe converges, not how it fares vs external strategies).
- Style precision is 500-game-limited; a 1,000-game re-probe would sharpen that layer only.

Artifacts: `/shared/diversity-fingerprints/diversity-20260711a/` (per-checkpoint fingerprints,
pairwise.json, style.json, matchup.json, report.html). Regenerate the report with
`scripts/diversity_report.py --analysis-dir <dir>`.
