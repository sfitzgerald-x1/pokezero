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

### Matchup layer — round-robin (Bradley-Terry + intransitivity)
_[filled when the round-robin lands: BT strengths, win matrix, and whether the outcomes are
non-transitive (rock-paper-scissors ⇒ strategic diversity the strength axis can't explain) or
transitive (a single strength axis)._]

## The matched-strength natural experiment (seqL200m vs lr15L200m)

The purest test in the roster: identical 200M architecture and schedule family, both pinned at
md≈0.85, but radically different training histories (seq-L spiralled into passivity and
recovered; lr15 warmed up cleanly at half peak LR). If trajectory history imprints strategy,
these two should differ. **Action layer: diverse (JS 0.134). Value layer: diverse (Pearson 0.83).
Style layer: same.** So the two histories produce policies that *choose differently and value
positions differently* while landing on *similar aggregate playstyle* — trajectory history
imprints the fine-grained policy, not the coarse behavioral signature.

## Preliminary verdict (Stages A complete; matchup pending)

On the fixed-corpus fingerprint layers (action + value), these equal-strength self-play
checkpoints are **diverse, not a single strategy** — every cross-run pair diverges beyond the
within-run drift baseline, with the recipe family clustering tightest. Style is a weaker,
noisier signal that partly agrees. The matchup layer (below) tests whether that fingerprint
diversity manifests as game-theoretic non-transitivity or collapses onto one strength axis —
the decisive read for whether a population buys anything over a single checkpoint.

Artifacts: `/shared/diversity-fingerprints/diversity-20260711a/` (per-checkpoint fingerprints,
pairwise.json, style.json, matchup.json, report.html). Regenerate the report with
`scripts/diversity_report.py --analysis-dir <dir>`.
