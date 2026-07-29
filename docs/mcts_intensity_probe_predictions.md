# `d6-s4096` intensity probe — predictions registered BEFORE the data

**Status: predictions frozen, probe launching.** Nothing in this file may be
edited after the first shard writes. Registered 2026-07-29, against the same
build as §12 (main @ `2103d65`, engine fingerprint `7909290e…`).

## What is being adjudicated

§12.4 found a residual seat asymmetry on the fixed build: a
difference-in-differences of **+0.152 [+0.018, +0.282]** at `d4-s1024`, meaning
search still helps the p1 seat ~15 points more than the p2 seat beyond what the
seed set's team advantage explains. It named three untested candidates. This
probe re-measures the same quantity at a much higher search intensity —
`d6-s4096`, 6× the simulations and 1.5× the depth — because the three candidates
disagree about how that number should move.

## Design

Same **220 within-seed mirrored pairs, same seeds** as the `d4-s1024` acceptance
cell (7800000–7800219). Reusing the seeds is deliberate:

* the seat comparison stays within-seed (both seats, same two teams);
* the **cell** comparison also becomes paired — `d6-s4096` vs `d4-s1024` on
  identical battles, which is what "is more search worth more?" requires;
* the raw-vs-raw control is **reused unchanged**. It is depth- and
  sims-independent by construction, so re-running it would consume ~1 GPU-hour to
  reproduce the same 440 games.

The reserved `probe-d6s4096` band stays unused and available.

## Discriminating predictions

Write the number each hypothesis expects for the DiD at `d6-s4096`, against the
`d4-s1024` value of **+0.152**:

| # | candidate | mechanism | **prediction** |
|---|---|---|---|
| **H1** | opponent-side priors stay uniform (`multiply_batched_encoded_core` applies model priors to the acting seat only; the model's `opponent_action_logits` head is discarded) | the defect lives in how the tree models the *opponent*, so its influence compounds with the amount of tree that is opponent-modelled | DiD **GROWS** with intensity: **≥ +0.25**, and the interval excludes the `d4` point |
| **H2** | seat-dependent fallback | the asymmetry tracks how often each seat falls back, not search intensity as such | DiD moves **in proportion to the per-seat fallback gap**. Confirmed only if `Δ(fallback_p2 − fallback_p1)` between the two cells moves the same direction and rough magnitude. Pre-fix `d6-s4096` shards ran fallback up to 0.217, so H2 expects a **large** DiD *and* a large fallback gap |
| **H3** | belief / world-construction asymmetry | world construction is per-decision and independent of depth and sims | DiD is **FLAT**: within **[+0.05, +0.25]**, not scaling with intensity |
| **H0** | noise | +0.152 was marginal on 220 pairs (lower bound +0.018) | DiD interval **contains 0** |

H1 and H3 are separated by magnitude; H0 by whether the interval clears zero; H2
by whether the fallback gap moves with it. **Per-seat fallback rates will be
reported for both cells** — without them H2 is untestable, and it was named in
§12.4 without that evidence.

## Secondary question, same run

Is more search worth more on the fixed build? Paired per-seed delta
`d6-s4096` − `d4-s1024`. Pre-fix this comparison was meaningless: both cells
were averaging a working p1 seat with an anti-optimising p2 seat, and the pooled
number fell with intensity (§4.2) for that reason alone.

* If search is genuinely working, the paired delta should be **≥ 0**.
* A **negative** paired delta on the fixed build would be a new finding — real
  diminishing or negative returns to depth+sims, no longer explainable by the
  seat inversion, and §4.2 would need re-opening on its own terms rather than
  being written off as an artefact.

## What would retract what

* DiD collapsing to ~0 ⇒ §12.4's residual was noise; say so and close it.
* DiD growing past +0.25 ⇒ H1 is live and the opponent-prior design becomes the
  next fix, not a cleanup.
* DiD flat with a flat fallback gap ⇒ H3, and belief/world construction is the
  place to look.
