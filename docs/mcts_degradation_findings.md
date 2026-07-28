# MCTS degradation: why search loses to its own prior

**Status:** cause localized. Search logic and belief aggregation are exonerated;
the evidence points at dynamics-model divergence compounding over rollout depth.

**Date:** 2026-07-28
**Checkpoint:** `v3hist-k64-enthalf-5m-20260723` @ `iteration-2657` (= 4.25M games)
**Eval build:** Python `046f58f`, crate image `mcts-eval-crate-20260726d`

---

## 1. Verdict

Engine MCTS scores **0.27–0.36 head-to-head against the raw policy it is built
on**, where 0.500 is the null. It is not marginally weaker; it is 14–23 points
below the policy whose decisions it is supposed to improve.

Three ablations, each on the same 100 seeds, localize the cause:

| axis | finding | implication |
|---|---|---|
| **depth** | 0.530 → 0.450 → 0.360 as depth goes 1 → 2 → 6 | **monotone decay with rollout depth** |
| **sims** | 0.360 → 0.310 → 0.270 as sims go 1024 → 2048 → 8192 | more search into the same model is worse |
| **worlds** | flat 0.354–0.427 across 1 → 16 worlds | belief aggregation is not the cause |

At depth 1 search is statistically indistinguishable from the raw policy and
from a raw-vs-raw control. Every additional ply of simulated dynamics costs
strength. That is the signature of an accurate policy being rolled forward
through inaccurate physics.

---

## 2. Provenance (established before any number was interpreted)

The eval ran Python commit **`046f58f`**, verified by simultaneous md5 match on
`engine_search.py`, `engine_world.py` and `showdown.py` — a three-way exact
match, not an assumption.

- `046f58f` descends from `29a0a6c`, the merge of #871 (fallback elimination).
- It carries the encoder-tables latch (`_latch_encoder_tables_to_model_config`).
- The only commit between #871 and the eval build is #872 (Ditto Transform).
- Crate value mapping `values01 = 0.5 * (v + 1.0)` confirmed correct
  (`model.rs:342`).

**Process weakness worth fixing:** the build was delivered as a working-tree tar
staged onto a PYTHONPATH overlay, not a tagged image. It happened to match a
commit byte-for-byte, but nothing enforced that. Overlays should stamp their
SHA into the artifact.

---

## 3. Harness validation

Before treating 0.500 as the null, it was measured:

| arm | n | score | Wilson 95% |
|---|---|---|---|
| **raw vs raw (control)** | 100 | **0.520** | [0.423, 0.615] |

The interval contains 0.500. Seats are mirrored by seed parity (`seed % 2`), so
seat advantage cancels. No harness bias; the deficits below are real.

---

## 4. Results

All cells: `v3hist-k64` @ 4.25M, batch 64, seeds 600000–600099, mirrored seats,
head-to-head vs the same checkpoint's raw policy. n = 100 unless noted.

### 4.1 Depth ablation — the discriminating experiment

| cell | score | Wilson 95% | s/decision |
|---|---|---|---|
| control (raw v raw) | 0.520 | [0.423, 0.615] | ~0.005 |
| **d1**-s1024-w4 | **0.530** | [0.433, 0.625] | 0.29 |
| **d2**-s1024-w4 | 0.450 | [0.356, 0.548] | 2.03 |
| **d6**-s1024-w4 | 0.360 | [0.273, 0.458] | 4.53 |

Depth 1 is at parity with both raw and the control. Score then falls
monotonically with depth. At depth 1 the engine's dynamics barely participate —
search is essentially a one-ply policy lookahead — and it performs like the
policy. The deficit appears only as the planner rolls forward.

### 4.2 Simulation scaling

| cell | score | Wilson 95% | s/decision |
|---|---|---|---|
| d6-**s1024** | 0.360 | [0.273, 0.458] | 4.53 |
| d6-**s2048** | 0.360 | [0.273, 0.458] | 8.16 |
| d6-**s4096** | 0.290 | [0.210, 0.385] | 15.83 |
| d4-**s2048** | 0.310 | [0.228, 0.406] | 7.52 |
| d4-**s4096** | 0.270 | [0.193, 0.364] | 13.82 |
| d4-**s8192** | 0.270 | [0.193, 0.364] | 22.72 |

An 8× simulation budget costs ~5× wall time and *loses* strength. Consistent
with the depth result: more compute spent inside a divergent model buys more
confidence in the wrong answer.

### 4.3 Depth past 6 is inert

`d6-s1024` and `d8-s1024` produced **identical per-seed outcomes** — same 36
wins, same winner on every seed. Two extra plies changed no decision. Either the
tree does not reach depth 6, or the cap is not binding. Worth a separate look;
it means the depth grid above effectively saturates at 6.

### 4.4 World-count ablation — refutes the aggregation hypothesis

| worlds | n | score | Wilson 95% | s/decision |
|---|---|---|---|---|
| 1 | 100 | 0.360 | [0.273, 0.458] | 1.20 |
| 2 | 100 | 0.360 | [0.273, 0.458] | 2.28 |
| 4 | 100 | 0.360 | [0.273, 0.458] | 4.53 |
| 8 | 96 | 0.354 | [0.266, 0.454] | 8.97 |
| 16 | 96 | 0.427 | [0.333, 0.527] | 17.33 |

Flat across a 16× range. **`w1` is decisive**: with one determinization the
cross-world vote is a no-op — `aggregated` receives a single world's visit
shares and the argmax is that world's own root choice — yet search is still 14
points below parity.

The identical 0.360 at w1/w2/w4 is coincidence, not an inert knob: w1 and w4
disagree on 20 of 100 seeds, w2 and w4 on 12. Different games won, same count.

**A prior hypothesis of mine — that strategy fusion in the visit-share vote
across determinizations caused the deficit — is refuted by this table.** It was
argued repeatedly before being tested. The plausibility-weighted-mean redesign
remains worth doing, but as a cleanup, not as the fix.

---

## 5. What is exonerated, what remains

**Exonerated by evidence:**

- *Selection / backup logic* — d1 at parity. A bug in PUCT, virtual loss, or
  backup would corrupt depth 1 as much as depth 6.
- *Leaf value orientation* — same argument. A perspective flip would break d1.
  (The mirrored-state crate test is still worth adding as a cheap guard; it is
  expected to pass and passing is not informative.)
- *Cross-world aggregation* — the w1 cell.
- *Fallback contamination* — 0.4–1.0% fallback across the grid, far too small to
  move a 14-point deficit.

**Implicated:** dynamics-model divergence between the vendored poke-engine and
the Showdown sim the policy was trained on. Three confirmed gen3 divergences,
none currently patched:

1. Rapid Spin hazard-clear leaks through Protect — `remove_effects_for_protect`
   (`choices.rs:20600`) clears effect fields but not `move_id`, while the hazard
   clear (`gen3/choice_effects.rs:343`) keys on `move_id`.
2. Leech Seed removal on switch unimplemented in gen3.
3. Partial-trap (Wrap-class) removal on switch unimplemented.

Leech Seed alone appears ~2–2.6×/game when carried. Over a 6-ply rollout the
exposure is large, and model error dominating policy error is sufficient on its
own to produce this table.

---

## 6. Fallback status

Fallback is no longer a confound, but is not zero. Across the grid: **0.4–1.0%**
of decisions. Residual causes, ranked:

| cause | share | note |
|---|---|---|
| `self_moveset_mismatch` (Transform, **our own** Ditto) | dominant | #872 fixed the opponent's Ditto; ours still desyncs |
| `trapped` (Mean Look / Spider Web) | ~380 | engine models no move-trap; needs an engine patch |
| `materialization_blocker: baton-pass` | 204 | fixed on `scott/baton-pass-and-trap`, not in this build |
| `belief_sample: opponent switch constraints inconsistent` | small | new, uninvestigated |
| `perish0` | small | Showdown's final tick has no engine counterpart |

---

## 7. Retracted claims

Recorded so they are not re-cited:

- **"FoulPlay results point the same way."** Withdrawn. The raw control arm
  alone moved 0.50 → 0.35 between two n=20 campaigns. That swing exceeds the
  effect being claimed; n=20 FoulPlay cells carry no direction.
- **Two 0/20 FoulPlay cells** (`d4-s2048`, `d4-s512`). Void — the bridge's
  `isinstance` gate handed engine-MCTS a `None` materialization state, so it
  played uniform-random every decision.
- **Strategy-fusion hypothesis.** Refuted by §4.4.

---

## 8. Next steps

1. **Patch the three gen3 divergences**, re-vendor, re-run the Track-C fidelity
   differential to quantify residual divergence.
2. **Re-bench the grid** on the patched engine. The prediction that makes this
   falsifiable: if dynamics divergence is the cause, the *depth decay should
   flatten* — d6 should rise toward d1, not merely shift upward.
3. Close the own-side Transform desync (largest fallback residue).
4. Cleanups, after the regression is resolved: plausibility-weighted mean value
   across worlds instead of visit-share vote; investigate why depth > 6 is inert.

**Acceptance:** search ≥ 0.500 vs raw at d4/s1024 over 200 mirrored-pair games,
on a build whose SHA is recorded in the report.
