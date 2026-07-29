# MCTS degradation: why search loses to its own prior

**Status:** mechanism **IDENTIFIED** (§10–§11), pending the §8 acceptance
re-bench. The leaf value was consumed with the wrong seat orientation whenever
PokeZero played p2, and the per-seat split of the recorded grids (§11) shows the
p2 seat carries essentially the whole deficit at every depth and every
simulation budget, on both builds. Four candidate causes were tested before it;
three are refuted, including this document's own original conclusion.

**Every strength number in §4, §7 and §9 was produced by an engine that played
one of its two seats backwards. Treat them as void, not as a baseline.**

**Date:** 2026-07-28, revised 2026-07-29 after the falsifying re-bench (§9),
then again 2026-07-29 with §10 (orientation audit) and §11 (the seat split)
**Checkpoint:** `v3hist-k64-enthalf-5m-20260723` @ `iteration-2657` (= 4.25M games)
**Eval builds:** original — Python `046f58f`, image `mcts-eval-crate-20260726d`;
re-bench — image `mcts-rb-bf72636` (main @ `bf72636`, crate + poke_engine
canary-verified before launch)

> **Read §9 before acting on §5, and §10–§11 before acting on anything.** The
> "implicated: dynamics divergence" conclusion below was tested by the
> prediction it generated, and failed. §5's exoneration of leaf value
> orientation was sound for a *global* flip and for a *per-ply* one (§10
> refutes the latter directly) but not for a flip that fires on one seat only —
> which is what it was. Both are retained unedited so the reasoning that
> produced wrong answers stays legible.

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

---

## 9. The falsifying re-bench (2026-07-29)

§8 committed to a prediction chosen so it could fail cleanly:

> if dynamics divergence is the cause, the *depth decay should flatten* — d6
> should rise toward d1, not merely shift upward.

**It did not rise.** Between the two runs, ~115 commits landed on main, including
22+ vendored gen3 engine patches and the whole simulator-fidelity campaign
(`docs/engine_divergence_ledger_20260728.md`).

| cell | n | score | Wilson 95% | pre-patch | change |
|---|---|---|---|---|---|
| control (raw v raw) | 100 | 0.480 | [0.385, 0.577] | 0.520 | −0.04 |
| `d1`-s1024 | 100 | 0.560 | [0.462, 0.653] | 0.530 | +0.03 |
| `d4`-s1024 | *in flight* | ~0.43 | — | — | — |
| **`d6`-s1024** | **66** | **0.364** | [0.258, 0.484] | **0.360** | **+0.004** |

The slope is fully intact: **0.560 → ~0.43 → 0.364** across d1 → d4 → d6, the
same monotone decay measured before any fidelity work existed. `d6` moved by
four thousandths.

**Therefore: dynamics-model divergence is NOT the cause of the depth decay.**
The §5 conclusion is retracted. Substantially fixing the divergences had no
effect on the curve they were supposed to explain.

### What still stands

- The depth decay is **real and reproducible** — measured twice, on independent
  builds ~115 commits apart, with a validated null both times.
- d1 remains at parity with raw and with the control on both runs.
- Something compounds per ply of rollout. That is the one durable fact, and it
  now has no confirmed mechanism.

### Caveats on this run

- `d6` is n=66 (three shards; ±0.11 half-width). A *partial* flattening could
  hide inside that. A flattening large enough to explain a 14-point deficit
  could not.
- The control drifted 0.520 → 0.480 on identical seeds. Raw-vs-raw never touches
  the search engine, so a pure engine patch should have left it byte-identical.
  It did not, because main also advanced on the observation side (charge-state
  surfacing, positional residual attribution, the Baton Pass parser change). The
  two runs are therefore paired on **seeds**, not on "everything but the
  engine", and 3–4 point differences carry no signal.
- `d4` is still filling and its number here is an in-flight log tally, not
  audited harness output.

### Hypotheses now refuted, in order of confidence spent on them

1. **Strategy fusion** in the cross-world visit-share vote — refuted by §4.4
   (`w1`, where the vote is a no-op, is still 14 points below parity).
2. **Dynamics-model divergence** compounding over depth — refuted by this
   section.

Both were argued at length before being tested. The pattern worth carrying
forward: each was plausible, each explained the shape of the table, and each was
wrong. The next candidate should be cheap to falsify *before* it is written up
as a cause.

### What has NOT been tested

- **Leaf-value orientation under depth.** §5 exonerated it by the d1 argument (a
  perspective flip would corrupt d1 too). That argument is sound for a *global*
  flip but not for one that depends on ply parity — a sign error applied per
  level would leave d1 clean and degrade even depths. The plan's Step 2 crate
  test (mirrored state, assert `v01` reflects about 0.5) is still unwritten and
  is now the cheapest untested candidate.
- **Backup at chance nodes** across the simultaneous-move branch structure.
- Whether depth > 6 being inert (§4.3) shares a cause with the decay.

**No third mechanism is asserted here.** The honest state is: reproducible
effect, two eliminated causes, no confirmed explanation.

---

## 10. Ply-parity value orientation: REFUTED. A seat-constant inversion: CONFIRMED.

**Date:** 2026-07-29. Crate rebuilt at the audited HEAD before any measurement;
probes ran against that build (module resolved out of a scratch prefix, model
feature verified live).

§9 named the cheapest untested candidate: a value-orientation error applied
**per level** rather than globally, which would leave `d1` clean and corrupt
increasingly with depth. It was tested. It does not exist.

### 10.1 The value path carries no per-level sign or seat change

Audited end to end (`rust/pokezero-search/`):

| site | finding |
|---|---|
| `tree.rs` backup (`finalize`) | the chance-node expectation is added to **both** seats' stats unchanged. `s2_stats[j].total_value += expectation - 1.0` is the REPLACEMENT of the `+1.0` virtual loss written in `traverse` — net per visit is `+expectation` on both sides. No negation, no `1 - v`, no per-ply seat swap. |
| `lib.rs` `MoveStats::puct` | the ONLY seat flip in the crate: `1.0 - self.mean()` for side two, applied identically at every level. This is the correct simultaneous-move form; it is not negamax. |
| `tree.rs` terminal pricing | `battle_is_over` (+1 = side ONE won, `state.rs:1485`) → `{0.0, 1.0}`. Side-one-absolute, depth-independent. |
| `fold.rs` | `perspective_slot` is set once at construction and is never written by `advance_in_place` / `process_line`; branch folds are clones of their parent's, so the seat perspective is inherited unchanged down every rollout. |
| `leaf.rs` | the leaf observation's seat is `LeafContext::self_is_p1`, a constant of the search. Nothing about the encode alternates with depth. |

**Also refuted while in `select()`:** side two's Q is NOT on a `[-1, 0]` scale.
Both seats' arms hold the same `[0, 1]` side-one quantity, so `c_puct` weighs
the exploration term against the same scale on both sides. Pinned by
`tree.rs::side_two_stats_accumulate_the_same_expectation_as_side_one`.

### 10.2 Pins (so this hypothesis stays dead)

Rust, model-free, expectations from fixture symmetry — not from engine output:

- `leaf_value_reflects_about_half_under_seat_mirror` — a leaf value and its
  seat-mirror sum to 1 on every fixture; a mirror-symmetric position reads
  exactly 0.5.
- `depth_parity_invariance_on_a_mirrored_position` — new fixture
  `symmetric.state` (side two is a verbatim copy of side one, so side one's
  exact win probability is 0.5 by construction). Backed-up root value at
  depths 1–6: **0.5000, 0.5000, 0.4981, 0.5004, 0.5004, 0.5004**, with the tree
  really growing (601 decision nodes by d4). No even/odd separation.
- `seat_mirror_maps_root_value_to_its_complement_at_every_depth` — root value +
  mirrored root value = **1.00006 / 1.00058 / 1.00049 / 1.00049** at d1–d4.
- `terminal_orientation_is_absolute_across_the_seat_mirror` — a guaranteed
  side-TWO KO prices at exactly 0.0, the complement of the side-one KO already
  pinned at 1.0.

Python (`tests/test_search_value_orientation.py`) drives the REAL encoded search
with a TorchScript stub whose orientation is fixed by construction
(`v = tanh(20 * (mean SELF-block HP fraction − mean OPPONENT-block HP
fraction))`, read out of the observation's own token blocks — the same
convention as the trained head, whose target is `+1` iff the observing seat won,
`dataset._terminal_value_for_player`).

### 10.3 What the audit did find: the seat boundary, not the ply boundary

The two conventions that meet at the leaf are not the same convention:

- the leaf observation is encoded from the **searching seat's** perspective
  (SELF / OPPONENT token blocks) and the value head is **self-relative**;
- the tree is **side-one-absolute** (terminal branches, `s1_stats`, one flip in
  `puct`);
- `engine_world.py:534` pins `slot_sides = {"p1": "side_one", "p2": "side_two"}`,
  so the searching seat is side TWO exactly when PokeZero plays p2 — and
  `mcts_eval/scoring.py` plays **both seats of every team seed**.

`multiply_batched_encoded_core` fed `values01` through unreflected. On a p2
root, every model leaf therefore disagreed with every terminal branch in the
same tree, and side two selected on `1 - v` of an already-inverted value.

Measured on drivable golden-corpus roots (`SIMS=96`, priors off, so the report
is about values alone). Ground truth for "which side leads" is read off the
constructed world spec:

| row | seat | side 1 leads | model v01 at root | root_value d1 | d2 | d3 | d4 |
|---|---|---|---|---|---|---|---|
| 2 | p1 | yes | 0.9752 | 0.6849 | 0.8225 | 0.8225 | 0.8225 |
| 3 | **p2** | yes | 0.2271 | **0.2234** | **0.2240** | **0.2240** | **0.2240** |

Row 3's tree reports side one *losing* while side one leads — and note the
shape: the error is **flat across depth**, present already at d1. That is the
refutation of the parity hypothesis and the confirmation of the seat one, in the
same four numbers.

The same rows after the fix: row 3 → 0.5820 / 0.6991 / 0.6995 / 0.6995, now on
the same side of 0.5 as row 3's p1-seated counterpart.

Both blocks are reproduced by `tests/test_search_value_orientation.py`, which
was run against an unfixed build as a negative control: it fails on the p2 row
at **every** depth (1, 2, 3, 4) and passes on the p1 rows.

**It changed the move.** Same roots, 256 sims, priors off, acting seat's argmax:

| row | seat | depth | as shipped | with the reflection |
|---|---|---|---|---|
| 1 | p2 | 2 | `switch smeargle` (45%) | `fireblast` (36%) |
| 1 | p2 | 4 | `switch smeargle` (44%) | `fireblast` (32%) |
| 3 | p2 | 2 | `switch smeargle` (36%) | `seismictoss` (29%) |
| 3 | p2 | 4 | `switch smeargle` (37%) | `seismictoss` (27%) |

(The "as shipped" column is produced on the FIXED build by feeding a
sign-negated stub, which on a side-two seat is algebraically the pre-fix path —
`1 - (1 - v01) = v01`. Validated: it reproduces the unfixed build's row-3 root
values to the last recorded digit, 0.223431 / 0.224028 / 0.224028 / 0.224028.)

### 10.4 Fix

One site, one reflection, at the seat boundary and never per ply
(`model.rs`, `multiply_batched_encoded_core`):

```rust
if self_side_one { output.values01 } else { output.values01.iter().map(|v| 1.0 - v).collect() }
```

Priors are untouched: they are policy over the SELF seat's own actions and were
already applied to the self side.

### 10.5 What this does NOT establish

Following §9's lesson — name the falsifier before writing the cause up:

- **This is not yet shown to be the depth-decay mechanism.** It is a confirmed
  defect with the right seat structure; the depth structure is a *prediction*,
  not a measurement: the priors carry the p2 seat at d1 (they were always
  correctly oriented), and inverted values gain leverage over them as depth and
  sims grow — which is also the shape of §4.2's sims ablation. Untested.
  **→ Now tested. See §11: the prediction held.**
- **The cheap falsifier, to run before this is called the cause:** split the
  ALREADY-RECORDED grid by seat. `mcts_eval/scoring.py` keeps `seat` on every
  result. If the `d6` deficit is symmetric between the p1 and p2 seats, this
  defect is not the cause and this section must be retracted the way §5 was. If
  the p2 seat carries essentially all of it, re-bench on the fixed build with
  the §8 acceptance criterion. **→ Run in §11. p2 carries it.**
- The `d1`-parity argument in §5 remains sound for global flips; it never
  covered a flip that only fires on one seat, because every reported cell
  averages the two seats.
- §4.3 (depth > 6 inert) reproduces here in miniature: `max_depth_reached`
  saturates at 4 for `max_depth` 5 and 6 on the symmetric fixture. Visit
  dilution, unrelated to orientation.

---

## 11. The seat split: the depth decay is the p2 seat, and only the p2 seat

**Date:** 2026-07-29. Source: the recorded shards under
`/shared/scott-experiment/mcts-power-overlay-20260728/results` (404 files),
read on the `olfusa` cluster. Reproduce with
`scripts/mcts_seat_split.py <results-dir>` — plain `python3`, no repo imports,
so it drops straight into a controller pod.

§10.5 named the falsifier: split the already-recorded grid by seat, and retract
§10's mechanism claim if the deficit is seat-symmetric. **It is not. The p2 seat
carries essentially all of it, at every depth and every simulation budget, on
both builds.**

### 11.1 Method

Each shard carries `per_game` = `{seed, search_seat, winner}`. Score is
`mcts_eval/scoring.py`'s: win 1, tie 0.5, loss 0; results deduped on
(arm, seat, seed).

- **Cell identity is the shard FILENAME, not the `config` field.** The control
  shards carry a leftover `config: d4-s1024-b64-w4` from the runner's defaults;
  keying on `config` silently merges 100 raw-vs-raw games into the d4 search
  cell. (This bit me first time through.)
- **Identity of each arm with the published cells was established by
  reproducing §4's pooled numbers**, not by file dates: `d2` 0.450, `d6-s1024`
  0.360, `d4-s2048` 0.310, `d4-s4096` 0.270, `d4-s8192` 0.270, `d6-s2048`
  0.360, `w1` 0.360, `w16` 0.430 — all exact or within one game of the table.
  The `fb-*` arms reproduce §9's re-bench (control 0.485 vs 0.480, `d1` 0.565
  vs 0.560, `d6` 0.360 vs 0.364 now that it filled from n=66 to n=100).
- **Seat assignment is by seed parity**: p1 = even seeds 600000–600098, p2 =
  odd 600001–600099. All seventeen n=50 arms use the *identical* two seed sets,
  so every ladder below is paired on seeds *within* a seat.

### 11.2 Depth ladder, s1024 w4 (original grid)

| cell | p1 seat | p2 seat | pooled |
|---|---|---|---|
| control (raw v raw) | 0.550 [0.413, 0.679] | 0.500 [0.366, 0.634] | 0.525 |
| `d1` | 0.560 [0.423, 0.688] | 0.510 [0.376, 0.643] | 0.535 |
| `d2` | 0.480 [0.348, 0.615] | 0.420 [0.294, 0.558] | 0.450 |
| `d6` | 0.460 [0.330, 0.596] | **0.260** [0.159, 0.396] | 0.360 |
| `d8` | 0.460 [0.330, 0.596] | **0.260** [0.159, 0.396] | 0.360 |

n = 50 per seat per cell. (§4.3 confirmed in passing: `d6` and `d8` share all
100 seeds with **identical winners, 100/100**.)

### 11.3 Simulation ladder — the decisive one

| cell | p1 seat | p2 seat | pooled |
|---|---|---|---|
| `d4-s512` (n=20) | 0.400 [0.219, 0.613] | 0.400 [0.219, 0.613] | 0.400 |
| `d4-s2048` | 0.480 [0.348, 0.615] | 0.140 [0.070, 0.262] | 0.310 |
| `d4-s4096` | 0.540 [0.404, 0.670] | **0.000** [0.000, 0.071] | 0.270 |
| `d4-s8192` | 0.540 [0.404, 0.670] | **0.000** [0.000, 0.071] | 0.270 |
| `d6-s1024` | 0.460 [0.330, 0.596] | 0.260 [0.159, 0.396] | 0.360 |
| `d6-s2048` | 0.520 [0.385, 0.652] | 0.200 [0.112, 0.330] | 0.360 |
| `d6-s4096` | 0.590 [0.452, 0.715] | **0.000** [0.000, 0.071] | 0.295 |

**On the p1 seat more search is better** — 0.400 → 0.480 → 0.540 → 0.540 at d4,
0.460 → 0.520 → 0.590 at d6, on the same 50 seeds throughout. **On the p2 seat
more search goes to zero** — 0.400 → 0.140 → 0.000 → 0.000, and 0.260 → 0.200 →
0.000. Three cells are **0 wins and 0 draws in 50 games**, in an arm whose own
`d1` cell scored 0.510 on those very seeds.

§4.2's "an 8× simulation budget costs ~5× wall time and *loses* strength" was
the average of a search that works and a search that anti-optimizes.

### 11.4 §9 re-bench — same split, independent build

| cell | p1 seat | p2 seat | pooled |
|---|---|---|---|
| control (raw v raw) | 0.430 [0.303, 0.567] | 0.540 [0.404, 0.670] | 0.485 |
| `d1-s1024` | 0.610 [0.472, 0.733] | 0.520 [0.385, 0.652] | 0.565 |
| `d6-s1024` | 0.540 [0.404, 0.670] | **0.180** [0.098, 0.308] | 0.360 |

The "fully intact slope" of §9 is a p2 slope. p1 goes 0.430 → 0.610 → 0.540 —
no decay at all.

### 11.5 Significance and attribution

Two-proportion z, p1 vs p2, and the p2 seat's share of each cell's pooled
deviation from the 0.500 null:

| cell | z (p1 − p2) | p2 share of the deficit |
|---|---|---|
| control (orig) | +0.50 | — (no deficit) |
| control (re-bench) | −1.10 | — (no deficit) |
| `d1` (orig / re-bench) | +0.50 / +0.91 | 14% / 15% |
| `d2` | +0.60 | 80% |
| `d6-s1024` | +2.08 | 86% |
| `d6-s2048` | +3.33 | 107% |
| `d6-s4096` | **+6.47** | 122% |
| `d4-s2048` | +3.68 | 95% |
| `d4-s4096` / `d4-s8192` | **+6.08** | 109% |
| `d6-s1024` (re-bench) | +3.75 | 114% |

Both controls are null, so the harness has no seat bias. Both `d1` cells are
null, matching §5's observation that d1 is at parity — and now explaining it:
at one ply the correctly-oriented priors still dominate the visit count, so an
inverted Q has almost no leverage. Every deep or high-sim cell separates, and
shares above 100% mean the p1 seat is *above* 0.500 there — the correctly
oriented seat was gaining from search while the pooled number fell.

### 11.6 Verdict

**Outcome (b). The seat-constant value inversion of §10 is the depth-decay
mechanism.** The three facts of §9's "what still stands" now read:

- the decay is real and reproducible — yes, and it is a **p2-seat** decay;
- `d1` is at parity — because priors, not values, drive d1;
- "something compounds per ply of rollout" — an inverted leaf value compounds
  per ply *and* per simulation, on the one seat where it was inverted.

The worlds ablation stays flat on **both** seats (p1 0.420/0.460/0.460/0.440/
0.460, p2 0.300/0.270/0.260/0.260/0.400 across w1→w16), so §4.4's refutation of
the aggregation hypothesis stands on its own. The mild w16 uptick is a p2
effect, consistent with more determinizations diluting a systematically
inverted Q.

### 11.7 What this still does not settle

- **A small residual on the correctly-oriented seat cannot be excluded.** p1
  sits at 0.480 / 0.460 / 0.460 for d2 / d6 / d8 against 0.550–0.560 for control
  and d1. Every interval overlaps and the sims ladder runs the other way, so
  this is consistent with noise — but a 2–5 point real effect would hide inside
  n = 50. Only the §8 acceptance re-bench resolves it.
- **The two seats hold disjoint seed sets** (parity split), so p1-vs-p2 is a
  between-seed comparison. The two controls bound that at +0.05 and −0.11, both
  null; and no seed-set luck produces 0/50 in a cell whose own d1 arm scored
  0.510 on the same seeds. But the seat gap itself should not be quoted to
  better than a few points.
- **Fallback rises in the largest cells** (up to 0.217 in one `d6-s4096`
  shard, against 0.000–0.005 typical). Far too small and too late to
  manufacture a 0/50, but it should be watched on the re-bench.

### 11.8 Next

Run the §8 acceptance criterion on the fixed build (PR #937): search ≥ 0.500 vs
raw at `d4`/`s1024` over 200 mirrored-pair games, **reported per seat**. The
prediction this makes falsifiable: p1 stays where it is and p2 rises to meet it.
If p2 does not recover, §10 and §11 both retract.

Every strength number in §4, §7 and §9 was produced by an engine that played one
of its two seats backwards. They should be treated as void, not as a baseline to
improve on.
