# MCTS degradation: why search loses to its own prior

**Status:** mechanism **IDENTIFIED and CONFIRMED** (§10–§11, §13). The §8 acceptance
criterion is **met** on the fixed build: 0.615 [0.573, 0.658] over 220 within-seed
mirrored pairs, +0.115 over its own prior. One residual seat asymmetry remains open
(§13.4). The leaf value was consumed with the wrong seat orientation whenever
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

> **Telemetry caveat — every model-mode `s/decision` and `search_wall` figure in
> this document is ~2x inflated.** `EngineMctsPolicy._search_model` accumulated
> `search_wall_seconds` twice per decision (and counted each world twice), fixed
> 2026-07-29. The tables below are left as recorded rather than rewritten: the
> inflation is uniform across model-mode cells, so every wall-time COMPARISON in
> here still holds, and only the absolute seconds are wrong. **Scores, Wilson
> intervals and every strength conclusion are unaffected** — the double count
> never touched the aggregated visit shares. Cite absolute wall numbers only from
> post-fix builds; the raw-vs-raw control rows were never model-mode and are
> exact.

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
  Note the symmetric-position 0.5 pin is a sanity/regression check only —
  0.5 is a fixed point of the reflection `v -> 1 - v`, so it cannot
  discriminate the §10.3 inversion; the discriminating pins are the
  seat-mirrored sums and the asymmetric p2 rows (review finding, 2026-07-29).
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

---

## 12. The handcrafted-leaf control: the tree itself was never the problem

**Date:** 2026-07-29, run concurrently with §10/§11 and reported independently.
Full write-up and artifacts: `docs/mcts_handcrafted_leaf_depth_findings.md`,
`docs/audit_artifacts/hc-depth-grid-20260729/`.

§11 shows the deficit is the p2 seat and that on p1 more search is better. That
is measured on the model path, so it says the *learned* value works when it is
oriented correctly. It leaves one thing open: whether the tree would convert
depth into strength for a value function that is not the network at all.

This section closes that. `EngineMctsConfig.leaf_eval="hp_fraction_crate"` runs
the crate's `puct_search_multi` — the same `MultiPlyConfig` reaching the same
`traverse`/`expand_edge`/`finalize`, the same chance-node backup, the same depth
and `c_puct` semantics — with the engine's handcrafted HP-fraction evaluation at
the leaves and uniform priors. Only the leaf changes. Same mirrored-seat harness,
seeds 600000+, `w4`, n = 400 per cell (the arm is cheap enough that the 100-seed
window's ±0.10 was avoidable; both windows are in the write-up).

| depth, s1024 | score | Wilson 95% | | d6, budget | score | Wilson 95% |
|---|---|---|---|---|---|---|
| control (raw v raw) | 0.496 | [0.448, 0.545] | | `s256` | 0.255 | [0.215, 0.300] |
| `d1` | 0.196 | [0.160, 0.238] | | `s1024` | 0.328 | [0.283, 0.375] |
| `d2` | 0.289 | [0.247, 0.335] | | `s4096` | 0.360 | [0.314, 0.408] |
| `d4` | 0.328 | [0.283, 0.375] | | | | |
| `d6` | 0.328 | [0.283, 0.375] | | | | |
| `d8` | 0.328 | [0.283, 0.375] | | | | |

Both axes rise (paired on the same seeds: d1→d4 p < 0.0001, s256→s4096
p = 0.0006). Levels are not comparable to §4 — different opponent pairing, and a
locally available v2.2 checkpoint as the raw opponent — so the slope is the
readout, and it points the other way on both axes.

**What this adds to §11:**

- **The tree is exonerated independently of the model being right.** Decoupled
  selection, the chance-node exact-expectation backup and the depth mechanics
  turn plies and simulations into strength with an arbitrary bounded value
  function. This retires "backup at chance nodes" from §9's *What has NOT been
  tested* without leaning on the fixed value path.
- **It closes the "maybe depth just helps weak values" reading.** One could have
  argued the handcrafted arm improves only because it starts weak. §11.3's p1
  ladder rules that out from the other side: with a strong, correctly oriented
  learned value, more search is also better. The two arms agree.
- **This arm's own seat orientation was verified, not assumed** — the leaf is
  `0.5 + 0.5*(hp_frac(s1) − hp_frac(s2))`, computed off the state with no
  seat-dependent conversion, and
  `tests/test_multiply_chance_search.py::test_seat_swap_reflects_about_one_half`
  pins root values and per-arm Q to reflect about 0.5 at d1/d2/d4/d6. No
  `leaf_eval="model"` cell was run here, so nothing in this section inherits the
  §10.3 inversion. Every cell is reported per seat in the write-up; the
  handcrafted arm shows nothing like §11's seat collapse (its one +0.11 gap at
  d4/d6-s1024 does not replicate at s4096, where it reverses sign).

### 12.1 §4.3 closed: the budget binds, not the cap

The crate has always counted `max_depth_reached`; nothing had read it. Node
depth, root = 0, and the cap bounds child *creation* (`depth + 1 >= max_depth`),
so a binding cap `d` tops out at `d − 1`.

| cell (s1024) | cap | max reached | mean reached | histogram (node depth → world-searches) |
|---|---|---|---|---|
| hc-`d1` | 1 | 0 | 0.000 | `{0: 66716}` |
| hc-`d2` | 2 | 1 | 0.999 | `{0: 102, 1: 71026}` |
| hc-`d4` | 4 | 3 | 2.508 | `{0: 97, 1: 364, 2: 32923, 3: 35640}` |
| hc-`d6` | 6 | 5 | 2.666 | `{0: 98, 1: 374, 2: 32965, 3: 26403, 4: 7468, 5: 1724}` |
| hc-`d8` | 8 | 7 | 2.672 | `{0: 98, 1: 374, 2: 32965, 3: 26403, 4: 7468, 5: 1381, 6: 264, 7: 79}` |

The cap binds at every setting — but only just: mean depth moves 2.508 → 2.666 →
2.672 across caps 4/6/8, and 0.11% of world-searches reach depth 7. The d6 and d8
histograms are the same tree one row longer (d6's 1724 traversals stopped at
depth 5 are d8's 1381 + 264 + 79), and hc-`d6` and hc-`d8` give the identical
winner on **400/400** seeds — §4.3 and §11.2's 100/100 reproduced with a cause.

Raising the budget at a fixed cap of 6 moves the distribution exactly as that
account predicts:

| budget | mean node depth | share at the cap (depth 5) |
|---|---|---|
| `s256` | 1.936 | 0.2% |
| `s1024` | 2.666 | 2.5% |
| `s4096` | 3.471 | 15.5% |

So §4.3's "either the tree does not reach depth 6, or the cap is not binding" was
a false dichotomy: the cap binds, and the subtree beyond node depth ~3 is too
visit-starved to change a root argmax. **Any future depth cell should report the
depth reached, not the depth configured.**

---

## 13. The acceptance re-bench: the prediction held

**Date:** 2026-07-29. Build: main @ `2103d65` (the #937 merge, plus #936/#938/#939),
engine fingerprint `7909290e14e065cda5cc38d5698c45a04db4862a416e1e2a52af86075104830b`
(29 patches, 8 crate sources), image pinned by digest and gated in-pod before the
first game of every shard. 440 games per arm, seeds 7800000–7800219, **within-seed
mirrored pairs** — each seed played from both seats, so the two seats face the same
two teams. Plan and prediction were registered before the run
(`docs/mcts_acceptance_rebench_plan.md`).

### 13.1 Result

| arm | p1 seat | p2 seat | pooled pair mean |
|---|---|---|---|
| control (raw v raw) | 0.448 [0.383, 0.514] | 0.552 [0.486, 0.617] | **0.500** [0.500, 0.500] |
| **search `d4-s1024`** | **0.639** [0.573, 0.699] | **0.591** [0.525, 0.654] | **0.615** [0.573, 0.658] |

220 complete pairs per arm, Wilson 95% per seat, percentile bootstrap on the pair mean.

**§8 acceptance criterion: MET.** `0.615`, and the whole interval sits above 0.500.
Paired against the control on the same seeds, search adds **+0.115** [+0.072, +0.158].

### 13.2 The prediction, adjudicated

Verbatim from §11.8, as registered:

> The prediction this makes falsifiable: p1 stays where it is and p2 rises to
> meet it. If p2 does not recover, §10 and §11 both retract.

**Held**, with the scope of that "held" stated precisely in §13.2.1 below.

The search seat gap (p1 − p2) collapsed from **+0.34 to +0.59** across the
pre-fix cells to **+0.048**. The p2 seat, which sat at 0.140–0.260 in the pre-fix
deep and high-simulation cells and at exactly 0.000 in three of them, reads
**0.591**. §10 and §11 stand.

#### 13.2.1 Two things the adjudication must not overclaim

**The run was larger than pre-registered.** §11.8 and §8 both say *200* mirrored
pairs; this ran **220**. The margin was reserved so that up to 20 pairs could be
excluded and still clear the bar. None were: all 220 pairs completed in both
arms, so every number above is on 220, not on a pre-registered 200 with 20
discarded. More than promised, but a deviation, and stated rather than left to
be noticed.

**"p1 stays where it is" is not directly measurable across this change, and it
did not hold literally.** p1 read **0.639** here against the **0.460–0.540**
range of §11's cells. But the comparison crosses both a seed block
(7800000–7800219 vs 600000–600099) and a cell (`d4-s1024` vs §11's `d6`/`d4`
cells at their own budgets), so the two are not commensurable and no conclusion
should be drawn from the difference in either direction.

The falsification criterion was the **p2 clause** — "if p2 does not recover, §10
and §11 both retract" — and p2 is what the run adjudicates: 0.140–0.260 (and
exactly 0.000 in three cells) before, **0.591** after. The p1 clause was a
statement of expectation, not a test, and it is honest to say so: had p1 been
the criterion, this run could not have decided it.

### 13.3 The control did something the old harness could not

Under within-seed pairing with two identical deterministic policies, both runs of a
seed are *the same battle with the labels swapped*. The pooled control mean is
therefore **exactly 0.500 by construction** — and it measured exactly 0.500 with a
**zero-width** bootstrap interval. That is a strict equality test of the pairing, not
a noisy null: any deviation would have meant seat- or label-dependent
nondeterminism in the harness.

Its per-seat split is the payload: these 220 seeds **favour the p2-slot team by
10.4 points** (0.448 / 0.552). Read against that baseline, search gains **+0.191**
on p1 and **+0.039** on p2. The parity-split harness of §4/§9 could not measure this
at all.

### 13.4 What is NOT closed: a residual seat asymmetry

The seat-gap collapse is not a seat-gap elimination. Difference-in-differences on
paired seeds — the search arm's within-seed seat gap minus the control's, which
removes team advantage exactly:

```
control seat gap (p1 - p2)   -0.1045
search  seat gap (p1 - p2)   +0.0477
difference-in-differences    +0.1523   bootstrap95 [+0.0182, +0.2818]
```

**The interval excludes zero.** Search still helps the p1 seat more than the p2 seat,
by about 15 points, beyond what the seed set's team advantage explains. The lower
bound is +0.018 and the interval is wide — this is a marginal result on 220 pairs,
not a second smoking gun — but it is a real signal and it should not be rounded away.

This is §11.7's open question, now measurable and pointing the *opposite* way from
how it was framed: the residual is not a depth effect on a healthy seat, it is a
remaining seat asymmetry. Candidates, none tested:

- the opponent side of the tree still runs uniform priors (`model_priors` applies to
  the acting seat only — `multiply_batched_encoded_core`), which is not seat-symmetric
  in its *effect* once the acting seat is side two;
- belief/world construction asymmetries between the self and opponent seats;
- residual fallback shape differing by seat.

The cheap next probe is the same difference-in-differences at a second cell — the
staged `d6-s4096` config exists for this and is owner-gated.

### 13.5 Consequences

- Every strength number in §4, §7 and §9 was produced by an engine that played one of
  its two seats backwards. They stay **void**, not a baseline.
- Search at `d4`/`s1024` is now **worth its compute**: +0.115 over its own prior on
  paired seeds.
- The depth and simulation ladders should be re-run on the fixed build before anyone
  reasons about depth again. §4.1–§4.3 describe an artefact.

### 13.6 Provenance notes

- **Encoder tables.** The materialized table's vocab is **1216**, matching this
  checkpoint's own `category_vocab` exactly (1216 + 16 OOV + 1 = its 1233 trained
  embedding rows), with `volatile:solarbeam` absent from both — the checkpoint
  predates that work. A table regenerated "fresh from the build" would have been the
  defect here, not the fix. Established by direct measurement, because
  `mcts_eval.resolver.validate_encoder_tables` — **as it stood when this campaign
  ran** — compared schema version, token count and feature counts but **not the
  vocab or its size**, and so could not catch a vocab renumbering on its own.
  (#945 has since merged a vocab check; whether the merged version covers size
  *and* content, and against which side, is the k0 lane's to confirm — not
  re-litigated here.)
- **Open hardening item: the export reuse key cannot see a pokezero-side
  vocabulary-enumeration change.** `export_reuse_key` keys on
  `checkpoint_sha256`, `model_device`, `observation_contract_sha256`,
  `showdown_source_sha256` and `exporter_revision`. The observation contract is
  `{schema_version, token_count, categorical_feature_count,
  numeric_feature_count, transition_token_count, feature_masks}` — **it carries
  no vocabulary at all**, neither the list nor its length. The vocab is produced
  by `pokezero.randbat_vocab.gen3_category_vocabulary`, pokezero's *own*
  enumeration over Showdown data. So a token added on the pokezero side —
  `volatile:solarbeam` is exactly this, from the charge-state work — moves
  neither hash: Showdown data did not change, and no contract field did either.
  The key then permits reuse of a table built before the enumeration change.
  Only `exporter_revision`, a hand-bumped constant, would catch it, and the
  comment above it records this class biting once already ("Missing this bump
  caused a trimmed run to reuse cached 87-token tables"). **This campaign was
  safe only because its checkpoint predates the token**, so the cached table and
  the checkpoint agreed at 1216. A checkpoint trained *after* such a change,
  served a table cached from before it, would be reused silently. The durable
  fix is to put the vocabulary hash (or its length) into the observation
  contract so the reuse key and the validator both see it.
- **Root encode was BUILD-anchored, not checkpoint-anchored (bounded caveat, not
  a retraction).** *[CLOSED 2026-07-29 by #954: the vocabulary axis is now latched
  for every consumption site and the latch is renamed
  `env_config_from_checkpoint_provenance`. The description below is retained as the
  state at the time of this finding — the old symbol name will not resolve.]* `scripts/mcts_acceptance_h2h.py` built its env as
  `LocalShowdownConfig(showdown_root=…, set_belief_source=True)` and passed
  **no `category_vocab`**; `env_config_with_checkpoint_masks` latches the mask
  and schema axes but not the vocabulary one, and `local_showdown` then falls
  back to `self.config.category_vocab or gen3_category_vocabulary(root, …)` —
  i.e. the *build's* enumeration. Both arms did this, identically. Scope, which
  is narrower than it first looks:

  - The **crate** path — the search arm's own model calls, root priors and every
    leaf value — ran on the materialized checkpoint-matched tables, measured at
    1216 (above). The root *inputs* handed to the crate are row-input JSON
    (species and ledger strings), re-encoded by the crate from those tables, so
    the build vocab does not reach the search arm's model evaluations.
  - What the build vocab did drive is the **raw policy's** own forwards: the
    opponent in the search arm, and both players in the control arm. Against a
    1233-row trained embedding, a build vocab carrying `volatile:solarbeam`
    shifts the 13-token volatile tail from index 1204 by one row.
  - The **control is unaffected in its pooled number by construction** — both
    runs of a seed are the same battle relabelled, so it is exactly 0.500
    whatever the encode. And because the search arm's opponent *is* that same
    raw policy, the control remains the right baseline for it: the comparison is
    internally consistent.
  - The direction of any residual bias is that a weakened raw opponent would
    **overstate** the +0.115 delta. The k0 lane's k64c reproduction bounds the
    shift's effect far below that margin, so this is a caveat on the margin's
    exact size, not on its sign or on the §8 verdict. The seat-split and DiD
    results are untouched: the shift is common-mode across seats.

  The fix for future runs is one argument — pass the checkpoint's
  `category_vocab` into `LocalShowdownConfig` — and it belongs with the
  observation-contract hardening item above, since both are the same root cause:
  vocabulary is not part of any latch.
- **Telemetry.** #939 single-counts model-mode s/decision. This campaign's wall
  figures are not comparable to the pre-#939 grid without saying so; strength scores
  are unaffected.
