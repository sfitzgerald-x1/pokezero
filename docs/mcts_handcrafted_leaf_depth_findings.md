# The handcrafted-leaf depth grid: the decay is not in the tree

**Status:** the depth decay of `docs/mcts_degradation_findings.md` is **not
reproduced** when the learned leaf value is replaced by the engine's handcrafted
HP-fraction evaluation inside the *identical* search tree. Both of the knobs that
study found to be negatively valued reverse sign: strength rises monotonically
with **depth** (0.196 → 0.328, saturating where the tree stops growing) and
monotonically with **simulation budget** (0.255 → 0.360 at d6). This eliminates
the search structure — decoupled selection, chance-node backup,
exact-expectation resolution, the depth mechanics themselves — as the cause.
What remains is on the value side.

**Date:** 2026-07-29
**Build:** branch `scott/mcts-hceval-ablation`, based on `main` @ `651699d`
(PR #931, the falsifying re-bench). The `pokezero-search` crate and the
gen3-patched `poke-engine` were rebuilt from that worktree into a dedicated venv
before any measurement, and every cell JSON records the resolved module paths it
actually imported plus the 27-patch engine build fingerprint
`32a9b325db5f2655…`.
**Raw-policy opponent:** `checkpoints/pz-v2-2-1m.pt`
(`foundation-midscale-iter-0312`, observation schema v2.2, 1M games), argmax.

> Read the **slope**, not the level. Handcrafted-search-vs-raw-NN-policy is a
> different opponent pairing from NN-search-vs-raw-NN-policy, so the absolute
> scores here are not comparable to the numbers in the degradation doc. Only the
> shape across depth is.

---

## 1. The question this was built to split

`docs/mcts_degradation_findings.md` records a reproducible, twice-measured fact
with no surviving mechanism: engine MCTS loses to the raw policy it is built on,
and loses *more* with every ply of rollout (d1 0.53 → d6 0.36; re-benched
0.56 → 0.364 after ~115 commits of engine-fidelity work). Two mechanisms were
argued at length and then falsified — cross-world strategy fusion (§4.4) and
dynamics-model divergence (§9).

Everything in that search has two halves: a **tree** (decoupled per-side PUCT
selection, chance nodes with engine-exact branch probabilities, exact-expectation
backup, a depth cap) and a **value** (the checkpoint's TorchScript leaf forward,
plus the acting side's model priors). No experiment had separated them.

This one does. `EngineMctsConfig.leaf_eval="hp_fraction_crate"` runs the crate's
own `puct_search_multi`: the **same** `MultiPlyConfig` reaching the **same**
`traverse`/`expand_edge`/`finalize` as model mode, with `HpFractionEval` pricing
the leaves instead of the network.

The prediction fork, fixed before the run:

- **(a)** handcrafted decays with depth like the NN arm → the defect is
  structural, independent of the value function;
- **(b)** handcrafted is flat or rising with depth → the defect is value-side.

---

## 2. Result: (b), and not marginally

Same harness discipline as the degradation doc: head-to-head against the raw
policy of the same checkpoint, seats mirrored by seed parity (`seed % 2`), seeds
from 600000. Every cell plays the same seeds against the same opponent, so cells
are paired.

### 2.1 Slope table — comparability window, seeds 600000–600099

| cell | n | score | Wilson 95% |
|---|---|---|---|
| control (raw v raw) | 100 | 0.500 | [0.404, 0.596] |
| **hc-d1**-s1024-w4 | 100 | **0.160** | [0.101, 0.244] |
| **hc-d2**-s1024-w4 | 100 | **0.320** | [0.237, 0.417] |
| **hc-d4**-s1024-w4 | 100 | **0.410** | [0.319, 0.508] |
| **hc-d6**-s1024-w4 | 100 | **0.410** | [0.319, 0.508] |
| **hc-d8**-s1024-w4 | 100 | **0.410** | [0.319, 0.508] |

(Per-decision cost and fallback are cumulative over the whole run and are
reported once, in §2.2.)

### 2.2 Slope table — full run, seeds 600000–600399

The grid is cheap enough (30–200× cheaper per decision than the model arm) that
the 100-seed window's ±0.10 half-width was not worth living with.

| cell | n | score | Wilson 95% | s/decision | fallback |
|---|---|---|---|---|---|
| control (raw v raw) | 400 | **0.496** | [0.448, 0.545] | — | — |
| **hc-d1**-s1024-w4 | 400 | **0.196** | [0.160, 0.238] | 0.0095 | 2.03% |
| **hc-d2**-s1024-w4 | 400 | **0.289** | [0.247, 0.335] | 0.0144 | 0.51% |
| **hc-d4**-s1024-w4 | 400 | **0.328** | [0.283, 0.375] | 0.0195 | 0.58% |
| **hc-d6**-s1024-w4 | 400 | **0.328** | [0.283, 0.375] | 0.0197 | 0.58% |
| **hc-d8**-s1024-w4 | 400 | **0.328** | [0.283, 0.375] | 0.0197 | 0.58% |

**Monotone rising, then flat.** 0.196 → 0.289 → 0.328 → 0.328 → 0.328.

### 2.3 Paired reads (the powerful test)

Interval overlap is the wrong instrument when every cell played the same seeds
against the same opponent. Discordant-pair counts over all 400 seeds, two-sided
sign test:

| A | B | A only | B only | agree | p |
|---|---|---|---|---|---|
| hc-d1 | hc-d2 | 40 | 78 | 282 | 0.0006 |
| hc-d1 | hc-d4 | 36 | 89 | 275 | <0.0001 |
| hc-d2 | hc-d4 | 34 | 51 | 315 | 0.082 |
| hc-d4 | hc-d6 | 1 | 1 | 398 | 1.000 |
| hc-d6 | hc-d8 | 0 | 0 | 400 | 1.000 |
| control | hc-d4 | 123 | 54 | 223 | <0.0001 |

Depth 1 → 2 and 1 → 4 are decisive gains. 2 → 4 is a real but marginal gain. Past
4, nothing changes at all.

### 2.4 Per seat

Mirroring cancels seat advantage in the pooled score, which also hides it. Seat
asymmetry is a live hypothesis for the decay (see §2.5), so every cell is
reported split: candidate on p1 = even seeds, candidate on p2 = odd seeds. The
halves are disjoint seed sets, so the test is an unpaired two-proportion z.

| cell | p1 | Wilson 95% | p2 | Wilson 95% | p1 − p2 | z | p (uncorrected) |
|---|---|---|---|---|---|---|---|
| control | 0.490 | [0.422, 0.559] | 0.502 | [0.434, 0.571] | −0.012 | −0.25 | 0.80 |
| hc-d1 | 0.225 | [0.173, 0.288] | 0.168 | [0.122, 0.225] | +0.057 | +1.45 | 0.15 |
| hc-d2 | 0.285 | [0.227, 0.351] | 0.292 | [0.234, 0.359] | −0.008 | −0.17 | 0.87 |
| hc-d4 | 0.380 | [0.316, 0.449] | 0.275 | [0.218, 0.341] | +0.105 | +2.24 | 0.025 |
| hc-d6 | 0.385 | [0.320, 0.454] | 0.270 | [0.213, 0.335] | +0.115 | +2.45 | 0.014 |
| hc-d8 | 0.385 | [0.320, 0.454] | 0.270 | [0.213, 0.335] | +0.115 | +2.45 | 0.014 |

n = 200 per half.

Two things to take from this, in order of confidence:

1. **The slope is not a seat artifact.** It rises with depth on both seats
   (p1 0.225 → 0.385, p2 0.168 → 0.270). The verdict in §5 does not depend on
   which half you read.
2. **A seat gap appears at d4/d6/d8-s1024 — and does not replicate.** p1 outruns
   p2 by ~11 points there (z ≈ 2.4, p ≈ 0.014 uncorrected; d4/d6/d8 are the same
   games, so that is one effective test, not three, and it does not survive
   correction across the cells tested). The obvious follow-up was already in the
   data: the s4096 cells at the same depths are near-independent replications
   (same seeds, same opponent, different search realizations), and they show
   **−0.038** at d4 and **−0.035** at d6 — the opposite sign, both n.s. The
   control is seat-clean (−0.012, p = 0.80), max-damage is seat-clean
   (+0.005, §2.6), d2 and d6-s256 are seat-clean, and §2.5 rules out a
   value-orientation cause in this path by direct measurement.

   Read: most likely noise, and recorded as such. It is reported rather than
   dropped because seat asymmetry is the parity lane's live hypothesis and a
   null from a path with no model value in it is itself worth having. If anyone
   wants to chase it, the cheap version is a dedicated seat read with per-seat
   fallback attribution — this run carries fallback pooled across seats.

### 2.5 Seat orientation of the handcrafted path, verified rather than assumed

The parity lane found a seat-constant inversion on the **model** leaf path (p2
roots entered the tree unreflected). The handcrafted path is structurally immune:
`HpFractionEval::eval` is `0.5 + 0.5 * (hp_frac(side_one) − hp_frac(side_two))`,
computed straight off the state and side-one-oriented by construction, and side
two's PUCT flips Q (`MoveStats::puct`, `for_side_one`). There is no
seat-dependent conversion that can be forgotten because there is no conversion.

That is the argument. The measurement is
`tests/test_multiply_chance_search.py::test_seat_swap_reflects_about_one_half`,
now a permanent gate: search a deliberately lopsided position and its
seat-swapped twin at depths 1/2/4/6 and assert the root values sum to 1, that the
swap flips which side is favored, and that each invested arm's Q reflects about
0.5. On the fixture used for the probe the root values summed to 0.9996, 0.9994,
1.0001 and 0.9999 at d1/d2/d4/d6.

**No `leaf_eval="model"` cell was run in this study**, so nothing here inherits
the inversion. The model-arm numbers quoted for contrast come from
`docs/mcts_degradation_findings.md` and carry whatever the inversion did to them
— which is one more reason the comparison here is deliberately slope-only.

### 2.6 The null is clean

The raw-vs-raw control lands at **0.496 [0.448, 0.545]** over 400 mirrored games.
The interval contains 0.500 and is four times tighter than the degradation doc's
control. Seat mirroring works; the deficits and gains above are not harness bias.

A second, independent check on the instrument, measured rather than asserted: the
same harness, same 400 seeds, same raw opponent, with **max-damage** in the
candidate seat scores **0.062 [0.043, 0.091]** (p1 0.065 / p2 0.060 — seat-clean
again). That places the arm correctly on the ladder:

    max-damage 0.062  <  hc-d1 0.196  <  hc-d4 0.328  <  raw ≈ 0.500

A one-ply HP-fraction search is a smarter greedy than max damage and much weaker
than the trained policy — exactly where a handcrafted agent belongs — and every
ply of tree moves it up that ladder rather than down it. The calibration cell is
committed alongside the grid
(`docs/audit_artifacts/hc-depth-grid-20260729/calibration/`).

---

## 3. Depth actually reached — the §4.3 loose end, closed

`docs/mcts_degradation_findings.md` §4.3 flagged that `d6-s1024` and `d8-s1024`
produced identical per-seed outcomes and asked whether the tree ever reaches the
cap. The crate already counts it (`max_depth_reached`); nothing had read it.

Reading convention: the counter is a **node depth with the root at 0**, and the
cap bounds child *creation* (`depth + 1 >= max_depth`), so a cap `d` that binds
tops out at `d - 1`.

| cell | cap | ceiling if binding | max reached | mean reached | histogram (node depth → world-searches) |
|---|---|---|---|---|---|
| hc-d1 | 1 | 0 | 0 | 0.000 | `{0: 66716}` |
| hc-d2 | 2 | 1 | 1 | 0.999 | `{0: 102, 1: 71026}` |
| hc-d4 | 4 | 3 | 3 | 2.508 | `{0: 97, 1: 364, 2: 32923, 3: 35640}` |
| hc-d6 | 6 | 5 | 5 | 2.666 | `{0: 98, 1: 374, 2: 32965, 3: 26403, 4: 7468, 5: 1724}` |
| hc-d8 | 8 | 7 | 7 | 2.672 | `{0: 98, 1: 374, 2: 32965, 3: 26403, 4: 7468, 5: 1381, 6: 264, 7: 79}` |

(~17k decisions × 4 belief worlds per cell.)

**The cap does bind — and that is not the point.** At every cap the deepest node
sits exactly at `d - 1`, so the knob is wired correctly all the way to d8. But
the *distribution* barely moves: mean node depth goes 2.508 → 2.666 → 2.672
between caps 4, 6 and 8, and the deep tail is vanishing — 1724/69032 = 2.5% of
world-searches reach depth 5, and 79/69032 = 0.11% reach depth 7.

The d6/d8 histograms make the mechanism explicit: the 1724 traversals that d6
stops at depth 5 are exactly the 1381 + 264 + 79 that d8 lets continue. Same
tree, one row longer, and those extra visits are far too few to move a root visit
argmax fed by 4096 simulations.

So the answer to §4.3 is not "the cap does not bind" but the sharper:

> **At s1024 the binding constraint is the simulation budget, not the depth cap.**
> Beyond node depth ~3 the subtree is too visit-starved to change a decision.
> Depth ≥ 4 is strength-inert *because the tree cannot afford to use it*.

Confirmed at the outcome level: **hc-d6 and hc-d8 produce the identical winner on
400/400 seeds**, and hc-d4 differs from hc-d6 on 2 of 400. The degradation doc's
d6≡d8 observation reproduces exactly on this arm, and now has a cause.

The budget explanation is testable in one move — change the budget and the depth
distribution should move. It does. Same cap of 6, three budgets:

| cell | mean node depth | share at the cap (depth 5) |
|---|---|---|
| hc-d6-s256 | 1.936 | 136 / 67 824 = 0.2% |
| hc-d6-s1024 | 2.666 | 1 724 / 69 032 = 2.5% |
| hc-d6-s4096 | 3.471 | 11 519 / 74 556 = 15.5% |

The cap is inert at s256, marginal at s1024 and genuinely binding at s4096. The
depth knob is not broken; at the budget the degradation study used, it was
mostly not being spent.

---

## 4. Second axis: simulation budget

`docs/mcts_degradation_findings.md` §4.2 reports the other half of the paradox —
*more* simulations into the same model made search *worse* (d6 0.360 → 0.360 →
0.290 from s1024 to s4096; d4 0.310 → 0.270 → 0.270 from s2048 to s8192). The
handcrafted arm was run on that axis too, same 400 seeds.

| cell | score | Wilson 95% | s/decision |
|---|---|---|---|
| hc-d6-**s256** | 0.255 | [0.215, 0.300] | 0.0104 |
| hc-d6-**s1024** | 0.328 | [0.283, 0.375] | 0.0197 |
| hc-d6-**s4096** | 0.360 | [0.314, 0.408] | 0.0512 |
| hc-d4-s1024 | 0.328 | [0.283, 0.375] | 0.0195 |
| hc-d4-**s4096** | 0.371 | [0.325, 0.420] | 0.0498 |
| hc-d2-s1024 | 0.289 | [0.247, 0.335] | 0.0144 |
| hc-d2-**s4096** | 0.319 | [0.275, 0.366] | 0.0180 |

Paired over the same seeds: d6 s256→s1024 p = 0.0046, s256→s4096 p = 0.0006;
s1024→s4096 is directionally right but not individually significant (52 vs 65,
p = 0.27). Every depth improves with budget; none degrades.

**Second axis, same reversal.** With a handcrafted leaf, both knobs the
degradation study found to be *negatively* valued — depth and simulations — are
positively valued, monotonically. Two independent axes with the opposite sign is
a much stronger statement than one, and it is hard to reconcile with any account
that puts the fault in the tree: a tree that mis-backs-up would waste extra
simulations too.

Depth still saturates near 4 even at the larger budget (d6-s4096 0.360 vs
d4-s4096 0.371, 10 vs 5 discordant, p = 0.30) — more budget buys depth *usage*
(§3) without buying much more strength from it. That is a property of the
handcrafted value, which stops carrying useful signal a few plies out, and is
exactly the kind of thing a learned value is supposed to fix.

---

## 5. Verdict

**Fork (b). The defect is not in the search structure.**

Given a value function that is not the network, the identical tree converts extra
compute into strength on **both** axes the degradation study found to be
negatively valued — depth (d1→d4 +13.2 points, paired p < 0.0001) and simulation
budget (d6 s256→s4096 +10.5 points, paired p = 0.0006) — and it stops improving
with depth precisely where the tree stops growing rather than at some earlier
point. Decoupled selection, virtual loss, the chance-node exact-expectation
backup and the depth mechanics are therefore not intrinsically
depth-degrading. A structural defect of the kind that could produce
0.53 → 0.36 while also making 8× the simulations *worse* would have to corrupt
this arm too, and it does not: this arm has the opposite sign on both axes.

That halves the hypothesis space. What survives, in the order I would test it:

1. **The learned leaf value's orientation — already found by another lane.**
   While this ran, the parity lane (PR #937) identified a **seat-constant
   inversion**: on p2 roots, model leaf values entered the tree unreflected. That
   is a value-side defect, discovered independently, and it is exactly the class
   this experiment localizes to. `docs/mcts_degradation_findings.md` §9 listed the
   mirrored-state orientation check as "the cheapest untested candidate"; it was
   the right instinct and it has now paid out. The open question is no longer
   *whether* the value path had an orientation bug but *how much of the depth
   decay it accounts for* — the honest answer needs the grid re-run on the fixed
   value path, because the published grid is a mix of one correct seat and one
   inverted seat.
2. **The learned leaf value under depth, beyond orientation.** Even with the
   inversion fixed, deeper leaves sit further off the distribution the value head
   trained on. If the re-run flattens the decay, this is moot; if a residual slope
   survives, this is where it lives.
3. **The model-prior/PUCT coupling.** This experiment does *not* exonerate it.
   The handcrafted arm runs with **uniform** priors, because `model_priors` has no
   meaning without a model. So what has been cleared is the tree *given uniform
   priors*. If the acting side's learned priors concentrate visits onto arms whose
   deep values are mispriced, the tree would be fine and the pairing would still
   decay. An ablation exists and is cheap: model leaves with `model_priors=False`.

An honest alternative reading, which I cannot exclude and which does not change
the verdict: it is possible that extra plies help a *weak* value function and hurt
a *strong* one, so the sign difference could track value quality rather than
"learned vs handcrafted". Note that this is still a statement about the value
side, not the tree. Either way, the tree is not where the fix is.

---

## 6. Design, and every deviation from the degradation doc's cells

Held identical:

- head-to-head vs the raw policy of the same checkpoint, seats mirrored by seed
  parity, seeds from 600000, `worlds=4`, `sims=1024`, `c_puct=1.4`,
  `deep_ko_split=true`, belief sampling on (`POKEZERO_BELIEF_SET_SOURCE=1`);
- the same `traverse`/`finalize`, the same per-world visit-share aggregation, the
  same `_map_choices` correspondence, the same loud fallback taxonomy.

Deviations, all forced and all on the record:

| deviation | why | effect on the slope read |
|---|---|---|
| **poke-engine's native `monte_carlo_tree_search` was NOT used** — the pre-existing `leaf_eval="hp_fraction"` mode is a time-budgeted search with no depth or sims knob. A new `hp_fraction_crate` mode was added. | The native MCTS would have changed the tree *and* the value at once, confounding the exact question being asked. | This is the whole point: only the leaf changed. |
| **Sequential driver** (`LeafPrice::Ready`), so `search_batch` is inert. The model arm uses batch-16 virtual-loss batching. | Handcrafted leaves are priced inline; there is nothing to batch. | The crate's own gate proves `b=1 ≡ sequential`. If anything the hc arm is the cleaner of the two. |
| **Uniform priors.** | No model, no priors. | Documented above as an explicit hole in the exoneration, not papered over. |
| **Raw opponent is `pz-v2-2-1m` (v2.2, 1M), not `v3hist-k64-enthalf-5m` @ 4.25M.** | The 5M checkpoint lives on the cluster; this ran locally. | Changes the level, not the shape. The opponent is fixed across every cell, so the within-experiment slope is unaffected. Cross-study comparison is slope-only by construction. |
| **n = 400** rather than 100. | The arm is 30–200× cheaper per decision, so the doc's ±0.10 half-width was avoidable. | Both windows are reported; the 100-seed window is the comparable one. |
| **d8 added.** | §4.3's open question is about d6 vs d8 specifically. | Bonus cell. |

Games that end with no winner (the 250-round decision cap) score 0.5. They are
rare and evenly spread: 3 in the control, 1–4 per handcrafted cell out of 400.

Fallback stayed in the 0.5–0.6% band the degradation doc reports, except hc-d1 at
2.03%. That cell's excess is one attribution bucket —
`self_moveset_mismatch … 'hiddenpower'`, 2560 world failures — and it is a
consequence, not a cause: d1 plays different moves, so it reaches different
battles. 1.5 percentage points of extra uniform-random decisions cannot account
for a 13-point spread, and the effect would push d1 *down*, i.e. away from the
direction that would fake a rising slope. It is reported rather than swept.

---

## 7. Reproducing

```sh
# venv outside the repo; crate + engine built from THIS worktree's HEAD
scripts/vendor_poke_engine_src.sh   "$VENV/bin/python"
scripts/setup_poke_engine.sh        "$VENV/bin/python"
(cd rust/pokezero-search && "$VENV/bin/maturin" build --release -i "$VENV/bin/python")

PYTHONPATH=src POKEZERO_BELIEF_SET_SOURCE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/hc_depth_grid.py \
    --checkpoint checkpoints/pz-v2-2-1m.pt \
    --showdown-root "$SHOWDOWN_ROOT" \
    --cells control,hc-d1,hc-d2,hc-d4,hc-d6,hc-d8 \
    --seed-start 600000 --games 400 --sims 1024 --worlds 4 --record-depths \
    --out docs/audit_artifacts/hc-depth-grid-20260729

# sims axis (§4): same command with --sims 256 / 4096 into its own directory,
# and the max-damage calibration cell (§2.6):
#   --cells "vs:max-damage" --out .../hc-depth-grid-20260729/calibration

python scripts/hc_depth_grid_report.py \
  --dir docs/audit_artifacts/hc-depth-grid-20260729 \
  --dir docs/audit_artifacts/hc-sims-grid-20260729/s256 \
  --dir docs/audit_artifacts/hc-sims-grid-20260729/s4096
```

`--dir` is repeatable; cells are labelled from the sims budget they recorded, so
folding the axes together cannot silently compare two different configurations
under one name.

Per-cell JSON — per-seed rows (seed, seat, winner, score, per-decision
`max_depth_reached` series), full engine telemetry including the depth-reached
histogram, and the resolved build provenance — is committed under
`docs/audit_artifacts/hc-depth-grid-20260729/` and
`docs/audit_artifacts/hc-sims-grid-20260729/`.
