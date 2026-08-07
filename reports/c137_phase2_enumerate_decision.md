# C137 — C116 Phase 2 decision: adopt roll enumeration for the differential harness only

The decision C116 assigned and the program has deferred since C119. It is taken here
because the measurement C116 required now exists. C119 declined for a reason that was
correct at the time and is now stale; §2 says why.

**Decision: option (b), adopt for the differential harness only.** Not a compromise
between (a) and (c) — the three measurements separate cleanly and each option is decided
by one of them.

## 1. What was measured

Runtime flag `POKEZERO_ENUMERATE_ROLLS`, read once via `OnceLock`, off by default, and
deliberately **not** a cargo feature — so one build serves both paths and every
comparison below is single-variable. The enumerated path emits one arm per distinct
`floor(max * r / 100)` for `r ∈ 85..=100` at mass `1/16`, pre-merged on equal integers,
bypassing the Case A / Case B partition cascade and not calling the residual-threshold
mirror at all.

This generalises a mechanism the engine **already ships**: the 32-arm enumeration used
for `pending_hp_reading_move` (Flail, Reversal, Substitute, Belly Drum, Pain Split). It
is not a new idea, it is an existing one applied to the general case.

### Acceptance criterion 1 — secondaries compose (C119's objection)

C119 objected that `count/16` cannot express a probabilistic secondary. It composes for
free, because `run_move` fans *each* arm through `get_instructions_from_secondaries`, so
masses become `count/16 × branch probability` **by construction** — there is no recipe to
hand-derive. Charizard Fire Blast (85 acc, 10 % burn, 1/16 crit) into a defender that
survives every roll, reconstructed independently in Python from `calculate_damage`'s two
scalars:

| | branches | predicted (damage × burn) cells matched |
|---|---|---|
| collapsed | 5 | **0 of 64** — the whole non-crit fan is one arm at 145, a value no legal roll deals |
| enumerated | 34 | **64 of 64**, to 1e-9; masses sum to 100 |

### Acceptance criterion 2 — A8's ordering concern evaporates

A8 worried that the residual threshold is evaluated before secondaries resolve. That is
a property of the **mirror**, and this deletes the mirror. Fixture where the burn the
move itself inflicts is what kills the low rolls (defender 206/404, tick 50, non-crit
rolls 133–157): of 32 outcomes, 16 lethal on the hit, **1 lethal only if the burn lands**,
15 never lethal.

| | independent KO mass | engine |
|---|---|---|
| truth | 5.810547 % | — |
| collapsed | | **5.312500 %** — short by exactly the burn-dependent roll |
| enumerated | | **5.810547 %, delta 0** |

The relocation question does not exist to be answered: per-roll outcomes are computed
where they happen. The arm at damage 157 + burn carries a tick the engine clamped 50 → 49
to land exactly on 0 HP.

### Measurement (i) — residue, both windows, 200 games each

Same build, same seeds; `boundaries_measured` and `boundaries_full_round` match to the
unit, confirming identical trajectories.

| window | | full_round | measured | diverged | engine_errors |
|---|---|---|---|---|---|
| dev `19,000,000–199` | collapsed | 15,968 | 15,503 | **2** | 0 |
| dev | **enumerated** | 15,968 | 15,503 | **0** | 0 |
| holdout `19,100,000–199` | collapsed | 16,155 | 15,579 | **4** | 0 |
| holdout | **enumerated** | 16,155 | 15,579 | **2** | 0 |

**Nothing opened on either window.** Closed: `19000074/27` (crit-straddle),
`19000191/63` (collapsed lethal arm), `19100107/135` and `19100191/5` (both
`limit:roll_divergent_lethality`). That is **all four collapse-class rows**, matching
C134 §3's "up to 4 of 5". The remaining two are the `itemleftovers` pair at `19100170`,
which enumeration does not touch and which has its own harness fix.

Harness cost: 705 s / 707 s enumerated against 743 s / 746 s collapsed, all four run
concurrently on one box. **Enumeration is free for the harness** — the sweep is
Node-bound, not `generate_instructions`-bound.

### Measurement (ii) — mass gate

All mass assertions pass under the flag. One test fails,
`test_matrix_is_not_vacuous`, and it is **not** a mass error: it asserts that at least one
fixture leaves a fan collapsed, as its own negative control. Under enumeration no fan is
ever collapsed, so the assertion is unsatisfiable by construction. Under harness-only the
collapsed configuration still exists, so the control is re-expressed against it rather
than weakened.

`engine_behavioral_probes.py` under the flag: 24 pass, 14 fail. All six
`residual-mass-*` probes — the independent-reconstruction family — pass. Every one of the
14 failures is an arm-**structure** assertion reading "expected 4 branches, got 18", with
correct masses. Those 14 encode the collapse and would need rewriting **only under
adopt-everywhere**; under harness-only search keeps the collapse and they stay green.

### Measurement (iii) — search throughput, depth 4 / 1024 sims

| position | collapsed | enumerated | ms/decision |
|---|---|---|---|
| minimal_1v1 | 3.32 M sims/s | 18,888 | 0.31 → 54.2 |
| **midgame_3v3** | 431 K sims/s | **115** | **2.38 → 8,881.8** |
| endgame_straddle | 3.76 M sims/s | 788,654 | 0.27 → 1.30 |

Two independent baseline runs bracket each other, so the ratios are not a load artifact.
The production-representative position regresses **~3,700×**, to 8.9 seconds per
decision, with 229× the leaf evaluations — despite the existing mitigation that
`branch_on_damage = depth < DAMAGE_BRANCH_DEPTH` already restricts enumeration to plies
1–2 plus deep-KO straddles.

## 2. Why each option is decided, and why C119 is stale

- **(a) adopt everywhere — rejected on measurement.** 8.9 s per decision at the
  production config is not a tuning problem.
- **(c) reject — cannot beat (b).** Rejecting costs the four rows (b) closes, and (b)'s
  throughput risk is **zero by construction**: the flag is a runtime env read on one
  build, so search takes the collapsed path bit-identically. A reject case would have to
  beat "four of five residue rows retired at no throughput cost", and nothing does.
- **(b) adopt harness-only — taken.**

C119 scoped this honestly for its era: 2 of 25 rows firmly absorbed, 3 conditional, 20
untouched — "about a fifth", and it was right to keep burning rows instead. That is now
stale by survivorship: the 20 untouched rows were the ones fixable by other means, and
they got fixed. What remains is dominated by the class enumeration is best at. C119's own
pre-registered prediction (2 firm, up to 5) was conservative in the direction it said it
would be.

## 3. What this decision also settles

- **The two `limit:roll_divergent_lethality` rows do not need a written demonstration.**
  They close. C134 §3 anticipated "enumeration fixes them or *constitutes* the
  demonstration"; the measurement says fixes. See `reports/c135_…` §7.
- **The queued crit-straddle sub-split for `19000074/27` should not be written**, and the
  status-aware threshold sketched in c135 §5 should not be implemented. Enumeration
  closes both rows without either.
- **The C134 §3 freeze has served its purpose and lifts.** It was correctly placed: this
  family had already burned three wrong hand-derived mass recipes, and the freeze
  prevented a fourth.

## 4. What this does NOT settle, stated

- **The f32 comparator (C116 M5) still executes in search.** Harness-only removes it from
  the fidelity path entirely — no fidelity claim passes through
  `compare_health_with_damage_multiples` again — but it still runs during search. M5 was
  re-derived independently from the shipped expression over max_damage 1..400: **173** max
  values where the top rung lands below `floor(max)`, **195** `(max, threshold)`
  kill-count mismatches, 22 at interior thresholds, **195 undercounts and 0 overcounts**;
  at max = 120 with threshold 108 it counts 10 kill rolls against a true 11. Every C115 /
  C116 figure confirmed. Closing it fully needs C116(c)'s remedy — rewrite the
  comparator's body in integers as `max * r // 100` — which is a small change to one
  function and independent of this decision.
- **Multi-hit semantics changed on the enumerated path** and were not separately pinned:
  enumeration applies a per-hit roll shared across hits, replacing the collapsed path's
  total→per-hit conversion. No multi-hit row was in either residue and no fixture
  regressed, but this is arguably a fix to the filed "multi-hit shared damage roll" gap
  and should be confirmed as one rather than assumed.
- **`test_matrix_is_not_vacuous` must be re-expressed, not deleted**, and the
  re-expression must still be able to fail.
