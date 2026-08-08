# C149 (prediction, registered before measurement) — split the residual-kill arm per roll where the killing residual is Leech Seed

**This file is the registered prediction for the G8 engine change. It is committed BEFORE the
fixed build is swept and is never edited afterwards; outcomes are appended in a separate file
(`reports/c149_g8_leechseed_band_split.md`) and in separate commits.** That discipline exists
because two patches in this program had a correct mechanism, a green suite, and opened dozens of
rows — one opened 78 to close 1 — and only the sweep caught it (C133 §7).

## 0. What is being changed

`reports/c140_last_dev_row_diagnosis.md` §7a, built. Where the gen 3 partition emits a
residual-**kill** arm *and* the killing residual **transfers HP to the other side** — which in
gen 3 is Leech Seed alone; Nightmare and Ghost-Curse are damage-only — the single arm priced at the
band's threshold becomes **one arm per roll in the band, each at mass `1/16`** of the crit-split
factor.

Why that class and no other: a residual that kills is capped by the HP that happened to be left, so
its magnitude inherits the damage roll. The harness already knows this on the drained side and
tags it `capped_lethal`, which is roll-scaled. Leech Seed **transfers** that capped amount, and it
arrives on the other side as a bare silent `heal` that the comparator checks **exactly** (c140 §5).
Over the lethal band the drain `min(maxhp/8, hp_after_move + leftovers)` is **injective in the
roll**, so one arm prices exactly one of the band's rolls and every other roll has no arm at all
(c140 §6a, ledger G8).

### Scope — two call sites, verified at source on this tree

`third_party/poke-engine-src/src/gen3/generate_instructions.rs`, at `1c94f071`:

| `residual_disjoint_bands(...)` call | `ceiling` arg | loop var | in scope |
|---|---|---|---|
| `:4073` (loop `:4085`) | `defender_active.hp` | `num_residual_only` | no |
| `:4176` (loop `:4185`) | **`i16::MAX`** | `num_residual_kill_rolls` | **yes** |
| `:4259` (loop `:4268`) | `defender_active.hp` | `num_residual_only` | no |
| `:4309` (loop `:4317`) | **`i16::MAX`** | `num_crit_residual_kills` | **yes** |

Read off the file, not carried from a handoff; `grep -n 'residual_disjoint_bands'` over that file
returns exactly those four call sites and the definition at `:1976`. The `i16::MAX` ceiling means
*this fan cannot kill on the hit*, so there is no hit-KO arm and every roll in a band dies to the
residual. The two `defender_active.hp` sites sit under a hit-KO arm that already owns the rolls at
or above the defender's HP.

**The two excluded sites are excluded for blast radius, not because they are immune.** The same
capped-lethal-drain arithmetic runs there. This prediction does not claim otherwise.

### Implementation, and the two properties that make it safe

* **Gate.** The defending side must carry `PokemonVolatileStatus::LEECHSEED`. Read from
  `defender_side` *before* `get_active()`, because `get_both_sides` hands back `&mut Side`.
* **Roll values come from the EXACT INTEGER fan** `floor(max * r / 100)`, `r` in `85..=100` — the
  expression `push_enumerated_rolls` and the shipped 32-arm `pending_hp_reading_move` enumeration
  already use. **Not** the f32 accumulator in `compare_health_with_damage_multiples`, which drifts
  below the true rung (C116 M5) and would price arms at damage amounts Showdown cannot deal.
* **Mass is still priced by the comparator's own count.** Each band keeps the roll count
  `residual_disjoint_bands` gave it, and the split emits exactly that many arms at `1/16` each.
  Before pushing anything, the helper checks that the integer fan really does put that many rolls
  in the band's half-open window `[t_i, t_i+1)`; **if it does not, it pushes nothing and the caller
  emits the single collapsed arm it emits today.**

That fallback is a **reachable path, measured, not a theoretical one**: over
`max_damage` in `10..=600` and every threshold strictly above the f32 fan's floor,
**288 of 27,318 `(max, threshold)` windows (1.054 %) disagree**, touching 256 of the 591
`max_damage` values. Derived in this branch by reproducing both counters in Python with
`struct`-backed float32; the glob is that rectangle and nothing wider. The consequence is the
property that matters here: **on a disagreeing band the engine keeps exactly the arm it has today,
including the case where that arm happens to sit on the observed roll — so the split is a strict
improvement or a no-op on any given band, never a trade.** That is the concrete difference from
route 7c, which c140 §6a(ii) measured as a wash.

## 1. Baselines this prediction is measured against

Both on `1c94f071`, tree clean, **73 patches**, fingerprint `de29e3dc79c80659`, `--matcher strict`,
`--games 200`, exit captured directly and never through a pipe.

| window | seeds | measured | matched | diverged | engine_errors | exit | artifact |
|---|---|---|---|---|---|---|---|
| dev | 19,000,000–19,000,199 | 15,503 | 15,502 | **1** | 0 | 1 | `reports/artifacts/c149_base_dev_sweep.json` |
| validation holdout | 19,100,000–19,100,199 | 15,579 | 15,579 | **0** | 0 | 0 | `reports/artifacts/c149_base_holdout_sweep.json` |

The dev divergence is `divergence_classes {'component_magnitude:heal': 1}`, repro `(19000191, 63)`.

**The dev baseline was run twice on this tree and the two agree.** A handoff run and an independent
re-derivation in this branch differ on exactly **three** leaves out of the whole artifact, compared
with no filter at all: `/elapsed_seconds` (493.99 → 488.55), `/games_per_hour` (1457.5 → 1473.7) and
`/repros[0]/protocol[1]`, a `|t:|` Unix-timestamp protocol line. Every counter, every verdict scalar
and the whole `checkpoint_provenance` block are identical. The committed artifact is the
re-derivation, not the handoff.

**The sweep exits 1 whenever any divergence exists.** That is the harness verdict
(`return 1 if (transitions_diverged or engine_errors or partition) else 0`), not a crash, so the dev
baseline legitimately exits 1 and the fixed dev sweep is predicted to exit **0**.

Residue is quoted only with its accept bar: **~9 % of measured boundaries are accepted via up to 64
enumerated hidden sleep-counter worlds** — dev `gating:support` 1,347/15,503 = **8.689 %**, holdout
1,431/15,579 = **9.185 %** — and the enumeration's coverage of the sleep-counter space is
**87.5 %**, not the ~96.6 % the metric self-reports.

## 2. The predictions

### 2a. Row level — `19000191/63`

Measured on the base build in this branch through the shipped
`cert_sweep_reread.reread_row` (which calls the shipped `evaluate_boundary_strict`; nothing is
reimplemented): **14 branches, `diverged`, 12 misses**, four of them
`observed_only=[('heal', 28)] engine_only=[('heal', 29)]` at 34.61 / 2.88 / 2.31 / 0.19 %.
That reproduces c140 §1 exactly.

| | predicted on the fixed build |
|---|---|
| verdict | `matched` |
| misses | 0 |
| **branch count** | **38** |

38 is re-derived here rather than carried from c140's arithmetic estimate, and it happens to agree
with it. The band is the seven rolls `[108, 109, 110, 111, 112, 113, 115]` of Hidden Power Grass's
115-max fan at threshold 108 — all seven are integer-fan members and the f32 count is also 7, so the
count guard passes and the split fires. Four upstream arms (Thunderbolt's paralysis × crit splits)
each carry one residual-kill arm, so `14 − 4 + 4 × 7 = 38`.

**If the branch count is not 38, the mechanism as described here is wrong** even if the row closes,
and the report must say which of the two claims failed.

### 2b. Dev window — exact counter deltas

Every key from `reports/artifacts/c149_base_dev_sweep.json` `counters`. Predicted **changed**:

| counter | base | predicted |
|---|---|---|
| `transition:diverged` | 1 | **0** |
| `transition:matched` | 15502 | **15503** |
| `divergence_class:component_magnitude:heal` | 1 | **key absent** |
| `strict:diverged_on_full_branch_set` | 1 | **key absent** |

Predicted **unchanged** — and this is the substance of the prediction. These are all of the
remaining keys, enumerated rather than summarised:

`boundaries_full_round` 15968 · `boundaries_measured` 15503 · `gating:exact` 14156 ·
`gating:support` 1347 · `hidden_counter_support:confusion` 1 · `hidden_counter_support:sleep` 1352 ·
`limit:world_substitute_health_unknown` 131 · `skip:single_seat_boundary` 1742 ·
`skip:unmappable_choice:struggle_not_submittable` 118 ·
`skip:world_unsupported:encore_move_unknown` 2 ·
`skip:world_unsupported:materialization_blocker` 18 ·
`skip:world_unsupported:self_request_state_unsupported` 13 ·
`skip:world_unsupported:volatile_unsupported` 144 · `strict:sleeptalk_union_branch` 126 ·
`world_prestate_mismatch` 39 · `world_prestate_mismatch:p1_hp` 7 ·
`world_prestate_mismatch:p1_status` 14 · `world_prestate_mismatch:p2_hp` 13 ·
`world_prestate_mismatch:p2_status` 5.

Plus: `engine_errors` 0, `divergence_classes` `{}`, no new key of any kind, and sweep exit **0**.

Rationale for "unchanged": the games are replayed from Showdown at fixed seeds, so
`boundaries_full_round` cannot move; the `skip:*`, `limit:*`, `world_prestate_mismatch*` and
`gating:*` families are decided from the observation and the world enumeration before the engine's
branch set is consulted, so they cannot move either. `strict:sleeptalk_union_branch` is a
Sleep-Talk-specific union and this change touches no Sleep Talk path.

### 2c. Validation holdout window — exact counter deltas

**Predicted: the whole `counters` block byte-identical to
`reports/artifacts/c149_base_holdout_sweep.json`**, i.e. 15,579 measured / 15,579 matched /
**0 diverged** / 0 engine_errors, `divergence_classes {}`, exit **0**.

Zero rows closed on the holdout is the *registered expectation*, not a disappointment: the holdout
baseline has no divergence left to close, so this window is a **safety measurement only** and it
cannot corroborate the mechanism. A reader who wants to falsify the mechanism must attack §2a and
§2b; §2c cannot discriminate.

## 3. The falsifiers

**F1 — nothing opened (the primary one).** Any boundary that matched on the base build and diverges
on the fixed build, on **either** window. Operationally: holdout `transition:diverged` > 0, or dev
`transition:diverged` > 0 (the dev target row closing must take the count to exactly 0, not to 1
with a different class), or any `divergence_classes` key on the fixed build that is not on the base
build, on either window.

**F2 — engine health.** `engine_errors` > 0 on either window, or any `COUNTER INTEGRITY:` line on
stderr (the `verdict_partition_failures` self-check), or a branch-mass sum that is not
100.000000 % on the target row.

**F3 — the mechanism.** `19000191/63` still `diverged`, or its branch count is not 38.

**F4 — the gate.** A non-Leech-Seeded boundary whose branch count changes. Pinned by a crate test
rather than by the sweep, because a verdict-level sweep cannot attribute a branch-count change to a
call site.

**If F1 or F2 fires, this change is WITHDRAWN and the report says so.** Two withdrawals in this
program were the right call. The honest fallback is then a limit with a written demonstration, and
that demonstration has to argue why *no* change should ship when c140 measured three that can —
which is the harder demonstration, and the reason 7a was preferred over 7b and 7c.

If F3 fires but F1 and F2 do not, the change is not withdrawn but the mechanism claim is, and the
report leads with that.

## 4. What this prediction does NOT claim

* It does not claim the two `defender_active.hp`-ceiling sites are unaffected by the same defect.
  They are excluded for blast radius. Unmeasured.
* It does not claim anything about `19200244/115`, the second confirmed G8 instance in the **final
  holdout** (ledger G8, `reports/c143_heal_attribution_diagnosis.md`). That row's arm is priced at
  the **survive representative**, not at a residual threshold — every threshold there lies below the
  fan minimum, so `residual_disjoint_bands`'s `min_roll < threshold` guard cannot pass and this
  change cannot reach it. The final holdout is not swept here and must not be.
* It does not claim a measured firing rate for the split across the two windows. The sweep is
  verdict-level and carries no counter for "the split fired"; how often the gate is true is
  **unmeasured**.
* It does not claim the count-guard fallback is exercised by the two windows. Its reachability is
  measured arithmetically (§0), not observed in a sweep.

## 5. Provenance of this file

Written and committed against `1c94f071` plus this branch's engine edit, with **no sweep of the
fixed build yet run**. The base-build row replay in §2a and the fan-disagreement census in §0 were
both run before this file was committed and are stated as measurements, not predictions.
