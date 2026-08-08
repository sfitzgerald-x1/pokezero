# C147 — registered prediction: the G33b residual-bucket gate, built in `ResidualPlan`

**Registered before any measurement of the gate exists.** Committed as its own commit, ahead of
the two gate sweeps and ahead of the row replay. §1–§7 are frozen; outcomes are appended in §8
and nothing above it is edited.

## 0. What is being predicted, and what these two windows can and cannot show

G33b is a renderer defect diagnosed and merged as a finding in PR #1161 (`1a929c57`),
`reports/c138_known_gaps_ledger.md` row **G33b**, mechanism in
`reports/c143_heal_attribution_diagnosis.md` §1–2 and §6. Its closure was measured there as a
**model applied to the renderer's output**, on one row of a **spent** final holdout
(`19200244/115`), under the enumeration oracle. This branch builds the gate as a real change in
`ResidualPlan::build` and runs C133 §7's discipline over it.

**The dev and validation-holdout windows cannot show a closure, and I am predicting that they do
not.** That is not a hedge added after the fact, it is arithmetic on the registered baseline
below: the dev window carries exactly **one** divergence and it is a G8 *magnitude* row
(`19000191/63`, class `component_magnitude:heal`) whose Leftovers label is already correct —
c143 §1a variant B, where the seeder's tick does appear after the faint. The validation holdout
carries **zero** divergences. There is no G33b row in either window to close.

So the two sweeps are a **safety** measurement, and the prediction they test is the
**"nothing opened"** falsifier. The gate's *closure* evidence is a separate measurement, §5: the
recorded row replayed from a committed artifact on both builds. Keeping the two apart is the
point — c143 §6 already recorded that a relabelling can only ever *widen* what matches, so a
green sweep vindicates nothing on its own and must be paired with a positive.

## 1. The gate, as built

`rust/pokezero-search/src/events.rs`, `leftovers_slot_truncated`, consumed at the one booking
site in `ResidualPlan::build`:

```rust
if active.item == Items::LEFTOVERS && !leftovers_truncated[i] {
    plan.heal[i].push("item: Leftovers".to_string());
}
```

The predicate walks the residual segment for the instruction that ends the battle and asks one
question — was the winner's order-**10.4** item slot behind that point? Its five arms are taken
from the engine's own section order in `add_end_of_turn_instructions`, and only two gate:

| what delivered the battle-ending faint | section | winner's 10.4 | gated |
|---|---|---|---|
| the shared weather entry | order 8 | not reached | yes |
| the loser's own 10.5 / 10.6 / 10.9 | order 10, loser's bucket | not reached **iff the loser is faster** | on that condition |
| the winner's 10.5 Liquid Ooze recoil | order 10, winner's bucket | already fired | no |
| Future Sight | order 11 | already fired | no |
| Perish Song | order 12 | already fired | no |

Two deliberate under-reaches, both retaining the pre-gate booking: an exact **speed tie** (the
engine forks both orders and keeps both when they differ, so there is no single answer), and a
**fatal weather chip with the winner faster** (a real instance of the same family, left unshipped
because it is unmeasured — the speed condition is what keeps it out).

## 2. The registered baseline, and its provenance

| | dev `19,000,000–19,000,199` | validation holdout `19,100,000–19,100,199` |
|---|---|---|
| artifact | `reports/artifacts/c142_base_dev_sweep.json` | `reports/artifacts/c142_base_holdout_sweep.json` |
| `boundaries_full_round` | 15968 | 16155 |
| `boundaries_measured` | 15503 | 15579 |
| `transitions_matched` | 15502 | 15579 |
| `transitions_diverged` | **1** (`19000191/63`) | **0** |
| `engine_errors` | 0 | 0 |
| `divergence_classes` | `{"component_magnitude:heal": 1}` | `{}` |
| `gating:exact` | 14156 | 14148 |
| `gating:support` | 1347 | 1431 |

Those two artifacts are on `main` and were measured at engine fingerprint
`5fa147ffa325c8872d47fbe3645125af9dac2c94f3b84ca6dc8be5f96539d341`. **`main` at `e0a23e4e` is
byte-identical to them on every input the fingerprint covers**: a fresh worktree of `e0a23e4e`
built with `scripts/build_search_crate_engine.sh` (exit captured directly, `exit=0`) reports
`engine build is current (71 patches, 5fa147ffa325c887)`, and the fingerprint's inputs include
every `.rs` under `rust/pokezero-search/src` as well as the patch stack
(`scripts/engine_build_fingerprint.py:93,153,168`). So the baseline is same-build, not merely
recent. A **same-build re-sweep of both windows is nonetheless run as a control**, and §8 records
whether it reproduces the table above. The gate build is
`a0796bdd0629f8c3babba7dbfc88d0f4c93dffd4c5379803110a438f75cd7545`.

If a quoted residue count is read out of these sweeps: the accept bar is C133/Constraint 7's, in
which ~9 % of measured boundaries are accepted via up to 64 enumerated hidden sleep-counter
worlds — `gating:support` **1,347 / 15,503** dev and **1,431 / 15,579** holdout.

## 3. Predictions

**P1 — dev window, gate build.** Every scalar in the dev column of §2 is **unchanged**:
`boundaries_full_round` 15968, `boundaries_measured` 15503, `transitions_matched` 15502,
`transitions_diverged` **1**, `engine_errors` 0, `divergence_classes`
`{"component_magnitude:heal": 1}`, `gating:exact` 14156, `gating:support` 1347. The surviving
divergence is still `19000191/63` and still that class.

**P2 — validation holdout, gate build.** Every scalar in the holdout column of §2 is
**unchanged**: 16155 / 15579 / 15579 / **0** / 0, `divergence_classes` `{}`, `gating:exact`
14148, `gating:support` 1431.

**P3 — the whole counter block is unchanged, not just the verdict scalars.** No `skip:*`,
`strict:*`, `limit:*`, `world_prestate_mismatch*` or `hidden_counter_support:*` key appears,
disappears or moves on either window. Predicted because the gate touches attribution only: it
changes which `[from]` tag a heal carries, never an HP number, never a branch's mass, never
whether a boundary is measurable.

**P4 — rows closed on the two windows: ZERO.** Predicted, in advance, as the *expected* result
rather than the disappointing one, for the reason in §0.

**P5 — the built gate reproduces the modelled closure on the replayed row** (§5).

**P6 — the seven crate pins.** `a_truncated_leftovers_slot_is_not_booked_so_the_drain_stays_silent`
is red on `origin/main` and green here. The other six are green on **both**, because each one is
a case the gate must *not* reach; they are the over-reach guards, and a gate that fired
unconditionally would break all six.

## 4. The falsifier — "nothing opened"

Any one of the following on **either** window falsifies this prediction, and the patch is
**withdrawn** rather than argued for:

1. `transitions_diverged` above its baseline (dev > 1, holdout > 0);
2. `transitions_matched` below its baseline (dev < 15502, holdout < 15579);
3. `boundaries_measured` different from its baseline in either direction;
4. any key in `divergence_classes` that is not in the baseline, at any count;
5. `engine_errors` > 0;
6. any counter key added, removed, or changed in value (P3).

This falsifier is the whole reason the sweep is being run. Two patches in this program have
shipped a correct mechanism behind a green suite and opened rows that only the sweep caught — v1
of the faint-cancels guard opened **78** rows to close **1**
(`reports/c136_faint_cancels_prediction.md`) — and the shape here is the more dangerous
direction, because the gate makes a previously-**unusable** plan reconcile. A side whose
`plan.usable` flips false→true stops receiving the constant fallback and starts receiving
positional labels, so every *other* entry in that side's heal list becomes load-bearing at
exactly the boundaries where it was previously ignored. If any of those entries is wrong, the
gate converts a correct-by-accident label into a wrong one. That is a row-opening mechanism, it
is specific to this change, and clause 1 is what would catch it.

## 5. The closure claim, on the replayed row, built rather than modelled

`19200244/115` is read from the committed artifact `reports/artifacts/c141_final_holdout_sweep.json`
and replayed against the local engine by the committed
`scripts/c143_heal_attribution_probe.py --enumerated --row …`, whose `shipped_renderer` block
calls `cert_sweep_reread.reread_row` through the **installed** crate. **No sweep is run at or
above seed 19,200,000**; this is the same read-and-replay c143 §0 used, on a row that has already
been spent.

| | base build `5fa147ff…` | gate build `a0796bdd…` |
|---|---|---|
| `shipped_renderer.verdict` | `diverged` | **`matched`** |
| `shipped_renderer.branches` | 416 | 416 |
| `shipped_renderer.misses` | 12 | **0** |
| `modelled_g33b_gate.verdict` | `matched` | `matched` |
| `modelled_g33b_gate.soundness.heals_relabelled` | 350 | **0** |

The last cell is the sharp one and it is the point of running the probe on both builds. The
model rewrites p1 heals tagged `[from] item: Leftovers` in truncated arms; if the built gate
covers exactly the set the model covered, the model finds nothing left to rewrite and the count
falls to 0. **A non-zero count on the gate build means the built gate is NARROWER than the
model**, and the difference is then the finding, reported as such rather than smoothed over.

## 6. Reach, and the condition under which the sweeps prove nothing

A sweep that never executes the changed line is the "checks that assert nothing" failure this
program has hit repeatedly. **Verdict-level sweeps cannot measure this gate's reach**, and the
reason is structural rather than incidental: at every site where the gate fires, the baseline
plan was *already* unusable, so any mutation that preserves the booking count reproduces the
baseline's output exactly. The only detectable change at a firing site is the one that makes the
count reconcile — which is the gate itself.

So reach is measured separately, by a **throwaway instrumented build** of this same predicate
that counts firings, swept over both windows. It is not part of the patch. Predicted: a
**non-zero** firing count on at least one window, since the shape needs only a seeded active
dying as its side's last Pokemon while a slower opponent holds Leftovers, and Leftovers is 72 %
of all generated items (`c138` G33b cell).

**If the measured firing count is zero on both windows, I will say so plainly**: the sweeps would
then be silent about the gate rather than supportive of it, P4 would be vacuous rather than
predicted, and the only positive evidence would be §5 plus the seven pins.

## 7. What would make me withdraw the patch

* any clause of §4 firing on either window;
* the built gate failing to close the replayed row (§5), i.e. `shipped_renderer.verdict` on the
  gate build not `matched`;
* the revert-failing pin (P6) turning out to be green on `origin/main` — that would mean it pins
  nothing, and it is verified by checking out `origin/main` into its own worktree and rebuilding,
  never by reasoning from a fingerprint.

A withdrawn patch with a clean falsifier is the intended outcome of this method when the
measurement says so, and two have already been withdrawn under it.

---

## 8. Outcomes — appended after the measurement, with §1–§7 untouched

Nothing above this line was edited. Full write-up:
`reports/c147_g33b_residual_bucket_gate.md`.

| prediction | outcome |
|---|---|
| **P1** dev unchanged (15968 / 15503 / 15502 / 1 / 0, `component_magnitude:heal` 1, 14156 / 1347) | **HELD**, every scalar |
| **P2** validation holdout unchanged (16155 / 15579 / 15579 / 0 / 0, `{}`, 14148 / 1431) | **HELD**, every scalar |
| **P3** the whole counter block unchanged | **HELD** — the gate's `counters` dict is *byte-identical* to the same-build base sweep's on both windows, 23 keys on dev and 21 on holdout |
| **P4** zero rows closed on the two windows | **HELD**, and it was the registered expectation |
| **P5** the built gate reproduces the modelled closure on the replayed row | **HELD** — `shipped_renderer` goes `diverged` / 12 misses → **`matched` / 0 misses** over 416 branches, and `modelled_g33b_gate.soundness.heals_relabelled` falls **350 → 0**, so the built gate covers exactly the set the model covered |
| **P6** one pin red on `origin/main`, six green on both | **HELD** — full crate suite on a separately-vendored `e0a23e4e` worktree: `135 passed; 1 failed`, the failure being `a_truncated_leftovers_slot_is_not_booked_so_the_drain_stays_silent`, printing `left: ["item: Leftovers"] right: []` |
| §4 falsifier, all six clauses, both windows | **did not fire** |
| §6 reach — predicted non-zero, with a zero to be reported as such | **non-zero: 52 slot skips on dev, 56 on holdout.** The gate executes 108 times across the 400 games and changes no verdict |

**One thing §2 got incomplete, recorded because the control is what caught it.** §2 argued the
committed C142 base artifacts were a same-build baseline because `main` at `e0a23e4e` reproduces
their engine fingerprint. Every verdict scalar reproduced. But two `counters` keys differ from
those artifacts — `strict:diverged_on_full_branch_set` (dev) and
`strict:lossy_render_marker:attract_empty_tail_ambiguous` (holdout) — because both were **added to
`scripts/engine_transition_differential.py`** after C142's sweeps were taken, and the engine build
fingerprint does not cover `scripts/`. Compared against the C142 artifacts rather than against the
fresh same-build, same-harness base, P3 would have read as falsified by a change this branch did not
make. See the report §5a.

**And one stale figure in §2, left standing rather than edited.** §2 names the gate build
`a0796bdd…`. That was the wheel built before the seven pins were added to `events.rs`; adding them
moved the *source* fingerprint, `engine_build_fingerprint.py --check` reported `STALE`, the wheel was
rebuilt, and every gate measurement reported here was re-run on **`b8ff1445…`**. The §2 value is not
corrected in place because §1–§7 are frozen; it is corrected here and in the report §0.

**No sweep was run at or above seed 19,200,000.**

### 8a. Re-derived after `main` moved — appended, nothing above edited

`origin/main` advanced to **`f1c3b3aa`** (#1166, the attract immobilizer marker) after §8's pair was
taken. It adds an engine patch (**71 → 72**) and changes the renderer, so it was merged in
(`git merge`, never a rebase) and **all four sweeps were re-run on both sides of the merge**. Every
prediction still holds, on the merged builds `770228825d53f717…` (base) and `b3b0fde0b3fda523…`
(gate), both `--check` green:

* **P1/P2** — merged dev 15968 / 15503 / 15502 / 1 / 0 and merged holdout 16155 / 15579 / 15579 /
  0 / 0, identical to §2's table.
* **P3** — the merged gate's `counters` block is byte-identical to the merged base's, 23 keys on
  dev and 19 on holdout.
* **P5** — the row replay on the merged builds gives `diverged` / 12 misses → **`matched` / 0**,
  relabels **350 → 0**, as before.

**And the merge was not a no-op, which is why carrying the pre-merge numbers would have been
wrong.** On the holdout window #1166 removes `strict:lossy_render` (3 → absent) and its
`attract_empty_tail_ambiguous` marker (3 → absent) and moves
`strict:sleeptalk_union_branch` 105 → 106; on dev it moves nothing. Compared against a merged base,
those three would have read as **this** branch opening something and clause 6 of §4 would have fired
on another PR's change.

The crate floor was re-measured on the merged tree (429 → **436**, summed from the CI step's own
expression) and `_EXPECTED_SWEEP_ARTIFACTS` re-derived a second time (79 → **87**, selector over
both trees, live at 86 and 88). This branch adds **no** engine patch, so
`PATCHED_TARGET_TREE_SHA256`, `EXPECTED_FINAL_SHA256`, `--test test_gen3` at 32 and the
`Engine lib suite` at 5 are untouched — verified by an empty
`git diff origin/main...HEAD -- third_party/ scripts/apply_poke_engine_patches.py`.

### 8b. And a second merge, at `99c77eb7`

`origin/main` moved again to **`99c77eb7`** (#1168, the ledger negative-claims audit), which took the
report number `c146`. This prediction and its report are renumbered **C147**; the `c146_g33b_*`
paths named in the earlier commits are the same files. #1168 touches no engine patch and no sweep
artifact, so no measurement above is re-derived by it — verified by
`git diff f1c3b3aa 99c77eb7 --name-only`, which lists four files, none of them under
`third_party/` and none a sweep. It does add a **second** exact corpus pin,
`_EXPECTED_COUNTER_ARTIFACTS`, bumped **347 → 360** by set difference over both trees and confirmed
live at 359 and 361.
