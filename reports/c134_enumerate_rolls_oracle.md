# C134 — roll enumeration as a flag-gated reference oracle (C116 Phase 2)

**Status: MEASURED on the post-#1152 base, and the residue is nearly gone.** Re-run on the
71-patch build after #1152 merged: dev **1 -> 0**, holdout **0 -> 0**, nothing opened. Six
of the seven rows this branch ever measured have been closed in the SHIPPING engine by
other work (#1144, #1148, #1152); one remains. The wrong-fan control is down to a single
subject and its verdict there is UNDETERMINED. The shipping engine now matches where
enumeration used to be the only thing that did.

**This document was retitled and its disposition reversed after independent review.** The
first version adopted enumeration *for the differential harness*. That was wrong, and the
reasoning is in `reports/c137_phase2_enumerate_decision.md`; the short form is that a
differential running the enumerated path certifies a code path production never takes, so
the fidelity gate would stop measuring the engine that ships. What lands instead:

> **Enumeration ships as a flag-gated reference ORACLE, default off, consumed by tests.
> The transition differential keeps measuring the SHIPPING configuration — the collapsed
> partition cascade — exactly as before this branch.**

The measurement below is still worth taking and still worth committing: it is the
evidence that the oracle and the shipping path agree about which rows diverge, and it is
the only place the two configurations have ever been swept head to head from one build.
It is *not* a certification sweep for the enumerated path, and nothing here proposes
making it one.

## What is being added

The engine gains one patch, `third_party/poke-engine-gen3-enumerate-damage-rolls.patch`,
which adds an **enumerate-then-merge** roll path behind the runtime environment flag
`POKEZERO_ENUMERATE_ROLLS`. Three hunks in `src/gen3/generate_instructions.rs`:

1. `push_enumerated_rolls` — one arm per distinct `floor(max * r / 100)` for `r` in
   `85..=100`, mass `1/16` each, pre-merged where rolls land on the same integer. Exact
   integer arithmetic; the expression is character-for-character the one the
   already-shipped 32-arm `pending_hp_reading_move` enumeration uses in this same
   function, so this generalises a mechanism the engine already runs rather than
   inventing one.
2. The `residual_lethality_threshold` mirror call returns `None` on this path. Under
   enumeration there is no threshold to place, because no roll is collapsed before its
   consequences are computed.
3. The Case A / Case B partition cascade is bypassed for the enumerated branch.

**The flag is a runtime env read through a `OnceLock`, deliberately not a cargo feature.**
One build then serves both paths, so a collapsed sweep and an enumerated sweep of this
commit differ in exactly one variable and carry the *same* engine fingerprint. Two builds
could not make that claim, and the whole value of this measurement is that it is
single-variable.

The cost of that choice is that the fingerprint **cannot say which path ran**, which is
why `enumerate_rolls` is now carried in the report and in every checkpoint record's
provenance, checked on `--resume`, refused on a mixed `--merge-from`, and required by
`scripts/cert_execution_manifest.py`'s shard evidence.

### Default OFF, and nothing turns it on for anyone else

There is **no enabler in the repository**. Turning enumeration on is a per-run act:
`--enumerate-rolls` on the differential, or an explicit environment set by a test that
wants the oracle in a child process.

The first version of this branch did it the other way — `os.environ.setdefault(...,"1")`
at module scope in `scripts/engine_transition_differential.py` — and review proved that
leaks: `os.environ` writes go through `putenv`, so the value is process-global and
inherited by every child, 21 tracked modules load that file, and `unittest` imports every
selected module before running any test. The demonstrated consequence was
`tests/test_engine_search_no_panic.py` running `python -m pokezero.engine_search` under
enumeration.

`tests/test_roll_enumeration_scope.py` now asserts the **runtime** property instead of
the textual one: it spawns a child per surface, imports the surface into it, and requires
the engine's fan back collapsed at `[112, 160]`. Mutation-checked both ways — see
§Gates.

### The oracle, and what it is for

`reports/c133_collapsed_roll_disposition.md` §3 and
`reports/c135_roll_divergent_lethality_adjudication.md` §5 specify two ENGINE fixes to the
collapsed cascade. That family has already burned three wrong hand-derived mass recipes,
which is why C134 §3 froze it. Enumeration answers that objection directly:

> For any fixture, enumerate the fan and compare the collapsed arms' masses against the
> enumerated truth. A wrong recipe stops being something review has to catch by reading
> and becomes a test that fails.

An oracle that is itself unpinned would silently bless a wrong recipe, so the oracle's own
arm structure is pinned in `tests/test_roll_enumeration_scope.py`: the fan is exactly
`floor(max * r / 100)` for `r` in 85..=100 on both crit and non-crit; a probabilistic
secondary composes to `count/16 x branch probability` on every (damage x burn) cell to
1e-6; and the **multi-hit semantic change** — one roll shared across the hits, replacing
the collapsed path's total-to-per-hit conversion — is pinned on Bonemerang. Each has a
paired negative control requiring the collapsed cascade to disagree.

### Why the harness does NOT take this path

Two independent reasons, and only the first was in the original version.

**Coverage.** The differential is the fidelity gate for the engine that ships. Search runs
the collapsed damage-branch and residual-partition surface; a 200-game sweep taken on the
enumerated path would attest a configuration production never takes, and a regression in
the shipping surface would become invisible to it. Keeping the default collapsed is what
resolves that, and it is asserted rather than intended: the differential has no
module-scope environment write, its `enumerate_rolls` defaults to `False`, and the sweeps
committed as the shipping-path evidence carry `enumerate_rolls: false` in every record.

**Throughput, for search specifically.** Adopt-everywhere was measured and rejected:
`midgame_3v3` at depth 4 / 1024 sims went **2.38 ms to 8.88 s per decision, ~3700x**.
Re-measured on this branch below. Note the honest scope of that figure, per c137 §1:
production is `search_depth: 2` / `search_sims: 256`, and enumeration is gated on
`depth < DAMAGE_BRANCH_DEPTH = 2`, so the production-config regression is unmeasured and
is plausibly one to two orders of magnitude smaller. The decision does not rest on the
size of that number.

### Why `count/16` composes with a probabilistic secondary (C119's objection)

C119 held that a count-over-sixteen mass cannot express a secondary. It does not have to.
Enumeration replaces the roll collapse **inside** each chance branch; `run_move` then fans
every one of those arms through `get_instructions_from_secondaries`, so an arm's mass
comes out as `count/16 × branch probability` **by construction**, with no recipe to
hand-derive. This is asserted rather than asserted-about:
`tests/test_roll_enumeration_scope.py::test_enumerated_arms_compose_with_a_probabilistic_secondary`
reconstructs every (damage × burn) cell of a Fire Blast fan in pure Python from two
`calculate_damage` scalars and requires agreement to 1e-6, and its paired negative control
requires the collapsed cascade to *disagree* — so the check cannot pass vacuously.

## Prediction

Four sweeps of `scripts/engine_transition_differential.py`, 200 games each, from **one
build**, differing only in `POKEZERO_ENUMERATE_ROLLS`:

| window | seed-start | collapsed `transitions_diverged` | enumerated `transitions_diverged` |
| --- | --- | --- | --- |
| dev | 19000000 | 2 | **0** |
| holdout | 19100000 | 4 | **2** |

Also predicted:

* `boundaries_measured` / `boundaries_full_round` **identical** between the two
  configurations on each window: dev `15503 / 15968`, holdout `15579 / 16155`.
* `engine_errors == 0` in all four sweeps.
* The two rows surviving on holdout are both
  `component_missing_in_engine:itemleftovers`, at seed `19100170`. Those are a harness
  defect, not a roll defect; a separate fix is in flight for them and is not touched here.
* `source_commit` and `engine_fingerprint` identical across all four sweeps — one build,
  as designed. A differing fingerprint refutes the single-variable claim outright.

The final holdout at seeds `>= 19200000` is **not** touched. `FINAL_HOLDOUT_SEED_FLOOR`
enforces that in the tool.

### FALSIFIER — "nothing opened"

This is the clause that decides the patch, and it is not the net count.

> **The enumerated divergent set must be a SUBSET of the collapsed divergent set on each
> window, compared row by row as `(seed, boundary_index)`. If ANY boundary is `matched`
> under the collapsed cascade and `diverged` under enumeration, this patch is REFUTED and
> does not ship — even if the total number of divergent rows goes down.**

Net counts hide exchanges. The previous engine fix in this residue passed every other gate
and then opened 38 rows in dev and 40 in holdout; it was caught by exactly this comparison
and by nothing else. A patch that closes four rows and opens one is a patch that changes
behaviour in a direction nobody has characterised, and roll enumeration touches every
damaging move in the game, so the surface for that is the whole engine.

Two further refuting outcomes, stated in advance:

* **Coverage moved.** If `boundaries_measured` differs between the collapsed and
  enumerated sweep of the same window, the comparison is not like-for-like and the row
  counts mean nothing. Enumeration changes the branch *support*, never which boundaries
  are eligible, so any movement here is a defect in the harness wiring.
* **Any `engine_errors > 0`**, in any of the four sweeps.

### What would NOT refute it

* Row counts differing from the table above while the subset relation holds. The spike ran
  on a different base (it carried `poke-engine-gen3-faint-cancels-opposing-switch.patch`,
  which is not on `main`); this branch is `main` plus the enumeration patch alone. The
  predicted numbers come from the spike and are re-derived here, not carried forward. A
  disagreement is reported as a disagreement.
* Slower wall-clock on the enumerated harness sweeps. The harness is not throughput
  constrained; search is, and search does not take this path.

### Post-rebase re-registration (written and committed BEFORE the re-run)

The branch has been rebased from `317822f1` onto `2ec0cb13`. `main` now carries
`poke-engine-gen3-faint-cancels-opposing-switch.patch` (#1144), the 69th patch, in the
same `generate_instructions.rs` the enumeration patch touches. All 69 apply through
`git apply` with no fuzz and no fallback movement; `PATCHED_TARGET_TREE_SHA256`, the
`generate_instructions.rs` per-file pin and the tail pin (11 -> 12) are re-derived from a
clean-room replay against the verified 0.0.47 sdist.

That changes the base the sweeps run against, so the previous numbers cannot be carried
forward. **One new prediction, registered here before the sweeps:**

> The collapsed holdout row `19100180/24` (`component_extra_in_engine:spikes`) was present
> in BOTH configurations on the old base and was explicitly attributed to the absence of
> #1144. #1144 is now in the base. So holdout collapsed should read **4**, not 5, and
> holdout enumerated **2**, not 3 — i.e. the branch should now reproduce the spike's
> original 4 -> 2 exactly. Dev should be unchanged at 2 -> 0.

This is a real risk of being wrong and is registered as such: if `19100180/24` is still
divergent on the rebased base, the c134 explanation for it was wrong and the report says
so rather than re-explaining it after the fact. (Measured below: it held. The row is gone
from the collapsed set and holdout reads 4 -> 2.) Everything else — coverage pairs,
`gating_exact` / `gating_support_based`, `engine_errors == 0`, and above all the
**"nothing opened" subset falsifier** — carries over unchanged.

## Measurement

Four sweeps, 200 games each, **one build**, `POKEZERO_ENUMERATE_ROLLS` the only variable.
Run from a **clean checkout of `6d390acb`** on the **71-patch** build `44ee1430`, after
#1152 merged. Every record carries `"source_tree": "clean"`.

| window | | collapsed | enumerated |
| --- | --- | --- | --- |
| dev 19000000 | `transitions_diverged` | **1** | **0** |
| dev | `boundaries_measured / full_round` | 15503 / 15968 | 15503 / 15968 |
| dev | `gating_exact / gating_support_based` | 14156 / 1347 | 14156 / 1347 |
| dev | `engine_errors` | 0 | 0 |
| holdout 19100000 | `transitions_diverged` | **0** | **0** |
| holdout | `boundaries_measured / full_round` | 15579 / 16155 | 15579 / 16155 |
| holdout | `gating_exact / gating_support_based` | 14148 / 1431 | 14148 / 1431 |
| holdout | `engine_errors` | 0 | 0 |

All four: `build_check: gated`, `acceptance_eligible: true`, **one** engine fingerprint
`44ee1430708cbb55…`, **one** `source_commit`. Coverage identical between configurations on
each window, including the gating split. Final holdout (`>= 19200000`) untouched.

Sole surviving collapsed row: `19000191/63`, `component_magnitude:heal`. Enumeration closes
it; nothing opened.

### #1152 CLOSED THIS RESIDUE IN THE SHIPPING ENGINE

`10856e0e` merged while this branch was in review — the crit-straddle sub-split and the
status-aware residual threshold, which change the shipping collapsed cascade on exactly the
mechanism behind `limit:roll_divergent_lethality`. Measured here, not read off its report:

| row | class | collapsed BEFORE (`f35c5928`) | collapsed NOW (`6d390acb`) |
| --- | --- | --- | --- |
| `19000074/27` | `component_missing_in_engine:sandstorm` | diverged | **closed by #1152** |
| `19100107/135` | `limit:roll_divergent_lethality` | diverged | **closed by #1152** |
| `19100191/5` | `limit:roll_divergent_lethality` | diverged | **closed by #1152** |
| `19000191/63` | `component_magnitude:heal` | diverged | still diverged |

**The holdout window now has no divergent rows at all**, in either configuration. The
enumeration-versus-collapsed comparison is no longer measurable there: there is nothing
left to compare.

That these rows are closed by #1152 rather than by anything here is checkable in one
direction — they are absent from the **collapsed** sweep, which is the shipping path.

### The registered prediction, and every departure from it

The prediction (`0733abe7`) said dev 2 → 0 and holdout 4 → 2, with two surviving
`itemleftovers` rows. Measured: dev **1 → 0**, holdout **0 → 0**. Every departure is a base
change landing on main while this branch was in review, and each is verifiable the same
way — the row is gone from the **collapsed** baseline, so enumeration did not close it:

| row | closed by | not by |
| --- | --- | --- |
| `19100180/24` | #1144 (faint cancels the queued action) | enumeration |
| `19100170/71`, `/72` | #1148 / C139 (Encore vs. post-Transform moveset) | enumeration |
| `19000074/27`, `19100107/135`, `19100191/5` | #1152 / C138 (collapse-class engine fixes) | enumeration |

Six of the seven rows this branch ever measured were closed in the shipping engine by other
work. One remains.

### The falsifier: NOTHING OPENED

| window | closed by enumeration | opened by enumeration |
| --- | --- | --- |
| dev | `19000191/63` (`component_magnitude:heal`) | **none** |
| holdout | — (nothing was divergent) | **none** |

Strict subset on both windows; both sweeps report `repros_complete: true`. **The falsifier
did not fire** — but note how little it now carries: on holdout it is vacuous, and on dev it
is one row.

### The wrong-fan control now has ONE subject, and it is the undetermined one

Everything below this heading was built on a four-row divergent set. **Three of those rows
no longer exist.** The analysis is kept as a record of how the instrument was validated —
six confounds, each of which produced a plausible verdict for a reason unrelated to the
property under test — but it describes a base the shipping engine has moved past, and none
of its conclusions transfer to the current one.

Re-run against the new repro set (`reports/artifacts/c134_wrong_fan_control.json`), one row:

| arm | `19000191/63` |
| --- | --- |
| collapsed | diverged (14) |
| enumerated | matched (1015) |
| `only_visible` (legal values) | matched (957) |
| wrong fan, NEAREST non-legal | **matched** |
| wrong fan, capped @2% | **diverged** |
| wrong fan, FAR | **diverged** |
| `isolable_only` (drop trailing-faint, values LEGAL) | diverged (551) |
| `drop_only_legacy` | diverged (377) |

**Verdict: UNDETERMINED**, unchanged. Two equally valid zero-contamination value-only
perturbations disagree, and the disagreement is not ordered by displacement — the NEAREST
arm at 10.6% matches while the 2%-capped arm at 1.95% diverges.

**The two rows this branch established as robustly not value-driven were `19000074/27` and
`19100191/5`. Both are among the rows #1152 closed.** The positive finding left with its
subjects; what survives is the row where nothing was established. Stated plainly rather
than reconciled.

`the_control_has_subjects` is now a required verdict. Without it every check is an
`all(...)` over the closed rows and `all([])` is `True`, so the whole battery would have
gone green while measuring nothing — the same false-green shape this control has already
produced six other ways.

### What this leaves

* **Enumeration and the shipping cascade now agree on both windows except one row**, and on
  that row enumeration closes and the collapsed path does not. That is the entire
  remaining delta.
* **The shipping engine now matches where enumeration used to be the only thing that did.**
  That is the honest summary of #1144, #1148 and #1152 against this residue, and it is the
  outcome the oracle was supposed to enable rather than a loss.
* **The lottery question is moot on five of six rows** because the rows are gone, and
  **undetermined on the sixth**. No structural control was run; that remains the experiment
  to build if anyone needs to settle it.
* `reports/c133_collapsed_roll_disposition.md` §3/§7 and
  `reports/c135_roll_divergent_lethality_adjudication.md` §2-3 remain the account of why
  each row was an engine gap — and #1152 acted on exactly that account, which is the
  strongest available corroboration of it.

### Correction to the merged #1152

#1152 (`10856e0e`) cites "enumeration also closes these rows" as sweep-side corroboration,
taken from an earlier revision of this document which claimed the closures were measured
not to be value-driven on all four rows. **That claim was over-generalised and is
withdrawn**: it held on two rows and was undetermined on two, and the two it held on are now
closed in the shipping engine, so the corroboration cannot be re-derived on the current
base at all.

**#1152's own correctness does not depend on it.** Its primary argument is a 480-state
differential against a partition-independent enumeration, with all residual error localised
to the f32 comparator; that stands untouched, and this branch's re-measurement independently
reproduces its headline result (holdout 2 → 0, dev 2 → 1). Only the secondary sweep-side
corroboration is affected.

`strict:sleeptalk_union_branch` moves 126 → 617 (dev) and 105 → 612 (holdout), reproducing
the movement c137 §4 recorded as unexplained. Explained by the same mechanism: it is
incremented once per rendered BRANCH, so it tracks branch multiplicity, not boundary
population. Every per-BOUNDARY counter is unchanged.

### Adopt-everywhere: re-measured, and still rejected

`scripts/bench_multiply_search.py --depths 4 --sims 1024 --seeds 5 --min-time 1.0`, same
build, flag off then on, run **serially** after the sweeps finished. Artifacts:
`reports/artifacts/c134_bench_{collapsed,enumerated}_search.md`.

| position | collapsed ms/decision | enumerated ms/decision | slowdown |
| --- | --- | --- | --- |
| `minimal_1v1` | 0.32 | 58.16 | 182x |
| `midgame_3v3` | **2.49** | **9327.16** | **~3746x** |
| `endgame_straddle` | 0.25 | 1.38 | 5.5x |

`midgame_3v3` leaf evals go **4,147 -> 950,803** (229x) at the same 1024 sims: the
per-branch factor multiplied down a depth-4 tree. Nine seconds per decision is not a
search.

And these are demonstrably **different engines**, not one engine at two speeds. Same
position, same five seeds, `minimal_1v1` root argmax:

```
collapsed:  ember, ember, tackle, tackle, ember     (argmax unstable across seeds)
enumerated: tackle, tackle, tackle, tackle, tackle
```

That is the coverage argument in one line: a differential run on the second column would
certify a player that plays `tackle` while production plays `ember`.

Enabling the flag for the second run is an explicit operator act on one command line —
no code in this repository would do it. Note the scope, per c137 §1: production is
`search_depth: 2` / `search_sims: 256`, and enumeration is gated on
`depth < DAMAGE_BRANCH_DEPTH = 2`, so **the production-config regression is unmeasured**.
The decision does not depend on the size of the number.

### The multi-hit semantic change, pinned

Enumeration applies a **per-hit** roll shared across the hits, replacing the collapsed
path's total-to-per-hit conversion. c137 §4 listed this as unpinned; it is pinned by
`tests/test_roll_enumeration_scope.py::test_enumeration_shares_one_roll_across_a_multi_hit`.

Bonemerang, two hits, defender at full HP so nothing clamps or merges:

| | per-hit damages emitted |
| --- | --- |
| collapsed | `[61, 124]` — one non-crit representative, one crit |
| enumerated | all 28 values of `floor(max * r / 100)` for r in 85..=100 over `max_regular 67` and `max_crit 135`, equal to the pure-Python reconstruction |

and in every branch the two `Damage` instructions are **equal** — one roll, reused. A
paired negative control requires the collapsed path to disagree.

### The oracle HAS a consumer now

A previous revision said it pinned only itself. **#1152 changed that.** It landed
`scripts/collapsed_arm_mass_oracle.py` and `tests/test_collapsed_arm_mass_oracle.py`, which
compare the shipping engine's outcome-mass functional three ways: a pin regenerated
out-of-process against an **enumerating build**, an independent pure-Python enumeration, and
the shipping engine. All three must agree.

That is the payoff written into this patch's manifest entry, landing — and it is why the
patch is worth keeping now that the differential residue it was measured against is
essentially closed. The oracle's value was never the sweep comparison; it was making a
wrong collapsed mass recipe fail a test instead of surviving review, and that is now a CI
step.

### Gates

All run locally against the build the sweeps used (`e97e661eaf9fb3b1`, 69 patches).

| gate | result |
| --- | --- |
| `tests.test_poke_engine_patch_stack` — clean-room replay vs. verified 0.0.47 sdist | **4 tests, OK**; 69 patches, all `git-apply`, no fuzz |
| `tests.test_roll_enumeration_scope` — the runtime scope gate | **17 tests, OK** |
| `tests.test_branch_mass_reconstruction` — the mass gate | **6 tests, OK** |
| `tests.test_engine_transition_checkpoint_provenance` | **19 tests, OK** (was 15; +4 for `source_tree`) |
| `tests.test_transition_differential_matcher` | **53 tests, OK** (was 52; +1 for the wrong-fan remap) |
| `tests.test_final_holdout_guard` | **14 tests, OK** |
| `tests.test_engine_search_no_panic` — spawns a real SEARCH child, the leak path review demonstrated | **1 test, OK** (179.2 s) |
| `tests.test_cert_sweep_readout_contract` | **39 tests, OK** |
| `tests.test_cert_historical_attestation` | **3 tests, OK** |
| `tests.test_engine_world_encore_transform` | **18 tests, OK** |
| `tests.test_collapsed_arm_mass_oracle` (#1152's oracle consumer) | **7 tests, OK** |
| `cargo test --release`, `RUSTFLAGS="-C debug-assertions=yes"`, `rust/pokezero-search` | **416 passed, 0 failed**, 34 suites, exit 0 |
| `cargo test third_party/poke-engine-src --features gen3 --test test_gen3` | **28 passed, 0 failed**, exit 0 |
| `scripts/engine_behavioral_probes.py`, flag off (the shipping path, and what CI runs) | **40 probes, 40 PASS**, exit 0 (38 -> 40 with #1152's two) |
| `scripts/engine_behavioral_probes.py`, flag on | 24 PASS / 16 FAIL, exit 1 — see below |
| `scripts/c134_wrong_fan_control.py` | **exit 1 — by design.** All preconditions pass across every arm (branch set fixed, zero legal values emitted, nothing failed to remap); it exits on `every_closed_row_is_value_independent`, FALSE because two of four rows are undetermined. Not in any workflow |
| Also green | `test_single_seat_coverage_bound` (3), `test_drag_limit_is_a_last_resort` (3), `test_a1_residuals_already_ran` (13), `test_sleep_talk_phaze_drag` (7), `test_instruction_event_mapping` (21), `test_crit_kill_split_patch` (8), `test_rest_sleep_refund_boundary` (6), `test_rest_sleep_refund_write_side` (6), `test_engine_stat_attestation` (16), `test_roll_cascade_predicate` (29), `test_matcher_tolerance_promotion` (32), `test_c26_archival_recalibration` (11) |

**One pre-existing failure, not caused by this branch.**
`tests.test_c26_damage_composition_readout::test_production_matcher_is_not_the_rejected_experiment`
pins `reports/certification_contract_lifecycle.json`'s
`successor_pending_identity.differential_sha256` against the live differential's digest.
On `origin/main` the live file hashes `3b6d3c60…` while the lifecycle pins `8f83a11f…`,
so it is RED on main before this branch touches anything. The module is in no workflow's
`FILTERS` and has no CI step, which is how the drift went unnoticed. Left alone — it
belongs to the certification lifecycle owner, and re-stamping it from this PR would hide
the drift rather than surface it.

**The 16 probes that fail under the flag** are all arm-STRUCTURE assertions
(`residual-partition-*`, `pending-read-*`, `ordered-walk-*`, `case-a-*`), each a variant of
*expected 4 branches, got 18*. They assert the collapsed cascade's partition shape; under
enumeration the fan is enumerated, not partitioned. Masses are correct throughout and the
mass gate passes in both configurations. Nothing ships enumerated, so the shipped
configuration is the one where all 38 pass.

**Both gates were mutation-checked, not merely run.**

| mutation | failures | which |
| --- | --- | --- |
| unmutated | 0 | — |
| module-scope `os.environ.setdefault` restored in the differential | **3** | AST check + both differential-import probes |
| enabler in `src/pokezero/engine_search.py` | **4** | + the search-surface probe |
| enabler in `src/pokezero/engine_env.py` (review's counter-example) | **4** | AST, ledger, core-surface and discovered-surface probes |
| **concealed** enabler in `scripts/hc_depth_grid.py` — `from os import environ`, flag name built by concatenation, so neither the ledger nor the AST check can see it | **1** | `test_no_discovered_search_surface_enumerates` **only** |
| mass gate: force the collapsed child ON | **2** | |
| mass gate: perturb the pinned representative 112 → 111 | **1** | |
| mass gate: unmutated | 0 | |

The concealed-enabler row is the one that matters: it is invisible to every static check
by construction, lands in a module that is not in the named core, and is caught **only**
by the exhaustive discovered-surface sweep. That is the measurement showing live discovery
is load-bearing rather than decorative.

Two earlier versions of these checks were measured insufficient and are recorded because
the failures are instructive. The first AST check matched only string literals and stayed
GREEN against `os.environ.setdefault(_ENUMERATE_ROLLS_ENV, "1")`. The second resolved
aliases of the flag constant but not of `os`, so `from os import environ` — an ordinary
idiom, not a dodge — passed it.

## Disposition

**Enumeration lands as a default-off reference oracle. The differential is NOT switched to
it.** On the post-#1152 base the falsifier did not fire, nothing opened, coverage and
identity are unchanged, and the shipping engine's divergent set on these two windows is a
single dev row.

What changed under this branch's feet, and what it means:

* **The residue is closed in the shipping engine.** #1144, #1148 and #1152 closed six of
  the seven rows this branch ever measured. The holdout window has none left. That is the
  right outcome and the one the program wanted — closing rows in the engine that ships
  rather than in the instrument, which is precisely the reversal `reports/c137` argued for.
* **The oracle now has its consumer.** `tests/test_collapsed_arm_mass_oracle.py` compares
  the shipping engine's outcome-mass functional against an enumerated pin and an
  independent Python enumeration. That, not the sweep comparison, was always the point.
* **The remaining analytical question is undetermined and now nearly moot.** The one
  surviving row, `19000191/63`, is the row the wrong-fan control could not decide; the two
  rows it did decide are gone. No structural control was run.

Still open:

* **`19000191/63`** (`component_magnitude:heal`) — enumeration closes it, the collapsed
  path does not, and whether that closure is value-driven is undetermined.
  `reports/c133` §3 disposes of it as a representative mis-pricing rather than a missing
  arm.
* **The f32 comparator (C116 M5)** still executes in search, untouched here.
* Enumeration's **production-config** (depth 2 / 256 sims) throughput cost is unmeasured.
