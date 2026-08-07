# C134 — roll enumeration as a flag-gated reference oracle (C116 Phase 2)

**Status: MEASURED, with one question left OPEN.** The falsifier did not fire and both
windows reach zero divergent rows under the oracle. The wrong-fan control that was meant
to show the closures are value-driven rather than a cardinality lottery is **confounded and
withdrawn** — see the retraction in §"Nothing opened" is weak on its own. Whether these
closures are a real fix or an artefact of a 9x-72x larger branch set is **not settled by
this branch**. The prediction was committed in `0733abe7` before any sweep on this branch
was started; the post-rebase re-registration below was committed in `21178990`, before the
sweeps that test it, which are the four artifacts landing with this section. The ordering
between "what I expected" and "what I measured" is a property of the history.

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
Run from a **clean checkout of `f35c5928`**, the commit carrying the code. Every record
carries `"source_tree": "clean"`, so that is a property of the artifact rather than a
claim here. (An earlier attempt at this re-run was discarded and restarted: a commit was
made while it was in flight, so its records would have stamped `clean` for a tree that
changed underneath them. Nothing but the sweeps' own output artifacts was written to the
tree during the run that produced these numbers.)

| artifact | window | roll path | `enumerate_rolls` in the artifact |
| --- | --- | --- | --- |
| `reports/artifacts/c134_collapsed_dev_sweep.json` | dev 19000000 | collapsed | `false` |
| `reports/artifacts/c134_enumerated_dev_sweep.json` | dev 19000000 | enumerated | `true` |
| `reports/artifacts/c134_collapsed_holdout_sweep.json` | holdout 19100000 | collapsed | `false` |
| `reports/artifacts/c134_enumerated_holdout_sweep.json` | holdout 19100000 | enumerated | `true` |

| window | | collapsed | enumerated |
| --- | --- | --- | --- |
| dev | `transitions_diverged` | 2 | **0** |
| dev | `boundaries_measured / full_round` | 15503 / 15968 | 15503 / 15968 |
| dev | `gating_exact / gating_support_based` | 14156 / 1347 | 14156 / 1347 |
| dev | `engine_errors` | 0 | 0 |
| holdout | `transitions_diverged` | **2** | **0** |
| holdout | `boundaries_measured / full_round` | 15579 / 16155 | 15579 / 16155 |
| holdout | `gating_exact / gating_support_based` | 14148 / 1431 | 14148 / 1431 |
| holdout | `engine_errors` | 0 | 0 |

All four: `build_check: gated`, `acceptance_eligible: true`, engine fingerprint
`e97e661eaf9fb3b1…` (**one** distinct value across all four), `source_commit` `f35c5928`
(**one** distinct value), `source_tree: clean`. Coverage is identical between
configurations on each window in the strong form: not merely the same *number* of
boundaries measured, but `gating_exact` and `gating_support_based` identical too.

The reserved final holdout (`>= 19200000`) was not touched.

### Where this DEPARTS from the registered prediction, and why

The prediction (`0733abe7`) said the two rows surviving on holdout would be
`component_missing_in_engine:itemleftovers` at `19100170/71` and `/72`, described as
"a harness defect, not a roll defect; a separate fix is in flight for them".

**That fix landed.** #1148 (C139, Encore against the post-Transform moveset) closed
exactly those two rows —
`reports/c139_encore_transform_move_index_results.md` registers them by name and measures
`component_missing_in_engine:itemleftovers` 2 → 0. Main was merged into this branch, so
they are gone from the **collapsed** baseline here. Holdout collapsed therefore reads 2,
not 4, and enumeration takes it to **0**.

Stated as a departure rather than presented as a better result: enumeration did not close
those two rows, another PR did, and the evidence that it was not enumeration is that they
are absent from the collapsed sweep as well. The post-rebase re-registration's claim about
`19100180/24` also held — that row is likewise absent from the collapsed baseline, closed
by #1144.

### The falsifier: NOTHING OPENED

Rows keyed `(seed, step)`, from the committed `repros` arrays; each sweep reports
`repros_complete: true`, so these are full divergent sets and not samples.

| window | closed by enumeration | opened by enumeration |
| --- | --- | --- |
| dev | `19000074/27` (`component_missing_in_engine:sandstorm`), `19000191/63` (`component_magnitude:heal`) | **none** |
| holdout | `19100107/135`, `19100191/5` (both `limit:roll_divergent_lethality`) | **none** |

Strict subset on both windows, and both windows reach zero. **The falsifier did not
fire.** Re-derivable from the committed artifacts:

```sh
python - <<'EOF'
import json
def rows(p):
    d = json.load(open(p)); assert d["repro_retention"]["repros_complete"]
    return {(r["seed"], r["step"]) for r in d["repros"]}
for w in ("dev", "holdout"):
    c = rows(f"reports/artifacts/c134_collapsed_{w}_sweep.json")
    e = rows(f"reports/artifacts/c134_enumerated_{w}_sweep.json")
    print(w, "opened:", sorted(e - c), "closed:", sorted(c - e))
EOF
```

### "Nothing opened" is weak on its own — and the control meant to fix that FAILED

**RETRACTION.** An earlier revision of this section claimed "cardinality of that order does
not buy acceptance; the values do", and the abstract claimed the closures were "measured as
a fix rather than merely consistent with one". **Both are withdrawn.** The control they
rested on was confounded. The lottery question below is **OPEN**.

The question is real. `evaluate_boundary_strict` returns `matched` on the **first** rendered
branch that matches (`scripts/engine_transition_differential.py`, the `if ok: return
"matched"` inside the per-branch loop), and enumeration multiplies the branch count. More
branches is more chances to match, so a subset relation is close to what you would expect
**even if enumeration were wrong**.

The size of the effect, by replaying each divergent row's own `engine_states` through
`pokezero_search.branch_events` in both configurations:

| window | row | collapsed | enumerated | factor |
| --- | --- | --- | --- | --- |
| dev | `19000074/27` | 3 | 29 | 9.7x |
| dev | `19000191/63` | 14 | 1015 | 72.5x |
| holdout | `19100107/135` | 8 | 544 | 68.0x |
| holdout | `19100191/5` | 4 | 34 | 8.5x |

#### How the control failed, and how that was established

`scripts/c134_wrong_fan_control.py` adjudicates each row with the same matcher on the same
recorded inputs under several branch-set manipulations. Artifact:
`reports/artifacts/c134_wrong_fan_control.json`. **It exits 1.**

| row | collapsed | enumerated | wrong_fan (set FIXED, values varied) | drop_only_legacy (same drop, values LEGAL) |
| --- | --- | --- | --- | --- |
| `19000074/27` | diverged (3) | matched (29) | **matched** (29) | **diverged** (22) |
| `19000191/63` | diverged (14) | matched (1015) | **matched** (1015) | **diverged** (377) |
| `19100107/135` | diverged (8) | matched (544) | **matched** (544) | **diverged** (352) |
| `19100191/5` | diverged (4) | matched (34) | **matched** (34) | **diverged** (28) |

**The superseded control's result came from a DROP, not from the values.** It dropped every
branch it could not remap — 6, 580, 176 and 5 branches, 57% of the fan on `19000191/63` —
and read "diverged". The `drop_only_legacy` arm deletes the same set and leaves the
survivors' values **legal and untouched**, and still reads **diverged** on all four rows.
So the earlier verdict was "delete the closing branch and the row reopens", which is true of
any branch set and silent on whether the values are right. Review identified this; the arm
above re-derives it here rather than quoting it.

**Holding the branch set fixed and varying only the values reads `matched` on all four
rows — and that is not evidence of a lottery either, because the fan is contaminated:**

| row | branches | still compatible with a legal roll | share |
| --- | --- | --- | --- |
| `19000074/27` | 29 | 7 | 24% |
| `19000191/63` | 1015 | 464 | 46% |
| `19100107/135` | 544 | 128 | 24% |
| `19100191/5` | 34 | 6 | 18% |

#### Why route (a) is not achievable on these rows

The obvious repair — remap a lethal branch onto a *disjoint lethal* value instead of
dropping it — does not close the gap, and the reason is structural rather than a coding
problem:

* A branch whose target faints **on the move** renders as `0 fnt`. The amount is not in the
  protocol, so every lethal roll produces a byte-identical branch. Remapping it onto another
  lethal value is a no-op at the only layer the matcher can see, and the branch stays
  compatible with the legal lethal rolls.
* A branch whose target survives the move and faints **later from a residual** cannot be
  given a different roll without changing whether it faints. That is not a value-only
  perturbation; it is a different branch, and synthesising the engine's downstream
  rendering for it by hand would introduce a fresh confound.

These four rows are `limit:roll_divergent_lethality` and its neighbours. **Lethality is the
mechanism under test**, so "hold the outcome class fixed and vary the value" is not merely
awkward here — it is not well defined. Restricting the control to rows with zero
contamination (route (c)) leaves **zero rows**, so there are no valid subjects.

#### What still stands, and what does not

* **Stands:** enumeration closes four rows and opens none, on both windows, with coverage
  and gating counters identical between configurations. That is a fact about the artifacts.
* **Stands:** the closing branch is a *visible-amount* branch, not a lethal one — the
  `only_visible` arm matches on every row while `only_lethal` diverges on every row. So the
  closure does depend on branches whose damage the protocol actually shows.
* **Does NOT stand:** any claim that the sweep distinguishes a real fix from a lottery. It
  does not, and this branch has not measured that.
* What carries the correctness account remains the per-row mechanism analysis in
  `reports/c133_collapsed_roll_disposition.md` §3/§7 and
  `reports/c135_roll_divergent_lethality_adjudication.md` §2-3 — read, not measured.

This is the fourth confound of the same family found in this one control (shift too large,
shift clamped and overlapping, wrong side perturbed, and now the drop doing the work).
Three were caught here; the fourth was caught by review. The pattern is worth naming: each
version produced the expected verdict for a reason unrelated to the property being tested,
and only an isolating arm — hold everything fixed except the one variable — exposed it.

`strict:sleeptalk_union_branch` moves 126 → 617 (dev) and 105 → 612 (holdout), reproducing
the movement c137 §4 recorded as unexplained. Explained by the same mechanism: it is
incremented **once per rendered branch** in that loop, so it tracks branch multiplicity, not
boundary population. Every per-BOUNDARY counter is unchanged.

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

### The oracle has no consumer yet

Stated plainly: **the oracle currently pins only itself.** The only `flag="1"` uses in the
tree are the four self-pins in `tests/test_roll_enumeration_scope.py` (fan structure,
secondary composition, multi-hit, and the runtime negative control), plus the
`--enumerate-rolls` comparison sweeps and the wrong-fan control. The payoff written into
the patch manifest — enumeration as an exact oracle for the collapsed path's mass recipes
— is **deferred to the c133 §3 / c135 §5 engine fixes, landing separately as #1152**.

### Gates

All run locally against the build the sweeps used (`e97e661eaf9fb3b1`, 69 patches).

| gate | result |
| --- | --- |
| `tests.test_poke_engine_patch_stack` — clean-room replay vs. verified 0.0.47 sdist | **4 tests, OK**; 69 patches, all `git-apply`, no fuzz |
| `tests.test_roll_enumeration_scope` — the runtime scope gate | **17 tests, OK** (was 14; +3 for discovered surfaces and the alias pins) |
| `tests.test_branch_mass_reconstruction` — the mass gate | **6 tests, OK** |
| `tests.test_engine_transition_checkpoint_provenance` | **19 tests, OK** (was 15; +4 for `source_tree`) |
| `tests.test_transition_differential_matcher` | **53 tests, OK** (was 52; +1 for the wrong-fan remap) |
| `tests.test_final_holdout_guard` | **14 tests, OK** |
| `tests.test_engine_search_no_panic` — spawns a real SEARCH child, the leak path review demonstrated | **1 test, OK** (179.2 s) |
| `tests.test_cert_sweep_readout_contract` | **39 tests, OK** |
| `tests.test_cert_historical_attestation` | **3 tests, OK** |
| `tests.test_engine_world_encore_transform` (arrived with the merge) | **18 tests, OK** |
| `cargo test --release`, `RUSTFLAGS="-C debug-assertions=yes"`, `rust/pokezero-search` | **416 passed, 0 failed**, 34 suites, exit 0 |
| `cargo test third_party/poke-engine-src --features gen3 --test test_gen3` | **28 passed, 0 failed**, exit 0 |
| `scripts/engine_behavioral_probes.py`, flag off (the shipping path, and what CI runs) | **38 probes, 38 PASS**, exit 0 |
| `scripts/engine_behavioral_probes.py`, flag on | 24 PASS / 14 FAIL, exit 1 — see below |
| `scripts/c134_wrong_fan_control.py` | **exit 1 — INCONCLUSIVE by design.** `wrong_fan_contains_no_legal_values` is false, so its verdict is uninterpretable; see the retraction above |
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

**The 14 probes that fail under the flag** are all arm-STRUCTURE assertions
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
it.** The falsifier did not fire; both windows reach zero divergent rows under the oracle
and neither opens anything; coverage and identity are unchanged.

**What this branch does NOT establish:** that the four closures are a real fix rather than
a consequence of enumeration's 9x-72x larger branch set. The control built to settle that
is confounded and its conclusion is withdrawn; route (a) is not well defined on these rows
because lethality is the mechanism under test. The question is open, and closing it is a
prerequisite for treating the oracle as authoritative about these rows — not for landing
the oracle itself, which is default-off and consumed by nothing that ships.

Still open, unchanged by this branch:

* **The engine fixes in c133 §3 and c135 §5 are un-cancelled and land as #1152.** Closing
  those rows in the SHIPPING engine is the remedy; this branch supplies the oracle that
  makes a wrong mass recipe fail a test instead of surviving review.
* **The f32 comparator (C116 M5)** still executes in search and is untouched here.
* Enumeration's **production-config** (depth 2 / 256 sims) throughput cost is unmeasured.
