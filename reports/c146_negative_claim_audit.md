# C146 — the known-gaps ledger's negative claims, audited as a class

`reports/c138_known_gaps_ledger.md` had a false "never fired" corrected three separate times on
2026-08-07 — H14 in #1163, §3.5's static-counter count in #1162, and H11's two `reports/`
negatives in #1165 — and each error propagated into other PRs, because citing the ledger felt
like verification. Three PRs independently inherited one of them.

This pass stops auditing them one at a time. **Entry H13 is the primary target**: it had never
been audited, and #1165 moved its line number while explicitly declining to corroborate it.
Secondarily, every remaining "never / nowhere / not observed / absent from all N" claim in the
document is sorted into **measured** and **merely asserted**, so the next reader knows which to
trust.

**Nothing here was swept.** This is static analysis of committed artifacts and committed source.
No engine was built, no sweep was run, and the reserved final holdout (`19,200,000+`) was not
touched — every seed referenced below comes out of an artifact that was already in the tree.

---

## 1. Method, and the glob

Every figure below comes from one corpus, stated once so that every claim in this report inherits
it:

> **347 committed JSON files** — every `.json` under `reports/` **or** `docs/`, recursively.
> That is 267 under `reports/` (196 direct + 71 in `reports/artifacts/`) and 80 under `docs/`
> (**79** of them in `docs/audit_artifacts/**`). Plus the non-JSON trees where a claim needed them:
> `reports/*.md`, `docs/**`, `scripts/**`, `tests/**`, `rust/**`.

Two scoping decisions, each of which is what the previous four errors turned on:

**The corpus is not `reports/artifacts/`.** Widened deliberately. `docs/audit_artifacts/**` is a
committed measurement corpus — the era-59 depth and sims grids — and it is where
`self_moveset_mismatch` reaches 2,560 and `transform_unexpressible` reaches 208. It was invisible
to every negative in the ledger. Scoping a glob to one directory and reporting the result as
repo-wide is this program's recurring defect, and §3 below shows it produced H15's error too.

**A name is matched, not a counter path.** The C32/C43 precedent is the whole lesson. Three
distinct shapes carry a counter value in this repo:

| shape | example | example artifact |
|---|---|---|
| the counter key itself | `counters.skip:world_unsupported:self_moveset_mismatch = 75` | `reports/artifacts/c133_withdrawn_switchcancel_dev_sweep.json` |
| a **differently-named field** | `coverage_diagnosis.coverage_reducing_skips.self_moveset_mismatch = 5058` | `reports/c32_fail_diagnosis.json` |
| the name in a **sibling** field | `decomposition.ranked[2] = {"counter": "world_unsupported:self_moveset_mismatch", "rows": 5058}` | `reports/c43_coverage_shortfall_diagnosis.json` |

Plus a fourth in `docs/`: the reason as a key **prefix** with an interpolated detail —
`engine_stats.world_failure_reasons."self_moveset_mismatch: p2: request-known move 'hiddenpower'
is absent from the sampled moveset (Transform/Mimic-class desync)" = 2560`.

An audit keyed on the first shape alone declares `strict_all_branches_lossy` never-fired and is
wrong, which is exactly what H14 did. The scanner used here admits shapes 1, 2 and 4 through one
rule (the name as a token anywhere in the leaf's dotted path, leaf value a nonzero number) and
shape 3 through a second (a mapping with a string field **equal** to the name after prefix
stripping, and a nonzero numeric sibling).

**One negative control, because it changed two answers.** A name appearing inside *prose* in a
JSON value, next to an unrelated number, is not evidence. `no_usable_branch` appears in
`reports/c9_decomposition.json` and `c12_decomposition.json` inside a `"basis"` narration
(`"documented mapper insufficiency (callee-union path / no_usable_branch / LOSSY)"`) beside a
`1500174`, and `BranchLegalRollError` appears in a `reports/c17_...` sentence beside a `1`. A
looser matcher reports both as fired. Both are genuinely never-fired. The pin carries
`test_prose_alone_is_not_evidence` as the standing control.

Variant spellings checked for every name: as written, prefix-stripped, `:`→`_`, `_`→`:`,
`_`→`.`, `_`→`-`, and with the `skip:`, `strict:`, `abort:`, `world_unsupported:`,
`unmappable_choice:`, `divergence_class:` prefixes both present and absent.

---

## 2. H13 — the primary target. It is false.

H13 read:

> **`self_moveset_mismatch`, `transform_unexpressible`, `status_unsupported` and 33 other
> world-construction refusal reasons are defined and never fire in either window.** Full list in
> §3.5. … Observed: **no** — 36 of 40 `world_unsupported` reasons are 0 in both windows

### 2.1 The taxonomy: 40, confirmed from source

Derived by AST over every `raise EngineWorldUnsupported(...)` in
`src/pokezero/engine_world.py` — **40 distinct literal reasons**, so the denominator is right.
(`payload_malformed` has 14 raise sites, `substitute_health_provenance_contradiction` 5,
`encore_move_unknown` 4; the count is of reasons, not sites.)

### 2.2 All three reasons H13 names by name have fired

| reason | value | where |
|---|---|---|
| **`self_moveset_mismatch`** | **75 dev / 24 holdout** | `counters.skip:world_unsupported:self_moveset_mismatch` in **27** committed sweep artifacts, `reports/artifacts/c121_*` through `c133_*` |
| | 108 | `reports/c{7,8,9}_summary.json`, `c10_{encore,explosion,kecleon}_differential.json`, `c11_statfloor_differential.json`, `c12_trace_toxic_differential.json`, `c13_{batch_e,rebaseline}_differential.json` |
| | 11 | `reports/c112_leaf_state_scenarios.json` `[0].counts.*` |
| | **5,058** | `reports/c32_fail_diagnosis.json` `coverage_diagnosis.coverage_reducing_skips.self_moveset_mismatch` |
| | 5,058, rank **2** of 10 | `reports/c43_coverage_shortfall_diagnosis.json` `decomposition.ranked[2].counter` |
| | up to **2,560** | `docs/audit_artifacts/hc-depth-grid-20260729/hc-d1.json` — **25** files under `docs/audit_artifacts/**` in total |
| **`transform_unexpressible`** | **23** | `reports/c32_fail_diagnosis.json` `coverage_diagnosis.coverage_reducing_skips.transform_unexpressible` |
| | 23, rank **8** | `reports/c43_coverage_shortfall_diagnosis.json` `decomposition.ranked[8]` |
| | **208** | `docs/audit_artifacts/k0-depth-grid-20260729/results/k0g-{a,c}-d1-1.json`, `world_failure_reasons."transform_unexpressible: side 'p1' copied 'Deoxys', absent from the sampled opposing party"` |
| **`status_unsupported`** | **2** | `reports/c32_fail_diagnosis.json`, same field family |
| | 2, rank **9** | `reports/c43_coverage_shortfall_diagnosis.json` `decomposition.ranked[9]` |
| | **9,071** and **3,453** | `docs/engine_divergence_ledger_20260728.md` §A / §D8 tables |

Note what the `transform_unexpressible` and `status_unsupported` evidence is: the **c32 field and
the c43 ranked list**. Those are the two artifacts that refuted H14 and produced the whole
"differently-named field" lesson, recorded in §3.5 of the same document, two paragraphs from
H13's own cell. H13 was refuted by the artifact its own document had already used to refute a
neighbouring row.

### 2.3 `self_moveset_mismatch` fired in **these** windows — not an older era

This is the part that makes H13 false on its own terms rather than on a technicality. The claim
is scoped "in either window". Read the `seeds` block of the artifacts:

| artifact | `seeds.min` | `seeds.max` | `distinct` | games | matcher | `boundaries_full_round` | `self_moveset_mismatch` |
|---|---|---|---|---|---|---|---|
| `c121_a5_dev_sweep.json` | 19000000 | 19000199 | 200 | 200 | strict | 15,968 | **75** |
| `c133_withdrawn_switchcancel_dev_sweep.json` | 19000000 | 19000199 | 200 | 200 | strict | 15,968 | **75** |
| `c136_faintcancels_fix_dev_sweep.json` | 19000000 | 19000199 | 200 | 200 | strict | 15,968 | 0 |
| `c121_a5_holdout_sweep.json` | 19100000 | 19100199 | 200 | 200 | strict | 16,155 | **24** |
| `c133_withdrawn_switchcancel_holdout_sweep.json` | 19100000 | 19100199 | 200 | 200 | strict | 16,155 | **24** |
| `c136_faintcancels_fix_holdout_sweep.json` | 19100000 | 19100199 | 200 | 200 | strict | 16,155 | 0 |

Byte-identical windows, identical budget, identical matcher. The counter fired 75 and 24 times in
the dev and holdout windows on **13 and 14** committed sweep artifacts respectively, and became zero
because it was **closed**.

### 2.4 It closed, and the closure reconciles

The commit is `29ca5697`, *"Stop a transformed request overwriting a Pokemon's own retained move
state"*, whose own message opens: *"Closes the dominant half of `self_moveset_mismatch`: 365
killed decisions in era 59, 44.8 % of the construction channel."* Mechanism per that message:
`actor_move_states_from_request_history` retained the **copied** moveset of a transformed Ditto
permanently (gen3 reverts Transform on switch-out, so no later request refreshes the entry), and
`engine_world._move_specs` then compared it against the root snapshot's `[transform]`.

The counter diff c133 → c136 on dev, taken from the artifacts:

```
boundaries_measured                        15,432 -> 15,503   (+71)
skip:world_unsupported:self_moveset_mismatch   75 -> 0        (-75)
limit:world_substitute_health_unknown         127 -> 131      (+4)
```

`-75 + 4 = -71` skips, `+71` measured. Holdout closes the same way: `-24` self_moveset_mismatch,
`-5` `encore_move_unknown`, `+1` substitute health = `-28`, against `boundaries_measured`
15,551 → 15,579 (+28). The freed boundaries **reappeared as measured**, which is what makes this
a repair and not a disappearance — and which is precisely the distinction "never fired" erases.

### 2.5 What survives, and it is a real result

Of §3.5's list of 33, **four have fired**: `self_moveset_mismatch`, `transform_unexpressible`,
`payload_malformed` (4, `reports/c112_leaf_state_scenarios.json`) and `pending_baton_pass`
(3 / 3 / 2, `reports/c112_leaf_state_golden_v2.json`, `_v4.json`, `_scenarios.json`).

**The other 29 have no nonzero record anywhere in the 347-file corpus.** So does the pair §3.5
calls "structurally diverted" — except that both of *those* have fired too
(`status_unsupported`, and `substitute_health_unknown` at 12 and 14 in
`reports/c112_leaf_state_golden_v{2,4}.json`).

Repo-wide, the 40 reasons partition **10 fired / 30 never**:

- fired **in the c136 windows** (4): `volatile_unsupported` 144/127, `materialization_blocker`
  18/8, `encore_move_unknown` 2/1, `self_request_state_unsupported` 13/0;
- fired **earlier, or only in `docs/`** (6): `self_moveset_mismatch`, `status_unsupported`,
  `substitute_health_unknown`, `transform_unexpressible`, `payload_malformed`,
  `pending_baton_pass`;
- **never fired anywhere** (30): the 29 above plus `future_sight_pending`, which R1 shows is
  additionally unreachable in this pool.

H13's arithmetic also never quite closed: "X, Y, Z and 33 other" is 36 reasons, and §3.5's 33
already **contains** X and Y while excluding Z. The two passages were counting different sets.

---

## 3. Secondary: the rest of the ledger's negatives

### 3.1 A second false one — H15, from a glob scoped to one directory

H15: *"only 6 of the 19 `divergence_class` values have ever fired — so 13 have never fired … the
remaining seven — `component_set_equal_but_unmatched`, **`limit:world_sample_drag_target`**, … —
are strict-path classes the program has simply never produced."* Quoted as "re-derived across
**all 31 committed sweep artifacts**".

`limit:world_sample_drag_target` **has** fired:

| value | field | artifact |
|---|---|---|
| **5** | `divergence_classes.limit:world_sample_drag_target` and `counters.divergence_class:limit:world_sample_drag_target` | `reports/c10_{encore,explosion,kecleon}_differential.json`, `c11_statfloor_differential.json`, `c12_trace_toxic_differential.json`, `c13_{batch_e,rebaseline}_differential.json` |
| **4** | `divergence_classes.limit:world_sample_drag_target` | `reports/c26_structural_probe_report.json` |
| **271** | `family_attribution.limit:world_sample_drag_target` | `reports/c14_cert_sweep_readout.json`, `c15_instrument_coverage.json`, `c24_final_classifier_c14_calibration.json` |

**None of those files is under `reports/artifacts/`, and that is the whole mechanism.** Run the
same selector at two scopes:

| scope | files | distinct `divergence_class` keys | static classes fired |
|---|---|---|---|
| `reports/artifacts/*sweep*.json` | 59 | 19 | **6** |
| `reports/artifacts/*.json` | 71 | 19 | **6** |
| all 347 under `reports/` + `docs/` | 347 | **35** | **7** |

So H15 is exactly right inside `reports/artifacts/` and wrong as stated. Corrected figures:
**7 of 19 have fired, 12 have never.** H15's parenthetical "the 18 distinct keys observed" is 19
at its own scope and 35 repo-wide.

The 19 is right: `classify_divergence` in `scripts/engine_transition_differential.py` has exactly
19 static `return` sites, derived by AST.

H15's two structural-unreachability sub-claims **hold**, re-verified:

- `mapper_lossy` — the trigger string `"every branch rendered lossy"` is returned on the
  **skip** path (`engine_transition_differential.py:2223`, `return "skip_lossy", [...]`), which
  `continue`s before the classification line. Nothing routes it into `classify_divergence`.
- `no_usable_branch` — its trigger `"mapper produced no usable branch"` appears at **exactly one
  site in the whole repo**: the classifier's own test of it
  (`engine_transition_differential.py:1915`'s guard). No producer exists. The *identifier*
  appears in 7 files, one of which reads as a hit to a prose-matching search; §1's negative
  control is what separates them.

### 3.2 A third — H17, false when written, in the same way H11 was

H17: *"`reports/c119_phase2_scoping.md`, `reports/c134…` and
`reports/c137_phase2_enumerate_decision.md` are cited by merged reports and absent from
`reports/`"*, Observed *"**yes** (verified by `ls`)"*.

- `reports/c137_phase2_enumerate_decision.md` — **present, and was present when this was
  written.** Added by `dc6e1e19` (#1147); `git merge-base --is-ancestor dc6e1e19 f876803e`
  succeeds, and `f876803e` is the commit that merged C138 itself. `git ls-tree -r f876803e --
  reports/` lists the file. This is the H11 failure exactly: a negative asserted over `reports/`
  from a search that missed a file already on `main`.
- `reports/c134_enumerate_rolls_oracle.md` — genuinely absent at `f876803e`, added one commit
  later by `6be52191` (#1149, not an ancestor of `f876803e`). **True when written, stale now.**
- `reports/c119_phase2_scoping.md` — **still absent.** `ls reports/ | grep '^c119'` returns
  nothing.

1 of 3. Verdict: **PARTLY RETRACTED**, and the load-bearing point now rests on c119 alone.

### 3.3 Verified negatives — six claims promoted from asserted to measured

A verified negative is a real result. These are recorded as such, each with the glob:

| claim | verdict | evidence |
|---|---|---|
| §3.5 **"Never-fired static counters (9)"** — `abort:no_legal_action`, `skip:no_action_candidates`, `skip:world_error:no_constructible_candidate`, `strict:no_damage_rolls`, `strict:branch_event_legal_error:BranchLegalRollError`, `engine_error`, `world_prestate_mismatch:side_conditions`, `mapper_lossy`, `no_usable_branch` | ✅ **VERIFIED** | all nine absent across the **347**-file corpus. C142 measured them over "260 committed JSONs under `reports/`"; the number survives a glob 87 files wider, including all of `docs/audit_artifacts/**` |
| §3.5 **"Never-fired dynamic families (6)"** — `skip:no_materialization:`, `skip:world_error:`, `strict:branch_events_error:`, `engine_error:`, `engine_error_choice:`, `world_prestate_mismatch:weather_` | ✅ **VERIFIED** | no key under any of the six prefixes carries a nonzero value in any of the 347 |
| §3.5 **`skip:unmappable_choice` 7 of 8 unobserved** | ✅ **VERIFIED** | seven absent; the eighth, `struggle_not_submittable`, present in **78** of the 347 — the control that makes the seven non-vacuous |
| H14's residual — **`engine_error` is an unexercised term of the verdict identity** | ✅ **VERIFIED** | 0 on every artifact, under both `engine_error` and `engine_errors`. ⚠ And there are **two**: `skip:rump_branch_set` is also 0 across all 347. H14 states the identity as four-term; `cert_sweep_readout.py:1451,1611` and the gate's own message make it **five**-term. Annotated at H14 — a stale arity in the cell that corrected the last arity error |
| H8 — **`strict:no_damage_rolls` is 0 in both windows** | ✅ **VERIFIED, and stronger** | 0 repo-wide, not just in the two windows. H8's own UNKNOWN (how much mass rides the *per-branch* window) is untouched by this — the counter bounds the state-level fallback only, exactly as H8 says |
| H15 — **`mapper_lossy` and `no_usable_branch` are structurally unreachable** | ✅ **VERIFIED** | see §3.1; one is `continue`d before classification, the other has no producer of its trigger string |
| §3.5's remaining **29 of 33** `world_unsupported` reasons | ✅ **VERIFIED** | no nonzero record in the 347-file corpus under any of the four shapes in §1 |

### 3.4 The inventory: which negatives in C138 are measured, and which are asserted

Every "never / nowhere / no cause / not observed / 0 in both windows / 0 of N"-shaped claim in
the document, by class. **Class A** is re-derivable from committed artifacts and is what this
audit settled. **Class B** is measured against an instrument outside this repo. **Class C** is
re-derivable by `grep` right now.

**Class A — artifact-observation negatives (12 claims). All now settled.**

| # | claim | verdict |
|---|---|---|
| A1 | H13 — 3 named + 33 other refusal reasons never fire | ⚠ **FALSE** (§2) |
| A2 | §3.5 umbrella — "the program has never seen it fire" | ⚠ **FALSE** for 6 of the names (§2.5) |
| A3 | H15 — 6 of 19 divergence classes ever fired / 13 never | ⚠ **FALSE**, 7 and 12 (§3.1) |
| A4 | H15 — 7 of 8 `unmappable_choice` never fire | ✅ VERIFIED |
| A5 | §3.5 — nine never-fired static counters | ✅ VERIFIED (wider glob) |
| A6 | §3.5 — six never-fired dynamic families | ✅ VERIFIED |
| A7 | §3.5 — 36 of 40 reasons zero **in both windows** | ✅ VERIFIED at that scope; the four are `volatile_unsupported`, `materialization_blocker`, `encore_move_unknown`, `self_request_state_unsupported` |
| A8 | G50 — `transform_unexpressible` 0 in both windows | ✅ VERIFIED at that scope; ⚠ annotated, since it is **not** never-fired |
| A9 | H14 — `engine_error` never fired | ✅ VERIFIED |
| A10 | H8 — `strict:no_damage_rolls` 0 in both windows | ✅ VERIFIED, and repo-wide |
| A11 | H15 — `mapper_lossy` / `no_usable_branch` structurally unreachable | ✅ VERIFIED |
| A12 | §3.5 — `self_request_state_unsupported` 13 dev, **absent** holdout; `hidden_counter_support:confusion` 1 dev, 0 holdout | ✅ VERIFIED in the c136 pair. Note the first fires **643** times in `reports/c43_...`, so it is one-sided *in this window*, not rare |

**Class B — pool-reachability negatives (27 occurrences across 23 rows: G9, G31, G34, G37, R1–R4,
R9, R11, R12, R15–R25). NOT re-derivable from this repo, and not audited here.**

Every "**0 of 220**" and "**0 of 393 sets**" is measured against `data/random-battles/gen3/sets.json`
in the vendored Showdown checkout at commit `f76228a1354b5d0f307ca2d16101294ad3a2308b`, which is
outside the repository. §1.4 already says the right thing about them ("Every '0 of 220' below is a
statement about `f76228a1`, not a theorem"), and §8's own rules cover the two ways they have gone
wrong before — the `target:` rule (R26/G49) and the movepool-is-an-upper-bound rule (G14). They
are marked here as **measured-with-a-stated-instrument, unverifiable from committed artifacts**,
which is a weaker status than Class A's ✅ and should be read as such. Re-deriving them needs the
Showdown checkout and is a separate task; a pin cannot hold them, which is why §8's rules for
them are prose and the counter lists' are not.

**Class C — repo-content negatives (5 claims). Re-derived by grep for this report.**

| # | claim | verdict |
|---|---|---|
| C1 | H17 — c119, c134, c137 absent from `reports/` | ⚠ **1 of 3** (§3.2) |
| C2 | H11 — "no written cause anywhere in `reports/`" | already ⚠ retracted by #1165; re-confirmed false — `reports/c139_encore_transform_move_index_prediction.md` is in the tree at `f876803e` |
| C3 | G31 — "`clearallboost` appears in the crate exactly twice, both inside that comment" | ✅ VERIFIED — `git grep -c clearallboost -- rust/` → `rust/pokezero-search/src/events.rs:2` |
| C4 | H15 — `no_usable_branch`'s trigger string "exists nowhere in the repo" | ✅ VERIFIED as written about the **trigger string** (`git grep -c "mapper produced no usable branch"` → 1, the classifier's own guard). The bare identifier is in 7 files, so the claim needs its "trigger string" qualifier to be true, and it has it |
| C5 | G1 — "`grep -c STICK` over the engine `src/` returns hits only for `STICKYHOLD`/`STICKYWEB`" | **UNVERIFIED — and unverifiable here.** `third_party/poke-engine-src/` is gitignored and regenerated. §1.3 cites the engine by symbol for exactly this reason; the claim is sound as a *procedure* and cannot be checked from a clean checkout |

**Two of the document's own ⚠ corrections are also Class A/C negatives, and both hold:** H14's
correction (`skip:strict_all_branches_lossy` fires at 2 in `reports/c{26,27}_structural_probe_report.json`
and 372 in `reports/c32_fail_diagnosis.json`) and §3.5's 10 → 9. Re-derived; both stand.

---

## 4. Mechanization

Five prose corrections to the same claim shape in three days is the argument. A pin re-derives;
prose restates.

**`tests/test_never_fired_counter_census.py` — 16 tests.**

- **Taxonomies from source, not transcribed.** The 40 refusal reasons come from every literal
  first argument to `raise EngineWorldUnsupported(...)` in `src/pokezero/engine_world.py`; the 19
  divergence classes from `classify_divergence`'s static `return` sites. Both counts are pinned,
  so a refusal added without a §3.5 row is red.
- **The corpus is the 347 files**, walked from `reports/` and `docs/`, and its size is an **exact**
  pin — following `_EXPECTED_SWEEP_ARTIFACTS`, and for its reason: a floor lets a member vanish,
  and a member vanishing is this pin's fail-open. `test_the_corpus_spans_both_trees` additionally
  refuses a selector that silently drops `docs/audit_artifacts`, which is where two of H13's three
  reasons fire.
- **The partitions are exact SET EQUALITY in both directions.** `_FIRED_WORLD_UNSUPPORTED` is
  pinned at exactly 10 names and `_FIRED_DIVERGENCE_CLASSES` at exactly 7. A counter that starts
  firing is red rather than a silent widening; a counter that stops being *found* is also red,
  because that means the scanner broke.
- **Both evidence matchers carry their own witness pin.** `test_the_c32_differently_named_field_shape_is_matched`
  asserts the exact `coverage_diagnosis.coverage_reducing_skips.self_moveset_mismatch = 5058`
  leaf, and `test_the_c43_sibling_field_shape_is_matched` the `rows: 23` under c43's
  `ranked[8]`. Without these, a matcher that stopped matching would turn every absence pin green.
- **H13's refutation is pinned to the windows**: that the c121/c133/c136 pairs share one
  `seeds.min/max/distinct`, that the counter is 75/24 pre-closure, and that the c133 → c136
  `boundaries_measured` delta reconciles at exactly +71 — so the claim cannot come back as prose.

**Gated.** `.github/workflows/engine-fidelity-gates.yml` gains a `Never-fired counter census`
step with `grep -qE 'Ran 16 tests'` and a clean-`^OK$` check, plus filter entries for the pin
file and for `reports/c32_fail_diagnosis.json` and `reports/c43_coverage_shortfall_diagnosis.json`
— so a PR whose only change deletes a witness cannot skip the gate. The taxonomy sources
(`src/pokezero/**`, `scripts/engine_transition_differential.py`) were already filters.

**A test that exists is not a test that runs**, so both the guard and the pins were exercised
rather than assumed:

| # | mutation | result |
|---|---|---|
| M0 | none (baseline) | `Ran 16 tests` / `OK`, exit 0, 15 s |
| M1 | `_EXPECTED_COUNTER_ARTIFACTS` 347 → 346 | ✅ caught (1 failure) |
| M2 | 347 → **348** | ✅ caught — the pin fires one lower *and* one higher |
| M3 | drop `docs/` from the corpus | ✅ caught (2 failures) |
| M4 | scope the corpus to `reports/artifacts/` — **H15's actual glob** | ✅ caught, **7 failures + 1 error** (the error is the c32 witness pin, which raises rather than asserts once `self_moveset_mismatch` stops being found). Includes the divergence-class partition, so this is direct proof the pin would have caught H15 |
| M5 | set `_FIRED_DIVERGENCE_CLASSES` to H15's six | ✅ caught |
| M6 | set `_FIRED_WORLD_UNSUPPORTED` to exclude H13's three named reasons | ✅ caught |
| M7 | kill the path matcher for `coverage_reducing_skips` (the H14 shape) | ✅ caught |
| M8 | kill the sibling-field matcher (the c43 shape) | ✅ caught |
| M9 | admit prose as evidence | ✅ caught (4 failures) |
| M11 | M5 **plus** a subset assertion instead of equality | ⚠ **PASSES** — the documented fail-open, and the reason the partitions assert equality |

10 mutations applied, **9 caught, 0 survivors**; M11 is not a survivor but a demonstration of the
design choice. The YAML guard was exercised directly against the real log: `grep -qE 'Ran 16
tests'` exits 0, `'Ran 15 tests'` and `'Ran 17 tests'` exit 1, and the clean-`OK` grep exits 1 on
a deliberately reddened run — captured with `echo "exit=$?"` immediately after each command, never
through a pipe, and with `__pycache__` cleared before every run.

**Scope note.** This is one test module and one CI step, and it covers only Class A. Class B (the
27 pool-census negatives) would need the vendored Showdown checkout at a pinned commit inside CI,
which is a materially larger change and is not attempted; §8's prose rules remain the only control
there, and §3.4 marks their status honestly instead.

---

## 5. What did not change

- **No §3 row was added or removed.** §3 stays at 78. H13, H15 and H17 are corrections to existing
  cells, not new gaps.
- **No sweep artifact was added**, so `_EXPECTED_SWEEP_ARTIFACTS = 79` in
  `tests/test_boundary_verdict_partition.py` is untouched. Confirmed by the selector rather than
  by inspection: no file in this PR carries a top-level `boundaries_measured`.
- **No engine, no sweep, no holdout.** The final holdout (`19,200,000+`) is spent and was not
  approached; nothing here needed it.
- **The fidelity numbers are unchanged.** The residue is still 2 dev / 4 holdout on the c136 pair,
  and it still sits on a widened accept bar: **~9 %** of measured boundaries (1,347 of 15,503 =
  8.689 % dev; 1,431 of 15,579 = 9.185 % holdout) are accepted via up to **64** enumerated hidden
  sleep-counter worlds, per Constraint 7.

## 6. What this does not settle

1. **Class B, the 27 pool-reachability negatives.** Measured against a Showdown checkout outside
   the repo; unverified here and marked as such in §3.4. **Settling measurement:** re-run the
   §1.3 static census and the generative census against `f76228a1` and diff every "0 of 220" and
   "0 of 393" in place.
2. **G1's `grep -c STICK`** (C5). Unverifiable from a clean checkout because
   `third_party/poke-engine-src/` is gitignored. **Settling measurement:** rebuild the engine
   source per §1.3 and re-run the grep, recording it in an artifact rather than in prose.
3. **Whether `self_moveset_mismatch` is fully closed or only dominantly so.** `29ca5697`'s message
   says "the dominant half", and the two c136 windows read 0 — but 0 on two windows is the exact
   evidential shape this whole report is about. **Settling measurement:** the counter is now
   pinned to 0 across all 347 artifacts, so a re-appearance in any future committed sweep is a red
   gate; that is a monitor, not a proof of closure.
4. **Whether any *other* C138 claim shape is systematically wrong.** This pass audited negatives.
   Positive claims — every "yes" in the Observed column, every reachability *grant* — were not
   re-derived.
