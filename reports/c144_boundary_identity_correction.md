# C144: the boundary identity was two-term and the instrument is four-term

> ⚠ **SUPERSEDED IN PART 2026-08-07 (C142).** The instrument is now **five**-term:
> `skip:rump_branch_set` was added as a post-measurement verdict
> (`reports/c142_rump_branch_adjudication.md`). Every finding below stands — the two-term form
> is still false, the case analysis is still the right method, and no number here moves — but
> two of this report's *premises* were falsified by that change and are corrected in the
> table in §3, following this report's own convention of correcting the site rather than
> only the conclusion.

**The false invariant.** For the whole C111–C141 era this program asserted, in reports, in
prediction clauses, and in a standing rule in the ledger, that

```
transitions_matched + transitions_diverged == boundaries_measured
```

is a property of `scripts/engine_transition_differential.py`. **It is not.** The identity is

```
transitions_matched + transitions_diverged + engine_errors
    + counters["skip:strict_all_branches_lossy"]  ==  boundaries_measured
```

Four terms, not two, and not the three the ledger's standing rule had. (C142 later added a
fifth, `skip:rump_branch_set`; see the banner above. The count is a property of how many
post-measurement verdicts `run_game` has, so it is expected to grow — which is why the
mechanized form, not the arity written in prose, is the thing to rely on.)

**It was already refuted by artifacts in the repo when it was last asserted.** Two committed
reports break the two-term form and close on the four-term one:

| artifact | measured | matched | diverged | engine_errors | `skip:strict_all_branches_lossy` | 2-term | 4-term |
|---|---|---|---|---|---|---|---|
| `reports/c26_structural_probe_report.json` | 4,738 | 4,672 | 64 | 0 | **2** | 4,736 ≠ 4,738 ✗ | 4,738 ✓ |
| `reports/c27_structural_probe_report.json` | 4,738 | 4,676 | 60 | 0 | **2** | 4,736 ≠ 4,738 ✗ | 4,738 ✓ |
| C141 final holdout (PR #1159) | 16,274 | 16,268 | 2 | 0 | **4** | 16,270 ≠ 16,274 ✗ | 16,274 ✓ |

C141's final-holdout sweep is what forced this audit, but it was not the first counterexample.
`reports/c138_known_gaps_ledger.md` H14 stated flatly that
"`skip:strict_all_branches_lossy` **has never fired**", and `reports/c26_structural_probe_report.json`
had it at 2 at the time. H14 is corrected in this PR. The lesson is the repo's own standing one:
the artifact that would refute the claim was on disk, and nobody opened it.

**The complete audit of the identity's own artifacts.** All 70 committed JSON reports carrying
`boundaries_measured` + `transitions_matched` + `transitions_diverged` were re-derived. The
four-term identity closes on **70 of 70**. The two-term form closes on **68 of 70** — it failed on
exactly the two probe reports above, and the reason it held on the other 68 is that both extra
terms are 0 there, on every dev and validation-holdout window the era iterated against.

---

## 1. What `skip:strict_all_branches_lossy` actually means

*The paragraph a future reader can rely on.*

The strict matcher does not compare the engine against Showdown directly; it asks the Rust
renderer (`rust/pokezero-search/src/events.rs`) to turn each engine branch's instruction list
back into protocol lines, and then compares *those* against the real protocol. When the renderer
cannot faithfully attribute part of what it emitted — a segment it could not split
(`segmentation_failed`), a confusion self-hit whose damage it cannot assign, an instruction list
that came back empty (`empty_instruction_list`) — it stamps the branch with a **lossy marker**.
`branch_render_is_usable()` admits a branch only if *every* marker it carries is on a narrow
allowlist of markers whose *telemetry* is incomplete while the public action window is still
exact (`sleeptalk_called_unidentified`, `attract_immobilization_source_unknown`); anything else
disqualifies that branch, counted as `strict:lossy_render`. If **every** positive-probability
branch across every candidate state is disqualified, `usable_branches == 0` and the matcher
returns the verdict `skip_lossy`, counted once as `skip:strict_all_branches_lossy`. Such a
boundary is **unadjudicable, not divergent**: the mapper has said it cannot reproduce this turn,
so there is no engine-side prediction to agree or disagree with, and calling it a divergence
would attribute a renderer gap to the engine. It nevertheless sits **inside**
`boundaries_measured`, because that counter increments as the last statement of
`_prepare_boundary` — it means "this boundary was fully prepared and handed to the matcher", and
every skip that *could* keep a boundary out of it (`skip:world_unsupported:*`,
`skip:unmappable_choice:*`, `world_prestate_mismatch`, `limit:*`, `skip:no_materialization:*`,
`skip:no_action_candidates`, `skip:world_error:*`) has already returned `None` by then. The lossy
verdict is discovered *later*, by the matcher, on a boundary the preparation stage had no reason
to refuse. So it is correctly **not** an exit from the coverage denominator (C132's claim, which
stands) and it **is** an exit from the verdict tally (which C132 does not say) — and reading the
first as licensing the second is precisely how the two-term identity spread. The two partitions
are different partitions of different sets, and the counter's `skip:` prefix, shared with seven
counters that behave the opposite way, is the trap.

## 2. The other terms inside `boundaries_measured`

Item 5 of the audit: is `skip:strict_all_branches_lossy` the only counter that sits inside
`boundaries_measured` without landing in `matched` or `diverged`? **No — there is exactly one
other, and the corrected identity includes it.**

Derived from `run_game`, which is the only place a prepared boundary can go. After
`_prepare_boundary` returns non-`None`, control reaches exactly four **counted** outcomes — plus a
fifth, **uncounted** one that never reaches a report, which §2a is entirely about and which must be
read alongside this table rather than after it:

| outcome | counter | in `boundaries_measured` | in a `transition:*` tally |
|---|---|---|---|
| the matcher raises (pyo3 panics included, caught as `BaseException`) | `engine_error` | yes | **no** |
| every branch rendered lossy | `skip:strict_all_branches_lossy` | yes | **no** |
| boundary matched | `transition:matched` | yes | yes |
| boundary diverged | `transition:diverged` | yes | yes |

`engine_error` is the second one, and the ledger's three-term standing rule omitted it. It is 0
on all 70 committed artifacts, so the omission has never been exercised — the same accident that
kept the two-term form alive. The four-term identity is therefore **complete for the verdicts
`run_game` had when this was written**, not patched: it is a case analysis of `run_game`, not an
empirical fit. ⚠ **C142** added a fifth verdict, so the *arity* was not durable even though the
*method* was; the case analysis was re-run and extended rather than refitted.

**Two facts make the case analysis airtight, and both are worth stating because each is a thing a
future change could take away.** First, `boundaries_measured` has **exactly one increment site in
the whole repo** and the verdict domain is **closed** — at three values when this was written, four since C142 added `skip_rump` (⚠ corrected) — `evaluate_boundary_strict`
and `evaluate_boundary` return only `"matched"`, `"diverged"` and `"skip_lossy"` — so the *dynamic*
key `f"transition:{verdict}"`, which looks like it could mint a term, cannot. Second,
`_prepare_boundary` has a **second caller**, `scripts/attest_materialized_damage_stats.py`, which
increments `boundaries_measured` and never records a verdict. It is harmless only because it passes
a throwaway `Counter()` and publishes no report. Hand it the live counter and the identity breaks
there silently.

## 2a. The seam: the fifth path, and the accident that hides it

A fifth path out of a measured boundary **does** exist, and the identity survives it only by an
accident of error handling. An invariant that holds by accident should say which accident.

Between the increment and the verdict there is a stretch of `run_game` that is **not** inside the
matcher's `try`: `env.step(actions)`, `_fold(cumulative)`, the `active_changed` comprehension, the
deliberate re-raise of `KeyboardInterrupt`/`SystemExit` out of the matcher's own handler, and
`classify_divergence` on the diverged path. An exception from any of those escapes `run_game` with
`boundaries_measured` already incremented and **no verdict counted** — a genuine fifth outcome, and
one with no counter of its own.

It is not a partition violation today for exactly one reason: **`counts` is a local `Counter`,
created at the top of `run_game`, and the sweep loop calls `run_game` with no `try`/`except` around
it.** A game that raises anywhere in that stretch propagates out and its counts are discarded
*wholesale* — the increment never reaches a checkpoint record or a report. The partition holds
because the evidence is thrown away, not because the path cannot be taken.

**The change that breaks it, named so it is recognisable in review:** *"salvage the partial counts
so a long sweep doesn't lose a game"* — wrapping the `run_game` call in `try`/`except` and recording
`counts` anyway, or hoisting `counts` out of `run_game` so the caller owns it. Either makes the
identity **five-term instantly**, and the fifth term has no name because nothing counts it. Anyone
making that change must count the escape explicitly (`boundary_abandoned_after_measure` or
similar) and add it to `VERDICT_PARTITION_SCALARS` / `VERDICT_PARTITION_COUNTERS`, rather than
discovering the drift later from a report that no longer reconciles. The same note is in the block
comment above `verdict_partition_failures`, where the person making the change will actually be.

**Both halves of the accident are now pinned, not merely described.**
`test_the_seam_that_hides_the_fifth_path_is_still_in_place` asserts, by parsing the source rather
than grepping it, that `counts` is still local to `run_game` (not a caller-owned parameter) and that
the single `run_game` call site is not inside a `try` with `except` handlers. A bare `try`/`finally`
— which is what is there today, closing the checkpoint handle — passes; adding handlers around the
call does not. Both mutations were built and both go red, and the failure message names the escape
that would have to be counted. The pin does not prove the escape unreachable; it proves the two
conditions that make it harmless cannot be removed silently.

## 2b. What still is not defended, stated plainly

A **code-level** break of the partition inside `run_game` — making `transition:matched` simply stop
incrementing, say — is caught today **only by the `differential_sha256` byte pin in
`reports/certification_contract_lifecycle.json`, not semantically.** That pin fires on *any* edit to
the differential and is re-stamped as routine, so a future PR could break the partition and
re-stamp the pin green in the same commit. Nothing in the test suite would object.

The runtime self-check at both of `main`'s report-emitting exits is the real defence: it evaluates
the identity against the report the run actually produced, so a mutated counter shows up as a
`COUNTER INTEGRITY` failure and a nonzero exit on the first real sweep. That is a *runtime* guard,
not a CI one, and this note exists so nobody mistakes the byte pin for the semantic guard it is
not. Closing the gap properly would need a synthetic-boundary harness that drives `run_game`
end to end; that is not attempted here.

**Counters that look like they belong in the identity and must not be added.** Every one of
these is inside the measured region, and none of them is a boundary verdict:

- `strict:lossy_render`, `strict:sleeptalk_union_branch`, `strict:no_damage_rolls`,
  `strict:branch_events_error:*`, `strict:branch_event_legal_error:*` — **per branch or per
  candidate state**, so one boundary can contribute many or none. C141's holdout has
  `strict:lossy_render` 14 against only 4 boundaries that lost *every* branch. Adding any of
  these breaks the identity on real data.
- `gating:exact` / `gating:support` — a **second, independent partition of the same set**:
  re-derived, `gating:exact + gating:support == boundaries_measured` on 70 of 70 artifacts. It
  answers "was the world exact or support-widened", not "what was the verdict".
- `divergence_class:*` — a partition of `transitions_diverged` alone. Re-derived,
  `sum(divergence_class:*) == transitions_diverged` on 70 of 70.
- `hidden_counter_support:*` — fires *before* `boundaries_measured` increments, so it is **not**
  a subset of the measured set and does **not** equal `gating:support`. Measured: it exceeds
  `gating:support` on 66 of 70 artifacts (e.g. 1,337 against 1,331 on
  `c121_a5_dev_sweep.json`), because some of those boundaries later exited on
  `skip:unmappable_choice:*` or `world_prestate_mismatch`. Anyone re-deriving the accounting will
  reach for this equality; it is false.
- every other `skip:*` counter, plus `world_prestate_mismatch` and `limit:*` — all fire *before*
  `boundaries_measured` increments and belong to the **coverage** reconciliation instead
  (`measured + in-path exits == boundaries_full_round`, pinned by
  `tests/test_single_seat_coverage_bound.py`).

  ⚠ **CORRECTED 2026-08-07 (C142).** The clause "every other `skip:*` counter … fire *before*
  `boundaries_measured` increments" is now false: `skip:rump_branch_set` is a `skip:*` counter,
  is not the lossy one, and fires **after**. It was true when written — every `skip:*` site then
  existing sat in `_prepare_boundary` and was immediately followed by `return None`, so
  `counts["boundaries_measured"] += 1` never ran, and `skip:single_seat_boundary` sits in the
  `else` branch where `_prepare_boundary` is not called at all. The repair is **not** to invert
  the timing rule: membership needs *two* conditions — fires after `boundaries_measured`, **and**
  is the boundary's terminal verdict (at most one per boundary, mutually exclusive with
  `transition:*`). Timing alone would admit `gating:*`, which increments on the very next line,
  and every `strict:*` counter in `evaluate_boundary_strict`. None of those is a verdict.

## 3. Where the false invariant appeared, and what each site was

The arithmetic was searched for, not just the phrase. Complete list.

### Wrong — asserted as a property of the instrument or registered as a forward clause

| site | what it said | disposition |
|---|---|---|
| `reports/c115_program_state.md` §"That the denominator held" | "every measurement below reports … the identity `matched + diverged == boundaries_measured`" — a standing rule for the whole era | corrected, with a note that no `matched` figure changes |
| `reports/c138_collapse_class_engine_fixes_prediction.md` §3 | registered as a **prediction clause** | corrected; the two-term clause would have read a lossy-skipped run as *falsifying the fix* |
| `reports/c139_encore_transform_move_index_prediction.md` clauses 3 and 5 | registered as **prediction clauses** | corrected, same reason |
| `docs/engine_divergence_ledger_20260728.md` standing rule 2 | **three**-term: omitted `engine_error` | corrected to four terms |
| `reports/c138_known_gaps_ledger.md` **H14** | had the four-term identity right, but claimed the lossy counter "has never fired" | corrected: it had fired twice in committed artifacts before the cell was written |
| `reports/c132_single_seat_coverage_bound.md` §2 | "`skip:strict_all_branches_lossy` is **not** an exit" — true of coverage, read as true of verdicts | clarified, with the distinction stated explicitly |
| **this report** §2 "every other `skip:*` … fire *before* `boundaries_measured`" | true when written; falsified by C142's `skip:rump_branch_set` | corrected in place above, with the two-condition membership rule that replaces the timing-only reading |
| **this report** §"closed at three values" and §"complete, not patched" | the verdict domain grew to four and the identity to five terms | corrected in place; the case-analysis method is unchanged and no number moved |
| `scripts/differential_denominator.py` rule 3 | two-term, and **correctly scoped** to four golden-corpus harnesses — but it publishes a rule about a counter literally named `boundaries_measured`, which is the vector by which the claim reached the transition differential | scope warning added; semantics unchanged, because they are right for those four harnesses |

### Correct but fragile — asserted of specific artifacts where the extra terms are 0

| site | disposition |
|---|---|
| `reports/c122_weather_entry_truncation.md` §6 | qualified; lossy re-derived as 0 on both committed artifacts, values quoted |
| `reports/c126_a6_white_herb.md` §5 | qualified; lossy re-derived as 0 on all four committed artifacts, values quoted |
| `reports/c139_encore_transform_move_index_results.md` ×4 | all four lines rewritten to show all four terms with measured values; **no number changed** |
| `reports/c111_residue_row_causes.md` §"Era and provenance" | qualified — and flagged as **unverifiable**, not merely fragile: `/tmp/sweep_f1.json` was never committed, so the lossy counter cannot be re-derived for that run |
| `reports/c117_validation_holdout_baseline.md` §1 | same: both artifacts are `/tmp` paths, never committed, so the reconciliation cannot be re-derived |

### Not this instrument — left alone

`scripts/leaf_vs_reality.py`, `scripts/leaf_root_parity.py`, `scripts/prior_mapping_assert.py`,
`scripts/fidelity_gate_events.py` and `tests/test_differential_denominator.py` /
`tests/test_denominator_adoption.py` all concern the golden-corpus denominator rule, whose
`attempted` counter has exactly two classifications. Two terms is right there.
`reports/c112_leaf_state_divergence_ledger.md`'s correction about `compared` is about the same
harness family. None of these was changed except for the scope warning above.

### No test asserted it at all — and that was the real hole

**Not one test in the repo pinned the boundary verdict partition, in either form.** So no test
was passing vacuously here; there was nothing to pass. The identity was carried entirely in prose
across the twelve files listed above, which is exactly why an artifact could refute it for months
without anything going red. `tests/test_boundary_verdict_partition.py` is new in this PR and is
the first mechanized form of it.

## 4. What is now mechanized

- `verdict_partition_failures()` in `scripts/engine_transition_differential.py` — the
  identity as a checkable function (four-term here; five since C142). It refuses a missing or non-integer `boundaries_measured`
  rather than defaulting it to 0, because a defaulted denominator makes the identity close on an
  unreadable report.
- `scripts/cert_sweep_readout.py` gates on it **per shard**, into `gate_failures`. Per shard
  rather than on the aggregate because two shards can violate it in opposite directions and sum
  to a clean total; there is a test for exactly that cancellation.
- `cert_sweep_readout.py`'s `in_support_rate` was
  `(boundaries_measured - transitions_diverged) / boundaries_measured` — the two-term identity in
  another costume, crediting every unadjudicable boundary as "in support". It is now
  `transitions_matched / (boundaries_measured - unadjudicated)`, with `boundaries_adjudicated`
  and `boundaries_unadjudicated` published beside it so the denominator adjustment is visible.
  On a run with no lossy verdict and no engine error the number is unchanged.
- `tests/test_boundary_verdict_partition.py` — **23 pins**, including four anti-vacuity ones: the
  globbed artifact corpus is pinned at an **exact** size (not a floor); some committed artifact
  really carries a nonzero lossy counter; every artifact in the corpus carries every verdict
  scalar; and `c26`/`c27` still **refute** the two-term form. Without the last, the module would
  pass identically in the repo state that produced the defect.

  > **The corpus selector was the one fail-open in review's ten-mutation battery, and it is
  > fixed here.** The selector required all three verdict scalars to be present, so it filtered on
  > exactly the keys the checker refuses: deleting `engine_errors` from
  > `reports/artifacts/c134_collapsed_dev_sweep.json` made that artifact **leave the corpus**
  > instead of failing it, and the whole 190-test suite stayed green. With a `> 40` floor against
  > 70 artifacts there were 29 more it could have eaten silently. Membership is now decided by
  > `boundaries_measured` alone — a key the checker does *not* validate — malformedness is the
  > checker's job, and the count is exact so a disappearance is a failure in its own right. The
  > reviewer's exact mutation now produces three distinct failures.
- `.github/workflows/engine-fidelity-gates.yml` — a `Boundary verdict partition` step with a
  pinned test count and a no-skip guard, plus filter entries for the test module,
  `cert_sweep_readout.py`, and the two counterexample artifacts, so deleting a counterexample
  cannot go green.

## 5. The pin that certifies this was already stale

`reports/certification_contract_lifecycle.json`'s `successor_pending_identity.differential_sha256`
was **broken before this PR touched anything**, and the record it carries was materially
incomplete. Bisected and confirmed independently:

- last accurate at **#1032** (`5d7f2ed0`, 2026-08-02);
- **broken by #1054** (`5475b2da`, 2026-08-03), so
  `tests/test_c26_damage_composition_readout.py::test_production_matcher_is_not_the_rejected_experiment`
  has been **red on `main` since 2026-08-03**;
- **seven** merged PRs touched the differential in between — #1054, #1059, #1086, #1107, #1122,
  #1135, #1149 — for a net **+470 / −12** lines;
- **four** of those are matcher or classifier changes whose classification effect was never
  declared at the pin: **#1054** (movepainsplit inherits the damage roll, 35 → 31), **#1059**
  (faint-without-damage synthesis and the Air Lock speed guard, 21 → 13), **#1086** (demote the
  drag limit to a last resort), **#1107** (forced-replacement ply runs no residual phase).

Re-deriving the hash here repairs the test, but the delta from `8f83a11f` **absorbs all four of
those undeclared changes**, so re-stamping this pin is not the routine no-op it looks like. That
whole record now lives in the artifact's own `why_pinned` field rather than only in a PR
description — which is the point of a repo whose thesis is that untested prose goes stale silently.
