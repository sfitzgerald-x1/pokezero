# C145 — H11 adjudicated: `19100170/71-72` is a world-construction row, closed by #1148

C116 Phase 4 item 12 requires every remaining divergence to end as an engine fix, a harness fix,
or **a limit with a written demonstration**, and says a classifier prefix is none of those. H11 is
the fourth of the five rows that condition covers, and it ended as **none of the three**: the
`reports/c138_known_gaps_ledger.md` cell records the rows as no longer observed, attributes the
closure to "**most likely #1148**", and says in its own words that this is "recorded here as an
observation, **not** a diagnosis" and that "nothing in `reports/` still explains the original rows."

This report replaces the guess with a measurement and the "no written cause" with a mechanism.

**Three findings, in order of how much they change the ledger.**

1. **The guess was right, and is now measured.** The closing commit is `d27316b6` (**#1148**,
   "Resolve the Encore lock against the post-Transform moveset"), bisected over the ledger's own
   `2ec0cb13..f876803e` range with a rebuilt, fingerprint-verified engine at each end of the
   transition. Both rows reproduce at `d27316b6`'s parent and neither survives `d27316b6`. §3.
2. **"Nothing in `reports/` still explains the original rows" is FALSE, and I withdraw it on the
   ledger's behalf.** `reports/c139_encore_transform_move_index_prediction.md` § Observation names
   this exact mechanism, on these exact two boundaries, by seed and step. It was merged **by
   #1148 itself** — i.e. inside the range the ledger was searching, and already on `main` at
   `f876803e` when the ledger cell was written. The ledger did not fail to find a diagnosis; it
   failed to find one that was two commits behind it. §2.
3. **The row was pinned and the pin was inert.** `tests/test_engine_world_encore_transform.py`
   shipped with the fix and holds the mechanism at unit level, but **no CI job runs that file** —
   zero matches for `encore` across all three workflows. It also stops at the lock *index* and
   never asserts that the index is what restores Showdown's `|-heal|` line. §5 wires the file into
   `mass-gate` and adds the missing cause-to-symptom assertion, red under a slot-0 mutant on
   current `main` and red at `dc6e1e19` against the real pre-fix code.

**No engine change ships with this report.** The one code change is a test file plus its CI step.

---

## 1. Scope, and what "measured" means here

Every number below was produced in this session from a rebuilt engine, not carried from any prose.
Four worktrees, four independent `scripts/build_search_crate_engine.sh` runs, each verified with
`scripts/engine_build_fingerprint.py --check` before any sweep:

| commit | PR | subject | patches | engine fingerprint |
|---|---|---|---|---|
| `2ec0cb13` | #1142 | Split the heal family into sub-cases | 68 | `907bea70abd1bf86…` |
| `dc6e1e19` | #1147 | C137: enumeration is the oracle, not the fix | 68 | `fdbf59379399b944…` |
| `d27316b6` | #1148 | Resolve the Encore lock against the post-Transform moveset | 68 | `fdbf59379399b944…` |
| `1a929c57` | #1161 | this report's `main` base | 71 | `5fa147ffa325c887…` |

`fdbf5937…` independently re-derives the fingerprint that
`reports/artifacts/c137_merged_base_holdout_sweep.json` and `reports/c139…_results.md` already
carry for `dc6e1e19`. Each worktree got its own `.venv` and its own build, so
`third_party/poke-engine-src` was vendored per tree by step `[1/8]`; nothing here was measured
against a tree a bare `git checkout` had left stale.

**`main` moved twice under this work, and every `1a929c57` below is left as written.** The branch was
cut at `1a929c57`, C143 (#1161) then C144 (#1163) landed, and the branch was rebased onto `ce962c6e`.
The mutant run and the `--check`ed build in §5 were genuinely taken at `1a929c57`, so restating them
as `ce962c6e` would be false. What licenses carrying them across is measured, not assumed:
`git diff a4e42034 HEAD -- rust/ third_party/` is **empty**, `--check` returns the same
`5fa147ffa325c887…` on the rebased tree, and the `19100170` sweep returns the identical 88 / 81 / 81 / 0
after C144's +189-line change to `engine_transition_differential.py`. The head-commit row of §3 and its
artifact were re-derived on the rebased tree rather than carried.

**The sweep window.** `19100170` is in the validation holdout (`19100000–19100199`), which is
sweepable. Nothing here ran at or above `19_200_000`; `FINAL_HOLDOUT_SEED_FLOOR`
(`scripts/engine_transition_differential.py:3085`) was never approached and
`--i-am-running-the-final-holdout` was never passed.

**Why one game and not 200.** The disposition is about two boundaries in one game, and
`--games 1 --seed-start 19100170` reproduces that game bit-for-bit: `boundaries_full_round` = 88
and `boundaries_measured` = 81 at all four commits, with an identical skip histogram. It is a
4.8-second measurement (`elapsed_seconds` 4.78 at `dc6e1e19`) against roughly 16 minutes for the
200-game form at that run's own measured 753.3 games/hour — which is what made a four-commit bisect
with a real build at each point affordable. The 200-game holdout censuses are **not** re-derived here;
they are already on `main` in `reports/artifacts/c137_merged_base_holdout_sweep.json` (4 diverged,
`itemleftovers` 2, at `dc6e1e19`) and `reports/artifacts/c138_collapsefix_mainhead_holdout_sweep.json`
(2 diverged, `itemleftovers` 0, at `f876803e`), and I re-read both to confirm the counters this
report leans on. The ledger's citation of the second filename is correct — it exists.

**The accept bar, stated because ~9 % of measured boundaries repo-wide clear it.** In game
`19100170`, 22 of 81 measured boundaries (27.2 %) are accepted through the Constraint-7 union over
enumerated hidden sleep-counter worlds — `gating:support` 22, `hidden_counter_support:sleep` 22,
`gating:exact` 59. **Neither of the two rows in question is one of them.** Both retained repros
carry `"gating": "exact"`, so a single candidate state was judged and the closure owes nothing to
the widened bar.

---

## 2. What the repo already knew, and where the ledger went wrong

Before writing a mechanism I grepped for one. `git grep 19100170` over `origin/main` returns, among
the sweep artifacts, three non-artifact files:

- `reports/c139_encore_transform_move_index_prediction.md` — § Observation states the mechanism in
  full, names `19100170/71-72` and the class `component_missing_in_engine:itemleftovers`, and
  identifies the suppressor as `end_of_turn_is_deferred`.
- `reports/c139_encore_transform_move_index_results.md` — the after-table, with the two rows listed
  as the two that close.
- `src/pokezero/engine_world.py` — the deferral comment inside `_build_side_spec` repeats the
  diagnosis at the site, citing the same two boundaries.
- and `tests/test_engine_world_encore_transform.py`, whose module docstring does the same.

All four arrived in `d27316b6`. So the ledger's two claims about the state of the record —
"no written cause anywhere in `reports/`" and "nothing in `reports/` still explains the original
rows" — were **already false when written**, by two commits. I state this as a documentation
defect rather than a research one: H11's *class* was correctly marked UNKNOWN under the ledger's own
"UNKNOWN with a named next measurement" rule, and its settling measurement was correctly specified;
the error is the negative, asserted over `reports/` as a whole from a search that missed a file
merged in the range the same cell was pointing at.

This is an instance of a family this effort keeps hitting — a report asserting that something is
absent from, or first in, the record when it is already written down. **I am deliberately not
attaching a count to that family here.** I was handed one and did not verify it, and repeating an
unverified figure is the same defect one level up; anyone who needs the tally should re-derive it
rather than cite this sentence.

What does generalise is the countermeasure, which is cheap and would have caught this one:
`git grep <the row id>` over **every tracked file** at the head being described, before writing any
negative about the record. Not a directory list — scoping a grep to chosen directories and reporting
the result as repo-wide is itself one of this program's recorded errors, and §4.5 and §5.1 below both
show the narrower version of my own searches missing files that the tracked-file version found.

---

## 3. The bisect: `d27316b6`, confirmed rather than accepted

Command, identical at every commit (strict matcher, no `--enumerate-rolls`, no
`--approximate-sleep`):

```
PYTHONPATH=src python scripts/engine_transition_differential.py \
    --games 1 --seed-start 19100170 --keep-repro 25 --json <out>
```

| commit | full_round | measured | matched | diverged | `divergence_classes` |
|---|---|---|---|---|---|
| `2ec0cb13` (range start) | 88 | 81 | 79 | **2** | `component_missing_in_engine:itemleftovers` 2 |
| `dc6e1e19` (`d27316b6`'s parent) | 88 | 81 | 79 | **2** | `component_missing_in_engine:itemleftovers` 2 |
| `d27316b6` (**#1148**) | 88 | 81 | **81** | **0** | `{}` |
| `662d9db8` (this branch, on `1a929c57`) | 88 | 81 | 81 | 0 | `{}` |

The boundary verdict partition closes on every line — and in its **four-term** form, because the
two-term reading is exactly what C144 refuted and this PR is rebased onto C144:
`matched + diverged + engine_errors + skip:strict_all_branches_lossy == boundaries_measured`, i.e.
`79 + 2 + 0 + 0 == 81` at the two pre-fix commits and `81 + 0 + 0 + 0 == 81` at the two post-fix
ones. A previous revision of this line asserted the two-term form. It gave the right answers *only*
because both other terms happen to be 0 in this game, which is the coincidence C144 exists to stop
anyone relying on. The skip
histogram is **byte-identical** across all four — `skip:single_seat_boundary` 8,
`skip:unmappable_choice:struggle_not_submittable` 6, `world_prestate_mismatch:p1_status` 1 — as is
the gating histogram. So the two boundaries were not converted into skips; they became `matched`.

Artifacts: `reports/artifacts/c145_g19100170_{2ec0cb13,dc6e1e19,d27316b6,head}.json`.

**`27609063` (#1150) was not built.** It is the only commit in the range I did not measure, and it
does not need measuring: it sits between `2ec0cb13` and `dc6e1e19`, both of which are measured
**red**, so no transition can lie in it. The transition is therefore uniquely located at
`d27316b6`, whose parent is red and which is green.

**A detail worth keeping, because it is the trap this bisect could have fallen into.** The engine
fingerprint *changes* across `2ec0cb13 → dc6e1e19` (`907bea70…` → `fdbf5937…`) and does **not**
change across `dc6e1e19 → d27316b6`. The crate moved without closing the rows, and then the rows
closed without the crate moving. A bisect that had trusted the fingerprint as a proxy for "did
anything relevant change" would have pointed at the wrong commit in both directions.

`git diff --numstat dc6e1e19 d27316b6` is the corroboration: ten sweep artifacts under
`reports/artifacts/`, two c139 reports, `src/pokezero/engine_world.py` (+182/−25) and
`tests/test_engine_world_encore_transform.py` (+590/−0). **No patch, no `rust/`, no crate source.**
The closure is Python world construction — which is also why the fingerprint could not move.

---

## 4. The mechanism, and the settling measurement H11 asked for

H11 specified: *replay both boundaries through `evaluate_boundary_strict` with the branch dump
retained and compare the engine's residual plan against Showdown's `|-heal|…|[from] item:
Leftovers` line.* Run at `dc6e1e19`, fingerprint `fdbf5937…`, via
`scripts/replay_residue.py` on the retained repros — which re-executes the same
`pokezero_search.branch_events` call `evaluate_boundary_strict` makes, on the exact recorded
candidate states. Full dump: `reports/artifacts/c145_settling_branch_dump.json`.

### 4.1 The state the world handed the engine

Both boundaries, side one, read out of the recorded state string itself:

- active expressed as `DELCATTY,100,…` with `pre_transform` `DITTO;153;153;153;153;153;TRANSFORM:15`
- volatiles `ENCORE:TYPECHANGE:TRANSFORMED`
- moves `BODYSLAM;false;5, HEALBELL;false;5, WISH;false;5, PROTECT;false;5`
- **`last_used_move=move:0`**

Showdown's own `|request|` for the same turn disables every move except Protect, i.e. Showdown's
Encore is locked on **Protect — donor slot 3**. The engine was told slot 0, **Body Slam**.

The cause is `_build_side_spec`'s pre-fix Encore bridge. Showdown locks Encore by move **id**; the
vendored gen3 engine locks by move **slot index**. For the self seat the bridge resolved the index
through `_resolve_encored_move_index(..., rows_for_active=_active_row_moves(side_payload))`, whose
rule is "the request disables every non-encored move, so exactly one enabled move identifies the
lock". But that payload row is deliberately the **pre-Transform** snapshot —
`local_showdown.actor_move_states_from_request_history` skips requests taken while transformed so
that PP stays honest — and for a gen3 randbats Ditto it is the single move `transform`. A one-move
list satisfies "exactly one enabled move" **spuriously**, yielding index 0; `_apply_transform` then
swapped the donor's moveset in underneath the surviving index, so `move:0` came to name Body Slam.

### 4.2 What the engine then did with it

One branch at `pct=100.00`, `lossy=[]`, at both boundaries — matching the retained
`branch_count: 1`. Raw instructions, `19100170/71`:

```
Switch SideTwo: P2 -> P3
SetLastUsedMove SideTwo: Move(M2) -> Switch(P3)
Damage SideTwo: 36            <- Spikes on the switch-in
DecrementPP SideOne: M0 1     <- M0 is Body Slam, not Protect
Damage SideTwo: 65            <- lethal: Delcatty was at 65 after Spikes
ToggleSideTwoForceSwitch      <- it fainted
```

and there the instruction list **ends**. There is no residual phase at all.

That is the whole row. The Encored lock is a *forced* choice, so the phantom Body Slam is not a
mispriced alternative — it is the only thing the engine can do. The load-bearing fact is simply that
it was **lethal**: the switch-in had taken Spikes down to 65 at step 71 and was a 2-HP Typhlosion at
step 72, and Body Slam off Delcatty's copied stats kills from there. The faint arms
`end_of_turn_is_deferred`, which defers the entire residual block to the forced-replacement boundary,
and **both** sides' Leftovers ticks disappear from a turn on which Showdown emitted them.

The component is labelled `capped_lethal`, and I am **not** offering "its magnitude equals the
target's remaining HP" as corroboration, as an earlier revision did.
`engine_transition_differential.py:551` *constructs* that component as `-remaining`, so **every**
lethal capped hit reads that way by definition. It is true and it is evidence of nothing.

Per-component, straight from the dump:

| | `19100170/71` | `19100170/72` |
|---|---|---|
| observed p1 | `itemleftovers +16` | `itemleftovers +16` |
| observed p2 | `spikes −36`, `itemleftovers +18` | `spikes −31` |
| engine p1 (`move:0`) | *(none)* | *(none)* |
| engine p2 (`move:0`) | `spikes −36`, `capped_lethal −65` | `spikes −31`, `capped_lethal −2` |

The recorded miss is `pct=100.00: p1 attributed components differ: observed_only=[('itemleftovers',
16)] engine_only=[]`, which names p1 only. **That is the matcher's report order, not the extent of
the loss.** `evaluate_boundary_strict` iterates slots `("p1", "p2")` and `break`s on the first
failure, so p2's lost tick at step 71 was never compared. The dump shows it lost too. Anyone
sizing this class from the miss string alone undercounts it — **by up to half, at boundaries where
both sides tick.** Scoped deliberately, because the stronger form is false: Showdown emits 2 ticks at
step 71 against 1 named, but only 1 at step 72 (Typhlosion holds no item), where the miss is
therefore complete. Across the class as observed that is **3 lost against 2 named — a third, not a
half.** Re-derived from `c145_settling_branch_dump.json`, not from the sentence it replaced.

### 4.3 One field, and the row becomes a component-identical match

Same engine build, same recorded state, same joint action — only side one's `last_used_move`
rewritten from `move:0` to `move:3`:

```
as built (move:0)                                   substituted (move:3)
|                                                   |
|switch|p2a: Delcatty|Delcatty, L96, M|101/290       |switch|p2a: Delcatty|Delcatty, L96, M|101/290
|-damage|p2a: Delcatty|65/290|[from] Spikes          |-damage|p2a: Delcatty|65/290|[from] Spikes
|move|p1a: Ditto|bodyslam|p2a: Delcatty              |move|p1a: Ditto|protect||[still]
|-damage|p2a: Delcatty|0 fnt                         |
|faint|p2a: Delcatty                                 |-heal|p1a: Ditto|161/258|[from] item: Leftovers
|                                                    |-heal|p2a: Delcatty|83/290|[from] item: Leftovers
                                                     |upkeep
                                                     |turn|65
```

The two right-hand heal lines are **character-for-character** Showdown's own:
`|-heal|p1a: Ditto|161/258|[from] item: Leftovers` and
`|-heal|p2a: Delcatty|83/290|[from] item: Leftovers`. The `move:3` component sets equal the observed
sets exactly on both slots at both boundaries — `p1 [itemleftovers +16]`, `p2 [spikes −36,
itemleftovers +18]` at step 71; `p1 [itemleftovers +16]`, `p2 [spikes −31]` at step 72 (Typhlosion
holds no item, and Showdown emits one heal line there, not two).

**This is what makes the disposition a world-construction row rather than an engine row.** The
engine build is byte-identical on both sides of that substitution — one field of one constructed
state is the entire difference between a 100 %-mass divergence and an exact match.

### 4.4 The disposition

**Row 4 is a fix, in shipped Python, and the fix was deliberate.** Not a limit: nothing about the
row was unknowable from public information — Showdown's request discloses the lock, and #1148 reads
it from `selfActiveMoves` (falling back to the payload's public `lastUsedMove`), resolving it
**after** `_apply_transform` against the moveset the engine will actually read, and still failing
closed as `encore_move_unknown` when the id is absent. Not a harness fix either:
`src/pokezero/engine_world.py` is production world construction, shared with
`engine_search.EngineMctsPolicy`, so the same wrong lock was reaching live search, not only the
differential. And **not incidental**: #1148 registered a prediction naming these two boundaries and
this class before measuring, with a "nothing opened" falsifier, and confirmed it on both windows.

**Could it regress?** Yes, and the answer to "without a pin" is §5: the mechanism *is* pinnable, it
*was* pinned, and the pin was not being executed anywhere. That gap is what this PR closes.

### 4.5 A second, smaller difference the settling measurement exposed

Showdown's protocol carries `|-fail|p1a: Ditto` on both turns, while the corrected (`move:3`) render
is a single 100 %-mass branch in which Protect **succeeds**. My first reading was that this was the
consecutive-Protect stall ladder. **It is not**, and the distinction matters:

- The reconstructed stall counter really is 0. Measured, not inferred: deserializing the recorded
  states gives `side_one.side_conditions.protect == 0` at both boundaries. At 0 the ladder's chance
  is 1, so the ladder cannot be what failed this Protect.
- Showdown's own gen3 `protect.onPrepareHit` is
  `return !!this.queue.willAct() && this.runEvent('StallMove', pokemon)` (`data/moves.ts`). **The
  ladder is the second conjunct and never ran.** p2 switched, a switch resolves before the move
  phase, so by the time Protect executes nothing is left to act, `willAct()` is false, and Protect
  fails on that clause alone. Consistent with the `[still]` tag on the `|move|` line.
- ⚠ **RETRACTED, and this is the correction that matters.** A previous revision of this section said
  the engine "renders the Protect as succeeding", that `side_conditions.protect += 1`
  (`generate_instructions.rs:5428`) "then fires", and that the engine therefore prices the next
  Protect at 1/2 against Showdown's 1. **All three are false, refuted by measurement on the recorded
  state.** I reached them by reading line 5428 instead of reading the instruction stream I already
  had in hand — and the effect was to *invent* an engine bug and point the next reader at it.

  Measured (`fdbf5937…` build, recorded `19100170/71` state, lock corrected to `move:3`,
  `poke_engine.generate_instructions`). **The actual boundary, p1 `protect` / p2 switches:**

  ```
  Switch SideTwo: P2 -> P3 · SetLastUsedMove SideTwo · Damage SideTwo: 36
  DecrementPP SideOne: M3 1 · Heal SideOne: 16
  ChangeVolatileStatusDuration SideOne ENCORE: 1 · Heal SideTwo: 18
  ```

  **No `ApplyVolatileStatus PROTECT`. No `ChangeSideCondition Protect`.** Single-variable control,
  only p2's choice changed from a switch to a move:

  ```
  DecrementPP SideOne: M3 1 · ApplyVolatileStatus SideOne: PROTECT · …
  RemoveVolatileStatus SideOne: PROTECT · ChangeSideCondition SideOne Protect: 1
  ```

  So the increment at `:5428` is reached only when the PROTECT volatile was applied, and on this
  shape it is not. **The ladder stays at 0, exactly as in Showdown, and there is no follow-on
  mispricing.** Behaviourally the engine *does* have the `willAct` equivalent here; the token search
  found no `willAct` because the gate is expressed differently, not because the behaviour is absent.

**What the difference actually is, then:** the engine omits Showdown's `|-fail|p1a: Ditto` on a
Protect the opponent's switch pre-empted. No volatile is applied, the stall counter does not move,
and no HP changes on either side — so on this shape it is a **protocol-render difference and nothing
more**, and both boundaries match at `d27316b6` and at `662d9db8`. I have not measured whether any
shape exists where it is more than that.

**Is the `willAct` clause already recorded?** My first search was
`protect_fail|protectfail|consecutive` over five named directories, and that was too narrow — it
would have missed a file that names the clause without those tokens. Widened to every tracked file:

```
git grep -l -iE "willact|will_act|protect_fail|protectfail|stallingmove" origin/main -- .
  docs/observation_v3_spec.md
  docs/silent_noop_sweep_plan.md
  src/pokezero/showdown.py
  third_party/poke-engine-gen3-protect-floor.patch
```

All four are about the **stall ladder**, which is the half that is *not* the cause here.
`docs/observation_v3_spec.md:76-96` derives the consecutive-successful-stall streak in detail and
notes `stallingMove: true` in passing; `silent_noop_sweep_plan.md:246` lists "consecutive-use
mechanics" as scope; `showdown.py` reconstructs the counter; the patch caps the ladder at 1/8. **None
of the four records the `!!this.queue.willAct()` conjunct.**

That glob was still the wrong shape, though, and a second widening is what a reader actually wants.
The repo *does* have prior art on "a Showdown **queue predicate** the engine lacks" — it is filed under
the sibling `willMove`, which `willact|will_act` cannot match:

```
git grep -n "queue\.will" origin/main -- .        # 5 tracked files
  docs/engine_fidelity_findings.md:472
  rust/pokezero-search/tests/gen3_encore_fidelity.rs:21
  scripts/gen3_switch_differential.py:1712
  third_party/poke-engine-gen3-encore-duration.patch:102
  third_party/poke-engine-gen3-patches.txt:254
```

All five are the Encore-duration compensation for `if (!this.queue.willMove(target)) duration++` — a
different predicate, and an *implemented* one. So the negative survives: `willAct` appears nowhere.
But `queue\.will` is the search term for this family, and stating only the narrower one would have
left a reader believing the repo had never met a queue predicate before.

**Why this is not filed as a §3 ledger row, corrected.** I first wrote that it "needs a
pool-reachability check", which is true but understates it: the ledger's own C125 standing rule
(`reports/c138_known_gaps_ledger.md:560`) *forbids* a §3 entry without a recorded reachability check,
so filing would have broken the rule rather than merely run ahead of it. Its correct home is §7's
undetermined list — not C125-gated, and exactly where "observed, mechanism known, incidence
unmeasured" belongs. It is filed there by this PR, which matters because this PR also *removes* an
item from §7 (H11's) and a §7 that only ever shrinks is a ledger of blind spots losing its blind
spots by attrition.

---

## 5. The pin: what existed, what did not, and the red/green

### 5.1 The existing pin is real, and CI never ran it

`tests/test_engine_world_encore_transform.py` (590 lines, 18 tests) shipped with #1148 and holds
the mechanism directly:
`test_transformed_self_encore_locks_the_post_transform_slot` asserts `move:3`, and
`test_the_fixture_really_is_the_spurious_one_move_snapshot` guards against the vacuous pass by
asserting the pre-Transform row really does hold exactly one enabled move.

But **nothing executes it.** The widest search I can run, over every *tracked* file rather than a
chosen directory list — the mistake this program has made before by scoping a grep to one directory
and reporting the result as repo-wide:

```
git grep -l -E "test_engine_world_encore_transform|encore_transform" origin/main -- .
  reports/c134_enumerate_rolls_oracle.md
  reports/c139_encore_transform_move_index_prediction.md
  reports/c139_encore_transform_move_index_results.md
  src/pokezero/engine_world.py

git grep -l -E "test_engine_world_encore_transform" origin/main \
  -- '*.yml' '*.yaml' '*.sh' '*.toml' '*.cfg' '*.ini' '*.mk' 'Makefile*' '*.py'
  src/pokezero/engine_world.py          <- a source COMMENT, not a runner
```

Three reports name the file (c134's gate table and both c139 reports) and one source comment cites
it. No workflow, no shell script, no `pyproject.toml`, no Makefile, no Python entry point runs it.
Corroborating from the other direction: a case-insensitive `grep -ci encore` against each of the
three workflow files at `1a929c57` returns **0, 0 and 0**, and the file appears in none of the 16
`python -m unittest` invocations across them (enumerated, not sampled).

`src/pokezero/**` *is* in the `mass-gate` path filter, so a PR touching `engine_world.py` triggers
the workflow — the workflow just never ran this file.

This PR adds a `mass-gate` step that runs the module, asserts the test count, and fails on any
`skipped`, following the pattern the A5 step already establishes in the same workflow ("Without
this, `OK (skipped=2)` reads as a pass. It did, locally, against a venv with no wheel"). It also
adds the test file to the `changes` filter, matching the convention the filter list already states
for pin files: *"Listed so a PR that ONLY weakens the pin file still trips this workflow."*

### 5.2 What the existing pin does not assert

Every test in the file stops at the lock **index**. None of them renders the constructed world, so
none asserts that the index is what puts Showdown's `|-heal|…|[from] item: Leftovers` line back.
The index was never the row; the suppressed residual block was. A future change could hold
`move:3` and still lose the tick — through `ResidualPlan`, through the deferral predicate, through
the renderer — and every test in this file would stay green.

`LockIndexToResidualBlockTests` (4 tests) closes that. It builds the world from the `19100170`
payload shape through `battle_spec_from_payload`, converts it with `build_poke_engine_state`, and
renders the joint action `(protect, switch)` through `pokezero_search.branch_events` — the same call
`evaluate_boundary_strict` makes. Two fixture facts are asserted rather than assumed: the
transformed Ditto sits more than one tick below max HP (so the heal is the full `maxhp // 16` and
not a clamp), and the opponent's switch-in is at 5 HP (so the phantom Body Slam is **lethal** and
actually arms the deferral). `pokezero_search` is imported **unguarded**, against the try/except
convention the rest of `tests/` uses, because a skip here would silently retire the only assertion
tying the lock index to the residual block.

### 5.3 Red before, green after — run, not reasoned

**(a) At the real pre-fix commit `dc6e1e19`.** The test module cannot be imported there —
`_sole_enabled_move_id` does not exist yet — so the fixture and the render were driven directly
against that worktree's own `engine_world.py`, built engine, fingerprint `fdbf5937…`:

```
last_used_move: move:0
|move|p1a: delcatty|bodyslam|p2a: swampert
|-damage|p2a: swampert|0 fnt
|faint|p2a: swampert
LEFTOVERS EVENTS: NONE
```

The same driver at `d27316b6` gives `move:3`, `|move|…|protect||[still]`, and both
`[from] item: Leftovers` lines. So the fixture discriminates the two eras on the real code.

**(b) On current `main`, as a mutant.** A separate worktree at `1a929c57` with one line of
`_apply_encore_locks` replaced by `index = 0` — the pre-fix behaviour, with every symbol still
present so the module imports. (No fifth build: the mutation is pure Python, so this run borrows
the `1a929c57` engine, fingerprint `5fa147ff…`. That is the point of the mutant — it isolates the
world-construction change from four commits of crate drift.) `9 of 22` tests fail, and the new pin
fails on its **symptom** assertion, `AssertionError: 0 != 2` (zero `item: Leftovers` events where
two are required), not on a restatement of the index. The index assertion is deliberately placed
**last** in that test so its failure message names the symptom rather than duplicating the index
test above it.

The mutant run is **paired with its own control in the same worktree**, because a mutation run
without one proves nothing about which change caused the red. Control: the shipped 22-test file with
`engine_world.py` *unmutated* → `Ran 22 tests … OK`. Then the one-line mutation, with
`git diff --stat` showing `src/pokezero/engine_world.py | 2 +-` as the only source change →
`FAILED (failures=9)`. Same worktree, same interpreter, same test bytes
(`md5 3f9f871ea0bd27ef75081861035347fe`, checked against the shipped file on every run).

My first attempt at that control was invalid and is recorded rather than quietly replaced: I reverted
the mutation with `git stash`, which also reverted the copied test file, so the "control" ran `main`'s
**18**-test version and its `Ran 18 tests … OK` said nothing about the new 4. Reverting only
`engine_world.py` gives the real control above.

**(c) Green, and not by skipping.** `Ran 22 tests … OK` at `1a929c57` + this PR, and `Ran 22 tests
… OK` with the shipped file copied into a clean `d27316b6` worktree whose engine
`--check`s as `fdbf5937…`. No `skipped` in either run.

| where | engine | result |
|---|---|---|
| `dc6e1e19`, real pre-fix code | `fdbf5937…` | **RED** — `move:0`, faint, zero Leftovers events |
| `1a929c57` + slot-0 mutant | `5fa147ff…` | **RED** — 9/22 fail, the pin on `0 != 2`; same-worktree unmutated control `Ran 22 tests, OK` |
| `d27316b6`, real post-fix code | `fdbf5937…` | GREEN — `Ran 22 tests, OK`, 0 skipped |
| `1a929c57` + this PR | `5fa147ff…` | GREEN — `Ran 22 tests, OK`, 0 skipped |

### 5.4 The new CI guard fires — constructed, not read off the branch

A guard added in another PR in this effort turned out to be dead code, so this one is not asserted
from its source text. The step's shell body was extracted verbatim and driven against a synthetic
log for each case it must catch, under `bash -e` — GitHub's default for `run:`, and this workflow
sets no `shell:` and no `defaults:` block that would change it:

| constructed case | step exit | which guard fired |
|---|---|---|
| `Ran 22 tests` / `OK` | 0 | none — passes, as it must |
| `Ran 21 tests` (a pin deleted) | 1 | `::error::expected 22 tests; the suite shrank or grew` |
| `Ran 23 tests` (a pin added) | 1 | same count guard |
| `Ran 22 tests` / `OK (skipped=4)` | 1 | `::error::an Encore-on-transform pin SKIPPED` |
| a pin genuinely failing (non-zero exit) | 1 | **the pipeline itself**, under `-e`, before any grep |

The last row is the one worth spelling out, because my first harness got it wrong: driven with a
`cat` that always exits 0, a log containing both `FAILED (failures=1)` and `Ran 22 tests` slipped
through both greps. Re-run with a stand-in that exits non-zero the way `python -m unittest` does, the
step aborts at the pipeline with exit 1 and never reaches the guards. So the hole is closed by `-e`
rather than by the greps — which is fine, but it is a property of the shell, and reading the two
`grep -q` lines alone would have suggested a gap that is not there and hidden the reason there isn't.

### 5.5 One pin this PR had to move, and why that is not a weakening

C144 (#1163) merged into `main` while this work was in flight and added
`tests/test_boundary_verdict_partition.py`, which pins the number of committed sweep artifacts
**exactly** — `_EXPECTED_SWEEP_ARTIFACTS = 70` — because a floor was the one fail-open in its own
mutation battery. The four bisect artifacts here take it to **74**, so the gate failed until bumped.

Bumping an exactness pin is the shape that deserves suspicion, so it was not done by arithmetic. The
selector was re-run over both trees: `origin/main` selects **exactly 70**, matching the pin, this
branch selects **74**, and the set difference is exactly the four `c145_g19100170_*.json` files with
**nothing removed** — so this is an addition, not the disappearance the pin exists to catch.
`c145_settling_branch_dump.json` is correctly *not* selected: it carries no top-level
`boundaries_measured`. All four then pass the checker itself rather than merely the count, closing the
four-term identity — `79 + 2 + 0 + 0 == 81` at the two pre-fix commits and `81 + 0 + 0 + 0 == 81` at
the two post-fix ones.

And the bumped pin was confirmed still live, by the same standard §5.4 applies to my own guard: set to
73 it fails with `AssertionError: 74 != 73`; at 74 the module is `Ran 23 tests … OK`. A bump that
silently disarmed the pin would look identical from the diff.

`#1159` also bumps that line, to 71 from its own 70. The edits conflict, and the resolution is
neither `71 + 4` nor `74 + 1`: re-derive from the merged corpus, which is the only reading an
exactness pin is worth anything under. Noted in the file and on the PR.

---

## 6. Ledger changes

- **H11** gets a real disposition: closing commit named and bisected, mechanism written,
  classified as a world-construction fix, with the "nothing in `reports/`" sentence withdrawn in
  place rather than deleted.
- **`c138` §7's list of undetermined items loses the `19100170/71-72` entry** (it was item 2). The
  remaining items are renumbered. `c138` `§7.1` is the only item cross-referenced anywhere — twice,
  from that ledger's §1 (`:49`) and §8 (`:586`) — and item 1 keeps its number, so nothing breaks.
- The **UNKNOWN roster in `c138` §1** drops from four rows to three: H8, H12 and H19. H11 is no longer
  UNKNOWN.
- **`c138` §7 gains an item as well as losing one** — the incidence of the missing Protect `|-fail|`
  line (this report's §4.5), as that ledger's new item 7, with its own settling measurement and the
  note that C125 (`c138` §8) is why it is not a `c138` §3 row. Added precisely because this PR also
  removes one: a list of blind spots that only ever shrinks is losing them by attrition rather than
  by resolution. Net `c138` §7 count is unchanged at nine.

## 7. What this does not claim

- It does not re-derive the 200-game holdout censuses. It reads the two that `main` already ships
  and re-derives, independently, the `dc6e1e19` engine fingerprint they carry.
- It does not measure `27609063`, and says why that is sound rather than a gap (§3).
- On §4.5: it **does** now settle what the difference is (a missing `|-fail|` line, with no volatile
  applied and the stall counter measured at 0 on both the boundary and its single-variable control),
  having previously and wrongly claimed a follow-on mispricing. What it does **not** measure is the
  *incidence* — whether any shape exists where the omission costs more than a protocol line — which is
  why the item goes to the ledger's §7 rather than §3. The unrecorded-ness of `willAct` is asserted
  from `git grep … origin/main -- .` over all tracked files, twice widened (once for
  `willact|will_act|protect_fail|protectfail|stallingmove`, once for `queue\.will`), not from a
  directory list.
- It says nothing about the four other rows of the C116 item 12 set. Row 4 is the subject; rows
  1–3 and 5 are elsewhere.
- No sweep at or above `19_200_000` was run. The final holdout is untouched by this report.
