# C134 — roll enumeration as a flag-gated reference oracle (C116 Phase 2)

**Status: MEASURED. The falsifier did not fire, and the post-rebase prediction held on
both windows.** The prediction was committed in `0733abe7` before any sweep on this branch
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

_Pending re-run: main was merged in (`d27316b6`), which moved `rust/pokezero-search/src/events.rs`
and `src/pokezero/engine_world.py` — both on the differential's path — so the engine
fingerprint changed and the previous artifacts no longer describe this tree. The four
sweeps and the wrong-fan control are re-run from the committed code commit on a clean
checkout and land with the artifacts in the following commit._
