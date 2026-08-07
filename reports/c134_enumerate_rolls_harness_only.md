# C134 — adopt roll enumeration for the differential harness only (C116 Phase 2)

**Status: PREDICTION REGISTERED, measurement pending.** Everything below the
`## Prediction` heading was written and committed before any sweep on this branch was
started. The commit that adds this file contains no code change and no artifact, so the
ordering is a property of the history rather than a claim in prose.

## What is being adopted

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

Default OFF. The only enabler in the repository is
`scripts/engine_transition_differential.py`, which sets it with `setdefault` before
`import poke_engine`. `tests/test_roll_enumeration_scope.py` holds that line by scanning
every tracked file, and separately measures a fan in a child process with the variable
unset to confirm the default build still collapses.

### Why the harness and not search

Adopt-everywhere was measured on the spike and **rejected on throughput**, not on taste:
`midgame_3v3` at depth 4 / 1024 sims went **2.38 ms → 8.88 s per decision, ~3700x**. That
is re-measured on this branch below. Harness-only carries no search-throughput risk at
all, because with the flag unset search executes the same instructions it does today.

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

## Measurement

*(to be appended after the sweeps complete; artifacts under `reports/artifacts/`)*
