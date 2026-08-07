# C137 Prediction: the Encore lock is bridged against the pre-Transform moveset

Registered **before** the measurement it commits to. Base commit
`aeaee2b1e3aa73a76c4054ae56e85dfd59269765` (`origin/main`), engine fingerprint
`07a3290d11ca14ecfa8c70f89a82a99e5bdc5a47d24136f740d54c59ab3122b4`.

## Observation

Showdown locks Encore by move **id**; the vendored gen3 poke-engine locks by move
**slot index** (`last_used_move = move:<i>`). `engine_world._build_side_spec` bridges the two,
and for a transformed self active it bridges against the wrong list.

For the SELF seat `encored_move` is `None` — `EngineMctsPolicy._public_effect_signals`
(`src/pokezero/engine_search.py`) only ever writes `encored[opponent_slot]`. Resolution
therefore falls through to `_resolve_encored_move_index(..., rows_for_active=_active_row_moves(side_payload))`,
whose self-seat rule is "the request disables every non-encored move, so exactly one enabled
move identifies the lock".

That payload row is, by deliberate design, the **pre-Transform** snapshot:
`local_showdown.actor_move_states_from_request_history` skips requests taken while
transformed so that PP stays honest. For a gen3 randbats Ditto the retained snapshot is a
single move, `transform`. A one-move list satisfies "exactly one enabled move" **spuriously**
and yields index 0. `_apply_transform` then swaps in the donor's moveset, so `move:0` now
names the donor's first move.

Reproduced on holdout seed `19100170` at the base commit: p1's active is a Ditto transformed
into Delcatty (`DITTO;...;TRANSFORM:15` as `pre_transform`, volatiles
`ENCORE:TYPECHANGE:TRANSFORMED`), Showdown Encored **Protect** (donor slot 3), and the built
world carries `last_used_move=move:0` — **Body Slam**. Steps 71 and 72 diverge as
`component_missing_in_engine:itemleftovers`: the phantom Body Slam KO makes
`end_of_turn_is_deferred` suppress the whole residual block, so the Leftovers tick never
appears on the engine side.

Turns 76-78 of the same game match only by coincidence — that Encore happened to lock Body
Slam, which *is* donor slot 0.

## Prediction

Resolve the Encore lock **after** `_apply_transform`, against the post-Transform moveset, and
for a transformed self active take the locked move from an **id-keyed** source
(the raw request's `selfActiveMoves`, whose single non-disabled entry is the lock; falling
back to the payload's `sides[slot]["lastUsedMove"]`) rather than from `_active_row_moves`.

Concretely, on the two sweeps (200 games each, `--keep-repro 25`, strict matcher):

1. **Holdout 19100000**: exactly the two rows `19100170/71` and `19100170/72`, both
   `component_missing_in_engine:itemleftovers`, close. `diverged` falls by exactly 2 and
   `matched` rises by exactly 2.
2. **Dev 19000000**: **no change at all** — identical `matched`, `diverged`,
   `engine_errors`, and an identical `divergence_classes` census.
3. `boundaries_full_round` and `boundaries_measured` are **unchanged** on both sweeps, and
   `matched + diverged == boundaries_measured` holds before and after.
4. The **skip histogram is unchanged** on both sweeps, including
   `skip:world_unsupported:encore_move_unknown` staying at its baseline value (expected: 1).

## Falsifier — "nothing opened"

This is the gate. The fix is **withdrawn, not shipped**, if ANY of the following is observed:

- **Anything opens.** Any `divergence_classes` key whose count *increases*, or any key that
  is present after and absent before, on either sweep. A net-zero trade (two rows close, two
  different rows open) counts as a failure, not a wash.
- **Dev moves at all.** Any change to dev's `matched`, `diverged`, `engine_errors`, or
  `divergence_classes` census.
- **Boundary counts change.** Any change to `boundaries_measured` or `boundaries_full_round`
  on either sweep — the fix must not convert a measured boundary into a skip, or the reverse.
- **The skip histogram changes**, in either direction, on either sweep. In particular,
  `skip:world_unsupported:encore_move_unknown` rising above its baseline would mean the fix
  bought its rows by failing worlds closed instead of by building them right.
- **`engine_errors` rises** above baseline on either sweep.
- **Fewer than 2 rows close on holdout.** Closing one of the two, or closing two rows other
  than `19100170/71` and `19100170/72`, falsifies the mechanism even if the totals look
  better.

The fail-closed `EngineWorldUnsupported("encore_move_unknown")` raise stays. If the locked id
cannot be found in the post-Transform moveset the world must still refuse to build; making it
fail open would trade a counted skip for an invented lock, which is the same class of defect
as the one being fixed.

Read `divergence_classes` for the census. `repros` is capped at `--keep-repro 25` and is not
a row count.

## Outcome

**CONFIRMED on every clause; the falsifier did not fire.** Holdout `diverged` 5 -> 3 and `matched`
15574 -> 15576, with `component_missing_in_engine:itemleftovers` 2 -> 0 and the two closed rows
being exactly `19100170/71` and `19100170/72`. Dev did not move on a single counter. Both
sweeps' `boundaries_full_round`, `boundaries_measured`, `engine_errors` and full skip histograms
are unchanged, `skip:world_unsupported:encore_move_unknown` included (holdout 1, dev 2). Nothing
opened on either sweep.

Full tables in `reports/c137_encore_transform_move_index_results.md`. Artifacts:
`reports/artifacts/c137_encore_transform_{dev,holdout}_sweep.json` (after) against
`reports/artifacts/c137_base_{dev,holdout}_sweep.json` (before, re-derived because no
`c136_faintcancels_fix_*` baseline exists on `main`), cross-checked against
`reports/artifacts/c137_base_pristine_{dev,holdout}_sweep.json`, which were produced in a
separate worktree of the base commit and agree on every counter.
