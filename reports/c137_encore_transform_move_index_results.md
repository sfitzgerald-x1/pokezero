# C137 Results: the Encore lock now indexes the post-Transform moveset

Outcome of `reports/c137_encore_transform_move_index_prediction.md`, which was registered
**before** any of these numbers were measured (commit `05aef35f`, one commit ahead of the base
`aeaee2b1`, adding nothing but the prediction).

**Prediction confirmed on every clause. The falsifier did not fire: nothing opened, dev did not
move, no boundary count changed, and the skip histogram is identical on both sweeps.**

## The baseline is the base commit, verified by fingerprint

No `c136_faintcancels_fix_{dev,holdout}_sweep.json` exists on `main`, so the baseline was
re-derived. It was re-derived **twice**, independently, because a bad baseline is the cheapest
way to manufacture a true-looking claim:

| artifact | `source_commit` |
|---|---|
| `reports/artifacts/c137_base_{dev,holdout}_sweep.json` | `05aef35f` (prediction commit; src identical to base) |
| `reports/artifacts/c137_base_pristine_{dev,holdout}_sweep.json` | `aeaee2b1` (base commit, separate worktree) |

Both carry engine fingerprint `07a3290d11ca14ecfa8c70f89a82a99e5bdc5a47d24136f740d54c59ab3122b4`,
the same wheel the after-runs used. The two baselines agree on **every counter, every
`divergence_classes` key, every skip bucket, and the identity of all repro rows** — they differ
only in `games_per_hour` and the recorded `source_commit`. The `_base_` pair is the one the
before/after tables below use; the `_base_pristine_` pair exists so that claim is checkable rather
than asserted.

## Dev, seed-start 19000000, 200 games

| | before | after |
|---|---|---|
| `boundaries_full_round` | 15968 | 15968 |
| `boundaries_measured` | 15503 | 15503 |
| `matched` | 15501 | 15501 |
| `diverged` | 2 | 2 |
| `engine_errors` | 0 | 0 |

`matched + diverged == boundaries_measured`: `15501 + 2 == 15503` before and after.

`divergence_classes` (complete census, both sides):

| class | before | after |
|---|---|---|
| `component_magnitude:heal` | 1 | 1 |
| `component_missing_in_engine:sandstorm` | 1 | 1 |

**No counter in the dev report changed at all** — not one of the 6 skip buckets, not one class,
not one total.

## Holdout, seed-start 19100000, 200 games

| | before | after |
|---|---|---|
| `boundaries_full_round` | 16155 | 16155 |
| `boundaries_measured` | 15579 | 15579 |
| `matched` | 15574 | **15576** |
| `diverged` | 5 | **3** |
| `engine_errors` | 0 | 0 |

`matched + diverged == boundaries_measured`: `15574 + 5 == 15579` before, `15576 + 3 == 15579`
after.

`divergence_classes` (complete census):

| class | before | after |
|---|---|---|
| `component_extra_in_engine:spikes` | 1 | 1 |
| `component_missing_in_engine:itemleftovers` | **2** | **0** |
| `limit:roll_divergent_lethality` | 2 | 2 |

The only counters that moved in the entire holdout report are the three implied by that one row:
`transition:matched` 15574 -> 15576, `transition:diverged` 5 -> 3, and
`divergence_class:component_missing_in_engine:itemleftovers` 2 -> 0.

The two rows that closed are exactly the two predicted, `19100170/71` and `19100170/72`. At
`diverged` = 5 and 3 the `repros` list is well under the `keep_repro` = 25 cap, so it enumerates
the residue in full:

```
before: 19100107/135 limit:roll_divergent_lethality
        19100170/71  component_missing_in_engine:itemleftovers   <- closed
        19100170/72  component_missing_in_engine:itemleftovers   <- closed
        19100180/24  component_extra_in_engine:spikes
        19100191/5   limit:roll_divergent_lethality
after:  19100107/135 limit:roll_divergent_lethality
        19100180/24  component_extra_in_engine:spikes
        19100191/5   limit:roll_divergent_lethality
```

## Skip histogram — unchanged, both sweeps

Confirming the diagnosis's claim rather than trusting it. `skip:world_unsupported:encore_move_unknown`
is **1** on holdout (the value the diagnosis named) and **2** on dev; both are unchanged. If the
fix had bought its rows by failing worlds closed, this is where it would show.

| bucket | dev before | dev after | holdout before | holdout after |
|---|---|---|---|---|
| `skip:single_seat_boundary` | 1742 | 1742 | 1813 | 1813 |
| `skip:unmappable_choice:struggle_not_submittable` | 118 | 118 | 233 | 233 |
| `skip:world_unsupported:encore_move_unknown` | 2 | 2 | 1 | 1 |
| `skip:world_unsupported:materialization_blocker` | 18 | 18 | 8 | 8 |
| `skip:world_unsupported:self_request_state_unsupported` | 13 | 13 | — | — |
| `skip:world_unsupported:volatile_unsupported` | 144 | 144 | 127 | 127 |

## The mechanism, observed directly

Instrumenting `battle_spec_from_payload` on seed `19100170` and printing every Encore lock built
for the transformed p1 active:

```
turn=64 moves=['bodyslam','healbell','wish','protect'] last_used_move=move:3
turn=65 moves=['bodyslam','healbell','wish','protect'] last_used_move=move:3
turn=76 moves=['bodyslam','healbell','wish','protect'] last_used_move=move:0
turn=77 moves=['bodyslam','healbell','wish','protect'] last_used_move=move:0
turn=78 moves=['bodyslam','healbell','wish','protect'] last_used_move=move:0
```

Turns 64-65 are steps 71-72, the two divergences: the lock moved from slot 0 (Body Slam) to slot 3
(Protect), which is what Showdown Encored. Turns 76-78 still resolve to slot 0 — and that is the
point: the diagnosis said those turns matched **by coincidence**, because that later Encore
happened to lock Body Slam, which really is donor slot 0. They still resolve to slot 0, now by
id rather than by accident, and they stay matched.

At the boundary itself the two id-keyed sources agree and the old source does not:

```
selfActiveMoves = [bodyslam disabled, healbell disabled, wish disabled, protect ENABLED]
sides.p1.lastUsedMove = "protect"
_active_row_moves(sides.p1) = [transform]      <- the pre-Transform snapshot, one move
```

A one-move list has exactly one enabled entry, so the self-seat rule "exactly one enabled move
identifies the lock" was satisfied spuriously and returned slot 0.

Single-seed control, same seed, same wheel: `81` measured boundaries before and after;
`79 matched / 2 diverged` before, `81 matched / 0 diverged` after.

## A coverage change on the OPPONENT seat, unobserved but real

Found by independent review of #1148, which correctly rejected an earlier claim in that PR's risk
assessment that opponent-seat behaviour was untouched. **It is not.** Deferral is keyed on
`transformed_active`, not on the seat, so a transformed *opponent* takes the new path too.

Reproduced directly, same fixture both times, only `src/pokezero/engine_world.py` swapped:

```
p2 in transformed_slots, encored_moves={"p2": "protect"}, p2 active = Ditto -> Delcatty

base aeaee2b1:  REFUSED encore_move_unknown: side 'p2' is encored but the locked
                move cannot be determined
HEAD  ca64be6e:  BUILT   moves=['bodyslam','healbell','wish','protect']
                last_used_move=move:3
```

Before deferral, the opponent's `encored_move` was matched against the *transformer's own*
moveset — Ditto's `[transform]` — so a genuine lock like `protect` was absent and construction
raised, producing a counted `encore_move_unknown` skip. It now resolves against the copy and
builds.

The direction is right (a correct world beats a refusal) and it still fails closed on an id the
copy does not contain, so no number in this report is wrong. But it **can convert an
`encore_move_unknown` skip into a measured boundary**, and in this harness a silent
world-construction change corrupts measurements rather than failing loudly. It did not fire in
either 200-game window — review independently enumerated all three `encore_move_unknown` refusals
across both windows and confirmed every one is on a *non-transformed* active — but it is plainly
reachable, since gen3 randbats sets carry Encore and opposing Ditto transforms are the reason this
code exists. It is now pinned by `EncoreOnATransformedOpponentTests` rather than merely unobserved.

**Refusal ordering** also changes for a transformed side, in one narrow way. The "no id at all"
refusal still fires inside `_build_side_spec`, ahead of `self_world_mismatch` and
`transform_unexpressible`. The "id absent from the moveset" refusal now fires in
`_apply_encore_locks`, i.e. *after* both — so a transformed side failing both checks is attributed
to the earlier one. Non-transformed sides are unaffected, and the measured skip histogram is
unchanged on both windows.

## Gates

- `tests/test_engine_world_encore_transform.py` — **18 tests** (8 original + 10 added under
  review). The original 8 were **RED at the base commit**: 5 failures + 1 error, the headline row
  failing `AssertionError: 'move:0' != 'move:3'`.
- **Mutation coverage.** Review ran six mutations against the original 8; five were caught and one
  survived — relaxing `_sole_enabled_move_id` from "exactly one enabled move" to "the first of
  many". That guard's spurious satisfaction on a one-move pre-Transform snapshot **is this
  defect**, so its survival mattered. Confirmed surviving all 99 tests then in force, and now
  caught by 4:

  | test | failure under the M6 mutation |
  |---|---|
  | `test_two_enabled_request_moves_do_not_identify_a_lock` | `AssertionError: 'move:0' != 'move:2'` |
  | `SoleEnabledMoveIdTests::test_two_enabled_rows_identify_nothing` | `AssertionError: 'bodyslam' is not None` |
  | `SoleEnabledMoveIdTests::test_four_enabled_rows_identify_nothing` | `AssertionError: 'bodyslam' is not None` |
  | `SoleEnabledMoveIdTests::test_the_non_transformed_path_inherits_the_same_rule` | `AssertionError: 0 is not None` |

  The first is the integration pin and reproduces the defect's own signature: with the rule
  loosened, an ambiguous two-enabled request resolves to slot 0 again. Three further mutations of
  the new code are caught too — `_apply_encore_locks` falling back to slot 0 instead of raising
  (2 failures), the transformed branch consulting the pre-Transform snapshot again (1 failure,
  4 errors), and deferral removed entirely (6 failures, 2 errors).
- Review also fuzzed `_resolve_encored_move_index` old-versus-new over 200,000 inputs and found
  byte-identical output, and established that Encore-before-Transform is unreachable (Transform
  carries `failencore: 1`) and that Baton Pass cannot carry Encore (`noCopy: true`).
- `tests.test_poke_engine_patch_stack`, `tests.test_branch_mass_reconstruction`,
  `tests.test_final_holdout_guard`: 23 tests, OK. Combined with the two world modules above:
  132 tests, OK.
- `tests.test_engine_world`: 91 tests, OK. `tests.test_engine_world_live_typechange`,
  `test_engine_world_stall_counter`, `test_transformed_move_state_retention`,
  `test_world_trace_and_toxic_seeding`, `test_engine_search`: 182 tests, OK.
- `RUSTFLAGS="-C debug-assertions=yes" cargo test` in `rust/pokezero-search`: exit 0, 33 suites,
  407 passed / 0 failed / 1 ignored (the pre-existing `a_near_full_hp_seeder_still_over_books_the_drain_slot`).

No run touched the reserved final holdout; the highest seed measured is `19100199`.

The review-driven additions are documentation and tests only. All 78 code objects in
`src/pokezero/engine_world.py` are byte-identical to the version the sweeps ran against once
docstrings are excluded, so the numbers above stand without re-measurement.
