# Poke Engine Feasibility Assessment

Status: assessment, not an integration decision.

Inspected on 2026-06-27:

- `poke-engine` upstream commit `60e1cf8a2c70`
- `foul-play` upstream commit `995525517668`

## Recommendation

Evaluate `poke-engine` as an optional simulation/search backend for CPU self-play and MCTS, but do
not replace the current Showdown-backed ground-truth harness until a mechanics-equivalence spike
passes on Gen 3 random-battle fixtures.

The main reason to investigate it is structural: `poke-engine` uses reversible instruction-based
state transitions. From Python it exposes `State.apply_instructions(...)`,
`State.reverse_instructions(...)`, state serialization, instruction generation, damage calculation,
and MCTS wrappers. That is exactly the primitive needed to reduce branch/snapshot cost for search
and to make larger from-scratch self-play experiments cheaper on CPU.

## What Looks Promising

- `poke-engine` is MIT-licensed, so direct integration is feasible from a licensing standpoint.
- `foul-play` is GPL-3.0, so it should be treated as reference material or an external benchmark,
  not copied or imported into this repo unless the project intentionally accepts GPL obligations.
- `poke-engine` has explicit Cargo features for `gen1` through `gen9`, including `gen3`.
- `poke-engine` has Gen 3 source tests behind `--features gen3`.
- A narrow local check passed:
  `cargo test --no-default-features --features gen3 test_regular_move_with_protect_side_condition`.
- The Python binding already exposes the useful low-level pieces:
  `State.from_string`, `State.to_string`, `generate_instructions`, `apply_instructions`,
  `reverse_instructions`, `calculate_damage`, and `monte_carlo_tree_search`.
- The command-line interactive mode also supports `apply`, `pop`, and `pop-all`, which confirms
  the intended make/unmake model.

## Main Risks

- The upstream README explicitly says the engine is not a perfect Showdown replacement. For
  training/evaluation, Showdown should remain the source of truth until equivalence is demonstrated.
- Gen 3 feature support exists, but Gen 3 random-battle equivalence is unproven. Random-battle
  integration needs correct set/level/item/ability/move translation, Hidden Power handling, public
  hidden-information treatment, and Showdown protocol-to-engine state reconstruction.
- The Python API exposes reversible state operations, but it does not expose every root-option helper
  directly. If the existing MCTS wrappers are too opinionated, we may need a small PyO3 extension for
  legal-option enumeration and lower-level search control.
- Mechanics mismatches would contaminate training if used as the rollout engine. Any adoption should
  start with side-by-side replay tests against Showdown.
- Foul Play's search/evaluation code may be useful as an external sparring bar, but not as a teacher
  to clone. The project goal is self-generated improvement, not a stronger imitation ceiling.

## Proposed Spike

1. Add an optional `poke-engine` dependency path outside the default install.
   The initial API preflight command is:
   `python -m pokezero.engine_cli doctor`.
   This verifies the Python reversible-state API seam and prints the recommended Gen 3 build
   command; it does not prove Gen 3 mechanics equivalence by itself.
   `python -m pokezero.engine_cli doctor --smoke` adds a real apply/reverse smoke: it builds a
   minimal Gen 3-compatible battle state (Charmander ember vs. Squirtle watergun), generates the
   instruction branches, applies and reverses several of them, and confirms the serialized state
   returns to the original while at least one branch actually mutated state. The smoke only runs
   when the API probe is ready and exits nonzero if the round-trip fails. This is still only a
   reversible-API smoke; it is **not** Showdown or Gen 3 random-battle equivalence, which the
   fixture-based steps below still own.
2. Build a tiny adapter from a curated Showdown Gen 3 battle fixture into a `poke_engine.State`.
   Done: `src/pokezero/poke_engine_adapter.py` adds curated `BattleSpec`/`SideSpec`/`PokemonSpec`/
   `MoveSpec` dataclasses, `build_poke_engine_state(spec, module=None)`, a `minimal_gen3_fixture()`
   helper (Charmander/Ember vs. Squirtle/Water Gun, matching the `doctor --smoke` state), and
   `run_adapter_reversible_smoke(...)`, which reuses the backend round-trip core to confirm the
   built state generates branches that apply/reverse cleanly. It is optional (lazy poke-engine
   import) and intentionally not wired into rollout/training/search. Most tests run against an
   in-process fake module so CI never needs the native wheel; the real-engine round-trip test is
   local/optional and skips when `poke-engine` is absent, so CI does not prove real-wheel
   compatibility on its own. This still only exercises the construction + reversible seam; it is
   **not** Showdown mechanics equivalence.

   Next step: legal-action equivalence (step 3 below). Partially addressed: the comparison
   scaffolding now exists, but the real engine binding cannot enumerate legal options yet.
3. Validate legal root actions against the Showdown request payload for both seats.
   **Partially addressed.** `src/pokezero/poke_engine_legal_actions.py` adds the singles-only
   comparison seam: `request_legal_actions(request)` derives expected labels from a Showdown-style
   request (active moves in request order minus disabled ones, plus legal bench switches honoring
   fainted/active/confirmed-`trapped`/`forceSwitch` rules; `maybeTrapped` remains switchable because
   it is not conclusive), `engine_legal_actions(state, side)` derives labels from the engine's own
   root-option enumeration, and `compare_legal_actions(...)` returns a `LegalActionEquivalence`
   (`supported`, `request_actions`, `engine_actions`, `missing_from_engine`, `extra_from_engine`,
   `reason`). The two sides are derived independently (request payload vs. engine state), so a match
   is real agreement rather than a tautology.

   The blocker is the binding: poke-engine 0.0.47's Python API exposes no root-option enumerator
   (no `get_all_options`/`root_get_all_options`; only `generate_instructions(state, m1, m2)`, which
   takes moves as input rather than listing legal ones). So the engine side currently returns
   `supported=False` with an actionable reason, and a real-engine test asserts exactly that instead
   of failing. The fake-provider tests prove the comparison itself is correct once options are
   available. **Remaining:** a small PyO3 wrapper over Rust `State::root_get_all_options` to export
   legal options to Python, then re-run `compare_legal_actions` against the real engine for both
   seats (including switches and forced/trapped cases). This is fixture-only and is not wired into
   rollout/training/search.
4. Validate one-turn instruction outcomes against Showdown for a small fixture matrix:
   damage move, status move, switch, forced switch, faint, Spikes, Toxic, Substitute, Hidden Power,
   Intimidate, and Flash Fire.
   **Partially addressed: the Showdown ground-truth side now exists.**
   `src/pokezero/showdown_fixture.py` adds a curated one-turn fixture runner: `FixturePokemon` +
   `pack_team(...)` build Showdown packed-team strings for simple Gen 3 sets, and
   `run_one_turn_fixture(...)` starts a one-battle `BattleStream` (custom Gen 3 format
   `gen3customgame`, discovered from the built checkout) with two supplied teams, two first-turn
   choices, and a deterministic seed, returning a structured `OneTurnFixtureResult` (omniscient
   protocol lines, both seats' opening requests, submitted choices, terminal flag, and any
   `|error|`/protocol error lines). To carry curated teams, `scripts/battle_bridge.mjs` `start` now
   accepts per-player `{name, team}` options and passes the packed team through to Showdown while the
   string-name form preserves existing random-battle behavior. Tests cover packed-team generation and
   the bridge start payload without the poke-engine wheel; a node + built-checkout integration test
   runs one deterministic turn (Charmander/Ember vs. Squirtle/Water Gun) and asserts both moves fire
   with no `|error|` lines. This is **Showdown ground truth only** — it does not yet build or compare
   against a `poke_engine` state. The runner submits one pair of choices and returns at the next
   boundary, so matrix rows that create a faint followed by a forced-switch request will need a
   follow-up replacement driver before they can be fully resolved.

   **First damage comparator exists and currently FAILS to match on poke-engine 0.0.47.**
   `src/pokezero/poke_engine_outcomes.py` adds the first engine-vs-Showdown one-turn outcome
   comparator. It builds a poke-engine `BattleSpec` from the **real opening request** of a
   `OneTurnFixtureResult` (species/level from `details`, hp/maxhp from `condition`, the five battle
   stats from `side.pokemon[].stats`, moves from the request, ability/item when present, and types
   from the existing Showdown dex loader, normalized with the shared id helper), parses the seeded
   Showdown turn into an observed final active HP tuple, enumerates poke-engine instruction branches
   for the same two move choices via `generate_instructions`, applies each branch with
   `apply_instructions` to read its final active HP, and returns a structured `OutcomeComparison`
   (`supported`, `matched`, `showdown_final_hp`, `engine_final_hp_outcomes`, per-branch
   percentages/descriptions, `reason`/`notes`, and a serializable `to_dict()`).
   `run_charmander_squirtle_outcome_comparison(...)` wires the Showdown one-turn runner and the engine
   comparison together for the curated Charmander/Ember vs. Squirtle/Water Gun damage smoke when both
   a built local Showdown checkout and poke-engine are available.

   On poke-engine 0.0.47 this comparator **reports a mismatch**: Showdown's seeded turn ends at
   Charmander 127/219 and Squirtle 209/229, while the engine's branches from the same request-derived
   stat state produce tuples like `(122, 207)` (most likely), `(25, 207)`, `(122, 179)`, etc. — the
   observed `(127, 209)` appears in no engine branch (`matched=False`). The damage numbers are close
   but not exact, so **one-turn outcome equivalence remains unproven/failed for now** on this fixture.
   This is honest feasibility evidence, not an adoption gate; it is fixture-only and is not wired into
   rollout/training/search/benchmarks/self-play. Unit tests cover request->`BattleSpec`/stat extraction
   (no wheel required) and the matching logic with a fake engine/branches; the optional real
   integration test runs only with a built Showdown checkout + node + poke-engine and asserts the
   comparator executes and reports the known mismatch (it does not fake a pass).

   **Mismatch diagnostic exists (diagnosis, not adoption).** `poke_engine_outcomes` now also builds a
   serializable `OneTurnDamageDiagnostic` (`build_one_turn_damage_diagnostic(...)`, with
   `run_charmander_squirtle_damage_diagnostic(...)` for the curated fixture). It records, for the
   compared turn: Showdown's observed final HP and per-side damage **deltas** from the opening request
   HP; an active-state summary for each seat (species, level, hp/maxhp, ability, item, atk/def/spa/spd/
   spe, and move ids/pp from the Showdown request, plus dex-derived types); each engine branch's
   percentage, final HP, per-side deltas, and description; and the engine's direct
   `calculate_damage(state, m1, m2, side_one_moves_first)` output for both turn orders when the binding
   exposes it (reported as an explicit unsupported reason, not an exception, when the function is
   missing, raises, or returns an unrecognized/non-finite shape).

   The diagnostic's `likely_mismatch_surface` is deliberately conservative. The active-state summaries
   are the *Python* request-derived spec fed into `build_poke_engine_state` (stats/hp/moves from the
   request, types from the dex); confirming they match the Showdown request shows the request-derived
   spec is faithful, but it does **not** prove the engine's internal state matches — no engine-state
   inspection is done. So the surviving mismatch is left on the broad surface of the engine's
   damage/data or the spec→engine state-translation path
   (`likely_mismatch_surface = "engine damage/data or state-translation path"`). The **exact** root
   cause — move base power/category, type-effectiveness table, stat usage, rounding, or a translation
   defect — remains **UNRESOLVED**; the diagnostic records the surface, not a proven cause, and still
   reports `matched=False`. Unit tests cover the assembler end-to-end against a fake engine (both
   matched and mismatched branches, engine-branch deltas, surface/notes, and direct-`calculate_damage`
   payload propagation with no native wheel or built Showdown), observed-delta computation, the
   request-derived active-state summary (no wheel), the strict-JSON `_coerce_jsonable` guard (non-finite
   floats rejected), and the direct-`calculate_damage` probe (simple shape plus graceful unsupported on
   missing/raising/unknown shape); an optional real integration test (built Showdown + node +
   poke-engine 0.0.47) asserts the diagnostic runs, records the known mismatch, and exposes a
   serializable, non-empty direct `calculate_damage` output (the local 0.0.47 binding supports it). It
   is fixture-only and is not wired into rollout/training/search/benchmarks/self-play.

   **Remaining:** extend the comparator across the full matrix above (status, switch, forced switch,
   faint, Spikes, Toxic, Substitute, Hidden Power, Intimidate, Flash Fire), and reconcile the Gen 3
   damage/data/mechanics difference the diagnostic now surfaces (or pin an engine build/config that
   matches) before engine outcome equivalence can be considered proven; today it is unproven.
5. Benchmark apply/reverse branch throughput against the current replay-from-root branch harness.
6. If equivalence and speed are good, add an optional search backend that keeps Showdown as final
   benchmark/evaluation truth.

## Decision Bar

Adopt `poke-engine` only if the spike shows:

- Gen 3 random-battle fixtures match Showdown closely enough for training/search use.
- Applying and reversing branches is materially faster than current replay-from-root branching.
- The adapter can preserve player-relative hidden-information boundaries.
- Integration stays optional so the current Showdown harness remains available as the correctness
  oracle.
