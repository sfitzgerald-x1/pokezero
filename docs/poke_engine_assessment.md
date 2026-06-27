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

   Next unowned step: legal-action equivalence (step 3 below). The adapter currently consumes a
   hand-authored fixture; nothing yet reconstructs a fixture from a Showdown request payload or
   compares the engine's legal root options against what Showdown offers each seat.
3. Validate legal root actions against the Showdown request payload for both seats.
   (Next unowned step. The fixture adapter from step 2 is the input side of this; what remains is
   translating a real Showdown request into a `BattleSpec` and asserting the engine's legal options
   match Showdown's for both seats, including switches and forced/trapped cases.)
4. Validate one-turn instruction outcomes against Showdown for a small fixture matrix:
   damage move, status move, switch, forced switch, faint, Spikes, Toxic, Substitute, Hidden Power,
   Intimidate, and Flash Fire.
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
