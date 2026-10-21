# GPU engine throughput spike

This isolated experiment asks whether a sampled-chance Gen 3 transition,
placeholder observation and the champion forward can sustain 100 completed
games per second on one GPU. It is not a production engine or training teacher.
Nothing here is imported by the existing battle collector or search runtime.

## Current implementation

Day 1: batched structure-of-arrays storage, per-battle PRNG streams, the
nine-action layout, an offline real-team generator/NPZ packer and incomplete
turn integration. No complete transition, encoder, policy forward, fidelity result
or games-per-second benchmark is implemented yet. Empty states
are storage fixtures and are terminal by default; they are not benchmark games.

State has six party members per side; all fields are JAX arrays with a leading
batch dimension. Types, status, items, abilities and moves require the eventual
team pool's vocabulary. Bench actions enumerate party order excluding the
active slot. Move0 represents synthetic Struggle only when no move is usable.
The initial mask handles PP, externally resolved disabled/trapped flags,
forced replacement and fainted slots; Taunt, Encore, recharge and other
restrictions still require the transition and are not claimed as supported.
For unilateral forced replacement, the initial transition suppresses the
other seat's action. The mask alone is not a complete request-phase protocol.

## Run the first device check

From the checkout root, in a fresh Python 3.12 environment on a supported NVIDIA GPU:

```sh
pip install -r gpu_engine/requirements-cuda.txt
python -m gpu_engine.probe
python -m unittest discover -s tests -p test_gpu_engine_state.py -v
```

Select the CUDA wheel against the installed driver using the
[official JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
The probe requires exactly one visible GPU and waits for a compiled operation.
It is a capability check, not an engine throughput measurement.

On October 21, 2026, JAX 0.12.0 on one NVIDIA GB200 (ARM64, driver 580.178.04)
passed the synchronized GPU probe and all nine state/action-mask tests. CUDA
reported an unavailable FABRIC memory-handle path and successfully retried with
simpler handle types; no permission or driver setting was changed. This proves
the first device capability, not engine speed or game correctness.

## Generate the offline team pool

Use a pinned, compiled Showdown checkout with its runtime dependencies installed.
The generator calls Showdown's real Gen 3 random-team generator, never a synthetic
species/move sampler. Node runs only during this preparation, not the device step.

```sh
python -m gpu_engine.team_pool --showdown-root /path/to/pokemon-showdown \
  --source-commit-label FULL_40_CHARACTER_COMMIT --teams 200000 \
  --out /path/outside/checkout/real-teams-200000.npz
python -m unittest discover -s tests -p test_gpu_engine_team_pool.py -v
```

The packer records the compiled source-tree and RNG-dependency digests, deterministic team seeds,
Gen 3 stats/PP/Hidden Power, numeric vocabularies and actual move-slot frequencies.
It requires the complete chronological stream and successful generator exit,
publishes exclusively without overwriting, and uses no pickled object arrays.
The source commit argument is a provenance label; the compiled-tree digest binds
the actual generator input. Keep the resulting NPZ outside git. The frequency
ledger starts with zero implemented coverage: generating teams does not establish
transition coverage, game correctness or throughput. An actual 200,000-team
generation remains a separate runtime validation, not a unit-test conclusion.
Short real movesets retain trailing zero-move/zero-PP slots, which are excluded
from coverage counts. Cosmetic forms such as Unown-G retain distinct species IDs.

## Damage and status core components

`mechanics.py` supplies batched Gen 3 damage arithmetic, separate accuracy/evasion
stages, paralysis speed, before-move status transitions and major-status residuals.
It preserves the pinned Showdown Q12 unsigned truncation and rounding, damage
modifier order and toxic's floor-before-stage rule. The caller must still resolve
items/abilities, move-specific effects and the full event scheduler. These helpers
do not constitute a battle step or increase the pool's implemented-move coverage.

```sh
NODE=/path/to/node POKEZERO_SHOWDOWN_ROOT=/path/to/pokemon-showdown \
  python -m unittest discover -s tests -p test_gpu_engine_mechanics.py -v
```

The optional source oracle calls the pinned Showdown arithmetic functions on
authored inputs. It is not the required 200-game divergence check. JIT/component
checks likewise supply no games-per-second result.

## Initial turn integration

`transition.py` initializes real teams from the pool and connects fixed-shape
action ordering, switches, PP, status action gates, static single-hit damage,
major-status residuals, forced-replacement pseudo-turns and party exhaustion.
Invalid live actions freeze that battle and return an explicit invalid flag;
terminal battles remain unchanged. Every executed move remains conservatively
counted as a placeholder while required mechanics are incomplete.

Confusion now deals typeless, noncritical self-damage before PP use; Struggle is
typeless and recoils for the rounded quarter of actual damage dealt, not max HP.
Any faint cancels the remaining ordinary action, as in Gen 3 singles. Flinch is
volatile bit 0, blocks before confusion and clears after an ordinary turn.

The explicit status/self-effect subset now applies Toxic, poison, paralysis,
burn and sleep moves; Rest; Recover-class floor-half healing; self boosts and
Haze. It handles existing-status/type/Substitute refusal, Gen 3 Electric-type
paralysis eligibility and Thunder Wave's Ground immunity, sleep counters and
boost clamping. Items/abilities can still alter these events and are not yet
integrated, so this is not a move-coverage qualification.

Substitute now costs the floored quarter of max HP, absorbs single-hit damage
without spill and preserves actual sub-damage for Struggle recoil. Protect and
Detect block moves according to the pinned catalog's protect flag; Endure limits
direct move damage to leave one HP, not residuals or recoil. Their shared Gen 3
stall chance halves to a minimum of one eighth and expires after an unrefreshed
ordinary turn. They fail as the last queued move. This remains a partial surface:
multi-hit, item/ability interactions, Baton Pass and the wider event system are
not qualified by these authored component fixtures.

This is an incomplete integration kernel. Items, abilities, weather, switch events,
the full residual scheduler and remaining special move effects are not
integrated. Its arithmetic/turn fixtures and one-turn real
team smoke do not qualify gameplay or supply the final throughput table. Pool
move coverage remains zero until move effects and their required mechanics are
implemented and verified. Do not use placeholder games for strength evidence.

```sh
POKEZERO_GPU_TEAM_POOL=/path/real-teams-200000.npz \
  python -m unittest discover -s tests -p test_gpu_engine_transition.py -v
```

## Early core step timing

```sh
python -m gpu_engine.benchmark --team-pool /path/real-teams-200000.npz \
  --out-prefix /path/outside/checkout/partial-core-step
```

This preliminary diagnostic requires exactly one GPU and the full real-team
pool. At batches 256, 1,024, 2,048 and 8,192 it repeats one fixed initial request,
with five warm-ups and 100 synchronized wall-time samples after compilation.
Host dispatch is included; setup and action sampling are excluded. The same
team prefix is reused across batch sizes. Repetitions do not advance games.
JSON retains samples, source/pool bindings and memory in use; Markdown reports
mean step latency. Encoder, forward, end-to-end, game length, game rate and
utilization remain unmeasured/null. Existing outputs are never overwritten.
These are incomplete-core timings, not the final table or a go/no-go result.

## Measurement contract

Use real Showdown-generated teams (at least 200,000), a frequency-led coverage
ledger and explicit placeholders. Target 85% move coverage before the final
throughput table. At batches 256, 1,024, 2,048 and 8,192 report synchronized engine,
placeholder encode, both-seat champion forward and end-to-end times separately;
games/second must use measured completed games/turns, never an assumed length.
Search-shaped expansion covers all 81 joint actions for 256, 1,024 and 4,096 nodes.

The predeclared go condition is at least 100 end-to-end games/second at batch 2,048,
engine step under 100 ms and fewer than 10% of 200 games diverging on covered
mechanics by turn 20. 30–100 games/second means reconsider; below 30 or no batch
scaling means no-go. These thresholds are not yet measured. Missing components
must remain missing in reports, not receive projected or placeholder timings.
No training, policy-quality or full-fidelity claim follows from this spike.
