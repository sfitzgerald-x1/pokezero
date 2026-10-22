# GPU engine throughput spike

This isolated experiment asks whether a sampled-chance Gen 3 transition,
placeholder observation and the champion forward can sustain 100 completed
games per second on one GPU. It is not a production engine or training teacher.
Nothing here is imported by the existing battle collector or search runtime.

## Current implementation

The spike has batched structure-of-arrays storage, per-battle PRNG streams, the
nine-action layout, an offline real-team generator/NPZ packer and partial
turn integration. No complete transition, encoder, policy forward, fidelity result
or games-per-second benchmark is implemented yet. Empty states
are storage fixtures and are terminal by default; they are not benchmark games.

Selected variable moves resolve before stat/type-dependent modifiers: Hidden
Power uses the Pokemon's accepted-pool type/power (not the metadata alias), and
Return/Frustration use happiness. The pinned generator emits team objects without
happiness, so Showdown defaults to 255: Return 102, Frustration 1. This is not
the text-team import path which may set Frustration happiness to zero. All Hidden
Power aliases are excluded from original-Fire thaw; resolved Fire still triggers
non-frozen Flash Fire. This repairs the accepted alias metadata's thaw edge
without changing the accepted pool or historical measurements. Wider variable
moves, Transform and complete gameplay qualification remain pending.

State has six party members per side; all fields are JAX arrays with a leading
batch dimension. Types, status, items, abilities and moves require the eventual
team pool's vocabulary. Bench actions enumerate party order excluding the
active slot. Move0 represents synthetic Struggle only when no move is usable.
The structural mask handles PP, Choice Band locking, linked move traps, externally resolved disabled/trapped flags,
forced replacement and fainted slots. Table-backed callers must use
`transition.action_mask(state, table)`, which additionally resolves current active
ability traps and is also used by the step validator and benchmark sampler.
Taunt, Encore, recharge and other
restrictions still require the transition and are not claimed as supported.
For unilateral forced replacement, the initial transition suppresses the
other seat's action. The mask alone is not a complete request-phase protocol.

Direct Baton Pass now suspends the original turn for a living forced switch,
then resumes the already selected opposing move with its original chance block.
It does not solicit a second action, re-sort two moves, or trigger Pursuit's
voluntary-switch interception. Two queued Baton actions suspend separately.
Supported transfer is seven boosts, Substitute HP, confusion counter, the
singles Leech Seed flag and direct move-trap links; recipient HP/status/item/ability remain its own, and
Flash Fire/Choice lock do not pass. This is not general volatile-copy support.
Entry hazards still hit through Substitute: an entry KO cancels the saved move,
then the existing replacement chain finishes before residuals. All moves remain
placeholders until the wider effects and coverage checks qualify them.

Direct Roar/Whirlwind select a living bench uniformly, cancel the displaced
actor's queued action and run entry before residuals. Protect and accuracy still
apply; Substitute does not block them. Suction Cups blocks both, Soundproof
blocks Roar only, and a represented Ingrain root blocks dragging/voluntary
switching (the Ingrain move itself remains a placeholder). Natural Cure and
outgoing volatile cleanup share the ordinary switch path. Hazard KO suspends
the original turn until a healthy replacement; incoming Lum/weather/Intimidate
resolve at entry. Broader reflection, partial traps and event effects remain pending.

Mean Look, Spider Web and Block now record the opposing owner's stable party
index rather than setting an anonymous trapped flag. Protect/Substitute block
them; Gen 3 Ghosts are not exempt. Baton transfers either linked end, including
mutual traps; an ordinary switch, drag or faint severs the links. Re-entering
with the same former owner cannot reactivate a stale trap. Ability traps remain
separate and are recomputed from current active abilities. Partial-damage traps
such as Wrap/Bind and broader volatile interactions remain placeholders.

Bench choice extends the fixed ordinary chance block with one uniform [0,60)
roll per seat from a dedicated fold-in. Since every possible living bench count
1..5 divides 60, rank selection is unbiased. The existing twelve chance columns
are unchanged, and the extra rolls are saved across a Baton pause rather than
redrawn from a later key. This is sampled chance, not engine/Showdown RNG parity.

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

The existing fixed critical roll now resolves ordinary 1/16 and catalog +1
high-crit 1/8 chances. Battle Armor and Shell Armor veto critical hits in both
ordinary damage and pre-switch Pursuit, using the actual outgoing holder.
Focus Energy, critical-boosting items and additional critical stages are outside
this subset; the accepted pool and PRNG columns are unchanged.

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

`champion.py` is an inference-only functional JAX port (no Flax dependency) of
the champion's one-frame, norm-first TransformerEncoder: exact GELU, key padding,
masked mean, action tokens13..21, policy/value/opponent heads, float32 eval mode.
The category embedding has916 rows TOTAL, including the16 OOV buckets, not932.
Export the unchanged trusted checkpoint once with optional Torch2.10 on CPU:
`python -m gpu_engine.champion --checkpoint /path/model.pt --out /outside/git/weights.npz`.
`PYTHONPATH=src python -m gpu_engine.check_champion_forward --checkpoint /path/model.pt
--archive /outside/git/weights.npz` checks the actual production Torch forward
on authored inputs. Load NPZ with `champion.load`; forward and per-seat masked
temperature1 sampling stay on device. No-legal rows return−1, never a valid action.
This is a cost workload, not production observation semantics, training or strength.

Explosion/Selfdestruct apply Gen 3's floor-halved defense after stat modifiers.
Damp on either living active holder vetoes the move after PP; otherwise the
user self-KOs even on a miss, Protect, Ghost immunity or Substitute. Action-phase
replacement/residual ordering and Gen 3 last-party ties follow the existing
turn scheduler. These component checks do not qualify full move/game coverage.

Sleep Talk uses its own priority, PP and Choice lock, then calls one eligible
own move without another BeforeMove or PP/Pressure charge. Catalog charge and
nosleeptalk flags exclude slots; selecting a zero-PP slot fails without rerolling.
Called Protect, Pursuit, phazing and Baton reuse the original queue semantics.
Gen3 sleep decrements, Early Bird, consecutive Sleep Talk skipped turns and
switch-in restoration are represented, including restoration after a hazard KO.
Natural Cure, waking and Lum clear that bookkeeping; it does not pass via Baton.
The dedicated [0,60) selection roll persists through pauses. Other move callers,
Snore's move effects and general called-move qualification remain pending.

Selected target secondaries cover Rock Slide flinch; Ice Beam/Punch freeze;
Thunderbolt/Thunder/Punch and Body Slam paralysis; Sludge Bomb poison; selected
Fire burns; and Gen 3 Crunch/Shadow Ball/Psychic Special Defense drops. Shield
Dust, Serene Grace, Inner Focus, Clear Body/White Smoke and selected status
immunities are resolved, including sun's freeze immunity and weather suppression.
Breaking a Substitute still blocks its hit's secondary, and Lum resolves before
the next action. One dedicated fixed secondary roll per seat is saved through
Baton pauses; the original twelve chance columns are unchanged. Other target,
self, multi-effect and contact-ability secondaries remain placeholders.

`transition.py` initializes real teams from the pool and connects fixed-shape
action ordering, switches, PP, status action gates, static single-hit damage,
major-status residuals, forced-replacement pseudo-turns and party exhaustion.
Invalid live actions freeze that battle and return an explicit invalid flag;
terminal battles remain unchanged. Every executed move remains conservatively
counted as a placeholder while required mechanics are incomplete.

Confusion now deals typeless, noncritical self-damage before PP use; Struggle is
typeless and recoils for the floored quarter of actual damage dealt, not max HP.
Any faint cancels the remaining ordinary action, as in Gen 3 singles. Flinch is
volatile bit 0, blocks before confusion and clears after an ordinary turn.

The explicit status/self-effect subset now applies Toxic, poison, paralysis,
burn and sleep moves; Rest; Recover-class floor-half healing; self boosts and
Haze. It handles existing-status/type/Substitute refusal, Gen 3 Electric-type
paralysis eligibility and Thunder Wave's Ground immunity, sleep counters and
boost clamping. Remaining items/abilities can still alter these events,
so this is not a move-coverage qualification.

Substitute now costs the floored quarter of max HP, absorbs single-hit damage
without spill and preserves actual sub-damage for Struggle recoil. Protect and
Detect block moves according to the pinned catalog's protect flag; Endure limits
direct move damage to leave one HP, not residuals or recoil. Their shared Gen 3
stall chance halves to a minimum of one eighth and expires after an unrefreshed
ordinary turn. They fail as the last queued move. This remains a partial surface:
multi-hit, broader item/ability interactions and the wider event system are
not qualified by these authored component fixtures.

The held-item subset applies Choice Band, species-gated Thick Club and Gen 3
Light Ball stat multipliers (Light Ball boosts only Pikachu's special attack).
Choice Band locks an attempted move until switching; an exhausted locked move
permits Struggle, not a different move. Lum Berry cures major status/confusion
and is consumed, including newly applied status before the slower move. Selected
residual events sort speed before suborder: on one mon (or equal speeds),
Leftovers precedes status, but faster poison can end the game before slower
Leftovers. Party exhaustion ends processing. Other items, item transfer
or suppression, and ability interactions still require integration.

Selected weather moves, suppression, damage/accuracy modifiers and chip are
integrated. The absorption subset applies Water Absorb, Volt Absorb and Flash
Fire after accuracy/Protect and before Substitute. Volt Absorb does not block
Thunder Wave in Gen 3. Frozen holders cannot activate Flash Fire; Will-O-Wisp
does not activate it when existing status, a Fire type or Substitute would
already make the status fail. Flash Fire's Fire-damage modifier persists until
switching. Original-catalog damaging Fire moves thaw a struck frozen target;
Hidden Power Fire and dynamically Fire Weather Ball do not. These are selected
events, not general ability coverage.

Pressure charges extra PP on actual eligible uses before accuracy, Protect or
Substitute checks, using the current living foe after switching. The pinned
target catalog—not damage category—distinguishes self/ally moves from attacks,
field moves and Spikes. Non-Ghost Curse remains self-targeted; Ghost Curse
targets the foe. Inability to act skips PP use, and exhausted moves use synthetic
Struggle without another charge. Called and multi-turn moves remain placeholders.

Speed Boost uses ordinary-turn age since entry: initial leads qualify on the
first ordinary turn, ordinary switch-ins skip that turn's bonus, and forced
replacement pseudo-turns do not age either side or run residuals. Selected
residual handlers retain their snapshotted speed order after a boost; the
ability does not re-sort already queued Leftovers or status events. This is
authored component coverage, not a full-game fidelity result.

Leech Seed applies after accuracy/Protect and fails on Grass or Substitute. It
drains into the current living occupant of the opposite singles slot, including
at full HP; switching the seeded holder clears it. Wish belongs to the side
slot, cannot refresh while pending, and expires using the recipient's max HP
before weather. Forced-replacement pseudo-turns do not tick either effect.
Seed damage and healing finish together before checking for a win. Direct Baton
passes the seed flag; Liquid Ooze and wider event interactions remain pending.

Pursuit intercepts an ordinary opponent switch against the outgoing state at
80 power with unconditional accuracy. In Gen 3 it is Special/Dark. A sleeping
or frozen actor at beforeTurn registration does not intercept; an interception
bypasses BeforeMove paralysis, confusion, flinch and counter changes. It spends
PP once, including the outgoing holder's Pressure, and retains capped Substitute
damage and successful-use Choice lock. Even after a Pursuit KO, Gen 3 installs
the already chosen replacement, with its hazards and entry ability. Forced
replacement pseudo-turns and direct Baton Pass are not intercepted. Broader
contact/item callbacks remain pending; ordinary non-switch Pursuit uses the usual path.

Action-phase Gen 3 KOs suspend the ordinary turn before residuals. A replacement
request installs the next living member before Wish, weather and order10 effects;
another hazard KO requires another replacement first. The saved ordinary chance
block, turn count and active ages are not redrawn/ticked during recovery. A
separate counted stream supplies replacement-entry tie coins. Once residuals
finish, their own faint replacements are standalone pseudo-turns with no second
residual. `turn` counts ordinary turns begun, including a suspended one. This is
the KO-resumption foundation used by direct Baton's queued-move continuation.

Guts now composes with the selected item multipliers in one Attack event,
including confusion self-damage. It overrides the physical burn penalty without
curing burn or preventing residual damage. Gen 3 Thick Fat modifies Fire/Ice
base power, not Attack or final damage. Arena Trap, Magnet Pull and Shadow Tag
restrict voluntary switches through the current request mask; forced switches
remain legal. Bench/fainted trappers do not bind, and Gen 3 does not give Ghosts
or other Shadow Tag holders their later-generation exemptions. Direct Baton
escapes these ability traps; its direct move-trap links transfer as described above.

Seismic Toss/Night Shade use level damage and Sonic Boom/Dragon Rage their fixed
constants, retaining immunity, accuracy, Protect, Substitute and Endure while
bypassing ordinary damage modifiers. Selected recoil moves use floored actual
damage (including capped Substitute damage), not later-generation rounding.
Rock Head prevents their recoil except Struggle. Nondivisible source fixtures
correct the earlier rounded-Struggle implementation and regression expectations.

This is an incomplete integration kernel. Remaining items, ability interactions, switch events,
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
