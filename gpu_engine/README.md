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

Heal Bell cures living own-party major status and sleep bookkeeping, leaving
confusion/other volatiles and the opposing party unchanged. Gen 3 Soundproof
blocks each holder's cure, including the user; fainted party members are not
cured. This is selected component support, not qualified move/game coverage.

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

Synthesis/Moonlight/Morning Sun heal floor(max HP / 2) in clear weather,
floor(2 × max HP / 3) in sun and floor(max HP / 4) in rain/sand/hail, minimum one.
Living Air Lock/Cloud Nine suppress weather for healing; the move resolves before
weather expiry/upkeep. Full-HP attempts still spend caller PP. This is selected
component support, not qualified move/game coverage.

`observation.encode(state, table)` builds the handoff's placeholder observation
on device for both seats: float32 `[B,2,23,132]`, int32 `[B,2,23,41]`, token
types/attention masks and the table-backed legal-action mask. Sections are one
field, six own mons, six opposing mons, nine actions and one tendency token;
token type IDs are 0/1/2/3/5. Representative columns contain HP fractions, level,
stats, boosts, PP, statuses, species/move/item/ability IDs, weather and legal flags.
Unfilled columns remain zero. Switch actions use the current ordered bench.
This is NOT the production v4 column/category mapping or belief encoder: opposing
private state is visible and raw engine IDs are not champion vocabulary rows.
It is only an encode/forward cost-shape workload, never strength/training input.
No RNG or battle state is changed; no game-rate or encoder-parity claim follows.

Explosion/Selfdestruct apply Gen 3's floor-halved defense after stat modifiers.
Damp on either living active holder vetoes the move after PP; otherwise the
user self-KOs even on a miss, Protect, Ghost immunity or Substitute. Action-phase
replacement/residual ordering and Gen 3 last-party ties follow the existing
turn scheduler. These component checks do not qualify full move/game coverage.

Sleep Talk uses its own priority, PP and Choice lock, then calls one eligible
own move without another BeforeMove or PP/Pressure charge. Catalog charge and
nosleeptalk flags exclude slots; selecting a zero-PP slot fails without rerolling.
An existing Encore or Choice lock rejects the called dispatch after caller PP;
the first Choice Band use can still call before installing that lock.
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

## Champion forward cost

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

## Selected Rapid Spin integration

Rapid Spin uses Gen 3 power 20 and clears represented Leech Seed/own Spikes
after a successful nonimmune hit, including unbroken Substitute and Endure at
one HP. Miss/Protect/Ghost leave them intact; Mean Look/ability/Ingrain traps
remain, and there is no modern Speed boost. Partial-trap clocks/damage are not
represented, so this does not qualify all of Rapid Spin or move/game coverage.

## Selected Solar Beam integration

Solar Beam charges outside effective sun, hard-locking its release even at zero
PP. The release has no second PP/Pressure cost; BeforeMove interruption cancels
the charge. Effective rain/sand/hail halve its base power. Live weather suppression,
move-time weather and ordinary-turn charge expiry are represented; forced and
deferred replacement requests do not consume a charge turn. Baton copies a
remaining charge, including to a recipient that does not know Solar Beam. Other charging moves
remain placeholders. These component checks do not qualify complete-game coverage.

## Joint action expansion cost shape

`python -m gpu_engine.benchmark_expansion --team-pool POOL --expected-pool-sha
POOL_SHA --out RESULT.json` writes JSON/Markdown for 256, 1024 and 4096 parents.
Each device call replicates each parent into 81 lexicographic action pairs,
including invalid requests, which freeze and are counted separately from valid
child transitions. Siblings retain the same parent chance key (common random
numbers). Timings include parent replication and complete output synchronization;
five warmups and 100 fixed-root samples exclude setup/compile. This measures a
search-shaped engine workload, not an actual search or qualified gameplay.

## Advancing cost shape diagnostic

After the separate champion-forward dependency is available, run
`python -m gpu_engine.benchmark_games --team-pool POOL --weights WEIGHTS
--expected-pool-sha POOL_SHA --out RESULT.json`. This writes exclusive JSON and
Markdown outputs for batches 256, 1024, 2048 and 8192. Each finite cohort uses
distinct prefix team pairs, five warmups and 100 component samples, followed by
at most 1000 advancing requests with temperature-one masked champion actions.
No game is refilled or redrawn. Actual completions divided by elapsed wall time
give games/second; capped games remain censored. Ordinary game turns and live
requests are separate counters because forced/Baton requests are not turns.
This is an unqualified mechanics/placeholder-observation diagnostic, not a
strength measurement or a go/no-go. After-only counters are not mean utilization.

## Warm start refill cost shape diagnostic

`python -m gpu_engine.benchmark_refill --team-pool POOL --weights WEIGHTS
--expected-pool-sha POOL_SHA --out RESULT.json` measures actual completed games
over a 60-second window following 30 seconds of advancing warmup, at each of
256, 1024, 2048 and 8192 slots. Refill/reset, sampling, all three champion heads
for both seats, transition, synchronization and host accounting are timed.
Each replacement receives the next unused pair from a fixed 100000-game roster;
there is no wrap or redraw. Each batch restarts the same ordered prefix, so
these are not independent cohorts. A 2000-request cap censors a game rather
than counting it completed; in-flight games at the window end remain separate.
Warmup in-flight state and per-game RNG counters persist into the timed window.
Pool exhaustion or invalid/nonfinite requests stop the protocol. The JSON keeps
retired IDs and final active IDs for exactly-once accounting. Thirty seconds of
warmup does not prove stationarity; this is unqualified mechanics with the
non-production placeholder observation, not production self-play or a go/no-go.

Add `--device-telemetry` for one-second polling during each measured refill
window. The CUDA driver must expose exactly one GPU, matching JAX's sole CUDA
ordinal; its UUID, not a presumed physical index, binds every device query.
Identity failure stops before measurement. Every query has a 0.5-second timeout;
query failures, monotonic start/end times, sampling gaps and boundary-straddling
samples are retained. Only successful queries wholly within the measured window
contribute to the mean of reported utilization samples and maximum sampled
device-used memory (MiB). These are not continuous utilization means, exact peaks
or live JAX-array memory. Device counters include other users unless separate
allocation evidence establishes exclusivity; no benchmark-only attribution is
inferred. Polling overhead is included in wall time; bounded shutdown/join is
outside it. Warmup, workload, roster, caps and timing protocol are unchanged.
New hardware/source results stay separate from historical receipts. The option
does not qualify mechanics, fidelity, strength or production deployment.

## Selected Encore integration

Focus Punch prepares its original queued move before switches and Encore's
override. A direct non-status hit breaks focus even when Endure prevents HP
loss; a Substitute, miss, Protect or immunity does not. Gen3 failure is after
caller PP/last-move recording and still Choice-locks. Preparation blocks added
flinch, survives an interrupted Baton request and expires with the turn; it is
not Baton-copyable. These components do not qualify full move/game coverage.

Gen 3 Encore chooses 3..6 turns, adds a tick when the target has no queued move,
and ticks at speed-ordered residual order10/suborder14. It forces the last
PP-spending normal caller, retaining original queued priority and charging the
forced move's slot. Sleep Talk records its caller; Pursuit interception records
Pursuit. Protect blocks Encore, Substitute/Ghost do not; missing/excluded/zero-PP
last moves fail and an active Encore does not refresh. PP exhaustion ends it at
residual; ordinary switching, Baton noCopy and faint cleanup clear its fields.
The existing [0,4) duration column and unused residual tie column preserve the
original fixed chance block; primary sleep and Encore cannot share a dispatch.
Other move locks and full move/game qualification remain outside this component.

`last_action_kind` and `last_action_index` retain Rust's side-owned
NONE, SWITCH(party index), MOVE(exact dispatch slot) and STRUGGLE history.
Initial entry callbacks preserve NONE; committed voluntary/forced/Baton/phaze
switches record their target before hazards. Moves record after BeforeMove/PP,
before accuracy or target failure, including Sleep Talk's caller and Pursuit's
separate dispatch. Fainting preserves this side history while clearing the
per-Pokemon `last_move` identity. Encore uses the exact retained slot for its
mask, queued override and PP expiry, including duplicate identities. Native
correspondence refuses legacy states missing the explicit history instead of
guessing from active age; original-constructor provenance requires NONE/-1.
These are ports of the Rust lastmove/PP/Sleep Talk regression rules, not new
Fake Out support, widened suspension admission, coverage or strength evidence.

Wish retains a side-owned `wish_amount` alongside its counter, matching the
native wire. Only a successful new Wish stores the caster's floored half-max-HP;
pending recasts leave it unchanged. Expiry, switching, faints and terminal
truncation preserve this history. Gen3 healing still uses the current recipient's
half-max-HP, so a stored150 can heal50 on a101-HP recipient. Both correspondence
directions retain the exact tuple, including expired nonzero and zero-amount
states; missing legacy history refuses rather than being reconstructed.
Original-constructor capture requires zero stored amount. The native conditional
continuation guard is unchanged and still refuses any nonzero Wish tuple.

## Selected effect ledger

`python -m gpu_engine.coverage --team-pool POOL --expected-pool-sha POOL_SHA
--out LEDGER.json` reconciles frequencies against every nonempty tensor move
slot and reports source-assessed selected effects, partial effects and
placeholders. Rows name runtime dispatch, authored check references and missing
intrinsic effects. Brick Break's screen removal, Rapid Spin's partial-trap
cleanup and Sing's Soundproof admission remain partial, with no whole-move
credit. Hidden Power aliases contribute their actual counts once each.
This is the declared singles measurement subset, not every ability, clause or
volatile cross-product. Implementation assessment is not qualified gameplay;
game-qualified coverage remains unset pending the separate divergence check.
The accepted pool and its historical implementation flags are never rewritten.

## Selected stat-event corrections

Soul Dew chains 1.5 into SpA and SpD for original Latias/Latios, after critical
stage cancellation and before damage arithmetic. Both chains apply to special
Pursuit interception on the outgoing target. Huge Power/Pure Power double
physical Atk, including confusion and its item chain; they do not bypass burn
or modify special damage. Physical confusion defense does not use Soul Dew.
These authored native-event fixtures do not qualify games. Transform/base-species
replacement, item transfer/suppression and other abilities remain outside scope.

## Fixed first-divergence diagnostic

`python -m gpu_engine.fidelity --team-pool POOL --expected-pool-sha POOL_SHA
--showdown-root PINNED_SHOWDOWN --node PINNED_NODE --out TRACE.json` binds clean
source and reconstructs all stored fields plus original packed sets for exactly
200 unfiltered pairs, game IDs 50200..50399. The earlier 50000..50199 panel is
retained unchanged after its authored faint-marker adapter correction; the
corrected panel is disjoint, not a replay or a pooled inference.
The stock local bridge is imported
through a diagnostic-only launcher; no simulator state is repaired or materialized.
Both engines receive the complete reference-legal action law, normalized to
original party and canonical move identities, only after independent requested
seat and full legal-set agreement. Known semantic raw chance roles are shared;
unknown/repeated native roles stop that original trajectory as unresolved.
Initialization randomness is audited separately: gender is inert only without
Attract/Cute Charm, and the unconditional Quick Claw roll only without a holder.
Other initial randomness remains unresolved, not a pass.
HP/faint agreement is independent of major status: Showdown's `fnt` marker is
not compared to a living major-status code. Historical traces are never repaired
or resumed after an adapter correction.

Each trajectory ends at its first mismatch, unresolved role, matched terminal,
completed twentieth ordinary turn or 200-request cap. Terminal/winner agreement
is required; unfinished forced/Baton/residual work cannot pass the turn cap.
Every original pair remains in the fixed denominator. Infrastructure errors
abort the panel without replacement. Exclusive JSONL records the entry before
play and each result immediately; exit zero means 200 classified trajectories,
not 200 completed games or a fidelity pass. Mechanic classification is a separate
read-only analysis; the runner never awards a go decision.

## Canonical historical report (no measurement)

`python -m gpu_engine.benchmark --report-games ADVANCING.json
--report-refill REFILL.json --report-expansion EXPANSION.json --report-cpu CPU.json
--report-fidelity FIDELITY.json --report-ledger LEDGER.json --out-prefix REPORT`
writes exclusive `REPORT.json` and `REPORT.md` tables from six existing receipts.
This reporting mode never calls a device or replays a game. It binds exact input
bytes and keeps source versions/workloads separate. Advancing component times
are fixed-root samples, not additive to advancing elapsed time or transferable
to a different refill source. Live requests, ordinary turns and nominal batch
slot capacity are distinct; games/s uses actual completions and wall time.
Unknown fidelity cases and censored games are never dropped to qualify a gate.
After-only GPU counters do not become average utilization/per-batch memory;
missing values remain unavailable. The historical CPU receipt has no embedded
source commit, so the report does not invent one from a filename.

The report recommends no production advancement: selected-effect source shares
are not qualified gameplay or proof of >=85% correct effects. It does not claim
the preset <30 games/s failure or <10% covered-divergence success. A negative or
unqualified spike readout does not authorize a full Gen3/production rewrite.
The existing `--team-pool` mode remains the explicitly partial core diagnostic.

## Fixed exact-outcome exploration panel

`python -m gpu_engine.panel_owner --entry ENTRY.json --expected-entry-sha SHA
--team-pool POOL.npz --out NEW_DIRECTORY` runs one prospectively bound attempt.
The entry pins the complete GPU source inventory, Git commit, pool, compiled
native profile and Python wrappers/dependencies, fixed roster, probability
tolerances, case cap and calendar bound. Exact-roll opt-in must precede import.
All source roots and actions close into one hashed artifact before comparison.

The comparator retains complete represented successor supports, rational JAX
masses and unnormalized native probabilities. Unsupported positive successors
invalidate the whole fixed slot; no prefix, redraw or renormalization passes.
The owner interrupts operations at the calendar bound, drains the owned process
group, records the actual exit and reconstructs terminal counts from valid
complete files. Corrupt/missing files remain unknown, even after worker exit0.
Requested move dispatch counts are exposure only, not executed-effect credit.
This JAX-sourced exploration cannot establish independent simulator truth,
85% correct-effect coverage or a production go decision by itself.
The compiled native/JAX seam profile is explicitly invoked as
`python -m unittest tests.test_native_gpu_panel_execution`; admission requires
all three actual controls and zero skips. JAX-only component CI does not build
the separate native oracle and is not this integration profile.

## Conditional continuation and original constructors

`conditional_law.plan_for_suspension` is an explicit opt-in adapter for the
bounded native continuation domain. It binds the full paused state, saved
chance capsule, legal replacement action, supplied move table and source pins.
It evaluates all four fresh entry-tie cases in the native oracle before JAX,
retains each case's original mass and refuses the whole slot if any positive
successor is unsupported. Cached native evaluation is actual execution, not
validation-only admission. A deterministic JAX table defect remains a divergence.

`source_provenance` retains original constructors separately from correspondence.
Load a byte-bound immutable pool with its Nature/EVs and optional happiness and
gender overrides. Build a prospective context from the source-bound Gen3 weight
catalog, both native extensions' runtime attestation and, for generated genders,
the complete source-bound gender policy. Capture after gender assignment and
before entry callbacks. Generated values are checked against a separate replay
of the bound policy/seed/domain/root; the live state and its keys are untouched.
Unresolved weights or constructor fields refuse their root rather than redrawing.

The capture is provenance only: it never clears `entry_pending` or issues a
native certificate. After entry, `native_input_with_provenance` checks constructor
identity and proves the complete physical mapping using the retained base types,
base ability, Nature, EVs, gender and native float weight. Current copied Trace
ability remains distinct from original base Trace; it does not widen the native
conditional domain. Original pool metadata and enriched mapping metadata retain
separate identities. The context expects independently verified prospective
source admission; a catalog's self-reported provenance is not independent proof.

These primitive APIs retain no action-selection or scientific launch authority.
Their authored tests run in the isolated `conditioned-oracle` CI profile with
both native bindings and zero skips. Historical panels remain unchanged; fresh
collector admission, broader suspension mechanics and full engine qualification
remain separate work.

New panel entries can explicitly opt into `constructor_provenance` with schema
`pokezero.prospective-constructors.v1`, the complete catalog, both-binding runtime
attestation, a bounded `root_start` and the simulator checkout `showdown_root`.
Before source generation the worker regenerates the weight catalog and optional
gender policy against that actual checkout/compiled tree and exact commit. Both
artifacts must equal the prospectively bound entry. It builds JAX tables from
the original pool metadata; correspondence uses the separately enriched metadata.

The collector captures originals before entry, retains the full post-entry state
before prefixes and closes every final root/action/source error plus provenance
and context identity in its hashed input artifact before native mapping. Missing
or unsupported originals remain whole-slot UNKNOWN without legacy fallback.
Unresolved generated genders keep their original slots; other root occurrences
and their draw addresses stay unchanged. For ordinary comparisons, the planner
receives a full current-root NativeInput with original statics. Suspended roots
first prove immutable constructor lineage, then use the original post-entry
certificate and cached conditional laws. All four native case witnesses remain
in the slot evidence. Owner reconstruction requires the referenced context file
and its digest for opt-in entries. Exclusive attempts, deadlines and actual exit
and cleanup checks remain in force.

This opt-in implements collection plumbing; it supplies no launch admission or
historical coverage credit. A new reviewed frozen entry and the standing saved-
root cost check are still required before any future scientific attempt. Broader
Trace effects, Sleep Talk and suspension contexts remain unfinished.
The native continuation validator admits current None/Static/Pressure/Levitate
abilities and None/Leftovers items on the entrant and remaining active. Pressure
adds deterministic PP cost and Levitate changes grounding in the admitted queued
move context. Retained original abilities are inert until departure, so their
values remain bound and compared without a current-consumer whitelist. Untouched
bench fields and outgoing-carrier abilities/items are retained; departure runs
Natural Cure and base-ability restoration explicitly. General status, side,
weather and saved-chance guards remain. The old all-party rule excluded every
team in the preserved pool; role separation removes that blanket restriction
without claiming measured real-pool acceptance or broader consumer support.
Authored full-law checks preserve complete party/carry state. The JAX state
retains immutable `base_ability` separately from current `ability`, initialized
from the selected pool before entry and verified against original constructor
evidence. Departure evaluates current Natural Cure before restoring the original
ability, including phazing and suspended replacements. Both JAX and native
correspondence reject changed originals; legacy state without that field refuses
instead of inferring it from a copied ability. The former isolated Trace
departure mismatch now agrees in both seats. Entry now copies a living foe’s
current ability after hazards without activating the copied Gen3 Start event.
Selected passive major-status cures and Own Tempo cleanup run on the gained
ability; Truant uses the pre-copy current ability and startup context. A failed
Trace does not retry on Update. The ordinary chance proof closes possible copied
abilities before declaring Static/contact chance inactive. Complete authored
ordinary laws cover both seats; this does not widen the native conditional
entrant/remaining-active whitelist. Broader ability events, Forecast/Color Change
representation, infatuation and Stick remain unfinished. These are implementation
controls, not historical coverage or fresh scientific credit.

### Explicit saved-root cost attempts

The collector and owner accept `--cost-admission` only with a separately frozen
cost entry. Ordinary scientific panels retain their minimum of1,000 slots and
reject a cost entry on the ordinary path. Cost preparation generates exactly the
registered roster of at least20 roots through the same source/constructor logic;
it does not generate a larger roster and choose a subset. All complete originals,
post-entry records, actions and source hashes close before the first comparison.
No source generation or attempt is authorized merely by these APIs existing.

A cost entry references an explicit `operation_budget` by its canonical SHA256.
The future panel must bind that identical operation budget and intended comparator
runtime/configuration. A cohort calendar cap is not a per-slot decision limit.
Timing begins before root transfers and binding, includes cached conditional
native evaluation, synchronized JAX work, canonicalization/compression and full
result/progress file and directory fsync. Source generation/setup precedes the
clock; no comparison warmup is performed by this mode. Timing-witness publication
is excluded and its overhead retained separately, excluding publication of the
overhead receipt itself to avoid self-reference.

An owned nonblocking pipe announces each attempt-bound START before in-scope
filesystem I/O. Immutable witnesses publish complete fsynced bytes atomically,
without overwriting prior evidence. The owner uses absolute deadlines and records
whether a stop concerns an operation, missing DONE/publication evidence or the
separate completion-liveness phase. That excluded phase is bounded by one further
operation-limit interval after DONE.end; it supplies no operation-latency inference.
Actual exits, reaping and process-group absence are retained. A proven overrun
stops the worker before another slot, and terminal reconstruction also rejects
late overruns that escaped live polling, even when the child exited0.

Numeric nearest-rank p95 bounds, at least20 timed roots, exercised comparison
paths/case counts and attempt integrity are separate facts. Missing DONE has
unknown bounds rather than an assumed60-second lower bound. A slow twentieth
observation does not turn rank19/20 into a maximum-latency gate. Fast refusals
remain refusal-path timings; unexercised comparison paths remain unmeasured.
Every fixed slot is retained without refill, normalization or an all-AGREE rule.
The accounting reports no automatic comparison-cost, fidelity or strength
qualification. Independently reviewed frozen source ranges, runtime, resources
and entries are still required before any scientific collection.

Rest-origin state is produced only by a successful Rest and ends on wake/cure.
It binds remaining1–3 and Sleep Talk refund0–3 to the corresponding native
Rest fields at ordinary boundaries; the refund follows the sleeping party
member and is spent only on re-entry. Natural sleep still refuses because
sampled remaining duration is not the native elapsed-hazard convention.
Native91 Lum's stale cured Rest fields remain a named refusal. This state
correspondence does not qualify Sleep Clause, all sleep consumers, or full-game
fidelity; the conditional native execution guards are unchanged.
Native91 also retains an overbroad Sleep Talk callee pool (charge moves are
not excluded). Its positive charge branch is preserved as a separate mismatch
control; the exact Rest carry cycles isolate the shared Rest/Sleep Talk/Splash
callee domain and do not qualify all called-move sets.

The ordinary comparator memoizes canonical successors within one root, using
every physical state field, dtype, shape and exact bytes plus the bound mapping
context. It keeps all chance cases and their original masses, including late
whole-slot refusals. At most1,024 distinct physical successors are retained;
eviction recomputes the mapping. Each engine records its first actual raw
representative once per canonical successor. These changes reduce repeated
serialization without narrowing the chance partition or supplying new cost
qualification.

Native92 charges accepted Protect/Detect/Endure attempts before their stall
success draw while retaining free BeforeMove failures. Sleep Talk pays only
the caller's PP and records the caller in move history. Protect and Endure
fail when no opposing queued move remains, including after a switch. Both-seat
full-law controls bind PP, history, saved Rest state and stall counters; this
does not change JAX's already matching stall behavior.
