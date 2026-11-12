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
resolve at entry. Broader reflection and event effects remain pending (partial
traps landed with fidelity M2).

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

Fidelity patch A1 (status, boost and cure effects; no new state or chance):
- Curse (non-Ghost, the only pool form): +1 Atk, +1 Def, -1 Spe on the user,
  silent at the caps. A Ghost-type user's Curse stays an unported placeholder.
- Superpower: -1 Atk, -1 Def on the user after a successful hit, Substitute
  hits and KOs included; not on a miss, Protect or type immunity. A White Herb
  holder (gen3 randbats gives every Deoxys and Deoxys-Attack one) restores the
  drop once at the move's AfterMove: fidelity WH, below.
- Charm: target Attack -2, blocked by Protect and Substitute, never by type
  immunity; Clear Body, White Smoke and Hyper Cutter block it (revealed only
  when the capped drop was nonzero).
- Lovely Kiss: primary sleep at 75% accuracy through the status table.
- Sleep Clause Mod (the comparator's `gen3customgame@@@sleepclausemod`): a
  foe's sleep move fails while another living member of the target's side
  sleeps from a foe; Rest's sleep (rest_origin) does not count.
- Insomnia and Vital Spirit refuse sleep moves (revealed) and fail Rest.
- Aromatherapy: Heal Bell's party cure without the Soundproof exemption.
- Refresh: cures the user's paralysis, burn or poison, not sleep or freeze.
- A fainted mon's Substitute is cleared with its other volatiles, so the
  forced-replacement request no longer sees it.

Fidelity patch A2 (damage-changing effects and Destiny Bond; new state
`item_knocked` and volatile bit `DESTINY_BOND`=64, no new chance):
- Knock Off: after the secondaries (Gen 3 AfterHit), a hit on the Pokemon (not
  a Substitute, KOs included) removes its item for good (`item_knocked`), so
  Leftovers, berries and Choice Band stop working; Sticky Hold blocks it
  (revealed). A Choice Band lock without the Band ends at endTurn (no TURN
  line yet while a replacement is pending). Device: the removed item is shown
  and marked removed, B2's removal credit counts per victim side, and the
  mon's item narrowing is rebuilt without the item (B2 drops a mutated item).
- Giga Drain: heals max(1, floor(dealt/2)), or ceil(sub damage/2) on a
  Substitute; silent at full HP. Liquid Ooze turns it into that much damage to
  the drainer, also at full HP, and can KO it (revealed).
- Counter and Mirror Coat: priority -5, twice the last foe hit on the Pokemon
  this turn (Counter: physical or any Hidden Power; Mirror Coat: special, not
  Hidden Power); Substitute hits, status moves and a called (Sleep Talk) or
  Encore-swapped Counter fail silently after PP; Ghost/Dark type immunity.
- Destiny Bond: set when it executes (reuse succeeds), cleared at the holder's
  next move attempt (a cant included), on switch-out and at a faint, never
  Baton-passed. A foe's move (Pursuit on the switching holder included) that
  KOs the holder faints the attacker; residual, recoil and Liquid Ooze KOs do
  not. Device: the destinybond bag bit follows B2 (expired on the holder's
  move line, Pursuit's included).

Fidelity patch R (the appended effect chance stream and Bonemerang; new state
`effect_draws`):
- `ChanceInputs.effect` ([B,2,4], bounds 16,16,4,2) is an appended stream:
  `fold_in(split key, 1732050)`, drawn beside the original fields without
  changing any of them or the key advance. A caller that omits it gets exactly
  the sampled value. A suspended turn saves it in `effect_draws` and its
  resumed step replays it, like the other saved blocks. The hand-written CUDA
  chance kernel does not draw it yet (deferred to CUDA option A), so
  `v2e_diff chance` and `v2e_cuda_check p2` report `effect` as differing.
- The exact-chance enumerators (`phase_d_chance`, `chance_consumers`,
  `ordinary_law`, `conditional_law`) have no `effect` axis: `finite_law`
  carries only the twelve draws and the auxiliary streams, and the consumer
  inventories list the effect coordinates without one. They refuse today on
  the stale `transition.py` pin; whoever re-pins it must add the coordinates.
- A root built without the leaf, or a capture that predates it, resumes a
  suspended turn with `effect_draws` 0, which is a second-hit critical for a
  queued Bonemerang. Root materialization should sample or carry the block.
  The conditioned native capsule (`conditioned_future.CAPSULE_FIELDS`, the
  Rust wire schema) does not carry it either; it joins the suspended-future
  keys instead.
- Bonemerang hits twice after one accuracy roll: the first hit uses
  `draws[s][4..5]`, the second `effect[s][0]` (critical) and `effect[s][1]`
  (damage roll). A first-hit KO ends the move. Each hit meets the Substitute
  on its own (a first hit that breaks it lets the second reach the Pokemon)
  and Endure on its own. Color Change runs after each hit, so a target
  retyped by the first hit meets the second as the move's type alone.
  Counter and Mirror Coat return twice the last hit that reached the
  Pokemon. Device: one damage line per hit.

Fidelity patch B (chance secondaries, and confusion switched on; no new state
or chance). Each move has one secondary entry, so the side's existing
`secondary_draws` roll is Showdown's single random(100) for it; Serene Grace
doubles the chance (Dunsparce's Headbutt: 60%):
- Iron Tail: 30% target Def -1. A Substitute hit, Shield Dust or a fainted
  target blocks it; Clear Body and White Smoke delete it with no line (the
  effect carries secondaries), so nothing is revealed; -6 is a no-op.
- Steel Wing (10% user Def +1) and Meteor Mash (20% user Atk +1): self
  entries, so they survive Shield Dust, the target's Substitute and a KO; +6
  is a no-op.
- Headbutt and Needle Arm: 30% flinch through the existing flinch path
  (Inner Focus, Focus Punch's focus, Shield Dust, Substitute); a flinch only
  stops a target that has not moved yet.
- Signal Beam: 10% confusion for 2+d moves, d the attacker's `draws[s][7]`
  (Signal Beam never also uses it for sleep or Encore). Own Tempo, Shield
  Dust, a Substitute hit, a fainted or an already confused target (no
  restart) block it. This switches on the existing confusion code: each move
  attempt decrements it first (not while asleep without Sleep Talk, frozen,
  loafing or flinched), 0 snaps out and the move proceeds, otherwise
  `draws[s][3] >= 50` is a 40 BP typeless physical self-hit with the
  confused mon's own stats, stages, Guts/Hustle/Huge Power, Choice Band and
  burn, its own `draws[s][5]` roll and no crit, through its Substitute; a
  self-KO cancels the foe's queued move. A Lum Berry eats it at once, Own
  Tempo cures a Baton-passed one on entry, and Baton Pass carries the clock.
- A faint's clearVolatile now also ends the fainted active's confusion and
  Choice lock at the step end (before, only its departure did), so a
  forced-replacement request sees neither, as in Showdown.
- Device: B2's `-end|X|confusion` (bag entry off, CONFUSION_TURNS reset) at
  the before-move snap-out, a Lum cure and an Own Tempo cure; the self-hit's
  `-damage ... [from] confusion` line feeds B2's absent-Leftovers evidence as
  action-phase HP.

Fidelity patch M1 (Hyper Beam's recharge and Trick; new state `recharge_counter`,
no new chance):
- Hyper Beam (move flag `recharge`): a hit sets `recharge_counter` 2 (Showdown's
  `mustrecharge` duration), a Substitute hit and a KO included, never a miss,
  Protect or an immunity. The next request is the single Recharge row (lane 0
  only, switching trapped); its action is the pseudo-move `recharge`, priority 0
  at the user's Speed (move id 0: no Focus Punch preparation, no Pursuit). At
  BeforeMove priority 11 it is `cant|X|recharge` ahead of sleep, freeze, flinch,
  confusion and Truant: no sleep or confusion tick, no Truant line, no PP or
  history, Destiny Bond ends. The counter falls by one at each completed
  residual, last (it has no handler), so a recharge cancelled by a Gen 3 faint
  expires there silently. A switch-out (a drag included) and a faint clear it.
  Slaking, the pool's only user, has Truant: the recharge turn is its loaf turn,
  and the loaf clock still flips at that residual.
- Trick: accuracy 100, Protect and a Substitute block it, a Status move ignores
  type immunity (Dark is hit), Sticky Hold makes it `-immune` (no ability line).
  It fails when either mon's item was knocked off (gen <= 4 takeItem) or neither
  holds one; otherwise the actives' items swap, an identical pair included. The
  Choice Band lock follows the current item: the Tricker no longer locks, a
  slower receiver locks into its move this turn, a faster one on its next move,
  an Encored one into its Encore move; a stale lock ends at endTurn.
- Device: the `mustrecharge` bag bit is set when the counter starts and cleared
  by the recharge turn (`recharge_turn`), an arrival or a faint, never at a silent
  expiry: B2's stale bit, matched on purpose (design 13 item 3). The recharge
  request collapses like the charge case (`move:recharge`, no mechanics, PP 1).
  Trick: B2's per-mon item ledger (`item_mutated`, `item_original`, Knock Off's
  removal included) and per-stint seat flags (`item_swapped` column 41,
  `choice_public`, `choice_locked` column 40 set on the holder's next own move
  line); an identical swap still counts (B2 folds its `-item` lines); a mutated
  item never narrows the belief again (design 13 item 4). The absent-Leftovers
  rule and the world sampler read the per-mon `item_mutated`.

Fidelity patch M2 (Perish Song, Yawn, Wrap and the residual scheduler; new state
`perish_counter`, `yawn_counter`, `partial_trap_counter`, `partial_trap_source`):
- Residual scheduler: the order-10 sorted list is 14 events (partiallytrapped at
  subOrder 9 and Yawn at 19 join Speed Boost 3, Leftovers 4, Leech Seed 5, status
  6 and Encore 14), keyed by Speed then subOrder with the multiplier 64; the old
  events keep their relative order. An equal-Speed Wrap pair takes `effect[0][3]`,
  the Yawn pair a fixed (inert) order. Perish Song is a two-event order-12 pass
  after the list (`effect[1][3]` on a tie). A faint clears all four leaves.
- Perish Song: every active without the volatile and not Soundproof (the user
  included) counts 4, then 3, 2, 1 at the order-12 residual, and faints at 0. Its
  end path runs no faintMessages, so both holders resolve even when the first
  faint empties a side (two last mons tie whichever goes first). Protect and a
  Substitute never block it; Baton Pass carries it; a switch clears it.
- Yawn: accuracy true, Protect and a Substitute block it, it fails on a status, an
  existing yawn and Insomnia/Vital Spirit (`-immune`, revealed on the device).
  Two residuals later `trySetStatus(slp)`: silent if statused, Sleep Clause Mod
  applies, sleep 2+`effect[s][2]`, Lum Berry cures at once. Not Baton-passed.
- Wrap: accuracy 85; a hit on a target that is not already trapped (never through
  a Substitute) traps it for 3+`draws[s][7]` (the attacker's column). Each
  residual lowers the count first; at 0 the trap ends, otherwise a trapper that is
  not the foe's living active (or entered this turn) releases it silently, else
  the holder takes max(1, maxhp/16) after Leftovers. Switches are illegal while
  the trapper is the foe's living active. Rapid Spin (a Substitute hit included)
  and the holder's own Substitute end it; Baton Pass carries it and its source.
- Device: `derived_volatiles` writes B2's perishN, yawn and partiallytrapped bag
  entries from the counters at each move and turn end, and B2's WRAP_TRAP_TURNS
  (column 33, `wrap_trap_turns`) counts TURN lines while trapped, resets on every
  `-end`, an arrival without Baton Pass and a faint, and rides a Baton Pass.

Fidelity WH (White Herb, generally, and C-lite: Sacred Fire's thaw and Facade;
no new state, no new chance):
- White Herb (`data/items.ts` whiteherb, the base item in Gen 3): when its holder
  has a negative stage and lives, `useItem()` consumes it (`-enditem`) and every
  negative stage goes to 0 (positive stages stay). Gen 3 triggers: every move
  that passed BeforeMove and PP (onAnyAfterMove, both actives, before runMove's
  faintMessages, so a Destiny Bond KO of the user comes after its restore); each
  entrant's runSwitch, whose SwitchIn event runs after hazards and before the
  entrant's Ability Start (so an entrant's Intimidate drop is not restored there,
  unlike Gen 5+); and residual order 29 (that drop with no later trigger in the
  turn). Gen 3 Pursuit on a switch is useMove: no AfterMove. A dragged-in mon's
  runSwitch is queued after the Roar's AfterMove. An equal-Speed pair is inert.
- Sacred Fire (the move flag `defrost`): the 20% thaw roll comes first; failing
  it, a frozen user still moves and frz's onModifyMove thaws it in useMove, before
  accuracy, Protect and immunity (a miss or a failure still thaws it).
- Facade: base power 140 when the user has par, brn, psn or tox at getDamage (not
  slp; frozen users cannot use it); Gen 3 burn halving still applies without Guts.
- Device: `white_herb` folds B2's `-enditem` consumption (revealed, removed, no
  ledger mutation, ITEM_END flags) and `-clearnegativeboost` on the holder.

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

Synchronize reflects newly inflicted burn, paralysis and poison from the
implemented primary, target-secondary and Static-contact status writers.
Gen3 Toxic reflects as ordinary poison; sleep, freeze and self-infliction do
not reflect. It uses the current ability and active recipient after switching,
retains source type/ability immunities and existing status, and bypasses the
source's Substitute. Reflection precedes each event's Lum cure, so a later
Static event can inflict status again after an earlier berry was consumed.
The chance proof closes reflected statuses over reachable Trace abilities
before dropping paralysis gates or the residual status-order coin. No new
random coordinate is introduced. Pinned simulator controls and complete
authored Rust/JAX laws cover the represented paths; native continuation
admission, Safeguard representation and broader contact abilities remain
separate gaps. Scientific rosters retain their original source and results.

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
remain, and there is no modern Speed boost. Since fidelity M2 it also ends the
spinner's own partial trap (Wrap), a Substitute hit included; this still does not
qualify move/game coverage.

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
intrinsic effects. Brick Break's screen removal and Sing's Soundproof admission
remain partial, with no whole-move credit (Rapid Spin's partial-trap cleanup
landed with fidelity M2). Hidden Power aliases contribute their actual counts once each.
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

### Complete ordinary carry comparison (opt-in)

`ordinary_law.compare_slot` compares complete ordinary move-request outputs,
including positive pending-residual successors. Supply the already validated
current-root `NativeInput` with retained original constructor provenance,
explicit move slots, case/batch/node bounds, and an owning operation deadline.
Both installed native consumers must match the registered Native98 build.
This API does not change the scientific collector or old panel admission.

The native API has a distinct ordinary context. It independently declares its
chance consumers and emits physical state, queue commitments, prebound address
advancement, and copy/constant expressions for the complete saved bank. The
common partition unions native cuts with the actual supplied JAX-table cuts.
`carry_law` retains each cell's primitive domains and compares complete output
laws without expanding their Cartesian bank; aliases and physical/carry
correlations remain observable. Closed outputs aggregate across cells. An
unsupported positive output, incomplete law or cap refuses the whole slot;
there is no prefix acceptance or normalization.

The initial native domain is live ordinary requests selecting Thunderbolt,
Body Slam, Explosion, Baton Pass, Protect, Endure, Splash or Will-O-Wisp, with the guarded
active abilities/items/statuses. Native98 admits move-order and residual-status
ties using distinct original physical coordinates (`draws.0.0` and
`draws.1.9`). The native residual scheduler freezes post-move speeds; actual
equal-speed 10.x handlers interleave by suborder. In this guarded domain,
Leftovers handlers commute and precede status damage; other tied callbacks
are inactive. Terminal checks remain between handlers. This is an opt-in
domain extension; the general default native tied scheduler is unchanged.
Native97 binds Will-O-Wisp to its native Gen3 75% accuracy cut on the original
accuracy coordinate, admits existing deterministic burn effects and the
species-gated Light Ball special-stat modifier, and preserves Fire immunity,
existing-status failure and Protect. Untouched bench fields remain retained. A representable pending output does
not promise that the next continuation is executable by the narrower resume
API. Native96 applies Gen3's final minimum one to successful nonimmune ordinary
hits, including maximum-zero/one resistance cases, with shared type/volatile
suppression and fixed-callback exclusions. The complete-law comparator admits
those repaired low-damage states and retains true immunity. Existing nonzero
Protect-streak storage at a Baton suspension remains a measured
authored DIVERGE, with equal capsules and differing physical counters. This
difference is retained by the mapper. These authored controls provide no new scientific
panel counts, strength evidence or full-engine qualification.


### Prospective complete ordinary-carry collector

A new entry can opt into the versioned method with retained constructor
provenance and an explicit exact-rational comparison criterion:

```json
"ordinary_carry": {
  "schema": "pokezero.complete-ordinary-carry-panel.v1",
  "criterion": "exact-rational-full-output-law",
  "max_output_nodes": 100000
}
```

The full input/context artifact closes before native root-domain preflight.
Supported roots select the complete ordinary path; the closed set of explicit
unsupported-root codes selects the existing legacy comparator. Pending roots
retain the conditional comparator. Once selected, caps, source drift or any
positive-output refusal cannot select another method. They preserve a whole
unknown slot or stop the owned attempt. Missing configuration preserves the
previous collector behavior.

Each ordinary result retains ordered paired emissions for every cell of the
common partition: domain binding, exact cell mass and both complete physical,
capsule and copy/constant outputs. Terminal audit reconstructs the partition
from the frozen root and retained originals, checks the full consumer
certificate and physical layout, binds the predetermined address advancement,
and recomputes both exact output laws. Removed, duplicated or projected cells
cannot become an accepted prefix. Exact rational equality is this versioned
route's declared criterion; the existing per-key tolerances apply to legacy
and conditional routes.

The operation clock includes binding, preflight, planning, both engines, symbolic
serialization and result/progress persistence. Cost receipts report
`ordinary_carry` separately and retain every refusal. This changed configuration
requires a new frozen cost cohort; existing cost/readout artifacts are unchanged.

### Public encoder input refactor

`public_belief.py` and `public_inputs.py` begin the device implementation in
[`gpu_observation_v4_design.md`](../docs/gpu_observation_v4_design.md). They do
not yet replace the omniscient `observation.py` timing placeholder or provide a
production observation. The complete protocol adapter, private requests, public
state columns, decision parity, champion forward and full-game gates remain.

The source compiler retains variant order and builds reveal/exclusion predicates
with the Python source's own matching rules. Hard evidence, standing damage pins
and effective candidates are separate. Off-source reveals recover the full
species pool with maximum uncertainty; conflicting pins retain the standing
evidence. Unavailable source and unknown species have distinct diagnostics.
Nonempty pins containing identities outside the frozen source fail admission
instead of silently losing identities. Candidate weights represent the current
uniform sampler, not generator frequencies.

The typed stream uses sixteen int32 words per event and drains chunks of 64
without truncation. Stream cursors bind the source table contents, battle ID,
sequence and phase. A malformed or repeated chunk is rejected atomically for
its battle. The fold supports exact turn, blank action-chunk and upkeep
boundaries, plus already-attributed set evidence. With a `public_mon_state`
ledger it also retains public encounter identities, arrival HP/status, condition
changes, bench cures, fainting and active boost operations. Arrival binds the
other observer's set-source slot without erasing prior evidence. These are
field consumers; the raw protocol adapter and remaining consumers, including
volatiles, clocks, item/ability attribution and tendencies, are incomplete.
A move declaration cannot be treated directly as a set reveal, since
called/copied moves need the Python belief attribution rules.

The optional `public_context` fold composes the move ledger with public ability
and Transform state. Original-set reveals, borrowed running abilities and the
parser's Trace feature remain separate. Trace can redirect an original reveal
to its public donor; Transform copies the donor's current public ability.
Pressure charges follow that context and the source's move-target rule, not a
sampled world or the wire target. Authoritative ability corrections rebuild the
current ability mask on retained non-ability evidence; stale ability masks do
not erase recovered candidates. Generic `SET_REVEAL` ability predicates refuse
when context is enabled, requiring the explicitly attributed ability event.
Malformed suffixes roll back the cursor, beliefs, mon ledger and both context
ledgers together. Secondary inference, raw attribution and full encoder gates
remain incomplete.

For the added event kinds, actor/target addresses occupy the same header words.
`MOVE` payload is actor encounter, move ID, caller code and last-move from-tag
flag. `ABILITY` uses the target address and payload encounter, ability ID and
authoritative flag. `TRACE` uses actor and optional donor addresses, with actor
encounter, copied ability ID, donor encounter and parser-feature flag; an absent
donor is address `(-1,-1)` and encounter zero. `TRANSFORM` requires both public
addresses and actor/donor encounter IDs. Remaining payload words are zero, and
all boolean flags are integer zero or one. Each static namespace is separately
bound across continuation. These typed events cover named consumers only and
are not a claim that every consumer of a corresponding raw line is implemented.

These components accept public encounter slots, never infer a slot from species
or the opponent's private team order, and do not read physical engine state.
The future full adapter must supply identity and evidence from public protocol
semantics before this path can pass the production encoder gates.
