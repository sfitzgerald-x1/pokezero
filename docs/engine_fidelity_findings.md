# Engine fidelity differential — findings (track C, v3 plan)

Status: 2026-07-18. First curated sweep of the Showdown-vs-poke-engine
one-turn differential (`pokezero.engine_fidelity`), 15 mechanics × 8 seeds
against the real Node sim and the gen3-feature poke-engine wheel (0.0.47,
unpatched upstream).

Repro:

```bash
python -m pokezero.engine_fidelity --showdown-root <showdown> --out report.json
```

## Result: 13/15 mechanics clean

Clean (all 8 seeds land inside the engine's branch support): basic damage
with crit/secondary branches, ground/Levitate immunities, Toxic vs Immunity,
Thunder Wave + full-para, Spikes set, Reflect and Light Screen set + same-turn
halving, Leech Seed drain routing, Sand Stream chip (engine state seeded with
entry weather), Explosion faint handling with gen3 defense halving, Protect,
Hidden Power typing/BP from IVs, Rest full-heal + sleep.

## Confirmed deviation 1: end-of-turn residual order vs Leftovers

poke-engine applies status residual **before** the Leftovers heal; gen3
(and Showdown) heal with Leftovers **first**, then apply poison/burn/toxic
damage. At full HP the difference is maximal — the engine nets the whole
residual to zero:

- Engine instruction stream (Toxic on a full-HP Leftovers holder):
  `ChangeStatus -> TOXIC, Damage 14, ChangeSideCondition ToxicCount 1, Heal 14`
  → net 0. Showdown: heal 0 (already full), damage 14 → net −14.
- Reproduces identically for burn (`burn_application` case: engine −24 net =
  −48 burn + 24 Leftovers; Showdown −48).

Both diverged cases (`toxic_residual` 2/8, `burn_application` 2/8) match on
exactly the status-miss seeds and fail on every status-hit seed — one
mechanism, fully explained.

**Impact:** systematic optimism about statused Leftovers holders in engine
rollouts (residual pressure halved or erased). Nearly every gen3 randbats set
holds Leftovers, and Toxic appears in 152 sets — this is on-distribution.

**Disposition: PATCHED (2026-07-18, revised after independent review).**
`third_party/poke-engine-gen3-residual-order.patch`, applied by
`scripts/setup_poke_engine.sh`. The first patch revision moved the whole
item/ability loop ahead of status damage; review caught that this dragged
the order-10 threshold berries (Sitrus/pinch) and Rain Dish along with
Leftovers, breaking berry timing on the crossover turn (reproduced). The
shipped patch therefore **splits the phases**: Leftovers (order 5) + Shed
Skin (5.3) resolve before Leech Seed (8) and status damage (9/10);
threshold berries, Rain Dish, and Speed Boost (order 10+) resolve after —
matching Showdown's gen3 residual table for every effect the gen3 engine
implements. Known residual gap: psn(9)-before-brn(10) cross-side
interleaving is not modeled (no cross-side coupling in the status block;
observable only in simultaneous last-mon faint tiebreaks — pre-existing,
negligible).

Regression gates: the differential stays **15/15 clean** (it only exercises
Leftovers among items — a scope limit, not full-ordering proof), and
`tests/test_engine_residual_order.py` pins the berry/Leftovers/Shed-Skin
orderings directly against the engine at mid-battle HP states the one-turn
differential cannot reach. Worth reporting upstream: the original
all-items-after-status ordering is a real gen3 bug in poke-engine 0.0.47.

## Confirmed deviation 3: Attract volatile was accepted but ignored

poke-engine 0.0.47 has `PokemonVolatileStatus::ATTRACT` in the gen3 enum
(`src/gen3/state.rs`) but **zero behavioral references** to it in
`src/gen3/generate_instructions.rs` / `choice_effects.rs` (nor in genx —
upstream never modeled infatuation immobilization). Probe-confirmed (walls
audit 2026-07-19): an attracted mon's turn is byte-identical to a mon with no
volatile — it moves 100% of the time. Real gen3 Attract immobilizes the holder
50% per turn (`data/moves.ts` `attract` condition, `onBeforeMove` priority 2,
`randomChance(1,2)`).

**Impact:** systematic optimism about the attracted seat (it "always moves").
On-distribution frequency is low — the Attract *move* is absent from the gen3
randbats pool; infatuation arises only via Cute Charm procs
(Clefable/Delcatty/Wigglytuff, gen3-mod 1/3 on contact, opposite gender) — a
proc-gated singleton in the 100-seed band. But when it hits, every turn the
attracted mon acts is over-valued by ~50%.

**Disposition: PATCHED (2026-07-19).**
`third_party/poke-engine-gen3-attract.patch`, applied AFTER the residual-order
patch by `scripts/setup_poke_engine.sh` / `scripts/vendor_poke_engine_src.sh`
(`--fuzz=0`). The patch adds a 50/50 **chance branch** in
`generate_instructions_from_existing_status_conditions`, directly mirroring the
confusion self-hit branch immediately above it: it clones the incoming
instructions, weights the clone 0.5 and pushes it as a terminal (the move never
executes), and reduces the surviving branch to 0.5. Unlike confusion the
immobilized branch carries an **empty delta** (attract deals no self-damage) —
the same shape as the fully-paralyzed branch. This is a weighted two-outcome
split, not a point mutation, so the exact-expectation backup and chance-node
machinery price it exactly, as they already do for full paralysis and the
confusion self-hit.

**Composition ordering (verified vs Showdown gen3 source).** Showdown resolves
`onBeforeMove` high-priority-first: flinch (8) → confusion (3) → attract (2) →
paralysis (1). The engine's order is: flinch/taunt hard-gate in
`cannot_use_move` (100% skip, before any status branch) → paralysis → freeze →
sleep → confusion → **attract** (the patch) → move. (Since 2026-07-28 the
paralysis roll moved to the END of that chain — deviation 4.) Consequences:

- **flinch**: a 100% "can't move" gate in BOTH sims; attract/confusion/par are
  never reached under a flinch. Exact.
- **confusion → attract**: same order as Showdown. A confused+attracted mon is
  exact in both probability AND reason attribution (50% confusion self-hit, then
  25% attract-immobilize, 25% move — identical to Showdown's
  confusion-before-attract resolution).
- **paralysis**: the engine resolved it *before* attract; Showdown resolves it
  *after*. The internal immobilized-reason split therefore differed (engine:
  25% par / 37.5% attract; Showdown: 50% attract / 12.5% par) but the net
  P(move) = 0.75 × 0.5 = **0.375** was identical (the two independent gates are
  commutative) and BOTH immobilized branches are empty-delta terminals, so the
  **leaf-state distribution was exact**. Only the rendered `|cant|` reason label
  (an events.rs concern) depended on which volatile was credited — invisible to
  search value. **Superseded (2026-07-28):** the confusion-duration patch moves
  the paralysis roll to last, matching Showdown's priority order exactly, because
  a bounded confusion counter makes the misordering stop being benign — see
  "Confirmed deviation 4" below.

**Source-leave handling (closed 2026-07-22).** Real Gen 3 Attract clears when
the infatuation source leaves play (`onUpdate` removes the volatile when
`effectState.source` goes inactive; Showdown emits a public
`|-end|mon|Attract|[silent]`). The engine does not store an infatuation source
id, but Gen 3 random battles are singles: if either active switches, that active
must be either the holder or the holder's sole possible source. The ability-
fidelity patch therefore clears Attract from the opposite side during switch
generation; the engine's existing switch cleanup clears it from the switching
side. A regression test pins the source-switch case. This removes the former
bounded over-model without adding hidden source identity to the state.

Regression gates (all in the dedicated `.venv-attract`, never the shared venv):
`scripts/attract_differential.py` is the residual-order-caliber ground-truth
gate — it drives two curated gen3 Custom Game scenarios (free attract; Thunder
Wave + attract composition) through the real Node sim over 100 seeds and the
patched engine, and asserts the branch probabilities match within 4σ, the
`|-activate|…|move: Attract|[of]…` line appears every measured turn, and BOTH
the move and immobilize branches (and, for para, both `cant Attract` and
`cant par`) actually occur. Latest: free 50/50 engine vs 54/46 Showdown, para
37.5/62.5 engine vs 42/58 Showdown, activate 100/100 both — PASS.
`tests/test_engine_attract_immobilization.py` pins the instruction-generation
output shape (exact 50/50 free split, empty-delta immobilized branch, 37.5%
para-composition move probability, 100% move without the volatile) so a wheel
rebuild cannot silently regress the immobilization back to a no-op.
`golden_corpus_scenarios.py::attract_snorlax` exercises the world-construction
path end-to-end (free + para composition on the search seat) in the fallback
sweep: 3/3 decisions searched, 0 walls. Known encoder follow-up: the Attract
*move* is outside the closed gen3 randbats vocab, so the scenario's
`move_effect:attract` / `belief:possible_move:attract` tokens hash to the
safety-net row (deterministic, graceful) — the move is used only because it is
the sole way to place attract deterministically on the search seat; real games
reach the volatile via Cute Charm, whose token IS enumerated.

## Confirmed engine contract 2: Hidden Power ids must be typed + base power

The gen3 engine move table only accepts fully-qualified ids
(`hiddenpowergrass70`); bare `hiddenpower` silently resolves as a weak
typeless hit, and the randbats set pool stores type-only ids
(`hiddenpowergrass`, 210 occurrences). The world constructor now translates
via `engine_world.hidden_power_engine_id` (type + BP derived from IVs, with a
fail-closed IV-consistency guard). This was a track-A bug found by track C —
without the differential it would have shipped as a silent damage-zeroing of
a very common move.

## Known engine deviation (low severity): Wish heal amount

poke-engine ignores the `wish` tuple's amount and heals the RESOLVING
ACTIVE's maxhp/2; gen3 heals by the CASTER's maxhp/2. Observable only when
the caster switches out before the wish lands and the recipient's maxhp
differs. Verified empirically on the patched wheel (amount 0/350/999 all
heal active maxhp/2). Documented rather than patched — low value impact,
and the world constructor records the timing exactly.

## Harness notes and scope (what "clean" does and does not mean)

- Damage matching uses a ±16% band around the engine's representative
  (average) roll. That band is tight ONLY because every curated case
  isolates its mechanic on a mon taking no other damage — the band scales
  with a branch's total damage, so a sub-16%-of-damage mechanic error
  riding alongside a big hit would be masked. Independently reviewed and
  confirmed: this is a latent false-CLEAN vector for any reuse of this
  matcher on non-isolated turns.
- Coverage is support-membership over 8 seeds: an engine that is MISSING a
  low-probability branch passes unless Showdown happens to roll it
  (a 10% branch goes unobserved across 8 seeds with p≈0.43). The current
  run did exercise freeze (~10%) and full-para, but that was luck, not
  design.
- Side conditions are compared presence-only (screen turns-remaining is
  never validated — needs a multi-turn case); boosts, volatiles, benched
  effects, and rest/sleep turn counts are invisible to the feature fold.
  "13/15 clean" means the tested observable effects match, not full effect
  fidelity for every rider on those turns.
- Entry abilities (Sand Stream) fire before the fixture turn; such cases seed
  the engine state (`spec_weather`), mirroring what the world constructor
  does from the public payload mid-game.
- The unpatched upstream wheel is deliberate for measurement; the Rest/Sleep
  Talk PP-underflow patch from `setup_foulplay_eval.sh` should be re-verified
  by a dedicated case when multi-turn fixtures land.

## Next (with prerequisites for tier 2)

Multi-turn curated cases (Sleep Talk, Baton Pass volatile transfer, Encore,
partial trapping, screen duration/expiry), then the tier-2 real-game sweep:
replay recorded decision points through `engine_world` and check each
observed Showdown outcome lies in the engine's branch support.

Tier 2 must NOT reuse this matcher as-is: real turns stack residuals and
chip on top of attack damage, exactly where the net-HP band goes blind.
Prerequisites before tier 2 can serve as a go/no-go read: per-instruction /
per-damage-source comparison (or a band tied to the mechanic under test,
not net active HP), branch-coverage assertions or a much larger seed count
for probabilistic effects, and turn-count validation for timed conditions.

## Multi-turn differential (tier-2 wave 1)

Status: 2026-07-18. Six curated multi-turn cases (3-7 scripted decision
boundaries, 4 seeds each) in `pokezero.engine_fidelity_multiturn`, run against
the real Node sim and the gen3-PATCHED wheel (0.0.47 + residual-order split).
Per step the observed Showdown turn must land in the engine's
`generate_instructions` branch support and the engine then CONTINUES from the
matched branch's applied state, so timed counters are validated by their
downstream effects (a wrong screen counter changes damage and misses the
support), plus per-step engine counter traces asserted on fully-matched seeds.

Repro:

```bash
python -m pokezero.engine_fidelity_multiturn --showdown-root <showdown> --out report.json
```

### Result: 6/6 cases clean (24/24 seed trajectories, every scripted step matched)

| Case | Steps x Seeds | Verdict | What it pinned down |
| --- | --- | --- | --- |
| `reflect_expiry` | 7 x 4 | clean | Engine reflect counter ticks 5->0 (trace `4,3,2,1,0,0,0` after steps 1-7); damage halved turns 2-5, un-halves turn 6+ in BOTH sims; a crit-through-Reflect branch (gen3 crits pierce screens) was hit and matched on seed 24. |
| `toxic_escalation` | 3 x 4 | clean | Residuals escalate 1/16 -> 2/16 -> 3/16 (engine `toxic_count` trace `1,2,3`), with the patched heal-BEFORE-status-damage Leftovers ordering holding at every stage. Seeds screened for the 85%-accuracy hit on step 1. |
| `resttalk_cycle` | 6 x 4 | clean | Rest = full heal + SLEEP + `rest_turns 0->3`; Sleep Talk branches (called Body Slam / called Curse / called Rest) all exercised across seeds; wake on the 3rd Sleep Talk turn in both sims. ALSO the PP-underflow canary — see below. |
| `baton_pass_transfer` | 5 x 4 | clean | Calm Mind x2 survives the mid-turn Baton Pass switch on the engine side (boost telemetry `+2` after the switch) and on the Showdown side (step-5 Surf at +2 doubles damage — far outside the roll band, and it matched the +2 branch). |
| `encore_lock` | 3 x 4 | clean | Engine auto-tracks `last_used_move` when Encore is in a moveset, redirects the target's already-chosen move to the encored one on the application turn (Showdown agrees), and holds the lock next turn. Duration NOT validated (below). |
| `sand_chip_multi` | 3 x 4 | clean | Sand chips 1/16 per turn on the itemless holder while the sand-immune Leftovers holder nets 1/16 back per turn, including the clamp at full HP. |

### PP-underflow canary: NOT reproduced on this wheel

The historical Rest/Sleep Talk PP panic does not fire on the patched 0.0.47
wheel via any path we can drive (`pp_underflow_canary`, attached to the
`resttalk_cycle` report row):

- `generate_instructions`/`apply_instructions` do not decrement PP at catalog
  PP at all (the engine only emits `DecrementPP` near zero), so the
  sleep-talk-called-Rest interplay never touches PP on realistic states;
- forcing Rest at 0 PP is ACCEPTED (the engine happily selects a 0-PP move)
  and `DecrementPP` wraps the stored PP to **-1** — a silent underflow, not a
  panic (mild contract note: the caller owns not submitting 0-PP moves);
- two forced Sleep Talk turns from that state and a 200 ms
  `monte_carlo_tree_search` burst (134k+ visits) complete cleanly.

Settled in passing: gen3 Showdown's sleep-talk-called Rest FAILS outright
while asleep (protocol shows `|move|...|Rest|[from] Sleep Talk` with no
effect), which is exactly the engine's 1/3 no-op branch — the two sims agree.

### Engine caller-contract sharp edges (confirmed, fail-silent/fail-late)

1. **Force-switch resolution drops the postponed move unless re-supplied.**
   When the slower side's move is postponed across a Baton Pass switch-out
   (`SideXMoveSecondSwitchOutMove` saved), the resolution call must pass the
   switching side's BARE species id (`"starmie"` — `"switch starmie"` raises
   `ValueError`) and must RE-SUPPLY the saved move for the waiting side:
   passing `"none"` returns a valid 100% branch in which the opponent's move
   silently never happens. Any search/world integration that resolves forced
   switches with `"none"` will corrupt its rollouts without an error.
   `engine_fidelity_multiturn.engine_step_choices` re-supplies from
   `side.switch_out_move_second_saved_move`.
2. **`Side(last_used_move=...)` takes a move INDEX, not a move id, and fails
   late.** The constructor accepts `"move:growl"` but `generate_instructions`
   later PANICS (`PanicException: Invalid PokemonMoveIndex: growl`,
   `state.rs:100`); the valid format is `"move:1"` (slot index). Engine-built
   trajectories are safe — `SetLastUsedMove` instructions are only emitted
   (and only when Encore is present in a moveset), always in index form.
3. **Encore duration is not modeled.** gen3 Showdown rolls 3-6 turns
   (`random(3, 7)`, counting the application turn); the engine applies the
   `ENCORE` volatile with `volatile_status_durations.encore` stuck at 0 and
   never expires it. Trajectories longer than the guaranteed lock prefix will
   diverge at Showdown's expiry roll. This, plus the index-form
   `last_used_move` requirement on world construction, keeps Encore
   **fail-close in `engine_search`**: a mid-game world would need the
   opponent's last-move slot index and a duration model the engine lacks.

### Harness additions over the one-turn matcher (and remaining limits)

- **Boost-delta matching** (`observed_boost_deltas` from `|-boost|/|-unboost|`
  lines vs per-branch engine stage deltas) — REQUIRED for correctness, not a
  nicety: Sleep Talk calling Curse vs calling Rest (no-op) are observationally
  identical in `TurnFeatures`, and without the filter the trajectory binds to
  the wrong applied state and falsely "diverges" one step later (observed as
  exactly that before the fix). Per-step deltas also sidestep absolute-stage
  tracking across Baton Pass (stages transfer with no protocol echo).
- **Drift correction with raw fallback**: the followed engine branch carries
  average rolls, so observed HP is shifted by the accumulated
  (engine - showdown) offset per side before matching (per-step delta
  comparison); a heal-to-full clamps both sims and makes the offset stale for
  one step, so the unadjusted observation is a fallback (fired exactly once in
  the sweep, on the sand case's Recover step). Offsets reset when a side's
  active changes.
- Still support-membership over 4 seeds per case — probabilistic branch
  COVERAGE remains the one-turn suite's (partially open) problem; sleep-talk
  call distribution (1/3 each) and encore/para/crit sub-branches were hit by
  luck of the scripted seeds, not asserted.
- Timed conditions validated here: screens (counter + expiry), toxic stage,
  rest/sleep-talk wake. NOT yet: Light Screen expiry (symmetric to Reflect but
  unexercised), Safeguard/Mist durations, encore expiry (engine has no
  counter), weather expiry for manual (non-ability) weather.
- `"switch N"` script entries resolve against ORIGINAL team order; fine for
  wave 1's single Baton Pass from the opening lineup, revisit before scripting
  multi-switch cases.

### Scope clarification (added after independent review)

"Clean" in the multi-turn sweep certifies that EACH TURN's observed delta
lands in the engine's branch support within the same ±16%-of-per-turn-damage
band, plus timed-counter fidelity (reflect ticks, toxic_count, rest_turns).
Because observed HP is re-anchored to the engine's trajectory every turn,
absolute HP tracking across N turns is NOT certified: a systematic engine
damage bias smaller than the per-turn band (e.g. ~10%/turn) would pass 6/6
clean. This is the one-turn doc's sub-band masking caveat applied with more
force, and it carries the same consequence for tier-2 reuse. Also: when two
engine branches are feature-identical AND both fall inside the HP band, the
matcher binds the FIRST in enumeration order (first-match, not best-fit) —
no ties were observed in wave 1's curated cases, but this is a latent
false-CLEAN vector for real-game turns. Encore wave-1 coverage: the
application-turn redirect and volatile persistence are validated; the
next-turn lock is only exercised trivially (the scripted choice coincides
with the encored move) and duration remains unmodeled.

## Confirmed deviation 4: CONFUSION was permanent (and never Baton-Passed)

poke-engine 0.0.47 models gen3 CONFUSION as an unbounded 50%-per-turn self-hit
that persists until switch-out — there is **no expiry path anywhere in
`src/gen3/`**. `VolatileStatusDurations.confusion` exists in the shared
`src/state.rs`, is serialized as field 0 of the duration blob, and
`increment_volatile_status_duration` already dispatches it — but no generation
ever reads or writes it. Confirmed by the PR #874 switch-out audit.

Real gen3 (`data/conditions.ts` `confusion`, whose `onStart` gen3 inherits
unchanged through gen4 → gen5 → …, plus `data/mods/gen4/conditions.ts`'s
`onBeforeMove`, which is the one gen3 resolves to) rolls
`this.effectState.time = this.random(2, 6)` **once**, at `addVolatile` — uniform
on {2,3,4,5}. Every `onBeforeMove` that actually runs decrements `time` FIRST,
snaps out and lets the move through when it hits zero, and only otherwise emits
`-activate` and rolls `randomChance(1, 2)` for the self-hit. So the number of
attacking turns that carry a self-hit roll is `time - 1` — **uniform on
{1,2,3,4}, never five**. (The gen7 change to 33% does not reach gen3: gen4's
`onBeforeMove` shadows it.)

**Impact:** systematic pessimism about any confused seat, unbounded in the
search horizon. A confused Pokemon that never switches was priced at a
50%-per-turn self-hit forever, so search over-valued Confuse Ray / Swagger /
Water Pulse lines and over-valued switching out of confusion. Reachable
everywhere in gen3 randbats.

**Disposition: PATCHED (2026-07-28).**
`third_party/poke-engine-gen3-confusion-duration.patch`, applied last by
`scripts/vendor_poke_engine_src.sh` (`--fuzz=0`). Modelled as a **hazard
ladder** over the already-present duration counter — `chance_confusion_ends(n)`
is `1 / (1 + MAX_CONFUSION_TURNS - n)`, the identical shape gen3 sleep already
uses in `chance_to_wake_up`, because gen3 sleep is rolled from the same
`random(2, 6)`. Given `n` attacking turns already burned, the chance this was
the last one carrying a self-hit roll is `P(time == n+2 | time > n+1)`: 1/4,
1/3, 1/2, forced. This is **exact, not an approximation** — it reproduces the
uniform-on-{1,2,3,4} marginal while holding the per-turn self-hit at 50%, and
costs one extra branch per confused turn instead of the four-way fan-out a
roll-at-application model would need.

**Where the fork is taken.** At END OF TURN (`add_end_of_turn_branches`), not at
the start of the next attacking turn. Forced by the engine's structure: a
snap-out branch has to CONTINUE through the move, and
`generate_instructions_from_move` advances exactly one surviving branch, whereas
at end of turn both outcomes are terminal. Equivalent for everything observable
— Showdown's snap-out turn carries no self-hit roll and lets the move through,
exactly like a turn on which the volatile is already gone. **The only residue:**
Showdown keeps an inert `time == 1` confusion visible for one decision boundary
longer than the engine does. That costs a one-boundary eval difference, and lets
the engine re-confuse a Pokemon that Showdown would refuse (`addVolatile` bails
with no `onRestart`) on exactly that boundary. Both are bounded and neither
changes the distribution of future self-hits.

**Gating.** The ladder fires only on turns the check actually ran, keyed on the
`+1` duration instruction the confusion block emits. Showdown's `onBeforeMove`
priority 3 is pre-empted by recharge (11), sleep and freeze (10), flinch (8),
Disable (7) and Taunt (5), all of which leave `time` untouched; the engine
reproduces that reachability exactly (the first four short-circuit in
`cannot_use_move` or push their no-move branches before the confusion block).
Gating on the counter's *value* instead would burn turns while standing still.

**Paralysis reorder (rides along).** The full-paralysis roll moves out of the
status match and after the confusion and Attract branches, so the engine now
matches Showdown's priority chain exactly (confusion 3 → attract 2 → par 1).
Required: rolling paralysis first meant a fully-paralyzed turn never reached the
confusion block, so the counter advanced on only 75% of a paralyzed Pokemon's
turns and its confusion outlasted Showdown's. It also restores the self-hit mass
for a paralyzed + confused Pokemon to Showdown's 1/2 (it was 3/8), since a
confusion self-hit aborts the move before paralysis is ever rolled. This
supersedes the "leaf-state distribution is exact" note under deviation 3.

**Baton Pass carry (rides along).** `copyVolatileFrom` copies every volatile
without a `noCopy` flag and shallow-clones the volatile object; `confusion`
carries no such flag anywhere in gen3's chain, so the remaining `time` rides the
pass and `onStart` never re-runs — the receiver gets no fresh roll. Upstream's
`remove_volatile_statuses_on_switch` retained only Substitute and Leech Seed.
The batonpass-perish patch deferred this deliberately: carrying a **permanent**
confusion would have been worse than dropping it, so the carry had to wait for
the duration. The counter is zeroed alongside the volatile on an ordinary
switch-out, on the Own Tempo cure (reachable now that a pass can hand a
confusion to an Own Tempo receiver), and on a fresh application.

**Verification.** Real gen3 Showdown via
`scripts/gen3_switch_differential.py --only confusionduration
confusiondurationcontrol confusionbatonpass confusionbatonpasscontrol`; engine
side pinned by `rust/pokezero-search/tests/gen3_confusion_fidelity.rs`
(11 tests). Seeds 1000-1003 of the duration scenario happen to cover all four
legal durations (1, 2, 3 and 4 self-hit-risk turns).

## Confirmed deviation 5: ENCORE never expired

poke-engine 0.0.47 applies the gen3 ENCORE volatile and enforces the lock, but
**never expires it**. `VolatileStatusDurations.encore` exists in the shared
`src/state.rs` and is serialized, and `increment_volatile_status_duration`
dispatches it — but the only code that advances it lives in
`src/genx/generate_instructions.rs` behind
`#[cfg(any(feature = "gen5", ..., feature = "gen9"))]`, which never compiles for
a gen3 build. So a gen3 Encore locked its victim for the rest of the battle.
Confirmed by the differential lane (D9, seeds 991000-991059: 3/4/5/6-turn ends
across 60 samples).

Real gen3 overrides the duration explicitly. `data/mods/gen3/moves.ts` gives
`encore.condition.durationCallback() { return this.random(3, 7); }` — uniform on
{3,4,5,6}. **This is the gen3-inherits-gen4-not-gen5 trap in its sharpest form**:
gen4's own override says `this.random(4, 9)` = {4..8}, and gen5-gen8 do not
define `encore` at all, so reading either neighbour gives the wrong window. Every
other handler (`onStart`, `onOverrideAction`, `onResidual`, `onEnd`,
`onDisableMove`, `noCopy`) comes unchanged from base `data/moves.ts`; gen4
contributes only `duration: undefined` (killing base's `duration: 3`) and the
residual ordering.

Unlike confusion this rides Showdown's **generic** duration machinery, not an
`onBeforeMove` counter: `sim/battle.ts` decrements `handler.state.duration` once
per turn in the RESIDUAL phase and calls `end` at zero. The tick therefore lands
whether or not the encored Pokemon moved, and the engine's end-of-turn block is
that same point in the turn — so unlike the confusion ladder there is no timing
approximation at all.

**Impact:** an encored seat was modelled as locked forever. Search under-valued
every line that plays through an Encore and over-valued Encore as an attack.
Reachable: 16 gen3 randbats species carry Encore.

**Disposition: PATCHED (2026-07-28).**
`third_party/poke-engine-gen3-encore-duration.patch`, applied last of the
behaviour patches. Hazard ladder over the existing counter,
`chance_encore_ends(n) = 0` while `n < 3`, else `1 / (1 + MAX_ENCORE_TURNS - n)`:

| n (ticks burned, incl. this one) | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| P(end) | 0 | 0 | 1/4 | 1/3 | 1/2 | 1 |

which multiplies out to a uniform marginal, with nothing surviving a seventh:

    P(3) = 1/4;  P(4) = 3/4 · 1/3 = 1/4;
    P(5) = 3/4 · 2/3 · 1/2 = 1/4;  P(6) = 3/4 · 2/3 · 1/2 · 1 = 1/4

**The `onStart` compensation.** `encore.onStart` ends with
`if (!this.queue.willMove(target)) this.effectState.duration!++`. The residual
tick at the end of the application turn is free when the target has already
moved — it burns a turn on which the target was never locked — so Showdown hands
one back. The net invariant is that **the victim is locked for exactly `duration`
turns regardless of speed order**, which is why the roll of {3,4,5,6} reads as
"Encore lasts 3-6 turns". Counting up instead of down, the patch reproduces this
by seeding the counter at `-1` when the encorer moved second and `0` when it
moved first, so the counter always means "locked turns elapsed". Pinned on both
sides: `the_application_turn_seeds_the_counter_by_move_order`, and the
differential pair `encoreduration` (span 3-6) vs `encoredurationslow` (span 4-7,
the same seeds shifted by exactly one).

**PP early termination.** The same `onResidual` ends Encore the moment the
encored move hits 0 PP. Modelled, because the engine's own option filter drops
0-PP moves — without it an encored Pokemon whose locked move is exhausted has no
legal move at all and falls through to "No Move". Showdown decrements the
duration *before* running `onResidual` and `continue`s if it hit zero, so the PP
check is skipped on the final turn; the patch checks PP first instead, which is
outcome-equivalent (both end Encore, both emit the same `-end`).

**No Baton Pass carry.** The condition carries `noCopy: true`, so — the opposite
of confusion — Encore is not passed. The switch-out arm drops it and zeroes the
counter, mirroring the LOCKEDMOVE / YAWN / TAUNT arms.

**Not fixed here (application-time, separate divergence):** Showdown fails Encore
outright when the target's last move carries `failencore` — in gen3 that is
Encore, Mimic, Mirror Move, Sketch, Struggle and Transform — and when the target's
last move is not in its move slots or already has 0 PP. The engine only checks
that the target's `last_used_move` is a move at all, so it will happily Encore a
Struggle. Bounded and orthogonal to duration.

**Caller-contract note.** `engine_world` does **not** fail Encore closed, contrary
to the D9 ledger row: it supports the volatile and declines only when the locked
move cannot be derived (`encore_move_unknown`). It seeds
`volatile_durations["encore"] = 1`, which was inert before this patch and is live
after it — the value now means "one locked turn already elapsed", a deliberate
floor. Deriving the true elapsed count from observation history is follow-up work
and only becomes measurable once the wheel carries this patch.

**Verification.** Real gen3 Showdown via `scripts/gen3_switch_differential.py
--only encoreduration encoreoutlivesshortest encoredurationslow
encoredurationcontrol`; engine side pinned by
`rust/pokezero-search/tests/gen3_encore_fidelity.rs` (9 tests).

## Confirmed deviation 6: Encore applied when Showdown refuses it

Deviation 5 bounded how long a gen3 Encore lasts. This one bounds when it may
start. Upstream's only application check was whether the target had ever used a
move (`move_has_no_effect`'s ENCORE arm), so the engine would happily Encore a
**Struggle** — locking a seat into a move it cannot select — or a move with no PP
left.

Showdown's `encore.condition.onStart` (base `data/moves.ts`; nothing in gen3's
chain overrides `onStart` itself) also refuses on the move's identity and PP:

    let move = target.lastMove;
    if (!move || target.volatiles['dynamax']) return false;
    const moveSlot = target.getMoveData(move.id);
    if (move.isZ || move.isMax || move.flags['failencore'] ||
        !moveSlot || moveSlot.pp <= 0) return false;

`dynamax`/`isZ`/`isMax` are unreachable in gen3. `!moveSlot` is structurally
unreachable in the engine, which stores `last_used_move` as a SLOT INDEX rather
than a move id, so it always denotes a real slot — the case it catches in
Showdown is Struggle, which is not in `moveSlots` and which the flag list rejects
anyway. The `!move` arms were already right: `LastUsedMove::None`, and
`LastUsedMove::Switch` for a fresh switch-in, since `Pokemon.clearVolatile()`
nulls `lastMove` on switch-out.

**The `failencore` set, resolved per chain level.** A mod's `flags` object
REPLACES its parent's wholesale rather than merging, so the set has to be walked
gen3 -> gen4 -> gen5 -> gen6 -> gen7 -> gen8 -> base and taken from the FIRST
level that declares flags for each move. Six moves carry it as gen3 sees them:

| move | gen3's flags come from | failencore |
|---|---|---|
| `mimic`, `mirrormove`, `sketch`, `struggle` | `data/mods/gen3/moves.ts` | yes |
| `encore`, `transform` | `data/mods/gen4/moves.ts` | yes |

**The replacement rule cuts both ways, and the inverse cases are the trap.** Four
more moves carry `failencore` in base `data/moves.ts` but lose it to a nearer
override that re-declares `flags` without it:

| move | nearest override | resulting flags | failencore |
|---|---|---|---|
| `assist` | gen3 | `{ metronome, noassist, nosleeptalk }` | no |
| `metronome` | gen4 | `{ noassist, failcopycat, nosleeptalk, failmimic }` | no |
| `naturepower` | gen4 | `{ metronome }` | no |
| `sleeptalk` | gen6 | `{ nosleeptalk, noassist, failcopycat }` | no |

So none of the four fails Encore in gen3. Reading base — or gen8, which re-adds
the flag to `sleeptalk` and `mirrormove` — would over-fail all four, which is why
`encore_succeeds_against_moves_that_lose_failencore_in_gen3` pins the negative
side explicitly.

**Regenerating this list needs care.** The six are hardcoded deliberately. A
Dex-derived `failencore` set for gen3 resolves **twelve** moves, because six more
leak in from the inherited base table despite having no gen3-legal existence at
all — `blazingtorque`, `combattorque`, `dynamaxcannon`, `magicaltorque`,
`noxioustorque`, `wickedtorque`, all gen8/gen9. Anyone re-deriving the set
programmatically must intersect against the gen3 move pool as well as walking the
flag overrides, or it will over-fail in both directions at once.

**Coupled with deviation 7.** This patch decides Encore's fate *from*
`last_used_move`, so a right failure set applied to a wrong `last_used_move`
still produces the wrong answer — and before deviation 7 that value was wrong on
exactly the turns where a target was immobilized. The two are one story, the
application gate and the value it reads, and should be reviewed and changed
together rather than as unrelated items.

**Impact:** search could commit to an Encore that Showdown refuses, and then plan
against a lock that never existed. Reachable: 16 gen3 randbats species carry
Encore, and Transform/Mimic/Mirror Move are all in the pool.

**Disposition: PATCHED (2026-07-28).**
`third_party/poke-engine-gen3-encore-failencore.patch`, applied after
encore-duration and before the test-only fixture refresh. Touches the Encore arm
of `move_has_no_effect` only.

**Not fixed here (pre-existing, orthogonal).** The engine records
`last_used_move` before resolving the paralysis roll, so a fully-paralyzed turn
still updates it, while Showdown sets `lastMove` only for a move that actually
executed. That can make Encore target a move Showdown would not have offered.
Bounded, unrelated to the failure set, and unchanged by this patch.

**Verification.** Real gen3 Showdown via `scripts/gen3_switch_differential.py
--only encorefailstruggle encorefailnolastmove encorefailmirrormove
encoreappliescontrol`; engine side pinned by
`rust/pokezero-search/tests/gen3_encore_failencore_fidelity.rs` (8 tests).

## Confirmed deviation 7: last_used_move recorded for moves that never executed

The engine recorded `last_used_move` before resolving the status branch, so a
turn spent fully paralyzed, asleep, frozen or hitting itself in confusion still
counted as "used". Showdown records only a move that actually got past the
`BeforeMove` gate.

The set site is singular and precise: `Pokemon.moveUsed()`, called from
`BattleActions.runMove` (`sim/battle-actions.ts:291`) after the gate and after PP
deduction:

    const willTryMove = this.battle.runEvent('BeforeMove', pokemon, target, move);
    if (!willTryMove) { runEvent('MoveAborted', ...); ...; return; }   // no moveUsed
    if (move.beforeMoveCallback) { ... return; }                       // no moveUsed
    if (!pokemon.deductPP(...) && move.id !== 'struggle') {
        this.battle.add('cant', pokemon, 'nopp', move); ... return;    // no moveUsed
    }
    pokemon.moveUsed(move, targetLoc);                                 // lastMove SET

The `runMove` / `useMove` distinction is what settles the "failed vs executed"
question: `moveUsed` runs in `runMove` BEFORE `useMove` is entered, so accuracy,
immunity, Protect and outright failure all happen downstream of the record. A
move that misses or fails still counts as used; only a move that never started
does not.

**Truth table** (gen3-chain handler deciding each row):

| turn outcome | lastMove | decided by |
|---|---|---|
| move executes — hit, miss, fail, or Protect-blocked | **SET** | `moveUsed` precedes `useMove` |
| fully paralyzed | not set | `data/mods/gen4/conditions.ts` `par.onBeforeMove` -> `false` |
| asleep, stays asleep | not set | `data/mods/gen3/conditions.ts` `slp.onBeforeMove` -> `false` |
| asleep, wakes and moves | **SET** | gen3 `slp` cures, returns undefined |
| asleep using Sleep Talk | **SET** | `move.sleepUsable` -> gen3 `slp` returns undefined |
| the move Sleep Talk CALLS | not set | called via `useMove`, which never touches `lastMove` |
| frozen, stays frozen | not set | `data/mods/gen4/conditions.ts` `frz.onBeforeMove` -> `false` |
| frozen, thaws and moves | **SET** | gen4 `frz` cures, returns undefined |
| flinched | not set | `data/conditions.ts` `flinch.onBeforeMove` -> `false` |
| confusion self-hit | not set | `data/mods/gen4/conditions.ts` `confusion` -> `false` |
| confusion snap-out, or no self-hit | **SET** | returns undefined, move proceeds |
| infatuation immobilized | not set | `data/moves.ts` `attract` -> `false` |
| no PP (`cant nopp`) | not set | PP gate returns before `moveUsed` |

**Impact:** Encore's `onStart` reads `lastMove`, so the engine could Encore a
move the target never got to make — and, since deviation 6, decide Encore's
success or failure from it. A fully-paralyzed Pokemon appeared to have "used" the
move it was about to use, which is exactly the move Showdown would not have
offered Encore.

**Disposition: PATCHED (2026-07-28).**
`third_party/poke-engine-gen3-lastmove-semantics.patch`. The record point moves
below `generate_instructions_from_existing_status_conditions` and below the
still-asleep early return, so only the branch on which the move really executes
reaches it — those immobilized branches are terminal and were already pushed to
`final_instructions`. It stays ABOVE `move_has_no_effect` and the accuracy roll,
matching `moveUsed` preceding `useMove`.

**Sleep Talk needed explicit handling**, not just relocation. Sleep Talk is
`sleepUsable`, so gen3's `slp.onBeforeMove` returns undefined and Sleep Talk
itself reaches `moveUsed`; the move it calls runs through `useMove` and must not
overwrite the record. A naive move of the record point past the Sleep Talk branch
would have inverted exactly this — recording the called move and not the caller —
so the caller records explicitly and the recursive sub-calls are excluded by a
`!choice.sleep_talk_move` guard.

**Composition with earlier patches.** The confusion ladder's paralysis reordering
(deviation 4) put the immobilizers in Showdown's priority order, which is what
makes a single record point below all of them correct; the pins assert the
stacked case (confused + paralyzed records on 1/2 x 3/4 = 37.5% of the mass).

**Still divergent (out of scope, newly measured).** PP is deducted *before* the
status branch in the engine but *after* the `BeforeMove` gate in Showdown, so a
fully-paralyzed, asleep or self-hitting turn still costs the engine a PP where
Showdown charges none. Flinch is already correct (it short-circuits in
`cannot_use_move` above the decrement). This is the same ordering family as the
fix above and the two lines sit adjacent, but it is a distinct observable and is
left for its own change.

**Verification.** Real gen3 Showdown via `scripts/gen3_switch_differential.py`;
engine side pinned by `rust/pokezero-search/tests/gen3_lastmove_semantics.rs`
(12 tests, one per truth-table row).

## Confirmed deviation 8: PP charged for moves that never executed

The sibling of deviation 7, and the last piece of the `runMove` prologue. The
engine deducted PP before the status branch, so an immobilized turn cost a PP;
Showdown charges only after the `BeforeMove` gate.

The deduction site sits in the same block as the `lastMove` record —
`BattleActions.runMove`, `sim/battle-actions.ts:282`, deduction first and
`moveUsed` second at 291:

    const willTryMove = this.battle.runEvent('BeforeMove', pokemon, target, move);
    if (!willTryMove) { ...; return; }              // no PP charged
    ...
    if (!pokemon.deductPP(baseMove, null, target) && move.id !== 'struggle') {
        this.battle.add('cant', pokemon, 'nopp', move); ...; return;
    }
    pokemon.moveUsed(move, targetLoc);

so the truth table is the same one as deviation 7: every immobilizer that returns
false from its `onBeforeMove` is free, and a move that misses, fails outright or
is blocked by Protect still pays, because the deduction precedes `useMove`.

**Pressure needs no separate treatment.** Showdown charges its extra point inside
`useMove` (`sim/battle-actions.ts:482`, via the `DeductPP` event), downstream of
the same gate, so an immobilized turn owes neither point and the engine's
combined 1-or-2 decrement moves as a single unit. gen3 has Pressure
(`data/mods/gen3/abilities.ts` inherits it and overrides only `onStart`).

**Impact.** Beyond the obvious PP drift, the phantom drain is load-bearing now
that Encore reads PP: `encore.onResidual` ends Encore the moment the locked move
hits 0, and `encore.onStart` refuses a target already at 0. Draining PP the sim
never spent frees a seat Showdown keeps locked.

**Disposition: PATCHED (2026-07-28).**
`third_party/poke-engine-gen3-pp-ordering.patch`, placed immediately ABOVE the
`last_used_move` record from deviation 7 (Showdown deducts first, records
second), below the still-asleep early return, and above `move_has_no_effect` and
the accuracy roll. `!choice.sleep_talk_move` reproduces Showdown's
`if (!externalMove)`: Sleep Talk pays for itself and the move it calls never
deducts.

**Still divergent (verified, left for its own change).** Showdown skips the
deduction entirely on a LOCKED continuation turn — `const lockedMove =
pokemon.getLockedMove(); if (!lockedMove) { ...deductPP... }` — so an Outrage /
Petal Dance / Thrash costs one PP for the whole lock. The engine charges every
turn. Confirmed empirically against the patched engine (a mid-lock turn still
emits `DecrementPP`). Distinct observable, needs its own pins and differential.

**Verification.** Real gen3 Showdown via `scripts/gen3_switch_differential.py
--only ppimmobilizedfree ppimmobilizedcontrol`; engine side pinned by
`rust/pokezero-search/tests/gen3_pp_ordering.rs` (14 tests).
## Confirmed deviation 9: PP charged on locked continuation turns

The tail of deviation 8, and the last known engine divergence. Showdown guards
the deduction on `getLockedMove()` (`sim/battle-actions.ts:280-283`):

    const lockedMove = pokemon.getLockedMove();
    if (!lockedMove) {
        if (!pokemon.deductPP(baseMove, null, target) && move.id !== 'struggle') { ... }
    } else {
        sourceEffect = this.dex.conditions.get('lockedmove');
    }

`getLockedMove` fires the `LockMove` priority event. Its providers in gen3's
chain are the `lockedmove` condition (Outrage / Thrash / Petal Dance,
`data/conditions.ts:282`), **`twoturnmove`** (Solar Beam, Sky Attack, Dig, Fly,
Razor Wind, Skull Bash, `data/conditions.ts:317`), `mustrecharge`
(`data/conditions.ts:377`), and `rollout` / `bide` in `data/moves.ts`. So a whole
lock costs ONE PP, charged on the turn that starts it — which is why Showdown
tags the second half of a two-turn move `[from] lockedmove`.

**Reachability changed the shape of this fix.** The `lockedmove` trio is
**absent from the gen3 randbats pool** — 0 carriers each for Outrage, Thrash and
Petal Dance (`data/random-battles/gen3/sets.json`). The reachable half of the
same bug is `twoturnmove`: **Solar Beam is on 4 species** (victreebel, exeggutor,
tangela, sunflora), and the engine charged **two PP per use instead of one**.
Hyper Beam (slaking) is the `mustrecharge` case and was already correct, because
the engine models the recharge turn as No Move and `Choices::NONE` returns before
the deduction. One guard covers all of them.

This also corrects an earlier, imprecise claim in this doc's deviation 8 entry:
the engine did not charge on *every* turn of an Outrage. It charged on the turn
that started the lock and on the first continuation; later turns often fell on a
confusion branch that (post-deviation 8) charges nothing. The divergence was two
PP where Showdown spends one, not N where Showdown spends one.

**PP exhaustion mid-lock, verified rather than assumed.** A two-turn move STARTED
on the last PP still completes, because the execute turn never consults PP again
— `getLockedMove()` short-circuits the whole deduction-and-abort block. Confirmed
against the real sim: 8 PP of Sky Attack yields eight complete uses, the eighth
beginning at 1 PP, and only then Struggle.

**Lock-end confusion does not interact with PP.** `lockedmove.onEnd` adds the
`confusion` volatile after the lock expires; that is a volatile application, not
a move attempt, so it reaches no deduction site.

**Disposition: PATCHED (2026-07-28).**
`third_party/poke-engine-gen3-lockedmove-pp.patch`. The charge-move execute turn
needs a local, because the engine clears the charge volatile before the deduction
runs; the LOCKEDMOVE volatile covers Outrage-class continuations, where it is
still present. Both are absent on the turn that starts the lock, so the first
turn pays exactly once and the next use pays again.

**Verification.** Real gen3 Showdown via `scripts/gen3_switch_differential.py
--only lockedmoveppdrain lockedmoveppcontrol`; engine side pinned by
`rust/pokezero-search/tests/gen3_lockedmove_pp.rs` (7 tests).

## Confirmed deviation 10: Solar Beam halved in clear weather

The damage half of the two-turn release gap (ledger J.4). The charge-state fix
made the release execute; this makes it deal the right number.

Showdown weakens Solar Beam in specific weathers only — `data/moves.ts`
`solarbeam.onBasePower`, inherited unchanged by gen3:

    const weakWeathers = ['raindance', 'primordialsea', 'sandstorm', 'hail', 'snowscape'];
    if (weakWeathers.includes(pokemon.effectiveWeather())) return this.chainModify(0.5);

Of those gen3 has rain, sand and hail. Clear weather is not in the list, and sun
is a separate mechanism: `solarbeam.onTryMove` skips the charge turn entirely
without touching power.

**Root cause: a self-comparison.** The engine asked
`state.weather_is_active(&state.weather.weather_type)`. That helper is
`self.weather.weather_type == argument && not suppressed by Air Lock / Cloud
Nine`, so passing the CURRENT weather back into it collapses to "weather is not
suppressed" — which is true in clear weather too, because `NONE == NONE`. Every
Solar Beam did half damage, in every branch.

The idiom is not wrong in general and is deliberately left alone at the three
other sites that use it — Morning Sun / Moonlight / Synthesis, the Chlorophyll /
Swift Swim speed boost, and `update_forecast` — each of which pairs it with a
`match` on the specific weather so `Weather::NONE` falls through harmlessly.
Only Solar Beam used it as "some weather is up".

**The 4x crit spread was a misdiagnosis, and is now pinned against.** The audit
measured Showdown −74 against engine −41 non-crit / −163 crit and inferred that
the non-crit branch was halving base power while the crit branch used full. It
was not: `calculate_damage` derives both branches from the same `choice`, so a
base-power error cannot separate them — it moves both by the same factor. The 4x
is **Light Screen**, which gen3 crits correctly ignore: the screen halves the
non-crit branch only, and 2x crit on top is exactly 4x. One root cause, not two.
−41 doubles to ~82, and Showdown's −74 is that at a 0.90 roll.

**Impact.** Every Solar Beam release in clear weather dealt half damage — the
common case, since sun is the only weather that changes the move's shape and
rain/sand/hail are the only ones that should weaken it. Reachable: Solar Beam is
on 4 gen3 randbats species.

**Disposition: PATCHED (2026-07-28).**
`third_party/poke-engine-gen3-solarbeam-weather.patch`. The `else if` now tests
`RAIN || SAND || HAIL` explicitly. The sun arm was already correct and is
untouched.

**Verification.** Real gen3 Showdown (Exeggutor into Blissey): clear −136, sand
−71, sun −131 with no charge turn. Pinned by
`rust/pokezero-search/tests/gen3_solarbeam_weather.rs` (6 tests, including the
2x/4x crit ratio in both directions and Air Lock suppression) and the
`solarbeamclear` / `solarbeamsand` / `solarbeamsun` differential scenarios.
