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

## Confirmed deviation 3: Attract volatile accepted but ignored

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
sleep → confusion → **attract** (the patch) → move. Consequences:

- **flinch**: a 100% "can't move" gate in BOTH sims; attract/confusion/par are
  never reached under a flinch. Exact.
- **confusion → attract**: same order as Showdown. A confused+attracted mon is
  exact in both probability AND reason attribution (50% confusion self-hit, then
  25% attract-immobilize, 25% move — identical to Showdown's
  confusion-before-attract resolution).
- **paralysis**: the engine resolves it *before* attract; Showdown resolves it
  *after*. The internal immobilized-reason split therefore differs (engine:
  25% par / 37.5% attract; Showdown: 50% attract / 12.5% par) but the net
  P(move) = 0.75 × 0.5 = **0.375** is identical (the two independent gates are
  commutative) and BOTH immobilized branches are empty-delta terminals, so the
  **leaf-state distribution is exact**. Only the rendered `|cant|` reason label
  (an events.rs concern) depends on which volatile is credited — invisible to
  search value.

**Source-leave decision (explicit).** Real Gen 3 Attract clears when the
infatuation source leaves play (`onUpdate` removes the volatile when
`effectState.source` goes inactive; Showdown emits a public
`|-end|mon|Attract|[silent]`). The engine does not track the infatuation
source, and adding source-identity tracking to the world state is out of scope
for a proc-gated singleton. **Chosen: a bounded in-search over-model** — attract
persists across the 2-3 ply search horizon even if the source switches out. This
is a small over-model strictly in the immobilizing (pessimistic) direction, the
exact opposite of the prior total no-op (which was optimistic), and it is bounded
to deep plies: the *live* attract state is always exact because the observation
parser clears the volatile on the real `-end`/switch (the volatile is seeded
verbatim from the public payload each decision). Only hypothetical in-search
lines where the source switches are affected, and there the pessimism is
self-limiting (the attracted mon usually cannot force the source out). This
matches the confusion / partial-trap no-expiry approximation class already
accepted in this engine (live state exact; in-search future duration a
pessimistic bound). The alternative — minimal source-tracking + clear-on-leave —
buys exactness on a singleton at the cost of new cross-side world state and a
parser-tracked source id; not justified now, revisit if post-fix sweeps show
attract above singleton rates.

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
