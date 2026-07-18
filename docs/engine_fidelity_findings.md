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

**Disposition: PATCHED (2026-07-18).**
`third_party/poke-engine-gen3-residual-order.patch`, applied by
`scripts/setup_poke_engine.sh`, moves the status-damage block after the
item/ability and Leech Seed blocks in `src/gen3/generate_instructions.rs`,
matching Showdown's gen3 residual orders (Leftovers 5, Leech Seed 8,
poison/toxic 9, burn 10 — the patch also corrects the latent
leech-seed-after-status ordering the differential had not yet exercised).
With the patched wheel the differential is **15/15 clean**; it is the
patch's regression gate.

## Confirmed engine contract 2: Hidden Power ids must be typed + base power

The gen3 engine move table only accepts fully-qualified ids
(`hiddenpowergrass70`); bare `hiddenpower` silently resolves as a weak
typeless hit, and the randbats set pool stores type-only ids
(`hiddenpowergrass`, 210 occurrences). The world constructor now translates
via `engine_world.hidden_power_engine_id` (type + BP derived from IVs, with a
fail-closed IV-consistency guard). This was a track-A bug found by track C —
without the differential it would have shipped as a silent damage-zeroing of
a very common move.

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
