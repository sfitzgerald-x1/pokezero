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
**Disposition options:** patch the residual order in our poke-engine build
(we already carry a local patch mechanism from `setup_foulplay_eval.sh`), or
accept as a documented exemption with a value-impact estimate. Patch is
recommended: the fix is an ordering swap, and stall/residual accuracy is
exactly what a value leaf reads.

## Confirmed engine contract 2: Hidden Power ids must be typed + base power

The gen3 engine move table only accepts fully-qualified ids
(`hiddenpowergrass70`); bare `hiddenpower` silently resolves as a weak
typeless hit, and the randbats set pool stores type-only ids
(`hiddenpowergrass`, 210 occurrences). The world constructor now translates
via `engine_world.hidden_power_engine_id` (type + BP derived from IVs, with a
fail-closed IV-consistency guard). This was a track-A bug found by track C —
without the differential it would have shipped as a silent damage-zeroing of
a very common move.

## Harness notes

- Damage matching uses a ±16% band around the engine's representative
  (average) roll — wide enough for the 0.85–1.0 roll spread plus residual
  rounding, narrow enough that wrong-mechanic deltas (wrong residual
  fraction, wrong effectiveness) still diverge.
- Entry abilities (Sand Stream) fire before the fixture turn; such cases seed
  the engine state (`spec_weather`), mirroring what the world constructor
  does from the public payload mid-game.
- The unpatched upstream wheel is deliberate for measurement; the Rest/Sleep
  Talk PP-underflow patch from `setup_foulplay_eval.sh` should be re-verified
  by a dedicated case when multi-turn fixtures land.

## Next

Multi-turn curated cases (Sleep Talk, Baton Pass volatile transfer, Encore,
partial trapping), then the tier-2 real-game sweep: replay recorded
decision points through `engine_world` and check each observed Showdown
outcome lies in the engine's branch support — same matcher, production
constructor path.
