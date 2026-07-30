# Prediction — recorded BEFORE implementation, committed before it

Base `05c4624` (origin/main). Engine **41 patches**, fingerprint `3204c777dec347aa` —
UNCHANGED by this work (parser + engine_world only, no vendored patch).
Population: the certification sweep's retained rows (seeds 2000000-2701249, 8 shards,
3821 rows, all `repros_complete`), re-read through the fixed build.

## Derived semantics (gen3 OWNS Truant — full override, trap direction 3)

`data/mods/gen3/abilities.ts` replaces base's volatile machinery outright
(`onStart: undefined`):

```js
onSwitchIn(pokemon) { pokemon.truantTurn = this.turn !== 0; }
onBeforeMove(pokemon) { if (pokemon.truantTurn) { add('cant', 'ability: Truant'); return false; } }
onResidualOrder: 27,
onResidual(pokemon) { pokemon.truantTurn = !pokemon.truantTurn; }
```

Three consequences the current implementation does not have:

1. **The bit is a FREE-RUNNING TOGGLE.** `onResidual` flips it every turn
   **unconditionally** — whether the mon moved, slept, flinched, was paralyzed,
   recharged, or did nothing at all.
2. **A fresh switch-in has a KNOWN phase, not an unknown one.**
   `truantTurn = (this.turn !== 0)` is exactly the compensation for the extra residual
   flip a mid-battle switch-in experiences before its first move opportunity. Both a
   turn-0 lead and a mid-battle switch-in therefore **ACT on their first move turn**.
   This answers the open question in the brief: "never acted" is not unknowable.
3. **Genuinely unknown = we never saw the switch-in** (truncated prefix). That is the
   only unknowable case, and it must stay distinct from "known to be acting".

## Why the current derivation drifts

`engine_search._truant_loaf_slots` uses **"publicly MOVED last round -> loafs now"**.
That is a proxy for the parity bit, not the bit. The correspondence breaks the first time
the mon fails to move for a **non-Truant** reason — sleep, paralysis, flinch, freeze,
recharge, or switching — because the sim flips the bit anyway and the proxy does not.
**Once broken the parity stays inverted for the rest of the stint**, which is why a
single mechanism produces tens of rows.

## Row identities (extracted by signature; the brief's count is 48)

Filtering the sweep's 304 unattributed rows for the loaf-parity signature finds **44**:

* **38** where Showdown's Truant holder ATTACKED and the engine's branch produced
  `engine=[]` (the engine loafed) — e.g. 2000054/49, 2000059/11, 2000393/28;
* **6** where Showdown emitted `|cant|...|ability: Truant` and the engine attacked —
  e.g. 2200291/42, 2201093/61, 2400315/29.

Two replayed at base, one per direction, both confirming:
`RemoveVolatileStatus SideOne: TRUANT` where Showdown dealt 182, and an engine attack
that KO'd where Showdown loafed.

**44 != 48**, and I am not reconciling that by assumption. My filter keys on
`engine=[]` in the majority miss and on the `|cant|` line; rows whose miss is worded
differently, or where the loaf changes a residual rather than a damage component, will be
missed. **The re-read is the count that matters**, and the gap is the first thing to
explain if it persists.

Note two of the six Case-A rows are **Porygon2**, not Slaking — a **traced** Truant. The
holder set must therefore include traced abilities, which is why this fix depends on the
`traced_ability` tracking added in #967.

## Predicted clearance

* floor: **44** (the identified rows), expected **48** if the brief's count is right
* newly divergent: **0**
* zero change to the documented limit classes and to any non-Truant family

Clearance signature is **structural** (a loaf changes whether the move happens at all), so
per the Z6.4 rule these rows may be scattered across matcher classes — they already are:
the 44 span `roll_scaled_component`, `component_missing_in_engine:itemleftovers`,
`component_missing_in_engine:sandstorm` and others. Under-attribution is therefore likely
and the identity diff, not the count, is the instrument.
