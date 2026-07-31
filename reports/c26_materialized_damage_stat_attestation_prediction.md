# C26 prediction: materialized damage-stat ownership

## Scope

This diagnostic investigates retained certification-tail rows labelled as status
or direct-damage magnitude mismatches. It does not change a production default
until a replay demonstrates one shared construction defect.

The target identities are `2800700/20`, `3301036/26`, `3401017/55`,
`3500021/19`, `3300207/69`, and, when retained evidence is available,
`3001000/57` and `3300122/21`.

## Prediction

For every materialized world that reaches a native engine state:

1. The Python `PokemonSpec` values for `attack`, `defense`,
   `special_attack`, and `special_defense` will exactly equal the corresponding
   fields in the constructed Rust `State`.
2. The active side's public boost stages will exactly equal the constructed
   Rust side boost fields.
3. Both sides and every party member will be attested; a match for only the
   active combatants is not sufficient.

The reason is structural: `engine_world._build_pokemon_spec` calculates the
five Gen 3 stats once, `PokemonSpec` carries those integers, and
`poke_engine_adapter._build_pokemon` forwards them verbatim. Boosts are a
separate `SideSpec` to `Side` mapping. A disagreement would therefore identify
a concrete Python-to-Rust construction defect. If all fields agree, this lane
will report that the outstanding magnitude ownership is downstream of that
seam rather than invent a stat patch.

## Evidence boundary

The historical target rows are not present in the locally retained public
reproduction archive at the time of this commit. The diagnostic must therefore
be run against a retained repro or a fresh replay before it can make a
row-level finding. Its tests prove only that it detects base-stat and
active-boost corruption at the materialization boundary.
