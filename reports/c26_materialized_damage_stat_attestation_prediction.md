# C26 prediction: bounded BattleSpec-to-native transport diagnostic

## Scope

This diagnostic is a reusable transport check for a *currently reproducible*
strict-matcher divergence. It takes a seed/step target, recreates its deterministic
action stream, and refuses to call the target transport-attested unless the same
boundary still diverges under the current build.

The public result records distinct constructed comparison states separately from
hidden-counter variants. If a support variant cannot be constructed, the result
is `dropped_variant_construction`; it is not zipped to another source spec and
does not receive a transport clearance.

## Prediction

For a current divergence whose native state was built successfully, each
`BattleSpec` field the adapter forwards will agree exactly with the native state:

- every reachable `PokemonSpec` member field: party identity, level, HP/max HP,
  five stored stats, current/base types, current/base ability, item, status,
  nature, gender, native-precision weight, Rest/sleep counters, move
  ID/PP/disabled slots, and the pre-Transform restoration snapshot;
- active party index, all seven active boost axes (including accuracy and
  evasion), active volatiles, transition flags/counters, and every exposed
  nonzero side condition on both sides; and
- weather, weather duration, terrain, and Trick Room.

Native enum display values are normalized with the adapter's production
lowercase-ID convention before comparison. Gender uses the adapter's compact
`M`/`F`/`N` to native-name mapping, the pre-Transform record is parsed into its
species/stats/four move slots, and weight is canonicalized to the binding's
`f32` representation. An extra native side condition is a mismatch, not an
ignored unknown.

## Falsification

A mismatch is a concrete adapter transport defect candidate. A clean result
rules out only that transport seam for the observed `BattleSpec`; it leaves
belief-world derivation, native branch generation, and Gen 3 damage arithmetic
open. No historical row-specific conclusion is claimed without a current JSON
artifact carrying reproducible target and build provenance.

The JSON provenance includes the exact command, public source commit, a
deterministic public-source tree hash, the two-consumer engine build
fingerprint, and the content-resolved Showdown/randbat source hash and paths.
When Git metadata is available, the recorded commit must equal checkout `HEAD`
and tracked files must be clean. A runtime image without Git metadata records
an explicit hash of the executable Python source closure instead.
