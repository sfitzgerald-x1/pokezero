# C26 result: bounded BattleSpec-to-native transport diagnostic

## Evidence Boundary

This artifact documents the diagnostic's scope, not a retained-row clearance.
The historical row identities previously listed here cannot be authoritatively
recovered from a tracked reproduction archive on this branch. They are therefore
not evidence that any current certification-tail boundary still diverges, nor
that any particular row has passed transport attestation.

When run, `scripts/attest_materialized_damage_stats.py` writes a machine-readable
JSON artifact containing its exact command, current source commit, engine
fingerprint/image provenance, target boundary verdict, candidate construction
status, and one attestation per constructed comparison state. A target is
eligible for a transport statement only when its current strict matcher verdict
is `diverged`; a matched, skipped, or construction-dropped target is reported as
ineligible rather than silently cleared.

`comparison_states` counts the constructed native states actually compared.
`hidden_counter_candidate_worlds` separately records the support-world count
when hidden-counter recovery was in use; it is zero for an exact world.

## What A Passing Result Establishes

For each constructed candidate world, the adapter forwarded the selected
`BattleSpec` values faithfully into the native `State`, including party stats,
level, HP, current/base types and abilities, item/status, move IDs/PP, active
index, boost stages, weather, and all exposed nonzero side conditions.

It does **not** establish either of the following:

- The belief-world builder derived the correct `BattleSpec` values from public
  and hidden game information.
- The native engine generated correct branches or applied correct Gen 3 damage
  arithmetic after state construction.

Accordingly this diagnostic can narrow a surviving divergence away from the
adapter transport seam only. It cannot clear a retained divergence's world
derivation or arithmetic owner.
