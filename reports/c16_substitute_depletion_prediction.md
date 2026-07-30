# C16 Substitute Depletion: Pre-registered Prediction

## Scope

This lane corrects only the public-state construction of an active Substitute.
It does not alter the battle engine's Substitute or recoil mechanics. Appendix
Z14.1 of the divergence ledger established that the engine already computes
recoil from the Substitute-clamped damage exactly.

The protocol gives three relevant facts:

1. `|-start|...|Substitute` proves that a newly-created Substitute has
   `floor(maxhp / 4)` HP.
2. `|-end|...|Substitute` proves that no Substitute remains.
3. A non-breaking `|-activate|...|Substitute|[damage]` proves that the
   Substitute was hit but does not publish the absorbed amount. Its remaining
   HP is therefore not reconstructable from public information.

The patch will carry this three-way public state from the replay fold to
`engine_world`. It will build a known-full Substitute exactly, represent a
known-broken Substitute as absent with zero HP, and reject an active
intermediate/unknown Substitute with the stable fail-closed reason
`substitute_health_unknown`. The differential harness must record that reason
as the existing `limit:*` comparison family, not silently rebuild the
Substitute at full health.

## Predicted Affected Certification Rows

The following 12 rows are the complete `CAND_recoil_vs_substitute_basis`
population in `reports/c14_cert_sweep_readout.json`:

| Identity | Current classification | Predicted post-patch result |
| --- | --- | --- |
| 2000031/60 | `CAND_recoil_vs_substitute_basis` | `limit:world_substitute_health_unknown` |
| 2100227/35 | `CAND_recoil_vs_substitute_basis` | `limit:world_substitute_health_unknown` |
| 2100568/48 | `CAND_recoil_vs_substitute_basis` | `limit:world_substitute_health_unknown` |
| 2101125/47 | `CAND_recoil_vs_substitute_basis` | `limit:world_substitute_health_unknown` |
| 2200232/51 | `CAND_recoil_vs_substitute_basis` | `limit:world_substitute_health_unknown` |
| 2400009/83 | `CAND_recoil_vs_substitute_basis` | `limit:world_substitute_health_unknown` |
| 2400484/36 | `CAND_recoil_vs_substitute_basis` | `limit:world_substitute_health_unknown` |
| 2400719/68 | `CAND_recoil_vs_substitute_basis` | `limit:world_substitute_health_unknown` |
| 2401166/75 | `CAND_recoil_vs_substitute_basis` | `limit:world_substitute_health_unknown` |
| 2600222/44 | `CAND_recoil_vs_substitute_basis` | `limit:world_substitute_health_unknown` |
| 2601023/51 | `CAND_recoil_vs_substitute_basis` | `limit:world_substitute_health_unknown` |
| 2701047/47 | `CAND_recoil_vs_substitute_basis` | `limit:world_substitute_health_unknown` |

Expected count: **12** reclassified from an unattributed world approximation
to the named public-information comparison limit. This is not a claimed
engine-fidelity clearance: it deliberately declines worlds whose Substitute
HP cannot be known.

## Predicted Controls

1. A fresh Substitute immediately after its public `-start` line builds with
   exactly `floor(maxhp / 4)` HP.
2. A public `-end Substitute` leaves no Substitute volatile and builds with
   zero Substitute HP.
3. A live Substitute after a non-breaking public `-activate ... Substitute`
   does not build, even when the old approximation flag is enabled.
4. A payload carrying an active Substitute but no explicit public health state
   does not inherit the old full-health approximation; it fails closed.
5. Non-Substitute construction and the pre-existing `volatile_unsupported`
   behavior when the opt-in flag is disabled remain unchanged.
