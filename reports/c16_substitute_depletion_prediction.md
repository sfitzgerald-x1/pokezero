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

The patch will carry this three-way public state and its provenance from the
replay fold to `engine_world`. It will build a known-full Substitute exactly,
represent a known-broken Substitute as absent with zero HP, and accept the
comparison limit `world_substitute_health_unknown` only for an active
Substitute carrying the explicit, valid provenance value `unknown`. An absent,
malformed, broken, or arbitrary health provenance while the volatile is active
is a terminal instrumentation/provenance contradiction, not a comparison
limit and not a reason to rebuild the Substitute at full health.

Where chronology supplies a publicly deterministic Gen 3 fixed-damage hit,
the fold will instead carry cumulative exact depletion since Substitute
creation. Each sampled world will derive its own remaining HP as
`floor(sampled maxhp / 4) - depletion`; replay-scale absolute remaining HP is
not portable across sampled max-HP variants. The predicted supported cases are
Seismic Toss, Night Shade, Dragon Rage, and Sonic Boom; this is deliberately
limited to hits whose exact public damage can be derived without private state.
All other non-breaking Substitute damage remains explicitly unknown. A sampled
world whose initial Substitute could not have publicly survived the exact
depletion fails with a distinct incompatibility reason, never the accepted
unknown-health limit.

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
to the named public-information comparison limit, provided each has explicit
valid `unknown` health provenance. Rows with contradictory provenance must
instead terminate as instrumentation/provenance failures. This is not a
claimed engine-fidelity clearance: it deliberately declines worlds whose
Substitute HP cannot be known.

## Predicted Controls

1. A fresh Substitute immediately after its public `-start` line builds with
   exactly `floor(maxhp / 4)` HP.
2. A public `-end Substitute` leaves no Substitute volatile and builds with
   zero Substitute HP.
3. A live Substitute after a non-breaking public `-activate ... Substitute`
   receives the accepted limit only when its health provenance is explicitly
   and validly `unknown`.
4. A payload carrying an active Substitute with missing, malformed, `broken`,
   or arbitrary health provenance fails as a terminal contradiction and is not
   counted in any `limit:*` bucket.
5. A `|faint|` line clears Substitute and its health provenance before the
   force-switch snapshot is built.
6. Publicly deterministic Gen 3 fixed-damage chronology accumulates exact
   depletion for Seismic Toss, Night Shade, Dragon Rage, and Sonic Boom, while
   non-deterministic damage remains unknown.
7. A replay with max HP 387 and 50 exact depletion materializes a sampled
   max-HP-370 world at `floor(370 / 4) - 50 = 42` Substitute HP, not the
   replay-scale absolute remainder 46.
8. A sampled Substitute with initial HP less than or equal to exact depletion
   fails with a precise sampled-world incompatibility reason that differential
   accounting does not count as `limit:world_substitute_health_unknown`.
9. Non-Substitute construction and the pre-existing `volatile_unsupported`
   behavior when the opt-in flag is disabled remain unchanged.
