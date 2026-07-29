# Prediction, recorded BEFORE the 300-game differential (Kecleon typechange seeding)

Base: fb1105c (origin/main merged, post-#959). Engine **34 patches**, fingerprint
5b29e611468d3baa930984d5b8557280835e72f1ce38d8dc3c6b183e15c344dc — UNCHANGED by this PR
(parser/engine_world only, no vendored patch).
Population: seeds 1500000-1500299, strict matcher. Post-#959 baseline: **92 outside-limits**
(independently confirmed by reports/c10_explosion_differential.json).

## Named rows (the floor)

The four `kecleon_typechange_world_drift` rows, all verified individually against this build
(each -1 divergence at an identical measured boundary count):

  * **1500074/12** — Kecleon's Return: Showdown 27 (no STAB), engine 43. ATTACKING side.
  * **1500074/32**
  * **1500191/20** — Hidden Power Ice into Kecleon: Showdown `-resisted` 16, engine 34.
    DEFENDING side.
  * **1500204/83**

## Predicted: >= 4, and I expect MORE

Predicting from the family label would say exactly 4. **Both of the last two fixes
under-counted their own family that way** (Encore: 11 labelled, 16 full-frame; Explosion: 2
labelled, 4 actual), because a mechanism that changes which branch is right surfaces in
whatever class the matcher lands in once the right branch is missing.

So the prediction is by **mechanism signature** instead: any boundary where the active mon
carries a live `typechange` the world did not reproduce. Concretely — any Kecleon boundary
after a Color Change, on either side of the damage calculation.

  * floor: **4** (the named rows)
  * outside-limits: 92 -> **88 or better**
  * newly divergent: **0**

**If clearance exceeds 4, identity-diff the extras and check each on the unpatched build
rather than calling them noise.** That is how 1500285/14, 1500188/57 and 1500286/38 were
attributed.

## Zero-change-elsewhere

`_apply_live_typechange` is a no-op for every side whose payload carries no `type:` override
(pinned four ways: absent key, empty string, empty sides, empty payload), and it consumes only
the `type:` form — `forme:` stays with `_apply_forecast_types`. So the 5 adjudicated
`limit:roll_divergent_lethality` rows and the 3 `limit_not_established_keeps_label` rows
must be untouched.

## Retention

Run with `--repros-per-game 40 --keep-repro 500` so the committed artifact carries the FULL
divergent set. The report now records `repro_retention` with `repros_complete`, so this is
verifiable from the artifact instead of asserted here — the gap this PR also closes for #959.
