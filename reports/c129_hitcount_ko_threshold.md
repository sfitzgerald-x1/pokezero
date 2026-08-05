# C129 — the KO partition threshold was hit-count-blind

Found by the independent review of #1098, which rejected my own explanation of why
`19100113/62`'s non-crit arm never partitioned and supplied a better one. This is the fix.

## 1. The defect

Both KO partitions compared a **per-hit** maximum against the defender's **full** HP, with no
`hit_count` scaling anywhere:

- Case A, `gen3/generate_instructions.rs`: `max_damage_dealt >= defender_active.hp && min_damage_dealt < defender_active.hp`
- Case B (crit): the same shape on `max_crit_damage` / `min_crit_damage`

`hit_count` is computed near the top of the move path and consumed only where the hits are
actually applied. **Neither partition referenced it.** So a multi-hit move whose single hit
cannot KO never partitioned, even when its hits together can.

Bonemerang on `19100113/62`: per-hit max **140** against **253** HP. `140 < 253`, so the
non-crit arm collapsed to one roll — but two hits reach **280**, and rolls 85–90 total 238–252
and *survive* while 91–100 total 254–280 and *kill*. That straddle is exactly what the partition
exists to represent, and it was invisible.

## 2. The fix, and the half-fix it took two attempts to get past

Scale both partition tests by `hit_count`, gated on `hit_count > 1` so every single-hit move is
bit-identical.

**That alone was not enough, and the failure is worth recording because it looked like success.**
`compare_health_with_damage_multiples` was then handed the scaled *total*, so it returned a
*total* — and `run_move` applies `damage_amount` **once per hit**. Feeding the total straight back
dealt it `hit_count` times:

| stage | branches produced |
|---|---|
| before any fix | `80 \| 80` = 160, one arm, always survives |
| threshold scaled only | `155 \| 10` = 165 → **kills**, and `165` → kills. Partition fired 52.73/37.27 and *both arms were lethal* |
| output converted back to per-hit | `77 \| 77` = 154 **survives**, and `165` **kills** |

The middle row is the trap: masses moved, the split appeared, and the fix accomplished nothing.
Only printing the damage numbers showed it. Both Case A's `survive_representative` and Case B's
`average_non_kill_crit_damage` needed the total→per-hit conversion.

## 3. Red run (M3)

The pin also took two attempts, and the first was vacuous. At a defender HP of 120,
`2 × min = 146` already kills, so **every roll kills with or without the fix** — it passed
against the unfixed engine. The straddle band for this matchup is
`2 × 0.85 × 87 = 148 < hp ≤ 174`, so the pin now uses **165** and asserts the band explicitly
(`per_hit < hp`, `2 × per_hit ≥ hp`, `2 × per_hit_min < hp`) so a future resize cannot silently
leave the band.

Red, before the fix: `non-crit mass did not partition: totals=[160]`. The crit arm's `155+10`
split was visible in the same listing — the control showing the machinery works when a threshold
*is* straddled, and that only the non-crit test was blind.

## 4. Gates

| gate | result |
|---|---|
| `tests/test_poke_engine_patch_stack` | Ran 4, OK |
| `tests/test_engine_gen3_abilities` | Ran 51, OK |
| `tests/test_branch_mass_reconstruction` (mass gate) | Ran 5, OK |
| `tests/test_crit_kill_split_patch` | Ran 8, OK |
| `tests/test_a1_residuals_already_ran` | Ran 13, OK |
| `scripts/engine_behavioral_probes.py` | exit 0, all PASS |

The mass gate matters here specifically: this change moves probability mass between branches, and
that gate reconstructs branch masses with arithmetic the engine does not share.

## 5. Sweep

Prediction registered before the results were read: **holdout 8 → 7** closing `19100113/62`,
**dev 2 unchanged** (neither dev row is multi-hit — Sacred Fire and Hidden Power are single-hit),
nothing opened.

| window | boundaries | matched | diverged |
|---|---|---|---|
| dev — before | 15,224 | 15,222 | 2 |
| dev — after | 15,224 | 15,222 | **2 (unchanged)** |
| validation holdout — before | 15,396 | 15,388 | 8 |
| validation holdout — after | 15,396 | **15,389** | **7** |

Closed exactly `19100113/62`. **Nothing opened in either window.** Identity holds on all four rows.

It closed despite a caveat I registered against it: the engine still applies **one roll to a whole
multi-hit move**, so it can only produce even totals, while Showdown rolled 128 and 121
independently for 249. I expected the row might narrow rather than close. The partition alone was
sufficient — but that shared-roll divergence is real, unfixed, and still unfiled.

Residue is now **dev 2 / holdout 7**.
