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
| `tests/test_engine_gen3_abilities` | Ran 54, OK |
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

> **Provenance correction (second review of #1116, BLOCK 2), and it vindicates this section
> rather than retracting it.** The `before` artifacts first committed here were produced by the
> **first push of this branch** — engine fingerprint `91c2b785…`, the version carrying a
> reachable divide-by-zero — not by `main`. They were labelled `source_commit: 0af35e28`
> regardless, which is exactly the stale-engine failure mode
> `scripts/engine_build_fingerprint.py` exists to prevent.
>
> Both windows have now been re-run on `main`'s engine, fingerprint
> `761133828dd5c3cc…`, and on the corrected branch engine, `599c68a31e373472…`. The table
> below is that measurement. The numbers came out **identical to the ones originally
> claimed**.
>
> On the strength of the bad baseline I had briefly rewritten the PR body to say this patch was
> row-neutral and that "closes `19100113/62`" was withdrawn. **That withdrawal was itself
> wrong.** I had compared the fixed engine against the *broken patched* engine — which also
> closes the row — saw no difference, and concluded the row never closed. Correcting a claim
> against an unverified baseline is the same error as making one.

| window | boundaries | matched | diverged |
|---|---|---|---|
| dev — `main` `761133828d…` | 15,224 | 15,222 | 2 |
| dev — branch `599c68a31e…` | 15,224 | 15,222 | **2 (unchanged)** |
| validation holdout — `main` `761133828d…` | 15,396 | 15,388 | 8 |
| validation holdout — branch `599c68a31e…` | 15,396 | **15,389** | **7** |

Closed exactly `19100113/62`. **Nothing opened in either window.** Identity holds on all four
rows and `engine_errors` is 0 in all four. Artifacts: `reports/artifacts/c129_hitcount_{dev,holdout}_sweep.json`
are the `main` runs; `c129_hitcount_fix_{dev,holdout}_sweep.json` are the branch runs.

It closed despite a caveat I registered against it: the engine still applies **one roll to a whole
multi-hit move**, so it can only produce even totals, while Showdown rolled 128 and 121
independently for 249. I expected the row might narrow rather than close. The partition alone was
sufficient — but that shared-roll divergence is real, unfixed, and still unfiled.

Residue is now **dev 2 / holdout 7**.

## 6. Filed, not fixed here

Both surfaced by the second review of #1116 as non-blocking, and both are real.

**N1 — Case B's residual arms are the last basis mix in the file.**
`generate_instructions.rs:3610` and `:3683` compare a **per-hit** `max_damage_dealt` against a
**total**-basis `residual_threshold`, and `:3618` / `:3691` push that unconverted threshold as
per-hit damage. There is no subtraction on those paths, so neither the negative count nor the
`0/0` can occur, and the unscaled `floor(x) < t ⟹ x < t` proof still holds — which is why this
is not a crash. It is pre-existing on `main` and untouched here, but it is now the only
inconsistency left. Reproduced on this branch at `bonemerang attack=200 hp=150 maxhp=400 poison`:
the 5.2734 % arm is `Damage SideTwo: 100 | Damage SideTwo: 50` = 150 = **a KO priced as a
residual death**.

**N2 — the survive arm truncates.** `average_non_kill_damage / hit_count` (`:3509`, `:3555`,
`:3663`) is integer division, so the survive arm under-deals by up to `hit_count - 1` HP.

**N4 — no pin catches a floor that is too high.** The three pins here catch a floor that is too
low (the panic) and a floor removed entirely, but a floor of `0.90` instead of `0.85` passes all
three. The straddle band's bottom edge is unpinned.

**The multi-hit shared roll, still unfixed and now filed.** The engine applies **one** damage roll
to a whole multi-hit move, so it can only produce even totals for a two-hit move, while Showdown
rolls each hit independently — 128 and 121 for 249 on `19100113/62`. The KO partition closed that
row without addressing this, so the row closing is not evidence that the shared roll is correct.
