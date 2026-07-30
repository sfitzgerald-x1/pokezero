# Prediction — recorded BEFORE any implementation, and committed before it

Base: `8e3d3a8` (origin/main). Engine **36 patches**, fingerprint
`bdb6ad30f2722540` — matches the c12 decomposition's, and is UNCHANGED by this work
(parser + engine_world only, no vendored patch).
Baseline: `reports/c11_statfloor_differential.json`, **78 outside-limits**, 117 total,
`repros_complete: true` — so the identity diff is over a full population on both sides.

## Diagnoses, from replay (not from the one-row provenance in the backlog)

**W2 — toxic-stage staleness. The PARSER half is the stale one, not the world's.**
Replay of 1500243/79: Zapdos wakes from Rest-sleep, is Toxic'd fresh, and Showdown ticks
**-15** (stage 1 = 255/16). The engine ticks **-75** (stage 5).
`_materialization_toxic_stage` is a pure `max(0, stage-1)` of the parser's value, so the
world cannot be the origin — the parser was holding ~6.
Cause: `_update_toxic_stage` resets only on `-curestatus`/`-cureteam`. **Rest replaces
`tox` with `slp` through a plain `-status` line, which resets nothing**, so the ramp
survived a status it no longer belonged to.

**W1 — traced ability never reaches the world.**
Replay of 1500248/77: Flareon's Flamethrower into Porygon2 is absorbed
(`|-start|p1a: Porygon2|ability: Flash Fire`) — Showdown deals **no damage at all**. Every
engine branch deals **-111**, because the world's ability is `TRACE`, not the traced
`FLASHFIRE`.

This falsifies a documented assumption. `_SUPPORTED_VOLATILES` in `engine_world` says
`flashfire`'s volatile is boost-only and so "never wrong, at worst incomplete if a sampled
world lacked the ability, **which cannot happen for the mono-ability Gen 3 randbats carriers**".
A **Trace** user acquiring Flash Fire is exactly a world lacking the ability. The comment was
true of native carriers and silently wrong about acquired ones.

## Predicted clearance, by mechanism class (Z6.4 rule)

| item | rows | class | prediction |
| --- | --- | --- | --- |
| W2 | 1500243/79 | **magnitude** (tick size inside a correct branch) | **exactly 1**, no scatter |
| W1 | 1500248/77, /78 | **structural** (immunity changes whether damage happens) | **>= 2**, may scatter |

- floor: **3**
- outside-limits: 78 -> **75 or better**
- newly divergent: **0**

W1 is the case to watch: structural mechanisms scattered in both previous cycles (Encore
11 -> 16, Explosion 2 -> 4). If clearance exceeds 3, identity-diff the extras and check each
on the **unpatched** build before attributing them.

## Zero-change-elsewhere

Both fixes are inert where their signal is absent: the toxic reset only fires on a
`-status` naming a non-tox status, and the ability seeding only fires where the public
protocol has actually revealed an ability differing from the sampled one. The 5 adjudicated
`limit:roll_divergent_lethality` rows and the 3 `limit_not_established_keeps_label` rows
must be untouched.

## gen3 nuance carried from #962 (patch 32)

gen3 does **not** fire the copied ability's Start event on acquisition. The world must carry
the traced ability itself **without** simulating its on-switch-in activation — so this seeds
the ability field only, and adds no activation instruction or volatile.
