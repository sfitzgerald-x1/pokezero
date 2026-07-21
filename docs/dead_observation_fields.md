# Dead observation fields (gen3 randbats)

**What this tracks.** Observation-tensor columns that encode a mechanic which is
**unreachable** in the current gen3 random-battle pool, so the column is
structurally always `0` ("dead"). This is the *inverse* of the silent no-op
sweep ([plan](silent_noop_sweep_plan.md)): that audit hunts for **missing**
encoding of **reachable** states; this doc records **spare** encoding of
**unreachable** ones.

Dead fields are **not bugs** — a column that is always `0` is harmless. They are
tracked because:

- they can be **reclaimed** at a future schema bump (v4+) instead of growing the
  tensor, and
- each row pins an explicit **reachability assumption**. If the pool ever gains
  one of these moves, the field goes live and the row must be revisited (and the
  encoder/engine re-checked for the now-reachable state).

**Reachability bar (same as the sweep).** A mechanic is dead iff its trigger has
**0 carriers** in `data/random-battles/gen3/sets.json` *and* there is no indirect
path. The only move-caller in the pool is Sleep Talk (40 carriers), and Sleep
Talk can only call a move the user already knows — so it cannot reach any move
with 0 direct carriers. There is no Metronome / Assist / Mirror Move / Nature
Power in the pool. Verified against the vendored gen3 pool (125 distinct moves).

## Dead numeric columns

Indices are into the per-token numeric vector in `src/pokezero/showdown.py`.
All listed columns are v2.x base columns (they predate the reachable-only
encoding discipline) and are carried into v2.1 / v2.2 / v3 unchanged.

| Mechanic | Trigger move(s) | Pool carriers | Dead column(s) | Notes |
| --- | --- | ---: | --- | --- |
| **Reflect** (screen) | Reflect | 0 | `NUMERIC_SELF_SCREENS` 24, `NUMERIC_OPP_SCREENS` 25 (count, shared w/ Light Screen); `NUMERIC_SELF_REFLECT_TURNS` 48, `NUMERIC_OPP_REFLECT_TURNS` 52 | No screen move in the pool at all. |
| **Light Screen** (screen) | Light Screen | 0 | screens 24 / 25 (shared count); `NUMERIC_SELF_LIGHT_SCREEN_TURNS` 49, `NUMERIC_OPP_LIGHT_SCREEN_TURNS` 53 | — |
| **Safeguard** | Safeguard | 0 | `NUMERIC_SELF_SAFEGUARD_TURNS` 50, `NUMERIC_OPP_SAFEGUARD_TURNS` 54 | — |
| **Mist** | Mist | 0 | `NUMERIC_SELF_MIST_TURNS` 51, `NUMERIC_OPP_MIST_TURNS` 55 | — |
| **Future Sight / Doom Desire** (delayed attack) | Future Sight, Doom Desire | 0 | `NUMERIC_SELF_FUTURE_SIGHT` 35, `NUMERIC_OPP_FUTURE_SIGHT` 36 | Engine **also refuses** this state: `EngineWorldUnsupported("future_sight_pending")` (`engine_world.py:479`) — safe only while the mechanic is unreachable. |
| **Hail** (weather) | Hail | 0 (+ no gen3 hail ability) | opponent weather-reveal hail pair (offset `NUMERIC_STAT_WEATHER_REVEAL_OFFSET` 97, `_WEATHER_REVEAL_ORDER` index 3): **103** (set-this-game bit), **104** (source-was-ability bit) | Hail is the one weather in `_WEATHER_REVEAL_ORDER` with **neither a move carrier nor a gen3 ability** (Snow Warning is gen4). The rain/sun/sand reveal pairs (97–102) are all live — via the Rain Dance / Sunny Day / (ability-only) Sandstorm sources and the Drizzle / Drought / Sand Stream abilities. |

**Total: 14 dead numeric columns** — indices `{24, 25, 35, 36, 48, 49, 50, 51, 52, 53, 54, 55, 103, 104}`.

## Swept and confirmed live (checked this pass — reachable, NOT dead)

Every mechanic-gated column was audited for the same "dead column" pattern.
The following were confirmed reachable, so their columns stay — several were
close calls where a naive read would have wrongly flagged them dead (noted):

| Mechanic (column) | Pool evidence | Note |
| --- | --- | --- |
| Wish (56 / 57, 166 / 167) | 16 carriers | — |
| Mean Look / Spider Web trap (`meanlook_trap` 165) | Mean Look 1 + Spider Web 2 | move-based switch-lock |
| Trap abilities (`trapper_alive` 62) | Shadow Tag 1 + Arena Trap 1 + Magnet Pull 3 | ability-based, distinct from Mean Look |
| Wrap partial-trap (`wrap_trap_turns` 162) | Wrap 1 (Shuckle) | — |
| Pursuit intercept (95 / 112 / 147) | 3 carriers | — |
| Substitute HP (`sub_hp_fraction` 137) | 87 carriers | — |
| Multi-hit `n_hits` (106 / 141) | Bonemerang (Marowak), 2-hit | **close call** — no 2–5-hit spread move is in the pool, but Bonemerang keeps it live |
| Choice-lock `cb` (119 / 138 / 150) | Choice Band | **close call** — not in `sets.json`; the gen3 item generator (`teams.ts`) assigns it to Trick users / physical attackers |
| Sleep Talk `called` (107) | 40 carriers | — |
| Transform (108 / 143) | Ditto in pool | — |
| Encore turns (161) | 16 carriers | — |
| Confusion turns / self-hit (160 / 168) | Signal Beam etc. | — |
| Consecutive stall (`stall_counter` 159) | Protect 55 + Endure 4 | Detect specifically = 0 carriers, but Protect/Endure keep the column live |
| Toxic stage (37) | 152 carriers | — |
| Spikes hazards (22 / 23, 113 / 114) | 15 carriers | — |
| Rain / Sun / Sand weather reveal (97–102), `weather_turns` 46, `weather_permanent` 47 | Rain Dance 7 / Sunny Day 4 moves + Drizzle / Drought / Sand Stream abilities | Sandstorm *move* = 0 carriers, but Sand Stream (Tyranitar) makes the sand pair live |

## Maintenance

Re-verify on any change to the gen3 randbats pool (`sets.json`). If a listed
move gains a carrier, delete its row here and confirm both the encoder and the
search engine handle the now-reachable state (for Future Sight, that includes
removing the `future_sight_pending` engine guard). Companion to the schema-freeze
audit ([plan](silent_noop_sweep_plan.md)).
