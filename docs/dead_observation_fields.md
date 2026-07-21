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

**Total: 12 dead numeric columns** — indices `{24, 25, 35, 36, 48, 49, 50, 51, 52, 53, 54, 55}`.

## Not dead (reachable — for contrast)

Superficially similar mechanics that **are** reachable, so their columns are
live. Do not confuse them with the table above:

| Mechanic | Live column(s) | Reachability |
| --- | --- | --- |
| Wish | `self/opp_wish_pending` 56 / 57, `self/opp_wish_turns` 166 / 167 | 16 pool carriers |
| Mean Look / Spider Web trap | `meanlook_trap` 165, `trapper_alive` 62 | reachable (landed #816) |
| Wrap (partial trap) | `wrap_trap_turns` 162 | Shuckle (sole carrier) |
| Weather turns | `weather_turns` 46, `weather_permanent` 47 | Rain Dance / Sunny Day + Sand Stream / Drought / Drizzle |

## Maintenance

Re-verify on any change to the gen3 randbats pool (`sets.json`). If a listed
move gains a carrier, delete its row here and confirm both the encoder and the
search engine handle the now-reachable state (for Future Sight, that includes
removing the `future_sight_pending` engine guard). Companion to the schema-freeze
audit ([plan](silent_noop_sweep_plan.md)).
