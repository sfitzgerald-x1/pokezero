# Bench Rest-provenance — investigation state, handed off mid-diagnosis

Base: `d245811` (origin/main, post-#972). Engine untouched. **No fix implemented.**
Handed off at the context ceiling per the established pattern; everything below is
probe-verified, not inferred (the binding rule from Z14.3).

## What the brief expected, and what the probes say

The brief's hypothesis: *"the tracking keys on the active slot or gets dropped at
switch-out"*. **Both halves of that are false.** Three of the four links in the chain were
tested directly and all three work:

| link | probe | result |
| --- | --- | --- |
| parser survives benching | feed Rest, bench, switch back | `{'p1:entei': 0}` persists throughout — **no switch-out drop** |
| stamping reaches benched rows | `_apply_rest_sleep_provenance` on a benched row | benched Entei row gets `restSleepAttempts = 0`; the active row correctly gets none |
| world arithmetic | `_rest_turns_from_row(stamped)` | returns **3** (= 3 − k), and `None` when unstamped |

The parser's drop conditions are `-curestatus slp`, `faint`, `-cureteam` and the
sleep-usable-move clause. **None of them is a switch**, which is why benching survives.

## Where the gap must therefore be

Everything upstream of world construction is correct **when the row carries the `slp`
condition and the species key matches**. That leaves exactly one candidate region:

**Opponent-side row construction.** Self-side rows come from the request
(`_request_materialization_rows`) and carry every benched mon with its condition; opponent
rows come from `[_pokemon_materialization_row(p) for p in replay.public_revealed[player]]`.
The open questions, in the order I would test them:

1. Does a benched OPPONENT row carry the live `slp` condition, or a stale/absent one? If the
   status is missing, `_hp_and_status` never reaches `_rest_turns_from_row` and the mon is
   seeded `rest_turns = 0` — which is exactly the reported symptom.
2. `_hp_and_status` takes an **`is_self`** flag; the self and opponent paths differ and the
   failing row was on a specific side. That flag is the first thing to vary in a repro.
3. Species-key normalization between `public_revealed` rows and
   `_induced_sleep_victim_key`'s ident-derived key. The docstring already flags that these
   coincide only under Nickname Clause.

**Do not re-test links 1-3 of the first table.** They are settled.

## gen3 bench-sleep semantics, derived for the required pin

From `data/mods/gen3/conditions.ts`:

* `time` decrements **only in `onBeforeMove`** — a move ATTEMPT by the ACTIVE mon. There is
  no residual or per-turn tick, so **sleep does NOT tick on the bench**. Confirmed; pin the
  non-ticking.
* gen3 additionally carries **`skippedTime`**: turns spent using Sleep Talk / Snore
  immediately before switching out while asleep are accumulated and **added back to `time`
  on switch-in** (`onSwitchIn`). So gen3 does not merely fail to tick on the bench — it
  actively refunds sleep-usable turns taken just before benching. Any bench-survival pin
  should cover this, and the parser's existing sleep-usable drop clause is adjacent to it.

## Suggested pins (not written)

bench-survival, switch-back, non-ticking-on-bench, and the `skippedTime` refund — the last
being the one most likely to be got wrong, since it is a gen3-only compensation.

## Second assigned item, not started

**Substitute depletion tracking** (~12 rows, the recoil family's inferable subset) is
untouched. Its shape is already established in Z14.1: `engine_world` seeds
`substitute_health = maxhp // 4` with no depletion tracking, the engine's clamp is sim-exact,
and part of the family is a genuine limit because Showdown publishes the `[damage]` activation
on non-breaking hits **without the amount**. The pre-registered split that assignment asks for
should therefore separate *known-full at creation* and *known-broken at `-end`* from the
unpublished middle.
