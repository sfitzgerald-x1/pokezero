# C124 — A6 item 11 decision: the item IS knowable, so A6 is an engine fix, not a limit

C116 Phase 3 item 11 asks a question before it asks for code:

> "**A6** — determine, from the harness's construction path, whether the defender's item is
> knowable at the boundary. If yes: implement gen3 `WHITEHERB` + fix the `UNKNOWNITEM`
> construction; if no: write the program's **first demonstrated limit**, to the M1 standard."

This report answers it. **Yes — knowable, deterministically, from species alone.** A6 takes the
engine-fix branch and is not the program's first demonstrated limit.

## 1. The answer

`data/random-battles/gen3/teams.ts:471`, inside `getItem()` (opens `:452`), in the block the
file itself labels "First, the high-priority items" — i.e. before any randomness:

```ts
if (species.id === 'deoxys' || species.id === 'deoxysattack') return 'White Herb';
```

Unconditional. Every gen3 randbats Deoxys and Deoxys-Attack holds White Herb, 100 % of the
time. The harness does not need to *observe* the item to know it; the species determines it.

That settles item 11 on the "yes" branch: implement gen3 `WHITEHERB`, and fix the
`UNKNOWNITEM` construction so a Deoxys is built holding it. C111 §A6 recorded the row's
mechanism already — Showdown restores the Superpower Defence drop with
`-enditem|p1a: Deoxys|White Herb` before Rock Slide lands, which moves the roll band from
"every roll kills" to "no roll kills" — so the remaining work is implementation, not diagnosis.

## 2. A methodological correction the plan's item 14 needs

Getting here nearly produced the opposite answer, and the near-miss generalises.

C116 item 14 introduces a pool-reachability filter and states it as: *"Verified against
`data/random-battles/gen3/sets.json`"*, using it to drop `futuresight`, `doomdesire` and Dry
Skin as non-gaps. Applying that same rule to White Herb returns **0 species**, which would have
made A6 a non-gap and closed it for the wrong reason.

**`sets.json` does not contain items at all.** A gen3 entry is exactly:

```json
{"level": 71, "sets": [{"role": "...", "movepool": [...], "abilities": [...], "preferredTypes": [...]}]}
```

Control, run over the real file — every one of these is obviously reachable and every one
returns `False`:

| item | present in `sets.json` |
|---|---|
| Leftovers | **False** |
| Choice Band | **False** |
| Thick Club | **False** |
| Sitrus Berry | **False** |
| White Herb | **False** |

Leftovers appears in the recorded protocols constantly. A filter that calls it unreachable is
not measuring reachability.

**So the rule is sound for moves and abilities — which is what item 14 actually used it on,
`futuresight`/`doomdesire` being moves and Dry Skin an ability — and invalid for items.** Item
reachability must be read from `teams.ts` `getItem()`, where items are assigned by species and
role.

Recording it because item 14's standing instruction is "no entry joins the known-gaps ledger
without a pool-reachability check recorded next to it". Applied to an item with the stated
instrument, that check returns a false non-gap every time.

## 3. What this does not claim

No engine change is made here and none is implied to be done. The White Herb implementation,
the `UNKNOWNITEM` construction fix, its red-run pin and its two-window sweep are the work item
11 calls for; this report only takes the fork item 11 puts first, and removes a way of taking
it wrongly.

The item-14 ledger entries the plan already settled — `futuresight` 0 species, `doomdesire` 0,
Dry Skin 0 — are **unaffected**: all three are moves or abilities, which `sets.json` does carry,
so those checks were run with the right instrument and stand.
