# C125 — the known-gaps ledger, pool-reachability filtered (C116 Phase 4 item 14)

C116 item 14: *"Known-gaps ledger, **filtered by pool reachability first**. The program is gen3
randbats only, so a mechanic absent from the pool cannot produce a boundary in any window …
Rule going forward: no entry joins the known-gaps ledger without a pool-reachability check
recorded next to it."*

This is that ledger. Every row carries its check and the instrument used.

> **On the C116 citation.** As in C121–C124: the plan is not in this repository, so it is
> provenance for why this exists, never evidence for a claim. Every check below is reproducible
> against the vendored Showdown at `data/random-battles/gen3/`.

## 1. Which instrument answers which question

Item 14 names `sets.json` as the instrument. That is right for **moves and abilities** and
wrong for **items**, because a gen3 `sets.json` entry carries no item field at all:

```json
{"level": 71, "sets": [{"role": "...", "movepool": [...], "abilities": [...], "preferredTypes": [...]}]}
```

Control — every one of these is plainly reachable and every one is absent from `sets.json`:
Leftovers, Choice Band, Thick Club, Sitrus Berry, White Herb. Items are assigned by
`teams.ts` `getItem()` (opens `:452`), by species and role. See `reports/c124_a6_is_knowable.md`;
applying the `sets.json` instrument to White Herb returns a false non-gap.

| question | instrument |
|---|---|
| is a **move** reachable? | `sets.json` → any set's `movepool` |
| is an **ability** reachable? | `sets.json` → any set's `abilities` |
| is an **item** reachable? | **`teams.ts` `getItem()`** — never `sets.json` |

## 2. Dropped as non-gaps — verified unreachable

Re-derived, not carried over:

| candidate | kind | instrument | species | verdict |
|---|---|---|---|---|
| `futuresight` | move | `sets.json` movepool | **0** | **not a gap** |
| `doomdesire` | move | `sets.json` movepool | **0** | **not a gap** |
| Dry Skin | ability | `sets.json` abilities | **0** | **not a gap** (gen4 ability) |

`futuresight` and `doomdesire` are the whole of gen3's delayed-damage class, so **residual order
11 is unreachable in this program** — not merely under-exercised. C115 carried Future Sight as a
live under-partitioning gap without applying this filter; it drops off.

## 3. Carried as reachable — real gaps

| candidate | kind | instrument | species | status |
|---|---|---|---|---|
| cross-side Leech Seed | move | `sets.json` movepool | **12** | **live, with rows** |
| Wish | move | `sets.json` movepool | **16** | ordered at 7 by #1066; watch |
| gen3 White Herb | item | **`teams.ts:471`** | **2** (deoxys, deoxysattack) | **live**, and *deterministic* |

**Cross-side Leech Seed** is the survivor item 14 predicted, and the current validation-holdout
residue contains exactly the shape it predicted — two of eleven rows:

- `19100014/35` — `component_missing_in_engine:leechseed`
- `19100193/46` — `component_mismatch:itemleftovers|leechseed`

Order 10.5 is cross-side and speed-major: the sap damages one side and heals the other, so the
heal is emitted at the victim's slot rather than the seeder's. Direction of error: the engine
under-attributes the cross-side heal, so a mislabel shows up as a *source* disagreement
(`itemleftovers` vs `leechseed`) rather than a magnitude one — which is exactly the shape of
`19100193/46`. Either fix the attribution or let Phase 2(a) enumeration absorb it; **not settled
here.**

**White Herb** is the strongest reachability in the table and the one the stated instrument
would have hidden: `teams.ts:471` returns it *unconditionally* for both Deoxys formes, so it is
not probabilistic at all. Disposition already decided as an engine fix in
`reports/c124_a6_is_knowable.md`.

## 4. Standing rule

No entry joins this ledger without a recorded pool-reachability check **and the instrument that
answered it**. A check run with the wrong instrument is worse than no check: it returns a
confident zero.
