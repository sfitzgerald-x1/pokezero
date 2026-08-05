# C123 — A9 is misfiled: the Wish heal is not unrendered, it is never generated

Row `19100113/62`, the sole A9 row, is filed in `reports/c117_validation_holdout_baseline.md`
as an **unrendered Wish heal** — a renderer defect — and `reports/c119_phase2_scoping.md`
counts it as "no" for Phase 2 on that basis. Both are wrong, and the correction moves the row
from the renderer to the Phase 2 decision.

> **On the C116 citation.** As in C121/C122: the C116 plan is not in this repository, so it is
> provenance for why work was queued and never evidence. Everything below is measured.

## 1. The measurement

The recorded engine state, replayed through the installed engine with the row's own choices
(`p1 seismictoss`, `p2 bonemerang`):

| mass | p1 damage | heal | 253 HP → | outcome |
|---|---|---|---|---|
| 10.0000 % | 0 | 0 | 253 | Bonemerang missed |
| 84.3750 % | 253 | 0 | **0** | dead |
| 3.8672 % | 253 | 0 | **0** | dead |
| 1.7578 % | 253 | 0 | **0** | dead |

Showdown, from the row's protocol:

```
|-damage|p1a: Registeel|125/253 slp      128
|-damage|p1a: Registeel|4/253 slp        121   -> survives at 4
|-heal|p1a: Registeel|130/253 slp|[from] move: Wish|[wisher] Umbreon    126
|-heal|p1a: Registeel|145/253 slp|[from] item: Leftovers                 15
```

**There is no engine branch in which the Pokémon survives the hit.** It is dead in 90 % of the
mass and untouched at full HP in the other 10 %, and a full-HP mon heals nothing. So the Wish
and Leftovers heals are not *rendered wrongly* — they are never *generated*. No renderer change
can produce them.

`reports/artifacts/c122_weather_holdout_sweep.json` records the miss as
`observed_only=[('itemleftovers', 15), ('movewish', 126)] engine_only=[]`, which is consistent
with both explanations; the replay is what distinguishes them, and it was one command away.

## 2. Why the engine has no surviving branch

The branch masses give the mechanism exactly. Bonemerang is 90 % accurate:

- `84.3750 = 90 × 15/16`
- crit mass `3.8672 + 1.7578 = 5.6250 = 90 × 1/16`

That is **one crit roll and no damage-roll fan at all**. The 85–100 roll fan lives behind the
gate at `gen3/generate_instructions.rs:3117`:

```rust
if branch_on_damage
    && choice.first_move
    && pending_hp_reading_move(defender_choice)
    && fixed_damage.is_none()
```

Neither `seismictoss` nor `bonemerang` reads HP, so the gate is closed and the whole fan
collapses to a single roll. That single roll deals **exactly 253** into exactly 253 max HP —
the three damaged branches differ only in how the total is split across the two hits
(`129+124`, `253`, `244+9`), which is per-hit clamping to remaining HP, not roll variation.

Showdown's roll totalled 249. **The margin between the two worlds is 4 HP out of 253.**

## 2b. Does widening the fan actually produce a surviving branch?

The diagnosis above would be worthless for Phase 2 if every roll were still lethal, so this is
measured rather than assumed. `poke_engine.calculate_damage` on the same state returns
`([78], [140, 282])` — p1's Seismic Toss at its fixed 78, and Bonemerang's per-hit maximum of
**140** non-crit. The fan scales that as `(raw * random / 100)` for `random` in 85..=100:

| roll | per hit | 2-hit total | outcome |
|---|---|---|---|
| 85 | 119 | 238 | survives at 15 |
| 88 | 123 | 246 | survives at 7 |
| 90 | 126 | 252 | **survives at 1** |
| 91 | 127 | 254 | dead |
| 100 | 140 | 280 | dead |

**Rolls 85–90 survive; 91–100 do not.** So widening the gate puts **6 of 16 rolls — 37.5 % of
the non-crit mass — on branches where the Pokémon lives and the residual walk runs.** The
conclusion follows.

One honest caveat this exposes, which is a separate question from A9: the engine's fan applies
**one roll to the whole move**, so a two-hit move always produces an even split and can only
land on the totals above. Showdown rolls **each hit independently** — 128 and 121, totalling
249, which is not reachable at all under a shared roll. That is a second, narrower divergence in
multi-hit damage; it does not affect this refiling (survival is reachable either way) and it is
recorded here rather than folded in.

## 3. Disposition

**A9 is refiled from "renderer omission" to the damage-fan class, and it is absorbed by Phase 2.**

This matters for the Phase 2 decision specifically. `reports/c119_phase2_scoping.md` lists A9 as
"**no** — a renderer omission" and that judgement is load-bearing in its 5 → 2 headline. On the
measurement above it is a **yes**: widening the `pending_hp_reading_move` gate so the roll fan
runs would create surviving branches, and the Wish and Leftovers heals follow from the ordinary
residual walk with no further change.

That also connects to the finding already recorded on #1088: the engine **already ships**
enumerate-then-merge at `:3117`, enumerating `for random in 85..=100` through `run_move` per
roll and merging with `combine_duplicate_instructions`. Phase 2 is therefore not "build a
mechanism" but "widen an existing gate", and this row is evidence about what widening buys.

**No fix is proposed here and none should be inferred.** Widening that gate is a throughput
decision — the engine's own comment at `:2911-2916` records a measured cost of "12 branches to
144, ~8x slower per call" — and it belongs to the Phase 2 measurement, not to a row-by-row
patch. This report changes an attribution, nothing else.

## 4. Note

C117 filed this row from its miss signature, which named the two missing components and was
correct about them. The inference from "these components are missing" to "the renderer omits
them" was never measured, and the replay that refutes it takes one `State.from_string` and one
`generate_instructions`. Same shape as C118 v1, C119 and C120: the artifact was available, the
structural story was cheaper to reach for, and it was wrong.
