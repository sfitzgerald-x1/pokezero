# C111 — source-level cause and disposition for every row in the current residue

Era: engine fingerprint `705e015d59d9f5ec6c21f753fb3a75393bc897427d5975a72d893d54cc92ba0a`,
58 patches, `main` at `9acc9d30` (PR #1062 merged). Window: seeds
19000000–19000199, 15,224 boundaries measured, 11 divergent, 0 engine errors,
`matched + diverged == boundaries_measured`. Artifact `/tmp/sweep_f1.json`.

**Documentation only. No classifier change, no fidelity change.** Every row below
was replayed with `scripts/replay_residue.py` against that artifact and read
against both implementations.

## The table

| row | class | cause | disposition |
|---|---|---|---|
| 19000008/54 | `limit:world_sample_drag_target` | Whirlwind drags a random replacement. Showdown realised Seviper; the engine enumerates the candidate set. | **limit** (registered) |
| 19000020/50 | `component_extra_in_engine:itemleftovers` | A1 — faint/forced-switch residual placement | matcher/harness |
| 19000052/36 | `component_missing_in_engine:brn,itemleftovers` | A2 — unmirrored residual source (**Burn**) | engine fix |
| 19000058/19 | `component_missing_in_engine:psn` | A3 — pending read against pre-switch state | engine fix |
| 19000059/27 | `component_extra_in_engine:psn` | A1 | matcher/harness |
| 19000074/27 | `component_missing_in_engine:sandstorm` | A4 — crit-survive arm mis-prices residual lethality | engine fix |
| 19000112/32 | `component_missing_in_engine:itemleftovers` | A1 | matcher/harness |
| 19000125/226 | `component_missing_in_engine:psn` | A5 — Poison Point does not fire for Wrap | engine fix |
| 19000147/125 | `limit:roll_divergent_lethality` | A2 — unmirrored residual **heal** (Leftovers precedes the tick) | engine fix — **reclassify, not a limit** |
| 19000191/63 | `limit:roll_divergent_lethality` | branch set vs single realised sample | **limit** |
| 19000198/33 | `limit:roll_divergent_lethality` | branch set vs single realised sample | **limit** |

Eight non-limit rows, **five shared causes**. Three rows survive as genuine
comparison limits, plus one currently misfiled as a limit that is not one.

## A1 — faint/forced-switch residual placement (3 rows)

`19000020/50`, `19000059/27`, `19000112/32`. All three carry
`ToggleSideOneForceSwitch`.

The engine defers the end-of-turn phase when a faint occurs
(`end_of_turn_is_deferred`, `poke-engine-gen3-residual-defer-on-faint.patch`),
so the tick lands on the *replacement* boundary. Showdown runs it on the
*faint* boundary and the replacement boundary carries no residual at all.
`19000059/27` is the clean case: Showdown shows two switches and `|turn|27` with
no residual, while the engine emits `Damage SideOne: 16` plus the Toxic counter
increment.

The deferral is **faithful gen3 behaviour**, verified against the real sim by
`gen3_switch_differential.py::faintresiduals`. So neither side is wrong: the
same damage is attributed to adjacent boundaries. This is the harness comparing
a boundary against the wrong one.

**Disposition: matcher/harness fix**, needs a "residuals already ran this turn"
marker so the comparison pairs the engine's deferred phase with Showdown's
earlier one. Requires a pin that fails when reverted. Not a limit — the
methodology can settle it, it simply lacks the state marker. This is the
`I5_boundary_truncation` family, now with three attributed rows instead of a
description.

## A2 — `pending_residual_damage` does not mirror the whole phase (2 rows)

Confirmed by independent review of PR #1062 and recorded there as an open gap.
The mirror handles hail, sand, poison, toxic and Leech Seed. It does **not**
handle Burn, partial trap, Future Sight, Wish, Rain Dish, Leftovers or threshold
berries — and the phase **heals before it damages** (10.3 abilities, 10.4 items,
then 10.5 Leech Seed, 10.6 status, 10.9 partial trap).

- **`19000052/36` — Burn.** Walrein 211 HP, `maxhp` 307, burned. Showdown's Surf
  rolled 163, leaving 48; Leftovers `+19`; Burn `−38` → survives at 10. The
  engine's collapsed representative is 177, and the Burn tick then kills. Burn is
  not in the mirror, so `pending_residual_damage` returned 0 for that component
  and the fan was never partitioned on the Burn threshold.
- **`19000147/125` — Leftovers heal.** Bellossom 116/293, `toxic_count` 2 → tick
  `22 × 3 = 66`, Leftovers `+18` at 10.4 *before* the tick at 10.6. True
  lethality is `116 − d + 18 − 66 ≤ 0`, i.e. `d ≥ 80`; Showdown rolled 78 and
  survived at 2. The mirror sums damage only, so it puts the threshold at
  `116 − 66 = 50` and prices the whole fan as lethal.

**Disposition: engine fix.** The remedy is an ordered simulation of the phase
with each clamp applied, and the threshold located by monotonicity in starting
HP — *not* a net sum, which is unsound because weather damage at order 8 can kill
before the 10.4 heal. Caveat carried from review: threshold berries break
monotonicity (Sitrus fires at `hp <= maxhp/2`, so lower starting HP can end
higher), so a bisection must special-case them or scan the band.

**`19000147/125` is currently filed `limit:roll_divergent_lethality` and that is
wrong.** It has a determinate cause above and a named fix, so it is a third-kind
disposition only for as long as the fix is unimplemented. Reclassifying it is a
*classifier* change and must be measured and committed separately from the fix,
per the program rule. Not done here.

## A3 — the pending read happens against pre-switch state (1 row)

`19000058/19`. Fearow switches in at 123/238 already Toxic; p2's Rock Slide then
hits it. Showdown rolled 106 → 17 left, `psn −14` → survives at 3. The engine's
84.38% arm is `move −112` then a residual clamped to the remaining 11, and it
dies.

The partition's own conditions are all satisfied here — tick 14, threshold
`123 − 14 = 109`, `max 121 ≥ 109`, `min 102 < 109` — yet the branch masses are
exactly miss 10 / non-crit 84.375 / crit 5.625, so **no split occurred**, which
means `pending_at_end_of_turn` was 0.

`pending_residual_damage` is bound at *function* level in
`generate_instructions_from_move`, and the boundary's pre-state HP is
`{p1: 179, p2: 205}` — 179 is the **outgoing** Pokémon, not Fearow's 123. The
mirror read the Pokémon that was active before the switch, which was not
poisoned, so it returned 0.

**Disposition: engine fix.** Review already established that the borrow
constraint used to justify the function-level binding does not exist — the
binding compiles at the later position, immediately above
`state.get_both_sides(&attacking_side)`, which reads `state` after this call's own
earlier mutations. That relocation is the candidate fix for this row. It can move
behaviour, so it needs its own single-variable measurement.

## A4 — the crit-survive arm mis-prices residual lethality (1 row)

`19000074/27`. Furret's Return crits Rapidash (`maxhp` 244) for 241, leaving 3;
the sandstorm tick (`244/16 = 15`) kills it. The threshold is sharp: a crit
leaving ≤15 dies, ≥16 lives. The engine's crit-survive arm collapses the whole
non-kill crit fan to one representative (227, leaving 17) and survives by two HP.
Six of the sixteen crit rolls (229, 231, 234, 236, 239, 241) survive the hit and
die to sandstorm, so that arm mis-prices half its own mass.

The crit-fan residual partition merged in #1062 is the mechanism that reaches
this outcome, but it only engages in the branch where the crit fan **cannot kill
on the hit**; here `max_crit ≥ hp`, so control goes to the crit-**kill** split
instead and the surviving crit arm is never partitioned on the residual.

**Disposition: engine fix** — extend the residual partition into the
crit-kill split's survive arm, the same way #1062 extended it to the crit fan.
Not a limit: a partition reaches the outcome, so the support is incomplete rather
than the methodology being blind.

## A5 — Poison Point does not fire for Wrap (1 row)

`19000125/226`. p2's Wrap (a contact move) hits Nidoqueen, whose Poison Point
poisons Shuckle; Showdown then applies the same-turn tick
(`198/8 = 24`, `194 → 170`). None of the engine's three arms — 93.75% of the mass
— carries `psn` on p2 at all, so the status is never applied and its tick is
absent.

Showdown's line is explicit about the direction:
`-status|p2a: Shuckle|psn|[from] ability: Poison Point|[of] p1a: Nidoqueen`.

**Disposition: engine fix, cause not yet pinned to a line.** Two candidates, both
readable from the two sources and neither yet checked: Wrap's `contact` flag in
`choices.rs`, or the Poison Point trigger's position relative to the partial-trap
handler. This is the one row whose *mechanism* is identified (contact-triggered
secondary absent) but whose *site* is not, so it is the next reading task rather
than a next implementation task.

## Stop-condition status, stated plainly

Every row has a written source-level cause and a disposition, which was the first
half of the condition. The second half is **not** met: the remaining divergence
is not only third-kind dispositions. Three rows are genuine limits
(`19000008/54`, `19000191/63`, `19000198/33`); eight are engine or harness fixes
that are attributed and unimplemented, and one of those eight
(`19000147/125`) is currently *misfiled* as a limit.

Queue, ordered by rows × search impact:

1. **A2 ordered residual simulation** — 2 rows directly, and it is the shared
   prerequisite for pricing Burn, partial trap, Wish and Leftovers anywhere in
   the fan. Largest blast radius.
2. **A1 residuals-already-ran marker** — 3 rows, harness-only, no engine risk.
3. **A3 binding relocation** — 1 row, smallest diff, already known to compile.
4. **A4 crit-kill survive-arm partition** — 1 row, symmetric with work already
   merged.
5. **A5 Wrap / Poison Point** — 1 row, read the two sources first.

Divergence count is reported as an outcome: **11**, from 208 at the era baseline.
