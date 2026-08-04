# C111 v2 — source-level cause and disposition for every row in the residue

> **v2 corrects v1 substantially after independent review.** Three of v1's
> attributions were wrong, its era stamp cited the wrong hash, and its central
> claim that three rows are genuine comparison limits is now **withdrawn — none
> of the three is demonstrated irreducible.** Every correction is marked below.
> v1 should not be cited.

## Era and provenance

Window: seeds 19000000–19000199 on `main` at `9acc9d30`, artifact
`/tmp/sweep_f1.json`: **15,224 boundaries measured, 11 divergent, 0 engine
errors**, `matched + diverged == boundaries_measured`.

**Correction (v1 error).** v1 stamped the era as engine fingerprint
`705e015d59d9…`. That string is the value inside the artifact's
`checkpoint_provenance` block — the fingerprint of the checkpoint that *played*
the games, recorded against `source_commit d6a148a5` — not the engine under test.
`compute_fingerprint()` is a pure function of tracked inputs, so the engine
fingerprint must be read from the build stamp written by
`scripts/engine_build_fingerprint.py --write` for the measuring tree, never from
`checkpoint_provenance`. Anyone reproducing an era from that block will get the
policy, not the engine.

Every row below was replayed with `scripts/replay_residue.py`. **Documentation
only: no classifier change, no fidelity change.**

## The table

| row | class | cause | disposition |
|---|---|---|---|
| 19000008/54 | `limit:world_sample_drag_target` | **B1** — mislabelled. The engine dragged the *same* Seviper; only a component **tag** differs (`spikes` vs `move`). | matcher/renderer fix — **not a limit** |
| 19000020/50 | `component_extra_in_engine:itemleftovers` | **A1** faint/forced-switch residual placement | matcher/harness fix |
| 19000052/36 | `component_missing_in_engine:brn,itemleftovers` | **A2** unmirrored residual source (Burn) | engine fix |
| 19000058/19 | `component_missing_in_engine:psn` | **A3** pending read against pre-switch state | **CLOSED** by #1065 |
| 19000059/27 | `component_extra_in_engine:psn` | **A1** | matcher/harness fix |
| 19000074/27 | `component_missing_in_engine:sandstorm` | **A4** crit-**kill** split's survive arm unpartitioned | engine fix |
| 19000112/32 | `component_missing_in_engine:itemleftovers` | **A6** White Herb absent from gen3 — *not* A1 | engine fix |
| 19000125/226 | `component_missing_in_engine:psn` | **A5** contact-ability trigger precedes the same-turn wake | engine fix |
| 19000147/125 | `limit:roll_divergent_lethality` | **A2** unmirrored residual **heal** | engine fix — misfiled as a limit |
| 19000191/63 | `limit:roll_divergent_lethality` | **A2** unmirrored residual **heal** | engine fix — misfiled as a limit |
| 19000198/33 | `limit:roll_divergent_lethality` | **A3** — same stale read | **CLOSED** by #1065 |

**Six causes. Zero rows demonstrated to be genuine comparison limits.** All three
rows v1 called limits are accounted for: one closed on an engine fix, one reduces
under A2, and one was never a drag divergence at all.

## A1 — faint/forced-switch residual placement (2 rows, was 3)

`19000020/50`, `19000059/27`. The engine defers the end-of-turn phase on a faint
(`end_of_turn_is_deferred`), so the tick lands on the *replacement* boundary;
Showdown runs it on the *faint* boundary. `19000059/27` is the clean case:
Showdown shows two switches and `|turn|27` with no residual, while the engine
emits `Damage SideOne: 16` plus the Toxic increment. The deferral is faithful
gen3, verified by `gen3_switch_differential.py::faintresiduals` — neither side is
wrong, the harness pairs the wrong boundaries.

**Disposition: matcher/harness fix** — a "residuals already ran this turn" marker,
with a pin that fails when reverted.

**Correction (v1 error): `19000112/32` is not an A1 row.** v1 filed it here on the
evidence that it carries `ToggleSideOneForceSwitch`. It does, but for an unrelated
reason, and **Showdown has no faint on that boundary at all** (`|turn|31`, Deoxys
alive at 47/188) — so there is no earlier Showdown phase for a marker to pair
with, and no matcher change can close it. Moved to A6.

## A2 — `pending_residual_damage` does not mirror the whole phase (3 rows)

The mirror handles hail, sand, poison, toxic and Leech Seed. It does **not**
handle Burn, partial trap, Future Sight, Wish, Rain Dish, Leftovers or threshold
berries — and the phase **heals before it damages** (10.3 abilities, 10.4 items,
then 10.5 Leech Seed, 10.6 status, 10.9 partial trap).

- **`19000052/36` — Burn.** *Arithmetic corrected.* Walrein's pre-state HP is
  **192**, not v1's 211, and the damaging move is **Roselia's Magical Leaf** —
  Surf is Walrein's own move, which dealt 45 the other way. Roll set
  `[163…192]`, true threshold **173**. Burn is unmirrored, so the mirror returned
  0 for that component and the fan was never partitioned.
- **`19000147/125` — Leftovers heal.** *Arithmetic corrected.* The Toxic tick is
  `293/16 = 18`, so `18 × 3 = 54` — v1 wrote "22 × 3 = 66", and its inequality
  `116 − d + 18 − 66 ≤ 0` yields `d ≥ 68`, not the 80 it concluded. The correct
  statement is `116 − d + 18 − 54 ≤ 0`, i.e. **`d ≥ 80`**; Showdown rolled 78 and
  survived at 2. The mirror's threshold is `116 − 54 = **62**`, not v1's 50.
- **`19000191/63` — Leftovers heal. NEW: this was v1's "genuine limit".** Raichu
  123/235, Leftovers `+14` at 10.4, Leech Seed `−29` at 10.5. True lethality
  `123 − d + 14 − 29 ≤ 0` → **`d ≥ 108`**; Showdown rolled 109 and died, the
  engine collapsed to `0.925 × 115 = 106` and lived. The mirror's threshold is
  `123 − 29 = 94`, and the guard `min_damage_dealt < residual_threshold` is
  `97 < 94` = **false**, so no split. The true threshold 108 lies inside the roll
  set `[97…115]`, so an ordered mirror straddles it and the kill arm contains
  Showdown's 109. This is character-for-character `19000147/125`'s cause, and v1
  declared one misfiled while calling the other irreducible — internally
  inconsistent, and the "limit" label was not earned.

**Disposition: engine fix.** An ordered simulation of the phase with each clamp
applied, threshold located by monotonicity in starting HP — *not* a net sum, which
is unsound because weather damage at order 8 can kill before the 10.4 heal.
Threshold berries break monotonicity (Sitrus fires at `hp <= maxhp/2`), so a
bisection must special-case them or scan the band.

Reclassifying `19000147/125` and `19000191/63` out of `limit:` is a **classifier**
change and must be measured and committed separately from the fix. Not done here.

## A3 — pending read against pre-switch state (2 rows, both CLOSED)

`19000058/19` and `19000198/33`, both fixed by #1065. The binding sat ~70 lines
before `state.apply_instructions(&incoming_instructions.instruction_list)`, so
the second mover's read predated the first mover's executed action, including a
switch. `19000198/33` is the important one: v1 called it a **genuine comparison
limit**, and it closed on an engine fix. Hitmontop switches in at 116/226
poisoned, tick 28, threshold 88; Showdown's 91 kills, the engine's collapsed 84
survives, and `min 77 < 88 ≤ 91` means the partition should have fired. It did not,
because the read saw the outgoing Pokémon.

**Lesson recorded, not softened:** a "limit" label survived only until someone
measured the mechanism. The program rule reserves "limit" for what the methodology
*genuinely cannot settle*; v1 applied it to three rows on the strength of a class
name, and all three failed that test.

## A4 — the crit-kill split's survive arm is unpartitioned (1 row)

`19000074/27`. A crit for 241 leaves Rapidash (`maxhp` 244) at 3 and the sandstorm
tick (15) kills it; the threshold is sharp at ≤15 dies / ≥16 lives. The engine
collapses the non-kill crit fan to 227, leaving 17, surviving by two HP. Six of
sixteen crit rolls survive the hit and die to sandstorm, so that arm mis-prices
half its own mass. The residual partition only engages where the crit fan *cannot*
kill on the hit; here `max_crit >= hp`, so control passes to the crit-**kill**
split and the surviving arm is never partitioned.

**Disposition: engine fix** — extend the residual partition into the crit-kill
split's survive arm, symmetric with #1062's crit-fan extension.

## A5 — contact-ability trigger precedes the same-turn wake (1 row)

`19000125/226`. **Correction: v1's framing and both its candidate sites were
wrong.** Wrap *is* `contact: true` under gen3 (`src/choices.rs`), and Poison Point
*does* fire for Wrap — measured at 1/3 of hit mass for an awake attacker.

The real site is **sleep-wake ordering**: `before_move` →
`ability_modify_attack_against` runs at `generate_instructions.rs:2682`, but the
wake `ChangeStatus SLEEP -> NONE` is not generated until
`generate_instructions_from_existing_status_conditions` at `:2694`. So
`contact_status_is_valid` (`src/gen3/abilities.rs:29-34`) sees `target.status ==
SLEEP` and refuses. Measured: an attacker that wakes this turn and lands Wrap is
poisoned in **0 of 2** wake-and-hit branches, versus 2 of 4 when already awake.
The row matches — `-curestatus|p2a: Shuckle|slp|[msg]` sits immediately before its
Wrap.

This generalises to **every** contact-triggered defender ability (Effect Spore,
Static, Flame Body, Cute Charm) against any same-turn waker or thawer, so "Poison
Point does not fire for Wrap" was far too narrow.

**Disposition: engine fix**, site now pinned to a line.

## A6 — White Herb is absent from gen3 (1 row) — NEW

`19000112/32`, moved out of A1. Golem's Rock Slide vs Deoxys at **undropped**
Defence is max 166 / min 141 against 188 HP, so the engine cannot kill it at all;
Showdown's realised roll was exactly **141**, the engine's minimum. With
Superpower's −1 Defence still applied (×1.5) it is max 249 / min 211 ≥ 188, so
**every** roll kills — which is exactly the artifact: one 84.38% arm,
`capped_lethal=-188`, and no Case A split, because that split requires
`min < hp`.

Showdown restored the drop before Rock Slide landed —
`-enditem|p1a: Deoxys|White Herb` then `-clearnegativeboost`. The engine never
does: `WHITEHERB` appears only under `src/genx/`, never `src/gen3/`, and the
engine's Deoxys item is `UNKNOWNITEM` anyway.

**Disposition: engine fix** (implement gen3 White Herb) **or hidden-item limit**,
depending on whether the harness can know the item. Not settled here — and note
that the hidden-item question is a genuine candidate for a third-kind disposition,
unlike any of v1's three.

## B1 — the drag label fires on the presence of a drag (1 row) — NEW

`19000008/54`. v1 called this a genuine limit: "Showdown realised Seviper; the
engine enumerates the candidate set." **The artifact says otherwise.** The engine
emitted **one** branch at 100% with `Switch SideTwo: P1 -> P2`, and
`party_display.p2[2]` is **Seviper** — the same Pokémon, with the same −68 Spikes
and the same +17 Leftovers. It enumerated nothing.

The only surviving difference is a component **tag**: Showdown files the 68 as
`spikes` (exact), the engine as `move` (rolled). The classifier applies
`limit:world_sample_drag_target` on the mere presence of a `|drag|` line
(`scripts/engine_transition_differential.py:1760-1761`), regardless of whether the
target actually diverged.

**Disposition: matcher/renderer fix** — tag the Spikes damage as `spikes`. This is
a classifier defect, so measuring it belongs in its own commit.

## Stop-condition status, stated plainly

Every row has a written source-level cause and a disposition. The second half of
the condition is **not** met, and v1 overstated how close it was: **zero rows are
demonstrated to be genuine comparison limits.** Two rows are closed (#1065); nine
remain, all with named fixes.

Queue, ordered by rows × search impact:

1. **A2 ordered residual simulation** — **3 rows** (was 2), and the shared
   prerequisite for pricing Burn, partial trap, Wish and Leftovers anywhere in the
   fan. Largest blast radius by a wide margin.
2. **A1 residuals-already-ran marker** — 2 rows (was 3), harness-only.
3. **A5 contact-ability vs same-turn wake** — 1 row, site pinned to two lines,
   and it generalises to four more abilities.
4. **A4 crit-kill survive-arm partition** — 1 row, symmetric with merged work.
5. **A6 gen3 White Herb** — 1 row; decide engine fix vs hidden-item limit first.
6. **B1 Spikes tag** — 1 row, classifier-only, must be measured separately.

Divergence count reported as an outcome: **9** after #1065, from 208 at the era
baseline.
