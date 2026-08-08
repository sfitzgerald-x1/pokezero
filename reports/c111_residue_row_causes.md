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

> **[CORRECTION 2026-08-07 — C144.]** The identity quoted above is the **two-term**
> form, which is not a property of the instrument: the real one is four-term
> (`matched + diverged + engine_errors + skip:strict_all_branches_lossy ==
> boundaries_measured`). The two-term form closes only when both extra terms are 0, and
> `/tmp/sweep_f1.json` was never committed, so **`skip:strict_all_branches_lossy`
> cannot be re-derived for this run** — the reported reconciliation is therefore
> unverifiable rather than wrong. It is very likely 0 (every committed artifact on this
> window era has it at 0), but "likely" is not a measurement. The row causes this report
> is about do not depend on it. See `reports/c144_boundary_identity_correction.md`.

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
| 19000191/63 | `component_magnitude:heal` ¹ | **A7** collapsed lethal arm discards the clamped sap | engine fix |
| 19000198/33 | `limit:roll_divergent_lethality` | **A3** — same stale read | **CLOSED** by #1065 |

¹ **Era note.** Every other class in this table is read at the era stamped
above (`main` 9acc9d30, artifact `/tmp/sweep_f1.json`, 11 divergent). This row's
class is re-read **post-#1066**, because A2 moved it off
`limit:roll_divergent_lethality`. Verified against the live classifier at that
later era, but its provenance differs from the other ten rows and must not be read
as contemporaneous with them.

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


---

# Addendum — A2 implementation spec, read out of the phase itself

Recorded because the ordering is the hard part of A2 and was only partially known
before: earlier notes had the heals and damage ticks but not their position
relative to Wish or the weather decrement. Line numbers are the vendored patched
tree at `main` 5953328d, `src/gen3/generate_instructions.rs`.

**The order the mirror must reproduce, for the defender's own HP:**

| # | step | site | effect |
|---|---|---|---|
| 7 | Wish | `:3615-3638` | heal, only when `wish.0 == 1` **and** `0 < hp < maxhp` |
| 8 | weather decrement, then chip | `:3647-3657` | **decrements first**; if `turns_remaining` hits 0 the weather ENDS and there is **no chip this turn** |
| 10.3 | abilities → `ability_end_of_turn` | `:3729` | Rain Dish heal |
| 10.4 | items → `item_end_of_turn` | `:3747` | Leftovers heal, threshold berries |
| 10.5 | Leech Seed | `:3756` | damage |
| 10.6 | status | `:3807` | brn `maxhp/8`, psn `maxhp/8`, tox `maxhp/16 * (count+1)` |
| 10.9 | partial trap | `:3914` | damage |

Two consequences the current `pending_residual_damage` gets wrong by
construction, both already visible in the residue:

1. **Heals precede every damage tick** (7, 10.3, 10.4 before 10.5, 10.6, 10.9),
   so a damage-only sum puts the threshold too low. This is `19000147/125` and
   `19000191/63`.
2. **The chip does not happen on the expiring turn**, because the decrement at
   `:3647-3652` runs before it. `weather_is_active` ignores `turns_remaining`, so
   a mirror that consults it counts a chip that never lands — the over-count
   direction, and the one that can make the engine *worse* than not partitioning.

**Why a net sum is unsound, concretely.** Weather at 8 can kill before the heal
at 10.4. With `hp_after_move = 10`, weather chip 15 and Leftovers 18, the net is
`-3` — "never dies" — but the defender is dead at step 8 and never reaches the
heal. So the mirror cannot sum; it must evaluate in order with each clamp applied.

**Shape of the fix.** Replace the magnitude question with a survival question:

    fn survives_residual_phase(state, side_ref, hp_after_move) -> bool

simulating the table above with every clamp (heals capped at `maxhp`, each damage
tick clamped to remaining HP, and `stop_residuals_if_battle_ended` semantics).
`survives` is monotonic non-decreasing in `hp_after_move`, so there is a unique
`h* = min{h : survives(h)}` and damage `d` is residual-lethal iff
`hp - d < h*`, i.e. the partition threshold is `hp - h* + 1`. Locate `h*` by
bisection over `[1, hp]` — about ten evaluations.

**Monotonicity caveat, and the safe default.** Threshold berries break the
monotonicity the bisection depends on: Sitrus fires at `hp <= maxhp/2`, so a
*lower* starting HP can finish *higher*. Until that is handled, the mirror must
**decline to partition** whenever the defender holds a threshold berry, rather
than bisect through a non-monotonic predicate. Declining is safe in the way
over-partitioning is not: under-counting can only fall back to the pre-partition
behaviour, which is where `main` already sat, whereas an over-count moves
probability mass onto an arm that does not happen. The same conservative default
should cover any phase member not yet mirrored, so the mirror is never *more*
wrong than no partition.

**Acceptance for the A2 fix**, so it is not gradeable on the counter alone:
`19000052/36`, `19000147/125` and `19000191/63` close; `boundaries_measured`
holds at 15,224; `transitions_matched` rises; zero newly divergent; and a mass
probe in the style of `probe_residual_partition_masses` confirms the new
thresholds against a reconstruction that shares no arithmetic with the mirror.

---

# Addendum 2 — A7, and a retraction of the phrase "roll granularity"

`19000191/63` was described after the A2 work as a "roll-granularity defect".
**That phrasing is wrong and is withdrawn.** It implies rounding or precision
loss. There is none: gen3 damage rolls are exact integers and every clamp in the
end-of-turn phase is integer arithmetic. Nothing is lost to precision anywhere in
this row.

## The actual mechanism

A2 fixed this row's *threshold* — 108 is correct, and the engine's lethality
verdict now agrees with Showdown, which is why the class moved off
`limit:roll_divergent_lethality`. What survives is a different defect.

The engine collapses the residual-lethal arm onto a single representative damage,
its minimum (108), because for **lethality** all seven rolls in that arm are
interchangeable — every one of them kills. That collapse is sound only if death is
the sole observable consequence. It is not. Leech Seed's sap is
`min(maxhp / 8, hp)`, clamped to the victim's HP *at step 10.5*, and that HP is
roll-dependent:

| roll | after move | +Leftovers (14) | sap `min(29, hp)` | heal delivered cross-side |
|---|---|---|---|---|
| 108 | 15 | 29 | 29 | **29** |
| 109 | 14 | 28 | 28 | **28** |
| 110 | 13 | 27 | 27 | **27** |

All three die. But 10.5 heals the *other* side by the sapped amount, so the
opponent's heal is 29 / 28 / 27. Showdown rolled 109 and emitted
`|-heal|p2a: Tangela|229/277 par|[silent]` = **+28**; the collapsed arm offers only
**+29**. Hence `component_magnitude:heal`.

## Why this is the program's recurring shape, not a special case

This is the same structural defect as the crit-kill split (C27) and the
residual-lethality partition (C109/A2): **a collapsed arm whose members differ in
an observable the collapse discarded.** Each of those fixes partitions one arm on
one threshold because the collapse was hiding one distinction. Here the discarded
distinction is the clamped sap.

**Disposition: engine fix — an ENGINE SUPPORT GAP, not a limit and not a
methodology artifact.** The clamp binds only when
`hp_after_move + leftovers_heal < maxhp / 8`; above that boundary every roll saps
the full `maxhp / 8` and the rolls genuinely are interchangeable. So the lethal arm
needs partitioning at exactly that boundary, in the same shape as every prior
partition in this stack.

Filed as **A7** and queued alongside the Case A three-way partition, not off to one
side as something the methodology cannot reach.

---

## Note from C112 (belief-surface follow-on) — filed, not fixed

`reports/c112_leaf_state_divergence_ledger.md` ledgered the leaf-vs-reality state divergences on
`corpus/golden-v4`. Three classes it encountered are owned **here and by the rust-fidelity lane**,
so it recorded them as skip counts and deliberately did not attribute or fix them. C112 said "a
note belongs in those ledgers"; this is that note, filed late — it was owed at C112's merge and
was not written until an audit caught the omission.

What C112 saw, and nothing more:

- **`self_moveset_mismatch`** — 11 skips on the scenarios corpus. Not attributed, not touched.
- **Residue rows** — appear only as skip counts in C112's tables.
- **Sleep Talk.** C112 has a cause named **P6**, `NUMERIC_SLEEP_TURNS` (self and opponent, 1 row
  each), and it *is* Sleep-Talk-adjacent, so read it before assuming an overlap. It is a
  **different mechanism** from this lane's `sleeptalk_called_unidentified:ambiguous_unrenderable`:
  P6 is `LeafMeta.sleep` not modelling gen3's `time += skippedTime` turn refund. C112 dispositioned
  it "encoder fix" and changed no code. If this lane's owner judges P6 to belong here instead, it
  is yours — C112 has no claim on it.

Nothing in these classes was modified by that work: the three commits that touched C112's
material (`d57a26ac`, `8eacf0da`, `61a1e946`) are docs and reports only, zero code.

**Provenance of the counts, stated exactly.** The `self_moveset_mismatch` figure of 11 is
C112's, from the **scenarios** corpus, not golden-v4 — quoted from
`reports/c112_leaf_state_divergence_ledger.md` rather than re-measured here. The golden-v4 run
(`python scripts/leaf_vs_reality.py --corpus corpus/golden-v4 --tables
corpus/encoder_tables_v4.json`) reports no `self_moveset_mismatch` skips at all; its skip classes
are `no_branch_match` 42 and `world_unsupported:{materialization_blocker 250,
substitute_health_unknown 14, encore_move_unknown 6, pending_baton_pass 3}`. An earlier draft of
this note attached the golden-v4 command to the scenarios figure, which is the mis-citation this
lane's own conventions exist to prevent.
