# C126 — A6 closed: gen3 White Herb implemented (C116 Phase 3 item 11)

Item 11 poses a fork before it asks for code. `reports/c124_a6_is_knowable.md` (#1099) answered
it: the item **is** knowable, deterministically, so A6 takes the engine-fix branch and is not
the program's first demonstrated limit. This is that fix.

> **On the C116 citation.** As in C121–C125: the plan is not in this repository, so it is
> provenance for why this work was queued, never evidence for a claim.

## 1. The defect

`WHITEHERB` existed only under `src/genx/` and was absent from gen3's own `Items` enum, so a
Deoxys never restored a stat drop. C111 §A6, row `19000112/32`: Deoxys uses Superpower, dropping
its own Defence; Showdown restores it with `-enditem` + `-clearnegativeboost` before Golem's Rock
Slide lands. With the drop still applied Rock Slide is max 249 / min 211 against 188 HP, so
**every** roll kills; with it restored it is max 166 / min 141, so **no** roll kills. Showdown's
realised roll was exactly 141, the minimum.

## 2. Semantics, and what is deliberately not covered

`data/items.ts:7653`. There is **no gen3 or gen4 override** — gen3's `scripts.ts` is
`inherit: 'gen4'` — so the base handler applies: zero the **negative** boosts, consume the item.
Positive boosts are untouched; it is not a reset.

Showdown fires it from four triggers: `onAnySwitchIn` (priority −2), `onAnyAfterMega`,
`onAnyAfterMove`, and `onResidual` at order 29. **This wires `onAnyAfterMove` only** — once after
`get_instructions_from_boosts`, for both sides, since that call applies `MoveTarget::User` and
`MoveTarget::Opponent` alike, so a self-drop and an opponent-inflicted drop are both caught.

Not covered, named in the patch rather than silently omitted: secondary-effect stat drops,
Intimidate on switch-in, and the residual-order-29 sweep. The reachability argument for stopping
here is item 14's rule applied honestly: White Herb is on **Deoxys and Deoxys-Attack only**
(`teams.ts:471`, unconditional), and their movepool — extremespeed, icebeam, psychoboost,
shadowball, superpower — carries exactly two self-droppers, Superpower (−1 Atk/−1 Def) and
Psycho Boost (−2 SpA). Both are after-move.

**Two placements were wrong on the way and are recorded so they are not regressed:** not
`apply_boost_instruction` (fires per stat, too early), and not the per-stat loop inside
`get_boost_instruction` (Showdown fires once per **move**, not once per boost application).

## 3. Red run (M3)

Against the 60-patch engine, before the patch existed:

| pin | failure |
|---|---|
| restores a self-inflicted drop and is consumed | `'Boost SideOne Attack: 1' not found` — only the `-1` drops present, no `ChangeItem` |
| leaves positive boosts alone | `'ChangeItem SideOne: WHITEHERB -> NONE' not found` |

Both **failed**, not errored — an earlier draft of the second pin *errored* on
`speed_boost` not being writable post-construction, which proves nothing; the boost is now
threaded through the `_state` helper's constructor. The second pin exists so an implementation
cannot pass by clearing **every** boost.

## 4. Gates

| gate | result |
|---|---|
| `tests/test_poke_engine_patch_stack` | Ran 4, OK — tail pin **grown** to 6 |
| `tests/test_engine_gen3_abilities` | Ran 48, OK (46 + the two new pins) |
| `tests/test_branch_mass_reconstruction` | Ran 5, OK |
| `tests/test_crit_kill_split_patch` | Ran 8, OK |
| `tests/test_drag_limit_is_a_last_resort` | Ran 3, OK |
| `scripts/engine_behavioral_probes.py` | exit 0, all PASS |

`src/gen3/items.rs` **joins the pinned digest set**. Without it a White Herb regression would
move no pinned digest at all.

## 5. Sweep

Prediction registered before the results were read: **dev 5 → 4** closing `19000112/32`,
**holdout 11 → 11 unchanged** — no holdout row is filed to A6, and White Herb is on Deoxys only,
so holdout movement would mean the hook fired where it should not.

| window | boundaries | matched | diverged |
|---|---|---|---|
| dev — before | 15,224 | 15,219 | 5 |
| dev — after | 15,224 | **15,220** | **4** |
| validation holdout — before | 15,396 | 15,385 | 11 |
| validation holdout — after | 15,396 | 15,385 | **11 (unchanged)** |

Row level: dev closed exactly `19000112/32`; **nothing opened in either window**; the holdout
closed and opened nothing. Identity `matched + diverged == boundaries` holds on all four rows.
Artifacts committed as `reports/artifacts/c126_whiteherb_{dev,holdout}_sweep.json`.

> **[CORRECTION 2026-08-07 — C144.]** The identity is four-term, not two-term:
> `matched + diverged + engine_errors + skip:strict_all_branches_lossy ==
> boundaries_measured`. It closes in two terms on these rows **only because
> `skip:strict_all_branches_lossy` is 0 on all four committed artifacts** (re-derived:
> dev 15,220 + 4 + 0 + 0 = 15,224; holdout 15,385 + 11 + 0 + 0 = 15,396, on both the
> plain and `_extended` pairs). Stated as a general property it is false —
> `reports/c144_boundary_identity_correction.md`.

Residue is now **dev 4 / holdout 11**, reported as an outcome.

## 5b. Self-review found the reachability argument false, and it is fixed

No independent reviewer was obtainable (twelve consecutive API 529s), so this was self-reviewed
against the same axes I had written into the reviewer's brief — one of which was *"can a Deoxys
receive a negative boost from a source the patch does NOT cover?"* It can, and the first version
of §2's argument was wrong.

That argument reasons only about **Deoxys' own moves**. It says nothing about what an
**opponent** can do to Deoxys, which is most of how a Pokémon acquires a negative boost. Two
live gaps, reproduced rather than theorised:

| gap | evidence | pool reach |
|---|---|---|
| secondary stat drops, applied *after* the boost-effect site | Deoxys hit by Crunch kept `-1 SpD`, no restore, no `ChangeItem` (gen3 Crunch drops **SpD**) | Ice Beam 79 species, Shadow Ball 59, Thunderbolt 50, Psychic 42, Flamethrower 23, Crunch 9, Iron Tail 7 |
| Intimidate on switch-in | `onAnySwitchIn` unwired | **18** species — arbok, arcanine, granbull, gyarados, hitmontop, masquerain, mawile, mightyena, salamence, stantler, tauros, … |

This is the same shape as the `sets.json`-for-items error corrected in
`reports/c124_a6_is_knowable.md`: right instrument, wrong question. There the reachability check
was applied to the item when that file cannot answer for items; here it was applied to the item
**holder** and not to the **sources** of the effect.

Fixed rather than re-scoped. `item_white_herb_after_boosts` is renamed `item_white_herb_check`
and called from **three** sites, each checking both sides: after a move's boost effect, after
secondaries (deliberately *not* gated on `does_damage` — a status move's secondary can drop a
stat), and after `ability_on_switch_in`. Still uncovered, and now genuinely narrow: `onResidual`
order 29, which needs a drop surviving to end-of-turn with no move, ability or switch in between.

Independent review of #1102 enumerated every producer rather than reasoning by example — the
five `BoostInstruction` sites and the three `apply_boost_instruction` callers — and found the
three call sites cover all of them bar two: the **Transform / Psych-Up boost copy**
(`choice_effects.rs:1205`) and **Trick-style item acquisition** (Showdown's `setItem` fires the
item's `onStart`). Both are unreachable here: the only holders are `deoxys` and `deoxysattack`,
whose single set's movepool is `{extremespeed, icebeam, psychoboost, shadowball, superpower}` —
no Transform, no Psych Up, no Trick, no Baton Pass. That enumeration also retroactively justifies
omitting order 29: if every reachable producer is one of the three sites, no negative boost can
survive to end-of-turn unconsumed. `onAnyAfterMega` is gen6+ and vacuous in gen3.

Two further pins, both seen red first: the secondary pin failed with the `-1 SpD` present and no
restore, and it asserts the drop actually happens so it cannot pass vacuously; the Intimidate pin
failed with `Boost SideOne Attack: -1` present and no `Boost SideOne Attack: 1`.

**The extension measures NEUTRAL, and that is the expected and correct result.** Re-sweeping both
windows against the §5 figures:

| window | boundaries | matched | diverged | closed | opened |
|---|---|---|---|---|---|
| dev | 15,224 | 15,220 | 4 | none | **none** |
| validation holdout | 15,396 | 15,385 | 11 | none | **none** |

No residue row involves a White Herb holder taking a secondary or an Intimidate drop, so nothing
was expected to close. The sweep's value here is the **negative** result: three call sites, each
firing on both sides, and nothing over-fires. Justification for the extension is correctness plus
pool reachability, **not** a residue reduction — recorded that way so it is not later cited as
one. Artifacts: `reports/artifacts/c126_whiteherb_{dev,holdout}_sweep_extended.json`.

## 6. Where Phase 3 stands

| item | state |
|---|---|
| 8 — A5 | closed, #1090 |
| 9 — A1 marker | **open** |
| 10 — B1 | closed, #1081 + #1086 |
| 11 — A6 | **closed here**, on the engine-fix branch |
