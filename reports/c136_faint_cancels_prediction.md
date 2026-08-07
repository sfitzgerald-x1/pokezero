# Gen-3 faint cancels a queued switch — attempt 3, registered before any sweep

Registered 2026-08-07, before the v3 patch exists. v1 and v2 both failed their falsifiers
(`switch_cancel_prediction.md`, `switch_cancel_prediction_v2.md`). The difference this
time is that the rule is no longer inferred from reading `battle.ts` — it is measured, on
both sides, by fixtures that are merged or in review.

## What the oracle established that v1 and v2 were guessing

Two Showdown fixture pairs, each passing 4/4 seeds:

| fixture | who faints | opposing switch | residual phase |
|---|---|---|---|
| `faintcancelsopposingswitch` (PR #1141) | the *other* side's incoming Pokémon, on entry | **cancelled** | does not run — block ends at `\|faint\|`, no `\|upkeep` |
| `pursuitkoswitcher` (merged, #1139) | the switcher *itself*, to Pursuit | **continues** | runs |

These are not in tension. They are one rule and one exception:

> A faint cancels the **opposing** side's queued switch. A Pursuit faint does not cancel
> the switch of the Pokémon it just killed — `sim/battle.ts:2790-2796` re-queues that
> switch at priority −101 rather than dropping it.

**v1 implemented the rule and broke the exception's neighbours; v2 implemented something
closer to the exception and broke the rule.** Neither had both measured.

## The v3 guard

Cancel only when the **opponent's** active newly fainted during the incoming
instructions — alive before, `hp <= 0` after — and never when this side owes a
replacement:

```rust
let attacker_owes_replacement = state.get_side_immutable(&attacking_side).force_switch;
if !choice.first_move && !attacker_owes_replacement {
    let defending_side = attacking_side.get_other_side();
    let opponent_fainted_before =
        state.get_side_immutable(&defending_side).get_active_immutable().hp <= 0;
    state.apply_instructions(&incoming_instructions.instruction_list);
    let opponent_newly_fainted = !opponent_fainted_before
        && state.get_side_immutable(&defending_side).get_active_immutable().hp <= 0;
    state.reverse_instructions(&incoming_instructions.instruction_list);
    if opponent_newly_fainted { /* cancel */ }
}
```

Why each clause, tied to a measured failure rather than a theory:

- **`opponent` rather than "either active"** is what preserves Pursuit. In the Pursuit
  shape the fainted Pokémon is the switcher's own active, so the opponent test is false
  and the switch proceeds. v2 tested either side and therefore cancelled it.
- **`newly` rather than `hp <= 0`** is what preserves forced replacements. A side owing a
  replacement enters the ply with a 0-HP active, which is not *newly* fainted. v1 tested
  the bare predicate and cancelled 78 rows' worth of replacements.
- **`force_switch`** is a second, independent guard on that same failure.

## Baseline, re-derived at this era

`main` `6b6fb368`, fingerprint-verified (`67 patches, 07a3290d11ca14ec`). Identical to the
`54e06fe8` measurement, so main's advance moved nothing. Artifacts:
`rust-fidelity-pass-2/artifacts/baseline_6b6fb368_{dev,holdout}.json`.

| window | measured | full_round | diverged |
|---|---|---|---|
| dev `19,000,000–199` | 15,503 | 15,968 | **2** |
| holdout `19,100,000–199` | 15,579 | 16,155 | **5** |

Holdout classes: `component_extra_in_engine:spikes` 1 (`19100180` — the target),
`component_missing_in_engine:itemleftovers` 2 (`19100170`),
`limit:roll_divergent_lethality` 2 (`19100107`, `19100191`).
Dev classes: `component_magnitude:heal` 1 (`19000191`),
`component_missing_in_engine:sandstorm` 1 (`19000074`).

## Prediction

| window | predicted |
|---|---|
| dev | **2 → 2, unchanged** — neither dev row is this mechanism |
| holdout | **5 → 4**, closing exactly `19100180`, the lone `component_extra_in_engine:spikes` |

Also predicted: `boundaries_measured` and `boundaries_full_round` unchanged
(15,503 / 15,968 and 15,579 / 16,155), `engine_errors: 0`, identity holding on all four
runs, and `19100170`, `19100107`, `19100191` all still open — none of them is this
mechanism and v3 must not touch them.

## Falsifier

**If anything opens, or dev moves at all, or either boundary count changes, or
`19100180` survives, v3 is wrong and is withdrawn.** This clause has now caught two
consecutive patches that every other gate passed, both of which had a correct mechanism,
a correct Showdown citation, and a green suite.

Additional falsifiers available this time that did not exist for v1 or v2, because the
oracle now covers both directions:

- `pursuitkoswitcher` / `pursuitnokocontrol` must stay green — if v3 reintroduces the v2
  defect, the merged Pursuit engine pin goes red before any sweep runs.
- `faintcancelsopposingswitch` is the shape v3 is supposed to fix; an engine pin for it
  is currently red on main and must go green with v3, having been verified red first.

A unit pin is still not the gate. Both windows must be swept on a fingerprint-verified
build before v3 is believed.
