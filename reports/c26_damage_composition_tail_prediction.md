# C26 Damage-Composition Tail Prediction

## Scope

This lane is limited to the retained damage-composition rows `2900889/126`
(Sleep Talk calling Earthquake) and `3400914/75` (Double-Edge followed by
Rest). It intentionally excludes the active confusion-before-Substitute row
`3401155/44` and the separately owned Rest/residual, Toxic-stage,
switch-prefixed-confusion, Trace/Truant, and partial-trap lanes.

## Reproduction

Both target seeds reproduce against the current public differential harness.
The relevant observed protocol tails are:

- `2900889/126`: Sleep Talk emits an executed `Earthquake` hit for 82 damage.
- `3400914/75`: Double-Edge deals 98, recoil deals 32, and Rest restores 234
  HP to full.

The engine's representative chance branches are compatible with those
outcomes: Earthquake has a legal 82-damage roll, and Double-Edge has a legal
98-damage roll with 32 recoil. This prediction therefore does **not** claim a
damage-calculation defect.

## Predictions

1. The strict matcher rejects the Sleep Talk branch because its pre-state roll
   support prices the selected action as `sleeptalk`, whose damage base is zero,
   rather than the executed `Earthquake` named by the branch's `|move|` line.
   Pricing that identified callee against the same pre-boundary state should
   admit 82 without widening unrelated components or ambiguous called-move
   branches.
2. The strict matcher rejects the Double-Edge/Rest branch because it compares
   each ordered component in isolation. The existing `*_to_full` rule needs
   the preceding direct-damage scale from the same ordered tail; isolated
   comparison sees no scale and incorrectly rejects the 234-vs-228 full-heal
   difference. Supplying that tail-local scale should admit the coherent
   direct-hit/recoil/Rest composition while retaining the existing exact event
   order and source checks.
3. The other retained rows in seed `2900889` (steps 3 and 93) are ordinary
   capped-heal tails and should remain divergent after this change. Clearing
   them would falsify the narrow mechanism and require separate analysis.

## Acceptance

The implementation is acceptable only if focused retained-shape tests show:

- a named called move is used for direct-roll support;
- a full-heal receives scale from an earlier same-tail roll component;
- no support leaks to another event, source, or target; and
- the two target rows replay as matched without clearing the predicted
  unrelated rows.
