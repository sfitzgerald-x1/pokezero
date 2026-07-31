# C26 Prediction: residual-lethality roll split

## Observation

A retained terminal boundary has a last opposing active that is badly poisoned.
The public move roll leaves that active inside its next Toxic tick, so Toxic
ends the battle before the winner's queued Leftovers handler. The reconstructed
engine state already carries the correct Toxic counter and Truant phase.

The engine instead collapses the regular move-roll range to its representative
nonlethal damage value. That representative leaves the poisoned active alive,
so it emits the winner's Leftovers. Its existing direct-KO branch is separate,
but the public result is a non-direct-KO move followed by a lethal residual.

## Prediction

For a damaging move whose ordinary roll range does not directly KO, split only
when a legal regular roll crosses a known, same-turn residual lethal boundary
for the defending last active. The crossing branch must retain the actual
roll-derived move damage, run the residual phase, and stop queued residual
handlers once that residual faint ends the battle.

The representative non-crossing branch must remain unchanged. Direct move-KO
splits, non-terminal residual faints with a replacement available, and turns
without a known imminent residual must retain their current behavior.

## Acceptance And Controls

Positive control:

- A faster, badly poisoned last active survives the engine's representative
  damage but faints after a higher legal regular roll. The crossing branch has
  the move damage and Toxic faint, and has no later winner Leftovers heal.

Negative controls:

- The same state with a living reserve still runs the remaining residual block
  after the poisoned active faints and owes a replacement.
- A regular non-crossing roll retains the existing representative branch and
  its normal residual handlers.
- A direct move-KO continues to use the existing direct-KO split rather than
  the residual-lethality path.

If the native state cannot distinguish the residual's exact lethal amount from
public information, or if the split changes an unrelated ordinary-damage
fixture, this prediction is falsified and no broad roll expansion is allowed.
