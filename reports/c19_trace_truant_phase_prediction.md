# C19: Trace-acquired Truant phase prediction

## Scope

This prediction addresses the two current-source retained identities `3400443/2`
and `3400443/69`. In both, Porygon2 publicly copies Truant before upkeep and
Showdown emits `|cant|...|ability: Truant` on its next move attempt while the
materialized engine world allows an attack.

Native Slaking/Slakoth phase handling is explicitly out of scope. Its
switch-in seed and replacement guard are already separately tested.

## Public mechanism

Gen 3's copied ability does not execute Truant's native `onSwitchIn` hook.
The public `|-ability|...|Truant|...|[from] ability: Trace` line therefore
identifies a holder whose simulator phase is reset to `false` at acquisition.
Each residual after that acquisition flips the phase exactly once.

The parser already exposes the only boundary needed to count those residuals:
`|upkeep|` opens a post-residual window and the following `|turn|` closes it.
Therefore:

| Trace acquisition position | Seed at Trace | Next `|turn|` flip | First move result |
| --- | --- | --- | --- |
| before `|upkeep|` | `False` | yes | loafs |
| after `|upkeep|` replacement | `False` | no | acts |
| incomplete/truncated trace chronology | unknown | no assertion | fail closed / existing fallback |

This uses only emitted Trace, upkeep, switch, and turn lines. It makes no
claim about unrevealed abilities or hidden counters.

## Predictions

1. Replaying the exact retained protocol shapes for `3400443/2` and
   `3400443/69` produces `truantPhase=True` at the next decision boundary.
2. A Trace-to-non-Truant line leaves no Truant phase assertion.
3. Native Slaking's existing lead, switch-action, and post-upkeep replacement
   phase results remain unchanged.
4. A post-upkeep Porygon2 replacement that copies Truant remains
   `truantPhase=False` at its first decision boundary; the parser must not
   double-count a residual which occurred before the replacement entered.
5. If the public chronology cannot establish an active Trace acquisition, the
   parser leaves the phase unknown rather than inferring a value.

## Acceptance evidence

The implementation must add exact protocol-state fixtures for both retained
identities plus the controls above, run focused parser/world/native tests and
the public invariant, and keep the worktree free of private deployment data.
