# C19: Trace-acquired Truant phase prediction

## Scope

This prediction addresses the two current-source retained identities `3400443/2`
and `3400443/69`. In both, Porygon2 publicly copies Truant before upkeep and
Showdown emits `|cant|...|ability: Truant` on its next move attempt while the
materialized engine world allows an attack.

Native Slaking/Slakoth phase handling is explicitly out of scope. Its
switch-in seed and replacement guard are already separately tested.

## Predicted public mechanism (withdrawn)

This section preserves the pre-implementation prediction. The measured outcome
and production disposition in the amendment below supersede it.

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
| incomplete/truncated trace chronology | unknown | no assertion | existing fallback |

This uses only emitted Trace, upkeep, switch, and turn lines. It makes no
claim about unrevealed abilities or hidden counters.

## Pre-implementation predictions

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

## Outcome amendment

**Prediction withdrawn; no production Trace phase seed is retained.**

The current-source replay did not support the proposed chronology rule:

- `3400443/2`: a one-sided action switch copied Truant before upkeep and loafed
  at the next decision.
- `3400443/69`: a move-KO forced replacement copied Truant before upkeep and
  loafed at the next decision.
- `2200291/41`: Porygon2 and Slaking switched simultaneously, Trace still
  appeared before upkeep, but Porygon2 acted at the next decision.

The last identity is ledger Z13.3's measured counterexample, but source drift
changed its surrounding replay. The ledger recorded a post-faint replacement
and a `3 -> 0` seed-level result; regenerating the same seed/step on current
source produces simultaneous switches. Untouched current `main` has four
unrelated divergences in seed `2200291`; the proposed boolean seed adds
`2200291/41` as a fifth and removes none from that game.
Therefore the visible `Trace -> upkeep -> turn` ordering does not determine
whether the copied ability participated in that residual's event queue. A bool
would overstate public knowledge at exactly the boundary under audit.

Production parser behavior remains non-asserting: Trace records the current
holder but leaves `truant_phase=None`. The downstream engine-world adapter
retains its pre-existing action-history proxy when the payload is `None`; this
is not a fail-closed materialization block and is why the selected rows remain
unresolved. A public own-move or
`|cant|...|ability: Truant` line anchors the phase, after which public turn
boundaries maintain it exactly. The native-holder replacement guard remains
valid and now snapshots/restores its post-upkeep window and pending skipped
flip.

The withdrawal resolves the prior measured counterexample rather than
reclassifying it: on current source, the amended branch no longer diverges at
`2200291/41` and matches untouched `main`'s four other divergences for that
seed. The selected `3400443/2` and `3400443/69` identities remain unresolved
under the legacy proxy fallback; that is the deliberate cost of declining to
assert a boolean the public chronology cannot determine. A fresh
full-population 10,000-game reread is required before making any
population-level improvement claim.
