# C26 Prediction: Substitute Ends an Active Partial Trap

## Scope

This is a prediction-first, narrow native-engine lane for retained identity
`3101100/29`. The public boundary is: a target is already under a successful
Wrap-family partial trap, then successfully uses Substitute. In Gen 3
Showdown, the successful Substitute ends the target's existing
`partiallytrapped` volatile before end-of-turn residuals. The target therefore
does not take another partial-trap chip on that boundary.

## Prediction

The current native engine retains `PARTIALLYTRAPPED` after a successful
Substitute and emits its residual damage. The expected fix is a
`RemoveVolatileStatus(PARTIALLYTRAPPED)` instruction on the Substitute user
only, emitted after a successful Substitute has been established and before
the residual phase.

The fix must not:

- remove an unrelated volatile;
- remove a partial trap when Substitute fails because the user cannot pay its
  HP cost;
- suppress ordinary residual chip on a trapped Pokemon that does not use
  Substitute; or
- change the existing rule that a partial trap ends when its trapper switches
  or Baton Passes away.

## Evidence And Acceptance

The implementation is accepted only if a direct native fixture proves the
four discriminators above and a Showdown-backed probe confirms the successful
Substitute lifecycle. The patch must apply through the frozen patch manifest
with no fuzz, keep fixture refresh last, and preserve apply/reverse behavior.

This artifact intentionally predicts behavior before inspecting or modifying
the relevant engine code. Results and residual limitations belong in the
implementation commit, not here.
