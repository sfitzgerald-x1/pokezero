# Own-policy-prior evidence audit

Date: 2026-09-17. This note records the completed evidence that motivated the
separate guided-MCTS-versus-raw-policy study. It is an audit of immutable
artifacts, not a new statistical analysis.

## What the completed comparison establishes

The source-bound own-prior strength readout at
`/shared/scott-experiment/mcts-own-prior-191af857-20260917-strength-derived-readout/READOUT.json`
has SHA-256
`5eb3433cd11e20fbeb5cc106e4c048fef594fdf9d580d4db8c1fe6f809c70759`.
Its completion receipt has SHA-256
`4a3a019cb0e0f61c9b4f534c393214d6036ebcbafdce2a615728306990b90d08`.

It contains 400 mirrored pairs (800 games) of one-second MCTS with own model
priors enabled against the otherwise matching uniform-prior MCTS control. The
guided score is 85.25% (95% paired-bootstrap interval 82.81% to 87.63%), or
+35.25 percentage points from neutral (95% interval +32.81 to +37.63 points).
There were three capped games. The readout's cap-loss and cap-win sensitivities
remain positive, at +35.06 and +35.44 points respectively. It therefore
establishes that own priors matter relative to deliberately uniform MCTS in
this configuration.

It does **not** establish that search improves on the champion's direct policy
action: both arms in that study searched. The fresh guided-versus-raw study is
the only registered contrast that can answer that question.

## Matched-state diagnostic

The replay PASS artifact at
`/shared/scott-experiment/mcts-own-prior-191af857-20260917-replay-r2/PASS.json`
has SHA-256
`bc7b71d23de4721f8ceee51c586864c3e8ac7897717efb08c70006d43d30146f`.
It replays 16 fixed public decisions in both execution orders, producing 32
samples per arm. Guided and uniform root allocations differed in all 32 paired
replays; every guided replay exposed a non-flat, multi-action root. The
replay's timing gate passed: guided/uniform mean wall-time ratio 0.9966, with
guided p95 1.053 s and uniform p95 1.050 s.

Those replays show a concrete mechanical effect—priors reshape finite-budget
visit allocation—without treating replayed decisions as independent strength
samples. They also do not say whether following the policy more closely is
better. That is why the live study measures fresh mirrored games against raw
policy, and why its final readout retains direct override and timing telemetry.

## Consequence for the active decision

The completed evidence justifies testing guided MCTS rather than assuming it
is redundant. It does not justify tuning PUCT, depth, or another search
parameter before the frozen guided-versus-raw result is complete. A useful
gain requires that study's preregistered +5 percentage-point lower-confidence
bound; otherwise the next work is a focused value/override diagnostic rather
than a parameter sweep.
