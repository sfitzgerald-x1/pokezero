# Guided MCTS versus raw-policy strength study

## Question

Does the final champion's policy-guided MCTS produce useful playing strength
over the same checkpoint's direct deterministic policy action, rather than
only outperforming an intentionally uniform-prior MCTS control?

The completed own-prior study is a prerequisite, not an answer to this
question.  Its source-bound derived readout is
`/shared/scott-experiment/mcts-own-prior-191af857-20260917-strength-derived-readout/READOUT.json`
(SHA-256 `5eb3433cd11e20fbeb5cc106e4c048fef594fdf9d580d4db8c1fe6f809c70759`).
It establishes that using model priors in the registered search configuration
matters relative to uniform priors.  This protocol tests the resulting search
against the raw policy itself.

The independent artifact audit, including the fixed-public-state diagnostic,
is recorded in [the own-prior evidence audit](mcts-own-prior-evidence-audit-20260917.md).

## Frozen comparison

* Candidate: model-leaf engine MCTS with `model_priors=true` and
  `use_opponent_model_priors=false`.
* Baseline: the same checkpoint's deterministic masked argmax policy
  (`epsilon=0`, `temperature=1`, no family-gated selection, no search).
* Both sides use the same verified champion checkpoint, source image, Showdown
  source, and checkpoint vocabulary/masks.
* The candidate uses the registered practical configuration: one-second soft
  decision budget, 64 ms native-batch guard, four worlds, 256 simulations per
  world, batch 16, depth 2, and PUCT constant 1.4.
* Every seed is played twice with candidate MCTS seated once as p1 and once as
  p2.  The two games share that seed's teams; pairs, rather than individual
  games, are the scoring units.

## Run shape and fault tolerance

The confirmation set is 400 fresh, predeclared seed pairs (800 games), split
into two disjoint 200-pair CPU-only workers.  Each game is atomically written
as an immutable record immediately after it ends.  A new launcher attempt may
resume a root only after an interruption before a terminal runner receipt;
already written game records are revalidated and reused.  A partial pair is
never scored.  A nonzero terminal runner exit, malformed receipt, source drift,
or restart is terminally nonbankable rather than silently retried.

Before the confirmation run, the exact same source/image/contract must pass a
four-pair (eight-game) dry preflight proving both seats, raw-selector receipt,
MCTS source binding, game persistence, and resume semantics.  The preflight is
not scored as confirmation evidence and its seeds are excluded from the 400
fresh pairs.

## Acceptance and interpretation

The finalizer may publish a result only when it has all 400 complete pairs,
both worker terminal receipts with exit zero and no restarts, matching immutable
source/checkpoint/engine/Showdown identities, and all required raw-selector and
MCTS receipts.

The primary statistic is the candidate's mirrored-pair score against neutral
50%, with a predeclared 10,000-resample paired bootstrap, seed 2026091701 and
95% confidence interval.  Caps are reported three ways (exclude, score as
loss, score as win) without changing the primary rule.  Candidate and baseline
decision-wall samples are retained; the report includes p50/p95 and the
candidate-to-raw mean latency ratio.

* **Useful search gain:** the 95% lower bound on the candidate score minus 50%
  is positive and at least +5 percentage points.  Register a separate
  external-opponent validation before proposing any production choice.
* **No demonstrated search gain:** otherwise do not tune MCTS around this
  result.  Preserve the readout and make value-head quality / decision-quality
  diagnostics the next investigation.

This is one registered contrast, not a parameter search.  It does not change
the production policy or claim a result for other search budgets or
policy-guidance schemes.
