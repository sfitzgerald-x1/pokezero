# Guided MCTS diagnosis: branch-prior and override evidence

**Status: working diagnosis, 2026-09-23.** This records model-leaf evidence
and completed branch-prior/override audits. It is neither a strength claim nor
a demonstrated causal explanation. The next source-bound measurements are
designed to distinguish a leaf-target problem from a search-mechanics problem
before choosing a correction.

## What the large branch-prior count means

The sealed override audit at
`/shared/scott-experiment/mcts-sealed-override-audit-db4bc213-20260920-r7`
contains **1,414** branch-prior fallbacks in its readable, complete ledger.
An earlier 1,457 figure referred to a different predecessor artifact and must
not be carried forward as the result of this audit.

All 1,414 are `unmapped_action` interior fallbacks:

| Scope | Observed count |
| --- | ---: |
| seed | `2026092006` only |
| candidate seat | `p2` only |
| affected candidate decisions | 5 of 198 |
| affected rounds | 118, 119, 124, 126, 127 |
| root-prior fallbacks | 0 |
| other fallback reasons | 0 |

The root was valid at every affected decision (`prior_authority=true`, no
missing root action indices), and the MCTS action equalled the raw policy
argmax at all five.  Thus the number is repeated simulated-node fallback
events, not 1,414 invalid live choices.

With opponent priors disabled, the supported source-level mechanism is an
**acting-seat interior action-surface mapping gap**.  At an evolved simulated
state, an engine option lacks a corresponding policy action index.  The
whole simulated node intentionally receives uniform priors rather than a
partial, silently wrong prior vector.  Its value backup and the live root
remain intact.  This can make those late-game trees less efficient, but it
cannot explain a global strength result on its own.

The prior artifact predates the topology witness, so it cannot distinguish a
missing move arm, switch arm, or `None` engine-option shape. A fresh
source-matched GPU replay did **not** reproduce the fallback (zero root and
branch fallbacks for the affected seed), so the old ledger cannot be
retroactively assigned a concrete topology. Keep the witness telemetry, but
do not treat this count as the central explanation for search strength unless
it recurs with an action-level effect. A mapping repair is not justified until
a source-bound witness identifies a repairable category and a controlled
comparison shows an effect on root actions or outcomes.

## Fixed-opponent override audit

The same sealed artifact recorded every guided-MCTS override and replayed the
fixed public joint step with the opponent held fixed, then continued each arm
with the deterministic raw policy. This is a descriptive local
counterfactual, not an independent game-strength estimate: many records arise
from the same four games.

| Result | Count |
| --- | ---: |
| total observed overrides | 65 |
| comparable paired continuations | 62 |
| correctly excluded non-simultaneous choices | 3 |
| MCTS wins / raw loses | 11 |
| MCTS loses / raw wins | 13 |
| both win | 19 |
| both lose | 19 |
| net MCTS-minus-raw wins | -2 |
| immediate fixed-step terminal resolutions | 0 |

All 62 comparable records retained valid root-prior authority. Their median
root Q gap was `0.0306275`, and median root visit-share gap was `0.151459`.
There is no supported claim that either gap identifies a beneficial override:
the records are too small and correlated, their continuation policy differs
from an independent Foul Play evaluation, and the older construction was later
found to reset policy histories. That history caveat makes its quantitative
rankings unsuitable as a proxy for the deployed raw-policy suffix.

Reading the 62 immutable `sealed-override-audits/*/round-*.json` records
directly gives a more specific negative result: the point-biserial correlation
between root-Q gap and the MCTS-minus-raw continuation result is `-0.0651`;
the corresponding visit-share-gap correlation is `0.0036`. The outcome cells
have Q-gap medians `0.024038` (MCTS-only wins), `0.037226` (raw-only wins),
and `0.029674` (same outcome). This is not enough data for a calibrated
effect-size claim, but it rejects the simpler account that larger native
root-Q or visit separations were consistently identifying better overrides.
The fallback-heavy `2026092006` contributes 43 of those paired records and is
near neutral (MCTS 21 wins, raw 22); it is a stress case, not a proof that all
of its overrides are losing. The history-preserving root-action audit added
in source PR #1450 supersedes this audit for the action-level question: it
records the live root ledger, preserves exact policy histories, freezes target
selection before outcomes, and declares the continuation target.

## Completed terminal-only model-leaf shadow

The complete R4 artifact at
`/shared/scott-experiment/mcts-model-leaf-shadow-repair-a60da9eb-20260922-r4`
ran the production model-value tree and recorded the learned leaf value beside
a terminal uniform-continuation result only when that continuation terminated.
It contains four paired seeds (`2026092004` through `2026092007`), eight games,
and **11,318,403** labelled leaves. **71,177** capped or dead-end continuations
are durable coverage exclusions, not labels.

| Metric against terminal uniform continuation | Value |
| --- | ---: |
| Pearson correlation | 0.29349 |
| Kendall tau-b | 0.08983 |
| mean absolute error | 0.40896 |
| RMSE | 0.51165 |
| learned-value mean | 0.44777 |
| terminal-outcome mean | 0.51684 |

This is a weak ranking signal against one uniform-continuation target on the
states the tree actually reaches. It is suggestive, not a calibration finding:
the leaves are correlated tree observations and uniform continuation may not
be the value head's intended target. It supplies a plausible route by which a
one-step policy can be strong while MCTS is weak—MCTS repeatedly ranks deep,
off-policy leaves and compounds errors through selection and backup—but does
not establish that route as the cause.

The R4 result is source-bound to `a60da9eb`, not the later `195a89c5` source
used by the failed opponent-prior R3 attempt. It is strong evidence about the
mechanism but not a replacement for a source-matched strength experiment. It
also uses uniform continuations, not Foul Play; it therefore does **not** prove
that the head disagrees with Foul Play on the same leaves.

## Why the B2a continuation bank is not yet a value-head training corpus

The completed B2a final-enthalf readout is valuable action-selection evidence:
on 400 paired source-game seeds, choosing among the policy's top-three actions
by its fixed-opponent policy-continuation outcome improved the reported score
from 55.875% to 67.5% (paired difference +11.625 percentage points, 95%
normal interval +5.471 to +17.779 points). Its readout explicitly remains
`MEASURED-NOT-A-LICENSE`.

That bank cannot yet justify a value-head update. It records root candidate
outcomes under a fixed opponent and subsequent raw-policy continuation, but
not the model-visible successor observations or a replay-bound training-cache
join. It also names the earlier public source commit
`3b3593f61419ab51640607e1d0c140b2d5371668`, not the current source-matched
MCTS commit `195a89c5`. Thus it is neither a sample of the search frontier nor
a source-compatible supervised data set.

The correct next use of B2a is as a specification for a new, separately
validated corpus: replay each selected fixed-opponent successor under the
pinned runtime, bind its exact encoded observation and history to the target,
keep held-out source seeds separate, and compare that policy-continuation
target with uniform-terminal labels. A bank-only conversion would train the
head on unidentifiable states and would not answer the leaf question.

## Calibration is not a metadata-only correction on the current engine path

The source-matched champion receipt carries no value-calibration transform.
That matters because the current MCTS crate maps its raw tanh output directly
to tree probability `(v + 1) / 2`; the Python **model-leaf** search boundary
explicitly refuses a checkpoint carrying any non-identity
`value_calibration_transform`. This is a protective fence against evaluating
the checkpoint on one value axis in Python and a different one in the native
tree.

Consequently, a measured calibration offset would not authorize attaching an
affine or isotonic metadata transform to the current champion. The corrective
paths are either (a) train and export a source-bound value head whose output
is already on the required tree axis, or (b) add a native calibration seam
with parity and source-binding tests before using it. This is a deployment
constraint, not evidence that calibration is the current causal mechanism.

## Opponent-prior applicability correction

The source-matched R6 opponent-prior job was terminally failed and is not
bankable. Its first failure was not a generic mapping error: at seed
`2026092006`, round 28, the public determinization could no longer prove the
opponent's active party permutation and reported
`lost_active_permutation`. The implementation correctly refused to invent an
opponent action-head ordering, but strict mode also discarded the entire
mirrored game.

The successor R8 experiment is source-bound to `a889930d` and keeps strict
mode. It accepts exactly the two native ledger encodings of the same harmless
case, and only after native has explicitly assessed the opponent root as
`no_choice`: either one root fallback with a null fallback reason, or a zero
fallback omission. Both must carry the matching source/native status
`lost_active_permutation`. A missing opponent head, an unassessed root,
self-prior fallback, mapped-action failure, multiple root fallbacks, or every
other opponent-order status remains terminal. Its immutable applicability
contract requires a witnessed no-choice root; it cannot be used as a strength
promotion.

R8 reached the same `2026092006` / round-28 boundary with both narrowly
scoped flags enabled and still wrote a durable terminal refusal with exactly
one `opponent_order_lost_active_permutation` fallback. Given the source-bound
flags, matching status, null fallback reason, and count one, this proves the
native assessment at that root was **not** `no_choice`; otherwise the guarded
exception would have admitted it. The refusal is therefore correct rather
than a missed use of the exception. R8's partial siblings remain preserved but
the strength result is nonbankable. Do not widen this exception: that would
invent opponent-prior ordering at a root where the opponent actually has a
decision. The supported outcome is to leave opponent priors unavailable for
this public-history state until a separately proven reconstruction mechanism
exists.

## Completed source-matched rollout-leaf diagnostic

The first `195a89c5` four-pair readout established the direction but did not
transport the realized rollout partition. A fresh, terminal, zero-restart
replication under commit `9a21e1b0fdb1aa52d41a056be26a1abeed63cd76` fixed
that evidence gap. Its complete, hash-linked readout is
`/shared/scott-experiment/mcts-source-matched-leaf-readout-9a21e1b0-20260923-r2`.

It uses the same checkpoint, seeds, raw incumbent, own-prior MCTS settings,
search depth, simulation count, and one-second decision limit. The rollout
arm enables `rollout_leaf_eval=true` and its required twelve rollout workers;
the control trusts the learned model leaf. Across the four mirrored pairs,
model-leaf MCTS scored 25% (`0.5, 0, 0, 0.5`) and rollout-leaf MCTS scored 75%
(`0.5, 0.5, 1, 1`). The paired rollout-minus-model estimate is **+50 percentage
points**, with the registered four-pair bootstrap interval **+12.5 to +87.5
points**.

The fresh rollout arm recorded **280,045,024** actual rollout rows: 254,552,650
terminal (90.90%), 25,492,374 ply-cap fallback (9.10%), and zero dead ends. The
model control recorded zero rollout rows. Thus the intervention and its
terminal/cap/dead-end denominator are now explicit rather than inferred. It is
still not a pure-terminal-leaf correction: about one in eleven backed-up
rollouts used the registered capped fallback. Four paired games, however, are
not enough to establish a repeatable leaf advantage or attribute the effect to
the value head rather than the changed rollout target and backup behavior. The
result establishes only that this leaf-path intervention can change outcomes
in this small matched sample; it is **not** a strength claim, a Foul Play
comparison, or enough evidence to approve value-head retraining.

The result is nevertheless a warning against assuming that a policy matching a
strong one-turn chooser automatically supports effective MCTS. One-turn policy
quality ranks the live root. MCTS repeatedly ranks off-policy frontier states
and backs those rankings up through the tree, which can require a separately
accurate leaf evaluator—but the current data has not separated that possibility
from opponent modelling, exploration, and backup effects.

## Decisive next evidence

The next evidence is deliberately bounded:

1. Run a source-bound, seed-balanced root-action inventory, then a small
   predeclared action audit. At each selected root, compare raw policy,
   MCTS's actual choice, and the leading visited alternative under the same
   fixed immediate opponent step. Measure a policy-consistent suffix and a
   uniform-own-side suffix separately. This asks whether continuation policy
   materially changes the relevant action ranking; it is not a direct value
   test or strength estimate.
2. Add a deterministic, history-preserving raw-policy anchor before using a
   policy-consistent result as a deployed-baseline claim. Repeated suffix RNG
   seeds vary suffix randomness, not independent opponent moves or chance
   worlds; report that limitation and root-level uncertainty explicitly.
3. The decisive attribution measurement is then a bounded frontier-state
   calibration study: compare learned V with repeated outcomes from identical
   search-frontier states, with player perspective and policy histories
   preserved, under both policy-matched and uniform targets. Poor agreement
   with policy-matched outcomes supports value work. Good agreement with poor
   root choices shifts the next intervention to opponent modelling,
   exploration, or backup.
4. Promote only a selected correction to a separately registered, durable
   paired GPU strength study with a zero-unexpected-fallback contract. Partial
   seed output is never evidence for either conclusion.
