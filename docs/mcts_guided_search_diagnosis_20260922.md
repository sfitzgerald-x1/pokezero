# Guided MCTS diagnosis: branch-prior and override evidence

**Status: updated, 2026-09-24.** This records model-leaf evidence and the
completed branch-prior/override audits. It is not a strength claim. The newer
source-root results below supersede the earlier small-sample conclusion that
the value head is the leading explanation.

## September 24 correction: actual raw-policy anchor and current decision

The previous multireply result was not a raw-policy-versus-search comparison:
its `raw` candidate reused the historical `search_argmax`.  The corrected
runner binds `raw_policy` to the sealed policy `model_argmax`, replays that
choice at the public boundary, and records aliases when two arms select the
same action.  The old artifact must not be used to decide whether MCTS beats
the raw policy.

The corrected terminal artifact is
`/shared/scott-experiment/mcts-source-root-multireply-raw-anchor-8a4a-20260924-r1`.
It completed seven deliberately selected model-leaf/rollout-leaf disagreement
roots, eight sampled raw-policy opponent replies per root, and 16 seeded
suffixes per reply, all terminal and uncapped.  The executed target is
`policy_consistent`; its manifest accidentally listed `uniform_own` as well,
which a follow-up contract repair prevents for future runs.  The completed
root evidence itself contains only the target that the runner actually
executes, so it is usable **only** as that single-target diagnostic.

| Candidate | Subject wins | Trials | Rate |
| --- | ---: | ---: | ---: |
| actual raw policy | 425 | 896 | 47.4% |
| model-leaf MCTS choice | 314 | 896 | 35.0% |

The paired cells are 201 raw-only wins, 90 MCTS-only wins, 224 both wins, and
381 both losses.  Raw policy was better at six roots and tied at the seventh.
These are repeated continuations from seven source positions, not 896
independent games; the opponent policy is sampled raw policy rather than Foul
Play.  They therefore do not quantify global strength.  They *do* rule out
promoting the current model-leaf choice at these changed roots and shift the
next diagnostic toward exact root-action/Q/visit accounting before value-head
retraining.

The table above is deliberately a **wins-only** count. It must not be read as
a game score because it omits draws. Recomputing every terminal outcome with a
draw worth one half point gives:

| Choice | Score | Rate |
| --- | ---: | ---: |
| actual raw policy | 433 / 896 | 48.3% |
| model-leaf MCTS choice | 325 / 896 | 36.3% |
| rollout-leaf MCTS choice | 433 / 896 | 48.3% |

Raw policy therefore exceeds the model-leaf choice by **12.05 percentage
points** on this selected panel. Rollout leaf does not establish a gain beyond
raw policy: it selected the exact raw action at six of seven roots, and its
different seventh action lost every measured continuation. Its supported
effect here is recovery from damaging model-leaf overrides, not discovery of a
better-than-policy action.

The persisted leaf-ablation witness had also discarded each allocation arm's
request action index after native search exported it.  That made the old Q and
visit rows impossible to join reliably to the selected raw/MCTS actions.  New
artifacts now retain unique in-range `action_index` values and fail closed if
they are missing, duplicated, or outside the nine-action policy surface.  No
claim about a selector/backup defect should be
made from an allocation table without that join.

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

The event count is highly concentrated rather than spread over those five
choices:

| round | repeated interior fallback events |
| ---: | ---: |
| 118 | 5 |
| 119 | 275 |
| 124 | 626 |
| 126 | 504 |
| 127 | 4 |

Thus 1,405 of 1,414 events (99.4%) were produced while exploring only three
late-game roots (rounds 119, 124, and 126). Each ledger records four native
belief-world invocations; the round-124 ledger coalesced two equivalent
belief records for one invocation. These are tree-internal occurrences, not
additional public choices.

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

The prior artifact predates the topology witness, so it cannot assign every
one of the 1,414 occurrences to a concrete shape.  A later source trace did,
however, expose a repairable member of the same class: filtering
`Choices::NONE` compacted sparse engine move slots, so a legal M2/M3 was
looked up at the wrong policy action slot.  The first repair preserved engine
slot holes, but independent review found the companion encoder stopped at the
first placeholder and again erased a later legal move.  The proposed
two-layer repair would preserve optional fixed slots through both the action
surface and action tokens, with a regression that asserts sparse legal M3
identity, presence, activity, and legal-mask alignment.  That repair is pending in PR
#1464 rather than landed on `main`; it is a correctness repair, not yet a
current-source claim.  It does
**not** yet prove that it explains every historical fallback or the global
strength plateau.  A fresh source-bound witness is still required to measure
its root-level effect.

## Fixed-opponent override audit

The same sealed artifact recorded every guided-MCTS override and replayed the
fixed public joint step with the opponent held fixed, then continued each arm
with the deterministic raw policy.  This is a useful local counterfactual,
not an independent game-strength estimate: many records arise from the same
four games.

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

All 62 comparable records retained valid root-prior authority.  Their median
root Q gap was `0.0306275`, and median root visit-share gap was `0.151459`.
There is no supported claim that either gap identifies a beneficial override:
the 62 records are too small and correlated, and the local continuation
policy intentionally differs from an independent Foul Play evaluation.

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
of its overrides are losing.

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

This is **consistent with** a weak ranking signal on the states the tree
actually reaches, plus a large calibration offset. It does not by itself prove
that the learned head is misranking its intended target: the labels are one
uniform-continuation draw per leaf, include continuation randomness, and come
from eight correlated games. It nevertheless supplies a concrete mechanism
worth testing: a head can be competitive with Foul Play or another one-step
chooser yet fail to improve MCTS because direct choice needs the correct
ordering near the live root, while MCTS repeatedly ranks deep, off-policy
leaves and compounds errors through selection and backup.

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

This is historical diagnostic evidence, not the current intervention choice.
Its four pairs are too few, and the corrected raw-anchor multireply comparison
above does not reproduce a general rollout/model advantage.  Retain it as a
reason to inspect leaf behavior, not as authorization to retrain the value
head or replace the deployed search leaf.

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
rollouts used the registered capped fallback. This is causal evidence that
changing the leaf-evaluation path materially changed these matched outcomes;
it is **not** a strength claim, a Foul Play comparison, or enough evidence to
approve value-head retraining.

The result is nevertheless incompatible with the claim that a policy that
matches a strong one-turn chooser must automatically support effective MCTS.
One-turn policy quality ranks the live root. MCTS repeatedly ranks off-policy
frontier states and backs those rankings up through the tree; that requires a
separately accurate leaf evaluator.

### Realized search-work audit

The configured `search_time_ms=1000` is not an end-to-end one-second decision
wall in this diagnostic.  A direct read of the complete per-seed summaries
under `/shared/scott-experiment/mcts-source-{model,leaf}-telemetry-9a21e1b0-r1`
shows that every candidate decision in **both** arms reached exactly 16,384
native iterations (four worlds times the 4,096 simulation cap) and exactly four
searched worlds.  Therefore the comparison did not give the rollout arm more
native search visits per decision.

The arms contain different numbers of decisions because their games have
different lengths (428 model-leaf decisions versus 322 rollout-leaf decisions),
so totals must not be compared as an equal game-work denominator.  On a
per-candidate-decision basis, model leaf recorded 26,615 model evaluations and
a 15.86 s median end-to-end wall (p95 23.66 s); rollout leaf recorded 25,539
model evaluations, 27,178 priced leaves / 869,705 rollout trials, and a 23.73 s
median wall (p95 45.27 s).  The latter used the registered twelve rollout
workers and still had the 9.10% capped-rollout fallback fraction above.

So the four-pair result is evidence about the **whole rollout leaf-evaluation
path at matched native search effort**, not an isolated numerical replacement
of the learned value.  It neither proves a value-head target nor licenses a
latency claim.  The next causal probe must hold the exact source roots, worlds,
priors, simulation cap, and random streams fixed; it should use the existing
rollout seam before building frozen-frontier machinery.

## Decisive next evidence

The result has a sharp interpretation:

1. Complete the predeclared root-action recovery audit. Its sixteen saved
   roots are measurement units, not sixteen independent observations: they
   cluster into four source seeds and eight games. It measures local target
   sensitivity under sampled-policy and uniform-own continuations with the
   opponent's committed action held fixed. Neither target is deployed raw
   argmax or continued MCTS, so it cannot choose a training target by itself.
2. On a small prespecified mix of beneficial and harmful overrides, reproduce
   the original search with worlds, priors, opponent assumptions, the fixed
   simulation cap, and random streams held fixed. Use the existing rollout
   seam first and measure backed-up Q, visit allocation, and selected move.
   Then validate any changed choice on separate continuation draws. Only build
   frozen-frontier repricing if this leaves a consequential ambiguity between
   evaluation ranking and allocation. Deployed selection aggregates normalized
   **visit** share, so repricing a frozen tree alone cannot show that the chosen
   move would change.
3. Validate any newly selected moves on separate continuation draws. If
   corrected leaf values improve both backed-up rankings and selected moves,
   prioritize a source-bound value-target study. If Q values improve but visit
   selection does not, investigate exploration/visit aggregation; if the
   difference appears only under hidden-state or opponent changes, prioritize
   beliefs/opponent modeling.
4. Any promoted search or value correction still needs a separately
   registered, durable paired GPU strength study with a
   zero-unexpected-fallback contract. Partial seed output is never evidence
   for either conclusion.

### Protocol for the next actual-search intervention

The next probe is deliberately a **search intervention**, not another audit.
After the R4 readout, its root-selection rule, root count, confirmation source
seeds, continuation policy, and success criterion must be fixed before work
begins.  It will then replay a small mix of MCTS-better, raw-better, and tied
override roots selected by that rule into the same source-bound engine
configuration twice:

* a model-leaf control that must reproduce the original root allocation,
  backed-up Q values, and selected action; and
* a rollout-leaf arm using the already-shipped leaf-evaluation seam, with the
  source root, checkpoint, worlds, own/opponent priors, depth, and simulation
  cap unchanged.

The registration must name three separate seed families: simulator/world-search
seeds, rollout-evaluator seeds, and held-out continuation seeds.  The two
search arms share only the simulator/world-search family.  In particular, an
evaluator must never consume the generator that drives tree expansion or belief
world construction; equal top-level seed values do not prove an equal search
when one evaluator makes extra random draws.

Every replayed root must write an independent, create-only terminal unit as
soon as its control and intervention are complete.  A later source-game failure
cannot erase an already completed root.  The terminal unit must bind the
replayed public-prefix witness, both exact engine configurations, all three
seed schedules, the model-control reproduction check, allocation/Q/action
outputs, and the separately drawn continuation outcomes.  The final root set
will reserve fresh source seeds for confirmation; the R4 roots are diagnostic
case studies rather than an independent strength sample.

The R4 sampled-policy and uniform-own continuations remain useful diagnostic
targets, but neither is deployed raw argmax or continued MCTS.  Their results
therefore cannot by themselves select the intervention's continuation policy
or a training target.

The decision rule is explicit.  If rollout leaf changes an action and that
action is better on the held-out continuations, prioritize a source-bound
value-target study.  If it improves Q rankings against independent
continuation estimates but does not improve action selection, investigate
exploration or visit aggregation.  If the effect only appears under particular
hidden-state or opponent assumptions, investigate belief or opponent
modelling.  Frozen-tree repricing is justified only if that controlled
intervention leaves a specific evaluation-versus-allocation ambiguity; a
value-head retrain separately requires positive evidence for a suitable value
target.
