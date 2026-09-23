# Guided MCTS diagnosis: branch-prior and override evidence

**Status: updated, 2026-09-22.** This records terminal-only model-leaf
evidence and the completed branch-prior/override audits. It is not a strength
claim. The leaf evidence supports prioritizing a policy-consistent value-head
measurement before assuming that deeper search will improve play.

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
missing move arm, switch arm, or `None` engine-option shape.  A fresh replay
on the later, witness-instrumented source did **not** reproduce the fallback,
so the old ledger cannot be retroactively assigned a concrete topology.  The
current source-matched leaf experiment uses commit `195a89c5`, including that
witness, but its purpose is the independent leaf-value intervention rather
than a fabricated reproduction claim.  A mapping repair is not justified
until a source-bound witness identifies a repairable category and a controlled
comparison shows an effect on root actions or outcomes.

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

This is a weak ranking signal on the states the tree actually reaches, plus a
large calibration offset. It explains how a value head can be competitive with
Foul Play or another one-step chooser yet still fail to improve MCTS: direct
choice needs the correct ordering near the live root; MCTS repeatedly ranks
deep, off-policy leaves and compounds small ordering errors through selection
and backup.

The R4 result is source-bound to `a60da9eb`, not the later `195a89c5` source
used by the failed opponent-prior R3 attempt. It is strong evidence about the
mechanism but not a replacement for a source-matched strength experiment. It
also uses uniform continuations, not Foul Play; it therefore does **not** prove
that the head disagrees with Foul Play on the same leaves.

## Opponent-prior applicability correction

The source-matched R3 opponent-prior job was terminally failed and is not
bankable. Its first failure was not a generic mapping error: at seed
`2026092006`, round 28, the public determinization could no longer prove the
opponent's active party permutation and reported
`lost_active_permutation`. The implementation correctly refused to invent an
opponent action-head ordering, but strict mode also discarded the entire
mirrored game.

The successor experiment keeps strict mode. It permits only one audited
exception: exactly one root fallback with a null acting-seat fallback reason
and matching source/native status `lost_active_permutation`. Self-prior
fallbacks, mapped-action failures, multiple root fallbacks, and every other
opponent-order status remain terminal. Its separate applicability contract can
report whether all retained fallbacks meet that exact condition; it cannot be
used as a strength promotion.

## Current causal intervention and decisive next evidence

The source-matched own-prior control is complete: it has four terminal,
zero-restart paired seeds under source commit `195a89c5`.  Its counterpart is
currently running with the same source, checkpoint, seeds, raw incumbent and
own-prior configuration.  It changes only the leaf evaluation path: instead
of trusting the learned leaf at the frontier, it runs bounded terminal
continuations (`rollout_leaf_eval=true`, 12 rollout workers).  It is therefore
a mechanism diagnostic, **not** a game-strength claim and not a Foul Play
comparison.

The result has a sharp interpretation:

1. If the terminal-continuation leaf improves the complete matched pairs, it
   supports and prioritizes the learned-leaf mechanism for these matched
   seeds. The next correction is a policy-consistent value target/calibration
   study, followed by a larger fresh durable paired strength evaluation before
   any general conclusion.
2. If it is neutral or worse, the current evidence does not support blaming
   the value head alone. The next causal arms must isolate backup,
   exploration, and opponent modeling while preserving the source-matched
   control.
3. Any promoted search or value correction still needs a separately
   registered, durable paired GPU strength study with a
   zero-unexpected-fallback contract. Partial seed output is never evidence
   for either conclusion.
