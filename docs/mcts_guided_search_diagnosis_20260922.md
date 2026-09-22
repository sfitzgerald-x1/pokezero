# Guided MCTS diagnosis: branch-prior and override evidence

**Status: preliminary, 2026-09-22.**  This records the evidence available
before the terminal-only leaf-shadow study completes.  It is not a strength
claim and it does not recommend retraining the value head yet.

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
missing move arm, switch arm, or `None` engine-option shape.  The active
successor records those categories per seat while preserving the same search
semantics.  A mapping repair is not justified until that witness identifies a
repairable category and a controlled comparison shows an effect on root
actions or outcomes.

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

## Decisive next evidence

The active source-bound job `mcts-model-leaf-shadow-752b2f84-r2` runs the
production model-value tree unchanged and records, for exactly reached leaves,
the learned value beside an independent uniform continuation **only when that
continuation terminates**.  Cap and dead-end continuations are durable
coverage exclusions, never labels.  Its held-out moments will distinguish:

1. poor leaf calibration or ranking on the tree-generated distribution, which
   supports prioritizing value-head data/training; from
2. adequate leaf calibration with weak overrides, which directs the next
   intervention toward opponent modeling, tree allocation, or backup/search
   mechanics.

The same job also carries the new branch action-topology witness, so it can
separate a repairable mapping hole from a benign or irreducible option shape.
