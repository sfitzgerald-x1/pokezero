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
leaf-shadow successor deliberately uses source commit `752b2f84`, which also
predates that witness; it must not be used to classify the mapping hole.  A
separate fresh, source-bound topology diagnostic is required after this
leaf-value readout.  A mapping repair is not justified until that witness
identifies a repairable category and a controlled comparison shows an effect
on root actions or outcomes.

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

## Decisive next evidence

The active source-bound job `mcts-model-leaf-shadow-752b2f84-r2` runs the
production model-value tree unchanged and records, for exactly reached leaves,
the learned value beside an independent uniform continuation **only when that
continuation terminates**.  Cap and dead-end continuations are durable
coverage exclusions, never labels. This is a real calibration measurement for
the tree-generated state distribution, but it estimates a *uniform-policy*
terminal target. It cannot by itself prove that the learned value head is
misaligned with the checkpoint-policy/Foul-Play continuation target that
matters to strength.

The second active source-bound job,
`mcts-model-leaf-shadow-topology-63d0eca5-r3`, repeats that observational
terminal-only measurement from source commit
`63d0eca5b8c625670e7c9ebae8293feff6da9d8f` and adds an observational
unmapped-action topology witness. It will classify each fallback as a
move/switch/other action-map shape without changing tree selection. A mapping
repair remains contingent on that classification and on a controlled action or
outcome effect.

The decisive leaf-value experiment after these jobs is therefore a
policy-consistent terminal continuation measurement over a preserved sample
of native frontier states. Only if the value is poorly ranked against that
target should we prioritize value-head retraining; otherwise the leading next
targets are opponent modeling, tree allocation, or backup/search mechanics.
