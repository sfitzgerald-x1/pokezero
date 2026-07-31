# C26 Damage-Composition Tail Results

## Current Result

The matcher implementation has focused tests for its two intended composition
rules, but the historical target identities are **not currently replayable as
identities**. The checked-in C26 prediction names `2900889/126` and
`3400914/75` without retaining their full protocol/state payload or pinning the
Showdown revision that generated them. A current gated one-game run reaches
only 119 full-round boundaries for the former and 66 for the latter, so neither
named step occurs. A zero-divergence run that never reaches the named boundary
is not evidence that the row matched.

The machine-readable [C26 readout](c26_damage_composition_tail_readout.json)
records the commands, boundary counts, every C15 WHAT identity, and the
fail-closed disposition. This lane changes matcher evidence only; it does not
change Rust battle mechanics or the Python battle engine.

## Mechanism

The matcher now:

1. prices the identified direct move from the rendered `|move|` event, so a
   named Sleep Talk callee uses its own legal roll support instead of the
   caller's zero base;
2. carries preceding ordered-tail damage scale into capped full-heal comparison;
   and
3. retains the original source of an engine terminal residual, permitting an
   observed non-terminal counterpart only after a legal selected direct roll
   and only when both sources match.

The last point was an honest refinement of the pre-registration. The initial
callee-support prediction explained the observed 82-damage Earthquake but did
not account for the representative 84-damage branch capping the subsequent
poison tick. The repair remains narrow because it requires an identified,
legal direct hit and the same residual source after that hit.

## Tested Matcher Shapes

The focused matcher tests establish only these implementation-level facts:

- a rendered named Sleep Talk callee is priced from the boundary pre-state;
- a capped full-heal keeps the earlier same-tail direct-damage scale; and
- a terminal residual can be relaxed only after a legal selected direct hit
  and only when its original source matches.

They are not substitutes for an exact retained-row replay.

## Identity Boundary

The C15 WHAT matrix is explicit rather than family-derived:

| Disposition | Exact identities |
| --- | --- |
| Closed by PR #980/current main | none |
| Active matcher scope, not exactly cleared | `2000298/23`, `2000561/67` |
| Exact poison/matcher tail cleared | none |
| C27/Rest | none |
| Still unresolved or refused | all 11 C15 WHAT identities |

`2601196/46` contains poison, but its divergence is an Ice Beam direct damage
gap and not C26's matching-source terminal-cap tail. It remains divergent.

## Verification

- `tests/test_transition_differential_matcher.py`: composition and
  source-isolation pins.
- `tests/test_c26_damage_composition_readout.py`: public identity-matrix and
  fail-closed replay contract.
- `tests/test_public_invariant.py`: public repository invariant guard.
