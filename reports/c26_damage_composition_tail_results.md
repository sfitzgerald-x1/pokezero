# C26 Damage-Composition Tail Results

## Result

The two requested target rows replay as matched under the strict public
differential matcher:

| Retained row | Before | After | Finding |
| --- | --- | --- | --- |
| `2900889/126` | divergent | matched | matcher composition defect |
| `3400914/75` | divergent | matched | matcher composition defect |

This lane changes matcher evidence only. It does not change Rust battle
mechanics or the Python battle engine.

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

## Retained-Shape Checks

Two related rows from the current public retained archive remain divergent:

| Retained row | Outcome | Reason it remains outside this change |
| --- | --- | --- |
| `1500037/28` | divergent | Moonlight attribution differs from an item-capped heal. |
| `1500174/72` | divergent | the rendered Sleep Talk branch marks the callee damage as unidentified, so no named-move support is available. |

These misses are expected and are not relabeled or absorbed by this lane.

## Verification

- `tests/test_transition_differential_matcher.py`: retained-shape and negative
  source-isolation pins.
- Fresh deterministic replays: `2900889`, `3400914`, `1500037`, and `1500174`.
- Broader mapper/fidelity tests and the repository public-invariant guard are
  recorded with the implementation commit.
