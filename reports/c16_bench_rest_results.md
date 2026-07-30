# C16 Bench Rest Provenance Results

## Result

The motivating archived identity, seed `2000281` step `99`, clears after the
parser preserves and refunds the public Sleep Talk attempt state for the
benched Rest sleeper.

The same deterministic one-game replay was run at the unpatched `origin/main`
commit and at this branch tip. Both runs used strict matching, complete repro
retention, the same Showdown checkout, and the same 41-patch engine build.

| Build | Full-round boundaries | Measured | Matched | Diverged | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| unpatched `origin/main` | 96 | 93 | 92 | 1 | `roll_scaled_component` at the motivating identity |
| patched branch | 96 | 93 | 93 | 0 | clearance |

The unpatched report hash is
`82e052ca06f227e1198b19d6cae84d2b850fc6b7ad0549929511fb9519c6ed26`.
The patched report hash is
`af3a3a2429af1233424c7fb0ee9bb4cf3fa9a420b7af113fcc58d7c2c5c8d58a`.
The reports are verification outputs, not tracked fixtures.

## Reproduction

Run `scripts/engine_transition_differential.py` once from the unpatched commit
and once from this branch with:

```text
--showdown-root <showdown-root>
--games 1
--seed-start 2000281
--max-steps 101
--keep-repro 100
--repros-per-game 100
```

The command exits `1` on the unpatched commit because one retained divergence
exists and exits `0` on the patched branch.

## Coverage Boundary

This is a targeted identity clearance, not a population-wide certification
claim. The broader tests pin:

- ordinary Rest attempt counts;
- active Sleep Talk/Snore states remaining fail-closed;
- the exact benched refund and one-time switch-in application;
- a later ordinary sleep turn cancelling the trailing refund;
- snapshot restoration;
- wake, faint, per-mon cure, and team cure cleanup;
- exact opponent-side public-row materialization.

The fresh certification re-sweep remains the global regression and recurrence
bound.
