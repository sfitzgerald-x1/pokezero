# The v4 rotation's real scope, measured

## Why this file exists

The plan this work serves opens with:

> Nothing here blocks the rotation being correct — gate 1 already proves it breaks zero tests by
> diff-proven set equality against main.

and describes the last PR as "the one-line rotation plus the bucket-B invariants it was designed to
be". **That premise does not hold for the rotation as a separable change.** This records the
measurement, because the difference changes what the remaining PR is.

## The measurement

Two full-suite runs, same tree, same interpreter, same command, on `main` at `07f0c238` (with all
five splits landed: #1239, #1243, #1232, #1228, #1230, #1244):

```sh
# baseline: main untouched
.venv/bin/python -m pytest tests/ -q -p no:randomly
#   11 failed, 6024 passed, 48 skipped, 2 xfailed, 18604 subtests passed in 947s

# rotated: main + exactly one line changed in src/pokezero/observation.py
#   OBSERVATION_SCHEMA_VERSION = OBSERVATION_SCHEMA_VERSION_V2_2
#   -> OBSERVATION_SCHEMA_VERSION = OBSERVATION_SCHEMA_VERSION_V4
.venv/bin/python -m pytest tests/ -q -p no:randomly
#   52 failed, 5983 passed, 48 skipped, 2 xfailed, 18604 subtests passed in 920s
```

Ids normalised identically on both sides and set-subtracted:

```
rotated ids   52
baseline ids  11
ATTRIBUTABLE  41
```

So the one-line rotation breaks **41** tests, not zero.

## Composition

| group | count | files |
|---|---:|---|
| schema-spec test files, where the identity pins live | 11 | test_observation*.py, test_turn_merged_encode.py, test_observation_schema_flag.py, test_transitions_fold.py |
| **consumers** | **30** | test_interaction_registry 10, test_batch_replay 4, test_remote_self_describe 4, test_distributed_training 2, test_neural_selfplay 2, and 7 files with 1 each |

The 11 are expected and largely sanctioned: a class-(iii) pin asserts `default == V2_2` and MUST break
under a rotation. (An exact pin/consumer split cannot be computed on `main` yet, because the rubric
names the POST-split pin names, which live in #1247 and are not merged — on `main` the pre-split names
exist. The file-level cut above is robust to that; a pin-level cut is not, and is deliberately not
claimed.)

The **30 consumers** are the part the premise did not anticipate, and they are the work.

## Why gate 1 said zero, most likely

Gate 1 measured #1231 as it stood: 42 files, 3710 insertions. The rotation travelled together with
the fixes for everything it broke, so the SET of failures was equal to main's — truthfully. Splitting
the rotation out, per step 1 of the plan, removes those fixes and the 41 reappear. The set equality
was real; it was a property of the whole branch, not of the one line.

That is the same error class this whole programme has been retiring: a figure that is true against the
baseline it was measured on and false against the one that matters.

## What the drill said, and why it is consistent

The rotation drill reported 12 attributable breakages under a SYNTHETIC rotation, 7 sanctioned. That
is not in conflict with 41:

- the drill's `identical` arm holds SHAPE constant by design — the synthetic schema is a clone of the
  outgoing default differing only in its version string — so it measures the NAMING half only, and
  says so;
- a real v4 rotation changes shape as well: 155 → 132 numeric, 51 → 41 categorical, 151 → 23 tokens;
- the drill's `differ` arm samples the shape half exactly ONE site wide (`test_from_dict_width_...`),
  which is a fact about how little of this suite pins the default's width, recorded in the verdict.

The consumers dominate the 41 precisely because they depend on shape, which the naming arm cannot see
and the differ arm barely samples. The instrument was not wrong; its scope was narrower than the
question, and the verdict states that scope.

## Consequence for the remaining PR

**#1231's remainder is the rotation plus ~30 consumer sites, not one line plus invariants.** It should
be scoped and reviewed as that. Folding 30 fixes into a PR whose description says "one-line rotation"
would reproduce exactly the reviewability problem step 1 was written to solve — a 42-file PR spanning
six concerns, which is where this began.

Recommended: land #1247 first, then rescope #1231 against this list, splitting the consumers by
concern if they do not share a root cause. The 10 `test_interaction_registry` failures are a single
file and may share one; that is worth checking before assuming 30 separate fixes.
