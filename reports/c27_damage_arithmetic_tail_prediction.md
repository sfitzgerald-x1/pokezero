# C27 prediction: retained damage-arithmetic tail

## Scope

The C26 materialization attestation established that all 21 candidate worlds for
the seven retained boundaries carry the same stored stats, active boosts, HP,
ability, item, status, type, weather, side conditions, and active volatiles at
the Python-to-Rust construction seam. This investigation begins downstream of
that seam.

The targets are `2800700/20`, `3301036/26`, `3401017/55`, `3500021/19`,
`3300207/69`, `3001000/57`, and `3300122/21`.

## Prediction

The retained magnitude differences will not establish one shared native Gen 3
damage-arithmetic defect. For each reproducible target, either:

1. the exact Showdown HP delta belongs to an instruction branch generated from
   the same fully attested state, making the mismatch an event-component
   attribution or branch-selection issue; or
2. the target cannot be compared exactly because the public step leaves a
   bounded hidden counter unresolved, which is recorded as a comparison limit,
   not evidence of arithmetic divergence.

## Falsification criteria

A native arithmetic defect is proven only if one target supplies all of the
following from the same pre-action state:

1. a transcribed Showdown oracle input including move, category, base power,
   attack and defense stats, boosts, ability, item, status, types, weather,
   side conditions, and relevant volatiles;
2. a concrete Showdown HP result within the oracle's legal Gen 3 roll support;
3. no native instruction branch with the same visible transition and any legal
   roll value; and
4. a common implementation mechanism shared by at least two targets.

Absent those conditions, no production arithmetic patch is justified. The
result should instead preserve a reusable replay attestation with its explicit
comparison limits.
