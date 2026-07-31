# C26 Damage-Composition Tail Prediction

## Outcome Amendment

This amendment withdraws the original reproduction, mechanism, and acceptance
claims. The historical rows `2900889/126`, `3400914/75`, `1500037/28`, and
`1500174/72` are absent from the retained certification archive, so none is
an admissible exact-row clearance. The two bounded gated runs for the first
two rows stopped before their required steps (119 of 126 and 66 of 75), which
is a refusal rather than a negative or positive replay result.

The original statement that `2900889/3` and `2900889/93` should remain
divergent is also withdrawn. Those steps were not independently adjudicated by
an exact retained-row replay, so they are undispositioned and must not be used
as an acceptance control or a clearance claim.

## Disposition

The complete retained-population reread showed that the experimental matcher
changed 91 verdicts relative to the pinned baseline: 88 divergent rows became
matched, but three previously matched rows became divergent. The experiment is
rejected. Its pre-state/called-move support widening, generic capped-source
promotion, and cumulative-tail comparison rule are all removed; no production
matcher code survives this lane.

The authoritative evidence is the machine-readable
[C26 readout](c26_damage_composition_tail_readout.json). It pins the immutable
baseline commit, full engine fingerprint, opaque archive-shard digests, exact
identity accounting, and the zero-delta final reread contract.
