# C26 Damage-Composition Tail Results

## Final Disposition

This is an evidence-only lane. The final matcher is byte-identical to immutable
baseline commit `d7a9c1a932366ef4b751dd5894ddfb61b91e58cd`; no experimental C26
production behavior remains. The full engine fingerprint is
`992186c85b4809f768830fa544209d5c31fee1bbc06be1587fe68698d074ba6e`.

Current main's retained-repro producer fields `party_display`, `slot_sides`,
and `turn` are included in that baseline source. They are provenance added
before this lane, not C26 branch behavior.

The retained population contains 3,821 rows in eight hash-pinned opaque
shards. The deterministic baseline reread tally is 3,599 `diverged`, 221
`matched`, and 1 `skip_lossy`. The final matcher rereads the same population
with exactly zero `diverged -> matched` and zero `matched -> diverged` changes.

## Rejected Experiment

The discarded experiment produced 3,514 `diverged`, 306 `matched`, and 1
`skip_lossy`: 88 apparent clearances and three regressions. The clearance
classes and hook-isolation observations are recorded exactly in the
[C26 readout](c26_damage_composition_tail_readout.json). The generic
capped-source promotion is associated with 87 of the 88 apparent clearances,
but removing it leaves all three regressions divergent. The cumulative-tail
scale accounts for one further apparent clearance. Removing the
pre-state/named-callee support hook restores all three regressions to matched;
its broader clearance effects overlap the other hooks and are deliberately not
assigned a separable benefit.

Across the full-population one-hook ablations (experiment to ablation), removing
pre-state/named-callee support changes 62 `matched -> diverged` and 10
`diverged -> matched`; removing capped promotion changes 87
`matched -> diverged`; removing cumulative-tail scale changes 1
`matched -> diverged`.

The regressions are `2200760/86`, `2300983/40`, and `2700145/92`. Each matched
on the baseline, diverged with the full and no-promotion configurations, and
returned to matched without pre-state/named-callee support. This is sufficient
to reject that support widening. These rows do not establish engine defects,
and this lane makes no engine-defect claim.

## Ownership And Refusals

All 11 retained C15 WHAT identities match current main with the exact branch
counts in the readout. They are closed only by current main and have no shared
ownership with poison-tail, matcher, C27, or Rest lanes.

All four historical/control rows are uniformly
`refused_archive_row_absent`: `2900889/126`, `3400914/75`, `1500037/28`, and
`1500174/72`. The two bounded gated runs did not reach their required steps;
there is no guessed clearance. The Outcome Amendment in the prediction also
withdraws the unadjudicated `2900889/3` and `2900889/93` assertion.

## Verification

- The read-only `scripts/c26_damage_composition_verifier.py` requires the
  hash-pinned retained inputs, checks the build stamp, rereads both pinned-main
  and final matcher sources, and rejects any verdict delta.
- `tests/test_c26_damage_composition_readout.py` pins the complete evidence
  contract and production-code equivalence. Its optional archive reread skips
  explicitly when the retained inputs are unavailable; it never converts a
  missing archive into a passing replay.
