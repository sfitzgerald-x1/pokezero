# Search over raw confirmatory statistical kernels

These kernels support prospective Phase B design and synthetic calibration.
They do not admit collection, validate game provenance, choose a test or stake,
authorize retries, or establish playing strength. Phase A's independently
reviewed per-arm bootstrap screening is a different procedure and is unchanged.

## Original primary cells and stopping

`adjudicate_primary_cells` consumes an exact prefix of the frozen seed order.
Each record supplies `seed`, `primary_cells`, `refused` and `provenance_ok`.
The primary mapping has the six `(arm, subject_seat)` keys for raw, incumbent
and reference against Foul Play, each from p1 and p2. Only a caller-validated
COMPLETE win/draw/loss cell contributes a point score. Missing, refused, capped
and unvalidated cells retain their full score interval [0, 1]. Validation flags
are caller attestations, not independently verified by this numerical module;
the owning study must validate opponent, arm, source and execution provenance.

Both contrasts reuse the same raw control. A missing raw game weakens both
lower endpoints, retaining known candidate scores. The cell entry derives
contrasts itself and rejects caller-supplied contrast overrides. Any uncertain
primary cluster counts against the refusal cap even if its supplied refusal
flag is false; secondary-only failures can also count without erasing known
primary scores. The default cap is two refused clusters. Provenance/worker-state
failure is immediately terminal, not a game failure that permits retries.

The specification permits 64, 128 or 256 unique seeds and registered looks
ending at the chosen cap. Evidence after a terminal boundary or outside the
exact seed order refuses. Unobserved suffix seeds remain uncertain in the full
roster's identification interval, not dropped from its denominator. Diagnostic
`primary_crossings` remain available after a failed attempt, while
`terminal_primary_claims` are false on nonterminal prefixes and on provenance
or refusal-cap failure. Even true terminal numerical claims do not confer
scientific admission: the explicit admission and Phase B authority flags are
always false.

## Candidate bounds and assumptions

The candidate design uses fixed Bonferroni allocation: family alpha 0.05, at
most 0.025 per arm. This is not Holm and is not adoption of the final Phase B
design. Fixed-time bounds additionally divide the per-arm budget over all
registered looks. Candidates are frozen-stake betting, Hoeffding and empirical
Bernstein; an uncalibrated t-test is not accepted by this numerical entry.

For frozen positive stakes below one, betting multiplies `1 + stake * lower`
on bounded contrast intervals. Under the stated conditional-mean null, replacing
an outcome by its lower endpoint cannot create a crossing that the complete
outcome path lacked. Shared controls need not make the two arms independent;
each arm still needs its null assumptions and the fixed prospective seed order.
The [bounded-mean betting paper](https://arxiv.org/abs/2010.09686) provides the
underlying nonnegative-supermartingale framework. This implementation's frozen
stakes and two-arm allocation are declared choices, not a reproduced optimized
algorithm or an outcome-fitted stake.

Empirical Bernstein uses the range-two version of
[Maurer and Pontil Theorem 4](https://arxiv.org/pdf/0907.3740), including the
`14 log(2 / alpha) / (3 (n - 1))` term and sample variance. It assumes IID seed
clusters. Its variance penalty is not monotone under arbitrary lower-endpoint
imputation, so it abstains on uncertain prefixes; it does not delete incomplete
clusters or switch to complete-case inference. Registered-look error allocation
does not supply validity for a post hoc method, stake, seed order or cap.

## Calibration and verification

The synthetic driver retains six clean bounded nulls, shared-control and partial
null scenarios, declared alternatives, refusal stress and an explicitly invalid
cross-cluster dependence control. It compares three frozen stakes and three
caps without adopting an optimized stake. Reported power is conditional on
each declared generator; optional hours are a labelled planning assumption,
not measured runtime. Invalid numeric inputs fail before output creation and
calibration outputs are create-only and source-bound.

For realizable six-game laws, calibration checks reconstruct the original
cell scores and known/unknown flags and compare terminal decisions through
the cell entry, scalar interval entry and vectorized computation. Generic
whole-contrast erasure and nonrealizable bounded stress laws are explicitly
excluded from original-cell parity. They remain stress tests, not fabricated
game rosters. The preserved earlier R2 calibration used a different generator
hash and wider-erasure missingness; its results do not calibrate this new
cell-faithful revision. A new prospectively reviewed current-revision execution
is still required before using its power figures for a scientific decision.

Focused verification uses no game, model or historical outcomes:

```sh
PYTHONPATH=src:tests:scripts python -m unittest test_search_over_raw_statistics -v
```

The parent passed all 32 tests in 1.278 seconds, and Astra independently passed
them in 1.272 seconds, exit 0 without skips. These checks cover the original-cell
join, shared missing controls, exact prefixes, invalid claims after later
failures, alpha allocation, refusal stops, all declared scalar/vector cases,
and output guards. They are implementation evidence, not a new full calibration
run, runtime qualification, reliability estimate or scientific measurement.
