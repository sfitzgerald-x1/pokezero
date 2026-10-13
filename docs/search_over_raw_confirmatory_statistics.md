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
cell-faithful revision. A prospectively reviewed current-revision execution on
October 12 completed 285 conditions in 52.6415 seconds, with source-bound local
contract/results and independent numerical audit. It did not adopt a method,
stake or cap, authorize games, or establish representative game runtime.

Focused verification uses no game, model or historical outcomes:

```sh
PYTHONPATH=src:tests:scripts python -m unittest test_search_over_raw_statistics test_search_over_raw_t_candidate -v
```

The parent passed all 32 tests in 1.278 seconds, and Astra independently passed
them in 1.272 seconds, exit 0 without skips. These checks cover the original-cell
join, shared missing controls, exact prefixes, invalid claims after later
failures, alpha allocation, refusal stops, all declared scalar/vector cases,
and output guards. They are implementation evidence, not a new full calibration
run, runtime qualification, reliability estimate or scientific measurement.

## Separate fixed 128 t candidate

`search_over_raw_t_candidate` is a separate, explicitly uncalibrated comparison
entry. It does not extend `JointDesign`'s accepted methods. Both candidates
have one terminal look at 128 unique seeds, fixed Bonferroni alpha 0.025 per arm
(not Holm), and the same third-refused-cluster/provenance stop. No claim is
computed on a shorter or failed prefix. Full roster identification intervals
retain its unobserved suffix. All adoption, scientific-admission and Phase B
authority flags remain false.

The plain comparator uses a Student-t critical value with 127 degrees of
freedom. The guarded candidate additionally requires adjusted Fisher–Pearson
absolute sample skewness at most 0.5, frozen before this calibration and tested
independently per arm. The adjusted statistic uses population-denominator
second/third central moments and the factor `sqrt(n*(n-1))/(n-2)`; this is the
[NIST adjusted definition](https://itl.nist.gov/div898/handbook/eda/section3/eda35b.htm).
A sample-skewness gate is not evidence of normality or general finite-sample
t-test validity. Neither candidate is adopted as the final Phase B rule.

Ordinary sample variance after lower-endpoint imputation is not a worst-case
t-test. This comparison instead explicitly adds a conservative moment envelope:
for positive lower mean L, any completion has mean at least L and squared sum
at most Qmax (the sum of maximum squared interval endpoints). Its sample
variance is at most `(Qmax-n*L²)/(n-1)`, bounding its t-statistic from below by
`sqrt(n)*L/sqrt(Vupper)`. This is an outer bound, not an exact completion
optimizer or a finite-sample validity theorem. Nonpositive lower mean abstains.
Zero variance abstains; so does any interval roster admitting a constant
completion, preserving that abstention under completion. The skewness gate
abstains on any uncertain contrast in that arm rather than inventing observed
skewness or switching to complete cases. Known scores in the shared-control
six-cell mapping remain known; all uncertain scores retain their full intervals.

Student-t tails use the regularized incomplete beta function and the
[NIST continued fraction](https://dlmf.nist.gov/8.17#v), with numerical inversion
for the [one-sided critical value](https://itl.nist.gov/div898/handbook/eda/section3/eda3672.htm).
Independent rounded table values, analytic Cauchy tails and direct normalized
density quadrature check the numerics; no SciPy installation is required.

## Synthetic t comparison results

The prospectively reviewed October 13 run froze seed 2026101301 and reused the
unchanged joint driver's 19 families: 20,000 generated cohorts per family, each
evaluated with both fixed rules. This is 380,000 generated cohorts and 38 result
conditions, not 760,000 independent cohorts. All conditions completed with
exit 0 in 10.4033 wall and 10.3752 CPU seconds. Generic bounded/erasure stress
is distinguished from realizable six-cell games; the dependent negative control
is excluded from in-assumption summaries. The gate's detections are a subset
of plain detections on every batch; stopping and expected costs are identical.

| Declared family at 128 seeds | Plain t | Skewness gated t |
| --- | ---: | ---: |
| Both +10 points, clean shared control: both detected | 51.38% | 50.835% |
| Both +10 points, 5% clusters lose shared raw seats: both detected | 1.765% | 0.105% |
| Positive-frequent skewed null: any false primary | 6.56% | 0%, with 100% per-arm gate abstention |
| Partial skewed null: true reference +10-point detection | 99.59% | 2.18%, with 97.41% reference gate abstention |

The largest in-assumption gated family false-positive point estimate is 954
of 20,000 (4.77%) on the low-variance null. Its pointwise 95% Wilson interval is
[4.4832%, 5.0742%]. That is not a simultaneous certificate, a universal bound or
an approved calibration acceptance criterion. Zero recorded skew-null false
positives is abstention, not useful detection on skewed distributions. The plain
skew-null estimate is 1,312/20,000 (6.56%), interval [6.2251%, 6.9115%].

Clean studies use all 128 clusters (1,536 planned games). Under the shared-raw
missingness law both methods stop for refusal in 95.68% of trials and use a mean
59.1442 clusters (709.7304 planned games). Short failed studies are not successful
runtime savings. Projected hours remain null: these are expected full-cluster
game budgets, not measured games or representative whole-harness costs.

The contract and result are retained in the existing local report root as
`pokezero-search-over-raw-t-candidate-contract-20261013.json` and
`pokezero-search-over-raw-t-candidate-results-20261013.json`. Their SHA256 digests
are `8819453a69556f5bbfa01eaff93d11b9fe902a3a018dfab196f89a1966b7399e`
and `40fffb59ccccd0ce14afc3ca80c484016de585bdb53af4889f1dc19912502adb`.
Those local outputs are not committed scientific evidence. Earlier calibration
outputs and their three bound source files are unchanged. Parent verification
passed all 52 old/new tests in 2.050 seconds; Astra independently passed the 20
new tests in 0.719 seconds, exit 0 without skips, before the full synthetic run.
Final procedure selection, runtime/reliability/fidelity qualification and
scientific execution remain outstanding.
