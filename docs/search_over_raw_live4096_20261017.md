# Search versus raw at 4096 fixed iterations

October 17, 2026. Independently reviewed saved-data results from a capped,
incomplete exploration. The goal is to establish whether deployable search
improves on raw argmax and identify its limiting bottleneck. This exploration
does not establish a gain.

## Measurement and closure

Model and HP leaves each reported 4,096 native iterations across four
public-belief worlds, not a one-second budget. Raw argmax drove both seats.
Every eligible nonopening p1 boundary was scanned in frozen source order.
Action differences triggered eight paired common-random-number continuations
per unique action; accepted agreements contributed exact zeros without games.

Collection hit its fixed eight-hour cap at 10:08 AM Pacific: supervisor and
outer exit 124, worker exit -9, no cleanup error. The original parent then
finished the canonical source, root and continuation audit and readout.
Analysis completion does not turn the capped attempt into a completed study.
The run will not be resumed or extended.

The journal has **915 reserved roots**: 872 accepted across 21 validated
sources, plus 43 uncertain roots from unfinished source twenty-two. The final
reservation has no selector result. All 43 retain unknown deviation membership
and contrast bounds of -1 to +1, including provisional action agreements.
Unused capacity is excluded. Source coverage is below the required 32;
neither leaf reached the 100-deviation target.

All 42 correspondence gates passed, covering 1,744 selected comparisons.
All 75 continuation jobs ended successfully, providing 1,200 validated
COMPLETE cells. The original exact native-prior and selected-action replay
requirements remained in force.

## Results against raw

The first three numerical columns use only accepted roots and matching
pooled-root weights. Effects are continuation win-score percentage points,
not full-game improvements.

| Leaf | Deviations | Mean on deviations | Mean over accepted roots | Bounds over all 915 roots |
| --- | ---: | ---: | ---: | ---: |
| Model | 50 of 872 or 5.73% | +1.38 pp | +0.079 pp | -4.62 to +4.77 pp |
| HP | 44 of 872 or 5.05% | +7.67 pp | +0.387 pp | -4.33 to +5.07 pp |

The last column contains identification bounds, not confidence intervals.
Accepted-only descriptive 95% seed-bootstrap intervals also cross zero:
model -0.415 to +0.653 pp; HP -0.082 to +1.034 pp. Full-scan conditional
intervals are unavailable: 6,370 of 10,000 resamples have uncertain membership.
These complete-case diagnostics and uncalibrated adaptive bootstraps authorize
no screen or leaf selection.

For accepted roots, raw margin is the largest minus second-largest legal
prior, without renormalization. Below are means on deviations, with deviation
counts in parentheses.

| Raw margin | Model | HP |
| --- | ---: | ---: |
| Below 0.1 | -3.00 pp (25) | +2.16 pp (26) |
| 0.1 through 0.3 | +11.25 pp (15) | +21.25 pp (15) |
| Above 0.3 | -2.50 pp (10) | -12.50 pp (3) |

Another 23 accepted roots have one legal action and no defined margin. The
43 uncertain roots remain unclassified. High-margin overrides lack the
hoped-for positive pattern; these small diagnostic groups cannot establish
harm or diagnose the value head. Medium-margin HP's descriptive conditional
interval has a lower endpoint of exactly zero, not a positive screen.

## Screening limitation and disposition

A bounded, posthoc synthetic sensitivity modeled first-hit stopping, source
validation, shared raw outcomes, refusals and independent exploration/holdout
leaf selection. In a prespecified skewed zero-effect family, selected rejection
was 26/250 or 10.4% (pointwise 95% Monte Carlo interval 7.2–14.8%) with pooled
weighting, and 9.6% (6.5–13.9%) with equal seed weights. These results do not
clear the nominal one-sided 5% bootstrap procedure.

This was an incumbent-only hypothetical procedure, not a measured-variance fit
or full A1/A3/reference calibration. The counterexample assumed 48 eligible
roots per source, 5% independent deviation membership per leaf, eight shared-raw
paired cells and source effect +0.10 with probability 0.8 or -0.40 otherwise
(mean zero). Capture and replay each cost an assumed two seconds per boundary.
The 48 scenarios used 250 cohorts and 10,000 bootstrap draws each; intervals
are pointwise, not simultaneous. No actual holdout was opened.

Separately, exact quota-stratified resampling of the measured 320/1024 source
profiles put fixed-200-root eligibility below 0.000000005% per 320 leaf and at
0.0200% for 1024 model. Those are empirical-law power ceilings for unchanged
action/deviation profiles, not population bounds. Missing cells were not
complete-case scored; the incomplete 4096 panel was not fitted as a population.

## Cost and next scientific decision

Across all 21 completed sources, matched selector calls took about 190 minutes
for capture and another 190 for correspondence replay. These are selector
timing sums, not whole-workflow costs. If that 21-source average repeated,
selector work alone would take about 9.7 hours for 32 sources, before
continuations; this projection is not a runtime guarantee.

No qualified candidate emerged. Do not select HP or repeat the unchanged budget
panel. Further collection needs a specific hypothesis and feasible source
coverage, not an extended cap. Cheaper correspondence work is optional, not an
infrastructure prerequisite: the 21 completed sources passed their gates. This
supersedes this document's earlier lower-work-gate-first recommendation.

Astra independently reproduced the saved-data arithmetic and cleared the
exact readout and supplement. Original receipts and the independent review
remain in the owner's local report collection under
search-over-raw-live4096-deviation-exploration-20261016-r1. This document does
not publish the underlying dataset. The subsequent independently reviewed
eligibility calculation and noncertifying sensitivity are retained under
pokezero-search-over-raw-parallel-checks-20261016 as
measured_screen_eligibility.json and timed_deviation_screen_sensitivity.json.

Related evidence: the [1024 panel](search_over_raw_live1024_20261017.md),
[320 panel](search_over_raw_live320_20261016.md),
[runtime preservation PR](https://github.com/sfitzgerald-x1/pokezero/pull/1508),
and [landed deviation arithmetic](https://github.com/sfitzgerald-x1/pokezero/pull/1532).
Different fresh-source panels do not isolate a budget effect. Holdout, Phase B
search arms and retraining remain closed. The [amended research-plan closeout](search_over_raw_closeout_20261017.md)
records completion of the prescribed work, not a search gain or completion of
this capped study.
