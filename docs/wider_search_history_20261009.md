# Historical wider search results from October 6 to 9

Published October 16, 2026. Evidence snapshot: October 9.

The wider study collected 90 accepted games from a fixed 256-game roster.
Its descriptive results favored the paper reference, but the unfinished,
mixed-source ledger establishes neither a reference advantage nor search
superiority over raw argmax. The design is retired.

## Design and accepted outcomes

The registration fixed 64 seed clusters, with both search arms playing both
seats against the same champion sampled from its full masked policy at every
opponent decision. There was no raw-argmax control arm and no Foul Play cell.
The nominal decision budget was ten seconds, not a strict latency ceiling;
reference used 20 workers and incumbent one, so this was not equal compute.

| Arm | Accepted games | Wins | Losses | Descriptive win rate |
| --- | ---: | ---: | ---: | ---: |
| Paper reference | 43 | 33 | 10 | 76.7% |
| Incumbent | 47 | 30 | 17 | 63.8% |

There were no accepted draws. The 12.9-point difference compares unequal
accepted subsets against a shared opponent; it is not a head-to-head result
or a valid full-roster effect estimate. Complete-case rates do not account
for potentially outcome-dependent missingness.

The 90 accepted cells comprise 83 retained results and seven separately
admitted local results. The remaining **166 cells stay unknown**: three
original refusals, one additional refusal, one unvalidated terminal labeled
COMPLETE, and 161 never-started games. Twenty clusters are fully known,
four partially known, and forty entirely unknown.

The validation pipeline exited **1** after the reference game at seed
566883983, seat two, produced a noncomplete terminal. The unvalidated
COMPLETE terminal is not promoted to accepted evidence. This record neither
retries those cells nor infers the producer's exit code from the observer's
historical process snapshot.

## Conservative bounds on the original ledger

Score a win as 1, draw as 0.5 and loss as 0. Let `y = 2 * score - 1`, and
give reference cells sign +1 and incumbent cells sign -1. The mean
reference-minus-incumbent contrast over all 64 seeds and both seats is
`D = sum(arm_sign * y) / 256`. Every unknown cell contributes an interval
`[-1, 1] / 256`; refusals are not assigned losses or dropped.

| Treatment | Known cells | Unknown cells | Contrast bounds in percentage points |
| --- | ---: | ---: | ---: |
| Retain accepted mixed-source outcomes | 90 | 166 | -60.94 to +68.75 |
| Make all 13 source-affected clusters uncertain | 40 | 216 | -81.25 to +87.50 |

The known signed contrast sums are 10 and 8 respectively, so the exact
endpoints are `(10 +/- 166) / 256 = [-39/64, 11/16]` and
`(8 +/- 216) / 256 = [-13/16, 7/8]`. The sensitivity makes all four cells
of each affected cluster unknown: 52 cells, including 50 formerly accepted
cells. Both calculations preserve the original denominator and retain
usable cells within other incomplete clusters.

These are deterministic identification bounds, **not confidence intervals
or hypothesis tests**. Both allow a substantial advantage in either
direction. Source changes also prevent treating the historical outcomes as
an estimate for a single frozen candidate. Later conditioning diagnostics
and qualification games are excluded from these 90 results.

## Evidence and remaining scope

The owner's local evidence is under
`/Users/scott/Documents/New project/reports/`:

- `search-statistical-amendment-20261009-r1/bounds/RESULT.json` and
  `REPORT.txt` contain the original cellwise analysis and accepted terminal
  bindings. RESULT SHA-256:
  `a54e1b2f1fd01cc9e027289fe48f45b07fffc69baccca86d903a49fa6bb79b63`.
- `wider-search-confirmation-20261007-r23-collection-amendment/registration.json`
  preserves the fixed roster, opponent, resource settings and source
  sensitivity clusters.
- `wider-search-evidence-audit-20261006-r1/registered-complete-stream-audits-collection-amendment/PIPELINE.json`
  records the seven local admissions and validation failure.

The printed counts and equations reproduce the bounds, but this document
does not bundle the original process receipts or make the historical native
build reproducible from the repository. It is the historical-record portion
of the search-over-raw plan, not completion of that plan's build and
prospective-validation requirements. The later
[raw versus Foul Play baseline](raw_foulplay_baseline_20261015.md) and
[live root-prior investigation](search_over_raw_root_prior_exactness_20261016.md)
are separate evidence, not additions to this ledger.
