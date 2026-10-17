# Search versus raw at 320 fixed iterations

October 16, 2026. Completed exploration, not holdout confirmation.

A fresh live-boundary panel completed all 200 planned roots from 32 source
seeds. Model-leaf and HP-leaf search each changed raw's action on only four
roots. Neither demonstrated a continuation advantage. This result does not
establish equivalence, leaf superiority or a retraining trigger.

## Design and results

The incumbent production selector ran at the actual p1 decision boundary,
using public belief, four sampled worlds and 80 iterations per world: 320
total, not a one-second budget. Iteration counts are production telemetry,
not independently qualified atomic-backup counts. Raw argmax drove both
seats of each source game; selection retained the fixed seed-priority rule
over the completed request catalog, with no replacement roots or seeds.

Each unique selected action received eight common-random-number paired
continuations: an initial sampled opponent reply, then raw-policy play.
Identical actions share cells rather than acquiring independent replicas.
These are decision-level continuation win-score contrasts against raw,
not full-game results against Foul Play.

| Search versus raw | Root-weighted mean in percentage points | Descriptive seed-clustered 95% interval in points | Changed actions |
| --- | ---: | ---: | ---: |
| Model leaf | 0.0000 | -0.1884 to +0.1856 | 4 / 200 |
| HP leaf | +0.0625 | 0.0000 to +0.1904 | 4 / 200 |

The intervals resample source seeds, retaining their roots: 10,000
percentile-bootstrap draws with seed 20261016. Both include zero. They
describe the root-weighted estimator, not the slightly different equal-seed
estimator. The fixed 32-seed estimates are -0.0093006 points for model and
+0.0651042 for HP. Their collapsed identification bounds describe this
finite sampled roster; they are **not population confidence intervals**.

Of model's four changed actions, one contrast is positive, one negative
and two zero. HP has one positive contrast and three zero. Three roots
have only one legal action. The panel contains 1,661 validated completed
continuation cells and three capped cells, with no refused, unstarted or
unvalidated-COMPLETE cells. All three capped scores remain null. They
occur at two roots where every policy chose the same action, so shared
aliases cancel exactly without assigning outcomes to the capped games.
Every root contrast is identified despite these three unknown scores.

## Correspondence and counter correction

All 64 source/leaf gates passed, comparing 400 selected native priors
against an independent live-native replay with zero total variation. All
200 selected priors per leaf were present and had a unique argmax equal
to raw. Raw-versus-native probability vectors need not be identical:
selected-root TV averaged 0.000348 and reached 0.018530. This is a
different comparison from collector-native versus independent-live-native.

The inherited readout incorrectly reports 200 unreported priors per leaf
because it reads an absent legacy field. Its disagreement counter reads
that field too. A separately reviewed saved-data companion corrects the
counters without changing the original readout, scores or intervals.
Each leaf recorded 1,527 original invocations: 32 ineligible openings and
1,495 eligible decisions, including the 200 selected roots. These rows
record no constructor refusals or world failures; this is not a population
reliability guarantee.

## Evidence and remaining scope

The evidence is retained outside this repository in the owner's local report
collection, under `search-over-raw-live320-exploration-20261016-r2/`.
It retains the entry, original source and continuation receipts,
`run/readout.json`, `saved-data-analysis.json` and
`independent-readout-review.json`. Original readout SHA-256:
`8b1b4d044ced33bd7de32be706a661e666672d8e169951d706271686e1ed1742`.
Independent review SHA-256:
`88861c81ebb508e6b65d3f16dd35427d6744ee2faa10d336b62fa78c8d1454d9`.

Astra's independent original-reader pass validated all 200 roots, 32 source
lifecycles, 64 gates and the continuation cells, and reproduced both
bootstrap intervals and fixed-seed estimates. Collection, final original
readback and independent verification each exited 0. The collection job
recorded about 3 hours 10 minutes; subsequent validation is additional.
Full-workflow cost, memory peaks, dependency closure and broad native
physics remain unqualified. The recorded scientific source is
`c689447a44ba6cd37dab541bef7970ee3d82a931`, using a separately pinned
repaired native runtime; this document alone does not reproduce that build.

The preceding failed 80-root roster and earlier reconstructed-selector
panels remain separate historical evidence, not extra observations here.
See the [root-prior investigation](search_over_raw_root_prior_exactness_20261016.md),
[raw versus Foul Play baseline](raw_foulplay_baseline_20261015.md) and
[retired wider study](wider_search_history_20261009.md) for their distinct scopes.

The prescribed next measurements are fresh, separately reviewed public-belief
model-leaf panels at 1,024 and 4,096 total iterations. No holdout, Phase B
search arm or retraining is opened by this result. The full search-over-raw
goal remains incomplete.
