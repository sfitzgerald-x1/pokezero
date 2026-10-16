# Raw champion versus Foul Play baseline

Published October 16, 2026.

The unchanged raw champion won **39 of 64 games (60.94%)** against the
registered Foul Play opponent over 32 fresh seeds, playing both seats.
This supplies the search-over-raw program's external baseline. It does not
measure a search arm, establish a search advantage, or trigger retraining.

## Result and uncertainty

Raw scored 19 wins and 13 losses as p1, and 20 wins and 12 losses as p2.
All 64 registered cells completed with no ties, caps or missing outcomes.
The descriptive 95% seed-cluster bootstrap interval is **51.56% to 70.31%**.
This is not a calibrated confirmatory test or significance claim.

The [publication extract](../reports/artifacts/raw_foulplay_baseline_20261015.json)
contains every paired score and the original evidence digests. Two seeds
lost both seats, 21 split their seats, and nine won both. To reproduce the
interval, convolve the empirical seed-score mass
`{0: 2/32, 0.5: 21/32, 1: 9/32}` 32 times and take the 2.5% and 97.5%
quantiles of the resampled mean. Treating the 64 games as independent would
ignore the paired design.

## Configuration and reliability

The format was Gen 3 random battles, using raw argmax from
`foundation-midscale-iter-9375` with its trained belief-set observation
source enabled. The extract identifies the unchanged checkpoint, runner,
Showdown and Foul Play revisions, and the registered Foul Play patches.

Foul Play was configured at 1000 ms with its original adaptive algorithm,
ordinarily about two seconds of search per move plus setup. This was **not**
a strict one-second whole-move limit, and its iteration telemetry was
unavailable under the observed process start method.

Both actual invocations exited 0. Zero game-ending failures were observed,
but decision-level refusal telemetry was unavailable, not zero. This result
does not clear an overall reliability or deployability gate.

## Historical failure and evidence scope

The earlier baseline entry stopped with exit 130 before completing a game:
its observation source was disabled. Its 64 planned scores remain uncertain.
This corrected entry used separately reviewed fresh seeds and is not a retry
or continuation of that roster; the failed entry is not pooled into this result.

Astra independently checked the original 64 seed-seat cells, source bindings
and descriptive interval. Original registrations, summaries, terminal output
and review remain in the owner's local report directory
`search-over-raw-foulplay-baseline-20261015-r2`. The compact extract permits
recomputing the numerical summary, but does not replace those process and
provenance receipts or make the complete runtime reproducible from this repository.

Holdout, search-arm Phase B and retraining are not authorized by this baseline.
