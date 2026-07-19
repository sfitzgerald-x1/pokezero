# Root-PUCT W4 scale-cost preliminary readout

Status: **pre-W5 preliminary cost evidence, not a strength benchmark**. This note records the
completed portion of W4 from [`test_time_search_plan_v2.md`](test_time_search_plan_v2.md) on the
former replay-from-root implementation. W5 has since moved the primary path to direct
materialization, so these rows are a useful scale baseline rather than a current serving-latency
claim. They do not establish that any budget improves play.

## Scope

The probe used one representative full game per model/budget with a 250-decision-round cap, four
belief-world samples, and nine opponent-action candidate scenarios. Both configurations used the
same search settings apart from the model checkpoint and extra root-visit budget:

- **M50:** a 50M-parameter checkpoint.
- **L200:** a 200M-parameter checkpoint.
- **Extra visits:** `0`, `24`, `120`, `480`, and, for M50 only, `1200`. Zero still performs the
  mandatory legal-action sweep; it is not a raw-policy timing measurement.

The L200/1200 phase was deliberately stopped before completion once the completed rows had already
placed 120 and above far outside the practical per-turn envelopes. All completed telemetry artifacts
were preserved.

## Measured Cost

`Decisions` includes both searched and fallback decisions. The timing columns are full policy
dispatch wall time, so they deliberately include fallback behavior. `Searches` and `fallback rate`
make that coverage limitation visible rather than allowing a fast fallback to look like cheap
search.

| Model | Extra visits | Decisions | Searches | Fallback rate | Mean seconds / decision | P95 seconds / decision | Visits / root-search second |
|---|---:|---:|---:|---:|---:|---:|---:|
| M50 | 0 | 44 | 31 | 29.5% | 2.25 | 4.65 | 1.87 |
| M50 | 24 | 38 | 25 | 34.2% | 3.77 | 8.72 | 5.11 |
| M50 | 120 | 53 | 40 | 24.5% | 19.21 | 39.59 | 4.92 |
| M50 | 480 | 53 | 40 | 24.5% | 69.32 | 144.32 | 5.28 |
| M50 | 1200 | 42 | 29 | 31.0% | 127.82 | 294.52 | 6.51 |
| L200 | 0 | 52 | 39 | 25.0% | 2.58 | 5.63 | 1.40 |
| L200 | 24 | 44 | 29 | 34.1% | 4.60 | 10.86 | 4.30 |
| L200 | 120 | 50 | 37 | 26.0% | 17.47 | 43.43 | 5.29 |
| L200 | 480 | 39 | 26 | 33.3% | 53.57 | 113.15 | 6.04 |

## Readout

The pre-W5 implementation did not demonstrate a reliable two-second search budget at either model
scale: even the smallest search setting had a mean above two seconds and a materially higher P95.
`0` extra visits was the closest configuration to an aggressive envelope, but remained a
cost/coverage tradeoff rather than a deployable two-second claim.

For a roughly ten-second ladder-like envelope, M50 with 24 extra visits is the only completed row
whose P95 is below ten seconds. L200 with 24 extra visits is close by mean but just above ten seconds
at P95. At both scales, 120 extra visits and above are already far beyond that envelope, so further
measurements at larger budgets would not change the operational conclusion.

At the time of this probe, `0` and `24` extra visits were the only plausible budgets for a later
strength-vs-cost tradeoff. W2 must refresh the curve on W5's direct-materialization path before
selecting a working search budget. The nontrivial fallback rates also mean that W1's fallback
baseline remains necessary before treating any timing row as a stable serving envelope.

## Verification Evidence

- The report was generated from the completed Root-PUCT telemetry rows for the two checkpoints and
  budgets listed above.
- The table preserves decision counts, search counts, fallback rates, full-dispatch mean/P95 wall
  time, and root-search visit rate from those artifacts.
- No win-rate or paired-strength result is asserted here; those require the shared-seed comparison
  harness described in the W2 plan.
