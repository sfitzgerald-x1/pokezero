# Full game search against raw on October 20

The fixed-work incumbent with opponent priors ON, four worlds × 256 iterations and depth six scored **53.91–54.69% against deterministic raw argmax** on the original 64 seeds, both seats. Independent review accepted 127 terminal outcomes and retained one capped game as unknown.

This is a completed secondary measurement, not a qualified improvement: the range describes the possible result on this fixed roster, not sampling uncertainty or statistical significance. The failed Foul Play primary comparisons remain separate and unqualified.

## Outcomes

| Candidate seat | Wins | Losses | Public ties | Caps |
| --- | ---: | ---: | ---: | ---: |
| First | 34 | 27 | 2 | 1 |
| Second | 34 | 30 | 0 | 0 |
| Both | 68 | 57 | 2 | 1 |

All 128 cells remain in the denominator. Seed 3829141002 in the first seat reached 250 decision rounds without a public winner or tie. Its legacy half-point is ignored; its scientific outcome is [0,1]. Averaging the two seats within each seed, then all 64 seeds, gives [69/128,70/128]. Excess over an even score is **3.91–4.69 percentage points** for this roster. No confidence interval or test was introduced after results.

## Work and cost

Both shard processes and the outer supervisor exited 0 with verified cleanup, after 3 hours 51 minutes overall. The collector retained all 6,643 searched decisions and 6,802,432 iterations. Candidate error, retry, fallback and recorded-refusal counters were zero with attached, trustworthy telemetry. Raw refusal telemetry is unavailable, not zero.

Mean bridge policy time was 1.93 seconds; nearest-rank p95 was 3.33 seconds. This was fixed work, not a one-second wall-clock search budget or deployment qualification.

Native opponent-prior projections were retained for 6,096 decisions. Of 547 omissions, 530 show opponent No Move, one is an acting No Move root, and 16 remain conservatively undisambiguated. These projections witness application in at least one contributing world, not every world.

## Scope and next step

The champion, native search budget and original secondary roster were unchanged. The Python public-slot construction repair preceded this collection; it is not byte-identical replication of the failed primary or proof of its historical cause. Validation covered saved public terminals, exact seed-seat membership, bound inputs/source and actual exits—not full protocol replay or simulator equivalence.

Continue the other independently admitted secondaries in their fixed order. Do not repeat this roster to resolve its cap, pool stopped cohorts, or use this result to authorize a teacher.

The [compact result and full seed inventory](audit_artifacts/phase0_on4_raw_secondary_20261020.json) preserve the descriptive calculation and evidence digests. Source repairs: [PR 1564](https://github.com/sfitzgerald-x1/pokezero/pull/1564) and [PR 1565](https://github.com/sfitzgerald-x1/pokezero/pull/1565).
