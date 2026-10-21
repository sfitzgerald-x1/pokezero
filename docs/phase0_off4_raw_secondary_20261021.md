# Four world search with priors off against raw

The incumbent with opponent priors OFF, four worlds × 256 iterations and depth six scored **58.59375% against deterministic raw argmax** on the original 64 seeds, both seats. Independent review validated all 128 public terminal outcomes: 74 wins, 52 losses and two genuine ties, with no caps or missing cells.

This is a descriptive secondary result, not a confidence interval, significance claim or qualified Foul Play improvement. It challenges the expected null on this fixed roster; it does not establish why search helped or authorize a teacher.

## Outcomes

| Candidate seat | Wins | Losses | Public ties | Unknown |
| --- | ---: | ---: | ---: | ---: |
| First | 41 | 22 | 1 | 0 |
| Second | 33 | 30 | 1 | 0 |
| Both | 74 | 52 | 2 | 0 |

The fixed seeds are 3829141000 through 3829141063. Genuine ties score one half; neither is a cap. Averaging both seats within each seed and then all 64 seed clusters gives **75/128 = 58.59375%**, or 8.59375 percentage points above even. No post-result confidence interval or test was added. The [ON four-world secondary](phase0_on4_raw_secondary_20261020.md) used the same roster but ran separately; the two results are not a prospectively specified inferential comparison between ON and OFF.

## Work and cost

Both 64-game shard processes and the outer supervisor exited 0 with verified cleanup. Collection ran from October 20 at 8:26 PM to October 21 at 12:27 AM Pacific, lasting 4 hours 1 minute. No clock extension, retry or replacement was recorded.

All 6,730 searched decisions retained 1,024 iterations, totaling 6,891,520. Opponent priors were OFF; opponent-prior projection fields were null as expected. Candidate error, retry, recorded-refusal, fallback and dropped-root counters were zero with trustworthy telemetry. Raw refusal telemetry is unavailable, not zero.

Mean bridge policy time was 2.00 seconds, median 1.83 seconds and nearest-rank p95 3.39 seconds. This was fixed work, not a one-second wall-clock budget or timing/reliability qualification for deployment.

## Evidence and next step

Independent review validated saved public terminals and exact seed-seat membership, 18 bound inputs, three source repositories and 471 compiled simulator files. It did not replay the full protocol or prove simulator equivalence. Collector source was `ac6e3c58c2a86d2db6739236600f1c355a67508d`; controller source was `03dcf0169ba7743e1d9295db666d91d3eae50dce`. Search and deterministic raw used the same immutable champion. The reviewed Python public-slot repair preceded collection; no runtime revision was recorded during it.

Local evidence is retained under `reports/phase0-secondary-opponent-priors-off1024-vs-raw-20261020-r1` in Scott's research workspace. The final `scientific-readout-review.json` has SHA-256 `cdcfae492298c49ebee4ee24402952bc6dc8e0bf56efca04103645e3b54012e8`; its entry and evidence inventory bind the retained records. This note does not publish raw battle records.

Preserve this completed roster without replay. Repair and prospectively qualify the Foul Play instrument before a fresh primary comparison; the failed primary cohorts remain closed and unqualified. Advancement, deployment and training remain gated by the original full goal.
