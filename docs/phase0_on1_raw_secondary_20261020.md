# Interrupted one world search against raw on October 20

The incumbent with opponent priors ON, one world × 2,048 iterations and depth eight stopped after **46 validated games: 29 wins and 17 losses**, all in the first seat. The original roster is 64 seeds, both seats, or 128 cells. Its 82 unknown outcomes prevent a strength conclusion; 29/46 is not a full-panel estimate.

## Outcomes and uncertainty

| Candidate seat | Wins | Losses | Unknown | Planned |
| --- | ---: | ---: | ---: | ---: |
| First | 29 | 17 | 18 | 64 |
| Second | 0 | 0 | 64 | 64 |
| Both | 29 | 17 | 82 | 128 |

The fixed seeds are 3829141000 through 3829141063. Only first-seat seeds through 3829141045 have accepted public terminal outcomes. Seed 3829141046 was interrupted; the remaining 17 first-seat and all 64 second-seat cells were unstarted. There were no saved ties or caps.

Assigning every unknown cell [0,1], averaging both seats within each seed and then all 64 seed clusters, gives **[29/128,111/128] = [22.65625%,86.71875%]**. Excess over even is −27.34375 to +36.71875 percentage points. These are descriptive missing-outcome bounds, not confidence intervals or significance. No stopped cohort is pooled, refilled or replayed.

## Interruption and cost

Collection ran October 20 from 5:22:56 PM to 8:21:32 PM Pacific. The last completion was at 7:42:08 PM. The supervisor reported `NO_COMPLETION_TIMEOUT`, terminated its worker with exit −15 and verified cleanup; the actual outer invocation exited 1.

Bounded OS power events show clamshell sleep at 7:38:51 PM, a brief dark wake around the last completion, maintenance sleep at 7:42:18 PM, and full lid/HID wake at 8:21:27 PM. Termination followed about five seconds later. Host suspension contributed to delayed watchdog enforcement. This does not establish a native-search hang, an algorithmic cost ceiling, or how much of every pause was caused by sleep. The original calendar deadline was not extended. An idle-sleep assertion did not prevent clamshell sleep.

The saved prefix retained 2,409 searched decisions × 2,048 iterations. Opponent-prior projections were witnessed on 2,217 decisions in at least one contributing world each; two acting No Move roots were unmeasured. Zero saved-prefix errors or refusals do not establish health of the unfinished game. A one-second read-only process sample during a different, eventually completed 195-round game adds unmeasured observer overhead; realized timing is not an isolated benchmark.

## Evidence and next step

The independent readout cleared only the stopped prefix, validating exact seed-seat membership and saved public terminal metadata, bound inputs, clean source, actual exits and cleanup. It did not replay the full protocol or establish simulator equivalence. Collector source was `ac6e3c58c2a86d2db6739236600f1c355a67508d`; controller source was `03dcf0169ba7743e1d9295db666d91d3eae50dce`. The same champion was used for search and deterministic raw.

Local evidence is retained under `reports/phase0-secondary-opponent-priors-on2048-vs-raw-20261020-r1` in Scott's research workspace. The final `scientific-readout-review.json` has SHA-256 `8500d52d2d90687abfdb7dc636cc6b81e4ae54b930af5d2eb44c94e7ce49a8fe`; its entry and evidence inventory bind the retained records. This note does not publish the raw battle records.

Keep this roster closed. Continue the already admitted OFF four-world secondary only after predecessor termination and cleanup, without overlapping timed collection. The [completed four-world ON secondary](phase0_on4_raw_secondary_20261020.md) remains a separate descriptive result. No Foul Play primary, advancement, deployment or teacher is qualified by this partial result.
