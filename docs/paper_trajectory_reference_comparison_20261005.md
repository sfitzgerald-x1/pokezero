# Paper trajectory MCTS comparison results

October 5, 2026

The bounded comparison was actually executed with the unchanged champion:
raw policy, incumbent MCTS and the trajectory reference at fixed work and a
nominal one-second decision budget. It provides no demonstrated action-quality
gain. Stop this candidate at the bounded screen; do not launch a larger game
pilot, train another model or change production policy from these results.

This is a comparison of search decisions on retained positions, not a complete
reproduction of the thesis's training or a whole-policy playing-strength test.
The reference's initial strict sampler refused; the completion repair is an
explicit adaptation, not evidence that the exact paper variant passed all roots.

## What was tested

The frozen roster contains 32 positions from 16 source battles in eight mirrored
seed pairs. It retains the original nine troublesome identities. Thirty-one
positions have simultaneous requests; one is an actor-only faint replacement.
No position was redrawn, no private opponent commitment was supplied, and no
failure was silently converted into a raw-policy fallback.

Fixed work means 200 completed reference trajectories versus four incumbent
worlds with 50 native iterations each, depth two and batch eight. Those counts
are not equivalent units of work. The reference uses 20 persistent single-thread
CPU workers; the incumbent uses one model-world worker. The laptop has 18 cores
and no CUDA. This is a wall-cost comparison, not equal CPU allocation.

Both outer timers include observation, preparation, search, communication and
legal choice serialization. Internal deadline scopes differ: the reference
subtracts public capture from its absolute budget, while the existing native
incumbent starts its model decision budget after outer preparation. The result
does not certify identical hard stopping deadlines. Startup and model warmup
are separate from warm decision timings.

The action audit evaluates eight seeded terminal continuations per selected
action. At simultaneous roots, the first opponent reply samples the unchanged champion's full legal
distribution; subsequent decisions use raw-policy argmax. Common randomness
does not imply identical trajectories after different actions. The auditor uses
the source snapshot only after selections, never as planner input. Identical
actions reuse the same intervention outcomes rather than become independent
samples. The request cap is 200; every accepted game reached a real terminal.

## Original failure and narrow repair

The first registration completed 170 of 192 timing cells. All raw and incumbent
cells completed. The reference completed 21 positions in both modes before a
worker refused on Crawdaunt's tenth forced completion: public maximum HP 257
conflicted with Hidden Power Flying's HP IV 30, which projected 256. The pool
then refused subsequent work instead of restarting or hiding the failure.

A default-off adaptation projects earlier templates from the same original ten
draws in reverse order, choosing the latest that satisfies every public hard
constraint. For the reproduced failure, template nine projects HP 257. It uses
no extra draw, catalog proposal, invented EV change or relaxed exclusion. The
strict tenth-template behavior remains the default.

Only the 22 refused reference cells were rerun. All 22 completed. The original
264 terminal games and all passing measurements were retained. The repair
reused 88 pinned outcomes and ran only eight additional games for one newly
selected action, yielding 272 unique terminal games with zero caps or failures.
The original strict failure remains recorded. Old 21-root and repaired 11-root
timings are separate cohorts, not a uniform-runtime 32-root rerun.

## Decision cost

Times below are full warm decision wall times in seconds. The reference rows
remain separate to avoid attributing old measurements to the adapted runtime.

| Arm and cohort | Fixed work p50 / p95 | One-second budget p50 / p95 | Budget decisions within one second |
| --- | --- | --- | --- |
| Raw policy, all 32 | 0.0044 / 0.0059 | 0.0043 / 0.0054 | 32 of 32 |
| Incumbent, all 32 | 0.5643 / 0.9907 | 0.9438 / 0.9512 | 31 of 32 |
| Strict reference, original 21 | 0.7384 / 0.9063 | 1.0312 / 1.0413 | 0 of 21 |
| Adapted reference, repaired 11 | 0.8639 / 1.2003 | 1.0298 / 1.0356 | 0 of 11 |

All reference budget decisions finished within the earlier 1.2-second tail
allowance, but none met a literal full-boundary one-second cutoff. Coordination
and cleanup overrun is reported, not removed from the timing. Fixed-work maximum
was 1.2866 seconds in the adapted cohort and 1.8122 seconds for the incumbent.
The reference spent substantially more CPU parallelism without demonstrating
additional first-action value.

## Conditional action quality

At fixed work, the repaired reference and incumbent each change one of 32 raw
choices. At the nominal deadline, the reference changes two and the incumbent
one. Both choose `move 1` instead of raw `switch 2` at position 27; this loses
two of eight paired continuations that raw wins. The reference's additional
deadline change at position 24 has identical terminal results in all eight pairs.

Across all 32 positions, the conditional win-probability contrast against raw
is **minus 0.78125 percentage points** for both reference modes and both
incumbent modes. Resampling the eight mirrored seed clusters gives a descriptive
95 percent bootstrap interval of **minus 2.34375 to zero percentage points**.
The negative effect comes from one source seed; removing it yields zero.
This small, clustered panel does not establish statistical whole-policy
inferiority, and it is not 272 independent root samples.

The reference summary combines selected actions from 21 strict positions and
11 adapted positions. It is an explicit repaired decision panel, not an
unmodified paper sampler's all-root estimate or a fresh test of the adapted
runtime on every position. No positive point estimate survives the screen,
so the preregistered benefit condition for a game pilot is unmet.

## Fidelity and interpretation limits

| Requirement | Implemented or tested status | Remaining limitation |
| --- | --- | --- |
| Own-policy opponent sampling | Full masked distribution, separate RNG | Continuations model this same champion, not an external adversary |
| Fresh hidden completion per trajectory | Exact pinned server draws and public constraints | Known draws use empty team context; exact opponent HP is a corpus information deviation |
| First-new-leaf or terminal expansion | Implemented and tested beyond depth two | Finite work still limits exploration; a safety cap refuses |
| Signed Q/N/M/P and prior exponent | Implemented with alpha 0.5 and beta 1 | Thesis selected parameter values are unpublished; no sweep was run |
| Persistent statistics and faint pruning | Implemented with battle isolation and tested reuse | Frozen-root comparison resets trees; live persistent-policy benefit is untested |
| Twenty workers and ten-trajectory exchanges | Actual execution, acknowledged incremental aggregation | Slight laptop oversubscription; not resource-matched to incumbent |
| Strict tenth-template forced completion | Preserved default; original refusal recorded | All-root repaired decisions require a separately labeled completion adaptation |
| Format, network and allowance | Gen 3, current transformer champion, practical one second | Thesis used Gen 4, its own training and ten seconds |
| Whole-policy and Foul Play strength | Not tested | No ladder, full-match or external-opponent superiority claim |

The finding is limited but actionable: this registered search configuration
does not justify its added cost or a larger benchmark from this panel. It does
not show that MCTS is inherently ineffective, that reward design has a
fundamental flaw, or that another budget, model or persistent policy would have
the same result. Priors, critic quality, hidden-state modeling and exploration
are possible explanations, not isolated causes established here.

The thesis source is Jett Wang, *Winning at Pokémon Random Battles Using
Reinforcement Learning*, MIT (2024), pages 21–22 and 26–28:
<https://dspace.mit.edu/handle/1721.1/153888>.

## Evidence and closeout

The experiment used original source `90fc58077838060d3ef1d213a1ac972cd01459f4`
and narrow repair `ceea5cef8007fae5c57dec8a9a8774be12da7e3e`. Champion SHA-256 is
`0fd095923b4ac7e05d6e2b3ccab9c1e6869dff4893c2dae456caff10dce690be`;
Showdown commit is `f76228a1354b5d0f307ca2d16101294ad3a2308b` and generator hash
`f5a5265143d423af`. Native incumbent, model export, encoder, roster, driver and
source inputs are pinned in the immutable registrations and readouts.

The owner's local experiment report directory retains the immutable evidence:

- Original `paper-reference-comparison-20261005-r1/READOUT.json`, SHA-256
  `e7cb3476cd3423c78e73adeb72482dc915333229d0138b53c663684d2544e30b`.
- Repair `paper-reference-comparison-20261005-r2-narrow-repair/READOUT.json`, SHA-256
  `fa184c1f447959042ae504e62e7d05b7fe256892397d5536fced2f274e960dba`.

Independent review verified the bounded adaptation, actual ten-seed regression,
runner identity checks, pinned outcome reuse, source binding and derived
clustered readout. The adapted reference and census regression suite passed
146 tests; seven runner preflight tests passed. The 75 census/citation tests
and 116 subsequent ledger/attestation tests pass. The separately disclosed
pre-existing scenario fixture/source mismatch remains a failed broader local
check; this is not a claim that every local test passes.

PR #1503 is merged. PR #1504 carries the opt-in reference and this report; it
must pass required CI before merging. Scientific disposition is
`STOP_NO_DEMONSTRATED_ACTION_GAIN`. No game pilot, promotion or production change
is authorized by the measured result. Historical artifacts, original refusal
records, historical measurements and verdicts, and cluster objects remain
unchanged. Source citations in two historical proof reports were refreshed
without changing their measurement or verdict fields.
