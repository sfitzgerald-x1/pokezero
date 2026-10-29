# A4 saved-root cost preparation

This is source preparation for a separately reviewed cost-only packet. Importing
these modules does not admit an experiment. There are no models, native queries,
Foul Play decisions, simulators or workers started by the source controls.

`pokezero.foulplay_saved_root_cost.SavedRootFoulPlayParser` feeds the existing
opponent parser while the owner reconstructs the immutable TRAIN source rows.
The source is the recorded mixed omniscient/request archive, not a player wire.
Only the opponent actor's own private requests cross the adapter. Other-seat HP
is projected to the pinned Gen3 Random Battle player percentage representation;
own HP stays exact. Unknown HP-bearing commands refuse. This projection uses
Showdown's near-full 99 guard, not an unqualified ceiling to 100.

The source bindings are Showdown commit
`f76228a1354b5d0f307ca2d16101294ad3a2308b`:
[`Pokemon.getHealth`](https://github.com/smogon/pokemon-showdown/blob/f76228a1354b5d0f307ca2d16101294ad3a2308b/sim/pokemon.ts#L1933),
[Gen3 Random Battle's Standard rules](https://github.com/smogon/pokemon-showdown/blob/f76228a1354b5d0f307ca2d16101294ad3a2308b/config/formats.ts#L4176),
[Gen3 Standard](https://github.com/smogon/pokemon-showdown/blob/f76228a1354b5d0f307ca2d16101294ad3a2308b/data/mods/gen3/rulesets.ts#L1),
and [Standard AG's HP Percentage Mod](https://github.com/smogon/pokemon-showdown/blob/f76228a1354b5d0f307ca2d16101294ad3a2308b/data/rulesets.ts#L9).
The actual compiled Showdown/source namespace must still match its separately
frozen attestation before real source reconstruction.

The optional Foul Play imports are lazy. The owner must bind the complete patched
opponent tree and dependencies before constructing the real parser API. A
predeclared nonempty randbats cache SHA is mandatory. The production set transform
loads the immutable JSON snapshot that was hashed, so a changed path cannot
validate one cache and load another, and a missing cache cannot trigger a live
download. The
upstream API anchor is submodule `9955255176688a877f8b3586ee7c6bda39df5fdb`:
`fp.run_battle.start_random_battle`, `async_pick_move`, `fp.battle.Battle`, and
`fp.battle_modifier.update_battle`. Initialization uses the first own request,
then the queued public messages, omitting the duplicate opening own switch, as
the production loop does. Every later own request flushes the queued public
updates through that same parser. No historical `find_best_move` call occurs.

Past recorded own choices update the production `LastUsedMove` bookkeeping on a
request-updated copy. The chosen root's recorded action is never consumed.
`selected_decision_copy()` mirrors the copy and request update performed by
`async_pick_move`; it refuses a seat that is not requested. It returns an input
object only. Only the historical prefix is prepared outside the timing window. The selected
request/public update and ensuing choice must be dispatched before our own
root observe/search without waiting for the opponent to start; this preserves
production overlap. The later, explicitly admitted owner must invoke the unchanged
Foul Play `find_best_move` under its fixed 1000 ms, one-thread, one-worker
configuration and capture actual work and CPU evidence. Calling that function
or reconstructing real source rows is outside the unit-test/source scope.

Production forwards Foul Play's request before dispatching our search, so its
think overlaps ours. The proposed cost geometry is twelve match workers, four
per raw/champion-search/refit-search arm, under one GB300, 12 CPU and 64 GiB.
Each worker uses the same twenty immutable roots without replacement. An
opponent think occurs only if the recorded counterpart is requested; one-seat
boundaries preserve that absence. Sleep or inference-only load is not an
opponent contention substitute.

`pokezero.a4_cost_contention.readout` refuses incomplete, duplicate or reordered
worker/root sequences and missing work/CPU evidence. It labels FP request/choice
windows as elapsed windows, not as CPU busy time. The proposed conservative
p95 uses the maximum of the four replicas for each distinct root, then the
nearest-rank p95 over those twenty roots. It creates no inferential interval
from repeated roots. The prospective bound is 15 seconds against an explicitly
justified 30-second offline admission reference. It adds no algorithmic cutoff,
early stopping or change to the August fixed search work.

Every arm must explicitly hold FP32, MHA fastpath off, TF32 off, highest matmul
precision and absent overrides before eager root/opponent and leaf work. The
qualified head artifacts remain immutable. A later cost owner must first verify
its actual combined capture build, known observed refusal lineage, fresh source
regression receipts, exact qualified artifacts, source, tables and root roster.
The readout helper always leaves cost/panel admission false; timing alone cannot
admit a panel.

The version-2 readout requires a pre-timing contract for each of the twenty
source boundaries: source/boundary hashes, whether the opponent is requested,
its legal-choice count, and the sample count/budget from production's random
battle sampler. Time-pressure variants remain valid; the configured budget is
1000 ms even when production divides it into 500 ms samples. The complete
fixed roster remains in every timing calculation.

A cost owner can use `native_call_timing` around the unchanged production
`find_best_move`. It times the existing native callable without changing its
arguments, result, sampling or executor. The actual platform's unmodified pool
start method must be `fork` so child processes inherit the instrumentation;
another method is a blocked geometry, never permission to switch it. The hook
restores the original callable in `finally`. It emits only native call timing,
CPU and visit metadata, with no state strings. The production sampling,
Iterations and Choice log records must reconcile with every measured native
call. The owner must retain and hash each complete decision-scoped log range.

Request-window overlap alone cannot witness native work. The readout separately
checks native call intervals, native CPU and visits against the endpoint's
measured process plus reaped-child CPU. It requires actual native-work overlap
witnesses in every arm and observed full twelve-worker cycle overlap involving
every declared worker. Those intervals are elapsed intervals; even a positive
CPU total does not establish which exact nanoseconds were CPU busy. Fully
serialized workers, an entirely pre-completed opponent, or absent work leave
contention ineligible and the decision-time gate false even if the arithmetic
p95 is below fifteen seconds. All 240 timing cells and the arithmetic result
remain available.

A legitimate forced/single-choice or early-return cell retains its actual zero
work. No positive visits or artificial 1000 ms delay is fabricated. A zero-work
multi-choice cell cannot establish the intended contention, so eligibility
remains false; this neither declares the opponent invalid nor permits replacing
that root. A roster consisting only of zero-work forced cells likewise cannot
qualify busy-opponent contention. These are source/readout controls only: the
actual parser, native instrumentation, process geometry and cost packet still
need separate prospective review before any real root or opponent work.

The plan binds one fixed opponent code/source SHA across all twenty root
contracts. This is separate from each boundary and player-wire digest. Native
sample receipts must show one actual child PID, sequential call counters and
non-overlapping call intervals; declaring parallelism one alone is insufficient.
