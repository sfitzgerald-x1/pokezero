# Champion search ladder

This is a separate, explicit search study on untouched champion 9375. It does
not wait for a Phase C winner. It changes neither D3's training teacher nor the
older exact-chance Phase D prerequisite.

The raw producer in `pokezero.phase_d_ladder_panel` reuses the original
`ChampionWorkerRuntime`, B2 owning encoder, controlled Foul Play runner and
literal terminal scorer. It claims a cell before numerical work, records its
owning wire before action selection, and writes each result exclusively. Its
500-cell screen is the actual common raw control. The full declaration is
1,000 seeds × both seats, with the first 250 seeds × both seats as the screen.
Every rung uses those same seeds and Foul Play at 1,000 ms. A CPU actor and
Foul Play process use distinct dedicated cores; compiled functions and model
persist, while every game gets fresh providers and history.

`phase_d_ladder_freeze.py` freezes 200 metadata-ranked roots from exactly the
first 64 closed declared raw cells. It opens only their captures and includes
partial, capped and failed captures. It never uses a score or a constructor
result. The four turn bands each contribute 50 roots; an unavailable band is
reported instead of being filled with easier roots. Missing capture games are
listed beside the actual root denominator.

An interrupted claimed cell with no result can close only through an explicit
`--retired-claims PATH --retired-claims-sha256 SHA` manifest. Its schema is
`pokezero.champion-search-ladder.retired-claims.v1`: the original declaration
SHA, a bound `closure: {path, sha256}`, and `cells` containing each original
`cell`, `request_sha256`, `claim_sha256` and `start_sha256`. The closure schema
is `pokezero.champion-search-ladder.attempt-closure.v1`; it records a nonempty
`attempt_id`, the same declaration SHA, `state: CLOSED`, `dispatch_stopped:
true`, `all_workers_terminated: true`, the exact `retired_ordinals`, and
nonempty `terminal_evidence: [{path, sha256}]` referencing original terminal
receipts. The root owner creates it after verifying closure. The freezer
checks those bytes and the original fixed-path request/claim/START bindings,
retains UNKNOWN and every durable partial capture, and creates no result or
score. An ongoing or unbound claim cannot make the prefix eligible.

The constructor's new `ProtocolInputs` route under
`v5_public_world.materialize` accepts genuine `PublicRoot` data, B2's public
carry and the exact own request. It calls the original pure
`battle_spec_from_payload` → `map_spec` mapping with the same original
public payload. The bounded batch-2 sleep projection and removal arguments are
described in [constructor batch 2](phase_d_constructor_batch2.md). This route
does not fabricate a driver Session, History, witness or true-opponent donor. Posterior draws are checked
against B2's public candidate bits. Unsupported clocks and mapping failures
remain named refused world slots. The original retained-driver route remains
unchanged. All requested 16/32-world roots remain in the construction table.

Own Hidden Power uses the device's typed move ID with the separately stored
IV-derived type and power, matching the existing pure `map_spec` convention.
Own happiness is read literally when provided. Otherwise a suffixed owning
Return/Frustration request ID can determine it only when inversion of the
pinned Showdown base-power callback leaves one integer in 0..255. The request
producer is `sim/pokemon.ts:getSwitchRequestData` and the callbacks are in
`data/moves.ts` at the pinned f76228a revision. No generator default is used:
plain or ambiguous powers, unreconstructable sleep evidence, unsupported replacement
phases and ambiguous own static facts remain refused. This bounded repair
shares the declared one-batch/one-remeasurement coverage budget.

`phase_d_constructor_coverage.py` runs that same construction on the host while
the shared GPU evaluates the value bank. It selects CPU before importing JAX,
accepts no model archive, and reuses one worker's compiled B2 functions. Use
`--worlds 16` or `--worlds 32` on the same frozen roster, with distinct output
directories and an optional persistent `--compilation-cache`. Interrupted
claims stay unknown and only not-attempted roots resume. This table reports
construction coverage/refusals, and makes no search latency or GPU performance
claim. It shares the original single coverage fix/retest budget. The GPU
Search delegates to the exact same `ProtocolConstructor.prepare` body using
its existing compiled functions, seen set and public session dictionary; the
separate saved-root search timing admission still requires the GPU.

The final named constructor tail is documented in
[final constructor tail](phase_d_constructor_tail.md): publicly resolved faint
replacement phases, positively fresh Substitute, and public Wish timing under
the pinned Gen 3 recipient-healing rule. It keeps the original 200 roots and
16/32 draw prefixes, targets 90%, and ends the constructor program after one
reviewed measurement regardless. Native/private and acting-latency admission
remain separate and unavailable. No other tail mechanics are opened.

The rung order is R1 (2 worlds, value), R3 (32 worlds, value), R4 (32 worlds,
8 sampled chances), R7 (32 worlds, depth 2), and R5 (8 worlds, 8 raw-policy
rollout leaves). All legal replies are priced by the frozen champion's
opponent-owned policy. The common root operator chooses the largest mean
completed Q across worlds, replies and chance samples. D's training target's
`raw logits + Q` policy target is not the acting operator. Raw B2 logits and Q
are recorded for the aggregation diagnostic. Depth 2 uses the best completed
second-ply action within each determinized successor, then averages those
values for the common root action. Rollouts use deterministic raw policy and
fresh declared chance keys, with caps and any required refusal making the
whole searched request unavailable. This is an explicit sampled-chance
ladder; it never silently substitutes samples for the original exact mode.

One GPU server owns the champion and dense world/branch batches for all
concurrent CPU panel games. No game owns a separate GPU. The constructor-only
server mode reports no search latency claim. Search timing uses five original
metadata-ranked roots per turn band (20 total), selected before constructor
results, with full-request p95 and untimed roots retained as unavailable. The
30-second admission bar remains against the 60-second acting deadline. The
panel client waits at most 45 seconds, leaving time for the explicitly counted
raw fallback; it never retries the root.

A rung's 500-cell point difference of at least three percentage points extends
its same declaration to 2,000. With missing cells, this implementation requires
the full-roster worst-case difference to meet three points, rather than using
a favorable survivor estimate. Reports retain known-pair intervals, missing
identification bounds, discordance, refusal/fallback shares and timing. The
paired interval clusters the two seats by seed. No interval is reported from
zero sample variance. Final arm choice uses reviewed lower bounds with the original missing-cell
caveat; this reducer does not automatically qualify strength.

The privileged/rollout value bank is a separate producer sharing the same GPU
allocation. It is not invoked by these import paths. The diagnostic order is
X1 children head, X2 leaf ablation, X3 oracle worlds, X4 world-count curve,
X5 leaf fidelity, X6 reply model, X7 aggregation and X8 coverage. One named
cause permits one fix and one ladder rerun; this source adds no open-ended
mechanics or visits sweep.

No numerical gate result, construction rate, p95 admission or strength claim
follows from source or inert tests. Scientific producer review and actual
fixed-roster measurements are required before dispatch.

## Reopened constructor: guard-defect batch

The Phase D host constructor keeps the engine, device trackers, D teacher and
shared determinization unchanged. A present fainted foe has B2's literal0/0
condition; only live foes require a100denominator. The own sampled packed team
is reordered to the current request before mapping. Whole packed entries move
without changing their fields, and the existing physical/request-order equality
remains an invariant. A literal typed own Hidden Power filters static variants
before the uniqueness test.

Live foe HP is sampled uniformly from the integer inverse of the actual Gen3
public display for that world's sampled maximum.100means full. Below full the
display is min(99,ceil(100*h/M)), so the99bin includes the below-full ceil100
tail. The world kernel seed owns these reproducible draws; no true opponent HP
or private packet is read. Public carry is rebased onto the resulting sample.

The one batch is measured on the original200roots at16and32worlds. Its coverage
bar is55%overall and40%in the latest turn band. Sleep and the other named tail
remain refusals for this batch. Construction success alone does not establish
oracle-support, own-state equality or value sensitivity: report unavailable
validation explicitly. The old captures do not contain certified true-opponent
state; no truth is reconstructed from guessed choices. Acting search remains
held until the full validation, coverage and unchanged latency gates pass.
