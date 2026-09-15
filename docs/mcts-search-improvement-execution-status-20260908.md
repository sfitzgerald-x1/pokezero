# MCTS search improvement: execution status and next decisions

Last updated: 2026-09-14. This is the live execution companion to the offline plan. It distinguishes implemented safety/correctness work, measured mechanics, and actual playing strength. Nothing below treats a green test, synthetic panel, or new runner as a strength result.

## Objective

Improve MCTS decisions at a fixed practical per-decision budget by taking the shortest valid proof path:

1. establish that a mechanism is correct;
2. show its decision or latency effect on representative development inputs;
3. run one attributable, source-bound MCTS-versus-MCTS comparison; and
4. retain or park the mechanism according to the declared result.

This avoids changing production choice from a toy observation and building evaluation machinery without a decision it can make.

The active corrected-versus-uncorrected backup comparison is **work-capped**,
not fixed-wall. A favorable result would be a strength signal under its
declared work cap only; it must not be relabelled as equal-deadline strength.

## Current evidence

| Line | Current state | Establishes | Does **not** establish |
| --- | --- | --- | --- |
| Batched backup repair | Merged in [#1337](https://github.com/sfitzgerald-x1/pokezero/pull/1337), merge `df4e3ce15ee69f922f6ae1b81c7b5e9861828319`. | Colliding batched chance visits no longer dilute a known terminal win or constant leaf value. | Better play. |
| Corrected action-choice panel | Merged in [#1338](https://github.com/sfitzgerald-x1/pokezero/pull/1338). Both seats and batches 1/2/8/64 pass immediate-terminal, nested-value, rare-decoy, and equal-value controls. | The repaired tree retains the correct action in its declared deterministic cases and exposes finite-budget value error. | Model-path behavior, multi-world behavior, or strength. |
| Raw Q-max selector | Merged in [#1354](https://github.com/sfitzgerald-x1/pokezero/pull/1354), merge `8616a2afb5eff426ec4f37a87e2fe4a8a649c723`. Its terminal-free, two-seat noisy-Q control shows visit-max taking the high-visit action while raw Q-max takes the noisier low-visit action, with a synthetic regret gap. | The conservative stop rule has an exact model-bias counterexample. | How often that error happens in games, or any game-strength estimate. |
| Encoder fast path | Merged in [#1340](https://github.com/sfitzgerald-x1/pokezero/pull/1340), merge `98faa804188289ecdcc5f5b87eaae8ad39de77eb`; full-path timing is a stable null and this line is parked. | Identifier normalization has bitwise-output coverage and the microbenchmark did not survive to a source-attributable full-path benefit. | A full-search speedup or extra useful work at the same clock. |
| Timing replay | [#1339](https://github.com/sfitzgerald-x1/pokezero/pull/1339) is merged and its fixed corpus was measured candidate-first twice, then baseline-first. | The candidate's apparent slowdown reverses in the baseline-first block; the result is order/warm-state dominated and is a valid null. | A source-attributable full-search win, numerical-tree parity, or extra useful work at the same clock. |
| Public-state approximation audit | A read-only audit of the canonical 16-decision timing corpus at `/shared/scott-experiment/mcts-encoder-fullpath-timing-corpus-cd1fbd57-20260908/timing-corpus.jsonl` (corpus SHA-256 `6d4be46153251e8a615275e600a9e557fb109609c9fd3111e7d087c27a6d8d11`; raw-file SHA-256 `a1930e513149d39166fc8fe5e0ddbbfd509cd0d4e50f0319f8f20f29d39a6203`) found no decision with an active public sleep, Substitute, partial-trap, confusion, or Yawn state. Historical event prefixes and move lists may mention Substitute or Rest, but they do not constitute an active target state. | None of these approximations is measurable with the present corpus; it is not a justified search-change or strength-study candidate. | That the states are unimportant in play, that their implementations are correct, or any performance effect. |
| MCTS-versus-MCTS runner | Merged in [#1341](https://github.com/sfitzgerald-x1/pokezero/pull/1341), merge `c43abac53c4fa6b9f5ca5db453f7e2e0e720ef92`, with source-isolated policy transport and receipt validation added afterward. | Fresh source-bound mirrored games, atomic game units, provenance binding, and fail-closed resumes are implemented. | A score or a timing-equality claim. |
| Whole-decision deadline | [#1356](https://github.com/sfitzgerald-x1/pokezero/pull/1356) introduced the model-only, fixed-work deadline; [#1359](https://github.com/sfitzgerald-x1/pokezero/pull/1359) hardened the native-invocation witness and fail-closed validation. The qualification source is their reviewed combined `main` merge `8609d301399738a8081f0e938a3cc4ed7d39abdd` (reviewed #1359 head `6e2bb3ac4fb50abf7fb358c250a7398686153cff`). The clock begins before folding and belief construction, reaches native setup and traversal, and records total elapsed time, overshoot, exhaustion, and skipped worlds. | A soft, whole-decision clock with a completed-tree-only native prefix; a started native batch may finish and its overshoot is visible. | A hard latency cap, a guaranteed nonzero prefix on every cold host, comparable same-deadline policy behavior, or stronger play. |
| Model-world parallelism R6 timing preflight | The source-bound R6 Job completed cleanly on two CPU workers at source `017fb434e62bd24fa1c0f2e6ff7b043a56bace0f`, with all three 16-decision serial/parallel pairs preserving fixed work and decision outputs. Warm serial medians were 10.802s, 11.191s, and 10.813s; two-worker medians were 5.408s, 5.488s, and 5.437s: 1.997x median speedup (range 1.989x–2.039x). | Two independent CPU model-world evaluators can nearly halve fixed-work decision wall time without changing the validated fixed-work decision results. | Same-deadline behavior, a hard latency limit, a strength gain, or permission to reinvest the saved wall time without a separate remaining-budget qualification. |
| Backup-repair strength pilot | The first frozen 12-pair/24-game corrected-versus-uncorrected run and its R18 replacement remain non-bankable for their independent terminal-contract defects. The fresh R20 replacement completed cleanly, with all 12 pairs, 24 games, and 48 policy receipts captured in an immutable terminal snapshot. Its work-capped candidate score was 0.375 (declared interval 0.2917–0.4583) and the frozen readout says `no_automatic_extension`. | The canonical launcher and terminal-capture path now work for a complete source-bound MCTS comparison; this particular backup-repair contrast did not produce a positive pilot signal. | A general no-effect claim, an equal-deadline result, a promotion decision, or use of the 50 reserved confirmation seeds. |
| Opponent-side model priors | R4's complete applicability result was a terminal `NONPASS`: candidate root-prior fallbacks were 104 and incumbent fallbacks were 4. The fresh R7 source-refusal diagnostic then completed cleanly on source `5ac2f0e68319b02aa7c2e53e3b1b9d879369242e`: its immutable `mcts-opponent-prior-refusal-diagnostic-r7-20260914-terminal` snapshot validates four mirrored pairs, eight games, 16 policy receipts, zero restarts, 575 candidate model-priced opponent arms, zero incumbent arms, and zero root-prior fallbacks for both arms. The separately registered P1 pilot then completed all eight fresh mirrored pairs (16 games, 32 policy receipts) with zero restarts, 750 candidate opponent-prior arms, zero incumbent arms, and zero root-prior fallbacks. Its point estimate was +0.0625, but its declared 80% paired-bootstrap lower delta was exactly 0.0, not strictly positive. | The current four-world/depth-2 configuration reaches the intended opponent-prior code path cleanly, but P1 does not earn a confirmation roster or a strength claim. | A playing-strength improvement, an automatic confirmation, a retry of P1, or production promotion. |

## Active route

### 1. Record the selector control and retire selector work

[#1354](https://github.com/sfitzgerald-x1/pokezero/pull/1354) passed its complete
fidelity gate and is merged. Its narrow result is that raw Q-max can prefer a
low-visit noisy action in a terminal-free, both-seat control. That supports
parking the mechanism; it neither estimates real-position error frequency nor
demonstrates a loss in games.

No new selector telemetry, representative selector read, or selector mutation
battery is authorized in this iteration.

### 2. Preserve both non-bankable pilots; R20 parks backup repair as a strength path

The first pilot completed all 24 durable game records and 48 matching worker
receipts, but its terminal handoff is invalid. The shell launcher wrote a valid
JSON object followed by literal `\\n` bytes, so `runner-terminal.json` is not
parseable even though its JSON prefix reports `COMPLETE` and exit code zero.
Do not repair, replace, or interpret that extant artifact, and do not inspect
its score or readout. The frozen contract did not authorize a retry.

The reusable launcher added with this status update creates an immutable
attempt receipt and log under `launcher-attempts/`, then atomically links
exactly one newline-terminated JSON terminal receipt after the child exits. A
reused attempt id or existing terminal is refused; a new attempt id can resume
an interrupted, non-terminal scorer root without overwriting prior diagnostics.
One root-scoped advisory lease is inherited by the scorer, so a hard-killed
launcher cannot admit a recovery attempt while its child still writes the root.
The child also receives the immutable launcher-attempt identity, receipt path,
and output-root path, allowing a scored-study contract to require this exact
handoff rather than relying on an optional wrapper convention.
Its focused tests cover successful and failed child exits, interrupted-attempt
preservation, orphan-child exclusion, existing-terminal refusal, and rejection
of a diverging runner `--out-dir`.

R18 was the independent, work-capped replacement study, freezing corrected
`6c1afe3b27d28f874c666a1ae34d5a92720bf4c0` against historical uncorrected
`aab7d480780fbfd3061458e17e35c1dc598b2c94`, the same iteration-9375 checkpoint,
12 fresh pilot seeds, and 50 disjoint confirmation seeds. It completed with
zero restarts and all 24 game records and 48 policy receipts under
`/shared/scott-experiment/mcts-backup-repair-current-pilot-r18-20260910`.

Its terminal summary is nevertheless invalid: the immutable manifest declares
10,000 resamples, seed `20260910`, and confidence level `0.8`, while its generic
summary reports the default 95% bootstrap lower bound (`0.375`) instead of the
declared 80% reconstruction (`0.4166666666666667`). The summary therefore
violates the predeclared analysis contract and cannot be captured, repaired, or
rerun. Its observed 0.4583 candidate point score is below neutral in either
interval, so no favorable pilot result exists and the reserved confirmation
seeds stay unused. Keep #1337 as a correctness repair; park it as a current
strength mechanism.

R20 was the fresh corrected-versus-uncorrected read with a new, disjoint roster
and the canonical launcher-terminal path. It completed with zero restarts and
its immutable terminal snapshot independently validates all 12 pairs, 24 games,
and 48 policy receipts. The work-capped readout reports candidate score `0.375`
with its declared interval `[0.2917, 0.4583]` and decision
`no_automatic_extension`. This is a clean negative pilot, not an inconclusive
handoff: retain #1337 as a correctness repair and park backup repair as a
current strength mechanism. It does not alter either preserved predecessor root
or authorize use of the reserved confirmation seeds.

### 3. Opponent-side model priors: terminal applicability NONPASS

The current model path explicitly defaults `use_opponent_priors` to false, so
the in-tree opponent is uniform even though the checkpoint exposes a separate
opponent-action head. This is a concrete modelling error, not a request for a
PUCT or budget sweep: the candidate changes that one boolean to true while
keeping source, checkpoint, belief construction, model priors, work cap, and
all other search settings fixed.

An older same-build study found a large opponent-prior effect, but that result
used an earlier checkpoint and passed through several now-repaired opponent
request-order bugs. It is hypothesis evidence only; it cannot establish the
current final-enthalf behavior. The existing implementation now fails closed
when it cannot construct a correct opponent request order, and it records
`opponent_prior_arm_decisions` when the native tree actually uses the opponent
head.

The source-isolated transport preserves that applied counter in every durable
policy receipt. A candidate with a missing, regressing, zero, or
fallback-contaminated applied counter is terminal `NONPASS`, regardless of its
games or score. Only a clean, applied read may register a fresh paired pilot
with new seeds. R18's pilot and confirmation rosters remain permanently
excluded.

The create-only applicability deployment, capture, and failure paths are
prepared and tested in pokezero-deploy PR #867. The fresh R3 applicability Job
was admitted on a spare arm64 engine node with a digest-qualified receipt and
immutable manifest, but its receipt selected source commit `3301f285` from
before the merged bootstrap-contract repair. It failed before its first game
with `NameError: bootstrap is not defined`; its preserved artifacts are a
terminal source-admission failure, not an applicability result. No game,
telemetry, or score evidence from R3 is bankable.

The fresh R4 replacement was create-only under new object and shared-artifact
names. Its renderer verified the clean checkout tied to the receipt contained
the bootstrap-contract assignment, recorded that runner-source hash in the
immutable manifest, and rejected the old `3301f285` tree before Kubernetes
objects could be created. It was bound to the independently successful R19
receipt: source commit `eaf37e7` and digest
`sha256:f0487300616c1c5e6ef466e53554360879f70e83db7fc9df3310ecb569734514`.

R4 then completed all eight mirrored games and all sixteen worker receipts with
zero restarts. Its candidate applied model-priced opponent arms 249 times and
the incumbent's counter remained zero, but the durable readout recorded 104
candidate and 4 incumbent root-prior fallbacks. The gate therefore wrote the
terminal `OPPONENT_PRIOR_APPLICABILITY_NONPASS` readout and the immutable
`mcts-opponent-prior-applicability-r4-20260911-terminal` snapshot; the capture
path is fixed in [pokezero-deploy #875](https://github.com/sfitzgerald-x1/pokezero-deploy/pull/875).
This is a complete, source-bound falsification of clean applicability, not a
partial run and not strength evidence. Park the one-boolean candidate. A later
source-level diagnosis may explain the fallback sites, but must not replay R4,
reuse its games as a score, or turn this result into a parameter sweep.

That bounded diagnosis is now complete. The candidate fallbacks are entirely
localized to `seed-2026092002`, searching as `p2`: 104 live-root fallbacks are
exactly 26 decisions across four sampled worlds, and the same side records
4,385 interior fallbacks. The other seven mirrored game-sides have zero
candidate root and branch fallbacks. That root-plus-interior shape identifies
the existing fail-closed path for an unavailable opponent request order: with
no verified order, the native mapper returns an all-unmapped opponent map at
the root and preserves that refusal through every branch. It is not evidence of
a bad policy head or a score effect. The sealed R4 artifacts cannot distinguish
why the upstream public-order walk declined the order (for example, a lost
permutation versus inconsistent public history), so no mechanics change is
justified from this run. A future, separately admitted source diagnostic must
record that refusal reason before attempting to repair the public-order path.
Its terminal evidence must distinguish an empty or duplicate sampled party, a
rejected public-order walk, a lost active permutation, and a non-permutation
result; a missing reason is a diagnostic failure, never an implicit zero.

R7 was that separately admitted source-refusal diagnostic. It used fresh,
four-seed mirrored development inputs and source
`5ac2f0e68319b02aa7c2e53e3b1b9d879369242e`; its only purpose was to retain the
source-order status alongside every live-root fallback while holding the
one-setting opponent-prior contrast fixed. Its immutable terminal snapshot
`mcts-opponent-prior-refusal-diagnostic-r7-20260914-terminal` now validates a
complete PASS: four mirrored pairs, eight games, 16 matching policy receipts,
zero restarts, 575 candidate model-priced opponent arms, zero incumbent arms,
and zero candidate and incumbent root-prior fallbacks. All 2,770 recorded
candidate source-order statuses were `resolved`; neither arm has an unclassified
root status.

This is a clean applicability decision only. The R7 game outcomes remain
diagnostic-only and must not be scored. The historical same-build null makes
this configuration eligible for exactly one separately registered, short,
fresh-seed, fixed-work strength pilot—not automatic confirmation or promotion.
That pilot must predeclare its roster, effect threshold, interval rule,
fallback/refusal gate, applied-prior denominator, and no-extension rule before
any outcome is read. A terminal failure or an unfavorable/inconclusive pilot
parks this one-boolean mechanism without reusing R7's games.

That P1 pilot is now complete. Its source-bound root contains all eight fresh
mirrored pairs, 16 games, 32 isolated-policy receipts, and a clean runner exit
with zero Pod restarts. The one-setting contrast retained the applicability
preconditions: the candidate recorded 750 model-priced opponent arms, the
incumbent recorded zero, all observed candidate request-order statuses were
`resolved`, and both root-prior-fallback counters remained zero. The candidate
won nine games and lost seven, yielding seven paired scores of 0.5 and one of
1.0: point score 0.5625, hence Δ=+0.0625 versus neutral. Under the predeclared
10,000-resample, 80% paired bootstrap (seed `20260924`), its score interval is
[0.5, 0.625] and Δ interval is [0.0, +0.125]. The interval rule was strict:
the lower bound must be greater than zero. P1 therefore earns **no extension**
and supplies no playing-strength claim.

P1 also exposed a runner visibility defect: its immutable manifest carried the
complete strength-readout contract, but the runner wrote only the generic
summary and applicability artifact. The deterministic calculation above is a
validation of the completed root, not a synthesized terminal artifact and does
not alter P1. [#1390](https://github.com/sfitzgerald-x1/pokezero/pull/1390),
merged as `973c3cc41e0181bcc42c2690beba34c7a9a92739`, closes that visibility
gap for future source-bound pilots: the runner now atomically writes
`OPPONENT_PRIOR_STRENGTH_PILOT_READOUT.json` before `COMPLETE.json`, binds its
SHA-256 in `COMPLETE.json`, validates the exact registered bootstrap and
applicability conditions, and records either
`ELIGIBLE_FOR_SEPARATE_CONFIRMATION_REGISTRATION` or `NO_EXTENSION`. A clean
null therefore terminates normally as `NO_EXTENSION` rather than requiring
manual interpretation. This repair neither alters P1 nor authorizes a retry,
extension, or production promotion of opponent-side priors. No further pilot
is currently registered because P1 did not meet its strict interval rule.

### 4. Keep the implemented fixed-wall mechanism ready for a future candidate

[#1356](https://github.com/sfitzgerald-x1/pokezero/pull/1356) supplies the
previously missing model-path decision clock without altering the live
work-capped pilot. It begins before live-fold advancement and belief-world
construction; the remaining time reaches the native boundary before parsing,
root setup, and traversal. The native tree checks before beginning each batch,
then finishes every selected traversal and backup before returning. Python
accounts for deadline-skipped worlds before native search, then measures the
complete decision through action mapping and records elapsed time, overshoot,
and exhaustion.

This is deliberately a **soft** whole-decision deadline, not a hard wall-time
guarantee: setup, a started native batch, mapping, and telemetry can overrun.
A one-millisecond integration smoke legally allows zero completed visits on a
cold host. Before a timed comparison, qualify the intended runtime with a
measured nonzero completed prefix, the actual latency-tail/overshoot
distribution, zero-completed-world refusal behavior, and the identical clock
configuration on both policies. Until then, report per-decision latency and
completed work as measurements only.

### 4a. R6 follow-up: deadline-safe model-world dispatch

R6's timing PASS is an engineering preflight, not a license to drop the
existing deadline guard. A queued world must compute its budget from the actual
remaining decision wall immediately before its native call; it must be skipped
if that wall has already expired. Each started world must retain that exact
per-invocation budget in the normal native witness, while report absorption
stays in source world order. A later source-bound deadline qualification must
require this remaining-budget dispatch mode, the configured worker count, and
agreement between its recorded native invocation count and the independent
deadline ledger. That is the minimum evidence needed before measuring whether
the R6 throughput headroom improves equal-deadline play.

### 4b. Public-state approximation candidates are unmeasured

A bounded source audit identified four potentially consequential public-state
approximations: generic induced-sleep duration, Substitute health, partial-trap
duration, and duration-bearing volatiles such as confusion and Yawn. This is a
candidate inventory, not evidence that any correction improves the search.
Rest already has its own exact, scoped provenance path (`restSleepAttempts`);
generic induced sleep is different because the public snapshot does not expose
its remaining duration. Replacing that uncertainty with a single guessed value
would be a new approximation, not a fidelity repair.

The pinned timing corpus has a manifest plus 16 decision records. A read-only
inspection of each decision's current `public_belief_inputs` found zero active
instances of all four target families. Some histories contain a completed
Substitute or a historical Rest event, but no selected decision presents the
state whose representation would be changed. The corpus therefore cannot
separate a correction from an inert code path.

Do not implement, benchmark, or strength-test these candidates now. A future
candidate first needs a source-bound corpus with explicit active-state coverage,
an exact declared oracle or counterfactual for that state, and an action-choice
control before it can enter a bounded MCTS study. This is a coverage stop
decision only: it does not claim that the states are rare or unimportant in
actual play.

### 5. Fixed-deadline qualification contract (only after a new candidate earns a timed study)

This is one bounded, development-only gate for the already-merged deadline;
it is neither a game trial nor evidence of stronger MCTS. It is required before
a future same-deadline contrast, but is not an automatic cluster run now that
the only active source-isolated candidate is parked. It uses the existing
public replay timing corpus and replay-backed model path, rather than a new
benchmark or evaluation runner. The corpus has 16 development decisions and is
not part of either the 12-pair pilot or its reserved confirmation roster. Its
canonical corpus SHA-256 is
`6d4be46153251e8a615275e600a9e557fb109609c9fd3111e7d087c27a6d8d11`
(raw-file SHA-256
`a1930e513149d39166fc8fe5e0ddbbfd509cd0d4e50f0319f8f20f29d39a6203`).

Freeze the following before the first timed decision:

- qualification source `8609d301399738a8081f0e938a3cc4ed7d39abdd`, including
  the reviewed #1359 deadline-witness head
  `6e2bb3ac4fb50abf7fb358c250a7398686153cff`, its matching native build, the
  final-enthalf iteration-9375 checkpoint and its registered SHA-256, the corpus
  hash, CPU runtime, and the exact Showdown/vocabulary receipts;
- fixed model allocation: depth 2, 256 requested simulations **per native
  world invocation**, batch 16, four worlds, `early_stop=false`, and both
  selector flags disabled. A collapsed duplicate is one native invocation with
  its request scaled by multiplicity; 256 is never misreported as a
  per-decision simulation cap; and
- one whole-decision request of **1,000 ms** (`model_decision_time_ms=1000`) on
  every corpus decision. This is a qualification target, not a budget to tune
  after observing its result.

The generated per-decision records must retain two distinct clocks. The
existing outer timing boundary excludes prefix replay and fold warm-up, then
measures through validated Showdown choice serialization. The inner deadline
witness covers the model decision and is not a substitute for that outer wall
time. Report both rather than treating either as the other. Each record must
add the decision's requested budget, deadline elapsed time, deadline overshoot,
exhaustion, searched and constructed worlds, deadline-skipped worlds, and exact
fallback/refusal cause. It must also retain every native invocation's
multiplicity, requested and completed iterations, remaining iterations,
`time_budget_exhausted` witness, and each seat's finalized root-visit total.
The terminal summary must retain p50/p95/max outer elapsed time and deadline
overshoot, the count of exhausted decisions, total and per-decision world
coverage, and the count of zero-completed-world refusals. A missing witness is
a terminal NONPASS, not a zero. A strict fallback that raises before emitting an
invocation receipt is likewise a terminal NONPASS; it must not be recast as a
receipt-bearing fallback or silently retried.

The qualification may permit a fully completed fixed-work decision, but it
passes only if at least one native invocation witnesses
`time_budget_exhausted=true` with a **nonzero, finalized** prefix
(`0 < completed_iterations < requested_iterations` for that invocation).
Every such prefix must retain per-seat root-visit conservation; a
decision-level exhaustion caused only by later mapping or skipped worlds does
not satisfy this gate. There must be no fallback, invalid-action, or
zero-completed-world decision. Deadline-skipped worlds are not failures or
silently discarded: their distribution is part of the result. A timed
MCTS-versus-MCTS contrast remains blocked unless that result records the same
clock contract and makes its candidate/incumbent belief-world coverage
comparable.

The historical `dacb635` predecessor cannot accept
`model_decision_time_ms`; the isolated transport correctly rejects it rather
than omitting a behavior-bearing field. Therefore a later timed baseline must
be an explicitly source-bound **derived predecessor** that ports only this
deadline contract, with its diff reviewed against `dacb635`. It must never be
relabeled as the historical predecessor, and the active work-capped pilot
remains unchanged.

## Resource, durability, and stop rules

All cluster work stays in `scott` on `olfusa`. CPU-heavy work first finds an engine node with actual spare capacity and binds there. New GPU work requests whole GPU groups, never fragments a node, and uses at most two nodes at a time. Every long run writes atomic progress and complete units, emits PASS/NONPASS terminal state, retains failure diagnostics, and is validated before it changes source. The pilot launcher atomically records the runner exit code, while the runner separately validates every source-bound game receipt; a superficial PASS marker is not sufficient.

- Keep the backup repair even if playing strength is flat: it is a correctness repair, not a claimed strength win.
- Preserve the completed first backup-pilot artifacts as non-bankable. The malformed terminal receipt is a process-handoff failure, not evidence that may be reconstructed from its valid prefix or its complete-unit files; its outcomes remain unread.
- Park raw Q-max after #1354's terminal-free noisy-Q control merges; do not rescue it with a threshold or selector grid.
- Keep the encoder line closed: full-path timing is null/order-sensitive.
- The stopped rollout-mutation sweep is preserved but invalid: its wrapper emitted PASS after interrupted, incomplete progress with zero controls. It is not a result or a prerequisite.
- Do not start a PUCT/depth/simulation sweep, broad cache rewrite, or retraining run as a substitute for these reads.
- The R74 Wave 03 recovery decision is recorded in
  [the recovery decision](r74-recovery-decision-20260911.md): its missing
  terminal snapshot makes that registered bridge incomplete and non-bankable.
  It remains separate from MCTS-improvement evidence and cannot be synthesized
  from partial artifacts.

## Immediate order

1. Record #1354's merged raw-Q control as the selector stop decision; do not extend selector work in this iteration.
2. Preserve both non-bankable pilots. Do not rerun or recapture R18, do not inspect the malformed first pilot's score, and do not spend the R18 confirmation seeds. R20 is terminally captured and negative, so retain the repair as correctness-only and do not extend that line.
3. Preserve R3's pre-game source-admission failure, R4's complete terminal `NONPASS`, and P1's completed root. Do not rerun, recapture, or extend any of them. R7 proved clean applicability, but P1's lower paired-bootstrap delta is exactly neutral, so park opponent-side model priors as a strength mechanism. The future-pilot strength-readout repair merged in #1390; it does not revive this parked candidate.
4. Retain #1356's fixed-deadline qualification as a prerequisite for a later timed contrast, but do not run it merely to produce more evaluation data. It follows only if a different cleanly applied candidate earns a timed study.
5. R18, the earlier backup-repair attempt, and clean R20 read did not promote backup repair. Retain the correctness repair; no PUCT/depth/simulation rescue grid, automatic cluster rerun, or reuse of R18 seeds is authorized.
6. Retain the public-state approximation coverage stop decision. Do not modify generic sleep, Substitute, partial-trap, confusion, or Yawn handling until a new source-bound corpus includes the active state and a declared action-choice oracle.
