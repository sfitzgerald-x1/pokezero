# MCTS search improvement: execution status and next decisions

Date: 2026-09-09. This is the live execution companion to the offline plan. It distinguishes implemented safety/correctness work, measured mechanics, and actual playing strength. Nothing below treats a green test, synthetic panel, or new runner as a strength result.

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
| MCTS-versus-MCTS runner | Merged in [#1341](https://github.com/sfitzgerald-x1/pokezero/pull/1341), merge `c43abac53c4fa6b9f5ca5db453f7e2e0e720ef92`, with source-isolated policy transport and receipt validation added afterward. | Fresh source-bound mirrored games, atomic game units, provenance binding, and fail-closed resumes are implemented. | A score or a timing-equality claim. |
| Whole-decision deadline | Merged in [#1356](https://github.com/sfitzgerald-x1/pokezero/pull/1356), merge `380066e8a8e7e16ab4e09f377f6802c11a4a2e85`. It is model-only and fixed-work-only: the clock begins before folding and belief construction, reaches native setup and traversal, and records total elapsed time, overshoot, exhaustion, and skipped worlds. | A soft, whole-decision clock with a completed-tree-only native prefix; a started native batch may finish and its overshoot is visible. | A hard latency cap, a guaranteed nonzero prefix on every cold host, comparable same-deadline policy behavior, or stronger play. |
| Backup-repair strength pilot | Active, frozen 12-pair/24-game corrected-versus-uncorrected comparison: depth 2, 256 simulations, four worlds, batch 16, CPU, final-enthalf iteration 9375. At the 2026-09-09 outcome-blind inspection, eight complete mirrored pairs (16 games and 32 receipts) were durable and the ninth pair's isolated workers were actively consuming CPU. | The live contrast is attributable and resumable; completed units have matching policy receipts. | Any outcome before all 12 pairs and one valid terminal readout. |

## Active route

### 1. Record the selector control and retire selector work

[#1354](https://github.com/sfitzgerald-x1/pokezero/pull/1354) passed its complete
fidelity gate and is merged. Its narrow result is that raw Q-max can prefer a
low-visit noisy action in a terminal-free, both-seat control. That supports
parking the mechanism; it neither estimates real-position error frequency nor
demonstrates a loss in games.

No new selector telemetry, representative selector read, or selector mutation
battery is authorized in this iteration.

### 2. Finish the backup pilot unchanged and read it once

Let the active pilot reach exactly 12 complete mirrored pairs. A valid terminal
read requires all 24 durable game records and 48 matching worker receipts, the
frozen checkpoint/source/engine contracts, and one complete PASS or NONPASS
terminal record. No partial read, retry reinterpretation, or outcome-driven
extension is permitted.

On a valid PASS, apply the frozen bootstrap rule: 10,000 resamples, confidence
0.80, and minimum effect delta 0.05. Promote only if that rule says so;
otherwise the result is inconclusive or negative as specified, not a reason to
change the rule. Any confirmation uses the reserved non-overlapping seed set
and its declared 50-pair bound.

### 3. Qualify the implemented fixed-wall mechanism before using it

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

## Resource, durability, and stop rules

All cluster work stays in `scott` on `olfusa`. CPU-heavy work first finds an engine node with actual spare capacity and binds there. New GPU work requests whole GPU groups, never fragments a node, and uses at most two nodes at a time. Every long run writes atomic progress and complete units, emits PASS/NONPASS terminal state, retains failure diagnostics, and is validated before it changes source. The pilot launcher atomically records the runner exit code, while the runner separately validates every source-bound game receipt; a superficial PASS marker is not sufficient.

- Keep the backup repair even if playing strength is flat: it is a correctness repair, not a claimed strength win.
- Park raw Q-max after #1354's terminal-free noisy-Q control merges; do not rescue it with a threshold or selector grid.
- Keep the encoder line closed: full-path timing is null/order-sensitive.
- The stopped rollout-mutation sweep is preserved but invalid: its wrapper emitted PASS after interrupted, incomplete progress with zero controls. It is not a result or a prerequisite.
- Do not start a PUCT/depth/simulation sweep, broad cache rewrite, or retraining run as a substitute for these reads.
- Keep the R74 missing-handoff recovery separate; it is not MCTS-improvement evidence and cannot be synthesized from partial artifacts.

## Immediate order

1. Record #1354's merged raw-Q control as the selector stop decision; do not extend selector work in this iteration.
2. Let the frozen backup pilot complete unchanged, validate its terminal artifact, then apply its predeclared analysis exactly once.
3. Before any same-deadline comparison, qualify #1356 on the intended runtime with a nonzero completed prefix, measured tail/overshoot, no-completed-world behavior, and identical policy clocks. This is a measurement gate, not a strength trial.
4. If the pilot promotes, run only its reserved confirmation contrast. If it does not, retain the correctness repair and choose the next mechanics investigation from a concrete remaining decision error.
