# MCTS search improvement: execution status and next decisions

Date: 2026-09-08. This is the live execution companion to the offline plan. It distinguishes implemented safety/correctness work, measured mechanics, and actual playing strength. Nothing below treats a green test, synthetic panel, or new runner as a strength result.

## Objective

Improve MCTS decisions at a fixed practical per-decision budget by taking the shortest valid proof path:

1. establish that a mechanism is correct;
2. show its decision or latency effect on representative development inputs;
3. run one attributable, source-bound MCTS-versus-MCTS comparison; and
4. retain or park the mechanism according to the declared result.

This avoids changing production choice from a toy observation and building evaluation machinery without a decision it can make.

## Current evidence

| Line | Current state | Establishes | Does **not** establish |
| --- | --- | --- | --- |
| Batched backup repair | Merged in [#1337](https://github.com/sfitzgerald-x1/pokezero/pull/1337), merge `df4e3ce15ee69f922f6ae1b81c7b5e9861828319`. | Colliding batched chance visits no longer dilute a known terminal win or constant leaf value. | Better play. |
| Corrected action-choice panel | Merged in [#1338](https://github.com/sfitzgerald-x1/pokezero/pull/1338). Both seats and batches 1/2/8/64 pass immediate-terminal, nested-value, rare-decoy, and equal-value controls. | The repaired tree retains the correct action in its declared deterministic cases and exposes finite-budget value error. | Model-path behavior, multi-world behavior, or strength. |
| Root-choice hypothesis | The panel's harsh-prior row has visit-max regret 0.4745 and shadow Q-max regret 0; the rare-decoy control rejects a simple lucky-terminal explanation. Shadow implementation is locally tested but not published. | One concrete completed-tree visit-lag mechanism is worth testing on real development positions. | Q-max is generally better or safe to deploy. |
| Encoder fast path | Merged in [#1340](https://github.com/sfitzgerald-x1/pokezero/pull/1340), merge `98faa804188289ecdcc5f5b87eaae8ad39de77eb`. | Identifier normalization has bitwise-output coverage and a narrow encoder improvement. | A full-search speedup or extra useful work at the same clock. |
| Timing replay | [#1339](https://github.com/sfitzgerald-x1/pokezero/pull/1339) is open. Its only failed CI gate is stale mutation evidence, not an engine-fidelity behavioral failure. A fresh source-bound battery is running. | The timing lattice and parity path are available after exact-source integrity evidence. | An end-to-end timing result. |
| MCTS-versus-MCTS runner | Merged in [#1341](https://github.com/sfitzgerald-x1/pokezero/pull/1341), merge `c43abac53c4fa6b9f5ca5db453f7e2e0e720ef92`. | Fresh mirrored games, atomic game records, provenance binding, and fail-closed resumes are implemented. | A native/checkpoint battle smoke or a score. |

## Active route

### 1. Close the timing-integrity gate without weakening it

The fresh battery must exit cleanly, restore exact source, and produce one complete artifact: every applied mutation killed; no survivor, skip, or not-applied outcome; matching controls. Only then may #1339 update checked-in evidence and re-run CI. A hand-edited hash or partial artifact is invalid. This is not a new search hypothesis; it is the bounded integrity prerequisite for using replay/timing evidence.

### 2. Turn the root-selector hypothesis into a representative read

After #1339 is green, rebase the shadow-only selector on current main and run fresh source-integrity evidence because it changes `engine_search.py`. The first read uses model-leaf, one-world, fixed-allocation searches with early stop disabled. It preserves production visit-max and records from the same completed tree: acting-seat Q-max and visit-max action; their visits and Q values; disagreement and unmeasured denominators with causes; all fallbacks/refusals; and low-visit-Q/noisy-control rows. The panel earns this read, not a production selector switch. Park Q-max if representative positions show no useful regret signal, noisy-Q regressions, or mostly unmeasured rows; do not rescue it with a threshold grid.

### 3. Obtain the first full-path encoder result

Once #1339 is green, execute its fixed representative timing panel serially, alternating baseline and candidate in isolated processes. Record parity, completed work, total decision wall, and existing subphase counters. Promote only a result with matching arrays, actions, values, and refusal/fallback behavior at fixed work. A vanished end-to-end gain is a valid negative result and closes this optimization line.

### 4. Use one attributable strength comparison

The first strength candidate must be selected by the representative selector read or the full-path timing result. Freeze final-enthalf iteration 9375 with SHA-256 `0fd095923b4ac7e05d6e2b3ccab9c1e6869dff4893c2dae456caff10dce690be` only after verifying exported bytes. Before outcomes are read, declare candidate, incumbent, mirrored seeds, draw scoring, fixed-work or fixed-wall contract, retries, failure rule, resource cap, and precision rule.

The merged runner supports two configurations of the *same* source build, such as a surviving selector versus corrected visit-max. It deliberately refuses two different native source builds in one process. Corrected-versus-uncorrected and encoder-versus-baseline contrasts therefore require minimal isolated-build orchestration rather than bypassing the binding guard. That is an explicit implementation gap, not permission to call different sources comparable.

Start with a small separately declared native smoke only to validate routing, provenance, freshness, latency, and durability. It has no strength conclusion. Then run the declared mirrored seed comparison, including both seats and fresh trajectories after divergent moves. Report score relative to 0.5, seat splits, paired uncertainty, tail latency, model work, and all exclusion/failure counts. An interval spanning zero is inconclusive, not equivalence.

## Resource, durability, and stop rules

All cluster work stays in `scott` on `olfusa`. CPU-heavy work first finds an engine node with actual spare capacity and binds there. New GPU work requests whole GPU groups, never fragments a node, and uses at most two nodes at a time. Every long run writes atomic progress and complete units, emits PASS/NONPASS terminal state, retains failure diagnostics, and is validated before it changes source.

- Keep the backup repair even if playing strength is flat: it is a correctness repair, not a claimed strength win.
- Park selector Q-max if its real-position read fails declared regret/noisy-Q controls.
- Close the encoder line if full-path parity timing is null or worse.
- Do not start a PUCT/depth/simulation sweep, broad cache rewrite, or retraining run as a substitute for these reads.
- Keep the R74 missing-handoff recovery separate; it is not MCTS-improvement evidence and cannot be synthesized from partial artifacts.

## Immediate order

1. Validate the running fresh mutation battery and merge #1339 only after exact-source evidence and CI pass.
2. Publish the selector as telemetry only, with fresh evidence after rebase; collect its one-world development read.
3. Complete the full-path timing read; park or promote the encoder on its own evidence.
4. Implement only the isolated-build adapter required by the first cross-source contrast, or use the existing same-source runner for a surviving selector. Then run one bounded, predeclared strength comparison.
