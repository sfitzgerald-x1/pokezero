# Next MCTS investigation: restore our own policy guidance

Date: 2026-09-15.

Status: **Plan written; no experiment launched or production setting changed.**
This is a new investigation, not an extension of either completed eight-pair
pilot. Final source/image identities and unused seed rosters must be recorded
before execution. The prior pilots and their `NO_EXTENSION` decisions remain
unchanged.

## Decision to make

Does enabling our own policy priors make the current MCTS stronger than
uniform-prior MCTS under the same practical decision-time budget?

Resolve this missing baseline comparison before making value-head training the
main next investment. A positive result identifies a useful search setting; a
negative or inconclusive result does **not** prove that the value head is the
bottleneck or that all search designs have been exhausted.

## Why this is the next step

- The corrected B2a bank yielded raw-policy score 55.875% versus oracle score
  67.5% over 400 paired outcomes: +11.625 percentage points, with the reported
  95% interval approximately [+5.5, +17.8] points. It demonstrates useful
  action-selection headroom under that oracle's conditions.
- B2a already used the policy to identify its top three candidate actions.
  However, it used a privileged simulator snapshot and one sampled immediate
  opponent reply for candidate scoring, followed by an independently sampled
  live reply. Its gain is neither a practical MCTS ceiling nor proof that
  choosing moves covering many opponent replies caused the improvement.
- The latest parallelism pilot changed one worker to two while disabling
  **our own** model priors in both arms. It obtained about 1.85x the iterations
  but an inconclusive strength result. Own-policy priors are normally enabled
  by default in the engine; their absence was frozen into that pilot's
  qualification configuration, not established as the strongest setting.
- The completed opponent-prior pilot tested a different flag. It cannot answer
  whether guidance from our own action head is valuable at the one-second clock.

The working hypothesis is that scarce search visits are being spent poorly
without policy guidance. Priors change exploration priorities, while the same
value head still evaluates the states search reaches. There is no assumption
that increasing agreement with the policy necessarily means better play.

## 1. Freeze one contrast

Both arms use the same reviewed source/build, environment, observations,
belief construction, checkpoint, model export, encoder tables, action mapping,
backup, and final action selector. Compare complete resolved configurations;
apart from arm labels, **only `model_priors` may differ**.

| Setting | Guided arm | Uniform-prior arm |
| --- | --- | --- |
| Own-policy priors (`model_priors`) | `true` | `false` |
| Opponent priors (`use_opponent_priors`) | `false` | `false` |
| Model-world workers | 1 | 1 |
| Whole-decision soft clock | 1,000 ms | 1,000 ms |
| Native batch guard | 64 ms | 64 ms |
| Requested belief worlds | 4 | 4 |
| Maximum simulations per world | 256 | 256 |
| Search batch / depth / PUCT coefficient | 16 / 2 / 1.4 | 16 / 2 / 1.4 |
| Leaf evaluator / device | Model / CPU | Model / CPU |
| Early stopping | Disabled | Disabled |

One worker retains the previous incumbent's setting and avoids introducing a
policy-prior-by-parallelism interaction. This study will not establish the best
two-worker configuration. Do not tune PUCT, depth, batching, opponent priors,
value weights, or selection while running this comparison.
The existing own-prior flag governs root and interior action guidance together;
this is not a root-only prior ablation.

Checkpoint: final v4 enthalf champion, iteration **9375**, SHA-256
`0fd095923b4ac7e05d6e2b3ccab9c1e6869dff4893c2dae456caff10dce690be`.
Planning source inspected: `a1dcdca67a36b4d32085d0c0e47934d5969ce62c`.
The executing source must be pinned separately if any preparation changes it.

This is an **equal requested soft-clock** experiment, not equal work or a hard
one-second latency guarantee. Include policy-pricing overhead inside the clock.
Do not give the guided arm extra time to compensate for that overhead.

## 2. Short unscored preflight

Reuse the existing prior-mapping tests, native reports, decision replay,
source-isolated H2H runner, and durable launcher. The old world-parallelism
contract deliberately rejects priors ON; leave it intact and use a separately
bound contrast. Do not build a new evaluator or loosen old contracts.

1. Run the focused prior ON/OFF, action-mapping, acting-seat value-frame, and
   config-transport tests. Missing native/model dependencies or skipped required
   tests are not passes. Reuse launcher/resume tests; do not kill an existing Job.
2. Replay 16 development decisions, eight per acting seat, selected without
   outcome labels and including ordinary multi-action and switch decisions.
   Use identical public observations/history and belief RNG seeds in each
   matched comparison. Replay both execution orders to expose warm-state bias.
3. Check that ON actually supplies mapped, normalized model priors at eligible
   roots and branches; OFF restores uniform exploration. Include a non-flat,
   multiple-action control where allocation differs. Equal-valued or forced
   decisions need not change their selected action.
4. Run four unscored mirrored development pairs (eight games) on the intended
   runtime. Verify fresh policy state per game, both seats, source bindings,
   complete durable game/worker receipts, zero decision/root-prior fallbacks,
   and a nonzero searched prefix on every search-eligible decision. Forced
   no-search decisions must be explicitly classified, not counted as searched.
5. Check the actual wall-time distribution, deadline ledger, and throughput.
   Admission targets: each arm's searched-decision p95 is at most 1.20 s and
   the guided/uniform mean searched-decision wall ratio is at most 1.05.
   Record cold starts and maxima separately; do not silently trim outliers.

The timing tolerances are proposed engineering limits for this new study, not
inherited guarantees. Freeze them before preflight; a miss is a timing issue
to diagnose, not permission to raise a limit after seeing results.

Cap preparation at one working day. If a real blocker needs more work, report
the specific missing capability and its effect on this decision. Do not launch
a chain of new instrumentation gates. Preflight outcomes are not strength
evidence and their seeds never enter the scored roster.

## 3. One fixed-size strength study

Run **400 fresh mirrored pairs: 800 games** of guided MCTS directly against
uniform-prior MCTS. Each team/environment seed supplies two games, with the
guided arm taking each seat once. Reset each isolated policy between games and
let both follow the actual trajectory after their actions diverge.

Before launch, materialize and hash the complete roster, deterministic seed
generation procedure, game order, shard assignment, and policy RNG rules. Check
against previous study/development and reserved confirmation rosters. Resolve
collisions before any scored game; never substitute seeds based on outcomes.
Balance which mirrored seat runs first. Keep both games of a pair on the same
node/shard, and give each node both arms rather than assigning one node per arm.

Scoring and analysis:

- Win = 1, genuine draw = 0.5, loss = 0, from the guided arm's perspective.
  Set a common safety cap of 250 decision rounds per game. Preserve capped
  games and score them as 0.5 under the existing convention; report caps
  separately from genuine draws. This estimates capped-match score if any caps
  occur, not uncapped battle win probability. Never drop capped games or count
  crashes as draws.
- Average the two mirrored games into one seed-pair score. The primary effect
  is **delta = mean guided pair score - 0.5**. This is head-to-head advantage,
  not the +11.6-point oracle-versus-raw-policy quantity.
- Use the existing paired percentile bootstrap: 10,000 resamples of whole
  pairs, 95% interval, fixed bootstrap RNG seed `20260915`. Do not treat 800
  games or thousands of decisions as independent pairs.
- Define **+5 percentage points** as the target useful head-to-head effect.
  Read the primary score once, after all 400 pairs validate. Operational
  monitoring may inspect progress, timings, and failures, not stop on wins.
  There is no pilot-to-confirmation ladder or automatic sample extension.
- If any games cap, also recompute the paired estimate/interval twice, assigning
  every capped game a guided loss and then a guided win. Any directional or
  target-effect conclusion must survive both assignments; otherwise classify
  it as cap-sensitive/inconclusive. A capped game is retained evidence, not a
  reason to replay that seed hoping for a different outcome.

Why 400 pairs: under an illustrative pair-score standard deviation of 0.354,
the normal-approximation 95% half-width is about 3.5 points, with roughly 80%
power for a five-point effect against zero. Because a pair score lies in
[0, 1], its standard deviation can be as high as 0.5: then the approximate
half-width is 4.9 points and the 80%-power effect is about seven points.
These are planning estimates, not a promised precision or a retrospective
reason to resize the study. An inconclusive answer remains possible.

## 4. Explain the result with a small diagnostic readout

Reuse the preflight's matched public-state replays to report:

- whether priors change root visit allocation and the selected action;
- visits allocated to the policy's top three legal actions;
- selected action's policy rank and its acting-seat Q/visit margin; and
- completed worlds, iterations per world, deadline skips, and latency.

Evaluate the reference policy once from each identical replay observation,
outside the timed comparison, to obtain top-three/rank labels for **both** arms.
OFF's existing `no_root_priors` telemetry is expected and must not be read as
agreement with the policy. Use native root reports where already available;
report unavailable fields rather than building a new trace collection system.

For scored games retain existing per-game/per-decision telemetry, seat splits,
fallback counters, and their full denominators. Aggregate game decisions are
descriptive because the arms encounter different states. They cannot replace
matched-state comparisons or act as independent statistical samples.

Greater policy agreement, a larger Q gap, or more simulations is not proof of
better choices. This step adds no oracle rollouts, privileged labels, opponent
reply grid, or claim about the “mid-ground” mechanism.

## 5. Execution, durability, and cost

- CPU-only on `olfusa`, namespace `scott`; no GPU needed. Use at most two nodes
  across all concurrent authorized work. Inspect live node health and spare
  engine capacity immediately before scheduling; bind each worker by affinity
  to its selected arm64 node. Do not reuse a historical node health verdict.
- Two balanced shards of 200 pairs each may run concurrently, each with exact
  48 CPU / 32 GiB requests and limits, one game process at a time. If only one
  suitable node is available, run the shards sequentially. Parallelize roster
  checks/readout preparation with preflight where they do not share mutable
  source or timing resources.
- The prior pilot took about 25 minutes for 16 games. Linear extrapolation is
  roughly 21 total worker-hours, or 10–11 hours on two comparable nodes before
  build, startup, priors overhead, and game-length variation. Re-estimate from
  unscored preflight before launch; this is not a live ETA. Proposed ceiling:
  48 hours per shard, with no live deadline extension.
- Reuse atomic progress and immutable completed-game records, isolated-policy
  receipts, and attempt logs. Progress should expose completed games/pairs,
  current game/decision, last update, elapsed time, and estimated remaining
  time. Persist a liveness/progress update at least every 60 seconds, including
  during a long decision. At most the in-flight game needs recomputation;
  preserve a completed first seat if its mirrored partner was interrupted.
- A clean final result requires every roster game and receipt to validate,
  source/config/checkpoint consistency, clean terminal attempts, and no
  unexplained restart or decision/root-prior fallback. Apply the registered cap
  sensitivity analysis. Report interior branch-prior fallbacks separately;
  they are not live-root failures.
- Infrastructure interruption may resume verified units with the same frozen
  experiment and fresh create-only attempt identity. Preserve prior logs and
  never overwrite completed units. A search/source defect cannot be repaired
  mid-study and pooled with earlier results; capture it and stop the strength
  claim. A summarization-only bug can be corrected in a new, explicitly derived
  readout from unchanged verified records, with the original retained; do not
  rerun good games just to repair report formatting or arithmetic.
- Inspect every 30 minutes only while a run exists; notify on completion,
  failure, or material ETA change. Remove the monitor after terminal handoff.
  Do not create a monitor for this plan-writing step.

## 6. Frozen interpretation and next action

First apply integrity and timing checks. A bound/receipt failure is an
**invalid experiment**, not a negative result about priors. If the full study
misses the same p95/mean-ratio timing limits used at admission, retain all
outcomes but label the readout timing-confounded; do not claim comparable
practical-budget strength or automatically rerun it.

For an eligible study, apply these cases in order:

| Result | Conclusion and next action |
| --- | --- |
| Delta >= +5 points and 95% lower bound > 0 | Useful positive evidence for own-policy guidance in this configuration. Use guided MCTS as the measured research baseline; leave production unchanged pending an explicit rollout decision. This is not evidence that a new value head is necessary. |
| 95% upper bound < 0 | Guidance harms this configuration. Inspect the matched-state allocation evidence for overconcentration/mapping/budget effects; no automatic PUCT or temperature sweep. Do not silently flip the engine's existing production default. |
| Otherwise, 95% upper bound < +5 points | The study rules out the target-sized benefit at its stated confidence. Report any smaller positive effect honestly. Prioritize a public-observation-valid value-quality investigation; do not claim the critic is proven guilty. |
| Otherwise | Inconclusive at this budget. Publish the estimate and interval, do not extend the roster, and provisionally prioritize value-quality work rather than another search grid. A later search hypothesis needs new evidence. |

Head-to-head superiority can be matchup-specific. Even a positive result does
not establish improvement against raw policy, FoulPlay, or a population of
opponents; that would be a separately scoped external validation, not an
automatic branch of this study.

## Deliverable and limits

Produce one concise result document plus a hashed derived readout: exact
contrast, all 400 pair scores, 95% interval, validity/timing verdict, compact
allocation diagnostics, and the selected next action. Preserve the original
inputs and unsuccessful attempts. No model training, two-worker retest, value
architecture change, source mechanics rewrite, or production rollout is part
of this investigation.

Planning is complete with this document. Execution still requires the short
preflight, final source/image/roster registration, and the scored run; none has
been performed by writing the plan.

## Evidence and existing implementation

- [Current execution evidence](mcts-search-improvement-execution-status-20260908.md),
  especially the opponent-prior and world-parallelism terminal readouts.
- [Engine configuration and root telemetry](../src/pokezero/engine_search.py):
  `EngineMctsConfig.model_priors`, `_aggregate_root_arms`, and root decision rows.
- [Existing H2H runner](../scripts/mcts_mcts_h2h.py):
  `_world_parallelism_expected_config`, immutable game writes, paired summary.
- [Prior ON/OFF tests](../tests/test_model_priors_search.py),
  [H2H tests](../tests/test_mcts_eval_head_to_head.py), and
  [durable launcher tests](../tests/test_mcts_h2h_durable_launcher.py).
- Corrected B2a bank SHA-256:
  `d4367a992dfeaad70cfd30599b24703cbf6735af49facbe4676f989092dfd5b2`;
  readout SHA-256:
  `efb8c8df5104b7efa28fec2be35d44961d7b74ba4d24122bf0f2c1dd028eebbe`.
  Root: `/shared/scott-experiment/root-policy-continuation-oracle-b2a-final-enthalf-20260830`.
