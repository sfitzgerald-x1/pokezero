# Test-time search: working plan v2 (lean)

Status: 2026-07-19. Successor to `test_time_search_plan.md` (v1, now closed —
see its Disposition section for what completed, what was descoped, and what
remains undone). v1's ceremony — frozen plans, provenance gating,
audit preconditions, reserved-seed choreography — is retired for everyday
questions. The measurement doctrine is now: **paired seeds, both arms, the
compare harness (`scripts/compare_root_puct_vs_foulplay.py`), provenance logged
but never gating.** A 200-seed paired read costs ~40 minutes locally; shard on
cluster for larger n. The primary capstone seed bands stay reserved and
untouched in case a formally defensible run is ever wanted.

## Scope boundary

This plan owns the existing Python/Showdown **V2** search path only. The V3
Rust/PokeEngine search work is owned separately and is intentionally excluded:
do not port V2 mechanics, reuse V2 artifacts as V3 evidence, or make V2 launch
decisions conditional on V3 results. V2 measurements remain useful on their
own terms, and any later cross-engine comparison must be an explicitly scoped
experiment with its own paired evaluation.

## Established results

- **Search works.** Root-PUCT-120 + frozen isotonic leaf, fallback enabled, 1M
  v2.2 checkpoint: **43.0% vs FoulPlay against the raw model's 33.0%** —
  paired **+10.0 pts** (200 seeds, McNemar b=46/c=26, p≈0.02). Search does not
  lose to its own prior.
- **Pure (fallback-disabled) search cannot run.** Opponent-action scenario
  materialization comes up illegal essentially always — the same wall as the
  hazard audit's world_unavailable (1680/1776). Today's +10 is earned on the
  minority of decisions where scenarios materialize, with the prior playing
  the rest. Known lead: the checkpoint scenario planner's requested-legal-mask
  path is a *privileged benchmark guard* (its own comment says so) — the
  deployment-honest (hidden-mask) path has no equivalent legality guard.
- **The search is one-ply**: root-PUCT over root actions with rollout tails /
  value-leaf evaluation. There is no multi-ply tree, in-tree chance node, or
  tree reuse across moves. The independent mandatory initial root-action value
  leaves can now share one NN batch; adaptive PUCT revisits remain sequential
  within each sampled world, although already-selected leaves can share a
  cross-world value batch. "Deeper search"
  today means more root visits, scenarios, and longer tails — not depth.
- **Fallback baseline (pre-W5): 19.4% of decisions**, dominated by
  missing_sampled_world (59% of failures); the first fix wave (force-switch
  screening, reserve candidates, world-sampling retries) brought live reads to
  ~15.3%. A bounded direct-materialization diagnostic later observed 1 fallback
  in 61 decisions (1.6%), but that short read is not a replacement for the W1
  baseline. Diagnosis is complete; taxonomy work is closed as a workstream.
- **Pre-W5 scale-cost baseline (W4): ~5–6.5 visits/sec at BOTH 50M and 200M**
  refuted H4 (NN-eval-bound) on the former replay-from-root path. Its conclusion
  that no 2s budget existed was valid for that implementation, but is not a
  current serving claim after W5's direct-materialization redesign. W2 must
  refresh the budget curve on the new primary path before selecting a budget
  (docs/w4_scale_cost_preliminary.md).
- **Former root cause, removed from the primary path (W5):** branch evaluation
  used to reset the env and replay the recorded prefix from the battle root per
  visit (the paused-search-era placeholder in `replay_branching.py`). Direct
  state construction now produces zero prefix replays in bounded telemetry. The
  latest strict CPU-only +120-visit read completed 136/136 direct
  materializations with zero replay and zero fallback across 34 decisions. It
  measured 5.19s mean / 5.57s p95 full-decision wall time and 97.9
  visits/root-search-second. Value evaluation (4.49s per decision, including
  3.50s of neural forwards) is now the dominant measured cost; observation
  encoding (0.96s), materialization (0.44s), and bridge round trips (0.07s) are
  secondary. This is mechanics telemetry, not a strength result.
- Value-leaf calibration (Step 0) is a repeatable job. The 1M frozen isotonic
  leaf: Pearson 0.503, sign 0.758, ECE 0.063 — with a known weak late-game
  slice (phase ECE 0.089/0.054/0.136). Timing and hazard artifacts exist; no
  new audits planned.

## Workstreams

### W1 — Fix opponent-scenario legality

The illegal-scenario wall blocks pure search and silently caps current
strength. In order:

1. **Instrument before fixing**: per-decision logging over ~50 games —
   fallback rate, and a failure taxonomy for illegal scenarios
   (perspective/mask mismatch, force-switch turns, belief–world desync, stale
   request state). The artifact must prove deployment-honest hidden opponent
   masks and fallback enabled: W1 measures the fallback rate rather than
   treating it as pure-search evidence. We have never measured what fraction
   of turns actually search; that number reframes every other result.
2. Root-cause and fix the top classes, starting from the privileged-guard
   lead above; unit-test each fixed class.
   - First public-information screen: when the protocol-visible opposing
     active is fainted, remove move hypotheses before replay and retain only
     the exchangeable replacement-switch bucket. This deliberately does not
     infer private move disablement, trapping, or party order.
3. Re-run the 200-seed paired baseline. Deliverables: fallback rate
   before/after, paired delta before/after. Hypothesis: if most turns fall
   back today, legality fixes multiply search's value.

### W2 — Throughput profile and the budget→value curve

1. Mechanics: per-decision wall-clock and visits/sec on the small model at
   extra-visits {0, 24, 120, 480, 1200}, with a stage breakdown (encode, NN
   forward, scenario materialization, rollout tails, sim stepping).
2. Value: paired winrate vs FoulPlay at each budget (~100–200 seeds/point).
   Deliverable: **winrate vs seconds-per-turn curve** and its knee — "how far
   can we reasonably search" answered in points, not visits.

**Execution status (2026-07-20):** the first two frontier W2 Jobs were
intentionally retired before completion. They predated the validated CPU-thread
and adaptive-batching W5 configuration, so their multi-hour partial rows are
not the fast-path curve and must not select W3's knee. Their persisted artifacts
remain diagnostic-only. The replacement five-point curve is now running with
the validated strict direct-materialization, initial/adaptive-batched,
adaptive-branch-reuse, eight-thread CPU profile. It uses fresh non-reserved
probe seeds, the final 3M checkpoint, and its checkpoint-matched frozen leaf;
the matching marker-backed W3 controller remains downstream of its selected
knee.

### W3 — Frontier small checkpoint (v2.2 @ 3M)

**Current readiness (2026-07-18):** `emeta-v2-2-lr3m-3m-belief` continued from
2M to 3,000,000 total games and produced its frontier checkpoint at iteration
625. It is the current final-checkpoint candidate. Its checkpoint-matched Step-0
capture/refit completed and selected an isotonic value leaf: Pearson `0.593`,
sign accuracy `0.773`, and ECE `0.143`. The ranking and sign gates pass; the ECE
miss is recorded as the plan's documented-proceed outcome rather than a clean
calibration pass. The pre-tuned matching W2 Jobs and their waiting W3
controllers were retired before producing a curve; the replacement W2 run must
use the validated W5 fast-path configuration. A marker-backed W3 controller
will submit the paired frontier read from that resulting knee. No frontier
paired-search result has been claimed yet.

1. **Refit the value leaf first** (Step-0 refit on the frontier checkpoint).
   A 1M-fitted isotonic map on a 3M value head confounds the read — the one
   prerequisite kept from the old program.
2. Paired baseline at the W2 knee budget: raw vs search at the frontier after
   the checkpoint-matched leaf passes its gate.
   Deliverable: does a stronger prior shrink search's edge (1M vs frontier
   delta comparison)?
3. If a later final checkpoint is designated, re-point the same capture,
   refit, and paired-read commands at that checkpoint rather than reusing this
   leaf. Those numbers feed final-checkpoint designation.

### W4 — Search cost at M (50M) and L (200M) scale

1. Per-decision search cost with M and L checkpoints on the validated W5
   direct-materialization, initial/adaptive-batched, adaptive-branch-reuse
   path: forward latency, visits/sec, achievable extra-visits within realistic
   per-turn budgets (~2s aggressive, ~10s ladder-like). Run the S/M mechanics
   profile CPU-first: batch-one inference is a small fraction of the current
   search wall and does not justify reserving a GPU. An L-scale profile may
   request exactly one GPU only if its CPU profile establishes that forward
   latency is material; it must not reserve a multi-GPU node. Deliverable:
   feasible search budget per scale — how much search the big models actually
   get.
2. Framing: compute allocation — strong prior with few visits vs weaker prior
   with many; the per-scale W2 curve says where search adds most. Optional,
   only if M-scale feasible budget is nontrivial: one paired probe at M with
   a refit M leaf.

The initial cost-only probe is recorded in
[`w4_scale_cost_preliminary.md`](w4_scale_cost_preliminary.md). It is deliberately
not a strength result or a final budget choice: W2 must supply the paired
strength-vs-cost curve, and W1 must reduce the visible fallback rate first.

### W5 — Replace replay-from-root materialization (owner-priority, 2026-07-17; not gated on W2)

The profile already told us what dominates: prefix replay. Supersedes further
W1 taxonomy slicing — the redesign removes the failure mechanism instead of
classifying it.

1. **Tier 1 — snapshot-per-decision:** materialize each determinized world
   once per decision (replay once if still needed), then snapshot it with the
   engine's own `State.serializeBattle`/`deserializeBattle`
   (pokemon-showdown `sim/state.ts`; `Battle#toJSON` exists for this) and
   clone the snapshot per visit and per scenario. Per-visit cost O(1);
   expected ~10× at late-game decisions.
   **Status (2026-07-17): landed and default-on** (bridge-resident snapshots,
   merges through `35395d4`). Independently verified on a local probe at
   extra-120: `prefix_replay_count = 0` (replay-from-root fully gone), world
   materialization down to ~5% of search wall, ~8s per search decision on
   CPU-only hardware vs the 19.2s GPU-cluster baseline. **Fresh strict
   mechanics read (2026-07-18):** direct materialization completed with zero
   prefix replays or fallbacks, but an extra-24 CPU-only probe still took
   ~37s per searched decision. Transformer forwards consumed ~34s per
   decision; bridge round trips (~0.1s) and world materialization (~0.4s) are
   not the limiting cost. The earlier residual interpretation is therefore
   superseded. **Instrumentation status (2026-07-18):** Root-PUCT now splits
   forward time into value leaves, root priors, opponent priors, and policy
   calls while retaining nested bridge timing. The next telemetry image also
   reports non-overlapping setup, initial-sweep, adaptive-visit,
   branch-validation, post-branch-history, direct-prefix, and
   scenario-dispatch stages, so the residual becomes only uncategorized work.
   The CPU-thread probe selected
   eight Torch threads for the next validation wave (8.22s to 3.31s per
   decision at extra-24, same direct materialization). Batched initial root
   value leaves are merged behind an explicit opt-in and retain scalar-equivalent
   inputs. Adaptive revisits remain sequential within each sampled world, while
   ready leaves may batch across worlds. **Strict batched smoke (2026-07-18):** one extra-120
   mechanics game completed with `prefix_replay_count = 0`, no mechanics
   fallbacks across 35 decisions, and 124/139 (89.2%) direct world
   materializations. It measured 15.61s mean / 17.99s p95 per decision, so it
   is a crash-free contract gate rather than a throughput conclusion. The
   companion one-game W1 diagnostic saw two force-switch fallbacks in 36
   decisions; the bounded W5 telemetry probe is the required coverage,
   fallback-rate, and residual-timing read before Tier 2 is declared validated.
   **Residual attribution (2026-07-18):** the first bounded read measured
   120/120 direct materializations and zero prefix replays, but left 9.66s
   (71%) of a 13.60s mean decision wall time in the raw residual.
   The next image partitions that residual into recorded branch-search
   internals, call time absent from recorded branch results, and outer policy
   orchestration. Optimize only the largest measured partition; do not infer a
   Tier 2 target from the blended residual.
   **Belief-group repair (2026-07-18):** that finer telemetry exposed a
   root-action-cap bug: all sampled belief worlds in the selected opponent
   action group were evaluated, but only the first was retained for root
   aggregation. The cap now applies after retaining the entire group, as the
   V2 PIMC contract requires. This is a correctness repair, not a claimed
   speedup; batching or otherwise accelerating the required retained worlds is
   the next throughput target.
   **Bounded telemetry (2026-07-18):** the CPU-only extra-120 read completed
   with `prefix_replay_count = 0`, 132/132 direct materializations in the W2
   mechanics game, and one fallback across the companion 61-decision W1
   diagnostic (1.6%, `illegal_action_for_current_request`). The W2 game itself
   had two non-search decisions out of 35; these are branch-legality outcomes,
   not replay or direct-materialization failures. This validates Tier 2's
   primary path and preserves the P-1 gate, but it is not a strength result.
   Cross-world initial-leaf batching also executed: it saved 995 neural-forward
   calls out of 16,868 value evaluations (5.9%). Adaptive revisits remain
   sequential, so that reduction alone does not settle the wall-clock problem.
   The resulting decision wall was 14.69s mean / 18.13s p95, still a bounded
   mechanics read rather than a ladder budget claim.

   **Cached-choice/single-view telemetry (2026-07-19):** the follow-up
   CPU-only extra-120 mechanics smoke completed with 124/124 direct
   materializations, zero prefix replays, zero fallbacks, and 31 searched
   decisions. Its decision wall was 11.69s mean / 14.39s p95, down from the
   preceding 14.69s / 18.13s bounded read. Snapshot-local legal
   action-to-Showdown-choice translations reduced action-choice encoding from
   the prior 2.69s per decision to 0.005s. Emitting only the searched player's
   branch observation retains full observations for rollout tails and falls
   back to normal legality validation for stale inputs. The remaining dominant
   measured stages are value evaluation (6.18s per decision) and post-branch
   result projection (3.53s); bridge round trips are 1.00s, local restore is
   0.36s, and raw residual is 0.01s per decision. This is a throughput
   diagnostic, not a strength result.

   **Adaptive cross-world value batches (validated as mechanics and telemetry):**
   each retained belief world's PUCT accumulator stays independent; at most one
   adaptive visit per world is selected, only already-selected non-terminal
   leaves batch together, and each result backs up before that world's next
   selection. The mechanism is opt-in (`--batch-adaptive-root-values`), requires
   the existing initial-leaf batch, non-time visit budgets, and zero rollout
   tails. With a batch-composition-invariant evaluator, it reproduces scalar
   per-world candidates, visit counts, and selected actions. A bounded
   extra-120 read completed with 120/120 direct
   materializations, zero prefix replays, a 6.25% fallback rate, and cross-world
   batches for all 14,400 adaptive leaves. Initial and adaptive batching reduced
   15,348 semantic value evaluations to 3,630 physical forwards. Its 10.05s
   mean / 12.35s p95 searched-decision wall is a one-game telemetry result, not
   a paired speed claim. It moved raw residual to 0.09s per decision and showed
   that the largest remaining nested branch stage is result projection (3.57s),
   so the next image splits player-view construction from the remainder before
   optimizing either.
   **Projection split (2026-07-19):** that follow-up completed with zero prefix
   replays. Player-view construction consumed 2.61s per decision while only
   0.004s remained outside it, so observation work is 99.9% of the measured
   result-projection stage. The next bounded mechanics image splits that work
   into state normalization, feature encoding, and belief-overlay projection;
   do not revisit replay or bridge transport until that split identifies the
   dominant substage.
   **State-normalization attribution (2026-07-19):** the strict CPU-only
   extra-120 mechanics smoke completed with 132/132 direct materializations,
   zero prefix replays, zero mechanics fallbacks, and 33 searched decisions.
   It measured 12.53s mean / 14.78s p95 full-decision wall time. Player-view
   projection was 4.13s per decision: state normalization was 3.15s, feature
   encoding was 0.94s, belief-overlay projection was 0.02s, and only 0.01s
   remained unattributed. This is a mechanics telemetry result, not a strength
   claim. The next image subdivides state normalization into incremental
   parser/belief sync, replay snapshotting, player-relative normalization, and
   post-normalization annotations; optimize only the dominant measured
   substage. No GPU expansion is justified by this CPU-side profile.
   **Direct fast-path selection (2026-07-20):** the subsequent strict
   CPU-only extra-120 telemetry read combined direct materialization, batched
   initial root values, batched adaptive root values, and adaptive branch reuse.
   Across 34 decisions it recorded 136/136 direct materializations, zero prefix
   replays, and zero fallbacks. Full-decision wall time was 5.19s mean / 5.57s
   p95, with 97.9 visits per root-search second. The stage ledger attributes
   4.49s per decision to policy/value evaluation (3.50s neural forward), 0.96s
   to observation encoding, 0.44s to materialization, 0.21s to simulator work,
   0.07s to bridge round trips, and 0.02s residual. This establishes the
   replacement W2 execution profile: CPU-only, eight Torch threads, strict
   direct reconstruction, both value-batching modes, and adaptive branch reuse.
   It is a bounded mechanics result, not a claim that any search budget improves
   strength.
2. **Tier 2 — direct state construction (implemented and primary-path
   validated):** `prepare_direct_materialization_prefix` and
   `LocalShowdownEnv.materialize_public_world` now start a fresh
   belief-sampled world and construct the public branch point from teams, HP,
   statuses, boosts, side conditions, and field state without replaying the
   recorded protocol prefix. Unsupported public effects fail closed rather
   than approximating simulator state. The strict W5 runner and bounded
   telemetry now establish direct-path coverage, zero prefix replay, fallback
   behavior, and remaining wall-clock cost. Branch-legality fallout remains a
   W1 concern rather than a direct-construction validation failure.
3. **Correctness constraint:** never serialize the live battle — only
   determinized worlds built from public information. The P-1 anti-leakage
   checksum gate must pass unchanged on the new path.
4. **Validation gates — right-sized (owner directive, 2026-07-17):** all
   iteration-loop validation must run in minutes, not hours. Per image:
   a 1–2 game mechanics smoke (crash-free, `prefix_replay_count = 0`,
   anti-leakage checksum) is the merge/image gate. Per fix wave: a ~10-game
   telemetry probe (a few hundred search decisions) re-measures fallback
   rate and the stage-timing breakdown — that resolution distinguishes 15%
   from low single digits and is sufficient. The 200-seed paired FoulPlay
   no-regression read runs **once**, when the redesign is declared done —
   not per iteration. No long-tailed validation batteries inside the
   redesign loop.

Deferred: tree reuse, early termination, vectorized stepping, and the multi-ply
question — all get re-priced after the adaptive-batching probe has a durable
result. Success remains **more winrate at fixed wall-clock**, not more visits.

## Order and cost

W5 (materialization redesign) is now first — W1's diagnosis and W4's cost
table are complete and both point at it. The previous W2 runs were retired
because they were not using the validated W5 fast path; W2 resumes from a fresh
configuration after the current bounded telemetry selects its next target, and
W3 follows that curve. Every strength
claim uses the paired harness, both arms on shared seeds; no strength claims
from unpaired or single-arm runs.

## Retired from the previous plan

Frozen-plan composition, provenance gates, audit-lineage preconditions,
marker-backed execution, the five-arm battery, Dirichlet secondaries. The
owner-criteria thresholds (+3 md / +5 fp, paired CI > 0, never losing to the
prior) remain the bar for any eventual *binding* claim, measured with this
harness at cluster scale (~1000+ paired games) once a final checkpoint is
designated. Late-game value-leaf ECE stays a named caveat on all value-leaf
results until a refit clears it.
