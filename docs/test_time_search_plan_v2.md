# Test-time search: working plan v2 (lean)

Status: 2026-07-18. Successor to `test_time_search_plan.md` (v1, now closed —
see its Disposition section for what completed, what was descoped, and what
remains undone). v1's ceremony — frozen plans, provenance gating,
audit preconditions, reserved-seed choreography — is retired for everyday
questions. The measurement doctrine is now: **paired seeds, both arms, the
compare harness (`scripts/compare_root_puct_vs_foulplay.py`), provenance logged
but never gating.** A 200-seed paired read costs ~40 minutes locally; shard on
cluster for larger n. The primary capstone seed bands stay reserved and
untouched in case a formally defensible run is ever wanted.

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
  because each selection depends on the preceding backup. "Deeper search"
  today means more root visits, scenarios, and longer tails — not depth.
- **Fallback rate measured (W1): 19.4% of decisions**, dominated by
  missing_sampled_world (59% of failures); first fix wave (force-switch
  screening, reserve candidates, world-sampling retries) brought live reads to
  ~15.3%. Diagnosis is complete; taxonomy work is closed as a workstream.
- **Search cost is scale-independent (W4): ~5–6.5 visits/sec at BOTH 50M and
  200M** — H4 (NN-eval-bound) is refuted. No 2s budget exists at any scale
  today; only extra-{0,24} fit a ~10s envelope (docs/w4_scale_cost_preliminary.md).
- **Root cause of both, identified in code**: branch evaluation resets the env
  and replays the recorded prefix from the battle root per visit
  (replay_branching.py — its own header calls this a paused-search-era
  placeholder). O(game-length) reconstruction per visit is the latency floor,
  and replay divergence *is* the missing_sampled_world fallback class. One
  mechanism, both symptoms.
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

### W3 — Frontier small checkpoint (v2.2 @ ~2.3M+, then the 3M final)

**Current readiness (2026-07-18):** no qualifying 2.3M+ frontier checkpoint
is available yet. The current `emeta-v2-2-lr3m-3m-belief` directory contains
a completed 1M-game run; its `3m` label denotes the learning-rate schedule,
not completed training games. Its existing Step-0 artifact targets an earlier
iteration and must not be reused as a frontier leaf. Refit only after a
checkpoint at the stated game count is available.

1. **Refit the value leaf first** (Step-0 refit on the frontier checkpoint).
   A 1M-fitted isotonic map on a 2.3M value head confounds the read — the one
   prerequisite kept from the old program.
2. Paired baseline at the W2 knee budget: raw vs search at the frontier.
   Deliverable: does a stronger prior shrink search's edge (1M vs frontier
   delta comparison)?
3. When the 3M final lands, the same two commands re-run there. Those numbers
   feed the final-checkpoint designation.

### W4 — Search cost at M (50M) and L (200M) scale

1. Per-decision search cost with M and L checkpoints on eval GPUs: forward
   latency, visits/sec, achievable extra-visits within realistic per-turn
   budgets (~2s aggressive, ~10s ladder-like). Deliverable: feasible search
   budget per scale — how much search the big models actually get.
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
   calls while retaining nested bridge timing. The CPU-thread probe selected
   eight Torch threads for the next validation wave (8.22s to 3.31s per
   decision at extra-24, same direct materialization). Batched initial root
   value leaves are merged behind an explicit opt-in and retain scalar-equivalent
   inputs; only the independent mandatory sweep batches, while adaptive revisits
   remain sequential. **Strict batched smoke (2026-07-18):** one extra-120
   mechanics game completed with `prefix_replay_count = 0`, no mechanics
   fallbacks across 35 decisions, and 124/139 (89.2%) direct world
   materializations. It measured 15.61s mean / 17.99s p95 per decision, so it
   is a crash-free contract gate rather than a throughput conclusion. The
   companion one-game W1 diagnostic saw two force-switch fallbacks in 36
   decisions; the bounded ten-game mechanics probe is the required coverage,
   fallback-rate, and residual-timing read before Tier 2 is declared validated.
   A one-round-trip retained-snapshot branch candidate is merged behind the
   belief-sampled bridge-handle guard; direct materialization remains the
   correct path but is no longer the primary throughput lever.
2. **Tier 2 — direct state construction (implemented; validation in
   progress):** `prepare_direct_materialization_prefix` and
   `LocalShowdownEnv.materialize_public_world` now start a fresh
   belief-sampled world and construct the public branch point from teams, HP,
   statuses, boosts, side conditions, and field state without replaying the
   recorded protocol prefix. Unsupported public effects fail closed rather
   than approximating simulator state. The strict W5 runner verifies that the
   direct path is exercised and that prefix replay is zero, but the fresh
   telemetry probe must still establish coverage, fallback behavior, and the
   remaining wall-clock cost before Tier 2 is treated as validated.
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
question — all get re-priced on the fast path once the batched-leaf validation
has a durable result. Success remains **more winrate at fixed wall-clock**,
not more visits.

## Order and cost

W5 (materialization redesign) is now first — W1's diagnosis and W4's cost
table are complete and both point at it. W2's value curve continues in
parallel (its data is still wanted and re-runs cheaply on the fast path); W3
proceeds once its leaf refit lands. Every strength
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
