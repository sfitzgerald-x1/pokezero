# Test-time search: working plan v2 (lean)

Status: 2026-07-15. Successor to `test_time_search_plan.md` (v1, now closed —
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
  value-leaf evaluation. No multi-ply tree, no in-tree chance nodes, no
  batched NN evaluation, no tree reuse across moves. "Deeper search" today
  means more root visits, scenarios, and longer tails — not depth.
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
   request state). We have never measured what fraction of turns actually
   search; that number reframes every other result.
2. Root-cause and fix the top classes, starting from the privileged-guard
   lead above; unit-test each fixed class.
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

### W5 — Search efficiency (strictly after W2's profile)

Attack what the profile says dominates. Candidates going in, confirmed or
discarded by data: caching/reusing determinized worlds instead of per-visit
materialization, batched leaf NN evaluation, tree reuse across consecutive
turns, early termination on forced lines, vectorized sim stepping. The bigger
structural option — an actual multi-ply tree — is only on the table if W2
shows the one-ply budget curve saturating while wall-clock headroom remains.
Deliverable: measured speedup, then the W2 value curve re-run — success is
**more winrate at fixed wall-clock**, not more visits.

## Order and cost

W1 ∥ W2 first (cheap; W1's fallback rate reframes everything else). W3 after
its leaf refit; W4 anytime on eval GPUs; W5 only after W2. Every strength
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
