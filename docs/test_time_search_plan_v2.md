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
2. **Tier 2 — direct state construction:** build the determinized battle
   directly from public state + the belief-sampled opponent (teams, HP,
   statuses, boosts, side conditions, field) as a constructed
   `deserializeBattle` payload — no protocol replay anywhere. This eliminates
   replay divergence, i.e. most of `missing_sampled_world`.
3. **Correctness constraint:** never serialize the live battle — only
   determinized worlds built from public information. The P-1 anti-leakage
   checksum gate must pass unchanged on the new path.
4. **Validation gates:** W2 mechanics stage-breakdown re-run on the new path
   (visits/sec up ~an order of magnitude; fallback toward low single digits);
   one 200-seed paired FoulPlay read confirming no strength regression.

Deferred until this lands: batched leaf NN evaluation, tree reuse, early
termination, vectorized stepping, and the multi-ply question — all get
re-priced on the fast path. Success remains **more winrate at fixed
wall-clock**, not more visits.

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
