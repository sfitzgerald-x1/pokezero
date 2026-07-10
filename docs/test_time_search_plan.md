# Test-time search: validation-first plan from root-PUCT to ladder budgets

Status: executable plan, 2026-07-10. Slots under `selfplay_mcts_roadmap.md`'s
"MCTS at inference" workstream. Every step validates a named assumption before the
next step spends effort on it; the capstone is a pre-registered strength test using
the **`emeta-v2-2-lr3m-1m-belief` 1M checkpoint (iteration-0312)** — the current
best pure self-play agent (93.2% max-damage / 41% foul-play low-fi) — as the search
value head. Later phases are funded by the capstone's outcome, not by default.

## Current state (verified in code, 2026-07-10)

Exists and works today:
- `RootPUCTSearchPolicy` (`search_policy.py`): one-ply PUCT over root actions —
  priors from the policy head, visit-based final selection, time budgets,
  override gates, fallback diagnostics (`mcts_diagnostics.py`).
- Simultaneous moves via opponent-action planners (greedy / weighted top-k
  scenario enumeration from the opponent policy's priors).
- Hidden information via `determinization.py` — belief-backed world sampling.
- Leaf evaluation: seeded rollout tails (`replay_branching`, prefix
  snapshot/restore) and/or an observation-value function hook.
- Runnable surfaces: `neural_cli root_puct` / `root_puct_counterfactual`
  (recorded-decision re-scoring + benchmarks), `foulplay_bridge` (live play),
  refutation mining (G4 consumer).

Does not exist: a multi-ply tree (no internal nodes/backup), in-tree chance
nodes, batched/served NN evaluation for search, tree reuse across moves.

## Assumptions under test

| id | assumption | validated in |
|---|---|---|
| H1 | root-only search adds measurable strength on a strong checkpoint ("MCTS is a topper") | Step 4 |
| H2 | search value concentrates at contested decisions (high policy entropy / small value margin) → adaptive budgets beat flat budgets | Steps 2, 4 |
| H3 | prior-guided search inherits systematic blind spots (near-zero prior ⇒ branch never visited at any budget) — root noise is load-bearing, not hygiene | Step 3 |
| H4 | per-simulation cost is dominated by NN eval (not sim stepping) at 10M+, and grows with model scale — the eval path, not the simulator, is the bottleneck | Step 1 |
| H5 | the sims→strength curve flattens quickly under a strong prior (few, well-aimed sims suffice) — bounding whether a fast simulator backend is ever required | Step 4 |

## Step 1 — Mechanics + cost profile (hours; no new code)

Run `neural_cli root_puct` (recorded-decision re-scoring) with the 1M checkpoint
on ~50 recorded games (its own eval games are fine).

Validates: the checkpoint loads under search (v2.2 latch), scenario planners and
belief determinization run end-to-end, and — the measured numbers this plan's
cost claims depend on — the per-move wall split into {prefix replay, per-branch
sim stepping, NN evals, rollout tails}. The scoping estimates (5–20 ms/edge,
seconds/tail) are hypotheses until this table exists.

Gate: any component >3× its estimate → update this doc's budget math before
proceeding (measure-don't-assume; twice this month the finer measurement
overturned the confident model).

## Step 2 — Prior-quality profile of the 1M net (hours)

Over a ≥2,000-decision corpus from recorded games: per-decision policy entropy,
top-1/top-2 prior mass, and value margin between the two best candidates.

Deliverable: the "contested-decision fraction" — what share of moves have
entropy > τ or value margin < δ (sweep τ, δ). This is the load factor for
adaptive budgets (H2's precondition) and the ladder wall-clock model
(budget ≈ contested-fraction × per-search cost). Also stratify by game phase:
the hypothesis says lategame/endgame decisions are the contested ones.

Gate: none (descriptive), but the number feeds Step 4's adaptive arm and the
ladder-budget arithmetic.

## Step 3 — Blind-spot entrenchment audit (half day)

From the hazard-probe state corpus (states where hazard/spin actions are
available and the ΔV work showed mispricing): measure the prior mass the 1M net
assigns to those actions, then run root-PUCT with root Dirichlet noise ON vs OFF
and count visits to the mispriced branches.

Validates H3 with numbers: if noise-OFF search never visits hazard lines the
priors bury (predicted), the caveat is confirmed — root noise is mandatory in
every search config, and search strength gains must not be read as "the blind
spots are fixed" (that remains G4/diversity's job). If noise-OFF search *does*
find them via the value head, the value repair is further along than ΔV implies —
worth knowing either way.

## Step 4 — CAPSTONE: strength test, 1M checkpoint as the value head

**Config**: `RootPUCTSearchPolicy` with priors AND leaf values from
`emeta-v2-2-lr3m-1m-belief` iteration-0312 (local convention:
`checkpoints/pz-v2-2-1m.pt`); belief-determinized worlds (K per Step 2's
uncertainty gating); root Dirichlet noise per Step 3; seeds fixed and shared
across arms.

**Arms** (each vs max-damage n=600 and foul-play @100 ms n≥600 — fp at n=300 is
±5 pts and ungateable):
1. Raw net (baseline — no search).
2. Net + search, **value-head leaves** (AlphaZero-style, no tails), flat budgets
   {8, 32, 128} sims/move.
3. Net + search, **rollout-tail leaves**, budget matched by wall-clock to arm 2's
   32-sim point (tails cost more per sim; match seconds, not sims).
4. Net + search, **adaptive budget** (search only contested decisions per
   Step 2's τ/δ; 128-sim cap) — H2's direct test.

**Metrics**: win-rate deltas ±95% CI; per-move wall (mean + p95); sims→strength
curve from arm 2 (H5); arm-2-vs-arm-3 delta (is the value head search-ready, or
do tails still carry it?); arm-4 vs arm-2 at matched wall (does adaptivity buy
the same strength cheaper?).

**Pre-registered decision rules**:
- **Fund the next phase** (batched-eval service integration + ladder pilot at
  2–5 s/move) iff search adds **≥ +3 pts max-damage or ≥ +5 pts foul-play** at
  ≤5 s/move (any arm). Both metrics move → strong go.
- **Stop** (net-alone remains the line; revisit after the next value-head
  generation) iff best arm adds <2 pts on both metrics — H1 fails; record it.
- **Redirect** iff tails (arm 3) clearly beat value-leaves (arm 2): the value
  head is not search-ready on the axes that matter → the finding feeds the
  value-repair program (reanalyze targets / refutation retraining) before more
  search infrastructure is built.
- H5 read: if 32→128 sims buys <1 pt, the strong-prior regime is confirmed —
  flat evidence **against ever funding a fast-simulator backend**; adaptive
  small budgets + batched evals are the end-state architecture.

## Gated follow-ons (not funded until the capstone reports)

1. **Batched leaf evaluation via the inference service** — the shared
   prerequisite for any deeper search at 50M+ (H4's consequence). Also speeds
   root-PUCT and refutation mining immediately.
2. **Ladder pilot**: root-PUCT + adaptive budgets + adaptive PIMC (many shallow
   worlds early, few deep late) at 2–5 s/move — inside ladder timers on
   existing infrastructure.
3. **Multi-ply tree on the native sim** (per-node snapshot/restore
   generalizing the restorable-prefix machinery) — only if the capstone shows
   search upside AND Step 1 shows edge costs make depth affordable; primary
   consumers are refutation depth and analysis, not latency-critical play.
4. **Fast-simulator rollout tails (hybrid)** — only if arm 3 wins decisively
   AND tail cost dominates the Step 1 profile. A full alternate-simulator tree
   backend is explicitly last-resort: it forks the observation encoder and the
   ground truth, and both the eval-bound trend (H4) and the strong-prior trend
   (H5) argue its advantage shrinks with every model generation.

## Standing constraints

- Search stays OUT of the collection loop (the roadmap's load-bearing lesson:
  the simulator is too slow to generate search-improved training targets at
  scale). Test-time, refutation mining, and reanalyze targets only.
- Every arm logs the search diagnostics (`mcts_diagnostics`) so failures are
  attributable; any parity-style gate failure is a finding to report, not a
  threshold to tune.
- Blind-spot honesty: a positive capstone does NOT mean the hazard axis is
  fixed — Step 3 exists to keep that claim impossible to make by accident.
