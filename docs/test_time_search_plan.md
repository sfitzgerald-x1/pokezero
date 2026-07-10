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
**Still not wired (review findings, 2026-07-10, verified in code)**: the
checkpoint scenario planner's requested-legal-mask path is a **privileged**
benchmark guard (its own comment says so). P-3's per-decision visit-budget hook
is implemented; its strength benefit remains unmeasured.

## Prerequisite implementations (small, test-gated; required before the steps that cite them)

- **P-0 Schema-matched external evaluation corpus (capture path implemented; frozen pool pending,
  required by Step 0)**: every value-readiness corpus must match the checkpoint's observation
  schema and numeric census. The historical `pool-fp-v1` and `pool-fp-v2` artifacts cannot score
  this v2.2/155-column capstone checkpoint, so they are explicitly ineligible. The controlled
  foul-play harness now normalizes turn-merged history for v2.2 and exposes
  `capture_controlled_foulplay_rollouts` for raw-policy, p1-only external-opponent capture; it
  writes each labeled terminal game immediately, excludes capped/tied outcomes, and stamps
  belief-source provenance. Before Step 0, freeze
  a v2.2 capture seed band plus a disjoint calibration-fit band, record both hashes and the capture
  checkpoint/config, and re-derive the E1 Pearson floor on that compatible corpus. A one-game
  v2.2 capture smoke validates plumbing only; it is not a gate result.
- **P-1 Belief-world wiring (implemented, required by Steps 2–4)**:
  `root-puct-play-benchmark --belief-start-overrides` wires the public Gen 3
  belief planner into replay search and explicitly enables the candidate-set
  source for the benchmark environment. `--belief-world-sample-cap` implements
  the pre-registered mapping **K = min(cap, public full-team combinations)**:
  surviving revealed variants plus the distinct-species unrevealed backline
  worlds the materializer can draw, respecting public switch/move constraints.
  Uncertainty bits and the resolved K are diagnostics, not policy uncertainty.
  **Anti-leakage gate (tested)**: the sampling profile is a function of public
  belief only; matched fixture contexts with different opponent-private request
  data have identical checksums/K and sampled worlds. Benchmark JSON logs the
  distinct public-belief checksum(s) for each game seed and refuses a
  belief-enabled result if any searched seed materialized no world.
- **P-2 Root Dirichlet noise (implemented; required by Step 3)**: alpha/mix/seed
  semantics are explicit and per-decision seeded for reproducibility; diagnostics
  record the legal-action noise draw and mixed priors. **Audit-only by default**:
  primary evaluation arms run deterministic priors; `--root-dirichlet-alpha`
  creates a separately labeled `+dirichlet` row. Noise never silently enters a
  strength row.
- **P-3 Per-decision budget hook (implemented; required by Step 4 arm 4)**:
  after the mandatory legal-action sweep, `EntropyMarginVisitBudgetSelector`
  can add visits only when normalized policy entropy and/or the initial top-two
  leaf-value margin crosses configured thresholds. The CLI records and labels
  adaptive-budget rows separately; fixed-budget behavior remains the default.

## Assumptions under test

| id | assumption | validated in |
|---|---|---|
| H0 | the chosen checkpoint's VALUE HEAD is a valid leaf evaluator (held-out ranking + calibration), independent of its policy strength | Step 0 |
| H1 | root-only search adds measurable strength on a strong checkpoint ("MCTS is a topper") | Step 4 |
| H2 | search value concentrates at contested decisions (high policy entropy / small value margin) → adaptive budgets beat flat budgets | Steps 2, 4 |
| H3 | prior-guided search inherits systematic blind spots (near-zero prior ⇒ branch never visited at any budget) — root noise is load-bearing, not hygiene | Step 3 |
| H4 | per-simulation cost is dominated by NN eval (not sim stepping) at 10M+, and grows with model scale — the eval path, not the simulator, is the bottleneck | Step 1 |
| H5 | the sims→strength curve flattens quickly under a strong prior (few, well-aimed sims suffice) — bounding whether a fast simulator backend is ever required | Step 4 |

## Step 0 — Value-head readiness gate for the capstone checkpoint (H0)

The 1M checkpoint was chosen for policy strength; nothing yet certifies its
VALUE head as a leaf evaluator — and project precedent (the E1 value-readiness
line) treats that as the prerequisite it is. On the frozen P-0 v2.2
external-opponent corpus: held-out value **ranking** (Pearson vs realized
outcomes) and **calibration** (ECE + sign agreement) for iteration-0312
specifically, with checkpoint/data provenance recorded. Historical v1/v2
encoded pools are invalid for this v2.2 checkpoint.
Pre-registered thresholds: Pearson ≥ the E1 floor re-derived on the P-0 pool;
sign agreement ≥ 0.75; ECE ≤ 0.10 (raw) — if raw calibration fails but ranking
passes, a **calibrated copy** (temperature/isotonic fit on the disjoint P-0
calibration band, never on capstone games) MAY be used as the leaf evaluator
and must be labeled as such in every capstone row. If ranking fails, the
capstone is re-pointed at the best value-ready checkpoint and the plan's title
claim changes accordingly.

## Step 1 — Mechanics + cost profile (hours; no new code)

Run `neural_cli root_puct` (recorded-decision re-scoring) with the 1M checkpoint
on ~50 recorded games (its own eval games are fine). This is a cost/profile
probe; use `root-puct-play-benchmark --belief-start-overrides` for the P-1
end-to-end determinization validation.

Validates: the checkpoint loads under search (v2.2 latch), and — the measured
numbers this plan's cost claims depend on — the per-move wall split into {prefix
replay, per-branch sim stepping, NN evals, rollout tails}. The scoping estimates
(5–20 ms/edge, seconds/tail) are hypotheses until this table exists.

Gate: any component >3× its estimate → update this doc's budget math before
proceeding (measure-don't-assume; twice this month the finer measurement
overturned the confident model).

## Step 2 — Prior-quality profile of the 1M net (hours)

Over a ≥2,000-decision corpus from recorded games: per-decision policy entropy,
top-1/top-2 prior mass, and value margin between the two best candidates.

Deliverable: the "contested-decision fraction" — what share of moves have
entropy > τ or value margin < δ (sweep τ, δ). **Also measured (P-1's input):
belief-candidate uncertainty per decision** (candidate-set entropy over
unrevealed opponent slots) — policy uncertainty gates the SIM budget; belief
uncertainty gates K, and the two are distinct populations by hypothesis. This is the load factor for
adaptive budgets (H2's precondition) and the ladder wall-clock model
(budget ≈ contested-fraction × per-search cost). Also stratify by game phase:
the hypothesis says lategame/endgame decisions are the contested ones.

Gate: none (descriptive), but the number feeds Step 4's adaptive arm and the
ladder-budget arithmetic.

## Step 3 — Blind-spot entrenchment audit (half day; requires P-2)

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

**Honest hidden-information mode (pre-registered, primary)**: no real opponent
action, no requested-opponent legal mask, no privileged fallback — opponent
legality is inferred from public state / belief only. Games where the engine
falls back to a privileged path are LOGGED and excluded from primary rows
(reported as a stratified "privileged-fallback" column, never blended).

**Arms** (each vs max-damage n=600 and foul-play @100 ms n≥600 — fp at n=300 is
±5 pts and ungateable). Budgets are defined RELATIVE to the mandatory initial
sweep — `puct_branch_search` must visit every legal root action once, and a
healthy gen3 position exposes up to 9, so absolute low budgets silently fall
back to raw play. Budgets = **legal_actions + {0, 24, 120}** extra visits, and
every comparison row requires a **zero fallback rate** to count:
1. Raw net (baseline — no search).
2. Net + search, **value-head leaves** (AlphaZero-style, no tails), budgets
   legal+{0, 24, 120}.
3. Net + search, **rollout-tail leaves**, budget matched by wall-clock to arm 2's
   legal+24 point (tails cost more per sim; match seconds, not sims).
4. Net + search, **adaptive budget** (requires P-3; search only contested
   decisions per Step 2's τ/δ; legal+120 cap) — H2's direct test.

**Statistical protocol (paired, pre-registered)**: shared seed set across all
arms; **mirrored seats** (every seed played from both seats); two disjoint seed
bands (order effects / band agreement reported); paired analysis on per-seed
outcome deltas (paired bootstrap CIs; McNemar-style check on flip counts);
ties/round-capped games counted as 0.5 and reported separately. Decision
criteria bind on the CI, not the point estimate: **go requires the 95% CI lower
bound of the delta > 0 AND the point estimate ≥ the threshold** (+3 md / +5 fp).

**Metrics**: paired win-rate deltas ±95% CI; per-move wall (mean + p95);
sims→strength curve from arm 2 (H5); arm-2-vs-arm-3 delta (is the value head
search-ready, or do tails still carry it?); arm-4 vs arm-2 at matched wall
(does adaptivity buy the same strength cheaper?); fallback and
privileged-fallback rates per row (must be zero in primary rows).

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
- H5 read: the legal+24 → legal+120 delta, WITH its CI — conclude "flat"
  only if the CI upper bound of that delta is < +2 pts (an unbounded "<1 pt
  point estimate" claim is not a conclusion). Flat ⇒ the strong-prior regime is
  confirmed — evidence **against ever funding a fast-simulator backend**;
  adaptive small budgets + batched evals are the end-state architecture.

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
