# Test-time search: validation-first plan from root-PUCT to ladder budgets

Status: active validation program, 2026-07-10. Slots under `selfplay_mcts_roadmap.md`'s
"MCTS at inference" workstream. Every step validates a named assumption before the
next step spends effort on it. The current 1M checkpoint is used only for a
non-binding directional probe. The pre-registered full strength capstone is deferred
to a final checkpoint the owner designates later; its primary seed bands remain
untouched. Later phases are funded by that binding capstone's outcome, not by
default.

## Current state (verified in code/artifacts, 2026-07-10)

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

Step 0 is closed. A frozen isotonic calibrated copy of iteration-0312 passed
the global held-out thresholds on the schema-matched external corpus: Pearson
0.503, sign agreement 0.758, and ECE 0.063. The raw head failed calibration,
so every subsequent value-leaf result must name the calibrated copy. Phase
stratification is retained as a risk diagnostic: early/mid/late ECE was
0.089/0.054/0.136 respectively. The late-game slice is above the global target;
it is not hidden or tuned away, and capstone reporting must preserve that
limitation alongside the global authorization.

## Prerequisite implementations (small, test-gated; required before the steps that cite them)

- **P-0 Schema-matched external evaluation corpus (closed for Step 0)**: every value-readiness corpus must match the checkpoint's observation
  schema and numeric census. The historical `pool-fp-v1` and `pool-fp-v2` artifacts cannot score
  this v2.2/155-column capstone checkpoint, so they are explicitly ineligible. The controlled
  foul-play harness now normalizes turn-merged history for v2.2 and exposes
  `capture_controlled_foulplay_rollouts` for raw-policy, p1-only external-opponent capture; it
  writes each labeled terminal game immediately, excludes capped/tied outcomes, and stamps
  belief-source provenance. The completed Step 0 read used disjoint v2.2
  calibration-fit and evaluation bands of 120 labeled games each, with corpus,
  source-checkpoint, observation-census, belief-source, and seed-range provenance
  frozen in the gate artifact. A one-game v2.2 capture smoke validates plumbing
  only; it is not a gate result.
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
| H3 | prior-guided search inherits systematic blind spots (near-zero prior ⇒ no post-sweep revisit at useful budgets); root noise is a separately tested remedy, not default hygiene | Step 3 |
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
passes, a **calibrated copy** (affine/isotonic fit on the disjoint P-0
calibration band, never on capstone games) MAY be used as the leaf evaluator
and must be labeled as such in every capstone row. The completed selection is
the isotonic copy described above. If ranking fails, the
capstone is re-pointed at the best value-ready checkpoint and the plan's title
claim changes accordingly.

## Step 1 — Mechanics + capstone-equivalent cost profile

Submit the persistent timing-audit job configured to run
`neural_cli root_puct-benchmark` with raw 1M policy priors, the frozen isotonic
copy as `--value-checkpoint`, and `--root-extra-visits 24` on ~50 recorded
games. This is the arm-2 comparator used to derive arm 3's rollout-tail wall
budget. A sweep-only timing read is still useful for mechanics attribution, but
cannot size a legal+24 tail match. Its durable timing artifact and terminal
marker, not a foreground command's exit status, are the record of this probe.
Use `root-puct-play-benchmark --belief-start-overrides` for the P-1 end-to-end
determinization validation.

Validates: the checkpoint loads under search (v2.2 latch), and — the measured
numbers this plan's cost claims depend on — the per-move wall split into {prefix
replay, per-branch sim stepping, NN evals, rollout tails}. The scoping estimates
(5–20 ms/edge, seconds/tail) are hypotheses until this table exists.

Gate: any component >3× its estimate → update this doc's budget math before
proceeding (measure-don't-assume; twice this month the finer measurement
overturned the confident model).

### Recorded timing evidence

The first capstone-equivalent timing read completed with raw policy priors, the
frozen calibrated value leaf, and `legal+24` root search over 50 games / 222
decisions. Mean wall time was 1.431 seconds per decision. The observed split
reversed H4's initial expectation: branch simulator steps consumed 235.193
seconds across 6,790 steps (34.64 ms per step), while policy-value evaluation
consumed 29.030 seconds across 5,848 evaluations (4.96 ms per evaluation).
Prefix replay added 40.292 seconds total (181.50 ms per decision).

The branch-step cost is above the initial 5–20 ms scoping range but below the
3x replan tripwire, so the frozen wall-budget procedure remains valid. H4 is
therefore currently **disconfirmed**: replay/simulation, not neural evaluation,
is the dominant root-search cost at this checkpoint and configuration. This is
mechanics evidence only; it does not claim a strength gain from search.

The frozen capstone plan records the timing artifact and the selected
legal+24 statistic used to round `root_time_budget_ms`. It must reject a
sweep-only, rollout-tail, or wrong value-leaf timing artifact rather than
silently accepting a hand-entered wall budget.

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

For the capstone's one adaptive arm, aggregate the decision-normalized
`entropy_or_margin` sweep across phases and choose the threshold pair closest
to a 20% contested rate. With legal+120 on contested decisions and no extra
visits otherwise, this matches legal+24's expected post-sweep work. An exact
rate tie resolves toward the lower contested rate, then the stricter threshold
pair. The frozen plan records the selected thresholds, full decision coverage,
and observed rate and rejects hand-entered values that do not reproduce this
rule from the profile artifact.

## Step 3 — Blind-spot entrenchment audit (half day; requires P-2)

From the hazard-probe state corpus (states where hazard/spin actions are
available and the ΔV work showed mispricing), first run the deterministic legal
sweep plus extra visits 24 and 120. Mandatory visits make entrenchment **no
post-sweep revisit**, never "never visited." Report E (low-prior target rate),
R_off at 24/120 (post-sweep rescue rate), and target-directed `DeltaChoice_on`.

Then run the same budgets with explicitly labeled, audit-only Dirichlet noise.
The primary capstone remains deterministic. If R_off is about 80% or higher,
there is no noise capstone arm. If R_off is low and Dirichlet materially changes
choices toward the mispriced line, add one separately labeled `+dirichlet`
secondary arm using audit-selected parameters; those parameters are tuned only
on the audit corpus, never capstone seeds. If R_off is low without a material
choice change, do not add a noise arm and route the finding to training/G4.
Hazards remain diagnostic targets, not rewards or terminal objectives.

An audit may resolve that routing only when the same low-prior state/world pair
reaches paired deterministic/Dirichlet search at every configured budget. A corpus with legal
targets or materialized belief worlds but zero paired searched worlds is
**inconclusive**, not evidence that there is no blind spot and not a basis for
either enabling or rejecting a secondary noise arm. The artifact records only
public-safe replay-rejection categories; recapture a replayable public probe
before drawing an H3 conclusion.

## Directional probe — current 1M checkpoint, non-binding

The current **`emeta-v2-2-lr3m-1m-belief` 1M checkpoint (iteration-0312)** is
the current best pure self-play agent (93.2% max-damage / 41% foul-play low-fi),
but its lineage is already continuing. Search deltas depend on the policy prior,
so a precise multi-arm measurement here would not transfer to the final model.

This checkpoint therefore receives one reduced, separately reserved directional
probe: raw base policy versus the strongest pre-registered fixed search arm,
`value-120`, at 200 mirrored games per opponent against max-damage and FoulPlay.
The probe reports paired deltas with 95% CIs, per-move wall mean/p95, and fallback
counts. Its red flags are deliberately simple: search must not lose to its raw
prior, and the FoulPlay delta should be positive. This is a directional smoke, not
a binding go/no-go verdict; it must not consume the full capstone's primary seed
bands or motivate further arm tuning on this checkpoint.

## Step 4 — Binding capstone, deferred to the final checkpoint

**Config**: `RootPUCTSearchPolicy` with priors AND leaf values from the
owner-designated final checkpoint; belief-determinized worlds (K per Step 2's
uncertainty gating); **deterministic root priors**; and seeds fixed and shared
across arms. Its selected calibrated value copy is named on every value-leaf row
and is passed as a leaf-only checkpoint: the raw checkpoint continues to supply
policy priors, action selection, and rollout behavior. The calibrated copy records
the immutable SHA-256 of its raw parent; every value-leaf run verifies that lineage
and records both input hashes plus the applied transform.
Fixed search rows use post-sweep extra visits, never an absolute visit cap, so
the actual budget is `legal_action_count + extra_visits` on every decision. A
Dirichlet row is secondary-only and exists only when Step 3's pre-registered
routing rule selects it.

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

**Owner amendment (2026-07-10)**: the binding outcomes of this plan are the two
paired capstone deltas, not the Step-0 calibration metrics. The two failure
shapes the owner cares about: (1) any search arm LOSING to the base-policy arm
(the strongest indictment — an uncalibrated/mispriced leaf evaluator is the
expected mechanism, via exact terminals vs optimistic leaves); (2) no search
arm beating the base policy against foul-play specifically. Foul-play-corpus
ECE is an instrumental precondition, not an accuracy claim about the model
(foul-play is not human play). Consequence for gating: the Step-0 calibrated
thresholds stand as the cheap defense against failure shape (1), but a
MARGINAL calibrated miss (e.g. ECE slightly above 0.10 with ranking and sign
intact) routes to owner review and may proceed to the capstone with the miss
documented in every row — it does not auto-stop the program. A ranking failure
still re-points the capstone per Step 0.

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

## Gated follow-ons (not funded until the binding capstone reports)

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

- **Persistent evaluation control:** long-running audits and capstone evaluations
  run under persistent cluster jobs, never an interactive Codex foreground
  session. The job owns progress, writes durable terminal artifacts, and emits
  its own notification. Codex submits the job and consumes those artifacts; it
  must not continuously poll a live evaluation. Resume follow-on work only
  after a job-produced artifact/notification or an explicit user prompt.
- Search stays OUT of the collection loop (the roadmap's load-bearing lesson:
  the simulator is too slow to generate search-improved training targets at
  scale). Test-time, refutation mining, and reanalyze targets only.
- Every arm logs the search diagnostics (`mcts_diagnostics`) so failures are
  attributable; any parity-style gate failure is a finding to report, not a
  threshold to tune.
- Blind-spot honesty: a positive capstone does NOT mean the hazard axis is
  fixed — Step 3 exists to keep that claim impossible to make by accident.
