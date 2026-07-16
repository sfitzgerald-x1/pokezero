# Test-time search: working plan v2 (lean)

<<<<<<< Updated upstream
Status: active validation program, 2026-07-10. Slots under `selfplay_mcts_roadmap.md`'s
"MCTS at inference" workstream. Every step validates a named assumption before the
next step spends effort on it. The current 1M checkpoint is used only for a
non-binding directional probe. The pre-registered full strength capstone is deferred
to a final checkpoint the owner designates later; its primary seed bands remain
untouched. Later phases are funded by that binding capstone's outcome, not by
default.
=======
Status: 2026-07-15. Replaces the staged validation program (Steps 0–4 and the
capstone apparatus). That program's ceremony — frozen plans, provenance gating,
audit preconditions, reserved-seed choreography — is retired for everyday
questions. The measurement doctrine is now: **paired seeds, both arms, the
compare harness (`scripts/compare_root_puct_vs_foulplay.py`), provenance logged
but never gating.** A 200-seed paired read costs ~40 minutes locally; shard on
cluster for larger n. The primary capstone seed bands stay reserved and
untouched in case a formally defensible run is ever wanted.
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
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
=======
### W4 — Search cost at M (50M) and L (200M) scale

1. Per-decision search cost with M and L checkpoints on eval GPUs: forward
   latency, visits/sec, achievable extra-visits within realistic per-turn
   budgets (~2s aggressive, ~10s ladder-like). Deliverable: feasible search
   budget per scale — how much search the big models actually get.
2. Framing: compute allocation — strong prior with few visits vs weaker prior
   with many; the per-scale W2 curve says where search adds most. Optional,
   only if M-scale feasible budget is nontrivial: one paired probe at M with
   a refit M leaf.
>>>>>>> Stashed changes

### W5 — Search efficiency (strictly after W2's profile)

Attack what the profile says dominates. Candidates going in, confirmed or
discarded by data: caching/reusing determinized worlds instead of per-visit
materialization, batched leaf NN evaluation, tree reuse across consecutive
turns, early termination on forced lines, vectorized sim stepping. The bigger
structural option — an actual multi-ply tree — is only on the table if W2
shows the one-ply budget curve saturating while wall-clock headroom remains.
Deliverable: measured speedup, then the W2 value curve re-run — success is
**more winrate at fixed wall-clock**, not more visits.

<<<<<<< Updated upstream
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
=======
## Order and cost
>>>>>>> Stashed changes

W1 ∥ W2 first (cheap; W1's fallback rate reframes everything else). W3 after
its leaf refit; W4 anytime on eval GPUs; W5 only after W2. Every strength
claim uses the paired harness, both arms on shared seeds; no strength claims
from unpaired or single-arm runs.

## Retired from the previous plan

<<<<<<< Updated upstream
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
`value-120` means raw 1M policy priors plus the frozen isotonic 1M value copy as
the leaf evaluator, deterministic root priors, the belief-determinized world
configuration frozen by the matching audits, and the mandatory legal-action sweep
plus 120 extra root visits. The controller records search diagnostics, wall time,
and ordinary/privileged fallback counts rather than silently blending fallback
games into the result. The probe reports paired deltas with 95% CIs and per-move
wall mean/p95. Its red flags are deliberately simple: search must not lose to its
raw prior, and the FoulPlay delta should be positive. This is a directional smoke,
not a binding go/no-go verdict; it must not consume the full capstone's primary
seed bands or motivate further arm tuning on this checkpoint.

## Step 4 — Binding capstone, deferred to the final checkpoint

**Config**: `RootPUCTSearchPolicy` with priors AND leaf values from the
owner-designated final checkpoint; belief-determinized worlds (K per Step 2's
uncertainty gating); **deterministic root priors**; and seeds fixed and shared
across arms. Its selected calibrated value copy is named on every value-leaf row
and is passed as a leaf-only checkpoint: the raw checkpoint continues to supply
policy priors, action selection, and rollout behavior. The calibrated copy records
the immutable SHA-256 of its raw parent; every value-leaf run verifies that lineage
and records both input hashes plus the applied transform.

**Final-checkpoint freeze and prerequisites**: before any binding capstone seed is
staged, the owner records the final checkpoint identity, immutable raw hash, and
selection rationale in the capstone decision record. That record is the checkpoint
selection rule for this one pre-registered measurement and cannot be changed after
primary seed access begins. The final checkpoint must then run its own frozen Step
0 calibration/readiness evaluation and Step 1-3 timing, prior/profile, and hazard
audits on disjoint non-capstone data. Those artifacts must all bind the final
checkpoint and its calibrated copy; the current 1M checkpoint's artifacts support
only its directional smoke and cannot be reused as final-capstone evidence.
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
=======
Frozen-plan composition, provenance gates, audit-lineage preconditions,
marker-backed execution, the five-arm battery, Dirichlet secondaries. The
owner-criteria thresholds (+3 md / +5 fp, paired CI > 0, never losing to the
prior) remain the bar for any eventual *binding* claim, measured with this
harness at cluster scale (~1000+ paired games) once a final checkpoint is
designated. Late-game value-leaf ECE stays a named caveat on all value-leaf
results until a refit clears it.
>>>>>>> Stashed changes
