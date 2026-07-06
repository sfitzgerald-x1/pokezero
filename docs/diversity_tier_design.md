# Diversity tier — prior-free strategy generation

Status: design for review, updated 2026-07-06 for the G4 pivot. Companion to
[`next_train_readiness_plan.md`](next_train_readiness_plan.md) (Horizon H2),
[`wse_shaping_and_coverage_design.md`](wse_shaping_and_coverage_design.md),
[`research_synthesis.md`](research_synthesis.md).

## 2026-07-06 pivot: G2 paused, G4 replaces D2

The named-axis G2 random-objective arm plan is no longer the active primary
diversity engine. Keep the reward-plumbing, gauntlet, payoff-ledger, and
dashboard work, but do not scale one-shot action-class axis arms to a 24-arm
wave. The current read is that marginal action-class rewards create play-style
caricatures rather than conditional strategy, and stronger curated variants
already failed to move the hazard/value probes.

The active D2 path is now **G4 refutation mining**: replay games the champion
won, search the loser seat for single-turn and then short-line deviations that
flip the outcome, certify those deviations with terminal rollouts over repeated
seeds, and archive the fragile states as ground-truth credit-assignment data.
This keeps generation game-supplied and prior-free while targeting the specific
self-play failure mode: unpunished mistakes in won games receive positive credit
unless a post-mortem finds the punishment.

## The three design axioms (fixed, from the project owner)

1. **No intentional priors.** We want networks that *learn* to use Spikes — but we
   will not build any condition hand-crafted to force that strategy to emerge. The
   environment must **randomly generate novel strategies that explore the game
   universe**, and the strategies we care about must emerge from that exploration,
   not from our thumb on the scale.
2. **Intuition is for measurement, not creation.** Our strategic knowledge —
   hazard/spin counterplay, stall vs hyper-aggression, reserving a sweeper, setup
   usage, phasing — defines the *observables* we use to measure how much of the
   strategy universe the population has explored. It never enters a reward, a
   selection rule, or a curriculum.
3. **Budget: diversity generation ≤ 3× train time** relative to a single
   foundation run.

Axiom 1 is a hard line that rules out three mechanisms this project previously
considered, including one this document's first draft was built around:

| ruled out | why it violates axiom 1 |
|---|---|
| Curriculum hazard bonus (pay for laying Spikes, anneal later) | a hand-crafted condition built to force one chosen strategy to emerge — the definition of an intentional prior; also Goodhart-bait and crutch-removal instability |
| Hand-seeded specialist (build/insert a hazard-stall agent) | the "novel" strategy is authored by us, not discovered; single-point diversity that generalizes nowhere |
| Semantic z-conditioning (pseudo-reward for following a descriptor over *our chosen* axes: hazard rate, status rate…) | the descriptor basis is our strategic intuition injected as a training signal — measurement axes smuggled into creation |

What axiom 1 *permits* is any mechanism whose content is supplied by the game
itself or by chance: competitive pressure (what beats the current population),
random search over reward/objective space, and unsupervised diversity objectives
whose axes are learned, not chosen.

## Why this tier exists (the measured premise)

The ΔV trajectory study (`runs/dv-pivot-trajectory-20260705/`) on our strongest
run (`foundation-emetamon-obsv2-500k-belief`, 85.3% max-damage) showed the value
head prices hazards **backwards** and hardens that mis-valuation with training
(correct-pricing vs games r = −0.65; rapid-spin response exactly 0.0 at every
checkpoint). The mechanism: a value head is a faithful mirror of its opponent
distribution. In a self-play monoculture nobody switches into hazards, so hazards
genuinely do not pay in that meta, so no outcome-derived signal can teach them —
and PBRS shaping provably cannot change the optimum (confirmed empirically by the
mc22 micro-arms: no shaping config moved ΔV; wse-arm1 damaged calibration with no
hazard payoff). Meanwhile the observation redesign moved max-damage +8 points:
**inputs fix tactics; only the opponent distribution can fix strategy.** The
diversity tier is that lever, and the corroborating contrast is that older
checkpoints trained against more varied opponents (belief-1.5m, fpdistill) priced
hazards correctly at +9–24% of spread.

One prediction this premise makes, which the tier will test: none of the
mechanisms below mention Spikes anywhere — if hazard play emerges anyway, the
thesis (strategy lives in the opponent distribution) is confirmed the right way.

## The generator: three prior-free mechanisms, one pipeline

The design separates three roles that previous drafts conflated:
**generate** candidate strategies (prior-free), **select** which ones enter the
opponent pool (prior-free), **measure** what the population explored (intuition
lives here, read-only).

### G1 — Population play (the substrate)

Replace pure mirror self-play with collection against a **pool**: the current
policy plus frozen checkpoints spanning runs, generations, and architectures.

- Mechanics: the collector gains opponent-sampling — per game, the opponent seat
  is drawn from a pool manifest (weighted; details in *Pool mechanics*). Today
  `rollout_cli collect-selfplay-training-cache` plays `--current-policy` against
  itself; this is the one substantive infra PR of the tier (see *Infra*).
- Free diversity to start: we already possess behaviorally distinct agents —
  v1-era 512d and metamon-S (different meta, correct hazard signs), the v2/v2.2
  arms, the mc22 micro-arms, and scripted baselines. Cross-*generation* play matters
  as much as cross-arm: past selves are the cheapest non-clones we own (and
  fictitious self-play over past checkpoints is the classical cure for cycling).
- Cost: pool opponents are inference-only (no training). Collection wall-clock
  rises modestly (checkpoint loading per game; mixed-schema opponents encode per
  their own latch). Estimate ≤0.2× overhead.

G1 alone breaks the monoculture's perfect mirror, but a pool of *near-clones*
converges back to one — G4/G3 keep filling it with genuinely new members.

### G2 — Random objective search (paused; gated fallback)

This was the direct implementation of "randomly generates novel strategies":
short micro-arms trained under **randomly sampled reward perturbations**,
following the reward-randomization line (RPG — Tang et al., arXiv:2103.04564).
It is now paused as the primary novelty engine. If revived, the accepted
invariant remains: its basis must stay behavioral, not tool-specific.

- Sample `w ~ random` over a **generic behavior-class basis**: damage
  dealt/taken, switch made, boost used, heal used, and KO. The basis must stay
  **behavioral, not tool-specific**: no hazard-laying, status-clicking, or
  strategy-label component is ever rewarded directly. Hazards/status remain
  observable measurement axes and may exist as PBRS potential components in
  `ShapingConfig`, but the G2 planner keeps `terminal_mode="zero"` and does not
  use those potential terms as behavior-changing rewards. If hazard play emerges,
  it must emerge instrumentally because it helps win the game by punishing
  switching/healing, not because the reward asked for hazards.
- Train a micro-arm per sample: 128d, ~13k games — the mc22 recipe, measured at
  ~40 min sequential, ~25 min pipelined. Most arms will be garbage or degenerate
  (a `w` that rewards taking damage produces a masochist). That is fine and
  expected; RPG's insight is that random objectives push populations into basins
  ordinary self-play cannot reach, and we only keep what survives selection.
- Micro-arms are *sacrificial explorers*: their networks never ship; they exist to
  become pool opponents. 128d is deliberately small — we need behavioral novelty,
  not strength (a weak-but-weird opponent still retrains the main value head:
  games against it are still outcome-labeled).

Axiom-1 audit of G2: the only human content is the generic *behavior-class basis*
and the *bounds* (scale hygiene). No sampled direction is chosen, weighted, or
filtered by what strategy we hope appears.

Revival condition: only reconsider G2 after G4 stalls for two full cycles, and
only as persistent lineage evolution over random-feature rewards, not one-shot
named-axis arms.

### G4 — Refutation mining (active D2)

G4 mines novelty and credit assignment from the champion's own wins:

1. Sample recent games the champion won.
2. Replay to each loser decision point.
3. Search the loser seat for a deviation, starting with single-turn legal
   deviations and extending later to short lines.
4. Score branches by **terminal rollouts**, never by the champion value head.
5. Certify only if the deviation wins over repeated terminal-rollout reseeds,
   defaulting to at least 20 rollouts and a >60% flip-rate threshold; R0
   acceptance specifically requires those rows to use simulator-RNG reseeding.
6. Emit a fragile-state archive containing replay coordinates, deviation action,
   flip rate, oracle/fair mode, and search statistics.

The R0 implementation is deliberately single-turn by default, with an explicit
bounded depth ladder up to three decision rounds. Under the current
replay-from-root backend, depth >1 forces the recorded continuation action
rounds after the first loser deviation before terminal rollout; it is a
certifiable short-line artifact, not a full post-deviation tree. That is enough
to catch many unpunished mistakes, gives a reproducible archive format, and
avoids prematurely depending on multi-ply tree search while replay-from-root is
still the forking backend. Later phases feed certified states back through value
retargeting, policy distillation at mined states, surprise weighting, and
curriculum starts.

`pokezero-refutation validate` is the artifact-level R0 gate: by default it
requires at least 200 sampled champion wins, at least 10 certified fragile-state
rows, terminal-rollout evaluation, no value-head use, ≥20 certification seeds per
row, simulator-RNG reseeding for every certified row, replay coordinates, and
archive/report count consistency. It does not rerun the simulations; it verifies
that a mined report is strong enough to treat as an R0 readout. Continuation
policy RNG alone is only an exploratory/development artifact; the validator
requires an explicit `--allow-continuation-only-reseeds` waiver before such a
report can pass, and waived reports are marked `r0_acceptance_eligible: false`.
Likewise, reports validated with relaxed smoke thresholds can pass validation
shape checks but remain non-R0-eligible. The CLI returns exit `0` only for
R0-eligible reports, exit `3` for exploratory/dev passes, and exit `2` for
validation failures.

`pokezero-refutation reproduce` is the rerun gate for the "reproducible from
artifacts alone" R0 requirement. Given the source rollout-record JSONL, the
fragile-state archive, and the same frozen continuation policy specs used for
mining, it rebuilds each archived `RefutationCandidate` from replay coordinates,
reruns the stored certification seeds through the simulator-RNG-reseeded terminal
rollout evaluator, and compares terminal winners/caps/turn counts plus the
archive's flip-rate summary. It exits `0` only when every reproduced row matches
the archive and `2` on mismatch. The intended flagship R0 readout should pass
both `validate` and `reproduce`; validation proves the artifact shape and
thresholds, reproduction proves the archived examples can be regenerated.

`pokezero-refutation cycle-report` is the cross-cycle G4 readout. It aggregates
one or more `refutation-report.json` files into per-mode refutation-rate trends,
including the "is the champion becoming less refutable?" decline signal, and it
computes oracle-vs-fair gaps when both modes are present for the same cycle id.
Cycle ids are naturally sorted for trend calculations, and duplicate
cycle/mode pairs fail loudly so the headline decline signal is not sensitive to
CLI argument order. This is the public artifact for the tier metric; deployment
may decide when to run miners, but the trend math stays repository-owned and
reproducible.

`pokezero-refutation training-cache` is the first R1 primitive. It turns a
validated fragile-state archive plus the source rollout records into a separate
cache of corrected loser-perspective examples: value targets are retargeted to
the certified terminal-rollout value, and `policy-value` mode also replaces the
action target with the certified deviation. `policy-distribution-value` mode is
the R1(b) distillation path: when fragile rows carry
`search_policy_distribution`, it emits weighted policy targets for the searched
action distribution rather than a single argmax action. This cache is
deliberately separate from the main rollout cache so deployment can mix it
behind explicit flags and enforce the ≤10–20% fragile-example cap. Distribution
rows are kept or dropped as a unit when capped so the searched policy target is
never silently truncated into an argmax-like partial distribution. `value` mode is
a value-target cache only and should be consumed by PPO/value-only paths, not
behavior cloning or reward-weighted objectives, because it intentionally leaves
the recorded loser action untouched.

The cache builder can also persist optional per-example `training_weights` for
R1 surprise weighting. The first source is certification strength: rows whose
terminal-rollout flip rate clears the miner's threshold by more can be upweighted
with `--surprise-weight-scale`, capped by `--surprise-weight-max`. This is
default-off and uses the generic training-weight field so later search-prior/KL
surprise can reuse the same ingestion path without changing the cache contract.

`pokezero-neural train --refutation-cache ...` is the opt-in ingestion path: it
streams certified refutation caches as capped auxiliary examples, hard-limits the
mix to at most 20% of emitted training examples, applies the same cache
schema/mask/shaping checks as primary training data, and leaves auxiliary caches
out of the primary cache-deletion lifecycle. This is ingestion plumbing only; no
R1 A/B result has landed yet.

`pokezero-refutation curriculum` is the R1(d) primitive. Given the same source
rollout records and fragile-state archive, it emits a separate rollout-record
JSONL slice whose games start from certified fragile decision boundaries instead
of turn 0. The `--total-games × --curriculum-fraction` product determines the
epsilon-sized slice count; deployment can concatenate or otherwise mix that
slice with normal collection records while keeping the default collection path
unchanged. These records are stamped with `trajectory.metadata.refutation_curriculum`
so later reports can separate curriculum-started games from ordinary self-play.

`pokezero-refutation behavior-seeds` is the R2 bridge into population work. It
turns certified fragile-state rows into a small manifest of replay coordinates,
deviation actions, flip rates, and intended population uses such as candidate
distillation, admission-gauntlet seeding, and held-out exploiter seeding. This
artifact is metadata only: it does not run admission, does not shape rewards, and
does not authorize legacy checkpoint strength evals. The manifest includes a
source fragile-archive digest and only admits terminal-rollout, no-value-head
rows whose certification counts match their terminal results. By default it can
include both oracle and fair rows while stamping each seed's mode; consumers that
need information-fair-only behavior should pass `--mode fair`. Population/eval
consumers still use the active gauntlet policy: max-damage, foul-play rungs, and
frozen current-family v2+ checkpoint pools rather than random/simple or legacy
families.

### G3 — Exploiters (the adversarial engine)

Periodically train an **exploiter**: a fresh (or branched) agent whose collection
opponent distribution is heavily weighted toward the *current main agent* rather
than the pool. Its objective is unmodified win/loss — the game itself defines
"whatever beats the champion", which is the purest prior-free novelty signal
available (AlphaStar's main-exploiter role, minus the human-data seeding).

- Cadence: ~2–3 exploiters per main-agent cycle, ~50k games each, 256d-class.
- An exploiter that achieves >τ win-rate vs main is frozen into the pool
  (whatever it found, the main agent must now answer it). One that fails is
  discarded — also information (main is locally robust).
- Risk (local search): a gradient-trained exploiter may only find nearby answers
  (e.g., speed-tier abuse) rather than deep strategy shifts. This is why G4 and
  G3 coexist: G4 mines concrete missed punishments from the champion's own wins;
  G3 digs sharply near the current meta.

### Selection: prior-free gatekeeping into the pool

Candidates (G4-derived policies, G3 winners, new main checkpoints, and any
future gated G2 survivors) enter the pool by
**game-theoretic novelty**, never by intuition metrics:

1. Play the candidate against the current pool (cheap: few hundred games).
2. Admit if it adds a new *interaction direction* to the empirical payoff matrix —
   operationally: it beats some pool member decisively that others don't, or its
   payoff-vector against the pool is far (cosine/rank criterion) from every
   existing member's. This is PSRO's effective-diversity notion: grow the span of
   the meta-game, not the head-count.
3. Cap pool size (~16–24 seats) with eviction of dominated/redundant members
   (payoff-vector nearest-neighbor merge), keeping a small never-evict core
   (scripted baselines + one frozen agent per generation) as anchors against
   drift and forgetting.

The payoff matrix is maintained incrementally (each admission test contributes its
column), giving the league an audit trail: at any time we can state the population
dimensionality (matrix effective rank) — itself a prior-free diversity measure.

**The Goodhart invariant (axiom 2, formalized):** the intuition observables below
are *read-only*. They may never appear in a reward function, an admission rule, an
eviction rule, or a matchmaking weight. Enforced structurally: the measurement
module has no import path into the training/selection configs, and every
selection decision is reproducible from game outcomes alone. If a future change
wants to couple them, it requires a new design doc, not a config flip.

## Measurement: the strategy-coverage dashboard (axiom 2)

Built from our strategic intuition, computed per-agent and per-population from
sampled games (the stats-token machinery, behavior probes, and meta-histograms
already compute most of these):

| axis | observables |
|---|---|
| hazard cycle | Spikes laid/game, spin/clear rate, hazard chip share of total damage, hazard-sack pivots |
| tempo | damage/turn, status-move share, heal share, mean game length vs pool |
| aggression structure | setup (boost-before-first-KO rate), sweeper-reserve (healthiest-mon-held-back patterns, last-mon win share), phasing (forced-switch rate: Roar/Whirlwind usage) |
| interaction | pivot rate, Pursuit interceptions, Protect scouting rate |
| generic (intuition-free cross-checks) | pairwise policy JS-divergence on a fixed state corpus; payoff-matrix effective rank; behavior-embedding cluster count |

Population coverage = how many axes show *live spread* (population variance
materially above a single-run baseline). The dashboard extends the existing
run-dashboard with a population view; the per-agent numbers ride the existing
milestone-probe cron.

**Success gates for the tier:**

- **primary** — ΔV hazard probe on the main agent flips to correctly-signed and
  grows materially. Operational definition: `correct_pricing :=
  (value_opp_hazard_response − value_self_hazard_response) / value_spread`, with
  sign convention self<0 / opp>0 (all three are existing `hazard_probe.py`
  scalars). Pass = `correct_pricing` positive and ≥10% of spread at two
  consecutive milestones, assessed as a **monotone trend over ≥5 milestone
  checkpoints** (a Pearson r over n≈6 points is noise; trend + level, not
  correlation). Reference band: +9–24% of spread on the historical
  varied-opponent checkpoints. Conjunction requirement: the flip must co-move
  with main-agent behavioral corroboration (`spin_hazard_response` off exactly-0
  and/or pivot-rate movement) so a narrow tempo-counter cannot pass the gate.
  This gate is a *validation observable*, never a target — nothing in G1–G3 or
  selection can see it.
- behavioral: ≥3 dashboard axes show live population spread; payoff-matrix
  effective rank grows past the near-1 of a monoculture.
- robustness: a fresh held-out exploiter (trained the G3 way, never pooled) gains
  less against the main agent than at tier start — the exploitability proxy
  (VGC-Bench: low-diversity agents are ~100% exploitable).
- non-regression: main-agent matched-milestone strength (max-damage,
  foul-play, and frozen v2+ pool reads) does not fall below the current vanilla
  trajectory beyond noise. Random-legal and simple-legal are saturated plumbing
  checks, not strength-gradient gates.

Kill criterion honesty: if after two main cycles the payoff rank grows but ΔV does
not move, the thesis "diverse opponents ⇒ correct strategic valuation" is wrong in
a way worth knowing, and the tier pauses for re-design rather than scaling up.

## Budget (axiom 3)

Per main-agent cycle (500k games ≈ 313 iterations; pipelined controller ≈ 140
s/iter ≈ 12 h):

| component | cost (train-time multiples of the 1× main run) |
|---|---|
| main run (unchanged) | 1.0× |
| G1 pool-collection overhead | ≤0.2× |
| G4: refutation mining + short terminal-rollout certification | ~0.05–0.1× |
| G3: 2–3 exploiters × 50k games (256d) | ~0.3–0.45× |
| selection gauntlets + dashboard sampling (inference only) | ≤0.1× |
| **total** | **≈ 1.75–1.85× < 3× cap** |

Two funding notes: the pipelining rework (deploy PR #41, measured target ~1.7×)
applies to every component, so the *wall-clock* of the full tier lands near
1.2–1.4× of today's sequential single run; and G4/G3 work is embarrassingly
parallel across available workers, so the cap binds on aggregate compute, not
calendar time.

## Pool mechanics (the details that bite)

- **Opponent sampling**: per game, opponent ∈ {current policy (self-play share,
  ~50%), pool draw (~50%)}. Pool draws weighted mildly toward recent admissions
  and anchors; *uniform-ish, not performance-prioritized* (PSRO-style f_hard
  weighting is a tuning knob to revisit only if the pool is ignored by training —
  and any such reweighting still uses outcomes only, per the invariant).
- **Mixed schemas**: pool members span v1/v2/v2.2. Each opponent encodes through
  its own checkpoint-latched spec (machinery exists); v1-era members need the
  pinned-tag runner or re-export — if that friction is high, generation anchors
  start at v2-era only and the point is moot within one cycle.
- **Value-target semantics** are unchanged: games are still outcome-labeled
  two-player zero-sum; the value head now estimates value *under the pool
  distribution* — exactly the point. (The eval yardsticks — fixed baselines,
  foul-play, frozen pools — stay fixed, so cross-run comparability survives the
  non-stationary training distribution.)
- **Provenance**: cache metadata gains the opponent identity per game (pool
  member id + checkpoint hash) — cheap, and it makes every diversity claim
  auditable from artifacts (which agent taught the main to respect Spikes?).

## Infra deltas (small, mostly existing machinery)

1. **Collector opponent-sampling PR** (pokezero): `--opponent-pool <manifest>` on
   `collect-selfplay-training-cache`; manifest = JSON list of (checkpoint path,
   weight, schema handled by latch). The public CLI keeps the design's current-policy
   mirror share via `--opponent-pool-self-play-share` (default `0.5`) and draws the
   remaining share from pool weights. The one real code change.
2. **Pool manifest + admission harness** (deploy repo): the gauntlet runner,
   payoff-matrix ledger, eviction logic — orchestration scripts in the mold of the
   milestone-probe suite.
3. **G4 refutation miner automation**: public code emits a report plus
   fragile-state JSONL archive; deploy orchestration should run it over recent
   champion wins, persist artifacts, and gate later R1/R2 use on certified
   examples that pass both artifact validation and reproduction. Public code can
   now also build a separate refutation training cache from those certified
   examples, mix it into standalone neural training behind capped flags, and
   emit an epsilon-sized curriculum rollout slice from fragile starts; deploy
   orchestration still owns the A/B run setup. The old G2 arm automation remains
   available only as fallback plumbing under the revival condition above.
4. **Collection ε-floor**: neural iterate and standalone training-cache
   collection keep a nonzero random legal-action floor on learned policies while
   benchmark/advancement specs stay deterministic. This is a completeness
   invariant, not a diversity reward.
5. **Dashboard population view** + per-agent behavior axes on the probe cron.
6. Rides the pipelined controller (deploy PR #41) once merged.

## Rollout

- **D0 (measurement first, no training)**: build the dashboard axes + payoff-matrix
  harness; baseline today's monoculture (expect: rank ≈ 1, hazard axis dead flat).
  Explicit build items surfaced by review: a move→class map (hazard / clear /
  setup / status / heal / phaze / attack) — `behavior_probe.py` today reports raw
  move names only — and the `correct_pricing` aggregation script over milestone
  checkpoints. Implementation handles: `behavior_probe.py` emits read-only
  `move_class_usage`, and `scripts/hazard_trajectory.py` aggregates
  `hazard_probe.py` JSON into the `correct_pricing` trend gate.
  `scripts/diversity_population_dashboard.py` combines behavior-probe rows with
  diversity-pool payoff vectors into the read-only population view
  (live-spread axes, behavior-embedding cluster count, policy JS-divergence, and
  payoff effective rank). `scripts/diversity_coverage_rate.py` then compares
  explicit milestone dashboard snapshots and reports payoff-rank growth,
  live-axis count growth, behavior-cluster growth, and policy JS-divergence
  growth as deltas per 100k games. `scripts/policy_js_divergence_probe.py` records
  checkpoint action-prior distributions over a fixed corpus for the generic
  pairwise JS-divergence cross-check. Without D0 the tier cannot demonstrate its
  own effect.
- **D1**: collector pool-sampling PR; populate the pool with existing checkpoints
  + anchors; continue the current main-agent line on pool collection. First read:
  ΔV probe at +50k games vs the vanilla trajectory.
- **D2**: G4 refutation mining on. First deliverable: R0 miner + report on a
  flagship checkpoint with a fragile-state JSONL archive and ≥10 certified,
  reproducible examples; "reproducible" means `pokezero-refutation reproduce`
  reruns the archived terminal results from source records and replay
  coordinates. Watch refutation rate and archive quality before feeding examples
  back into training. The feed-back primitives are now a separate refutation
  training cache plus capped `pokezero-neural train --refutation-cache`
  ingestion, default-off certification-strength surprise weights, and explicit
  `pokezero-refutation curriculum` rollout slices from fragile starts; they are
  not yet evidence that R1 improves value calibration or strength.
- **D3**: G3 exploiters at cycle cadence; held-out-exploiter robustness read.
- **D4 (gated)**: only if D1–D3 plateau on the dashboard with ΔV unmoved —
  unsupervised skill discovery (DIAYN-class, *learned* z with no semantic axes;
  axiom-1-compatible where semantic z was not) as a stronger novelty engine.

## Risks

- **Narrow-counter mirage** (the sharpest transferred critique from the prior
  draft's review): a bounded-share opponent that uses hazards can be answered by a
  hazard-*agnostic* counter (out-tempo it) — win-gates pass while ΔV stays wrong,
  silently reconstituting the monoculture inside the augmented pool. The
  structural answer is that this tier's pool is *generative, not static*: if the
  main finds a narrow counter, that counter is now the exploitable meta, and the
  next G4/G3 admission is whatever beats *it* — the arms race keeps moving where a
  single hand-seeded specialist would freeze. The measurement answer is the
  primary-gate conjunction below (ΔV flip **and** behavioral corroboration on the
  main agent — spin/pivot movement — so a tempo-only answer cannot pass).
- **Convergence-to-clones**: pool fills with near-copies → payoff-rank admission
  is the guard; watch rank, not head-count.
- **Weak-opponent pollution**: too many weak generated candidates make training too easy
  → cap the pool share of any single class of member; non-regression gate.
- **Cycling/forgetting**: rock-paper-scissors churn instead of accumulation →
  anchors never evict; fictitious-play share of past selves.
- **Eval drift**: training distribution is now non-stationary → all yardsticks
  (baselines, foul-play, frozen Pearson pools) stay fixed; matched-milestone
  comparisons are against the vanilla v2.2 control line.
- **The thesis itself fails**: rank grows, ΔV doesn't flip → kill criterion above;
  that result would redirect the program honestly rather than quietly.
