# Diversity tier — manufacturing strategic diversity so the value head can price strategy

Status: design, 2026-07-05. For review; not an implementation contract. Companion to
[`wse_shaping_and_coverage_design.md`](wse_shaping_and_coverage_design.md) (#487),
[`research_synthesis.md`](research_synthesis.md) (T1–T7),
[`alphastar_training_context.md`](alphastar_training_context.md),
[`no_human_data_selfplay_context.md`](no_human_data_selfplay_context.md), and
[`next_train_readiness_plan.md`](next_train_readiness_plan.md) (Horizon H1/H2/H4).
Motivated by the 2026-07-05 ΔV-hazard-probe trajectory across the best-yet 500k run
(`runs/dv-pivot-trajectory-20260705/`).

This doc extends those; it does not restate them. Read #487 first — the diversity tier is the
*third* thing #487 stops short of. #487 designed two arms (dense shaping; hazard-coverage
curriculum) that widen *input signal* and *data coverage*. This doc argues the remaining failure
is neither: it is a property of the **opponent distribution**, and the only lever that moves it is a
population.

## 0. The premise (state it precisely)

The ΔV hazard probe (`scripts/hazard_probe.py`; primitives shared with the shaping ranker via
`pokezero.checkpoint_factors.with_self_spikes` / `with_opp_spikes`) was run as a *trajectory* across
milestones of the best foundation run to date — `foundation-emetamon-obsv2-500k-belief`, 85.3% max-
damage at peak, the strongest net we have measured. The read is not "hazard-blind" (the #487 credit-
assignment story). It is worse and more specific:

1. **Backwards sign, negligible magnitude.** The value head prices *opponent-side* hazards with the
   wrong sign and at ~0.4% of `value_spread` — i.e. it treats Spikes on the opponent's side as
   roughly neutral-to-slightly-bad for us, when the correct sign is positive (opp-side hazards help
   us). The self-side response is likewise near-zero. This is not "can't see hazards"; it is "has
   priced them, and priced them wrong."
2. **Training hardens the error.** Across checkpoints, the correlation between *correct-pricing* and
   games-trained is **−0.65**: more training makes the mispricing *worse*, monotonically. The
   `spin_hazard_response` (dP(Rapid Spin) as self-side Spikes go 0→3) is **exactly 0.0 at every
   checkpoint** — the policy never learns to clear hazards because there are none worth clearing.
3. **The mechanism is an equilibrium, not a defect.** The value head is a *faithful mirror of its
   own opponent distribution*. In a from-scratch self-play monoculture, nobody switches often enough
   for Spikes chip to accumulate, so hazards **genuinely do not pay in this meta** — and the head
   correctly-for-this-meta prices them at ~zero/negative. No amount of input redesign or additional
   training fixes this, because the *only* training signal is game outcome, and in the equilibrium
   the population found, the outcome does not reward hazards. This is a chicken-and-egg: you would
   only set Spikes if opponents switched into them; opponents only switch into them if Spikes are
   set. Neither side of the loop is ever the first mover under mirror self-play.

**The contrast that isolates the lever.** The observation redesign (v2 / v2.2) delivered **+8 pts**
max-damage. That fixed *tactics* — it made the net a better in-battle player by giving it better
features. Strategy is **orthogonal to features**: strategic value does not live in the input tensor,
it lives in the **opponent distribution the net is scored against**. You cannot feature-engineer a
payoff that the meta does not contain. This is the same conclusion #487 reached from a weaker read
(the ecology caveat: "low Spikes value is partially *correct* internally"), the same conclusion
`self_play_convergence_findings.md` reached from weight-space drift ("escaping the basin needs a
different signal … not more self-play or more capacity"), and the same conclusion T2 states as the
program thesis. **The diversity tier is the only lever for strategic valuation.** Dense shaping and
better observations are necessary and done; they are not sufficient and never will be.

### 0.1 The corroborating evidence — why older, *weaker* checkpoints priced hazards *right*

The probe's own historical table (#487, "Evidence this design responds to") is the tell. Two
checkpoints that are **weaker battlers** than the 500k flagship priced opponent-side hazards
**correctly and materially**:

| checkpoint | opp-Spikes ΔV (% of spread) | lineage |
|---|---|---|
| belief-1.5m | **+10.9%** | from-scratch belief self-play, but earlier / less converged |
| fpdistill-1.5m | **+24.1%** | BC distillation of foul-play (an MCTS search bot) + RL |
| foundation-emetamon-obsv2-500k-belief (this run) | **~0.4%, wrong sign** | strongest battler; deepest into the self-play basin |

This inverts the naive expectation (stronger net ⇒ better value head) and is *predicted* by the
mirror thesis. Read against `self_play_convergence_findings.md`:

- **fpdistill priced hazards best because its opponent distribution was widest.** fpdistill is a
  behavior-clone of foul-play — a search bot whose play includes hazard-respecting, switch-heavy
  lines. Early in its life its self-play opponent (and its own policy) still carried foul-play's
  varied, hazard-aware distribution; in *that* distribution, hazards paid, so the value head learned
  the correct sign. `self_play_convergence_findings.md` Finding 1 then shows fpdistill **froze by
  ~1M** (policy KL 1M→1.5M = 0.0075, ~60× less than clone→1M): it stopped best-responding and its
  correctly-signed hazard valuation was *preserved* rather than eroded, because it never finished
  collapsing into the pure-mirror monoculture.
- **The 500k flagship prices hazards worst because it is the deepest into the searchless self-play
  fixed point** — from scratch, no foul-play scaffolding, a monoculture that converged hard on
  hyper-offense. The `−0.65` correct-pricing-vs-games correlation *is the collapse into the basin,
  measured on the hazard axis* — the same collapse `self_play_convergence_findings.md` measured on
  the weight-drift / policy-KL axis.

**Investigation task (chase in `runs/` where the artifacts survive):** re-run the ΔV probe on the
fpdistill and belief-1.5m milestone checkpoints *if they are still captured* (per
`model_versioning.md`, runs prune all but the last ~8 iterations — the milestone copies in
`checkpoints/curated/` and `/shared` are the only survivors), and confirm (a) the opp-Spikes sign
was correct *throughout* fpdistill's life, and (b) it correlates with a *wider* self-play opponent
distribution (measure move-class / switch-rate spread of each checkpoint's self-play games via
`scripts/behavior_probe.py`'s `move_usage` + `pivot_rate`). If that holds, it is direct causal
evidence that **opponent-distribution width, not net strength, sets hazard pricing** — which is the
entire premise of this tier. This is cheap and should run before the tier launches; a *negative*
result (fpdistill priced hazards right for some other reason) would force a redesign.

## 1. The objective: "make hazards pay," as a measurable gate

The diversity tier succeeds **iff the ΔV hazard probe flips**. Concretely, on the tier's main
agent, measured by `scripts/hazard_probe.py` on both `pool-self-v1` and `pool-fp-v1` (per
`next_train_readiness_plan.md` WS-3):

- **Primary (sign + magnitude):** `value_opp_hazard_response` **> 0** (correct sign) **and ≥ 10% of
  `value_spread`** — the #487 bar, and the level the older belief-1.5m checkpoint already cleared
  (+10.9%). Stretch: match fpdistill's +24.1%.
- **Self-side corroborant:** `value_self_hazard_response` **< 0** and ≥ 5% of spread (self-side
  hazards should read as *bad* for us). Today it is near-zero.
- **Policy follow-through (not just value):** `spin_hazard_response` **> 0** (the policy raises
  P(Rapid Spin) when its own side is stacked) and Spikes `argmax_rate` **> 0** on the coverage arm —
  the value head seeing the payoff is necessary; the policy acting on it is the point.
- **Monotone, not transient:** the flip must *hold or grow* with continued training — the failure
  we are fixing is that training *hardens* the mispricing (−0.65). The tier's success signature is
  that same correlation going **positive**. This is the single most important read: a tier that
  makes hazards pay briefly and then re-collapses has not worked.

Why the opp-side response is the headline and not the self-side: the self-side ΔV also requires the
credit-assignment machinery #487 arm 1 (dense shaping) is separately building, so it confounds two
levers. The opp-side response is a purer read of "does the value head believe hazards change who
wins" — and it is exactly the number that is backwards today.

**Non-goal (stated to prevent reward-hacking of the gate):** the objective is *not* "maximize Spikes
usage." A monoculture that spams Spikes because a bonus paid it to (see §2c) also fails the program —
it is a different monoculture. The gate is the *value head pricing hazards correctly in equilibrium*,
which is only meaningful if the equilibrium still contains diverse, strong play. Hence §4's diversity
and strength guards are gates, not nice-to-haves.

## 2. Three composing levers

Each lever is designed, critiqued, then composed. None uses human data as a policy teacher (T5,
`no_human_data_selfplay_context.md`); the constraint is honored per-lever below and audited in §4.4.

### 2a. Specialist-in-pool — a hazard/stall agent that punishes the monoculture

**Mechanism.** Seed the collector's opponent pool with an agent built around hazard-setting and
stall, so that the main agent *loses games* when it ignores hazards, forcing it to respect them. In
opponent-distribution terms: inject a mode where hazards pay, so the value head sees states where
opp-side Spikes precede a loss.

**How it is created without human data — three sources, in preference order:**

1. **Scripted hazard-stall bot (bootstrap, available today).** `policy.py::ScriptedTeacherPolicy`
   already has exactly the branches needed: `spikes_available` (scores `62.0`, "hazard pressure"),
   `rapid_spin_clear_hazards` (`min(76.0, 58.0 + 10·hazard_count)`, i.e. scales with self-side
   layers), `rapid_spin_blocked_by_ghost`,
   `spikes_maxed`, plus `recovery`/`status_pressure`/`switch`. A thin subclass that *up-weights* the
   hazard/stall/recovery branches (a `hazard-stall` variant, biasing toward setting Spikes early and
   switching to force chip) is a **scripted heuristic bot, not human data** — it is admissible under
   the same clause that admits `simple-legal`/`max-damage` (`eval_opponents.md`: scripted policies
   are floor-to-mid opponents, never teachers). It plays *against* the main agent; it never labels
   the main agent's actions, never contributes to the main agent's gradient except through win/loss
   outcomes. This is the cleanest bootstrap and needs no new training.
2. **A frozen curriculum-bonus checkpoint (§2c), promoted into the pool.** Once §2c produces a
   hazard-aware net, freeze it and add its `neural:<path>` spec to the pool. This is strictly better
   than the scripted bot as a specialist (it is a *strong* hazard player, not a heuristic), and it is
   self-generated. This is the intended steady-state specialist; the scripted bot is training wheels.
3. **An exploiter-lite arm (AlphaStar main-exploiter analog).** A learner trained *only* to beat the
   current main agent (§3 roles), which will discover whatever the main agent is blind to — and the
   main agent is blind to hazards, so an exploiter will find hazard/stall lines *if the reward and
   pool let it*. Gated on levers 1–2 first showing ΔV movement (per the AlphaStar transfer map's
   "exploiter-lite only if cross-arm pools move ΔV").

**Critique / failure modes.**

- **The June result (scripted-teacher in pool was worse for strength) is real and must be
  respected.** `selfplay_mcts_roadmap.md` records that adding `scripted-teacher` to the pool *hurt*
  max-damage (139/600 vs 159/600), and #487 arm 2 already caveats this. The resolution is the same
  as #487's: **strength is read at the end; ΔV and behavior are read along the way**, and the
  specialist share is *bounded* (25%, §3) so it pressures without dominating the learned
  equilibrium. But this design goes further: the specialist is a **means to seed the pool with
  hazard-aware *frozen nets* (source 2)**, then can be *removed*, exactly like §2c's bonus is
  annealed. The scripted bot's job is to break the chicken-and-egg *once*; the sustained pressure
  comes from frozen self-generated hazard players, which do not carry the scripted bot's strength
  tax.
- **Over-fitting to the specialist.** If 25% of games are vs one scripted stall bot, the main agent
  may learn a narrow counter to *that bot* rather than general hazard respect (this is the VGC-Bench
  single-opponent-overfit pathology, `recent_pokemon_agents_survey.md`). Mitigation: the specialist
  is *one member* of a pool that also contains historical self-snapshots and (later) z-modes; PSRO
  `f_hard` (§3) then samples it in proportion to how much it still beats the main agent, so its
  weight *decays as the main agent learns to handle it* — which is the correct dynamic.

### 2b. z-conditioning — self-derived strategy descriptor, no human source

**Mechanism (the AlphaStar z, human source removed).** Condition the policy and value on a strategy
descriptor `z`, and pay a pseudo-reward for *following* `z`, so that distinct `z` values induce
distinct strategic modes *within one net*. Crucially: when some `z`-modes set hazards, the *other*
modes are trained against them (via self-play over the z-population), so switching-heavy play gets
punished by the hazard modes → **hazards pay inside the population the single net generates.** This
is the no-N-learner way to manufacture the opponent distribution the value head needs (the AlphaStar
transfer map calls it "the principled version of 'diverse versions of current strength'").

**The z space (self-derived — this is the whole no-human-data argument).** The v2.2 observation spec
*already computes* the behavioral tendency stats this needs:
`transitions.py::extract_tendency_stats` produces (count, opportunity) pairs, and
`OpponentMonTendency` tracks `switched_out_before_attacking` / `stayed_and_attacked` /
`turns_active`; these are encoded via `showdown.py::_encode_mon_tendency`. So the descriptor
vocabulary is *the game's own measured behavior*, not human replays. Propose a **3-D z** over the
same axes the probes already watch:

- `z_hazard` ∈ {low, high} — target hazard-setting rate (Spikes / side-condition moves per
  opportunity).
- `z_status` ∈ {low, high} — target status-infliction rate.
- `z_switch` ∈ {low, high} — target voluntary-switch (pivot) rate, the `pivot_rate`
  `behavior_probe.py` already computes.

Start with a small discrete grid (2³ = 8 modes; collapse to the 3–4 that matter after the first read
— a hazard-high/switch-high "stall" mode and a hazard-low/switch-low "hyper-offense" mode are the two
poles that must exist). z is sampled per-game for the acting seat; **z is zeroed some fraction of the
time (AlphaStar used 10% in SL) so an unconditional mode exists** and the deployed policy is not
forced to condition.

**The pseudo-reward.** Per AlphaStar, a *separate* value head and loss per pseudo-reward, active with
probability ~25% per episode. Reward = negative distance between the sampled z and the *executed*
behavior descriptor for that game (e.g. |target_hazard_rate − realized_hazard_rate|), computed
post-game from the same `extract_tendency_stats` machinery. This is a **behavior-matching** reward,
not an outcome reward: it pays the policy to *act out* the assigned strategy, independent of whether
that strategy won — which is exactly what forces a hazard mode to *exist* even before hazards pay.
Once the hazard mode exists and plays against the non-hazard modes, the *outcome* reward on the main
head does the rest (hazards start paying → the main value head learns the correct sign).

**How it stays no-human-data.** Every input to z and to the pseudo-reward is measured from
pokezero's own self-play games via existing extractors. There is no replay corpus, no BC target, no
KL-to-human. z is sampled either uniformly over the grid (for coverage) or from the agent's *own*
league history's descriptor distribution (`no_human_data_selfplay_context.md`'s self-derived-z
recipe). This is the T5 substitution ("human-derived z → z descriptors enumerated from the game
itself") implemented against machinery that already exists.

**Critique / cost.**

- **The net has no z input today.** `neural_policy.py::TransformerPolicyConfig` /
  `TransformerSoftmaxPolicy` take the observation stream and emit policy/value/opponent-action heads;
  there is **no conditioning channel**. Adding one is the real build cost here: a small z-embedding
  fused into the trunk (concat to the pooled representation, or a learned bias token), plus N extra
  value heads for the pseudo-rewards, plus checkpoint-compat guards (z-zeroed load path for old
  checkpoints). This is a `neural_policy.py` change of similar weight to the multi-γ head #487 defers
  to phase 2 — non-trivial but bounded, and it is the *principled* coverage mechanism, so it earns
  the build if levers 2a/2c prove the diversity direction moves ΔV.
- **Reward-hacking of the descriptor.** The policy can satisfy `z_hazard=high` by spamming Spikes
  into already-maxed sides (the `spikes_maxed` state) without strategic intent. Mitigation: define
  the hazard-rate descriptor over *productive* opportunities (Spikes-legal-and-not-maxed), i.e. reuse
  `hazard_probe.py`'s `spikes_layer_sensitivity` notion of "stops when stacked"; and cap the
  pseudo-reward weight so descriptor-matching never dominates the outcome objective.
- **Mode collapse across z.** If the trunk ignores z, all modes converge (the descriptor reward is
  cheap to ignore if the outcome reward is uniform across modes). The §4 diversity gate
  (move-class divergence *between* z-modes) is precisely the instrument that catches this; if modes
  don't separate, z-conditioning has failed and the tier falls back to 2a+2c only.

### 2c. Curriculum bonus — the non-potential-based hazard reward that *actually moves the optimum*

**Why PBRS could not teach hazards, and this is different.** #487 arm 1 is *potential-based reward
shaping* (PBRS): `f = γ·Φ(s') − Φ(s)` over hp/faint/status differentials (`pokezero.shaping`, the
reward-candidate ladder in #487). PBRS has a celebrated property: it is **policy-invariant** — it
changes the value *function* (adds Φ) but provably leaves the optimal *policy* unchanged (Ng, Harada
& Russell 1999). That is a feature for credit assignment (it can make the value head *see* hp chip
faster without biasing what the agent does) and a fatal **bug** for teaching hazards: *if the
underlying meta does not reward hazards, the PBRS-optimal policy still does not set them.* PBRS can
make the value head price the hp-chip *consequences* of hazards slightly better once hazards are
already being set — but it **cannot make the agent set them in the first place**, because it cannot
change the optimum. This is why #487 arm 1's ΔV read is "self-side hazard credit assignment" (a
value-visibility fix), and why #487 *explicitly declines* to add a hazard shaping term to arm 1
("rewarding Spikes placement directly would hand-craft the answer the probe is supposed to detect").

The diversity tier makes the opposite, deliberate choice for a **bootstrap-only** lever: a
**non-potential-based** hazard reward that is *not* a telescoping difference and therefore **does
change the optimum**. Pay the agent directly for (a) setting Spikes on the opponent's side, and/or
(b) hazard chip damage actually dealt on opponent switch-ins. Because this is not of the form
`γ·Φ(s') − Φ(s)`, it is **not policy-invariant**: it shifts the optimal policy toward hazard-setting.
That is the entire point — we *want* to bias behavior here, precisely because the meta's own reward
is stuck at the wrong equilibrium and we need a first mover to break the chicken-and-egg.

**Role: a one-time bootstrap, then removed.** The bonus is *not* a standing reward (that would be
reward-hacking by construction). It is used to **seed a hazard-aware population**: train a net (or a
few) under the bonus until it reliably sets and exploits hazards, freeze it, and **retire the bonus**
(anneal to zero on a schedule, or hard-cut and continue from the frozen checkpoint under pure outcome
reward). The frozen hazard-aware net becomes §2a's specialist (source 2) and/or a z-mode seed. After
the bonus is gone, hazard-awareness is sustained **not by the bonus** but by the *pool*: the frozen
hazard player is now in the opponent distribution, so hazards keep paying on the main agent's outcome
signal even with zero shaping. This is the composition that matters (§2.4).

**Critique / the reward-hacking risk, addressed head-on.**

- **It teaches Spikes-spam.** Yes — under a naive "+r per Spikes move" the agent will set Spikes into
  maxed sides, set them when switching would win faster, etc. Three mitigations, all necessary:
  1. **Reward the *consequence*, not the *action*, where possible.** Prefer "hazard chip damage dealt
     to opponent on switch-in" over "Spikes move played." Chip-dealt is only earned when the opponent
     actually eats hazards, which requires the opponent to switch — so the bonus is *self-limiting*
     against a static opponent and only pays when hazards are genuinely doing work. (This is the same
     philosophy as #487's "hazards become visible through their consequences.")
  2. **Cap and clip** into `[-1, 1]` alongside the terminal return (the existing shaping clip), so
     the bonus can *tilt* the optimum toward hazards without *swamping* the win/loss signal — the
     agent must still win.
  3. **Anneal on a schedule tied to the ΔV read, not a fixed step count.** Retire the bonus once the
     probe shows the *outcome* value head has picked up the correct sign (opp-ΔV > 0) — i.e. once the
     pool can sustain hazard-awareness without the crutch. If ΔV re-collapses after annealing, the
     pool composition (§3) is too thin, not the bonus schedule.
- **Selection protocol.** The bonus is a reward candidate; it must earn its way through the #487
  Tier A–E ladder (`pokezero.shaping` oracle-fit → supervised ranker → micro-RL → 100k confirm),
  with **one crucial difference in the pass criterion**: unlike PBRS candidates, the non-PBRS bonus
  is *expected* to change the policy (Spikes argmax should *rise*), so the ranker's "corrected
  Pearson within 10% of control" gate is relaxed for this candidate — we are explicitly buying a
  policy change. Its kill criterion is instead behavioral: if it produces Spikes-spam *without* the
  opp-side ΔV sign flipping (spam that doesn't teach the value head anything), it is a dead
  bootstrap.

### 2.4. Recommended composition

The three levers are not alternatives; they are a **sequence that hands off**. The chicken-and-egg is
broken once and then sustained:

> **curriculum-bonus (§2c) forces the first hazard-aware net into existence → that net, frozen,
> becomes the specialist in the pool (§2a source 2) → the pool now contains a mode where hazards pay,
> so the main agent's *outcome* value head learns the correct sign with the bonus removed →
> z-conditioning (§2b) generalizes single-specialist diversity into a *continuum* of strategic modes
> so the main agent respects hazards robustly (not just vs one bot) → PSRO/`f_hard` matchmaking (§3)
> keeps the whole population non-transitive and sustains the diversity as strength climbs.**

Each arrow is a first-mover breaking a dependency the previous stage could not break alone:

- The **bonus** is the only lever that can move the *optimum* (2a's scripted bot pressures but is
  weak; 2b's descriptor reward makes a mode *exist* but that mode has no reason to be *good* at
  hazards without the bonus). The bonus is the ignition.
- The **specialist-in-pool** is the only lever that makes the flip *durable without ongoing shaping*
  — it converts the bonus's one-time policy change into a standing feature of the opponent
  distribution. It is the flywheel.
- **z-conditioning** is the only lever that makes the diversity *cheap and continuous* (one net, many
  modes) rather than a growing zoo of frozen specialists, and prevents single-opponent overfit. It is
  the generalizer.
- **PSRO/`f_hard`** is the only lever that keeps it from re-collapsing as the main agent strengthens
  (beat-everyone, not beat-on-average). It is the ratchet.

You could stop after bonus+specialist and likely pass the §1 gate (belief-1.5m passed it with *no*
diversity machinery, just a wider early distribution). z-conditioning and full PSRO are the
generalization/robustness layer, gated on the bootstrap working. That gating is the phased rollout,
§5.

## 3. Pool / league mechanics

The core new machinery is **opponent sampling in collection**. Grounding it in the actual code:

**What exists (do not rebuild).** The training loop *already* has an opponent pool.
`opponents.py::opponent_pool_policy_specs` builds a pool from `fixed_policy_specs` (named or
`neural:`/`linear:` checkpoint specs) plus `historical_opponent_policy_specs` (frozen self-snapshots
drawn from the promotion registry, `promotion.py::PromotionRegistry.selection_checkpoint_policy_specs`),
capped at `--max-historical-opponents` (default 3) with `selection_mode ∈ {recent, spread}`.
`collection.py::policy_from_spec` loads any of these into a live policy. `neural_cli iterate` /
`rollout_cli collect-selfplay-training-cache` already accept `--current-policy` and repeated
`--opponent-policy` flags. So *heterogeneous pools of scripted bots + frozen nets already work* —
adding the §2a specialist is `fixed_policy_specs += ("hazard-stall",)` or
`+= ("neural:<frozen-hazard-ckpt>",)`.

**The one seam that is missing — per-game weighted matchmaking.** Today, opponent assignment per game
is **`selfplay.py:833`: `opponent_spec = opponent_specs[game_index % len(opponent_specs)]`** — a
*deterministic round-robin*. Every pool member gets exactly `games / |pool|` games, regardless of how
hard it is to beat. This is the core thing the diversity tier changes. The sketch:

```
# selfplay.py, replacing the modulo cycle in _run_selfplay_game_record
opponent_spec = matchmaker.sample(game_index, rng)   # weighted, not modulo
```

where `matchmaker` implements a policy over the pool:

- **Uniform** (baseline / ablation): the current round-robin, made stochastic. Weight ∝ 1.
- **PSRO `f_hard`** (recommended): weight opponent `B` ∝ `f_hard(P[main beats B]) = (1 − p_B)^c`,
  where `p_B` is the main agent's estimated win rate vs `B`. Zero weight on already-beaten opponents;
  a smooth **max-min** (learn to beat *everyone*) rather than max-average — the AlphaStar mechanism
  that lets a rare-but-strong exploit (the hazard specialist, early) actually enter the learning
  signal instead of being averaged into noise. `p_B` comes for free from the benchmark harness we
  already run (`collection.benchmark_rollouts` / the per-opponent win rates in `evaluation.py`); the
  matchmaker reads the latest per-opponent win-rate estimates and re-weights each iteration.
- **`f_var`** (for exploiter/struggling arms): weight ∝ `p_B·(1 − p_B)` (opponents near own level),
  per AlphaStar's curriculum for exploiters when win prob is low.

This is a bounded change: a new `Matchmaker` abstraction consumed at the single call site
(`selfplay.py:833`), plus a win-rate source (already computed), plus manifest plumbing to record the
sampling weights (mirror the existing `opponent_pool_config_dict` in `run_manifest.py`). It does
**not** touch the collection throughput path, the env pool, or the PPO contract.

**Roles (AlphaStar league, budget-scaled).** Full AlphaStar was 12 concurrent learners + ~900 frozen
players — prohibitive (`alphastar_training_context.md` transfer map: "full: prohibitive"). The
budget-scaled roles:

- **Main agent** (1, never reset): the flagship. Sampling = 35% mirror/self + ~50% `f_hard` over the
  full frozen pool + ~15% vs the hazard specialists (bounded, per §2a). Snapshots into the frozen
  pool every ~200k games. This is the net the §1 gate is read on.
- **Frozen pool** (the "players"): every promoted main snapshot (already captured by the promotion
  registry) + the §2c frozen hazard net(s) + the §2a scripted bot(s). Refresh: add on promotion;
  cap total with `spread` selection so the pool stays diverse across training time, not clustered at
  recent-and-similar checkpoints (the `_spread_policy_specs` machinery already does this).
- **Exploiter-lite** (0–1, reset on success): trains `f_var` *only against the current main*; added
  to the pool on beating the main >70% or timeout, then reset. Purpose: find the main's blind spots
  (hazards, early). Gated — only spun up if bonus+specialist show ΔV movement but plateau (§2.4's
  "generalization layer").
- **League-exploiter** (deferred): finds *systemic* blind spots nobody in the pool beats. Out of
  scope for the first tier; noted as the H2 escalation.

**How the two live control checkpoints + future arms populate it.** The two live control checkpoints
(the 512d-vanilla and 512d-shaping/coverage arms from `next_train_readiness_plan.md` WS-4) are
*already* promotion-registry snapshots — they enter the frozen pool for free as `neural:` specs the
moment they promote. Future arms (R-NaD, multi-γ) likewise contribute snapshots. The pool is the
union of every arm's milestones plus the manufactured specialists — which is exactly the "cross-arm
opponent pools" #487 and the research synthesis strategy-stack item 1 call for, now with (a)
`f_hard` weighting instead of uniform and (b) *manufactured* hazard modes rather than only
whatever-the-arms-happened-to-learn.

**The multi-schema constraint — a real dependency, flagged.** `model_versioning.md` ("Deferred until
there's a concrete need") explicitly defers *backward-compatible multi-schema inference* — loading
old + new checkpoints in one process for "a cross-version league / model zoo" — as "heavy, ongoing
maintenance." **The diversity tier is that concrete need.** Every frozen pool member must be loadable
alongside the current-schema main agent *in the same collection process*. Consequences the design
must accept:

- **First tier stays within one observation schema.** Populate the pool only from checkpoints that
  share the main agent's schema (the v2.2 wave onward). The pre-v2 belief-1.5m / fpdistill
  checkpoints are **not** poolable as live opponents without the deferred multi-schema loader — they
  are usable as *probe targets* (§0.1) and *eval yardsticks* (they run in their own process), but not
  as in-loop opponents. This is a hard constraint, not a preference.
- **If cross-schema pooling is later wanted**, the `model_versioning.md` multi-schema loader is a
  prerequisite build, and its cost should be weighed against just retraining a specialist under the
  current schema (usually cheaper — a scripted bot or a fresh bonus-bootstrap is schema-current by
  construction). Recommendation: **do not build the multi-schema loader for the first tier**; keep
  the pool schema-homogeneous and manufacture specialists in-schema.

## 4. Measurement & gates

The tier is instrumented before launch (the WS-3 discipline: reads wired into the milestone cron,
both pools, every ~100k games). Four gate families:

### 4.1 Primary — the ΔV hazard flip (§1)

`scripts/hazard_probe.py` on the main agent, both pools, every milestone. Pass = opp-ΔV > 0 and ≥10%
of spread, *and* the correct-pricing-vs-games correlation is **positive** (the inverse of today's
−0.65). This is the headline; everything else guards it.

### 4.2 Behavioral diversity — does the population actually contain distinct strategies?

A flipped ΔV is meaningless if the "diversity" is cosmetic. Instruments:

- **Meta-histograms** (`next_train_readiness_plan.md` WS-3 already schedules these): move-class usage
  distribution over the pool's games. The hazard/setup/switch classes must have *non-trivial mass*,
  not a delta at hyper-offense.
- **Move-class divergence *between* z-modes** (new read, the z-conditioning acceptance test):
  KL / total-variation between the move-class distributions of `z_hazard=high` vs `z_hazard=low`
  self-play games (built from `behavior_probe.py::move_usage`). If z-modes are indistinguishable, 2b
  has failed (trunk ignores z) — the tier falls back to 2a+2c. Target: the hazard-high mode sets
  Spikes at ≥N× the rate of the hazard-low mode.
- **`pivot_rate` spread** across the pool (`behavior_probe.py`): a real stall mode should show high
  voluntary-switch rate; the hyper-offense mode low. A pool with uniform pivot rate is a monoculture
  wearing costumes.

### 4.3 Exploitability — does league training reduce it?

The VGC-Bench finding (`recent_pokemon_agents_survey.md`): a single-team / mirror-optimized agent is
**~100% exploitable** and gets *more* exploitable as opponent diversity widens; generalization-
oriented (population) training trades peak strength for robustness. Read: train a fresh best-response
exploiter against the frozen tier checkpoint and measure its win rate — the exploiter's ceiling
should be **lower** for a league-trained main than for a mirror-only main at matched strength.
(This is the exploiter-lite arm doing double duty as a metric.) A league that does not reduce
exploitability is not buying what the program wants.

### 4.4 Guards (kill criteria)

- **Strength guard (the diversity-must-not-tank-strength gate).** Matched-milestone max-damage /
  foul-play vs the historical control curves (the WS-4 controls). Diversity that costs >3 points
  foul-play at matched milestone (the #487 / 4L-lesson band) without an ΔV flip by 250k is killed —
  same kill criterion as #487 arm 2, applied to the whole tier. The June scripted-teacher regression
  is the specific risk this guards.
- **No-human-data audit.** A mechanical check that no pool member, z source, or pseudo-reward derives
  from human/foul-play data *as a policy input to the main agent's gradient*. Allowed: foul-play as
  an eval-only yardstick (separate process), a human world-model as a *search-time opponent prior*
  (H4, never in training collection), fpdistill checkpoints as *probe targets*. Forbidden: fpdistill/
  human checkpoints as in-loop training opponents (they *are* admissible as exploiter opponents per
  the quarantine — but note that even then they shape the main agent only through outcomes, and the
  first tier does not need them; keep them out to keep the audit trivially green). The audit is a
  manifest lint, not a judgment call.
- **Descriptor-hacking guard** (§2b/2c): Spikes usage must correlate with *productive* hazard states
  (not `spikes_maxed` spam) — reuse `spikes_layer_sensitivity`. Spam without ΔV flip = kill.

## 5. Infra dependency

**A league needs many concurrent agents in the collection process.** Two cost axes:

- **Throughput.** Each collection game now loads a *sampled* opponent policy (a frozen net forward
  pass) instead of a mirror copy. With `f_hard` weighting the pool is small (main + ~3 historical +
  1–2 specialists ≈ 5–6 live policies), but the collector must hold them all resident and the
  benchmark harness must estimate per-opponent win rates each iteration. The **pipelining throughput
  win (deploy PR #41, ~1.7×)** is what makes this affordable: the diversity tier should **ride the
  pipelined controller**, so the extra per-game opponent-forward cost and the per-iteration win-rate
  estimation overlap with training rather than serializing. Without pipelining, N-agent collection
  serialized against the train step would erase the throughput headroom the recipe-scale budget
  needs (`selfplay_mcts_roadmap.md`: ~200 games/s is the current floor; league sampling must not drop
  below the ~46 g/s recipe rate).
- **Compute of N-agent leagues.** The budget-scaled roles (§3) are deliberately *not* AlphaStar's 12
  learners: **one main + one gated exploiter** are the only *learners*; everything else is a *frozen*
  forward pass (cheap). This keeps the tier at ~1–2× a single-arm's training compute, not 12×. The
  frozen-pool memory is bounded by `--max-historical-opponents` + specialists (~6 nets × ~1.5 MB =
  negligible; `model_versioning.md`). The exploiter, when active, doubles the train step — hence it
  is gated, not standing. **Recommendation:** first tier runs with **zero standing exploiter** (main
  + frozen pool + matchmaker only); add the exploiter-lite arm only when the phased gate (below)
  says the bootstrap worked but generalization plateaued.

## 6. Phased rollout and gates

Phases hand off on gates, not dates (the Horizon convention). Watchdogs (strength guard, game-length
drift, no-human-data audit) live from phase 0.

- **Phase 0 — matchmaker + probe wiring (no new learning).** Land the `Matchmaker` seam at
  `selfplay.py:833` (uniform-stochastic ≡ today's behavior; `f_hard` behind a flag), the per-opponent
  win-rate source, and the z-mode / exploitability reads into the milestone cron. Ship the scripted
  `hazard-stall` variant (§2a source 1). *Gate: round-robin ≡ stochastic-uniform reproduces the
  current arm's curve (no regression from the plumbing); probes fire on both pools.* Also: run the
  §0.1 investigation (probe fpdistill/belief-1.5m milestones) to confirm the premise before spending
  training compute.
- **Phase 1 — the bootstrap (the chicken-and-egg break).** Curriculum-bonus arm (§2c) through the
  #487 Tier A–E ladder; freeze the resulting hazard-aware net; add it to the pool as the specialist;
  run the main agent with `f_hard` over {historical + specialist}, bonus **removed** on the main.
  *Gate (the make-or-break read): main-agent opp-ΔV flips to > 0 and ≥10% of spread by 250k, and the
  flip **holds** with the bonus gone (correct-pricing-vs-games correlation ≥ 0), with no >3-pt
  foul-play regression at matched milestone.* If this passes, the tier has done its core job. If it
  fails — ΔV does not flip, or flips then re-collapses — the pool is too thin: escalate specialist
  share or add a second frozen hazard net *before* touching z.
- **Phase 2 — generalize (z-conditioning).** Build the z channel in `neural_policy.py` (embedding +
  per-pseudo-reward value heads + compat guard); train the main with z-conditioning + descriptor
  pseudo-rewards over the 3-D grid, on top of the phase-1 pool. *Gate: move-class divergence between
  z-modes clears the §4.2 bar (modes are distinct), opp-ΔV holds ≥10% robustly across `z`, and the
  unconditional (z-zeroed) mode is strength-neutral vs the phase-1 main.* This buys robustness /
  anti-overfit, not the initial flip.
- **Phase 3 — ratchet (PSRO-proper + exploiter-lite).** Spin up the exploiter-lite arm (`f_var` vs
  current main), promote its finds into the pool, run full `f_hard` PSRO. *Gate: exploitability
  (§4.3) drops vs the phase-2 main at matched strength; strength non-regressing.* This is the
  standing steady state and the entry to H2's PSRO-proper.

## 7. Horizon connections

The diversity tier is the precondition for two Horizon tracks (`next_train_readiness_plan.md`):

- **H1 — search revival.** Search is *paused* today, and specifically **gated on a value head that
  prices hazards**: `mcts_design.md` records that a miscalibrated / mis-signed leaf value makes 1-ply
  search **no better than (or worse than) the net** — a mis-signed value head makes search actively
  *worse* on exactly the hazard lines where it would help, because 1-ply search consults the value
  head at leaves and inherits its backwards hazard sign (`alphastar_training_context.md`: "expert
  iteration would distill the blindness, faithfully"). The E0-oracle margin is already thin (+4.0
  pts, p≈0.056). **Fixing hazard valuation is the precondition for search earning its keep** — the
  #487/roadmap trigger "a head passes the ΔV ≥10%-of-spread bar" is *this tier's primary gate*. When
  the tier's main agent flips ΔV, re-run E0-oracle on it (H1's first trigger); a correctly-signed
  value head is what makes matrix-game root search (T3c) worth building.
- **H4 — opponent world-model (the one sanctioned human-data use).** Once the population contains a
  hazard/stall mode, the *spin-block branch* (set hazards → opponent must switch or spin → hazards
  chip the switch-in) becomes a live strategic line the agent plays — but modeling *what a human/
  foul-play opponent would do* at that branch (switch vs spin vs stay) is a search-time opponent
  prior, and that is the H4 world-model's job. The diversity tier *opens the branch that H4 then
  models*: today the branch is shut because self-play blindness never explores it. H4 trains a "what
  would they click" model from the PokéChamp corpus **as a world model inside search only, never as
  policy teacher** — admissible precisely because it teaches the policy nothing (T5,
  `no_human_data_selfplay_context.md`). The tier unlocks H4 by making the strategic states H4 prices
  actually occur in play.

## 8. Open questions (each cheap, none answered)

1. **§0.1 causal check:** did fpdistill price hazards right *throughout* its life, and does the sign
   track its self-play opponent-distribution width? (Run the probe + `behavior_probe.py` on surviving
   milestones before launch. A negative result forces a redesign.)
2. Does the non-PBRS bonus's policy change *survive annealing*, or does the main re-collapse the
   moment the crutch is gone — i.e. is a single frozen specialist enough, or is a *continuum* (z)
   required for durability? (Phase 1 vs Phase 2 gate answers this.)
3. What specialist **share** (§2a bound) maximizes ΔV movement per point of strength cost? The June
   result says 100% scripted-teacher is too much and hurts; #487 proposes 25%. Sweep 10–30% on the
   phase-1 arm.
4. Does `f_hard` matchmaking alone (Phase 0 + specialist, no bonus) move ΔV, or is the optimum-
   changing bonus strictly required to break the egg? (A cheap ablation: Phase 0 with the scripted
   specialist at `f_hard` weight, no §2c bonus — if ΔV moves, the bonus is optional and the tier
   simplifies.)
5. Is one observation schema enough for a useful pool, or does the deferred multi-schema loader
   (`model_versioning.md`) become worth building to admit the older correctly-priced checkpoints as
   live opponents? (Default answer: no — manufacture in-schema; revisit only if in-schema specialists
   prove insufficient.)
