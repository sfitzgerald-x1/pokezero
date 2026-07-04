# Research synthesis — what the literature + our measurements now say

Status: position doc, 2026-07-03. Distills
[`alphastar_training_context.md`](alphastar_training_context.md),
[`no_human_data_selfplay_context.md`](no_human_data_selfplay_context.md),
[`recent_pokemon_agents_survey.md`](recent_pokemon_agents_survey.md), and
today's measurement runs (E1 full, ΔV probe, cross-pool Pearson surveys,
E0-oracle partial) into seven theses and a strategy stack. Each thesis
names its evidence; when a thesis dies, strike it here rather than
deleting it.

## Theses

**T1 — From-scratch superhuman is feasible here, and randbats is a
favorable arena for it.** Poker (Pluribus: uniform-random start, ~12.4k
CPU-core-hours) and Stratego (DeepNash: model-free, no search, 10^66
hidden deployments) reached superhuman/expert play with zero human
teaching data; bluffing and deception emerged from equilibrium-seeking.
Randbats is *more* tractable than either on the hidden-information axis:
the generator's set catalogs make the hidden space finite and enumerable,
and the public belief engine already tracks it. The AlphaGo-Zero framing
is not romantic; it is supported.

**T2 — The binding constraint is strategic coverage of the self-play
equilibrium, not aggregate value quality or architecture.** Four
architecturally distinct belief heads landed within 0.003 Pearson of ~0.53
on a shared pool (saturation); meanwhile the ΔV probe shows those same
heads at ≤4% of value spread on self-side hazards with a fully flat Spikes
policy — a feature-level blindness invisible to the aggregate metric. The
field agrees three ways: AlphaStar's air-units diagnosis (coherent
multi-step deviations cannot be found by noise), VGC-Bench's measurement
that mirror-optimized agents get *more exploitable* as opponent diversity
widens, and Metamon's result that intentionally unrealistic
coverage-oriented self-play data beat distribution-realistic data online.
Every arm we run should be judged first on whether it widens coverage.

**T3 — Vanilla AlphaZero self-play is structurally wrong for this game on
three axes, each with a proven no-human-data fix.** (a) Best-response
self-play cycles in non-transitive metas → R-NaD regularized dynamics
(KL toward a lagged copy of self) or population/PSRO methods; (b)
imperfect-info equilibria are *mixed* — deterministic argmax play leaks
information and is exploitable → training and deployment must represent
mixed strategies; (c) turns are simultaneous — sequential PUCT plays the
wrong game → matrix-game solves per node (Simultaneous-AlphaZero-style)
replace our opponent-action-scenario heuristic. None of these fixes
requires human data.

**T4 — Search is a validated amplifier and an invalid teacher.** Validated:
foul-play's MCTS lineage won the PokéAgent Gen 9 OU tournament; the MIT
thesis got its strength jump from inference-time search; our E0-oracle
partial shows root-PUCT at 12.4% vs raw 8.0% over 250 paired games
(discordants 25–14, p≈0.054 — final read pending). Invalid as teacher:
search consults the value head at leaves, and 2-round rollouts are far
shorter than delayed-strategy payoff horizons, so expert iteration with a
hazard-blind head distills the blindness. Sequencing follows: value and
coverage work precede search spend; in-loop search (periodic distillation
stages, affordable at ~0.58 s/searched decision) is gated on a head
passing the ΔV bar and on E0 showing a margin worth distilling.

**T5 — The human-data quarantine costs us nothing that matters.** Every
role human data played for AlphaStar has a self-generated substitute:
KL-to-supervised → KL-to-own-reference (R-NaD); human-derived z strategy
statistics → z descriptors enumerated from the game itself (hazard/status/
switch-rate conditioning with pseudo-rewards); human validation agents →
exploiter arms + external bots. The one principled human-data role is as
*world model and yardstick* — opponent modeling inside search (PokéChamp's
architecture demonstrates the module boundary) and evaluation material —
which teaches the policy nothing.

**T6 — Scale means finishing the recipe's volume axis, not growing the
model.** *(Revised 2026-07-03 after review: the first draft leaned on
Metamon's 200M-param results; the MIT thesis recipe refutes the capacity
half.)* The thesis is an existence proof that model capacity is not the
constraint at our target: a 3-layer MLP (hidden 256) at ~3M battles with
recipe-fidelity knobs reached rank 8 / 1693 Elo peak on the gen4 randbats
ladder. Metamon's 200M transformers belong to a different regime — offline
RL absorbing 25M mixed-quality trajectories needs capacity that
from-scratch on-policy PPO demonstrably does not. What survives from the
first draft: **game volume is unfinished business** — our arms run at
0.5–1.5M battles against a recipe defined at ~3M, the gap the config audit
itself calls dominant — so no strength verdict on the flagship line is
valid short of recipe volume. Re-attribution of today's plateau: the
3m-belief head's Pearson flatline at ~0.53 (2M→3M games) occurred *at*
thesis-scale volume with thesis-scale capacity, which points the blame at
coverage/data-distribution (T2) or the pool's noise ceiling — not at model
size. Caveat both ways: the thesis's ladder number includes inference-time
search and never measured value-head quality, so it bounds policy
strength, not value learnability.

**T7 — Measurement discipline is a first-class result.** This week's
lessons, now standing rules: (a) never gate on an unmeasured or stale
number (the ~0.12 Pearson steered a week of search engineering; the real
candidates measured 0.31–0.52); (b) never compare value heads across
architectures on their own pools (own-pool Pearson inverted the true
ranking — the strongest battler had the *top* shared-pool head); (c)
Pearson is a readiness gate, not a leaderboard — past the bar, select on
strength; (d) strength reads need ≥300 paired games, matched milestones,
and a named opponent rung (`eval_opponents.md` convention); (e) prefer
feature-specific probes (ΔV, argmax rates, panic-switch) beside any
aggregate — aggregates hid the hazard failure completely.

## Strategy stack (near → far, each gated)

1. **WS-E arms** (#487): dense shaping (status-delta term) + coverage
   curriculum, cross-arm opponent pools so the reward-diverse arms serve as
   each other's population. Gate: ΔV self-response and Spikes argmax move
   by 250k without foul-play regression. *(T2)*
2. **R-NaD-style regularization arm**: reward-shaping KL toward a lagged
   reference copy of self — anti-cycling, anti-washout, no human data.
   Gate: matched-milestone foul-play non-regression + reduced forgetting
   vs historical snapshots. *(T3a, T5)*
3. **Self-derived z conditioning**: descriptor-conditioned policy with
   pseudo-rewards, descriptors enumerated from the game (hazards, status,
   switch rate). The principled coverage mechanism if arm-level shaping
   (stack item 1) proves insufficient or too blunt. *(T2, T5)*
4. **Matrix-game root search**: replace opponent-action scenarios with a
   per-node matrix solve (regret matching); also revisit deployment-time
   action sampling for mixed play. Gate: E0 final verdict says search
   earns compute. *(T3b, T3c, T4)*
5. **Volume completion**: the flagship recipe (belief features, current
   trunk) run to its defined ~3M battles before any further architecture
   arms. Model-scale escalation is explicitly off the table unless the
   thesis regime demonstrably breaks at recipe volume. *(T6)*
6. **PBS/ReBeL endgame**: belief-state value nets + depth-limited
   re-solving over the belief engine — the bet that randbats' closed
   universe makes uniquely winnable. Entry condition: stack items 1–4
   delivering a value head that sees strategy and a search layer that
   plays the right game. *(T1, T4)*

Standing quarantine: human/foul-play-derived models appear only as eval
yardsticks, search-time opponent models, or exploiter opponents.

## Open empirical questions (each cheap to answer, none yet answered)

1. E0-oracle final verdict at 300 pairs (running) — and, if positive, the
   same read at the honest hidden-info setting.
2. Re-derived Pearson bar on an external-opponent pool (`pool-fp-v1` from
   captured foul-play evals) — does 0.30 survive the distribution change?
3. Does dense shaping move ΔV *without* degrading terminal-head
   calibration (the phase-2 multi-γ trigger)?
4. Rung calibration of the FP ladder (is FP-10 > scripted-teacher, is the
   ladder monotone?).
5. Does the fpdistill head's ranking erosion under self-play fine-tuning
   (0.505 → 0.489) reverse under R-NaD-style regularization? A direct,
   cheap test of T3a's mechanism.
