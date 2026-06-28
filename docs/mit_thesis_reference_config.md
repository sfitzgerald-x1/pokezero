# MIT thesis reference configuration & our current gap

Source: Jett Wang, *"Winning at Pokémon Random Battles Using Reinforcement Learning"* (MIT EECS
MEng, 2024). This is the de-risked recipe we are reproducing for Gen 3. Page references are to the
thesis PDF. This document exists so that "are we even running the recipe?" is answerable from
concrete numbers rather than memory — see [`selfplay_mcts_roadmap.md`](selfplay_mcts_roadmap.md) for
how it feeds the plan.

## The recipe shape (unambiguous, p.23)

> *"…our approach diverges from that of AlphaZero in that **MCTS is not used to train the neural
> network**. Instead, **the neural network is trained via PPO, then MCTS is used purely at inference
> time as a policy improvement operator.** This was done because simulating the environment is very
> slow… generating gameplay using MCTS would not likely lead to enough samples for a neural network
> to converge…"*

Two cleanly separated phases:

1. **Train** an actor-critic via **PPO self-play with NO MCTS in the loop.** This is the
   load-bearing phase — it is what produced a *strong net on its own* (see results below).
2. **At inference only**, wrap the trained net in MCTS as a policy-improvement operator.

Running searchless PPO self-play is therefore *the recipe*, not a shortcut. In-loop MCTS (true
AlphaZero) is explicitly out of scope for the recipe and would be a larger, off-plan bet — only
reconsider it if a fast reversible engine removes the simulator-speed constraint that drove the
thesis's choice.

## Training budget (p.24, 29) — the dominant variable

- **150M environment steps ≈ 3,000,000 self-play battles**, over **~4 days**.
- Hardware: **one NVIDIA A6000 48G GPU + 80 CPU workers** (≤1 GB RAM each).
- Self-play: both players use the latest policy and both record trajectories → **2 training
  trajectories per game**.
- Parallelism: 39 games / 78 workers, **async (not lockstep)**; once ≥half the rollout buffers are
  full → compute returns/advantages → **7 epochs** of gradient descent over the batch.
- Validation: every 20k steps, win rate over 200 games vs `SimpleHeuristicsPlayer` (a weak-but-
  smooth baseline, *not* max-damage). **Most progress came in the first 40M steps (~1 day)**,
  reaching ~80%; 150M steps reached ~85%.
- Reward: **+1** on a winning turn, **−1** on a loss, **0** otherwise (sparse/terminal).
- Markov restoration: multi-turn effect durations encoded as one-hots in the state.

## PPO hyperparameters (Table A.3, p.43)

| parameter | thesis value | notes |
|---|---|---|
| `learning_rate` | **10^−4.23 ≈ 5.9e-5**, annealed `ℓ(x)=10^−4.23/(8x+1)^1.5` | annealing had *"a massive impact"*: constant LR plateaued ~55% vs SimpleHeuristics, annealed reached ~80% (p.25) |
| `n_epochs` | **7** | gradient passes per batch |
| `gamma` | **0.9999** | discount factor |
| `gae_lambda` | **0.754** | bias/variance tradeoff |
| `clip_range` | **0.0829** | policy-ratio clip (aggressive) |
| `clip_range_vf` | **0.0184** | value-function update clip |
| `entropy_coef` | **0.0588** | exploration pressure |
| `value_coef` | **0.4375** | value-loss weight |
| `max_grad_norm` | **0.5430** | gradient clip |
| `n_steps` | **78·512 ≈ 39,936** | experience per update |
| `batch_size` | **1024** | |
| `features_dim` | 896 | per-Pokémon/battle embedding width |
| `hidden_dim` | 256 | |

Hyperparameters were tuned via Bayesian optimization on a **3v3 surrogate task** (half-length
battles) because full 6v6 training is too slow to tune directly (p.41).

## Architecture (Appendix A)

- Embedding feature extractor (species/ability/item/move via `nn.Embedding`) → 896-vector per
  Pokémon + battle = 13·896.
- 3-layer MLP, hidden 256, ReLU.
- Actor head (2-layer MLP → action distribution) + critic head (2-layer MLP → scalar value).
- **Not recurrent** (the thesis lists recurrence as future work).
- State: a single **3725-dim** vector; HP binned into 7 states (0 + 6 bins), PP binned `⌊∛x⌋`,
  multi-turn effects as duration one-hots. Action space 495 (199 move slots + 295 switch slots),
  ≤9 valid per turn, invalid logits masked to −∞.

## MCTS — inference only (§3.2, p.26–28)

- 10 s/move time budget → **1000–2000 rollouts** per move; 20 workers + an aggregator sharing
  partial trees every 10 rollouts. Prior `P` is *recomputed* per worker (cheap) rather than shared;
  **the env step, not GPU inference, is the rollout bottleneck.**
- **Opponent modeled by the trained NN policy** during search (simple, but weak vs opponents who
  play unlike the net — noted as a limitation).
- **Determinization:** at the start of each rollout, sample one possibility for all unknown opponent
  info using **Showdown's exact randbats team generator** + rejection sampling (10 attempts, then
  "force" known traits). This turns each rollout into a (near) perfect-information game.
- Tree pruned by monotone fainted-count; 2k–15k nodes live. Final action = **max visit count**
  (mixed-strategy noted as future work). Rollouts terminate at a terminal *or* a leaf node.

## Results to reproduce (Table 4.1, §4.3)

- **Net-alone is already strong:** ~80% vs SimpleHeuristics by ~40M steps (~85% at 150M); net-alone
  beats the heuristic head-to-head (.786) and random (1.0).
- **MCTS adds a topper:** MCTS+NN beats raw NN .809 head-to-head (the thesis flags this as
  *inflated* because the NN is also the opponent model), beats the heuristic .908, random .996.
- Ladder: **rank 8, 1693 Elo peak** (~1615 avg over 200 games, 79.5% GXE) on gen4 randbats.

The takeaway: **the bulk of strength lives in the PPO-trained net; MCTS is a meaningful but modest
test-time addition on top of an already-strong net.**

## Gap analysis — our current config vs the thesis

Our default PPO knobs are `TransformerTrainingConfig` defaults (`neural_policy.py`); the teacher-cut
variant flips `objective` to `ppo`. The first teacher-cut pilot ran **3 iterations × 256 games = 768
battles** and reached **0.2825 vs max-damage** — with the *default* (off-recipe) hyperparameters
below.

The `default` column is what an unconfigured `neural iterate` / teacher-cut run uses. The
`recipe-fidelity` column is what the new `--experiment-preset recipe-fidelity` path (and
`neural foundation-plan/run --recipe-fidelity`) sets — see "Recipe-fidelity preset" below.

| dimension | thesis | default (off-recipe) | recipe-fidelity preset | gap / risk |
|---|---|---|---|---|
| **training scale** | **~3,000,000 battles** | **768 battles** | **unchanged (scale is orthogonal to config)** | **~3,900× under-budget — the dominant gap.** Config fidelity does not change scale; the preset only aligns *knobs*. |
| `entropy_coef` | **0.0588** | **0.0** | **0.0588 ✅** | **was no exploration pressure at all** — now aligned |
| `n_epochs` | **7** | **1** | **7 ✅** | now aligned |
| `learning_rate` (base) | 10^−4.23 ≈ **5.9e-5** | 3e-4 | **5.9e-5 ✅** | base LR aligned |
| `learning_rate` annealing | **annealed** (massive impact) | constant | **`mit-thesis` ✅** | implemented as completed-game progress against the recipe-scale denominator: `ℓ(x)=10^−4.23/(8x+1)^1.5` |
| `gamma` | 0.9999 | 1.0 | **0.9999 ✅** | now aligned (audit uses a tight tolerance so 1.0 is never read as 0.9999) |
| `gae_lambda` | 0.754 | 0.95 | **0.754 ✅** (with `ppo_target_mode=gae`) | now aligned |
| `clip_range` | 0.0829 | 0.2 (`clip_epsilon`) | **0.0829 ✅** | now aligned |
| `clip_range_vf` | 0.0184 | (none) | **(none) ❌ (unsupported)** | our PPO value loss is an unclipped MSE — listed in `unsupported_knobs` |
| `value_coef` | 0.4375 | 0.25 (`value_loss_weight`) | **0.4375 ✅** | now aligned |
| `max_grad_norm` | 0.5430 | (none) | **0.5430 ✅** (new `--max-grad-norm` knob) | now aligned |
| `batch_size` | 1024 | 64 | **1024 ✅** | now aligned |
| collection temperature | standard (1.0) | 1.0 (arms-race uses 1.4) | **1.0 ✅** | recipe preset reverts the arms-race 1.4 to standard sampling |
| validation opponent | SimpleHeuristics (weak/smooth) | max-damage (harder/noisier) | unchanged | the thesis tracked a smoother strength signal during training; still a difference |
| architecture | embedding + 3-layer MLP, non-recurrent | entity-token transformer | unchanged | different (not necessarily worse); a variable to hold in mind |

### Recipe-fidelity preset (what is now aligned, what remains off)

`neural iterate --experiment-preset recipe-fidelity` (and the foundation wrapper's
`--recipe-fidelity` flag, usable with the `teacher-cut` variant) bundles the thesis Table A.3 knobs
that our config can express directly: `entropy_coef=0.0588`, `epochs=7`, `discount=0.9999`,
`gae_lambda=0.754` (with `ppo_target_mode=gae`), `clip_epsilon=0.0829`, `value_loss_weight=0.4375`,
`max_grad_norm=0.5430`, `learning_rate=5.9e-5`, `learning_rate_schedule=mit-thesis`,
`learning_rate_schedule_total_games=3_000_000` by default, `batch_size=1024`, plus standard
`collection_temperature=1.0`. For cheap midscale reads, the foundation wrapper can override
`learning_rate_schedule_total_games` to the read's own total game count so the annealing schedule
actually sweeps the full progress range during the read.
It reuses the arms-race self-play scaffolding (PPO+GAE, mirror self-play, latest-policy collector,
held-out Pearson value selection + calibration, max-damage yardstick). Like the arms-race preset, it
only fills options not explicitly passed on the command line, so existing commands are unchanged.

**Audit, not just a name.** Every knob is recorded in the run manifest (per-iteration
`training.config` and the run-level `invocation_config.training_config`). `neural report` prints a
`recipe_fidelity:` block, `neural foundation-run` summaries carry a `recipe_fidelity` audit, and
`recipe_fidelity_audit()` compares the *actual* resolved config against the reference table — so a
run is verifiable as recipe-fidelity rather than merely labeled that way. The audit reports
`aligned=true` only when the expressible knobs match, and **always** reports `fully_on_recipe=false`
plus an `unsupported_knobs` list.

**What remains off-recipe / scale-limited (intentionally surfaced, not hidden):**

- **Value-function clipping** (`clip_range_vf=0.0184`) — our PPO value loss is an unclipped MSE.
  Reported under `unsupported_knobs`.
- **Training scale (~3M battles)** — config fidelity is independent of scale. The preset changes
  *knobs*, not battle count; reaching recipe scale is WS-B.
- **Validation opponent / architecture** — still max-damage (not SimpleHeuristics) and an
  entity-token transformer (not the thesis MLP).

### What this means

1. **The teacher-cut 0.2825 is not a plateau or a ceiling signal.** It is ~0.026% of the recipe's
   training budget, run with materially off-recipe hyperparameters (zero entropy, single epoch, and
   pre-annealing constant LR). No conclusion about whether self-play "works" can be drawn from it.
2. **The de-risked load-bearing work is the *training* half at recipe fidelity + scale**, not
   test-time search. The thesis's net-alone was already strong; search was a topper. Aligning
   hyperparameters and reaching a meaningful fraction of ~3M battles must come before reading any
   strength conclusion.
3. **Reaching ~3M CPU battles is exactly where horizontal scaling (WS-B) becomes on-recipe and
   justified** — the thesis used 80 CPU workers for collection; our cluster is the equivalent. This
   is no longer premature optimization; it is the recipe's collection budget.
4. **Test-time MCTS (and its value-head prerequisite) is phase 2** — a topper on an already-strong
   net, with determinization driven by Showdown's randbats generator as above.
