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

Our PPO knobs are `TransformerTrainingConfig` defaults (`neural_policy.py`); the teacher-cut variant
flips `objective` to `ppo`. The first teacher-cut pilot ran **3 iterations × 256 games = 768
battles** and reached **0.2825 vs max-damage**.

| dimension | thesis | ours (current) | gap / risk |
|---|---|---|---|
| **training scale** | **~3,000,000 battles** | **768 battles** | **~3,900× under-budget — the dominant gap.** The thesis needed ~1 day / 40M steps just to reach 80% vs SimpleHeuristics. |
| `entropy_coef` | **0.0588** | **0.0** | **no exploration pressure at all** — directly relevant to escaping a local equilibrium |
| `n_epochs` | **7** | **1** | 7× less optimization per batch |
| `learning_rate` | 10^−4.23, **annealed** | **3e-4, constant** | the thesis credits annealing for 55%→80% — a first-order effect |
| `gamma` | 0.9999 | 1.0 | undiscounted vs near-1 |
| `gae_lambda` | 0.754 | 0.95 | different bias/variance point |
| `clip_range` | 0.0829 | 0.2 (`clip_epsilon`) | ours ~2.4× looser |
| `clip_range_vf` | 0.0184 | (none) | thesis clips value updates; we don't |
| `value_coef` | 0.4375 | 0.25 (`value_loss_weight`) | |
| `batch_size` | 1024 | 64 | 16× smaller |
| validation opponent | SimpleHeuristics (weak/smooth) | max-damage (harder/noisier) | the thesis tracked a smoother strength signal during training |
| architecture | embedding + 3-layer MLP, non-recurrent | entity-token transformer | different (not necessarily worse); a variable to hold in mind |

### What this means

1. **The teacher-cut 0.2825 is not a plateau or a ceiling signal.** It is ~0.026% of the recipe's
   training budget, run with materially off-recipe hyperparameters (zero entropy, single epoch, no
   LR annealing). No conclusion about whether self-play "works" can be drawn from it.
2. **The de-risked load-bearing work is the *training* half at recipe fidelity + scale**, not
   test-time search. The thesis's net-alone was already strong; search was a topper. Aligning
   hyperparameters and reaching a meaningful fraction of ~3M battles must come before reading any
   strength conclusion.
3. **Reaching ~3M CPU battles is exactly where horizontal scaling (WS-B) becomes on-recipe and
   justified** — the thesis used 80 CPU workers for collection; our cluster is the equivalent. This
   is no longer premature optimization; it is the recipe's collection budget.
4. **Test-time MCTS (and its value-head prerequisite) is phase 2** — a topper on an already-strong
   net, with determinization driven by Showdown's randbats generator as above.
