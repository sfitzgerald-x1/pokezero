# Superhuman self-play without human data in imperfect-info games — reference notes

Status: reference notes, 2026-07-03. Companion to
[`alphastar_training_context.md`](alphastar_training_context.md), written
after the project direction was fixed: **AlphaGo-Zero-style from-scratch
superhuman play; no human data as a teaching source**. Human models are
admissible only as (a) evaluation yardsticks, (b) opponent models inside
search/simulation, (c) opponents for exploiter-style robustness training —
never as policy teachers. The AlphaStar doc's human-anchor recommendations
should be reread through that lens; this doc gives the no-human-data
replacements.

## Is from-scratch superhuman feasible in imperfect info? Yes — three times over

The strongest evidence for the purist position comes from games *harder* on
the hidden-information axis than randbats:

1. **Poker (CFR line — Libratus, Pluribus).** Zero human data. Pluribus
   starts from *uniform random* and reaches superhuman 6-max no-limit
   hold'em via Monte-Carlo CFR self-play (~12,400 CPU-core-hours — tiny) +
   depth-limited real-time search. Bluffing is not taught; it *emerges*
   from equilibrium computation, because protecting information is part of
   the equilibrium in any imperfect-info game
   ([Libratus, Science 2017](https://noambrown.com/papers/17-Science-Superhuman.pdf);
   [Pluribus, Science 2019](https://www.science.org/doi/10.1126/science.aay2400)).
2. **Stratego (DeepNash / R-NaD, Science 2022).** Game tree 10^535 (10^175×
   Go), 10^66 possible private deployments. Model-free self-play, **no
   search, no human data**, top-3 on the Gravon ladder vs humans. Deceptive
   play (bluff deployments, information-protecting moves) emerged
   ([arXiv 2206.15378](https://arxiv.org/abs/2206.15378);
   [DeepMind blog](https://deepmind.google/blog/mastering-stratego-the-classic-game-of-imperfect-information/)).
3. **Dota 2 (OpenAI Five).** Pure PPO self-play from scratch at extreme
   scale; imperfect info handled by nothing more clever than scale. Proof
   that brute force works; not a template at our budget.

AlphaStar's human data was a compute-saving exploration crutch for an
*unenumerable* strategy space, not a requirement of imperfect-info play.
Which brings in the key structural fact about our game:

**Randbats is a closed, enumerable universe.** Every species has a finite
generator catalog of sets; the public belief engine already narrows the
opponent's hidden space from observations. Stratego's 10^66 deployments and
poker's continuous bet space had to be *approximated*; our hidden space can
be tracked nearly exactly. This is the technical basis for the superhuman
thesis — and it is precisely the property that makes the strongest
algorithm class below (belief-state search) tractable here when it isn't in
OU.

## Why naive AlphaZero breaks here (and what each paper fixes)

Pokémon differs from Go on three axes, each of which breaks a piece of
vanilla AlphaZero self-play:

1. **Best-response self-play cycles.** In non-transitive strategy spaces
   (rock-paper-scissors metas), latest-vs-latest self-play chases cycles
   instead of converging. Fixes: fictitious play averaging (NFSP), league /
   population Nash (PSRO, AlphaStar), or **regularized dynamics** (R-NaD).
2. **Equilibrium play is mixed.** In imperfect-info games, optimal
   strategies randomize — deterministic argmax play leaks information and
   is exploitable (why Pluribus bluffs). Consequences for us: the
   *deployment* policy at equilibrium should not be pure argmax, and any
   training scheme must be able to represent and preserve mixed strategies.
3. **Simultaneous moves.** Both players commit actions at once; sequential
   PUCT (opponent replies *after* seeing our move) is the wrong game. Our
   root-PUCT's opponent-action-scenario averaging is a heuristic patch.
   The sound version solves a **matrix game at each node** (payoff = reward
   + future value), e.g. via regret matching —
   [Simultaneous AlphaZero, arXiv 2512.12486](https://arxiv.org/abs/2512.12486).
   Also note: a state's *value* in imperfect info depends on both players'
   beliefs, which is why naive state-value search is unsound and the E0
   run's oracle framing (known worlds) sidesteps rather than solves this.

## Algorithm menu (no human data), mapped to PokeZero

| algorithm | one-line mechanism | maps to | cost / fit |
|---|---|---|---|
| **R-NaD** (DeepNash) | PPO-like updates but reward is transformed with a KL term toward a slowly-updated *reference copy of the agent's own past policy*; converges to Nash instead of cycling | drop-in evolution of our PPO loop: the AlphaStar "KL anchor to human" becomes "KL to own reference policy" — same stabilizing machinery, zero human data | **best near-term fit**: model-free, no search dependency, proven at Stratego scale |
| **PSRO** ([Lanctot et al. 2017](https://arxiv.org/abs/1711.00832)) | league as empirical game: population of policies, iteratively add approximate best responses to the population meta-Nash | formalizes the cross-arm opponent-pool plan; `opponents.py` historical pools + win-rate-weighted (`f_hard`) sampling is PSRO-lite | cheap now, principled scaling path |
| **NFSP** | mix of best-response net and time-averaged policy net | simplest anti-cycling fix; superseded by R-NaD/PSRO | reference only |
| **MCCFR / blueprint + re-solving** (Libratus/Pluribus) | tabular regret minimization over abstracted game + depth-limited subgame re-solving at play time | full-game CFR is out (game too long/branchy), but *depth-limited re-solving with a value function at leaves* is the shape our search phase should grow toward | concept import, not code import |
| **ReBeL** ([arXiv 2007.13544](https://arxiv.org/abs/2007.13544)) | AlphaZero-for-imperfect-info: self-play RL + search over **public belief states** (public state + both players' belief distributions); provably converges to Nash in 2p zero-sum | our public belief engine ≈ a PBS factorization already; randbats enumerability makes PBS value nets realistic here when they aren't in OU | **the principled superhuman endpoint** for this game; heavy build, phase-3 |
| **Student of Games** ([Science Advances 2023](https://www.science.org/doi/10.1126/sciadv.adg3256)) | growing-tree CFR + sound self-play, one algorithm for chess/Go/poker/Scotland Yard | the "sound search" alternative to ReBeL; Scotland Yard is the closest published analog to Pokémon's pursuit-of-hidden-info flavor | heavier than ReBeL for our purposes; watch, don't build |
| **Simultaneous AlphaZero** ([arXiv 2512.12486](https://arxiv.org/abs/2512.12486)) | MCTS where each node solves a matrix game with a regret-optimal bandit solver | the correct replacement for root-PUCT's opponent-action scenarios | medium; natural next search milestone after E0 |

## What replaces AlphaStar's human-data roles

Every job human data did for AlphaStar has a self-generated substitute:

- **Exploration prior / washout protection** (their KL-to-supervised) →
  R-NaD reference-policy regularization (KL to own past self). Same
  mechanism, internal source.
- **Strategy diversity** (their z from human replays) → **self-derived z**:
  behavior descriptors are enumerable from the game itself (hazard usage,
  status usage, switch rate, game length). Sample descriptors from the
  agent's *own league history* (or uniformly over descriptor space for true
  coverage), condition the policy on z, pseudo-reward distance to the
  sampled descriptor, separate value head — AlphaStar's machinery with the
  human source swapped out. This is the no-human-data version of the
  coverage curriculum: "play a hazard game today" as a conditioning
  instruction rather than a scripted teacher.
- **Robustness targets** (their validation agents) → human/foul-play models
  quarantined to evaluation and exploiter-opponent duty (per the stated
  constraint), plus PSRO exploiter arms trained against the main.
- **Opponent model inside search** — the one place a human model *belongs*
  in the purist frame: search must predict the opponent's action
  distribution; using our own net assumes the opponent plays like us
  (the MIT thesis flags this exact limitation). A model of "what would a
  human/foul-play do" used strictly as a *world model* inside
  simulation teaches our policy nothing — it is environment, not teacher.
  Long-run, the equilibrium answer (ReBeL-style) models the opponent as
  also-rational-under-their-beliefs instead.

## Bluffing in Pokémon, concretely

The poker/Stratego results predict information-protecting play emerges from
equilibrium-seeking without being taught. Randbats analogs: revealing moves
has cost (belief engine narrows *our* set too — the opponent's model of us
sharpens with every reveal); staying in to represent a coverage move one
may not have; switch mind-games as the simultaneous-move mixed equilibrium;
sacrifice sequencing to conceal win conditions. None of this is learnable
by a deterministic-argmax agent evaluated only on win rate against a fixed
bot — it requires mixed strategies and an opponent that *adapts* (league /
exploiters), which is an argument that some population machinery is not
optional for the superhuman goal, human data or no.

## Suggested no-human-data sequencing

1. **R-NaD-style regularization arm** — reference policy = EMA/lagged
   snapshot of self; the anti-cycling, anti-washout backbone. (Replaces the
   retracted "KL to foul-play-BC" suggestion.)
2. **PSRO-lite**: cross-arm pools + `f_hard` win-rate-weighted sampling
   (already planned; unchanged — it never needed human data).
3. **Self-derived z conditioning** for coverage (hazard/status/switch
   descriptors enumerated from the game, not from replays).
4. **Matrix-game root search** (Simultaneous AlphaZero style) replacing
   opponent-action scenarios — after E0's verdict on whether search earns
   its keep at all.
5. **ReBeL-style PBS value + depth-limited re-solving** over the belief
   engine — the phase-3 bet that randbats' closed universe makes uniquely
   winnable here.
6. Human/foul-play models: eval yardsticks, search-time opponent models,
   exploiter opponents. Nothing else.
