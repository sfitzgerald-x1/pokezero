# AlphaStar (Vinyals et al., Nature 2019) — mechanisms that apply to PokeZero

Status: reference notes, 2026-07-03. Source: "Grandmaster level in StarCraft II
using multi-agent reinforcement learning" ([Nature](https://www.nature.com/articles/s41586-019-1724-z),
[open PDF](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf)).
Companion to [`wse_shaping_and_coverage_design.md`](wse_shaping_and_coverage_design.md)
and the population-diversity discussion. Numbers below are from the paper, not
paraphrase.

## Why this paper is on point

AlphaStar's central exploration problem is *our Spikes problem*, in their own
words: a policy skilled at ground-unit micro will find that "any deviation
that builds and naively uses air units will reduce performance. It is highly
improbable that naive exploration will execute a precise sequence of
instructions, over thousands of steps, that constructs air units and
effectively utilizes their micro-tactics." Substitute "sets Spikes and then
forces switches over many turns" and the failure mode is identical: a
coherent multi-step line whose payoff never materializes under noisy
deviation, so RL prunes the first step.

Their answer was **not** search and **not** entropy. It was (a) human data as
a standing exploration prior, (b) a league of diverse opponents. Both have
cheap partial transfers to us.

## Mechanism inventory (exact)

### 1. Human data as exploration scaffolding

- **Supervised init**: every RL agent starts from a supervised policy trained
  on 971k human replays (MMR > 3500 = top 22%), fine-tuned on 16k winning
  high-MMR (> 6200) replays — fine-tuning alone moved Elite-bot win rate
  87% → 96%.
- **Continual KL anchor**: during RL, agents pay a penalty "whenever their
  action probabilities differ from the supervised policy" — the human prior
  never washes out.
- **Strategy statistic z**: from each replay they extract z = build order
  (first 20 buildings/units) + cumulative statistics (units/buildings/
  upgrades present). The policy and value are *conditioned on z*
  (`π(a|s,z)`, `V(s,z)`); z is zeroed 10% of the time in SL so an
  unconditional mode exists.
- **Pseudo-rewards toward z**: during RL, main agents sample a human z and
  receive pseudo-rewards = edit distance between sampled and executed build
  order, Hamming distance between sampled and executed cumulative stats.
  Each pseudo-reward type is active with probability **25%**, with
  **separate value functions and losses** per pseudo-reward.
- Ablation (their Fig. 3E): the human-data mechanisms were "critical" — the
  single most load-bearing component group.

### 2. League training (population diversity)

- Composition: **3 main agents** (one per race), **3 main exploiters**,
  **6 league exploiters**. Each trained on 32 TPUv3 for 44 days; ~900 frozen
  "players" snapshotted into the league.
- **Main agents**: 35% self-play, 50% PFSP vs all past players, 15% PFSP vs
  forgotten main players + past main exploiters. Snapshot every 2·10⁹ steps.
  Never reset.
- **Main exploiters**: train *only against current main agents* (with an
  f_var-PFSP curriculum over main-lineage players when win prob < 20%).
  Added to league on beating all mains > 70% or timeout (4·10⁹ steps), then
  **always reset to supervised weights**. Purpose: find weaknesses of the
  mains; make the mains robust.
- **League exploiters**: PFSP vs everyone; added on > 70% league win rate or
  timeout; **25% chance of reset to supervised**. Purpose: find *systemic*
  blind spots of the whole league (strategies nobody in the league beats).
- **PFSP matchmaking**: sample frozen opponent B ∝ f(P[A beats B]).
  Default `f_hard(x) = (1−x)^p` — zero weight on already-beaten opponents;
  a smooth max-min (beat *everyone*) rather than FSP's max-average, which is
  what lets rare-but-strong exploits actually enter the learning signal
  instead of being averaged away. Alternative `f_var(x) = x(1−x)` (opponents
  near own level) for struggling agents and exploiters.
- Ablations: naive self-play reaches high Elo but is *forgetful* (loses to
  past versions); PFSP beats FSP on strength, exploitability, and final
  performance (their Extended Data Fig. 5).

### 3. Privileged (centralised) value function

During training only, "the value function is estimated using information
from both the player's and the opponent's perspectives" — the critic sees
the opponent's observations; the policy does not. Values are discarded at
inference, so nothing leaks to play. Ablation (Fig. 3K) shows a clear
variance-reduction win.

### 4. Explicit rejection of search

AlphaStar is fully model-free: it "sidesteps the difficulties of
search-based methods due to imperfect models." No MCTS anywhere in training
or inference. Note carefully *why*: StarCraft has no usable forward model.
This rejection does **not** transfer to Pokémon, where a perfect simulator
exists — but their diagnosis that the *strategy-discovery* problem is a
data/exploration problem (solved by human priors + league, not by search)
does transfer, and matches our ΔV finding.

### 5. RL machinery (context, lower priority for us)

Terminal win/loss reward, undiscounted, no hand shaping (all dense signal
comes from the z pseudo-rewards). TD(λ) for values, V-trace for policy,
UPGO (self-imitation on better-than-expected partial returns). Off-policy
corrections split per action argument due to the huge action space.

## Transfer map to PokeZero

| AlphaStar mechanism | PokeZero analog | cost | verdict |
|---|---|---|---|
| Supervised init from human replays | fpdistill lineage (foul-play BC) already does this; metamon-style human replays via `pokezero-replay-import` are the truer analog | exists / small | already partially in place |
| Continual KL anchor to supervised policy | KL penalty in PPO toward a frozen `foul-play-distill-base` (or human-BC) policy — prevents self-play from washing out taught strategies (our fpdistill Pearson erosion 0.505→0.489 with self-play FT is this exact washout, measured) | small: one extra forward per batch + a loss term | **high value, cheap — strong candidate** |
| z conditioning + pseudo-rewards (25% active, separate value heads) | z = behavior descriptor from human/foul-play games (hazard usage, status usage, switch rate, game length); pseudo-reward = distance between executed and sampled descriptor; separate value head per pseudo-reward | medium: descriptor extraction + conditioning input + extra heads | **the principled version of "diverse versions of current strength"** — one model, diversity by conditioning, no N-learner cost. Directly targets the Spikes class. |
| League: mains + exploiters, ~900 players | full version is 12 concurrent learners — far beyond budget. Budget version: cross-arm opponent pools over the #487 reward-diversified arms + historical snapshots (already supported by `opponents.py`) | full: prohibitive; lite: ~free | lite version now; exploiter-lite (one arm rewarded to beat the current main) only if cross-arm pools show ΔV movement |
| PFSP `f_hard` weighting | replace uniform historical-pool sampling with win-rate-weighted sampling (we already estimate per-opponent win rates in evals) | small: sampling-weight change in pool selection | worth folding into any pool work |
| Privileged critic (opponent obs into value fn, training only) | critic input = both players' observations (or true hidden state — we own the simulator); policy unchanged; inference unchanged | medium: value-head input plumbing + checkpoint compat | **directly attacks value credit assignment** — complements WS-E shaping; belief features already being the top value differentiator (0.48–0.53 vs 0.22–0.25 no-belief) says fuller state info helps this head |
| UPGO / V-trace / TD(λ) | PPO+GAE serves; revisit only if off-policy pools (league) make PPO strain | — | not now |

## On "MCTS in training" (expert iteration) — where this leaves it

The hope that in-loop search would *discover* Spikes value inverts the
dependency. Search consults the value head at leaves (plus short rollouts).
With hazard-blind heads (ΔV self-response ≤ 4% of value spread) and 2-round
rollouts — far shorter than the multi-turn hazard payoff horizon — root
search cannot see the payoff either; expert iteration would distill the
blindness, faithfully. AlphaStar's diagnosis applies: strategy discovery is
a data/prior problem. Search amplifies what the evaluator already knows; it
does not teach the evaluator new strategic concepts at these depths.

Cost, from today's E0 measurements (not the MIT thesis's 10 s/move MCTS):
root-PUCT averaged **0.58 s per searched decision** (~30 searched decisions
per game → ~17 s search overhead per game, vs ~1 s/game raw collection).
In-loop search on every collection game ≈ 10–30× collection cost —
the thesis's write-off stands for the *continuous* form. A **periodic**
form (every N iterations, generate a small search-labeled batch under
oracle-ish settings and distill it, ~5–10% of game budget) is affordable —
but only worth scheduling after a value head passes the ΔV bar, and only if
E0-oracle shows a margin worth distilling (current partial read: raw 8.0%
vs root-PUCT 12.4% at 250/300 pairs, p≈0.054 — thin).

## Suggested sequencing (cheap → expensive)

1. **KL anchor** to `foul-play-distill-base` in one WS-E arm (measured
   antidote to strategy washout).
2. **Privileged critic** arm (training-only full-information value input) —
   pairs naturally with the dense-shaping arm's credit-assignment goal.
3. **Cross-arm opponent pools** with PFSP-style `f_hard` weighting over the
   #487 arms + historical snapshots.
4. **z pseudo-rewards** from foul-play/human behavior descriptors (the full
   AlphaStar exploration mechanism, medium build).
5. **Exploiter-lite** and periodic expert-iteration stages — both gated on
   the above moving ΔV/behavior probes and on E0's final verdict.
