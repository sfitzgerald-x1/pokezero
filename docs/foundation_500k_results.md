# Foundation 500k run results

Status: current evidence from the completed recipe-fidelity foundation run that ended at
**500,800 self-play games** on June 30, 2026.

This note records evaluation evidence only. It intentionally omits private operational details.

## Source papers

Two papers shaped this run and the surrounding roadmap:

- **Training recipe inspiration:** Jett Wang, *Winning at Pokémon Random Battles Using
  Reinforcement Learning* (MIT EECS MEng thesis, 2024),
  <https://dspace.mit.edu/handle/1721.1/153888>. The recipe we are testing is the thesis's
  separation of PPO self-play training from inference-time MCTS, with sparse game-outcome reward,
  value-function clipping, strong entropy pressure, LR annealing, GPU training, and CPU-parallel
  battle collection.
- **Transformer/input inspiration:** Jake Grigsby, Yuqi Xie, Justin Sasek, Steven Zheng, and Yuke
  Zhu, *Human-Level Competitive Pokémon via Scalable Offline Reinforcement Learning with
  Transformers* (RLC 2025 / UT Austin RPL), <https://arxiv.org/abs/2504.04395> and
  <https://metamon.tech/>. This project does **not** depend on Metamon or use its offline-human-data
  training recipe, but its first-person trajectory framing and transformer-over-battle-history
  approach are useful inspiration for PokeZero's entity/action/history token input shape.

## Run shape

This run was intended as a mid-scale recipe-fidelity read before spending a full multi-million-game
budget.

- **Training games:** 500,800.
- **Update cadence:** 1,600 games per PPO update.
- **Final iteration:** 313.
- **Training device:** GPU for the central PPO train step.
- **Evaluation cadence:** standard low-fidelity yardstick reads at 10k-game thresholds, plus
  independent high-fidelity foul-play reads at 50k-game milestones.
- **Recipe-fidelity knobs:** value-function clipping (`clip_range_vf=0.0184`), 7 PPO epochs,
  `entropy_coef=0.0588`, `gamma=0.9999`, GAE lambda `0.754`, annealed LR over the scheduled run,
  and batch size 1024.

This is still **not** the full MIT recipe: it is roughly one sixth of the thesis's ~3M-battle
training scale, uses PokeZero's entity-token transformer rather than the thesis MLP, and uses
PokeZero's scheduled 1,600-game PPO update cadence rather than the thesis's async rollout-buffer
cadence.

## High-fidelity foul-play milestones

These are independent 1,000-game foul-play reads at 50k-game checkpoints unless noted otherwise.
The training-game column is the nominal milestone threshold; the evaluated checkpoint is the listed
iteration, which lands on the first 1,600-game update boundary at or after that threshold.

| Training games | Iteration | Status | Wins / games | Win rate |
|---:|---:|---|---:|---:|
| 50,000 | 32 | complete | 22 / 1000 | 2.2% |
| 100,000 | 63 | complete | 26 / 1000 | 2.6% |
| 150,000 | 94 | complete | 33 / 1000 | 3.3% |
| 200,000 | 125 | complete | 31 / 1000 | 3.1% |
| 250,000 | 157 | complete | 33 / 1000 | 3.3% |
| 300,000 | 188 | complete | 39 / 1000 | 3.9% |
| 350,000 | 219 | complete | 27 / 1000 | 2.7% |
| 400,000 | 250 | partial artifact | 34 / 1013 | 3.4% |
| 450,000 | 282 | complete | 37 / 1000 | 3.7% |
| 500,000 | 313 | complete | 37 / 1000 | 3.7% |

The 400k read completed more than the requested 1,000 games, but the older foul-play runner wrote it
as `partial-result.json`. Treat the 34/1013 row as useful directional evidence, but normalize or
rerun it before using it as a clean plotted point.

## Standard yardstick context

The standard milestone benchmark is lower fidelity than the 1,000-game foul-play reads, but it is
useful context because it ran regularly throughout the training job. Each row aggregates the mirrored
benchmark orientations for that opponent: 300 games with the checkpoint in one seat plus 300 games
with the checkpoint in the other seat, for 600 games total.

### Max-damage milestone progression

Rows marked `high-fidelity` are independent 2,000-game mirrored reads. Later 50k milestones did
not yet have high-fidelity max-damage backfills recorded, so those rows use the regular 600-game
mirrored yardstick and are labeled `standard`.

| Training games | Iteration | Source | Wins / games | Win rate |
|---:|---:|---|---:|---:|
| 50,000 | 32 | high-fidelity | 627 / 2000 | 31.4% |
| 100,000 | 63 | high-fidelity | 739 / 2000 | 37.0% |
| 150,000 | 94 | high-fidelity | 789 / 2000 | 39.5% |
| 200,000 | 125 | high-fidelity | 843 / 2000 | 42.2% |
| 250,000 | 157 | high-fidelity | 859 / 2000 | 43.0% |
| 300,000 | 188 | high-fidelity | 914 / 2000 | 45.7% |
| 350,000 | 219 | standard | 303 / 600 | 50.5% |
| 400,000 | 250 | standard | 300 / 600 | 50.0% |
| 450,000 | 282 | standard | 320 / 600 | 53.3% |
| 500,000 | 313 | standard | 309 / 600 | 51.5% |

The max-damage curve improved materially through the first few hundred thousand games and remained
around the low 50s in the lower-fidelity late standard reads. That is the main actionable signal
from this run: the recipe has produced useful strength against a simpler fixed opponent, and the
500k checkpoint is not enough evidence to call the recipe exhausted.

### Final standard yardstick snapshot

| Milestone | Iteration | Opponent | Wins / games | Win rate |
|---:|---:|---|---:|---:|
| 500k | 313 | max-damage | 309 / 600 | 51.5% |
| 500k | 313 | simple-legal | 554 / 600 | 92.3% |
| 500k | 313 | random-legal | 592 / 600 | 98.7% |

## Readout

The 500k readout is constructive for the current recipe. The clearest leading signal is the
max-damage curve: it improved materially across the run and was still around the low 50s in the late
lower-fidelity standard reads. That is evidence that the policy learned useful strength against a
simpler fixed opponent. This is not a recipe rejection while max-damage remains non-collapsed; it is
the reason to continue the recipe-faithful run and look for a clearer large-scale trend.

The high-fidelity foul-play series remains in the low single digits through 500k games, but that is
not surprising at this stage. Foul-play is a much stronger, higher-quality bot than max-damage, so it
is expected to beat PokeZero until the policy reaches a substantially higher level of play. The
working expectation is that foul-play progress may be delayed and nonlinear: the score may stay low
for a long time and then move once the model crosses a practical competence threshold. Because the
max-damage curve is still the healthier near-term signal, low early foul-play scores should be
treated as a high-bar lagging readout rather than evidence that the recipe is already exhausted.

The MIT thesis recipe used a much larger training scale, so the right next readout is PokeZero's
scheduled continuation toward 1M rather than treating the 500k foul-play score as a rejection of the
recipe. As long as max-damage keeps holding or improving at larger scale, the working hypothesis is
that the recipe remains promising. Foul-play should stay in the report because it is the higher
quality bar, but the expected path is max-damage improvement first and foul-play movement later.

Concrete follow-ups:

- Continue the recipe-faithful run toward 1M games and track whether the max-damage curve keeps
  holding or climbing.
- Keep recording high-fidelity foul-play milestones, but interpret them as a high-bar lagging signal
  that may move only after the policy reaches a stronger level of play.
- Normalize or rerun the 400k over-complete partial artifact so plots do not mix clean and partial
  statuses.
- Use max-damage holding or improving as the near-term evidence that the recipe remains viable,
  while watching foul-play for delayed nonlinear movement.
- Keep the MIT recipe and UT Austin transformer/input paper as inspiration, but continue to verify
  PokeZero-specific assumptions with fixed-opponent curves.
