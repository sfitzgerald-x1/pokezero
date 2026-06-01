# Bootstrap Strategy

PokeZero now has enough harness infrastructure to run repeatable self-play experiments, but the next strategic question is data quality. The current linear reward-weighted loop is useful as a plumbing baseline; it should not be treated as proof that cold self-play from weak policies will produce a strong agent.

## Decision Fork

There are two viable paths for the next phase.

## Path A: Cold Self-Play Baseline

Continue from random/simple baselines and let the current policy improve through repeated self-play.

This path is simplest operationally:

- collect games from the current policy against fixed and historical opponents
- train on current-policy examples
- benchmark against fixed baselines
- promote checkpoints only when benchmark win rates improve

Risks:

- early trajectories are low quality
- reward-weighted regression ignores losing and capped examples, so it can fail to learn from many games
- self-play can amplify degenerate habits before the model learns useful tactics
- capped games are now mildly penalized by default in self-play, but reward-weighted regression still ignores non-positive examples

Use this path as a control condition and harness smoke, not as the only route to a strong policy.

## Path B: Imitation Bootstrap Then Self-Play

Start from stronger trajectories before self-play fine-tuning.

The bootstrap data could come from:

- a stronger non-learning policy used only to seed examples
- a future search/planning teacher used to generate target actions
- curated Gen 3 random battle replays converted into player-relative trajectories
- filtered self-play games from later checkpoints once a stronger pool exists

The current harness already supports this in two stages:

1. Train a checkpoint from offline rollout JSONL with `linear_cli train`.
2. Start self-play from that checkpoint with `selfplay_cli iterate --initial-policy linear:<checkpoint>`.

Held-out validation JSONL can now be passed into self-play iterations with `--validation-data`, so reports can distinguish in-sample fit from validation fit.

Validation fit is still an imitation metric. It answers whether the checkpoint matches the held-out rollout labels, not whether the policy is stronger. Benchmark win rate, capped-game rate, and head-to-head results remain the promotion signal.

## Gen 3 Randbat Corpus Constraint

The easiest replay corpora to find are usually ladder or tournament games in constructed formats. Those are not drop-in training data for Gen 3 random battles because the team-generation distribution, hidden-information assumptions, and common tactical situations differ.

Curated Gen 3 randbat replays may still be useful, but the corpus source is unresolved. Until we know that a large enough randbat-specific replay set exists, replay import should not be the critical path for bootstrapping.

## Current Recommendation

Keep Path A running as a baseline, but build Path B around the scripted Gen 3 randbat teacher. The next learner will need stronger early signal than random/simple rollouts are likely to provide, and a teacher can generate format-matched trajectories immediately without waiting on replay corpus availability.

Replay import remains valuable after a randbat replay source is identified. It should share the same rollout JSONL schema and player-relative observation path, but it should not block the first bootstrap iteration.

## Near-Term Implementation Plan

- Use the initial capped-game scoring policy: self-play defaults to `--capped-terminal-value -0.25`, a mild double-loss penalty that can be tuned later.
- Expand the deterministic scripted teacher for Gen 3 randbats beyond the initial metadata-backed version, covering more battle context such as hazards, status value, and safer switch participation.
- Collect teacher-vs-baseline and teacher-self-play trajectories through the normal rollout JSONL path.
- Train bootstrap checkpoints from teacher trajectories with held-out validation.
- Start self-play from the bootstrap checkpoint and compare against cold-start runs using the self-play report command.
- Benchmark each candidate against `random-legal`, `simple-legal`, historical self-play checkpoints, and the static bootstrap checkpoint.
- Track benchmark win rate, capped-game rate, validation fit, and games per hour for both paths. Treat validation fit as imitation-health only.
- Add a replay-to-trajectory importer after a useful Gen 3 randbat replay corpus is identified.

## Supported Command Shape

Train a bootstrap checkpoint from offline data:

```bash
python -m pokezero.linear_cli train \
  --data runs/bootstrap-train.jsonl \
  --validation-data runs/bootstrap-validation.jsonl \
  --out checkpoints/bootstrap-linear.json \
  --objective behavior-cloning \
  --window-size 4
```

Start self-play from that checkpoint:

```bash
python -m pokezero.selfplay_cli iterate \
  --run-dir runs/bootstrap-selfplay \
  --initial-policy linear:checkpoints/bootstrap-linear.json \
  --validation-data runs/bootstrap-validation.jsonl \
  --iterations 5 \
  --games-per-iteration 200 \
  --workers 4 \
  --evaluation-games 50 \
  --showdown-root /path/to/pokemon-showdown
```

Inspect the run:

```bash
python -m pokezero.selfplay_cli report --run-dir runs/bootstrap-selfplay
```

Collect initial teacher data directly from the local Showdown harness:

```bash
python -m pokezero.rollout_cli collect \
  --games 200 \
  --out runs/scripted-teacher-vs-baseline.jsonl \
  --showdown-root /path/to/pokemon-showdown \
  --p1-policy scripted-teacher \
  --p2-policy simple-legal
```

## Open Questions

- What source should be the first bootstrap corpus?
- Is `--capped-terminal-value -0.25` enough pressure, or should capped games become a stronger double-loss or explicit stall penalty?
- What benchmark win-rate delta should promote a checkpoint?
- How much imported data is needed before self-play fine-tuning is useful?
- Should bootstrap data continue to mix into later self-play training, or only initialize the first checkpoint?
