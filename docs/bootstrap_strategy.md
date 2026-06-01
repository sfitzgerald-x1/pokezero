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
- capped games currently provide no explicit anti-stall gradient

Use this path as a control condition and harness smoke, not as the only route to a strong policy.

## Path B: Imitation Bootstrap Then Self-Play

Start from stronger trajectories before self-play fine-tuning.

The bootstrap data could come from:

- curated human or ladder replays converted into player-relative trajectories
- a stronger non-learning policy used only to seed examples
- a future search/planning teacher used to generate target actions
- filtered self-play games from later checkpoints once a stronger pool exists

The current harness already supports this in two stages:

1. Train a checkpoint from offline rollout JSONL with `linear_cli train`.
2. Start self-play from that checkpoint with `selfplay_cli iterate --initial-policy linear:<checkpoint>`.

Held-out validation JSONL can now be passed into self-play iterations with `--validation-data`, so reports can distinguish in-sample fit from validation fit.

## Current Recommendation

Keep Path A running as a baseline, but build Path B next. The transformer/PPO phase will need stronger early signal than random/simple rollouts are likely to provide. The immediate goal should be to make replay/imported trajectory conversion reliable, then use those trajectories as a bootstrap corpus.

## Near-Term Implementation Plan

- Add a replay-to-trajectory importer that produces the same rollout JSONL schema as live collection.
- Keep all imported observations player-relative and player-knowable.
- Add provenance metadata for imported examples, including source, replay id, winner, rating/quality fields when available, and conversion version.
- Train bootstrap checkpoints from imported data with held-out validation.
- Start self-play from the bootstrap checkpoint and compare against cold-start runs using the self-play report command.
- Track capped-game rate, benchmark win rate, validation fit, and games per hour for both paths.

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

## Open Questions

- What source should be the first bootstrap corpus?
- Should capped games be treated as tie, double loss, or explicit stall penalty?
- What benchmark win-rate delta should promote a checkpoint?
- How much imported data is needed before self-play fine-tuning is useful?
- Should bootstrap data continue to mix into later self-play training, or only initialize the first checkpoint?
