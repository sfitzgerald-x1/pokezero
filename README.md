# PokeZero

PokeZero is an experiment to train a model to play Pokemon Showdown random battles through self-play.

The initial focus is Gen 3 random battles. The goal is to build a training loop where agents repeatedly battle each other, learn from the resulting games, and improve decision quality with fast learned policies.

This repo will hold the self-play, training, evaluation, and model artifacts for that work.

## Rollout Collection

Collect local random-vs-random self-play trajectories as JSONL:

```bash
python -m pokezero.rollout_cli collect \
  --games 10 \
  --out runs/random-vs-random.jsonl \
  --showdown-root /path/to/pokemon-showdown
```

The Showdown checkout must be built so `dist/sim/index.js` exists. Each JSONL row contains one battle trajectory plus seed, policy ids, terminal outcome, decision-round count, simulator turn count, and elapsed time.

The printed throughput metrics use wall-clock collection time, including JSONL serialization. Use `pokezero.collection.iter_rollout_records(path)` for streaming reads of large trajectory files.

The `--p1-policy` and `--p2-policy` options accept `random-legal`, `simple-legal`, or a trained linear checkpoint spec:

```bash
python -m pokezero.rollout_cli collect \
  --games 10 \
  --out runs/linear-vs-random.jsonl \
  --showdown-root /path/to/pokemon-showdown \
  --p1-policy linear:checkpoints/linear-softmax.json \
  --p2-policy random-legal
```

Linear checkpoint specs default to stochastic softmax sampling for collection. Add query options when needed, for example `linear:checkpoints/linear-softmax.json?deterministic=true` for argmax evaluation-style collection, or `linear:checkpoints/linear-softmax.json?epsilon=0.1&temperature=1.5` for exploratory self-play.

Run baseline rollout benchmarks without writing trajectory JSONL:

```bash
python -m pokezero.rollout_cli benchmark \
  --games 100 \
  --showdown-root /path/to/pokemon-showdown
```

The benchmark command runs `random-legal` and `simple-legal` against each other in both seats, reports win/cap/turn-count metrics, and uses the same seed range for each matchup so results are easier to compare. The default `--games 20` is a throughput smoke. Use hundreds of games before treating the mirror-aggregated head-to-head rows as policy-quality evidence.

## Trajectory Dataset Loading

Rollout JSONL can be streamed into fixed-shape training examples and batches:

```python
from pokezero.dataset import TrajectoryDatasetConfig, iter_training_batches

config = TrajectoryDatasetConfig(window_size=4, discount=1.0)
for batch in iter_training_batches("runs/random-vs-random.jsonl", batch_size=64, config=config):
    ...
```

Each example contains a left-padded per-player history window, the current legal-action mask, selected action, immediate reward, terminal-derived discounted return, opponent action metadata, and source identifiers. The batch objects are dependency-free tuple containers so they can be converted to NumPy, PyTorch, or another tensor runtime later without adding a training-framework dependency yet.

Padding uses zero-shaped observation values plus `history_mask=False` for missing history slots. Training code must gate temporal attention or pooling with `history_mask`; categorical id `0` is not reserved as a universal padding token. The streaming order is also game-sequential, so gradient training should add a shuffle buffer before consuming batches directly.

## Linear Policy Baseline

Train the first dependency-free masked softmax policy from collected rollout JSONL:

```bash
python -m pokezero.linear_cli train \
  --data runs/random-vs-random.jsonl \
  --validation-data runs/heldout.jsonl \
  --out checkpoints/linear-softmax.json \
  --epochs 3 \
  --objective behavior-cloning \
  --window-size 1
```

Evaluate the checkpoint offline against rollout labels:

```bash
python -m pokezero.linear_cli evaluate \
  --data runs/random-vs-random.jsonl \
  --checkpoint checkpoints/linear-softmax.json
```

Benchmark it in live local self-play against the fixed baselines:

```bash
python -m pokezero.linear_cli benchmark \
  --checkpoint checkpoints/linear-softmax.json \
  --games 20 \
  --showdown-root /path/to/pokemon-showdown
```

This baseline uses hashed observation-window features, a streaming shuffle buffer, and legal-action-masked linear objectives. It is intentionally small and CPU-only; its purpose is to validate the train/save/load/evaluate loop before adding a heavier learner.

The default `behavior-cloning` objective can only imitate the data source. Training on `random-legal` or `simple-legal` rollouts is useful as a plumbing smoke test, but it should not be expected to produce a stronger agent than those policies. The optional `reward-weighted` objective is an offline reward-weighted regression mode: it reinforces positive-return actions and ignores non-positive-return actions. It is not a replacement for a stronger imitation source or a full self-play optimizer. Use held-out validation data for reported accuracy.

## Self-Play Iteration Harness

Run the first linear-policy collect/train/evaluate loop:

```bash
python -m pokezero.selfplay_cli iterate \
  --run-dir runs/selfplay-smoke \
  --iterations 3 \
  --games-per-iteration 100 \
  --evaluation-games 20 \
  --showdown-root /path/to/pokemon-showdown
```

Each iteration writes `rollouts.jsonl`, `linear-policy.json`, and `manifest.json` under `iteration-NNNN/`, plus a top-level run manifest. The current checkpoint plays both seats across the collected games against a fixed opponent pool and a bounded history of older checkpoints. This is still a small linear-policy harness; it exists to make the improvement loop auditable before moving to a larger neural model.

## Gen 3 Belief Sidecar

The read-only sidecar can attach to a local Showdown battle room and display the public Gen 3 random-battle belief state:

```bash
python -m pokezero.sidecar serve \
  --room battle-gen3randombattle-123 \
  --showdown-root /path/to/pokemon-showdown \
  --showdown-url ws://localhost:8000/showdown/websocket
```

The Showdown checkout must be built so `dist/data/random-battles/gen3/teams.js` exists. The sidecar serves a local webview on `http://127.0.0.1:8010` and does not submit battle choices.
