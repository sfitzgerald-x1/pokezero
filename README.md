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

The `--p1-policy` and `--p2-policy` options accept `random-legal`, `simple-legal`, `scripted-teacher`, or a trained linear checkpoint spec:

```bash
python -m pokezero.rollout_cli collect \
  --games 10 \
  --out runs/linear-vs-random.jsonl \
  --showdown-root /path/to/pokemon-showdown \
  --p1-policy linear:checkpoints/linear-softmax.json \
  --p2-policy random-legal
```

Linear checkpoint specs default to stochastic softmax sampling for collection. Add query options when needed, for example `linear:checkpoints/linear-softmax.json?deterministic=true` for argmax evaluation-style collection, or `linear:checkpoints/linear-softmax.json?epsilon=0.1&temperature=1.5` for exploratory self-play.

`scripted-teacher` is a deterministic Gen 3 bootstrap policy backed by local Showdown dex metadata. It scores legal moves with Gen 3 type/category rules and uses switches mainly for force-switches or when legal attacks are poor. It is intended for bootstrap data generation, not as the target policy.

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

Linear checkpoints include the current action-space, observation, and linear-feature schema versions. Loading a stale checkpoint with mismatched runtime schemas fails fast instead of silently producing incompatible features.

The default `behavior-cloning` objective can only imitate the data source. Training on `random-legal` or `simple-legal` rollouts is useful as a plumbing smoke test, but it should not be expected to produce a stronger agent than those policies. The optional `reward-weighted` objective is an offline reward-weighted regression mode: it reinforces positive-return actions, ignores ordinary losing actions, and applies the configured capped-game return as an explicit anti-stall update. It is not a replacement for a stronger imitation source or a full self-play optimizer. Use held-out validation data for reported accuracy.

## Self-Play Iteration Harness

Generate a scripted-teacher bootstrap checkpoint in one command:

```bash
python -m pokezero.bootstrap_cli teacher \
  --run-dir runs/scripted-teacher-bootstrap \
  --train-games 1000 \
  --validation-games 200 \
  --workers 4 \
  --showdown-root /path/to/pokemon-showdown \
  --window-size 4
```

The bootstrap workflow writes full audit rollouts, current-teacher-only train and validation JSONL, a linear behavior-cloning checkpoint, and `manifest.json`. By default it collects against a teacher mirror plus `simple-legal` and `random-legal`, runs a short strict-teacher preflight before the full collection, and benchmarks the resulting checkpoint. It is the preferred way to seed the first checkpoint from the scripted teacher before moving into self-play.

Run the first linear-policy collect/train/evaluate loop:

```bash
python -m pokezero.selfplay_cli iterate \
  --run-dir runs/selfplay-smoke \
  --iterations 3 \
  --games-per-iteration 100 \
  --workers 4 \
  --validation-data runs/heldout.jsonl \
  --evaluation-games 20 \
  --showdown-root /path/to/pokemon-showdown
```

Each iteration writes full-audit `rollouts.jsonl`, current-policy-only `training-rollouts.jsonl`, `linear-policy.json`, and `manifest.json` under `iteration-NNNN/`, plus a top-level run manifest. The current checkpoint plays both seats across the collected games against a fixed opponent pool and a bounded history of older checkpoints. Training accumulates current-policy examples across iterations and warm-starts from the prior checkpoint. This is still a small linear-policy harness; it exists to make the improvement loop auditable before moving to a larger neural model.

Use `--workers N` to collect games in parallel within each iteration. Result files are still written in deterministic seed order, and the default remains `--workers 1` for simpler debugging.

Use `--validation-data` to attach one or more held-out rollout JSONL files to every training step. Validation metrics are stored in each iteration manifest and surfaced by the report command.

Validation metrics measure imitation fit against the held-out rollout labels, not policy strength. Use benchmark win rate, capped-game rate, and head-to-head evaluation results for checkpoint promotion decisions.

When reusing teacher bootstrap validation data during self-play, treat it as a teacher-retention regression check. A policy that improves past the teacher may become less teacher-faithful, so promotion still depends on benchmark strength and capped-game health.

Self-play training defaults capped games to a mild double-loss return with `--capped-terminal-value -0.25`. This keeps capped games from being free neutral outcomes while preserving a CLI override for experiments.

Pass `--resume` with the same `--run-dir` to continue from the latest manifest checkpoint. Existing run directories are not overwritten unless resume is explicit.

Summarize an existing run without opening the manifest JSON:

```bash
python -m pokezero.selfplay_cli report --run-dir runs/selfplay-smoke
```

Add `--json` to print the raw formatted run manifest for downstream scripts.

Evaluate whether a bootstrap or self-play manifest clears basic promotion gates:

```bash
python -m pokezero.eval_cli gate runs/bootstrap-selfplay \
  --min-benchmark-win-rate 0.55 \
  --min-benchmark-games 50 \
  --max-collection-capped-rate 0.10 \
  --max-benchmark-capped-rate 0.10
```

The gate command treats per-opponent benchmark win rates as the strength signal and capped-game rates as health checks. It returns exit code `0` for pass and `2` for fail, so shell scripts can use it before promoting a checkpoint. Bootstrap manifests also check teacher degradation counters by default.

By default the gate checks every fixed-opponent benchmark row independently and requires a minimum game count per opponent. Use `--benchmark-opponent scripted-teacher --opponent-win-rate scripted-teacher=0.50` when a specific fixed-opponent comparison is the promotion target. For self-play manifests, the gate auto-derives the incumbent from the previous iteration when possible; use `--registry runs/promotions.json` to default the incumbent to the latest promoted policy, or `--incumbent-policy <policy-id>` to override it. Incumbent checks use a separate point-estimate floor, minimum game count, capped-game limit, and Wilson lower-bound check via `--min-incumbent-win-rate`, `--min-incumbent-games`, `--max-incumbent-capped-rate`, and `--min-incumbent-win-rate-lower-bound`.

When `--evaluation-games` is enabled during self-play, the benchmark includes the fixed random/simple baselines plus a direct candidate-vs-incumbent comparison whenever the incumbent policy is a previous linear checkpoint or bootstrap checkpoint. Fixed baselines are not duplicated as incumbents because they are already benchmarked. Aggregate benchmark win rate and capped rate exclude the incumbent row; the incumbent is reported and gated separately.

Record a gate-passing checkpoint in an append-only promotion registry:

```bash
python -m pokezero.eval_cli promote runs/bootstrap-selfplay \
  --registry runs/promotions.json \
  --artifact-dir runs/promoted-checkpoints \
  --min-benchmark-win-rate 0.55 \
  --min-benchmark-games 50 \
  --label bootstrap-selfplay-0005
```

Inspect promoted checkpoints:

```bash
python -m pokezero.eval_cli promotions --registry runs/promotions.json
```

`promote` embeds the full gate result in the registry entry and refuses duplicate checkpoint entries by default. With `--artifact-dir`, it copies the accepted checkpoint into a stable artifact directory, stores that managed copy as the registry checkpoint path, and keeps the original source checkpoint path for audit. Without `--artifact-dir`, the registry stores references to existing run artifacts. Passing `--promotion-registry runs/promotions.json` to `selfplay_cli iterate` makes the historical opponent pool draw from promoted checkpoints instead of every raw prior iteration checkpoint.

Start self-play from a bootstrap checkpoint by first training offline data with `linear_cli train`, then passing the resulting checkpoint as the initial policy:

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

See `docs/bootstrap_strategy.md` for the cold self-play versus imitation-bootstrap plan.

## Gen 3 Belief Sidecar

The read-only sidecar can attach to a local Showdown battle room and display the public Gen 3 random-battle belief state:

```bash
python -m pokezero.sidecar serve \
  --room battle-gen3randombattle-123 \
  --showdown-root /path/to/pokemon-showdown \
  --showdown-url ws://localhost:8000/showdown/websocket
```

The Showdown checkout must be built so `dist/data/random-battles/gen3/teams.js` exists. The sidecar serves a local webview on `http://127.0.0.1:8010` and does not submit battle choices.
