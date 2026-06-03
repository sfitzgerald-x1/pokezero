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
- reward-weighted regression ignores ordinary losing examples, so it can fail to learn from many games
- self-play can amplify degenerate habits before the model learns useful tactics
- capped games are now mildly penalized by default in self-play, including under reward-weighted training

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

Generate the initial scripted-teacher bootstrap checkpoint in one command:

```bash
python -m pokezero.bootstrap_cli teacher \
  --run-dir runs/scripted-teacher-bootstrap \
  --train-games 1000 \
  --validation-games 200 \
  --workers 4 \
  --showdown-root /path/to/pokemon-showdown \
  --window-size 4
```

This writes full audit rollouts, current-teacher-only train and validation JSONL, a linear behavior-cloning checkpoint, baseline benchmark results, and `manifest.json`.

Default teacher bootstrap collection includes three opponent families:

- teacher mirror games, which reduce immediate covariate shift between bootstrap data and clone-vs-clone deployment
- `simple-legal`
- `random-legal`

This does not eliminate DAgger-style compounding error. The first self-play iterations still exist partly to correct states that the behavior-cloned checkpoint did not see in the teacher corpus.

The CLI keeps the scripted teacher strict by default. A short preflight run executes before the full collection so missing dex metadata, unresolved moves, or missing observation metadata fail early. If a policy spec deliberately enables `allow_unknown_moves=true` or `allow_fallback=true`, the manifest records teacher decision counters for unknown-move and fallback decisions so degraded data is visible.

The default benchmark is intentionally small and serial. Increase `--benchmark-games` for promotion decisions; set it to `0` only for smoke runs where the manifest does not need strength evidence.

Use the generated checkpoint as the first self-play policy:

```bash
python -m pokezero.selfplay_cli iterate \
  --run-dir runs/bootstrap-selfplay \
  --initial-policy linear:runs/scripted-teacher-bootstrap/linear-bootstrap.json \
  --validation-data runs/scripted-teacher-bootstrap/validation-rollouts.jsonl \
  --iterations 5 \
  --games-per-iteration 200 \
  --workers 4 \
  --evaluation-games 50 \
  --showdown-root /path/to/pokemon-showdown
```

Bootstrap validation data measures teacher imitation retention during self-play. It is useful for detecting catastrophic teacher forgetting, but it is not a promotion signal. If self-play improves past the teacher, validation fit against teacher labels can legitimately decrease.

The manual two-step path remains useful when a custom corpus has already been collected.

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

Gate a candidate before promotion:

```bash
python -m pokezero.eval_cli gate runs/bootstrap-selfplay \
  --min-benchmark-win-rate 0.55 \
  --min-benchmark-games 50 \
  --max-collection-capped-rate 0.10 \
  --max-benchmark-capped-rate 0.10
```

The gate is a configurable guardrail, not a final research threshold. It requires benchmark evidence by default, checks each candidate-vs-opponent benchmark row independently, enforces a minimum game count per opponent, checks collection and benchmark capped-game rates, and checks bootstrap teacher-degradation counters when present. Use `--json` for automation and `--allow-missing-benchmark` only for smoke runs.

Use opponent filters when a specific comparison matters more than broad baseline health:

```bash
python -m pokezero.eval_cli gate runs/bootstrap-selfplay \
  --benchmark-opponent scripted-teacher \
  --opponent-win-rate scripted-teacher=0.50 \
  --min-benchmark-games 100
```

Use an incumbent gate when deciding whether a self-play checkpoint should replace the prior promoted checkpoint:

```bash
python -m pokezero.eval_cli gate runs/bootstrap-selfplay \
  --registry runs/promotions.json \
  --min-incumbent-win-rate 0.55 \
  --min-incumbent-games 200 \
  --min-benchmark-games 100
```

When self-play runs with `--evaluation-games`, each iteration automatically benchmarks the candidate against the policy it just replaced if that incumbent is a linear checkpoint. The gate auto-derives that incumbent from self-play manifests when possible. Passing `--registry runs/promotions.json` instead defaults the incumbent to the latest promoted policy, with `--incumbent-policy` available as an explicit override. The incumbent row is gated separately from fixed baselines by point-estimate win rate, minimum games, capped-game rate, and a Wilson lower-bound no-regression check. Fixed random/simple baseline rows continue to use the normal per-opponent benchmark floors, and aggregate benchmark health excludes the incumbent row.

Record accepted checkpoints in a promotion registry:

```bash
python -m pokezero.eval_cli promote runs/bootstrap-selfplay \
  --registry runs/promotions.json \
  --artifact-dir runs/promoted-checkpoints \
  --min-incumbent-win-rate 0.55 \
  --min-incumbent-games 200 \
  --min-benchmark-games 100 \
  --label bootstrap-selfplay-0005
```

The registry is append-only by default and embeds the full gate result for each accepted checkpoint. With `--artifact-dir`, promotion copies the accepted single-file checkpoint into a stable artifact directory, records that managed copy as the checkpoint used by later self-play, records a SHA-256 checksum, and preserves the original source checkpoint path for audit. The embedded gate result still reflects the source manifest and source checkpoint that were evaluated. Without `--artifact-dir`, the registry stores references to existing checkpoint files.

The registry is the checkpoint-pool index for accepted policies: `selfplay_cli iterate --promotion-registry runs/promotions.json` uses promoted checkpoints as historical opponents instead of every raw prior iteration checkpoint.

Collection capped rate and benchmark capped rate are separate checks. Collection capped rate measures training-data health for the latest iteration or bootstrap corpus. Benchmark capped rate measures the candidate policy's evaluation-time stall tendency. Win rate intentionally uses all benchmark games as the denominator, so capped games hurt both win rate and capped-rate health.

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
- Is the default incumbent gate strict enough for long runs, or should it require a larger lower-bound margin above 0.50?
- How much imported data is needed before self-play fine-tuning is useful?
- Should bootstrap data continue to mix into later self-play training, or only initialize the first checkpoint?
