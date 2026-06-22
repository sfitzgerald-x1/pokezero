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

Replay import remains valuable after a randbat replay source is identified. A normalized replay-to-rollout scaffold now exists and writes the same rollout JSONL schema used by self-play collection. This normalized shape deliberately sits below raw Showdown replay parsing: it expects player-relative observations and action indices that a future corpus-specific converter must reconstruct. Raw Showdown replay discovery, curation, and conversion are still unresolved, so replay import should not block the first bootstrap iteration.

## Near-Term Implementation Plan

- Use the initial capped-game scoring policy: self-play defaults to `--capped-terminal-value -0.25`, a mild double-loss penalty that can be tuned later.
- Continue expanding the deterministic scripted teacher for Gen 3 randbats beyond the initial metadata-backed version. It now covers team-status cure value, status-aware switch targets, low-HP active preservation, and first-pass Spikes/Rapid Spin awareness; richer hazard planning and matchup context remain future work.
- Collect teacher-vs-baseline and teacher-self-play trajectories through the normal rollout JSONL path.
- Train bootstrap checkpoints from teacher trajectories with held-out validation.
- Start self-play from the bootstrap checkpoint and compare against cold-start runs using the self-play report command.
- Benchmark each candidate against `random-legal`, `simple-legal`, historical self-play checkpoints, and the static bootstrap checkpoint.
- Track benchmark win rate, capped-game rate, validation fit, games per hour, average decision-round length, and best-effort process peak RSS high-water marks for both paths. Treat validation fit as imitation-health only.
- Use `python -m pokezero.eval_cli audit-calibrate <run-dir>` after pilot runs to derive starting audit thresholds from observed history before enforcing them on longer unattended experiments.
- Extend the normalized replay-to-rollout importer with a raw Showdown replay converter after a useful Gen 3 randbat replay corpus is identified.

## Supported Command Shape

Import normalized replay decisions into standard rollout JSONL:

```bash
python -m pokezero.replay_import_cli import \
  --input data/normalized-replays/battle-001.json \
  --output runs/replay-bootstrap/rollouts.jsonl
```

The importer expects one battle per input file, with player-relative observations and fixed action indices already encoded in the normalized replay file. Raw Showdown replay conversion remains the harder corpus-specific step.

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

Quickly benchmark the scripted teacher itself against fixed baselines before running a full bootstrap:

```bash
python -m pokezero.bootstrap_cli teacher-benchmark \
  --games 50 \
  --showdown-root /path/to/pokemon-showdown
```

Use this as a cheap quality check after changing scripted-teacher heuristics. It reports teacher fallback and unknown-move counters alongside win rates, but it does not train a checkpoint or write a manifest.

Default teacher bootstrap collection includes three opponent families:

- teacher mirror games, which reduce immediate covariate shift between bootstrap data and clone-vs-clone deployment
- `simple-legal`
- `random-legal`

This does not eliminate DAgger-style compounding error. The first self-play iterations still exist partly to correct states that the behavior-cloned checkpoint did not see in the teacher corpus.

The CLI keeps the scripted teacher strict by default. A short preflight run executes before the full collection so missing dex metadata, unresolved moves, or missing observation metadata fail early. If a policy spec deliberately enables `allow_unknown_moves=true` or `allow_fallback=true`, the manifest records teacher decision counters for unknown-move and fallback decisions so degraded data is visible.

Linear training can train a supervised opponent-action auxiliary head via `--opponent-action-loss-weight`, but it is off by default (weight `0`). The linear policy's action weights are independent of this head, so enabling it does not change play; it exists as opt-in scaffolding so the later transformer policy can carry the same prediction task on a shared representation. Set a positive weight only to collect opponent-prediction metrics or to train the head in isolation.

The first transformer scaffold is available behind the optional neural dependency extra:

```bash
pip install -e '.[neural]'
python -m pokezero.neural_cli describe
```

The neural CLI can train an entity-token transformer checkpoint from rollout JSONL. Neural checkpoints can be used in rollout and benchmark policy specs as `neural:/path/to/checkpoint.pt`. The linear self-play iteration CLI intentionally rejects `--initial-policy neural:...`; neural checkpoints use the separate neural iteration command below.

Benchmark a neural checkpoint through the same local Showdown harness:

```bash
python -m pokezero.neural_cli benchmark \
  --checkpoint runs/neural-bootstrap/entity-transformer.pt \
  --games 50 \
  --showdown-root /path/to/pokemon-showdown
```

Run a first neural self-play iteration loop:

```bash
python -m pokezero.neural_cli iterate \
  --run-dir runs/neural-selfplay \
  --initial-policy linear:runs/scripted-teacher-bootstrap/linear-bootstrap.json \
  --iterations 3 \
  --games-per-iteration 200 \
  --workers 4 \
  --evaluation-games 25 \
  --promotion-registry runs/promotions.json \
  --promotion-artifact-dir runs/promoted-checkpoints \
  --auto-promote \
  --min-incumbent-games 50 \
  --showdown-root /path/to/pokemon-showdown
```

This neural loop collects current-policy-only rollout data, trains a transformer checkpoint from accumulated training rollouts each iteration, benchmarks the checkpoint when `--evaluation-games` is positive, and writes per-iteration manifests. Multi-iteration runs require evaluation games because each candidate must pass the advancement gate before it becomes the next rollout collector. With `--auto-promote`, passing candidates are recorded in the shared promotion registry and can be copied into the managed artifact directory; failed candidates remain saved and measured but do not become the collector. The example uses `--min-incumbent-games 50` because 25 evaluation games produce 50 mirrored incumbent games. Use `--resume` to continue an interrupted neural run from the latest manifest.

This is still supervised/value-head training over rollout records, not PPO.

The standalone neural benchmark is intentionally small and serial. Increase `--games` for strength checks. In the neural iteration command, increase `--evaluation-games` for per-iteration benchmark evidence; set it to `0` only for one-iteration smoke runs.

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
  --promotion-registry runs/promotions.json \
  --promotion-artifact-dir runs/promoted-checkpoints \
  --auto-promote \
  --audit-after-iteration \
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
  --promotion-registry runs/promotions.json \
  --promotion-artifact-dir runs/promoted-checkpoints \
  --auto-promote \
  --showdown-root /path/to/pokemon-showdown
```

With `--auto-promote`, each iteration evaluates the same promotion gate used by `eval_cli promote` after the benchmark is written. Passing checkpoints are recorded in `--promotion-registry`, copied into `--promotion-artifact-dir` when supplied, and become eligible historical opponents for later iterations in the same run. `--allow-missing-benchmark` bypasses the win-rate signal in this path too, so reserve it for smoke runs.

When `--initial-policy` points at a linear checkpoint, self-play retains that checkpoint as a static benchmark reference across later iterations. This keeps the original bootstrap checkpoint visible after the incumbent benchmark rotates from bootstrap-vs-candidate to previous-iteration-vs-candidate. Use `--benchmark-reference-policy` to add any other fixed reference checkpoints that should remain in every iteration benchmark. These references are informational by default: they stay visible in benchmark reports, but they do not add promotion-gate floors unless explicitly named as required benchmark opponents.

Inspect the run:

```bash
python -m pokezero.selfplay_cli report --run-dir runs/bootstrap-selfplay
```

Audit a run for regression-health checks that are cheap to run on CPU:

```bash
python -m pokezero.eval_cli audit runs/bootstrap-selfplay \
  --min-latest-benchmark-win-rate 0.55 \
  --min-latest-benchmark-games 50 \
  --max-latest-collection-capped-rate 0.10 \
  --max-latest-benchmark-capped-rate 0.10 \
  --max-latest-average-decision-rounds 200 \
  --max-latest-benchmark-average-decision-rounds 200 \
  --max-benchmark-win-rate-drop 0.05 \
  --max-consecutive-promotion-failures 1
```

The audit command reads linear or neural self-play manifests and does not run new games. It is intended for long CPU experiments where the latest checkpoint should be checked for benchmark availability, capped-game health, optional collection and benchmark average decision-round upper bounds, same-opponent regression from the previous best benchmark against each shared opponent, and repeated promotion failures before the run is treated as healthy. The average decision-round checks catch slow or stall-heavy runs; they do not by themselves detect degenerate-short games.

By default, the audit also requires the latest benchmark to retain fixed baseline opponents, currently `random-legal` and `simple-legal`, once they have appeared in prior benchmark evidence. This prevents a run from looking healthy after silently dropping fixed baselines, while still allowing incumbent or historical checkpoint opponents to rotate. Use `--allow-missing-benchmark-opponents` only when intentionally changing the benchmark set.

Use `--audit-after-iteration` on `selfplay_cli iterate` or `neural_cli iterate` to enforce a per-iteration version of that same audit after each completed iteration. The run writes the latest manifest first, then stops before starting the next iteration if any audit check fails. The per-iteration CLI defaults are intentionally looser than the standalone end-of-run audit for noisy early experiments: benchmark win-rate drop tolerance defaults to `0.15`, and consecutive promotion failures default to `3`. Prefix audit thresholds with `--audit-`, for example `--audit-min-latest-benchmark-games 50`, `--audit-max-latest-average-decision-rounds 200`, `--audit-max-latest-benchmark-average-decision-rounds 200`, or `--audit-require-latest-promotion`.

Compare cold-start, teacher-bootstrap, and neural iteration runs side by side:

```bash
python -m pokezero.eval_cli compare \
  runs/cold-selfplay \
  runs/bootstrap-selfplay \
  runs/neural-selfplay
```

The comparison report reads existing manifests and surfaces latest and best benchmark win rate, capped-game rates, collection and benchmark games-per-hour, latest process peak RSS high-water when recorded, average decision-round length, latest promotion or advancement state, and latest checkpoint paths. The RSS value is a platform process high-water mark, not phase-isolated memory attribution, and resumed runs may reset the process counter. Best-run labels require at least `--min-benchmark-games` benchmark games by default, and malformed or not-yet-started manifests are reported as row-level errors without hiding healthy runs. Use it to decide which run deserves deeper audit or benchmark expansion; do not treat validation fit as a strength signal.

Named evaluation profiles can be used instead of repeating every threshold flag:

```bash
python -m pokezero.eval_cli profiles
python -m pokezero.eval_cli audit runs/bootstrap-selfplay --profile long-run
```

Profiles provide defaults only. Explicit threshold flags and boolean requirement flags such as `--require-benchmark` or `--allow-missing-benchmark` still override profile values. Use `--profile smoke` for plumbing checks, `--profile default` for current guardrails, and `--profile long-run` for stricter provisional CPU run checks.

Gate a candidate before promotion:

```bash
python -m pokezero.eval_cli gate runs/bootstrap-selfplay \
  --min-benchmark-win-rate 0.55 \
  --min-benchmark-games 50 \
  --max-collection-capped-rate 0.10 \
  --max-benchmark-capped-rate 0.10
```

The gate is a configurable guardrail, not a final research threshold. It requires benchmark evidence by default, checks each candidate-vs-opponent benchmark row independently, enforces a minimum game count per opponent, checks collection and benchmark capped-game rates, and checks bootstrap teacher-degradation counters when present. Use `--json` for automation and `--allow-missing-benchmark` only for smoke runs; use `--require-benchmark` to tighten a permissive profile.

The same gate can use a named profile:

```bash
python -m pokezero.eval_cli gate runs/bootstrap-selfplay --profile long-run
```

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

The registry is the checkpoint-pool index for accepted policies: `selfplay_cli iterate --promotion-registry runs/promotions.json` uses promoted checkpoints as historical opponents instead of every raw prior iteration checkpoint. With `--auto-promote`, that pool is refreshed after each passing iteration during the run.

Preview the historical opponent pool that self-play would draw from before starting a long run:

```bash
python -m pokezero.eval_cli promotions \
  --registry runs/promotions.json \
  --opponent-pool-size 3
```

By default, the preview assumes the latest promoted checkpoint is the current collector and excludes it from the historical opponent slice, matching the steady-state auto-promotion loop. Add `--current-policy-spec linear:/path/to/current.json` to preview a different current collector. The report annotates each promotion entry with whether it is the latest checkpoint, part of the previewed opponent pool, and whether verification has checked checkpoint existence, checksums, loadability, and policy-id consistency.

Verify that recorded promoted checkpoints still resolve and match stored checksums before using the registry for a long run:

```bash
python -m pokezero.eval_cli promotions --registry runs/promotions.json --verify --verify-loadable
```

This check is CPU-only and read-only. Relative promoted checkpoint paths are resolved deterministically from the current working directory, the registry directory, and the promotion entry's manifest location. Self-play uses the same resolved registry policy specs after verification, so the registry audit and promoted-opponent selection agree even when a long run is launched from a different current directory. It fails when registry sequences are malformed, a promoted checkpoint path no longer resolves, an embedded gate result is not passing, a promoted policy spec cannot be loaded, the loaded policy id disagrees with registry metadata, or a stored checkpoint checksum no longer matches the file on disk. Add `--require-checksum` when every promoted entry is expected to come from a managed artifact copy with checksum metadata.

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
