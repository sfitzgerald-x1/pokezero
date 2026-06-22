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
- Continue expanding the deterministic scripted teacher for Gen 3 randbats beyond the initial metadata-backed version. It now covers team-status cure value, status-aware switch targets, low-HP active preservation, max-layer Spikes suppression, and Rapid Spin hazard clearing; richer hazard planning and matchup context remain future work.
- Collect teacher-vs-baseline and teacher-self-play trajectories through the normal rollout JSONL path.
- Train bootstrap checkpoints from teacher trajectories with held-out validation.
- Start self-play from the bootstrap checkpoint and compare against cold-start runs using the self-play report command.
- Benchmark each candidate against `random-legal`, `simple-legal`, historical self-play checkpoints, and the static bootstrap checkpoint.
- Track benchmark win rate, capped-game rate, validation fit, games per hour, average decision-round length, and best-effort process peak RSS high-water marks for both paths. Treat validation fit as imitation-health only.
- Use `python -m pokezero.eval_cli cpu-pilot-run ...` to run multiple seeded CPU smoke pilots, calibrate starting audit thresholds from their manifests, and immediately replay those thresholds against the same pilot suite before enforcing them on longer unattended experiments.
- Extend the normalized replay-to-rollout importer with a raw Showdown replay converter after a useful Gen 3 randbat replay corpus is identified.

## Supported Command Shape

Run a tiny CPU smoke validation before spending time on larger experiments:

```bash
./.venv/bin/python -m pokezero.eval_cli cpu-smoke-run \
  --run-root runs/cpu-smoke \
  --showdown-root /path/to/pokemon-showdown
```

This executes the teacher bootstrap, linear self-play, report, smoke audit, audit-calibration/profile, and calibrated audit-config replay steps sequentially, stopping on the first non-zero exit. Use a fresh `--run-root`; the command does not delete existing artifacts. The wrapper writes `RUN_ROOT/cpu-smoke-run-summary.json` with the executed recipe, git source metadata for the PokeZero package source that ran when available, per-step exit codes, timestamps, and final pass/fail status. It also writes `RUN_ROOT/smoke-audit-config.json` by default, then immediately runs `audit --audit-config` against the smoke run so the reusable audit-config path is exercised. Pass `--summary-path` to write the wrapper artifact somewhere else, and pass `--audit-config-path` to place the generated smoke audit config elsewhere. Teacher bootstrap, linear self-play, and neural self-play manifests also record the package source snapshot directly. A dirty source marker is a reproducibility warning; it does not include the uncommitted patch contents.

Inspect that wrapper summary later:

```bash
./.venv/bin/python -m pokezero.eval_cli cpu-smoke-report runs/cpu-smoke
```

The report command accepts either the run root or the summary JSON path. It exits `0` for a recorded passed run with all requested preflight evidence readable, exits `2` for recorded failed, running, unknown, or missing requested preflight-artifact states, and exits `1` when the summary cannot be read or has an unsupported schema. Shell automation can use any non-zero exit as a wrapper-level health check failure, or distinguish `1` as "no valid summary was available" versus `2` as "a valid summary recorded a non-passing run or incomplete requested preflight evidence." When teacher branch gates were requested, the report also summarizes `RUN_ROOT/teacher-branch-preflight.json`, including artifact availability, pass/fail state, failed branch checks, and observed teacher branch counts.

Inspect the generated commands without running them:

```bash
./.venv/bin/python -m pokezero.eval_cli cpu-smoke-plan \
  --run-root runs/cpu-smoke \
  --showdown-root /path/to/pokemon-showdown
```

Both commands use intentionally small counts. They are plumbing validation aids, not strength evidence; the generated smoke audit config proves the config path works but should not be reused as a long-run policy. By default the smoke recipe uses the Python interpreter running the CLI; pass `--python-binary` when another interpreter or virtualenv should run the commands. Use `cpu-smoke-plan --json` when another script should consume the recipe.

When changing scripted-teacher heuristics, add teacher branch gates to the smoke or pilot wrapper instead of running a separate manual preflight:

```bash
./.venv/bin/python -m pokezero.eval_cli cpu-smoke-run \
  --run-root runs/cpu-smoke \
  --showdown-root /path/to/pokemon-showdown \
  --require-teacher-branch status_pressure \
  --min-teacher-branch-count status_pressure=1
```

These flags insert a `teacher-benchmark` branch-coverage step before the teacher bootstrap step. `cpu-pilot-run` accepts the same flags and passes them through to each seeded smoke pilot.

Run a slightly broader CPU pilot suite when a single smoke run is not enough evidence to tune audit thresholds:

```bash
./.venv/bin/python -m pokezero.eval_cli cpu-pilot-run \
  --run-root runs/cpu-pilots \
  --showdown-root /path/to/pokemon-showdown \
  --pilot-count 2 \
  --seed-start 1 \
  --seed-stride 10000
```

This wrapper runs `cpu-smoke-run` repeatedly under `RUN_ROOT/pilot-0001`, `RUN_ROOT/pilot-0002`, and so on, using deterministic seed offsets for each pilot. After the pilots finish, it compares `RUN_ROOT/pilot-*/selfplay/manifest.json`, writes `RUN_ROOT/pilot-audit-config.json` through compare-time audit calibration with envelope aggregation and sufficiency requirements, then reruns `compare --audit-config --fail-on-audit` against the pilot manifests. Envelope aggregation is intentional here because the suite-level replay must keep every supplied pilot passable. That replay is a plumbing check for config loadability and wiring, not a held-out policy-quality signal; the generated config becomes meaningful when applied to later runs. The suite writes `RUN_ROOT/cpu-pilot-suite-summary.json` with the executed recipe, source metadata, per-step exit codes, timestamps, and final pass/fail status. It also persists the two compare JSON payloads as `RUN_ROOT/pilot-calibration-compare.json` and `RUN_ROOT/pilot-audit-replay.json`, so later reports can inspect the exact calibration suggestion and replay audit result without rerunning the suite.

For pilot suites, `--audit-config-path` controls the suite-level calibrated audit config. Each nested smoke pilot still writes its own `PILOT_ROOT/smoke-audit-config.json`. Pilot seed offsets must stay within the smoke recipe's seed band, so `(pilot-count - 1) * seed-stride` must be less than `1_000_000`; this prevents pilot collection seeds from colliding with validation, benchmark, preflight, self-play, or evaluation seed bands.

Inspect or preflight the pilot suite without rerunning games:

```bash
./.venv/bin/python -m pokezero.eval_cli cpu-pilot-plan \
  --run-root runs/cpu-pilots \
  --showdown-root /path/to/pokemon-showdown
./.venv/bin/python -m pokezero.eval_cli cpu-pilot-report runs/cpu-pilots
```

The pilot report surfaces the calibrated audit config path, persisted compare artifact paths, calibration sufficiency/write status, replay audit status, and a derived `audit_config_ready` verdict. It also rolls up each nested smoke pilot summary, including per-pilot smoke pass/fail state and any requested teacher-branch preflight counts; aggregate branch counts include only passing preflight artifacts. That verdict means only that the pilot suite passed, the calibration artifact says the audit config was written from sufficient pilot evidence, and the replay artifact says the generated config passed against the same pilots. It is a reuse/readiness check for the generated guardrail config, not a policy-strength claim. Use `cpu-pilot-report --json` when automation needs the same derived artifact and per-pilot smoke reports. By default, the report exit code still reflects the wrapper summary status for backward compatibility; add `--require-ready` when shell automation should also fail unless `audit_config_ready` is true, and add `--require-smoke-ready` when automation should also fail unless every discovered nested smoke pilot has a readable passing summary and every requested teacher-branch preflight passed. Add `--require-calibration-run-count`, `--require-calibration-benchmark-iterations`, and `--require-calibration-min-benchmark-games` when the report should also re-check the generated audit config's saved calibration metadata before treating it as reusable. Like the smoke wrapper, the pilot suite is still CPU plumbing and threshold-calibration evidence, not proof of policy strength. Increase `--pilot-count`, per-pilot game counts, and calibration sufficiency floors before treating the generated audit config as a long-run guardrail.

After a pilot suite is ready, generate the guarded long-run command from that same summary instead of hand-copying the audit config path:

```bash
./.venv/bin/python -m pokezero.eval_cli cpu-long-run-plan runs/cpu-pilots \
  --run-dir runs/linear-long-run \
  --initial-policy linear:runs/bootstrap/linear-bootstrap.json \
  --validation-data runs/bootstrap/validation-rollouts.jsonl \
  --iterations 20 \
  --games-per-iteration 100 \
  --evaluation-games 200 \
  --require-calibration-run-count 2 \
  --require-calibration-benchmark-iterations 4 \
  --require-calibration-min-benchmark-games 50
```

The long-run plan is read-only. It fails closed unless the pilot suite passed, the generated audit config is ready under the requested calibration floors, and the requested `--evaluation-games` can satisfy the audit config's benchmark-game floor during post-iteration audit. When ready, it emits a `selfplay_cli iterate` command wired with `--audit-after-iteration --audit-config <generated-config>`, `--auto-promote`, and managed promotion artifact paths under the requested long-run directory. The `--initial-policy` remains explicit because the pilot suite calibrates guardrails; it does not decide which checkpoint should seed a longer experiment.

Use the paired execution wrapper when the validated plan should actually launch and leave behind a wrapper artifact:

```bash
./.venv/bin/python -m pokezero.eval_cli cpu-long-run-run runs/cpu-pilots \
  --run-dir runs/linear-long-run \
  --initial-policy linear:runs/bootstrap/linear-bootstrap.json \
  --validation-data runs/bootstrap/validation-rollouts.jsonl \
  --iterations 20 \
  --games-per-iteration 100 \
  --evaluation-games 200 \
  --require-calibration-run-count 2 \
  --require-calibration-benchmark-iterations 4 \
  --require-calibration-min-benchmark-games 50
```

`cpu-long-run-run` builds the same readiness-checked plan, writes `RUN_DIR/cpu-long-run-run-summary.json` by default, and only invokes the self-play command when the plan is ready. On terminal outcomes it also records a best-effort `derived_run_report` in the wrapper summary, using the nested self-play manifest and the recorded runtime audit source when those artifacts are available. If readiness checks fail, it exits non-zero without launching self-play and still writes a failed summary containing the rejected plan and reasons. If the self-play process itself fails, the wrapper propagates that exit code and records the failed step.

The launcher defaults to `--profile long-run`, which intentionally requires enough evaluation games for the stricter promotion gate and uses the calibrated pilot audit config at runtime. Use `--profile smoke` only for cheap local rehearsal of the wrapper path; non-`long-run` profiles still require the pilot audit artifact for launch readiness, but run self-play with `--audit-profile <profile>` instead of the calibrated long-run audit config. Smoke-profile runs validate command wiring, not long-run policy quality.

Inspect the long-run wrapper summary without reading raw JSON:

```bash
./.venv/bin/python -m pokezero.eval_cli cpu-long-run-report runs/linear-long-run
```

The report accepts either the run directory or the summary JSON path. It exits `0` only for a recorded passed wrapper run, exits `2` for valid failed, running, or unknown wrapper summaries, and exits `1` when the summary cannot be read or has an unsupported schema. It uses the persisted `derived_run_report` snapshot when the wrapper summary has one, and falls back to resolving the nested self-play `manifest.json` from the recorded `run_dir` for older summaries. The derived report surfaces benchmark/capped-rate/decision-round health and failed audit checks. By default, the command exit code still reflects the wrapper summary status. Add `--require-derived-audit` when shell automation should also fail unless the selected derived report is readable and passing; in that mode, unreadable derived run artifacts return `1`, while a readable nested run with failing derived audit health returns `2`. Add `--refresh-derived-audit` when the report should ignore any persisted snapshot and recompute current audit health from the live nested manifest and current audit config/profile. Use `--json` when automation needs the full summary payload, resolved summary path, derived run report, and `derived_run_report_source`.

Compare several long-run wrapper summaries directly:

```bash
./.venv/bin/python -m pokezero.eval_cli cpu-long-run-compare \
  --summary-glob 'runs/linear-long-run-*/cpu-long-run-run-summary.json'
```

The compare command reads wrapper summaries, uses each summary's persisted `derived_run_report` when available, falls back to recomputing derived nested audit health for older summaries, and surfaces wrapper status, audit pass/fail, latest benchmark win rate, capped rates, decision-round metrics, RSS high-water, and load errors. It is a run-health comparison tool, not a standalone policy-strength claim. By default, it returns `0` when all requested summaries load even if some runs are non-passing; add `--fail-on-non-passing` when shell automation should return `2` for failed wrappers or failed derived audit health. Load errors return `1`. Add `--refresh-derived-audit` when every row should ignore persisted snapshots and recompute current derived health from each live nested manifest. Refresh mode marks rows non-passing when the live nested manifest or audit config is unavailable, so pair it with `--fail-on-non-passing` when that should fail a shell workflow. Use `--json` for automation.

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
  --showdown-root /path/to/pokemon-showdown \
  --teacher-policy 'scripted-teacher?allow_fallback=true&allow_unknown_moves=true' \
  --min-teacher-win-rate 0.55 \
  --max-capped-rate 0.10 \
  --fail-on-degraded-decisions \
  --require-teacher-branch damaging_move \
  --min-teacher-branch-count damaging_move=1 \
  --out runs/scripted-teacher-benchmark.json
```

Use this as a cheap quality check after changing scripted-teacher heuristics. It reports teacher fallback and unknown-move counters plus low-cardinality teacher branch counts and top teacher decision reasons alongside win rates, so heuristic-specific branches can be audited without reading raw rollout JSONL. It does not train a checkpoint or write a manifest. Optional threshold flags make it usable as a CPU preflight gate: it exits `2` when any requested win-rate, capped-rate, degraded-decision, or teacher-branch coverage check fails, while still writing the JSON report requested by `--out`. Use `--require-teacher-branch BRANCH` when a branch must appear at least once, and `--min-teacher-branch-count BRANCH=N` when a branch needs a minimum sample count. Unknown branch names fail explicitly so typos do not look like heuristic regressions.

The scripted teacher remains strict by default. With the default `scripted-teacher` policy, unresolved moves or missing metadata fail fast with exit `1` before a benchmark report is produced. Use `allow_fallback=true` and/or `allow_unknown_moves=true` only when the goal is to measure degraded decisions via `--fail-on-degraded-decisions`.

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

The text report includes benchmark health, fit metrics, capped-game counts, process RSS when recorded, source provenance when recorded, and the invocation/opponent-pool metadata used to launch or resume the run. Use `--json` when the full manifest is needed for deeper audit.

Inspect a neural self-play run without loading torch:

```bash
python -m pokezero.neural_cli report --run-dir runs/neural-selfplay
```

The neural report is read-only and summarizes the run manifest's current policy, latest checkpoint, source provenance, per-iteration blended benchmark win rate, incumbent win rate, advancement state, promotion state, and transformer training metrics. Use `--json` for the raw manifest.

Audit a run for regression-health checks that are cheap to run on CPU:

```bash
python -m pokezero.eval_cli audit runs/bootstrap-selfplay \
  --min-latest-benchmark-win-rate 0.55 \
  --min-latest-benchmark-games 50 \
  --max-latest-collection-capped-rate 0.10 \
  --max-latest-benchmark-capped-rate 0.10 \
  --max-latest-average-decision-rounds 200 \
  --max-latest-benchmark-average-decision-rounds 200 \
  --max-latest-process-peak-rss-mb 8192 \
  --max-benchmark-win-rate-drop 0.05 \
  --max-consecutive-promotion-failures 1
```

The audit command reads linear or neural self-play manifests and does not run new games. It is intended for long CPU experiments where the latest checkpoint should be checked for benchmark availability, capped-game health, optional collection and benchmark average decision-round upper bounds, optional process peak RSS ceilings, recorded promoted-pool preflight requirements, same-opponent regression from the previous best benchmark against each shared opponent, and repeated promotion failures before the run is treated as healthy. Audit output also surfaces recorded source provenance when the manifest has it, so a run's branch, head, and dirty marker can be checked alongside health status. The average decision-round checks catch slow or stall-heavy runs; they do not by themselves detect degenerate-short games. The process RSS value is a platform high-water mark over the current run process, not phase-isolated memory attribution. When the local platform or an older manifest does not expose RSS, the RSS ceiling check is skipped rather than treated as a run failure.

By default, the audit also requires the latest benchmark to retain fixed baseline opponents, currently `random-legal` and `simple-legal`, once they have appeared in prior benchmark evidence. This prevents a run from looking healthy after silently dropping fixed baselines, while still allowing incumbent or historical checkpoint opponents to rotate. Use `--allow-missing-benchmark-opponents` only when intentionally changing the benchmark set.

Use `audit-calibrate` on one or more pilot manifests to derive a starter audit config from observed CPU runs before enforcing stricter profiles. With one path, the command reports thresholds for that run. With multiple paths, the default `--aggregate-mode median` combines per-run calibrations with a median-style reducer so one noisy pilot does not silently loosen every threshold. Use `--aggregate-mode envelope` only when the intended output is a permissive floor/ceiling set that keeps every supplied pilot run passable under the requested margin:

```bash
python -m pokezero.eval_cli audit-calibrate runs/pilot-a runs/pilot-b --margin 0.10
python -m pokezero.eval_cli audit-calibrate --manifest-glob 'runs/pilot-*/manifest.json' --margin 0.10
python -m pokezero.eval_cli audit-calibrate runs/pilot-a runs/pilot-b --compare-profile long-run
python -m pokezero.eval_cli audit-calibrate runs/pilot-a runs/pilot-b --aggregate-mode envelope
python -m pokezero.eval_cli audit-calibrate runs/pilot-a runs/pilot-b \
  --require-run-count 2 \
  --require-benchmark-iterations 4 \
  --require-min-benchmark-games 50
```

Use `--manifest-glob` when pilot runs are stored under a shared root and should be discovered in sorted order instead of manually enumerated. The flag may be repeated and can be mixed with explicit paths; duplicate resolved manifest identities are ignored, including a run directory and its globbed `manifest.json`. Empty glob patterns emit a warning when another explicit path or glob supplies manifests, but fail when no manifests are selected. Use `--compare-profile smoke`, `--compare-profile default`, or `--compare-profile long-run` to report whether each pilot manifest would pass a named audit profile before copying thresholds into unattended runs; this per-run profile comparison is independent of the calibration aggregate mode. Add `--fail-on-profile` when that profile comparison should be enforceable in a shell preflight. Use `--require-run-count`, `--require-benchmark-iterations`, and `--require-min-benchmark-games` before copying calibration output into long unattended runs. The command still prints the suggested thresholds when a requirement fails, but exits non-zero and reports the sufficiency failure so thin pilot evidence is not silently accepted. Requiring benchmark iterations also fails if any contributing run had no benchmark iterations, because that aggregate would otherwise allow missing benchmarks.

Write calibrated thresholds into a reusable audit config after the sufficiency checks pass:

```bash
python -m pokezero.eval_cli audit-calibrate runs/pilot-a runs/pilot-b \
  --require-run-count 2 \
  --require-benchmark-iterations 4 \
  --require-min-benchmark-games 50 \
  --write-config runs/audit-configs/pilot-long-run.json
python -m pokezero.eval_cli audit runs/new-run --audit-config runs/audit-configs/pilot-long-run.json
```

The config file is versioned JSON and records the suggested audit thresholds plus source and calibration metadata. `--write-config` requires at least one sufficiency requirement, such as `--require-run-count`, `--require-benchmark-iterations`, or `--require-min-benchmark-games`, so thin pilot evidence is not accidentally turned into a reusable long-run policy. `audit`, `compare --audit-config`, and post-iteration self-play flags can consume it. Explicit threshold flags still override values from the config file.

Before applying a calibrated config to a longer unattended run, inspect and optionally replay it against the pilot manifests that produced it:

```bash
python -m pokezero.eval_cli audit-config-report runs/pilot-audit-config.json \
  --require-source \
  --require-calibration
python -m pokezero.eval_cli audit-config-report runs/pilot-audit-config.json \
  runs/pilot-*/selfplay/manifest.json
```

The report prints the config thresholds, source provenance, calibration metadata, and optional per-manifest preflight results. Use `--json` for automation. `--require-source` and `--require-calibration` make missing provenance or calibration metadata fail fast. Add `--require-calibration-run-count`, `--require-calibration-benchmark-iterations`, and `--require-calibration-min-benchmark-games` when automation should enforce that the saved config was calibrated from enough pilot evidence before reuse. Add `--require-preflight` when automation should also fail unless at least one manifest was supplied and every supplied manifest passed the config. When manifests are supplied, the command returns non-zero if any manifest fails the config or cannot be loaded.

Use `--audit-after-iteration` on `selfplay_cli iterate` or `neural_cli iterate` to enforce a per-iteration version of that same audit after each completed iteration. The run writes the latest manifest first, then stops before starting the next iteration if any audit check fails. The per-iteration CLI defaults are intentionally looser than the standalone end-of-run audit for noisy early experiments: benchmark win-rate drop tolerance defaults to `0.15`, and consecutive promotion failures default to `3`. Add `--audit-profile smoke`, `--audit-profile default`, `--audit-profile long-run`, or `--audit-config runs/audit-configs/pilot-long-run.json` when the run should enforce a reusable audit policy during iteration; explicit `--audit-*` thresholds and boolean requirement flags still override the selected profile or config file. Because `--evaluation-games` is per benchmark matchup while audit benchmark-game thresholds are aggregate counts, the CLI rejects benchmark-required audit configs whose requested evaluation games cannot satisfy the configured audit floor. Prefix audit thresholds with `--audit-`, for example `--audit-min-latest-benchmark-games 50`, `--audit-max-latest-average-decision-rounds 200`, `--audit-max-latest-benchmark-average-decision-rounds 200`, `--audit-max-latest-process-peak-rss-mb 8192`, `--audit-require-benchmark`, `--audit-allow-missing-benchmark`, `--audit-require-benchmark-opponents`, `--audit-allow-missing-benchmark-opponents`, `--audit-require-latest-promotion`, or `--audit-allow-missing-latest-promotion`.

Compare cold-start, teacher-bootstrap, and neural iteration runs side by side:

```bash
python -m pokezero.eval_cli compare \
  runs/cold-selfplay \
  runs/bootstrap-selfplay \
  runs/neural-selfplay
python -m pokezero.eval_cli compare --manifest-glob 'runs/pilot-*/manifest.json'
```

The comparison report reads existing manifests and surfaces latest and best benchmark win rate, capped-game rates, collection and benchmark games-per-hour, latest process peak RSS high-water when recorded, average decision-round length, latest promotion or advancement state, latest checkpoint paths, and recorded source provenance. The RSS value is a platform process high-water mark, not phase-isolated memory attribution, and resumed runs may reset the process counter. Best-run labels require at least `--min-benchmark-games` benchmark games by default, and malformed or not-yet-started manifests are reported as row-level errors without hiding healthy runs. Use it to decide which run deserves deeper audit or benchmark expansion; do not treat validation fit as a strength signal.

Add `--audit-profile smoke`, `--audit-profile default`, or `--audit-profile long-run` to include per-run audit pass/fail status and failed audit checks in the same comparison output. When an audit profile is supplied, best-run labels ignore audit-failing rows so a run is not highlighted as best while failing the selected health profile. The compare command remains read-only and does not run new games; audit failures are shown as row health signals unless `--fail-on-audit` is supplied or a manifest itself cannot be loaded.

Add `--suggest-audit-calibration` when comparing pilot runs to include starter audit thresholds derived from the same valid manifests. This reuses the `audit-calibrate` logic inline with comparison output, so malformed comparison rows remain visible but are excluded from calibration suggestions:

```bash
python -m pokezero.eval_cli compare \
  runs/cold-selfplay \
  runs/bootstrap-selfplay \
  runs/neural-selfplay \
  --suggest-audit-calibration
```

The compare command still returns non-zero when any manifest cannot be loaded. In that case, inspect the reported errors before consuming a printed calibration suggestion from the remaining valid runs.

For compare-time calibration, use `--calibration-require-run-count`, `--calibration-require-benchmark-iterations`, and `--calibration-require-min-benchmark-games` to make the comparison fail when too few valid compared runs, benchmark iterations, or benchmark games contributed to the suggested thresholds.

Add `--write-audit-config runs/audit-configs/pilot-comparison.json` when a passing comparison calibration should produce the reusable versioned audit config directly. Like `audit-calibrate --write-config`, compare-time config writing requires `--suggest-audit-calibration` plus at least one calibration sufficiency requirement, and it refuses to write when any compared manifest failed to load or when sufficiency checks fail.

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

The registry is append-only by default and embeds the full gate result for each accepted checkpoint. Recording holds a per-registry file lock across the registry reload, duplicate check, artifact copy, sequence assignment, and JSON write, so same-host concurrent auto-promotion attempts cannot reuse sequence numbers or clobber each other. This serializes registry recording, not the gate decision itself: if multiple independent runs evaluate candidates at the same time, they may still have evaluated against the same previous incumbent before the recording lock is acquired. The lock relies on local `fcntl` file-lock behavior and should not be treated as a distributed lock for shared network filesystems. With `--artifact-dir`, promotion copies the accepted single-file checkpoint into a stable artifact directory, records that managed copy as the checkpoint used by later self-play, records a SHA-256 checksum, and preserves the original source checkpoint path for audit. The embedded gate result still reflects the source manifest and source checkpoint that were evaluated. Without `--artifact-dir`, the registry stores references to existing checkpoint files.

The registry is the checkpoint-pool index for accepted policies: `selfplay_cli iterate --promotion-registry runs/promotions.json` uses promoted checkpoints as historical opponents instead of every raw prior iteration checkpoint. With `--auto-promote`, that pool is refreshed after each passing iteration during the run.

Add `--require-promoted-opponent-pool-size N` to `selfplay_cli iterate` or `neural_cli iterate` when launching a long run from a promotion registry, including the registry used by `--auto-promote`. The run fails before rollout collection unless at least `N` promoted historical opponents are selectable after current-policy exclusion and `--max-historical-opponents` capping. The required size cannot exceed `--max-historical-opponents`.

Preview the historical opponent pool that self-play would draw from before starting a long run:

```bash
python -m pokezero.eval_cli promotions \
  --registry runs/promotions.json \
  --opponent-pool-size 3 \
  --require-opponent-pool-size 3 \
  --verify \
  --verify-opponent-pool-only \
  --lifecycle \
  --retention-plan \
  --write-opponent-pool runs/opponent-pool-snapshot.json
```

By default, the preview assumes the latest promoted checkpoint is the current collector and excludes it from the historical opponent slice, matching the steady-state auto-promotion loop. Add `--current-policy-spec linear:/path/to/current.json` to preview a different current collector. Add `--require-opponent-pool-size N` when using the command as a long-run preflight; it exits non-zero if fewer than `N` promoted historical opponents appear in the capped preview after current-policy exclusion. The requirement cannot exceed `--opponent-pool-size`. Use `--verify` or `--verify --verify-loadable` when the preflight must also prove promoted checkpoints exist, match checksums, and can be loaded by the runtime. Add `--verify-opponent-pool-only` when the long-run preflight should fail only for broken checkpoints in the selected opponent pool, any registry entry excluded as the current collector, or registry-level integrity failures, while still annotating stale entry-specific failures elsewhere in the registry. An external `--current-policy-spec` that is not a registry entry is used for pool exclusion but is not itself verified by registry verification. The report annotates each promotion entry with whether it is the latest checkpoint, part of the previewed opponent pool, and whether verification has checked checkpoint existence, checksums, loadability, and policy-id consistency.

When an opponent pool is previewed, each entry also includes an `opponent_pool_status` such as `selected`, `unselectable`, `excluded_current_policy`, or `available_outside_requested_size`. Add `--lifecycle` to include compact counts for latest, selected opponent-pool, unhealthy selected opponent-pool, selection-eligible, unselectable, excluded-current, stale-available, entry-level failed-verification, and registry-level failed-verification entries. The lifecycle block also records whether an opponent pool was requested, so `--lifecycle` without `--opponent-pool-size` is distinguishable from a clean pool preflight. Use these statuses and lifecycle counts to diagnose why a required pool is undersized before starting a long run.

Add `--retention-plan` with `--opponent-pool-size` to print a non-destructive cleanup preview. The plan marks selected opponent-pool entries, the assumed current collector, and the latest promotion as `retain`; stale entries outside the requested opponent-pool window are marked `verify_before_cleanup` until full verification has passed. A stale entry becomes `cleanup_candidate` only when per-entry verification is `pass` and registry-level verification has no structural failures. Broken stale entries, structurally broken registries, and partially verified stale entries are marked for review or further verification rather than cleanup.

To act on verified cleanup candidates, add `--apply-retention-plan`. This still defaults to a dry run and reports what would be archived:

```bash
python -m pokezero.eval_cli promotions \
  --registry runs/promotions.json \
  --opponent-pool-size 3 \
  --retention-plan \
  --apply-retention-plan \
  --verify \
  --verify-loadable \
  --verify-opponent-pool-only
```

Apply the archive only after inspecting the dry-run output:

```bash
python -m pokezero.eval_cli promotions \
  --registry runs/promotions.json \
  --opponent-pool-size 3 \
  --retention-plan \
  --apply-retention-plan \
  --retention-apply-confirm archive \
  --retention-archive-dir runs/retention-archive \
  --verify \
  --verify-loadable \
  --verify-opponent-pool-only
```

The apply command only moves entries already marked `cleanup_candidate`, only for managed promoted artifact copies, and leaves source run checkpoints untouched. Confirmed archive requires the same promotions preflight to pass before any file is moved. It archives the stale artifact and rewrites that promotion entry's checkpoint path to the archive location so registry verification and historical auditability continue to work. Already archived entries are retained on later retention-plan runs instead of being re-archived. It is not a permanent deletion policy; remove or compact archive directories separately only after deciding those checkpoints are no longer needed.

Use `--write-opponent-pool` to save a compact, versioned snapshot of the selected policy specs, selected promotion entries, current-policy exclusion, size requirement, and verification/preflight status. The snapshot is written even when the preflight exits non-zero, so failed long-run launch checks leave behind the exact pool state that was rejected. Snapshots include `generated_at`, so compare `policy_specs` and selected entries rather than whole-file equality when checking whether the selected pool changed.

Verify that recorded promoted checkpoints still resolve and match stored checksums before using the registry for a long run:

```bash
python -m pokezero.eval_cli promotions --registry runs/promotions.json --verify --verify-loadable
```

This check is CPU-only and read-only. Relative promoted checkpoint paths are resolved deterministically from the registry directory and the promotion entry's manifest location, with the current working directory kept only as a compatibility fallback. Self-play uses the same resolved registry policy specs after verification, so the registry audit and promoted-opponent selection agree even when later code loads policies after a working-directory change. It fails when registry sequences are malformed, a promoted checkpoint path no longer resolves, an embedded gate result is not passing, a promoted policy spec cannot be loaded, the loaded policy id disagrees with registry metadata, or a stored checkpoint checksum no longer matches the file on disk. Add `--require-checksum` when every promoted entry is expected to come from a managed artifact copy with checksum metadata.

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
