# CPU Self-Play Roadmap

This document summarizes the current state of the local CPU-based self-play proof of concept and the remaining work needed before it should be treated as an experiment loop rather than scaffolding.

## Goal

The near-term goal is a local, CPU-only Gen 3 random battle loop that can:

- generate teacher-bootstrap and self-play data through the local Showdown harness
- train and resume linear baseline policies without GPU dependencies
- benchmark candidates against fixed baselines and promoted historical checkpoints
- gate checkpoint advancement with auditable promotion criteria
- stop bad long-running jobs through post-iteration audits
- compare completed runs with enough metadata to decide what to try next

This goal is narrower than the eventual transformer/PPO direction. The CPU loop is successful when it can produce reproducible, benchmarked runs that are safe to leave running locally and informative enough to compare.

## Current State

The CPU loop is now mostly wired end to end.

Implemented:

- The local Showdown BattleStream environment can run Gen 3 random battles and expose stable player-relative observations for either side.
- The rollout path records JSONL trajectories with legal masks, selected actions, opponent-action labels, terminal outcomes, capped-game markers, policy identifiers, and collection metrics.
- Dataset loading supports temporal windows and terminal-winner-derived returns, so winning actions are reinforced even when the immediate step reward is zero.
- A dependency-free linear softmax policy can train from behavior-cloning or reward-weighted objectives and save/load versioned checkpoints.
- A scripted Gen 3 randbat teacher exists for bootstrap data. It uses local Showdown dex metadata and first-pass context heuristics for damage preference, status pressure, team-status cures, low-HP preservation switches, status-aware switch penalties, recovery/setup, Spikes, and Rapid Spin.
- Teacher bootstrap can collect train and held-out validation rollouts, run strict-teacher preflight, train a linear checkpoint, benchmark it, and write a manifest.
- Linear self-play can run multiple iterations from random, simple, or checkpoint initial policies with current-policy-only training data, frozen historical opponents, checkpoint warm starts, resume support, validation metrics, and parallel collection workers.
- Neural scaffolding exists behind the optional `neural` extra, including offline transformer training, neural checkpoint benchmarking, and a first neural iteration loop. This is not required for the CPU baseline milestone.
- Promotion gates check benchmark win rates, minimum game counts, incumbent comparisons, capped-game rates, teacher-degradation counters, and Wilson lower-bound no-regression criteria.
- The promotion registry records accepted checkpoints, can copy managed artifacts, verifies registry/checkpoint integrity, feeds promoted opponent pools back into self-play, previews pool selection, and supports recoverable retention archiving.
- Run audits can check latest benchmark health, capped-game rates, same-opponent regressions, repeated promotion failures, decision-round length, missing benchmark opponents, and best-effort process RSS high-water marks.
- CPU smoke, pilot, and long-run wrappers can generate plans, execute guarded runs, persist wrapper summaries, calibrate audit configs from pilot evidence, replay those configs, launch readiness-checked long runs, and report or compare run health from summaries.
- CPU smoke and pilot execution wrappers now reject non-fresh run roots before launching nested work, so accidental reruns do not overwrite wrapper summaries or collide later with existing bootstrap/self-play artifacts.
- CPU pilot planning and execution now reject calibration benchmark-game floors that the selected `--evaluation-games` cannot guarantee before launching nested work.
- CPU long-run planning can decouple the promotion gate profile from the post-iteration audit source, so a feasible smoke/default gate profile can still enforce the stronger pilot-derived or summary-derived audit config during the run.
- `cpu-readiness-report` can roll up core pilot readiness, long-run derived audit health, and promotion-registry opponent-pool readiness from existing artifacts without launching games.
- Source provenance is recorded in major run artifacts so dirty or unexpected code snapshots are visible during review.
- A read-only Gen 3 randbat belief sidecar and compact belief observation features exist, but the detailed sidecar state is not required for the current linear CPU loop.
- A real local smoke-scale `cpu-pilot-run` has passed against a local Showdown checkout with two seeded pilots, deterministic teacher scenario preflight, rollout-backed `status_pressure` branch gates, audit calibration, and calibrated audit replay. This proves the wrapper path works on local artifacts; it is still too thin to define long-run quality thresholds.
- A stronger local `cpu-pilot-run` has passed with two seeded pilots, six total self-play benchmark iterations, a 50-game calibration sufficiency floor, calibrated audit replay, and `cpu-readiness-report` marking `pilot_suite_ready` as PASS.
- A small standalone scripted-teacher bootstrap checkpoint exists with held-out validation data, source provenance, and fixed-baseline benchmarks against random-legal, simple-legal, and scripted-teacher opponents.
- A first matched-configuration local cold-start versus teacher-bootstrap comparison has completed under the smoke audit profile. It validates the run/report/compare/readiness path for both arms and provides early relative evidence, but it is not yet a calibrated long-run quality threshold.
- A first expanded comparison has exercised `--runtime-audit-config` with summary-derived envelope guardrails. Both arms launched correctly, wrote wrapper summaries, and failed closed under the provisional audit config, which is useful negative evidence for the next calibration pass.

## What Is Left

The remaining work is less about wiring and more about making the loop empirically usable.

1. Recalibrate provisional guardrails after the first expanded comparison failed closed.

   The summary-based envelope config was useful because it stopped runs that violated explicit health gates, but it was calibrated from only two smoke-profile summaries. On the expanded run, cold-start failed after iteration 1 on average decision-round ceilings, while teacher-bootstrap reached iteration 2 and failed on same-opponent benchmark regression. The next guardrail pass should separate "hard stop" thresholds from "diagnostic warning" thresholds until there is more evidence, especially for decision-round ceilings and same-opponent regression. Validation fit remains imitation-health only.

2. Use promotion registry discipline for historical opponents.

   Long runs should use `--auto-promote`, a promotion registry, a managed artifact directory, and a required promoted-opponent-pool preflight once enough promoted checkpoints exist. Historical opponents should come from accepted checkpoints, not every raw iteration.

3. Validate scripted-teacher data quality with measurements.

   The teacher now has multiple tactical heuristics, but heuristic branches firing is not the same as better bootstrap data. The next useful evidence is teacher-vs-baseline and teacher-self-play benchmark deltas, branch coverage, degradation counters, and capped-game health.

4. Decide the first local success threshold.

   Before treating any CPU run as progress, define provisional minimums for benchmark game count, fixed-baseline win rate, incumbent win rate, capped-game rate, decision-round limits, throughput, RSS ceiling, and required promoted-opponent-pool size. These should come from pilot calibration plus manual judgment, not from arbitrary defaults.

5. Keep the neural path optional until the CPU baseline is trustworthy.

   The neural scaffold can run CPU smoke tests when `.[neural]` is installed, but it is not the blocking path for the local CPU proof of concept. PPO, GPU training, and distributed rollout orchestration remain later milestones.

## Suggested Next Tasks

The next implementation tasks should be chosen in this order unless a real pilot run exposes a more urgent failure.

1. Recalibrate or split the provisional runtime audit thresholds using the expanded failed summaries, keeping benchmark/capped/RSS checks strict while deciding whether decision-round and same-opponent regression should be hard stops this early.
2. Rerun the expanded cold-start versus teacher-bootstrap comparison under the adjusted guardrails.
3. Add focused scripted-teacher improvements only when they can be measured with deterministic scenarios and rollout-backed branch coverage.

## Progress Updates

- Added `cpu-readiness-report` so the core pilot, long-run, and promotion readiness artifacts can be evaluated in one read-only command. This closed the immediate reporting gap for existing artifacts and made the first local pilot inspection straightforward.
- Ran a real local smoke-scale CPU pilot suite at `runs/cpu-pilots-local-20260622-smoke-2` against `/Users/scott/workspace/pokerena/vendor/pokemon-showdown`. The suite passed in 138.6 seconds with two seeded pilots, deterministic teacher scenario preflights passing 13/13 scenarios per pilot, rollout-backed `status_pressure` branch gates passing with 13 aggregate observations, zero capped games in the nested smoke self-play runs, generated audit calibration, and calibrated audit replay. `cpu-pilot-report --require-ready --require-smoke-ready --require-calibration-run-count 2 --require-calibration-benchmark-iterations 4 --require-calibration-min-benchmark-games 1` passed. `cpu-readiness-report --pilot-summary ...` reports the pilot item as PASS while the overall checklist remains not ready because no long-run summary or promotion registry was supplied.
- The local pilot validation exposed a stale-run-root failure mode when rerunning into an existing ignored artifact directory. The smoke and pilot execution wrappers now fail fast on non-fresh run roots before starting child commands.
- Ran a stronger local pilot attempt at `runs/cpu-pilots-local-20260622-strong-1` with two seeded pilots, three self-play iterations per pilot, stricter teacher branch coverage, and a 50-game calibration floor. Both nested pilots completed successfully, but suite-level calibration failed after about 711 seconds because the selected `--evaluation-games 8` produced a 48-game observed floor. A manual compare at a 48-game floor succeeded and wrote a provisional config, but that artifact should not become the long-run config. The failure is now captured as a preflight validation gap: pilot plan/run reject calibration floors that cannot be guaranteed by the selected evaluation-game count before launching nested work. For a 50-game floor the current guaranteed recommendation is `--evaluation-games 13`; this is intentionally conservative because it relies on the minimum four post-iteration benchmark matchups rather than incidental extra reference matchups.
- Reran the stronger local pilot at `runs/cpu-pilots-local-20260622-strong-2` with `--evaluation-games 13`, which guarantees 52 benchmark games for the requested 50-game calibration floor. The suite passed in 976.5 seconds with two seeded pilots, six total self-play benchmark iterations, deterministic teacher scenario preflights passing 13/13 scenarios per pilot, rollout-backed `status_pressure` branch gates passing with 37 aggregate observations, zero capped games, generated audit calibration, and calibrated audit replay. `cpu-pilot-report --require-ready --require-smoke-ready --require-calibration-run-count 2 --require-calibration-benchmark-iterations 6 --require-calibration-min-benchmark-games 50` passed. The generated pilot audit config has `min_latest_benchmark_games: 78`, `min_latest_benchmark_win_rate: 0.475962`, and recommends `--evaluation-games 20` for post-iteration audit use. At that point, `cpu-readiness-report --pilot-summary ...` reported `pilot_suite_ready` as PASS while the overall checklist remained blocked on the long-run summary and promotion registry inputs.
- Ran a standalone scripted-teacher bootstrap at `runs/scripted-teacher-bootstrap-local-20260622-medium-1` with 64 train games, 16 held-out validation games, 16 benchmark games per direction, four workers, `window-size 4`, `feature-count 131072`, deterministic seed bands, and local Showdown. It wrote `linear-bootstrap.json`, train/validation rollout JSONL, and `manifest.json` with source provenance. The 96-game fixed-baseline benchmark had zero capped games and measured 32-game head-to-head aggregate linear-bootstrap win rates of 0.8125 vs random-legal, 0.65625 vs simple-legal, and 0.15625 vs scripted-teacher. Held-out teacher imitation accuracy was 0.3767; this remains an imitation-health metric, not a policy-strength signal.
- Ran a matched-configuration local smoke-profile comparison at `runs/cpu-comparison-local-20260622` using two iterations, 16 collection games per iteration, four workers, `--evaluation-games 20`, `window-size 4`, `feature-count 131072`, and shared seed bands. The cold-start arm completed in 426.8 seconds with two recorded promotions, latest benchmark win rate 0.5583 over 120 latest benchmark games, latest benchmark capped rate 0.0167, latest average decision rounds 68.19, benchmark average decision rounds 68.60, and peak RSS 443.11 MB. The teacher-bootstrap arm started from `runs/scripted-teacher-bootstrap-local-20260622-medium-1/linear-bootstrap.json`, completed in 439.6 seconds with two recorded promotions, latest benchmark win rate 0.5875 over 160 latest benchmark games, zero latest benchmark capped games, latest average decision rounds 60.00, benchmark average decision rounds 58.89, and peak RSS 578.66 MB. The aggregate benchmark win rates are not directly comparable because the teacher-bootstrap arm also benchmarked against the static `linear-bootstrap` reference, producing four opponent groups versus three for cold-start; on the shared `random-legal` and `simple-legal` baselines, teacher-bootstrap scored 0.80 and 0.60 versus cold-start at 0.625 and 0.50. `cpu-long-run-report --require-derived-audit`, `cpu-long-run-compare --fail-on-non-passing`, and `cpu-readiness-report --require-ready` all passed for these artifacts. Because the runtime audit source was the smoke profile rather than the pilot-derived audit config, this is path validation plus early relative evidence, not final long-run strength evidence.
- Calibrated provisional long-run thresholds from the two completed comparison summaries with `cpu-long-run-calibrate --aggregate-mode envelope --require-run-count 2 --require-benchmark-iterations 2 --require-min-benchmark-games 100`. The envelope config is intentionally conservative while evidence is thin: `min_latest_benchmark_win_rate: 0.5025`, `min_latest_benchmark_games: 120`, capped-rate ceilings of 0.10 for collection and benchmark, `max_latest_average_decision_rounds: 75.00625`, `max_latest_benchmark_average_decision_rounds: 75.46`, `max_latest_process_peak_rss_mb: 636.521875`, `max_benchmark_win_rate_drop: 0.05`, `max_consecutive_promotion_failures: 1`, benchmark required, fixed-baseline opponent coverage required, and latest promotion optional. The generated post-iteration recommendation is `--evaluation-games 30`. Median calibration was not selected because it would set an RSS ceiling below the teacher-bootstrap observed peak and would not keep both early arms passable.
- Ran the first expanded comparison at `runs/cpu-comparison-local-20260623-expanded` with two requested iterations, 32 collection games per iteration, four workers, `--evaluation-games 30`, `window-size 4`, `feature-count 131072`, shared seed bands, `--profile smoke`, and `--runtime-audit-config runs/cpu-comparison-local-20260622/envelope-audit-config.json`. This verified the summary-derived runtime-audit path in a real wrapper launch, but both arms failed closed under the provisional guardrails. The cold-start arm stopped after iteration 1 with latest benchmark win rate 0.5667 over 120 games, zero capped games, peak RSS 315.78 MB, and failed checks `latest_average_decision_rounds` at 80.71875 versus 75.00625 and `latest_benchmark_average_decision_rounds` at 75.8583 versus 75.46. The teacher-bootstrap arm reached iteration 2 with latest benchmark win rate 0.525 over 240 games, best benchmark win rate 0.6056, zero capped games, latest average decision rounds 66.28125, latest benchmark average decision rounds 62.925, peak RSS 619.53 MB, and failed only `benchmark_win_rate_drop_by_opponent` at 0.10 versus 0.05. Teacher-bootstrap iteration 2 still promoted under the smoke promotion gate, and `cpu-readiness-report` confirmed the promotion registry pool was readable, but the wrapper correctly reported the long-run derived audit as not ready.

## Out Of Scope For This Milestone

- GPU training.
- PPO-style actor-critic updates.
- Distributed self-play orchestration.
- Raw Showdown replay corpus discovery and conversion.
- Treating validation accuracy as a policy-strength signal.
- Permanent deletion of old checkpoint artifacts.

## Readiness Checklist

The local CPU loop is ready to be used as a serious prototype when all of these are true:

- A strict teacher scenario preflight passes.
- A rollout-backed teacher benchmark passes required branch and degradation gates.
- A `cpu-pilot-run` passes with at least two seeded pilots and writes a calibrated audit config from sufficient evidence.
- `cpu-pilot-report --require-ready --require-smoke-ready` passes under the selected sufficiency floors.
- A teacher-bootstrap checkpoint exists with held-out validation, fixed-baseline benchmarks, and clean or intentionally dirty source provenance.
- A guarded `cpu-long-run-run` launches from the ready pilot artifact and completes without post-iteration audit failure.
- `cpu-long-run-report --require-derived-audit` passes for the completed long run.
- `cpu-long-run-compare --fail-on-non-passing` can compare cold-start and teacher-bootstrap runs without load errors.
- A promotion registry preflight verifies the selected promoted opponent pool before any longer unattended run relies on it.
- The selected non-smoke long-run thresholds are explicit enough that a passing run means more than plumbing health.
