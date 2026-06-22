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
- `cpu-readiness-report` can roll up core pilot readiness, long-run derived audit health, and promotion-registry opponent-pool readiness from existing artifacts without launching games.
- Source provenance is recorded in major run artifacts so dirty or unexpected code snapshots are visible during review.
- A read-only Gen 3 randbat belief sidecar and compact belief observation features exist, but the detailed sidecar state is not required for the current linear CPU loop.
- A real local smoke-scale `cpu-pilot-run` has passed against a local Showdown checkout with two seeded pilots, deterministic teacher scenario preflight, rollout-backed `status_pressure` branch gates, audit calibration, and calibrated audit replay. This proves the wrapper path works on local artifacts; it is still too thin to define long-run quality thresholds.

## What Is Left

The remaining work is less about wiring and more about making the loop empirically usable.

1. Run a stronger local CPU pilot suite for long-run threshold evidence.

   The first smoke-scale local pilot proved that the wrappers, local Showdown path, teacher bootstrap, linear self-play, audit calibration, and config replay work together outside unit tests. The next pilot should increase pilot count, games, benchmark games, and calibration sufficiency floors enough to produce a credible starter audit config.

2. Promote a calibrated audit config from stronger pilot evidence.

   The smoke-scale pilot did write and replay a calibrated config, but it used the minimum benchmark-game floor needed for plumbing validation. A reusable long-run config should come only after enough pilot manifests, benchmark iterations, and benchmark games are present. Thin smoke thresholds should not become the long-run policy.

3. Generate a current scripted-teacher bootstrap checkpoint.

   Run the teacher bootstrap path with strict defaults, held-out validation, baseline benchmarks, and source provenance. This checkpoint is the intended seed for the first meaningful local linear self-play runs.

4. Run a cold-start control and a teacher-bootstrap run.

   The CPU baseline should include at least one cold-start run and one teacher-bootstrap run under comparable audit settings. The comparison should use benchmark win rate, capped-game rate, average decision rounds, throughput, RSS high-water marks, and promotion outcomes. Validation fit remains imitation-health only.

5. Use promotion registry discipline for historical opponents.

   Long runs should use `--auto-promote`, a promotion registry, a managed artifact directory, and a required promoted-opponent-pool preflight once enough promoted checkpoints exist. Historical opponents should come from accepted checkpoints, not every raw iteration.

6. Validate scripted-teacher data quality with measurements.

   The teacher now has multiple tactical heuristics, but heuristic branches firing is not the same as better bootstrap data. The next useful evidence is teacher-vs-baseline and teacher-self-play benchmark deltas, branch coverage, degradation counters, and capped-game health.

7. Decide the first local success threshold.

   Before treating any CPU run as progress, define provisional minimums for benchmark game count, fixed-baseline win rate, incumbent win rate, capped-game rate, decision-round limits, throughput, RSS ceiling, and required promoted-opponent-pool size. These should come from pilot calibration plus manual judgment, not from arbitrary defaults.

8. Keep the neural path optional until the CPU baseline is trustworthy.

   The neural scaffold can run CPU smoke tests when `.[neural]` is installed, but it is not the blocking path for the local CPU proof of concept. PPO, GPU training, and distributed rollout orchestration remain later milestones.

## Suggested Next Tasks

The next implementation tasks should be chosen in this order unless a real pilot run exposes a more urgent failure.

1. Run a stronger `cpu-pilot-run` with higher evidence floors, then inspect it with `cpu-pilot-report` plus `cpu-readiness-report`.
2. Add missing preflight or audit checks discovered while scaling that pilot beyond the smoke profile.
3. Run and document a reproducible teacher-bootstrap experiment.
4. Run and document comparable cold-start and teacher-bootstrap linear self-play experiments.
5. Add focused scripted-teacher improvements only when they can be measured with deterministic scenarios and rollout-backed branch coverage.
6. Tighten long-run audit thresholds based on the stronger pilot and early long-run artifacts.

## Progress Updates

- Added `cpu-readiness-report` so the core pilot, long-run, and promotion readiness artifacts can be evaluated in one read-only command. This closes the immediate reporting gap for existing artifacts; the next task is to run a real local CPU pilot suite and use the report to identify any missing preflight or audit checks.
- Ran a real local smoke-scale CPU pilot suite at `runs/cpu-pilots-local-20260622-smoke-2` against `/Users/scott/workspace/pokerena/vendor/pokemon-showdown`. The suite passed in 138.6 seconds with two seeded pilots, deterministic teacher scenario preflights passing 13/13 scenarios per pilot, rollout-backed `status_pressure` branch gates passing with 13 aggregate observations, zero capped games in the nested smoke self-play runs, generated audit calibration, and calibrated audit replay. `cpu-pilot-report --require-ready --require-smoke-ready --require-calibration-run-count 2 --require-calibration-benchmark-iterations 4 --require-calibration-min-benchmark-games 1` passed. `cpu-readiness-report --pilot-summary ...` reports the pilot item as PASS while the overall checklist remains not ready because no long-run summary or promotion registry was supplied.

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
