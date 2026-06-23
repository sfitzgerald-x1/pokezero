# Goal: Meaningful CPU Progress Past `simple-legal`

This is a self-contained goal specification to be handed to the autonomous `/goal` loop. It
defines the objective, the definition of done, the operating procedure, and the environment.
Follow `docs/autonomous_task_loop.md` for the per-task mechanics; this document is the goal
and the constraints layered on top of it.

## Goal

Produce a **CPU-trained PokeZero policy that meaningfully beats the `simple-legal` strategy**
in Gen 3 random battles, demonstrated with reproducible shared-opponent benchmark evidence —
not merely a healthy run that fails to advance a stronger checkpoint.

The work must be reachable on CPU only. GPU training, PPO at scale, distributed rollout
orchestration, and replay-corpus conversion remain out of scope for this goal.

## Definition Of Done

All of the following must hold, with artifacts committed as evidence (benchmark JSON paths
recorded in `docs/cpu_self_play_roadmap.md`):

1. **Baseline established.** A shared-opponent mirrored benchmark records the current
   strongest reference (`expanded-teacher-bootstrap-linear-iter-0001`) against `simple-legal`
   and `random-legal` at >= 120 games per orientation with zero/low capped games. This fixes
   the concrete bar that "meaningful progress" must clear.
2. **A candidate clearly beats `simple-legal`.** A CPU-trained checkpoint beats `simple-legal`
   in a mirrored shared-opponent benchmark with a **Wilson 95% lower bound >= 0.55** over
   **>= 240 total games**, with capped-game rate <= 0.05.
3. **Reproducible.** The result holds across **two independent seed bands** (or one 480-game
   mirrored row with a clear point-estimate lead and clean capped-game health).
4. **No regression sanity.** The same checkpoint still beats `random-legal` decisively, and
   does not fail runtime-health audit checks (RSS, capped rates, benchmark coverage).
5. **Honest reporting.** No strength claim is made from wrapper aggregate metrics alone;
   every claim is backed by a targeted shared-opponent benchmark.

If reaching bar (2) is shown by evidence to be infeasible for the linear policy class (the
roadmap already indicates linear sits near `simple-legal`), the goal extends to standing up
the first neural Stage-1 path from `docs/learning_architecture_exploration.md` far enough to
clear the bar on CPU, or to reporting a clear, evidence-backed blocker.

## Roles

- **Executor — Claude Code (this agent).** Selects tasks, implements, tests, opens PRs,
  applies review findings, merges per the policy below.
- **Reviewer — Codex (GPT-5.5, xhigh).** Adversarial, read-only review of every code PR.

## Operating Environment

- **Executor working directory:** `/Users/scott/workspace/agents/pokezero-agent` (isolated
  git worktree). Branch from `origin/main` with `scott/` names. Do not touch the primary
  checkout at `/Users/scott/workspace/pokezero` or other in-progress branches/worktrees.
- **Python env:** `uv sync` in the executor worktree.
- **Showdown checkout:** `/Users/scott/workspace/pokerena/vendor/pokemon-showdown` (pass as
  `--showdown-root`).
- **Reference checkpoints (local, not in git):** under the primary checkout's `runs/`, e.g.
  `runs/cpu-comparison-local-20260623-warning-audit/teacher-bootstrap/promoted-checkpoints/000001-expanded-teacher-bootstrap-linear-iter-0001.json`
  and `runs/scripted-teacher-bootstrap-local-20260622-medium-1/linear-bootstrap.json`.
  Reference them by absolute path; keep generated run artifacts out of git.

## Reviewer Command

Codex is a desktop app; use the absolute path (it is not on the executor's PATH):

```bash
/Applications/Codex.app/Contents/Resources/codex exec review --base main \
  -m gpt-5.5 -c 'model_reasoning_effort="xhigh"' \
  "$(cat review-prompt.txt)"
```

`exec` runs with approval `never` automatically and review is read-only; do not pass
`--ask-for-approval`. Use the prompt template in `docs/autonomous_task_loop.md`. Run with a
15-minute timeout; on failure, capture stderr and fall back to an independent adversarial
Claude review so the loop is not blocked.

## Merge Policy

- Run **serially**, one task at a time.
- Under this `/goal`, merge a **code PR** once it passes local verification **and** Codex
  signs off (no unresolved blocking findings) — no per-PR user approval needed.
- **Documentation-only PRs** may be merged without a Codex review.
- Serialize merges: merge, re-sync `origin/main`, write the roadmap progress entry, then take
  the next task.

## Task Order

1. **Baseline measurement** (Definition of Done #1) — measurement-only, docs PR.
2. **Cheapest credible attempts to clear the bar with the existing policy class** — targeted
   bootstrap-recipe or training changes, each validated by a shared-opponent benchmark, not
   wrapper aggregates. Same-teacher/same-recipe reruns are variance checks, not candidates.
3. **If linear is shown to be at its ceiling**, begin neural Stage-1 (entity-token
   transformer + model-free RL, CPU proof-of-life) per
   `docs/learning_architecture_exploration.md`, scaling only as far as CPU allows.
4. **Confirm and reproduce** any candidate that clears the bar (#2/#3 of Definition of Done).

## Constraints And Scope

- CPU-only. Stop and report if the next meaningful step requires GPU.
- Keep generated artifacts, local runs, and credentials out of git.
- Do not claim improvement without benchmark/incumbent/capped-rate evidence.
- Treat validation/imitation accuracy as health only, never as a strength signal.

## Stop Conditions

Stop and report (do not continue automatically) when: the next meaningful task requires GPU;
a required local prerequisite is missing; the reviewer path is unavailable for a code PR;
Codex (and the Claude fallback) find a blocking issue that cannot be safely fixed in the same
PR; or the plan docs conflict with no clear CPU-compatible next task. See
`docs/autonomous_task_loop.md` for the full list.

## Source-Of-Truth Documents

- `docs/autonomous_task_loop.md` — per-task procedure and reviewer mechanics.
- `docs/cpu_self_play_roadmap.md` — current CPU loop state, strength bars, progress log.
- `docs/learning_architecture_exploration.md` — the post-linear architecture direction.
- `docs/first_iteration_design.md`, `docs/bootstrap_strategy.md`, `docs/goals.md`.
