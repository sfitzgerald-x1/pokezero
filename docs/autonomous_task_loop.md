# Autonomous Task Loop

This document defines how the autonomous engineering loop continues the CPU-first PokeZero
proof of concept. The loop chooses the next task from the project plan, implements it on a
feature branch, opens a PR with verification evidence, runs an adversarial Codex review,
applies necessary fixes, merges when the goal-mode rules allow it, and continues.

## Roles And Operating Environment

The loop uses two different models in fixed roles so review is genuinely independent
(cross-model review catches uncorrelated blind spots):

- **Executor — Claude Code.** Selects tasks, branches, implements, tests, opens PRs,
  applies review findings, and merges per the goal-mode rules below.
- **Reviewer — Codex (GPT-5.5, xhigh reasoning).** Runs an adversarial, read-only review of
  each code PR. The reviewer must not modify files and must not merge.

Operating environment:

- The executor works out of an **isolated git worktree** at
  `/Users/scott/workspace/agents/pokezero-agent`, not the primary checkout at
  `/Users/scott/workspace/pokezero`. This prevents working-tree collisions with any other
  process that shares the primary checkout.
- Each task branches from `origin/main` with a `scott/` branch name. Do not use `codex/`
  branch names, and do not check out, edit, or delete branches or worktrees owned by other
  in-progress work.
- The loop runs **serially** (one task at a time) for now. Parallel multi-track execution is
  deferred until the single-track executor/reviewer cycle is proven; if it is enabled later,
  each track must use its own worktree and merges must stay serialized.
- The executor owns `docs/cpu_self_play_roadmap.md`: progress entries are written by the
  executor at merge time, so the roadmap is never a parallel-edit contention point.

## Goal

The active goal is to **show meaningful progress on a CPU-trained model past the
`simple-legal` strategy** — a checkpoint that clearly and measurably beats `simple-legal`
under shared-opponent benchmarking, not merely a healthy run. "Meaningful" is defined by an
explicit baseline measurement (the first task) and the strength bars in
`docs/cpu_self_play_roadmap.md`.

This remains CPU-first: the goal must be reachable without GPU access. GPU-dependent PPO
work, large-scale training, raw replay corpus conversion, and distributed orchestration
remain later milestones.

The working source of truth is:

- `docs/first_iteration_design.md`
- `docs/bootstrap_strategy.md`
- `docs/cpu_self_play_roadmap.md`
- `docs/learning_architecture_exploration.md`
- `docs/goals.md`

At the start of each task, read those docs and identify the smallest unfinished item that
moves a CPU model measurably past `simple-legal`.

## Per-Task Workflow

1. Sync and inspect the repository.
   - From the executor worktree, fetch and branch from `origin/main`.
   - Run `git status --short --branch`.
   - Preserve unrelated user work and other in-progress branches/worktrees.

2. Select the task.
   - Read the plan docs listed above.
   - Pick the next task that most directly advances a CPU model past `simple-legal`
     (baseline measurement, training-recipe or data-quality change, benchmark evidence).
   - Write a one-sentence task statement before implementing.

3. Create the branch.
   - Branch from `origin/main` with a `scott/` branch name.

4. Implement the change.
   - Keep the change scoped to the selected task.
   - Add focused tests when a suitable test framework exists.
   - Update docs when behavior, workflow, or plan status changes.

5. Verify the change.
   - Run the narrowest meaningful tests first.
   - Broaden verification when shared harness behavior, policy loading, manifests, promotion
     gates, or CLI flows are touched.
   - Record non-zero exits as failures with what remains incomplete.

6. Open the PR.
   - Push the branch and open a PR against `main`.
   - Include these sections: `Summary`, `Task Attempted`, `Changes Introduced`,
     `Risk Assessment`, `Verification Evidence`.

7. Run the adversarial Codex review (code PRs).
   - From the executor worktree with the PR branch checked out, run the reviewer command
     below with a hard 15-minute timeout.
   - The review is read-only (`--sandbox read-only`), so the sandbox itself enforces that
     the reviewer cannot modify files. After it exits, run `git status --short` and treat any
     unexpected change as output to inspect before proceeding.
   - If Codex is unavailable, unauthenticated, fails, or does not finish within the timeout,
     record that and fall back to an independent adversarial Claude review so the loop is not
     blocked; swap back to Codex once it is reachable.
   - Documentation-only PRs do not require a Codex review.

8. Apply review findings.
   - Treat concrete correctness, test, verification, or scope findings as changes to address.
   - Ignore findings that conflict with the plan, repo policy, or current CPU-first scope,
     and say why.
   - Push updates to the same PR and update the PR description when verification changes.

9. Merge or stop according to the active mode.
   - Under an explicit long-horizon `/goal`, merge a code PR once it has passed local
     verification **and** Codex has signed off (no unresolved blocking findings), without
     waiting for per-PR user approval. Documentation-only PRs may be merged without a Codex
     review.
   - Merges are serialized: merge one PR, then re-sync `origin/main` before the next task.
   - Outside explicit goal-mode work, stop for user approval before merging.
   - After merge, write the roadmap progress entry, re-sync, and repeat the loop.

## Codex Review Command

Reviewer binary and verified invocation (Codex is a desktop app, so use the absolute path;
it is not on the executor's PATH):

```bash
# Purpose-built review of the PR branch's diff against main, run from the executor worktree:
/Applications/Codex.app/Contents/Resources/codex exec review --base main \
  -m gpt-5.5 -c 'model_reasoning_effort="xhigh"' \
  "$(cat review-prompt.txt)"

# General read-only form (also verified working) when reviewing by explicit context/diff:
/Applications/Codex.app/Contents/Resources/codex exec \
  -m gpt-5.5 -c 'model_reasoning_effort="xhigh"' \
  -C /Users/scott/workspace/agents/pokezero-agent --sandbox read-only \
  "$(cat review-prompt.txt)"
```

Notes: `exec` mode runs with approval `never` automatically; do not pass `--ask-for-approval`
(it is not an `exec` flag). If a run fails, capture exact stderr; likely causes are binary
path, config quote parsing (`-c 'model_reasoning_effort="xhigh"'`), auth/session, or Codex
refusing the repo/sandbox state.

## Codex Review Prompt Template

```text
Review this PR adversarially and skeptically.

Task attempted:
<ONE_SENTENCE_TASK_STATEMENT>

Be skeptical of whether this PR actually achieves the intended task. Focus on:
- correctness gaps
- missing tests
- hidden regressions
- whether the verification evidence proves the claim
- whether the change moves a CPU model measurably past simple-legal

Return prioritized findings only. Do not modify files and do not merge the PR.
```

## Acceptance Criteria

Every task PR should satisfy these unless it explicitly explains why one does not apply:

- The selected task is traceable to the plan docs and to the active goal.
- The PR advances a CPU model past `simple-legal` without requiring GPU access, or it is an
  explicitly scoped process/docs PR that improves how future loop work is executed.
- The PR includes focused tests or a clear reason tests are not meaningful.
- The PR does not claim self-play or model improvement without benchmark, incumbent,
  capped-rate, or comparable evidence.
- The PR keeps generated artifacts, local runs, and credentials out of git.
- The PR description states what was attempted and what verification passed.

## Stop Conditions

Stop and report status instead of continuing automatically when:

- The next meaningful task requires GPU access.
- The task depends on missing local prerequisites such as a Showdown checkout or an optional
  install that is not present.
- The plan docs conflict or no next task is clearly CPU-compatible.
- Codex (and the fallback Claude reviewer) find a blocking issue that cannot be safely fixed
  in the same PR.
- The reviewer path is unavailable for a code PR and the goal mode requires sign-off.
- A PR is ready to merge outside explicit goal-mode work, but explicit user approval has not
  been given.
