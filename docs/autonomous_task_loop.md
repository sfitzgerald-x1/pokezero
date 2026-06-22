# Autonomous Task Loop

This document defines how an autonomous engineering loop should continue the CPU-first PokeZero proof of concept. The loop should choose the next task from the project plan, implement it on a feature branch, open a PR with verification evidence, run an adversarial Claude Code review, apply necessary fixes, and then stop for explicit merge approval.

## Scope

The current objective is a proof-of-concept self-play stack that can be developed and validated without requiring GPU utilization. Tasks should prioritize harness correctness, observability, resumability, promotion discipline, CPU smoke tests, and experiment reporting before moving into GPU-dependent training.

The working source of truth is:

- `docs/first_iteration_design.md`
- `docs/bootstrap_strategy.md`
- `docs/goals.md`

At the start of each task, read those docs and identify the smallest unfinished item that improves the CPU self-play loop's validity, measurement, or operability.

## Current Task Priority

Based on the current plan, the next CPU-compatible work should focus on closing the neural-iteration orchestration gap without requiring GPU training:

- Integrate neural iteration outputs with the existing promotion registry and gate path.
- Strengthen long-run benchmark thresholds and regression detection.
- Improve managed checkpoint lifecycle tooling where it affects promoted-policy selection.

GPU-dependent PPO work, large-scale training, and distributed orchestration remain later milestones.

## Per-Task Workflow

1. Sync and inspect the repository.
   - Check out `main`.
   - Pull the latest `main`.
   - Run `git status --short --branch`.
   - Preserve unrelated user work.

2. Select the task.
   - Read the plan docs listed above.
   - Pick the next CPU-first task from `Known limitations`, `Not implemented yet`, or the bootstrap near-term plan.
   - Write a one-sentence task statement before implementing.

3. Create the branch.
   - Branch from updated `main`.
   - Use a `scott/` branch name.
   - Do not use `codex/` branch names.

4. Implement the change.
   - Keep the change scoped to the selected task.
   - Add focused tests when a suitable test framework exists.
   - Update docs when behavior, workflow, or plan status changes.

5. Verify the change.
   - Run the narrowest meaningful tests first.
   - Broaden verification when shared harness behavior, policy loading, manifests, promotion gates, or CLI flows are touched.
   - Record non-zero exits as failures with what remains incomplete.

6. Open the PR.
   - Push the branch.
   - Open a PR against `main`.
   - Include these sections:
     - `Summary`
     - `Task Attempted`
     - `Changes Introduced`
     - `Risk Assessment`
     - `Verification Evidence`

7. Run adversarial Claude Code review.
   - Run Claude Code from `/Users/scott/workspace/pokezero`.
   - Use the command:

```bash
claude --dangerously-skip-permissions --model claude-opus-4-8
```

   - Prompt Claude to review the PR adversarially and skeptically, focusing on whether it achieves the selected task's goals.
   - Claude should review only. It should not merge the PR.

8. Apply review findings.
   - Treat concrete correctness, test, verification, or scope findings as changes to address.
   - Ignore findings that conflict with the plan, repo policy, or current CPU-first scope.
   - Push updates to the same PR and update the PR description when verification changes.

9. Stop for merge approval.
   - Do not merge without explicit user approval.
   - After an approved merge, check out `main`, pull, then repeat the loop.

## Claude Review Prompt Template

Use this as the default prompt after opening the PR:

```text
Review PR <PR_URL> in /Users/scott/workspace/pokezero adversarially.

Task attempted:
<ONE_SENTENCE_TASK_STATEMENT>

Be skeptical of whether this PR actually achieves the intended task. Focus on:
- correctness gaps
- missing tests
- hidden regressions
- whether the verification evidence proves the claim
- whether the change moves the CPU-first proof-of-concept self-play loop forward

Return prioritized findings only. Do not modify files and do not merge the PR.
```

If the Claude CLI is unavailable, unauthenticated, or fails, record that failure in the handoff and continue from local verification evidence.

## Acceptance Criteria

Every task PR should satisfy these criteria unless the PR explicitly explains why one does not apply:

- The selected task is traceable to the plan docs.
- The PR improves the CPU-first proof-of-concept loop without requiring GPU access.
- The PR includes focused tests or a clear reason tests are not meaningful.
- The PR does not claim self-play improvement without benchmark, incumbent, capped-rate, or comparable evidence.
- The PR keeps generated artifacts, local runs, and credentials out of git.
- The PR description states what was attempted and what verification passed.

## Stop Conditions

Stop and report status instead of continuing automatically when:

- The next meaningful task requires GPU access.
- The task depends on missing local prerequisites such as a Showdown checkout, optional PyTorch install, or Claude CLI authentication.
- The plan docs conflict or no next task is clearly CPU-compatible.
- Claude finds a blocking issue that cannot be safely fixed in the same PR.
- The PR is ready to merge but explicit user approval has not been given.
