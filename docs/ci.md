# CI policy

`premerge` is the single merge gate. On a ready pull request, it requires all eight core lanes and hygiene to pass on the proposed merge result. Its budget is **15 minutes of wall time**, including queueing. Each core lane has a **20-minute timeout**. Split existing work or move eligible tests to nightly if a lane approaches the cap. Keep the cap at 20 minutes.

| Lane | Coverage |
| --- | --- |
| `turn` | Engine state, transitions, move execution, chance, and turn order |
| `collect` | V2 collection, actor banks, device rollout, and model integration |
| `encode` | Device observations, trackers, public hooks, batched transitions, and self-move encoding |
| `encode-parity` | Observation parity at the Showdown boundary |
| `search` | DepthN live search, pricing, reductions, and hedging |
| `oracle` | Showdown-backed accuracy and ability identity checks |
| `learn` | V2 optimizer identity, checkpoint/resume, actor export, and distributed layout |
| `serve` | DepthN server and legal fallback, Showdown and Foul Play adapters, and battle progress |
| `hygiene` | Python syntax floor, credential scanning, and the CI allowlist |

## Where tests belong

- **Premerge** runs fast, deterministic guards for the live path. Eligible surfaces are V2 collection, learning and encoding, DepthN search and serving, and the Showdown and Foul Play adapters. Keep the approved workflow and lane list in [the allowlist](../.github/ci-premerge-allowlist.json). A new PR workflow or lane requires a deliberate policy change with the allowlist updated in the same PR.
- **Nightly** is the default for new tests. Put slow or compile-heavy checks, memory checks, real wall-clock budgets, citation or source-line checks, artifact or report consistency, and historical certification here when they still provide value.
- **Manual** covers affected native build and differential qualification, heavy environment-specific tests, and extended qualification for a scored bind. Before an S0, 100M, or scored screen bind, run extended against the exact source commit. This gate keeps the public-world regression that escaped in #1967.

Once the main-branch workflow is registered, set `BIND_COMMIT` to the exact bind commit and dispatch pre-bind extended.

```sh
gh workflow run prebind-extended.yml --repo sfitzgerald-x1/pokezero --ref main -f source="$BIND_COMMIT"
```

## Adding a test

1. Put it in a nightly suite. GPU component and DepthN suites discover matching modules. Add other modules to the named nightly job selection.
2. Promote a test to an existing premerge lane only when it is fast on a hosted runner, deterministic under contention, and guards one of the live paths above. Use the smallest selector that covers the behavior and keep the lane and total wall time within budget.
3. Run the hygiene guard locally with PyYAML installed if a workflow changes. Run `python -m unittest tests.test_ci_premerge_policy`.

## Why this budget exists

Before the 2026-11-20 prune, the median merge cycle was **207 minutes**. Premerge alone took **84.4 minutes** and postmerge extended took **86.8 minutes**. The lean gate now targets **about 15 minutes**. The first hosted cutover passed in **17m16s** including queueing, and the encode split passed in **15m29s** from first job to aggregate. The merge-cycle target is **10–20 minutes**. Keep correctness checks such as Showdown fidelity and bitwise identity on the affected work while moving slower checks off the merge path.
