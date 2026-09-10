# MCTS search performance plan — evidence to strength

Date: 2026-09-08, updated 2026-09-10. This replaces the now-stale execution order in the September
7 working plan. It is deliberately narrow: get one attributable answer about
whether the repaired MCTS is stronger, then invest only where the answer points.
It is not a deployment plan and does not turn any diagnostic or synthetic result
into a playing-strength claim.

## Objective

Improve the choice MCTS makes within a practical per-decision budget. A change
advances only when it clears the next relevant question:

| Question | Evidence that answers it | Current state |
| --- | --- | --- |
| Does the tree compute completed values correctly? | Invariants and declared action-choice panel | **Yes, for the batched-backup bug.** |
| Does it select a better action in a known mechanism case? | Predeclared simple-regret panel | **No for raw Q-max.** The deep noisy control falsified it; the selector is parked. |
| Does less encoding work make real search faster without changing it? | Full-path warm/cold timing and parity | **No reproducible benefit shown.** The first ordered development read is parked. |
| Does a candidate win more paired games than its frozen MCTS incumbent? | Fresh source-isolated mirrored MCTS-versus-MCTS games | **Unknown. This is the next decisive outcome.** |

The scorecard is intentionally not “tests passed,” simulation count, or PR
count. A source change earns credit only for the column it directly measures.

## What is established

### Corrected backup baseline

PR [#1337](https://github.com/sfitzgerald-x1/pokezero/pull/1337) merged as
`df4e3ce15ee69f922f6ae1b81c7b5e9861828319`. It fixes dilution of values when
batched visits collide. The prior behavior turned a guaranteed win into Q=0.75
and a constant 0.8 leaf into Q=0.6. This is a correctness repair, not yet a
measured strength win.

The repaired tree passes the declared action-choice panel for both acting seats
and collection batches 1, 2, 8, and 64. In particular it retains an immediate
terminal winning action instead of hiding it. The panel also reports finite-tree
Q error rather than mistaking a correct final action for exact convergence.

### Efficiency and evaluation plumbing

PR [#1340](https://github.com/sfitzgerald-x1/pokezero/pull/1340) merged an
identifier-normalization improvement. Its five-fixture boundary microbenchmark
reduced median encoding time from 88.252 to 81.434 microseconds (7.73%), with
bit-exact encoded arrays. That is not yet an end-to-end MCTS latency result and
must not be reinvested as extra search work until full-path parity is measured.

PR [#1341](https://github.com/sfitzgerald-x1/pokezero/pull/1341) provides the
durable paired MCTS-versus-MCTS runner. It records provenance and rejects
incomplete mirrored pairs from scoring. PR
[#1343](https://github.com/sfitzgerald-x1/pokezero/pull/1343) adds the
source-isolated mode required for different native candidate and incumbent
builds. Together they are infrastructure for the first strength comparison,
not strength evidence.

The first fixed-work full-path encoder read is a null result. Across two
candidate-first blocks, candidate timing looked slower; in a baseline-first
order flip it was effectively tied (6.131s candidate versus 6.174s baseline
median). Because the direction reverses with order and times fall sharply over
the sequence, the result is dominated by warm-state/order effects rather than
a source-attributable win or loss. The exact receipts, counters, and limits are
in [the timing readout](mcts-fullpath-timing-readout-20260908.md). Do not
reinvest encoder time into extra search or claim full numerical parity from this
timing artifact.

### A bounded selector hypothesis

The corrected decision panel includes a deliberately harsh-prior one-world row:
visit-max selected `toxic` (regret 0.4745), while shadow acting-seat Q-max
selected the known winning `seismictoss` (regret 0). That narrow row was a
hypothesis, not a promotion result.

The required deep-noisy control falsified raw Q-max for both acting seats:
Q-max selected a 10-visit `tackle` with reported Q=0.632018 against its declared
payoff of 0.4 (simple regret 0.15 and Q error 0.232018), while visit-max selected
`splash` with payoff 0.55 and zero regret. Raw Q-max is therefore parked. PR
#1346 remains shadow-only and default-off; it is not a path to a game-level
trial or production selection.

An independent post-merge review also found that PR #1346 is not a valid
integrity fallback: its shadow path can attest a truncated native report as a
full-budget tree, and its committed mutation receipt covers rollout-leaf witness
behavior rather than selector guards. Those gaps need not be repaired while the
falsified selector is parked. They must be fixed and independently re-reviewed
before any future selector evidence is used.

## Execution order

```text
corrected backup baseline
          |
          +-- fixed-work, source-isolated MCTS-v-MCTS pilot --> strength readout
          |                    |
          |                    +-- positive + precise --> confirmation only
          |                    +-- null/negative --> park mechanism
          |
          +-- full-path timing/parity --> decide whether to reinvest saved time
          |
          +-- raw Q-max selector --------------------------> parked (falsified)
```

These branches share provenance and durability rules but not outcome claims.
The only serial dependency is that a shared mutation runner must mutate and
restore one source tree at a time; timing preparation and result analysis do
not justify parallel mutation of those files.

### 1. Keep the selector conclusion bounded

PR #1345's source-isolated lifecycle/reset work and PR #1346's default-off
shadow implementation have merged. The shadow integrity review was not clean,
but its failure does not undermine the direct deep-noisy panel result: raw
Q-max is parked. Do not repair its telemetry, add a threshold family, collect a
selector development corpus, or run selector games in this iteration.

If a future independent mechanism reopens selector work, first require an
exact-head, selector-specific mutation battery and a full-budget native-report
invariant, then obtain a fresh independent review. Receipts never transfer
across a changed head.

### 2. First decisive experiment: backup repair versus frozen predecessor

Run the repaired-backup candidate against the frozen pre-repair implementation
only. Do not bundle identifier normalization or Q-max selection into this
contrast. Freeze before outcomes are read:

- exact candidate and predecessor commits; the final-enthalf checkpoint, whose
  SHA-256 is `0fd095923b4ac7e05d6e2b3ccab9c1e6869dff4893c2dae456caff10dce690be`;
  engine, vocabulary, and environment receipts;
- paired seed list, mirrored side/team assignment, draw scoring, per-decision
  work cap, and measured latency accounting;
- retry and failure rules that preserve completed games but score neither member
  of an incomplete mirrored pair; and
- fixed, disjoint pilot and confirmation seed rosters. Development/pilot seeds
  cannot become confirmation seeds; and
- decision fallbacks and **root-prior** fallbacks, distinct from the separately
  reported interior simulated-branch prior fallbacks. A nonzero root count
  invalidates the strength claim; interior counts remain visible but are not
  mislabeled as failures of the live chosen action.

The frozen runtime revisions used for the earlier attempt only emitted the
aggregate fallback counter. They are not eligible for this scoped gate: the
replacement pair must be source-bound to the two reviewed, instrumentation-only
runtime revisions that emit `prior_fallbacks = root_prior_fallbacks +
branch_prior_fallbacks`. This is a compatibility requirement, not a search
treatment; the same normalized delta is applied to both sides and the original
backup-repair contrast remains the only mechanical difference.

Use source-isolated policies for both sides. After divergent actions, each side
must follow the actual resulting trajectory; replaying the incumbent’s later
decisions after a candidate divergence is invalid. The outcome report must
include paired score relative to 0.5, a confidence interval, seat splits,
completed work, p50/p95 latency, fallback/refusal counts, source receipts, and
the complete/incomplete-pair ledger.

For this first contrast, make those words operational before a game is run: use
exactly 12 mirrored pilot pairs with 10,000 paired-bootstrap resamples. All
thresholds apply to **Δ = mean candidate paired score − 0.5**, not the runner's
raw candidate score. The smallest useful effect is Δ=+0.05. The pilot proceeds
to confirmation only if every pair is complete, there are no decision or
root-prior fallbacks/refusals, the point estimate is at least +0.05, its
predeclared 80% interval for Δ lies
wholly above zero, and both sides have zero decision and root-prior fallbacks.
Interior simulated-branch fallbacks are reported as topology telemetry, not
silently discarded or treated as live-root failures. The confirmation uses a separate, already-reserved
roster of 50 mirrored pairs and the same 10,000-resample paired bootstrap at
95%; an improvement is claimed only if its interval for Δ has a lower bound of
at least +0.05. Run every pair in a roster before reading that roster's outcome;
the pilot is inspected once after all 12 pairs and the confirmation once after
all 50. No pilot seed enters the confirmation roster, and neither roster is
extended after outcomes are read. An imprecise result is inconclusive, not
evidence of no benefit. Fixed-work remains the primary comparison; latency tails
are reported but cannot be relabelled as a fixed-wall result without a
separately enforced decision clock.

### 3. Encoder line is parked unless a new measurement need emerges

The first declared development corpus was measured in serial source-bound
processes, including an order flip. It does not provide a reproducible
end-to-end direction, so no cache, batching, or fixed-wall follow-up is
authorized from this optimization. A future measurement must randomize order,
separate cold and warm treatment, and add full numerical tree/array parity
capture before it can reopen the line.

A partial lattice cell, missing frozen corpus, or a benefit that vanishes outside
encoding is a useful null result, not a reason to tune cache or batch parameters.
Only a reproducible full-path reduction with full parity capture permits a
separate fixed-wall experiment that reinvests saved time.

### 4. Q-max is parked without altering production selection

The deep-noisy control is the relevant negative result: a low-visit, lucky
continuation can produce an overstated root Q. The raw Q-max recommendation is
not eligible for a representative development readout, a game-level candidate
trial, or production selection. Do not reinterpret the harsh-prior fixture as
countervailing evidence.

## Operating safeguards

- Each long run writes atomic progress, per-unit durable records, a terminal
  PASS/NONPASS result, source/hardware receipts, and a handoff that can be
  independently checked. A job loss must retain completed units and diagnostic
  state; no partial result can be scored as a completed study.
- Run at most two nodes in namespace `scott` on `olfusa`. CPU-heavy work first
  selects a node with measured spare engine capacity and pins affinity there.
  Any GPU job requests multiple GPUs together and never fragments a node.
- Use only new development inputs for mechanism and timing work. Keep protected
  confirmation seeds/positions out of pilots, root-cause probes, and threshold
  selection.
- Record negative results plainly. Broad PUCT/depth/simulation grids, new cache
  rewrites, and model retraining remain out of scope until a completed contrast
  identifies a concrete bottleneck or mechanism.

## Near-term deliverable

The only active decision is the registered, source-bound backup-repair pilot's
complete paired-game readout. It can permit consideration of the disjoint
confirmation roster only under its frozen threshold; it cannot itself support
an improvement claim. The encoder and Q-max lines are parked, and neither
authorizes follow-on tuning or a second game experiment.
