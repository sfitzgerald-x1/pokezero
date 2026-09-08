# MCTS search performance plan — evidence to strength

Date: 2026-09-08. This replaces the now-stale execution order in the September
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
| Does it select a better action in a known mechanism case? | Predeclared simple-regret panel | **One selector hypothesis worth testing; not promoted.** |
| Does less encoding work make real search faster without changing it? | Full-path warm/cold timing and parity | **Unknown.** A microbenchmark is insufficient. |
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
source-isolated, durable paired MCTS-versus-MCTS runner. It prevents source
mixing, rejects incomplete mirrored pairs from scoring, and records provenance.
It is infrastructure for the first strength comparison, not strength evidence.

### A bounded selector hypothesis

The corrected decision panel includes a deliberately harsh-prior one-world row:
visit-max selected `toxic` (regret 0.4745), while shadow acting-seat Q-max
selected the known winning `seismictoss` (regret 0). The rare-terminal-decoy and
equal-value controls did not let Q-max win by chasing a lucky terminal leaf.

That is useful enough to measure on representative development positions, but
too synthetic and too narrow to change production selection. PR #1346 keeps the
selector shadow-only and default-off. Its fresh exact-head mutation receipt is
the remaining integrity gate; no game claim depends on it.

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
          +-- selector shadow on development positions --> only then a separate
                                                     MCTS-v-MCTS candidate trial
```

These branches share provenance and durability rules but not outcome claims.
The only serial dependency is that a shared mutation runner must mutate and
restore one source tree at a time; timing preparation and result analysis do
not justify parallel mutation of those files.

### 1. Close integrity, not more mechanics

Finish the exact-head mutation receipt for PR #1346, commit the fresh receipt,
and obtain its already-requested independent review. Merge only after its CI
uses the receipt tied to the exact PR head. Rebase PR #1345 from current `main`
and regenerate its own receipt; receipts never transfer across a changed head.

This prevents a regression in the new source-isolation/durability path. It does
not warrant another selector feature or another broad evaluation framework.

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
- a small pilot budget and a confirmation-size rule based on pilot paired
  variance. Development/pilot seeds cannot become confirmation seeds.

Use source-isolated policies for both sides. After divergent actions, each side
must follow the actual resulting trajectory; replaying the incumbent’s later
decisions after a candidate divergence is invalid. The outcome report must
include paired score relative to 0.5, a confidence interval, seat splits,
completed work, p50/p95 latency, fallback/refusal counts, source receipts, and
the complete/incomplete-pair ledger.

**Decision rule:** a positive, sufficiently precise pilot earns one independent
confirmation run. A null, imprecise, or negative result does not justify a
hyperparameter sweep; it parks the claimed practical benefit while preserving
the correctness repair.

### 3. Measure the encoder in the real search path

Build a small, fixed development timing corpus before collecting numbers: early
and late game, both acting seats, branch-heavy and branch-light requests, and
representative histories. In an isolated compatible runtime, run baseline and
optimized builds in alternating order with fixed work and RNG. Record warm
search time and its existing subphases (tree, model, encoding, folding,
rendering, tensor writes, and action mapping); record cold start separately.

Require bit-exact encoded arrays, identical completed-tree values/actions, and
identical refusal/fallback behavior. A partial lattice cell or a benefit that
vanishes outside encoding is a useful null result, not a reason to tune cache
or batch parameters. Only a reproducible full-path reduction permits a separate
fixed-wall experiment that reinvests the saved time.

### 4. Qualify or park Q-max without altering production selection

After PR #1346’s integrity gate, collect shadow recommendation telemetry from
representative one-world development positions using one completed tree per
position. Visit-max and Q-max must see exactly the same legal visited actions,
completed visits, and work. Keep early stop disabled; its current visit-lock
rule is not validated for Q-max.

Advance Q-max only if a predeclared noisy/deep-continuation control and the
development panel show lower regret without an unacceptable low-visit error
pattern. It then gets its *own* paired MCTS-versus-MCTS comparison against the
corrected visit-max incumbent. Otherwise retain the telemetry if useful and
park the selector; never silently enable it because of the harsh-prior fixture.

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

## Near-term deliverables

1. Fresh and independently reviewed source-isolation mutation receipts for the
   two open implementation PRs.
2. A registered, source-bound backup-repair pilot contract and its first
   durable paired-game readout.
3. A compact full-path timing/parity report that either validates or falsifies
   the encoder microbenchmark’s practical effect.
4. A shadow-selector development report with an explicit promote/park decision.

The first item is an integrity prerequisite. The second is the first deliverable
that can establish improved MCTS playing strength.
