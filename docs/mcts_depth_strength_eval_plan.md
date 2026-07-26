# MCTS Depth, Throughput, and Strength Evaluation Plan

**Status:** planned
**Scope:** fixed-budget Rust engine MCTS only; dynamic per-turn allocation is deferred
**Primary question:** which fixed search configurations average less than 15 seconds per
PokeZero decision while providing the strongest directional result against FoulPlay?

## 0. Decision This Plan Supports

For any compatible frozen checkpoint, measure the tradeoff among:

- configured and realized search depth;
- breadth, expressed as simulations per belief world;
- end-to-end decision wall time; and
- playing strength against FoulPlay.

The output is a Pareto frontier, not a single hard-coded ladder setting. A configuration is on
the frontier when no other measured configuration is both faster and stronger.

This study deliberately does not design the eventual ladder time-bank controller. Every arm uses
one fixed depth, simulation count, batch size, and belief-world count for every searchable
decision. Confidence-based early stopping and state-dependent budget changes stay off so they do
not confound the depth-by-breadth comparison.

## 1. Existing Evidence and Constraints

The tracked
[`spike-sac` GPU scenario sweep](../evals/spike-sac-gpu-depth-breadth-sweep.png)
established useful starting bounds:

- depth 2 can become confidently wrong as breadth increases;
- depth 4 is a fast stable regime on that scenario;
- depth 6 through 10 changes the selected action in useful ways;
- depth 8 or 10 at 8,192 simulations took about 11 to 12 seconds for one fully revealed world;
- configured depth above 8 had almost no extra wall cost because visits did not reach it.

Those numbers are scenario evidence, not a ladder estimate. Live random battles add belief-world
construction, several worlds per decision, variable branching, and longer observations. The
FoulPlay study must therefore gate configurations on measured end-to-end decision time rather
than extrapolating from the scenario.

The native Rust path currently evaluates TorchScript leaves in-process. The existing WS-L1
inference service serves Python policy forwards, but it is not a drop-in leaf evaluator for
`pokezero-search`. Shared GPU inference is therefore a promising optimization experiment, not a
prerequisite that can be assumed to exist.

## 2. Frozen Experiment Contract

Freeze these inputs before any timing or strength row is accepted:

- checkpoint path, SHA-256, policy id, and model configuration;
- checkpoint observation contract, history budget, encoder-table hash, and feature masks;
- Showdown source hash used to export the encoder tables;
- `pokezero-search`, Pokemon Showdown, FoulPlay, and image revisions;
- belief-world count and belief sampling mode;
- leaf calibration and model-prior settings;
- search batch size, `c_puct`, chance branching, and deep-KO splitting;
- deterministic search seed derivation;
- FoulPlay battle and bot-RNG seed reservations;
- FoulPlay strength rung, fixed at **FP-1000** (`--search-time-ms 1000`) for the primary read;
- FoulPlay decision cap, fixed at `max_decision_rounds=250`;
- accelerator type, inference mode, CPU/GPU allocation, Torch thread count, and worker
  concurrency.

Only maximum depth and simulations per world vary in the primary lattice. If batch size or world
count is later varied, that is a separately labeled experiment because either can change search
semantics as well as wall time.

Fix belief worlds at **4**, the current `EngineMctsConfig.worlds` default. Because worlds are
searched serially today, the 15-second aggregate gate implies an approximate 3.75-second budget
per world. The upper lattice cells are allowed to fail this gate; that is a result, not a harness
failure.

`EngineMctsConfig.search_time_ms` and `threads` govern the handcrafted `hp_fraction` path, not the
fixed-simulation model path in this plan. Record their inert configured values for provenance, but
do not interpret `search_time_ms` as a model-search deadline.

Primary rows require:

- model leaf evaluation and model priors enabled;
- no adaptive budget or early-stop feature;
- no search fallback;
- no checkpoint, encoder, or engine provenance drift;
- both seats represented on the same team seeds.

### 2.1 Reusable checkpoint input

The harness must not hard-code a lineage, architecture, observation width, history budget, policy
id, or checkpoint storage path. One invocation accepts:

- an immutable checkpoint reference plus expected SHA-256;
- an optional pre-exported TorchScript artifact and encoder tables;
- a Showdown source reference plus expected source hash;
- the desired fixed belief-world count and matrix override, if any;
- an output root and a reserved seed-band identifier.

The materialization stage loads the checkpoint's own model configuration and derives:

- observation schema and token dimensions;
- transition-history budget and feature masks;
- architecture and policy id;
- required TorchScript export shape and device;
- encoder-table contract.

If the checkpoint is engine-compatible but exported artifacts are absent, the job creates them
once, validates them, and records their hashes. Export reuse is keyed by checkpoint hash,
accelerator device, observation contract, Showdown source hash, and exporter revision. If the
schema is unsupported, the checkpoint hash does not match, or root/leaf contracts disagree, the
run fails terminally before launching the matrix.

The checkpoint reference is data. Changing it creates a new result namespace while using the same
orchestration code. Separate experiment identity from execution resources:

```text
experiment_id = sha256(frozen_contract_without_resource_profile + matrix_manifest)
execution_id = sha256(experiment_id + resource_profile)
```

Stages 1 through 4 record `experiment_id` and may be reused when only the resource profile
changes. Stages 5 through 9 record `execution_id`. A marker mismatch at its applicable identity
scope is terminal and is never reused. Reducing concurrency creates a new execution under the
same experiment rather than invalidating checkpoint materialization, validation, smoke, or corpus
artifacts.

## 3. Statistical Meaning of 100 Games

Each FoulPlay entry is limited to exactly 100 games: 50 team seeds, with PokeZero playing both
seats for every seed. All entries reuse the same seed schedule and the same per-game FoulPlay RNG
schedule.

Score each game using the existing controlled-FoulPlay convention: win = 1, tie or decision cap =
0.5, loss = 0. Average the two seats for each team seed to produce 50 independent pair scores.
The headline score and interval are the mean and a 95% bootstrap interval over those 50 pair
scores. Also report raw wins, ties, caps, and losses separately.

Use 10,000 deterministic percentile-bootstrap resamples. Freeze the bootstrap RNG seed in the
matrix manifest and use the same resampled pair indices for every configuration delta.

One hundred games is a screening sample, not proof of parity. With no ties or caps, 50 wins in 100
games has an approximate 40.4% to 59.6% binomial Wilson interval, illustrating the uncertainty.
Report rows using these terms:

- **clearly below parity:** the 95% interval upper bound is below 50%;
- **parity-compatible:** the 95% interval contains 50%;
- **directionally above parity:** the point estimate is above 50%, without claiming proof;
- **confirmatory evidence:** reserved for a later, larger pre-registered read.

Do not call a 100-game row "parity achieved." The useful result is whether a configuration is
worth a confirmatory run and whether additional compute improves paired outcomes.

For comparisons among configurations:

- bootstrap the 50 seed-pair score deltas;
- report the paired delta and 95% interval against the no-search checkpoint and against the
  fastest lower-depth configuration;
- retain wins, losses, ties, caps, and opponent crashes separately;
- make no multiplicity-adjusted significance claim from the screening lattice.

These are per-row descriptive screening labels. Family-wise error is unadjusted by design.

## 4. Phase A: Mechanics and Timing Funnel

### A1. One-image mechanics smoke

Run one or two FoulPlay games before the matrix:

- crate import and TorchScript load succeed;
- root and leaf observations honor the checkpoint's schema and history budget;
- legal actions map correctly in both seats;
- model priors and values are finite;
- `max_depth_reached` and search counters are emitted;
- no belief-world, fold, encoder, or engine fallback occurs.

Any failure stops the image. It must not be retried as if it were an unlucky battle.

### A2. Representative decision corpus

Build `pokezero.engine-mcts-timing-corpus.v1` from held-out FoulPlay games. It contains exactly
256 legal PokeZero decisions. Every record carries:

- the full public event prefix through the current request;
- the acting player's current request-derived action candidates and legal mask;
- battle, team, seat, and bot-RNG seeds;
- the public belief inputs required to reconstruct the search worlds;
- no opponent-private request or hidden team data.

The 256 decisions are stratified across:

- six, four, two, and one Pokemon remaining;
- high, medium, and low team HP;
- no boost and material offensive or defensive boosts;
- forced switches and ordinary move requests;
- low and high hidden-world uncertainty;
- early, middle, and late battle phases.

Strata may overlap. The corpus manifest records the deterministic held-out seed range, selection
algorithm, and count in every bucket; changing any of them changes the corpus hash. This
stratification is for coverage only. It does not create a dynamic policy.

Timing a record replays its public game prefix through the normal observation path so the
incremental fold is warm and path-faithful. Prefix replay and warm-up are excluded from the
decision timer. A cold per-record fold or reuse of `public-decision-corpus.v1` is not valid for
this study because that artifact omits the request-derived action mapping.

### A3. Broad fixed-configuration lattice

Measure this initial lattice on the same decision corpus:

- depths: `2, 4, 6, 8, 10`;
- simulations per world: `512, 1024, 2048, 4096, 8192`;
- batch: the current validated fixed batch, initially `16`;
- belief worlds: `4`.

The 25 cells are timing probes, not 100-game strength entries. After at least 64 decisions, stop a
cell early when its running mean decision wall is at least 15 seconds. Persist it as
`gate_failed`, including its partial sample size and complete telemetry; this is a result, not a
stage failure. Give each cell a 90-minute infrastructure deadline. A deadline before 64 decisions
is retryable once and then terminal as a harness failure. Cells may run in parallel only when each
receives the frozen resource profile and does not contend for the same inference queue or CPU
allocation. Otherwise run cells in waves or serially on one device.

Define end-to-end decision wall as:

```text
acting-player request available -> validated Showdown choice string ready to submit
```

This includes observation/legal-action construction, belief-world construction, native search,
root-action mapping, and choice serialization. It excludes prefix replay for the timing corpus and
network delivery, because the latter is not present in the offline harness. The existing
`policy_elapsed_seconds` begins after observation/context construction, so it is a component, not
the headline metric.

Every cell records:

- end-to-end decision wall mean, median, p95, and maximum;
- search-only wall mean and p95;
- request-to-context, belief-world construction, native encoding, model forward, tree work,
  action mapping, and choice-serialization timing;
- configured depth and the distribution of native `max_depth_reached`;
- simulations requested and completed;
- model evaluations, expansions, decision nodes, and chance nodes;
- inference time, queue time if applicable, and non-inference search time;
- worlds attempted and searched;
- fallback, timeout, and invalid-action counts;
- root argmax and visit distribution for action-stability analysis.

Add native encode/model/tree phase timers before running A3, then freeze that crate revision for
all rows. Do not estimate a missing phase by subtracting an assumed model cost.

Native `max_depth_reached` uses root depth 0 and saturates at `search_depth - 1`. The distribution
is over the scalar maximum from every `(decision, belief world)` search. `cap_hit` means
`max_depth_reached == search_depth - 1`; it is not a per-visit depth histogram.

### A4. Eligibility and pruning

A cell is eligible for a FoulPlay strength read only when:

- mean end-to-end PokeZero decision wall is strictly below 15 seconds;
- all corpus decisions complete;
- fallback and invalid-action counts are zero;
- the requested checkpoint and encoder provenance remain exact.

The 15-second condition is an experiment gate, not a ladder safety guarantee. Always report p95
and maximum latency so a later controller can reserve network and submission time. The actual
timer-bank and delivery constraints remain defined in
[`ladder_search_timing_constraints.md`](ladder_search_timing_constraints.md).

Prune dominated cells before playing games:

- discard a cell that is slower and reaches no deeper than another cell at the same breadth;
- discard a breadth increase that leaves root decisions unchanged and adds no realized depth on
  the timing corpus, unless one diagnostic row is retained to confirm saturation;
- for each configured depth, retain the eligible cell with the largest simulation count, breaking
  ties by lower mean wall time;
- if fewer than seven search cells are selected, add the fastest remaining eligible cell and then
  the remaining cell nearest the median eligible wall time;
- stop at seven search configurations.

This caps Phase B at eight entries including the no-search checkpoint.

Before accepting timings, replay a fixed 32-decision corpus subset at the most expensive eligible
cell, once at concurrency 1 and once at the maximum configured concurrency. Mean and p95 decision
wall must each agree within 10%. Otherwise reduce concurrency and start a new execution of A3;
contention is not part of a search configuration.

## 5. Phase B: 100-Game FoulPlay Screen

### B1. Entries

Evaluate:

1. the raw checkpoint with no search;
2. up to seven screening candidates selected by A4.

Each entry receives the same 100 games. A depth label is valid only alongside realized-depth
telemetry; a `depth=10` configuration that never passes depth 6 is reported as a depth-10 cap
with its observed depth distribution, not as demonstrated depth-10 search.

### B2. Persistent cluster orchestration

The complete experiment is submitted once as a persistent cluster orchestration Job. After
submission, no foreground interactive session or polling loop is required to keep it alive. The
controller owns this resumable state machine:

1. `materialize-checkpoint`
2. `validate-contract`
3. `mechanics-smoke`
4. `build-or-validate-timing-corpus`
5. `run-timing-lattice`
6. `merge-timing-and-select-candidates`
7. `run-foulplay-matrix`
8. `merge-strength-results`
9. `publish-report`

Every stage writes its output atomically and writes a completion marker containing its applicable
`experiment_id` or `execution_id` last. On restart, the controller validates and reuses complete
stages, then resumes the first incomplete stage. It classifies failures as:

- **retryable:** transient scheduling, transport, storage, or opponent-process failure; retry with
  bounded exponential backoff under the same run and shard id;
- **terminal:** provenance mismatch, unsupported checkpoint contract, invalid matrix, conflicting
  duplicate seed result, deterministic engine failure, or artifact validation error; stop the run
  and publish the exact failure.

The controller emits a machine-readable `status.json` with both identities, current stage,
completed and total tasks, retry counts, latest artifact paths, and terminal failure details.
Optional notifications consume that status artifact; they are not required for progress.

The public repository owns the checkpoint-parameterized runner, manifest schema, validation,
merge, and reporting logic. The private deployment repository owns the cluster Job wrapper,
resource requests, storage wiring, and notification Secret references.

### B3. Parallel game execution

Represent one entry as 50 mirrored seed pairs. Shard it into ten tasks of five seed pairs:

- each task runs ten games, five in each seat;
- each task owns a disjoint seed-pair range;
- tasks write partial results atomically and write a completion marker last;
- each opponent crash receives one retry;
- retries resume missing seeds and never discard completed games;
- duplicate `(config_id, seed, seat)` results are idempotent when their canonical outcome fields
  match: result, score, turn count, provenance hashes, and chosen-action sequence;
- timing and telemetry differences between canonically matching duplicate attempts are retained as
  retry diagnostics, not treated as conflicts;
- canonical outcome conflicts, missing seats, provenance drift, and marker-less shards fail
  closed.

Pre-register ten spare seed pairs after the primary 50. If any pair still crashes after its retry
in any entry, exclude that pair from every entry and promote the next spare pair for every entry.
Continue until all entries share exactly 50 complete mirrored pairs or the spare band is
exhausted. Exhaustion is terminal. The substitution order is fixed before outcomes are observed.
Each promoted pair runs as an appended repair shard keyed by `(entry, spare_index)`. Excluded pairs
remain in the ledger with `excluded_reason` and are omitted from every entry's shared 50-pair
scoring set.

Run entries and shards concurrently under the frozen per-game CPU/GPU allocation and Torch thread
count. Parallelism must not change a configuration's effective inference batch, queue delay, or
FoulPlay compute allocation without recording that as a different execution mode. When resources
cannot isolate all entries, run equal-sized waves rather than oversubscribe.

Before the first full wave, replay one fixed mirrored pair at concurrency 1 and at wave
concurrency under the same resource limits. Record FoulPlay iterations-per-move if the opponent
exposes it; otherwise record its CPU allocation and observed move wall. A material resource or
opponent-compute discrepancy creates a new, lower-concurrency execution before strength games
begin.

The matrix runner is a persistent cluster Job or equivalent durable controller, not a foreground
interactive process. Progress and completion come from job-produced artifacts.

### B4. Strength and timing report

Produce one table with:

| Field | Meaning |
|---|---|
| `config_id` | Immutable depth, simulations, batch, worlds, and inference mode |
| `foulplay_rung` | `FP-1000` for the primary read |
| `record` | Wins, ties, caps, and losses out of 100 |
| `score_95ci` | Mean pair score and 50-pair bootstrap interval |
| `delta_vs_raw_95ci` | Paired bootstrap over 50 mirrored seed pairs |
| `mean/p95/max_s` | End-to-end PokeZero decision wall |
| `realized_depth` | Mean, p95, maximum, and cap-hit rate |
| `sims/evals` | Completed search work per decision |
| `fallbacks/timeouts` | Must be explicit |

Also publish:

- pair score and raw win rate versus mean seconds per decision, with 95% intervals on score;
- realized depth versus simulations;
- p95 latency versus strength;
- root-action agreement between adjacent depth and breadth cells;
- a machine-readable manifest and merged per-game ledger.

## 6. Shared Inference Decision

### Recommendation

Yes, configurations evaluating the same checkpoint should eventually share one GPU-backed model
service rather than each reserving a GPU. Do not make that refactor a blocker for the first timing
funnel.

The native search crate currently owns its TorchScript leaf evaluator. Reusing the existing
Python inference server would require a new Rust `BatchLeafEval` transport and would put a
latency-sensitive RPC inside tree expansion. Build it only after measuring the current split.

Use this decision sequence:

1. Profile in-crate model time versus encoding, tree work, and belief-world construction.
2. First try one evaluator process per checkpoint running several game shards while reusing one
   loaded native model. This tests checkpoint load-time and memory amortization only; it does not
   batch independent search forwards and is not expected to improve inference throughput.
3. If one process cannot feed enough parallel games, prototype a shared leaf-inference service
   with dynamic batching.
4. Compare local and served modes on identical encoded leaves and the same timing corpus.

The served-inference pilot must prove:

- logits and values match the accepted numeric tolerance;
- root actions match on a deterministic parity corpus;
- one immutable checkpoint id and observation contract are latched for the run;
- request queue time, forward time, serialization time, and end-to-end latency are measured;
- at 1, 4, 8, and 16 concurrent search workers, mean decision wall regresses by no more than 10%
  while GPU utilization and aggregate games/hour improve;
- no request is silently retried into a different search result.

Before the multi-game process is considered viable, add a deterministic concurrent
`eval_batch` thread-safety test over one shared native model. The local mechanism is several
`asyncio.to_thread` searches sharing that evaluator; passing this test does not waive the measured
throughput gate for served inference.

Configurations for one checkpoint should share a service. Distinct checkpoints should not hot
reload through one queue during a timing comparison. If several small checkpoints later share one
physical GPU, keep each model resident with a separate queue and report per-checkpoint queue
latency so cross-model contention cannot masquerade as a search effect.

## 7. Implementation Deliverables

The public repository needs:

1. A checkpoint resolver that derives and validates the observation, export, and encoder contract
   from any compatible checkpoint.
2. A controlled FoulPlay engine-MCTS policy mode that accepts the frozen Rust search
   configuration and emits the native search telemetry.
3. A decision-corpus timing runner for the Phase-A lattice.
4. Native per-phase encode/model/tree timers, landed before the crate revision is frozen.
5. A versioned matrix manifest, deterministic `experiment_id`, deterministic `execution_id`, and
   deterministic `config_id`.
6. A resumable stage controller whose state is fully represented by persisted artifacts.
7. A sharded FoulPlay runner and fail-closed merger for 50 mirrored seed pairs.
8. A report generator for the tables and frontier plots.
9. Optional served-leaf inference only if the profiling gate justifies it.

Deployment-specific manifests, storage paths, credentials, and cluster identifiers remain in the
private deployment repository. The public interface is parameterized by worker count, shard id,
seed range, checkpoint, output path, and inference endpoint.

The private deployment repository needs one checkpoint-parameterized launcher that creates the
persistent orchestration Job. Its user-facing contract should be one command with a checkpoint
reference, checkpoint hash, seed reservation, and output location. The image must contain the
public runner, FoulPlay, Pokemon Showdown, and the model-enabled Rust search crate so an agent does
not install or coordinate dependencies between stages.

## 8. Stop Conditions

The fixed-budget study is complete when:

- the same unmodified harness has completed smoke runs for two checkpoints with different model
  or observation configurations;
- one frozen checkpoint has a valid 25-cell timing lattice;
- no more than seven eligible search rows plus the raw policy have complete 100-game reads;
- every row has exact provenance, 100 unique games, 50 mirrored pairs, and no hidden fallback;
- the timing and strength Pareto frontier is published;
- parity language follows Section 3;
- one unattended cluster rehearsal reaches a published probe-scale report after the submitter
  disconnects, and a forced retryable shard failure resumes without losing completed work;
- the inference-server decision is supported by a measured local-versus-shared comparison or is
  explicitly deferred because model inference is not the bottleneck.

Do not continue adding grid cells after the frontier is stable. The next work item is either a
larger confirmatory read for the best fixed configuration or the separate dynamic ladder
time-allocation study.
