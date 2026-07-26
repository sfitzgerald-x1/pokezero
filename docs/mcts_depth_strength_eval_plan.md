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

The `spike-sac` GPU scenario sweep established useful starting bounds:

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
- `pokezero-search`, Pokemon Showdown, FoulPlay, and image revisions;
- belief-world count and belief sampling mode;
- leaf calibration and model-prior settings;
- search batch size, `c_puct`, chance branching, and deep-KO splitting;
- deterministic search seed derivation;
- FoulPlay battle and bot-RNG seed reservations;
- accelerator type and inference mode.

Only maximum depth and simulations per world vary in the primary lattice. If batch size or world
count is later varied, that is a separately labeled experiment because either can change search
semantics as well as wall time.

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
- the desired fixed belief-world count and matrix override, if any;
- an output root and a reserved seed-band identifier.

The materialization stage loads the checkpoint's own model configuration and derives:

- observation schema and token dimensions;
- transition-history budget and feature masks;
- architecture and policy id;
- required TorchScript export shape and device;
- encoder-table contract.

If the checkpoint is engine-compatible but exported artifacts are absent, the job creates them
once, validates them, and records their hashes. If the schema is unsupported, the checkpoint hash
does not match, or root/leaf contracts disagree, the run fails terminally before launching the
matrix. A rerun with the same checkpoint and manifest reuses validated exports.

The checkpoint reference is data. Changing it creates a new run id and new result namespace while
using the same orchestration code.

## 3. Statistical Meaning of 100 Games

Each FoulPlay entry is limited to exactly 100 games: 50 team seeds, with PokeZero playing both
seats for every seed. All entries reuse the same seed schedule and the same per-game FoulPlay RNG
schedule.

One hundred games is a screening sample, not proof of parity. At 50 wins, a 95% Wilson interval
is approximately 40.4% to 59.6%. Report rows using these terms:

- **clearly below parity:** the 95% interval upper bound is below 50%;
- **parity-compatible:** the 95% interval contains 50%;
- **directionally above parity:** the point estimate is above 50%, without claiming proof;
- **confirmatory evidence:** reserved for a later, larger pre-registered read.

Do not call a 100-game row "parity achieved." The useful result is whether a configuration is
worth a confirmatory run and whether additional compute improves paired outcomes.

For comparisons among configurations:

- collapse the two seats of each team seed into one paired score;
- bootstrap the 50 seed-pair score deltas;
- report the paired delta and 95% interval against the no-search checkpoint and against the
  fastest lower-depth configuration;
- retain wins, losses, ties, caps, and opponent crashes separately;
- make no multiplicity-adjusted significance claim from the screening lattice.

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

Build a checkpoint-independent timing corpus from held-out FoulPlay games. It should contain at
least 200 legal PokeZero decision states and be stratified across:

- six, four, two, and one Pokemon remaining;
- high, medium, and low team HP;
- no boost and material offensive or defensive boosts;
- forced switches and ordinary move requests;
- low and high hidden-world uncertainty;
- early, middle, and late battle phases.

This stratification is for coverage only. It does not create a dynamic policy.

### A3. Broad fixed-configuration lattice

Measure this initial lattice on the same decision corpus:

- depths: `2, 4, 6, 8, 10`;
- simulations per world: `512, 1024, 2048, 4096, 8192`;
- batch: the current validated fixed batch, initially `16`;
- belief worlds: the current production fixed count.

The 25 cells are timing probes, not 100-game strength entries. Cells may run in parallel because
they read the same immutable corpus and checkpoint.

Every cell records:

- end-to-end decision wall mean, median, p95, and maximum;
- search-only wall mean and p95;
- configured depth and the distribution of `max_depth_reached`;
- simulations requested and completed;
- model evaluations, expansions, decision nodes, and chance nodes;
- inference time, queue time if applicable, and non-inference search time;
- worlds attempted and searched;
- fallback, timeout, and invalid-action counts;
- root argmax and visit distribution for action-stability analysis.

### A4. Eligibility and pruning

A cell is eligible for a FoulPlay strength read only when:

- mean end-to-end PokeZero decision wall is strictly below 15 seconds;
- all corpus decisions complete;
- fallback and invalid-action counts are zero;
- the requested checkpoint and encoder provenance remain exact.

The 15-second condition is an experiment gate, not a ladder safety guarantee. Always report p95
and maximum latency so a later controller can reserve network and submission time.

Prune dominated cells before playing games:

- discard a cell that is slower and reaches no deeper than another cell at the same breadth;
- discard a breadth increase that leaves root decisions unchanged and adds no realized depth on
  the timing corpus, unless one diagnostic row is retained to confirm saturation;
- retain at least one eligible row at depths 2, 4, 6, and 8 when available;
- retain no more than seven search configurations for Phase B.

This caps Phase B at eight entries including the no-search checkpoint.

## 5. Phase B: 100-Game FoulPlay Screen

### B1. Entries

Evaluate:

1. the raw checkpoint with no search;
2. up to seven Pareto candidates from Phase A.

Each entry receives the same 100 games. A depth label is valid only alongside realized-depth
telemetry; a `depth=10` configuration that never passes depth 6 is reported as a depth-10 cap
with its observed depth distribution, not as demonstrated depth-10 search.

### B2. Persistent cluster orchestration

The complete experiment is submitted once as a persistent cluster orchestration Job. After
submission, no laptop, terminal, Codex session, or agent polling loop is required to keep it
alive. The controller owns this resumable state machine:

1. `materialize-checkpoint`
2. `validate-contract`
3. `mechanics-smoke`
4. `build-or-validate-timing-corpus`
5. `run-timing-lattice`
6. `merge-timing-and-select-candidates`
7. `run-foulplay-matrix`
8. `merge-strength-results`
9. `publish-report`

Every stage writes its output atomically and writes a completion marker last. On restart, the
controller validates and reuses complete stages, then resumes the first incomplete stage. It
classifies failures as:

- **retryable:** transient scheduling, transport, storage, or opponent-process failure; retry with
  bounded exponential backoff under the same run and shard id;
- **terminal:** provenance mismatch, unsupported checkpoint contract, invalid matrix, duplicate
  seed ownership, deterministic engine failure, or artifact validation error; stop the run and
  publish the exact failure.

The controller emits a machine-readable `status.json` with current stage, completed and total
tasks, retry counts, latest artifact paths, and terminal failure details. Optional notifications
consume that status artifact; they are not required for progress.

The public repository owns the checkpoint-parameterized runner, manifest schema, validation,
merge, and reporting logic. The private deployment repository owns the cluster Job wrapper,
resource requests, storage wiring, and notification Secret references.

### B3. Parallel game execution

Represent one entry as 50 mirrored seed pairs. Shard it into ten tasks of five seed pairs:

- each task runs ten games, five in each seat;
- each task owns a disjoint seed-pair range;
- tasks write partial results atomically and write a completion marker last;
- retries resume missing seeds and never discard completed games;
- the merger rejects duplicate seeds, missing seats, provenance drift, and marker-less shards.

Run entries and shards concurrently, bounded by measured simulator, FoulPlay, and inference
capacity. Parallelism must not change a configuration's effective inference batch or queue delay
without recording that as a different execution mode.

The matrix runner should be a persistent cluster Job or equivalent durable controller, not a
foreground Codex process. Progress and completion come from job-produced artifacts.

### B4. Strength and timing report

Produce one table with:

| Field | Meaning |
|---|---|
| `config_id` | Immutable depth, simulations, batch, worlds, and inference mode |
| `wins/games` | PokeZero wins out of 100 |
| `win_rate_95ci` | Wilson interval |
| `delta_vs_raw_95ci` | Paired bootstrap over 50 mirrored seed pairs |
| `mean/p95/max_s` | End-to-end PokeZero decision wall |
| `realized_depth` | Mean, p95, maximum, and cap-hit rate |
| `sims/evals` | Completed search work per decision |
| `fallbacks/timeouts` | Must be explicit |

Also publish:

- win rate versus mean seconds per decision, with 95% intervals;
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
   loaded native model. This avoids both checkpoint reloads and GPU-per-pod reservations without
   adding a network hop.
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
4. A versioned matrix manifest, deterministic `run_id`, and deterministic `config_id`.
5. A resumable stage controller whose state is fully represented by persisted artifacts.
6. A sharded FoulPlay runner and fail-closed merger for 50 mirrored seed pairs.
7. A report generator for the tables and frontier plots.
8. Optional served-leaf inference only if the profiling gate justifies it.

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
