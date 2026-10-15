# Controlled Foul Play benchmark cost

The opt-in benchmark owner connects a fixed controlled Foul Play recipe to an
actually executed child, its original summary, and terminal cost readback. It
does not authorize scientific execution, replace the staged Phase A collector,
or establish search improvement. The unchanged benchmark remains the producer;
this module owns its execution and consumes its output.

## Execution contract

Run or read through `python -m pokezero.controlled_benchmark_cost`. `run` requires
`--recipe /absolute/recipe.json --output /absolute/new-attempt`; `read` takes
`--output /absolute/existing-attempt`. Scientific runs require separate authority.
An existing output directory is never reused. There is no arbitrary-command,
retry, resume, source-generation, registration, or fallback option.

The recipe is a closed JSON mapping with these fields:

- `kind`: `pokezero.controlled-benchmark-recipe.v1`.
- Absolute paths: `python`, `checkpoint`, `showdown_root`, `foulplay_root`,
  `foulplay_python`, and `node`.
- `policy_mode`: `raw` or `engine-mcts`; `seat`: `p1` or `p2`.
- Integer `games`, `seed_start`, and `max_decision_rounds`, and positive finite
  `timeout_seconds`. The original sequential seed band is fixed, not redrawn.
- `native`: null for raw, or a closed mapping of absolute `model` and `tables`
  paths, integer `depth`, `sims`, `batch`, `worlds`, `workers`, positive `cpuct`,
  and boolean `opponent_priors`. `workers` must be one. This is native fixed-work
  execution, **not** the plan's timed-deployment qualification.
- `input_sha256`: prospective lower-case SHA256 digests keyed by canonical
  absolute file paths. It must cover the checkpoint, interpreter, Foul Play
  interpreter, Node executable, Showdown `dist/sim/battle-stream.js`, Foul Play
  `run.py`, and native model/tables when applicable. Additional explicitly
  reviewed input files can be bound too. This subset is not full package
  qualification or proof of the native extension actually loaded.

The bridge arguments freeze CPU, temperature 1, public belief sources on, no
search fallback, Gen 3 random battles, the external Foul Play opponent, and its
1000 ms move budget. No policy settings are borrowed from another arm. The
recipe hash and exact arguments are checked before calling the bridge and on
durable readback. Source bindings cover the checkout's Python package and
BattleStream bridge JavaScript, not all installed dependencies. The child's
import path is set to that checkout; inherited PokeZero overrides are removed.

Trajectory-reference stays unavailable here even if another PR adds parser
support. Its separately isolated workers need an adequate owning cleanup
handshake before this envelope can safely supervise them.

## Measured cost and failure handling

The parent exclusively observes its exact child with non-reaping `waitid`, then
reaps once with `wait4`. It never equates disappearance, a STOP event, or a
progress file with terminal success. The measured wall interval includes the
child's launch, imports, benchmark call and teardown. Child CPU includes that
child and descendants it actually reaped, without adding overlapping child
counters. Kernel maximum RSS is labelled in platform units and is not a
physical process-tree peak. Separate startup and cleanup CPU remain null.

The worker records actual directly spawned subprocess handles. If a handle is
still live after the benchmark returns, it requests parent cleanup while still
alive. The parent confirms the owner is stopped before checking and signalling
registered direct session leaders, preventing a concurrent reap/PID-reuse race.
It never signals a reparented or guessed group. Forced cleanup, timeout and
interruption refuse, preserve observed exit codes and keep partial summaries.
Unobserved or unreaped descendant scopes are not declared closed or costed.

`MEASURED_CHILD` requires a clean exact child exit, consistent worker terminal
record and uncapped complete original summary. It is deliberately not full
workflow success: `complete_workflow_cost`, `descendant_cleanup_complete`,
`loaded_native_extension_qualified`, and `scientific_qualified` remain false.
Missing/malformed/partial/capped summaries cannot promote exit zero to success.
Native c-puct has a launch binding only because the existing native summary has
no corresponding field; the unrelated Python root c-puct is not substituted.

Readback preserves reported partial game scores separately from their scientific
status. Every original cell still has a scientific score interval of [0,1] and
`independently_validated=false`; this consumer accepts zero scientific outcomes.
Independent outcome, runtime, refusal-rate and provenance qualification is still
required before any inference or budget commitment.

## Verification

The focused suite exercises real owned subprocesses with authored bridge calls
and the actual bridge argument parser, config and result serializer. It covers
reaped-child CPU, exact original seed prefixes, failure after completed games,
startup and cleanup failure, missing/capped/malformed summaries, source/input
drift, recipe/argument substitution, timeout, separate sessions, frozen-owner
signal ordering, reparented identities and stopped-event rejection. It runs no
model, Showdown or Foul Play games. Local installation and model builds are not
required; hosted tests install only the bridge protocol dependency.

Initial verification failures remain failures: a noncanonical fixture pin,
the invented native c-puct field caught by independent review, and stopped-event
handling during cleanup. One hung disposable test was explicitly interrupted
with exit 130 after exact-handle cleanup. Later passing regressions do not erase
those results. This feature measures a useful owned scope, but complete runtime
and scientific qualification remain unfinished.
