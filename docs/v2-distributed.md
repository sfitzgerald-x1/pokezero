# V2 synchronous fused ranks

`gpu_engine.v2_distributed` runs one fused collector/learner process per
allocated CUDA device on one host. It requires exactly the requested device
count, gives each child one GPU UUID, and stops all owned children on a failed
rank or the finite parent deadline. Rank-local logs and `rank-exit.json` remain
under the output directory. This implementation does not authorize a training
declaration or satisfy its scientific qualification gates.

```sh
python -m gpu_engine.v2_distributed --ranks 4 --timeout-seconds 3600 \
  --declaration "$DECLARATION" --declaration-sha256 "$DECLARATION_SHA" \
  --runtime "$RUNTIME" --runtime-sha256 "$RUNTIME_SHA" --plan-arm B1 \
  --reference-run "$INITIAL" --out "$OUTPUT" --max-iterations 2 \
  --recipe R5 --lanes 2048 --row-order file --order ordered --mode cudagraphs
```

The finite horizon above is an example validation invocation. Production
timeouts, declarations and initial snapshots must be bound separately. The
original `gpu_engine.v2_fused` entry remains the single-device path.

S0 GPU runs can enable `--s0-idle-guard` on the parent launcher and set
`JAX_COMPILATION_CACHE_DIR` to the workers' persistent cache. The guard
requires fifteen minutes of startup time and more than fifteen minutes of
overlapping zero utilization on all allocated GPUs, near-zero CPU on every
rank, and unchanged cache and rank logs. An unavailable sample restarts the
affected observation interval. The finite wall deadline and normal rank-exit
handling still apply. An accepted idle timeout appears in `rank-exit.json`.

## Games and batches

A round's declared games divide into equal contiguous rank shards. Four ranks
collect 5,000 games each for a 20,000-game round. Assignment and action RNG use
the original global stream identities/ordinals. A constructor block crossing
a shard boundary is constructed at its original full shape on the affected
ranks, then sliced; constructing smaller blocks would change constructor RNG.
Only each rank's shard is collected. No rank repeats the full global game set.

Before learning, ranks exchange compact row addresses and derive the original
file order `(construction block, frame, row within block, owner)`. Original
construction-block boundaries remain optimizer segments. An all-to-all
redistributes feature columns once per round so each global batch divides
contiguously across ranks. Short segment tails use zero-weight padding, never
discarded or duplicated real rows. Global counters exclude padding. This
exchange costs bandwidth and transient memory; it is not a measured scaling
claim.

`--recipe` explicitly selects a package from a validated 512×5 reference:

| Recipe | Global batch | Epochs | LR/reference LR | Local batch at four ranks |
|---|---:|---:|---:|---:|
| R5 | 512 | 5 | 1 | 128 |
| R2 | 2,048 | 2 | 4 | 512 |
| R1 | 4,096 | 1 | 8 | 1,024 |

Every other optimizer/loss control is preserved. The original reference
checkpoint's exact hash, model, loss and training metadata are verified before
using its initial tensors. Fresh optimizer/RNG state uses the explicit selected
recipe, whose effective plan hash is recorded. The LR window uses global
declared games, not per-rank games. A checkpoint cannot silently change the
recipe, source/runtime bindings, row semantics or rank count on resume.

The existing declaration loader still enforces its Phase C reference contract;
this adapter does not relax it to admit an unbound 100M or S0 declaration.
Recipe choice does not replace S0 or add support for undeclared input flags.

## Learner synchronization

Each masked loss term has its own global sum of active training weights.
Ranks all-reduce those denominators, PPO active count and weighted advantage
sum, then a second centered variance reduction. Each local loss is its weighted
numerator divided by the global denominator. Summing the local gradients gives
the global-batch gradient. Averaging rank-local normalized losses would be
incorrect when weights, masks or tails differ.

Gradient summation precedes clipping and AdamW, so every rank applies the same
update and maintains the same optimizer state. The forward/backward remains
compiled locally; collectives execute outside the CUDA graph. Metrics are
summed across ranks and nonfinite flags use a global minimum. The legacy
replicated-row-store DDP mode remains separate and cannot load this mode's
checkpoint with its old loss semantics.

Rank dropout streams are independent deterministic functions of the initial
seed and rank. Consequently a different rank count is not bitwise identical to
a single global dropout draw. Mathematical equivalence tests disable dropout,
use uneven masks/weights/tails, and state numerical tolerances. A separate
same-rank test enables dropout and requires exact checkpoint-resume tensors and
metrics. These controls do not qualify S0 at a different rank count.

## Checkpoints and qualification

All ranks call the checkpoint writer; rank zero publishes one checkpoint with
all rank RNG states, shared optimizer state, global cursor and counters. Rank
zero exports the actor; its snapshot metadata is broadcast before any next
collection. All ranks load that same actor/history. Rank zero commits
`state.json` only after the complete global round, followed by a barrier.
Interrupted rounds restart only from the last committed boundary under the
run's retry policy. Orphaned checkpoint/export files fail closed; they are not
silently overwritten or retrained.

CPU controls cover original one-rank update statistics versus four ranks,
global row accounting, constructor slicing, recipe bindings, and exact
same-rank resume. Loss/update-statistic tolerance is absolute 2e-5 in float32;
prediction tolerance is absolute/relative 2e-4. Analytically invariant parameter
directions can amplify reduction roundoff under Adam, so predictions and update
statistics are compared instead of mistaking those shifts for model drift.

Required before launch: same-node GPU equivalence/collection and resume
validation, actual per-rank throughput/memory/utilization, independent review,
the canary route decision, S0 at the launch rank count, and the final signed
declaration. CPU gloo results alone are not GPU qualification.

The v5 builder retains one custom module type family per complete immutable
model configuration for the process lifetime. Lookup and class construction
are synchronized; every call still creates independent model parameters and
buffers in the original initialization order. Caching types avoids a known
Dynamo type-identity guard failure when an equal-config runtime is rebuilt.
It changes neither checkpoint schema nor optimizer, dropout, or RNG handling.

Stable types are a candidate correction for unnecessary recompilation, not
proof of exact GPU resume. Qualification must compare the uninterrupted suffix
with both a new runtime in the same process and a fresh worker process loading
the same prefix checkpoint. A cold process does not inherit type identities,
compiled state, or CUDA graphs. Record compiler-cache policy and preserve
strict state/statistic comparisons before proceeding to production smoke.


### Actor archive source compatibility

The installed Torch model and contract must match the actor loader's exact
committed source hashes. Model class-family caching changed the Torch source
bytes while preserving initialized state, tensor schema and inference math.
The loader accepts archives from the explicitly named pre-cache producer and
the current producer under the same contract; this does not allow the old or
an unknown implementation to run. Archive/checkpoint SHA bindings, namespaces,
public evaluation flags and tensor validation remain required.

New exports record the current Torch source hash. Loading a pre-cache archive
preserves its producer hash and original archive SHA; the returned execution
receipt separately records `loader_torch_source_sha256` and
`loader_contract_source_sha256`. Existing plans and archives are not rewritten.
Actor loading needs no Torch installation; checkpoint export still needs Torch.
Source compatibility does not attest runtime libraries or qualify production
collection: execution receipts must also bind source, runtime and test evidence.
