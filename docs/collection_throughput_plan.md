# Collection throughput plan — validation-gated optimization of the CPU rollout path

Status: executable plan, 2026-07-08. Written to be run **independently by an agent**;
every workstream specifies implementation, the exact verification evidence required,
and hard stop rules. Fleet/deployment rollout is out of scope here and handled in the
private deployment planning docs — this plan is local-machine code work only.

## Goal

Reduce per-decision cost of the collection path (self-play rollout on CPU) so a ~5×
larger model holds today's iteration cadence. Target: **≥3× combined speedup of the
`policy_select` phase** and an instrumented, attributed `env_step` phase, with zero
behavior change (parity-gated).

## Ground rules (read before any code)

1. **Measure → gate → optimize, in that order.** Twice this week a confident
   hypothesis was overturned by a finer measurement (the storage-cap directory walk
   masqueraded as materialize cost; a predicted 10–15 s asarray was actually 1.4 s).
   No optimization lands without before/after numbers from the standard harness below.
2. **Parity gates are hard requirements.** If a gate fails, STOP and report the
   failure — never loosen a threshold to pass. A parity failure is itself a finding.
3. **Collection path only.** Eval, benchmark, and training forwards are out of scope
   and must be provably untouched. No changes to rewards, environment semantics,
   observation schema, or cache formats.
4. **One PR per workstream, sequential.** Each PR body carries the evidence tables.
   Follow repo conventions (feature branch, tests, no unrelated diffs).
5. Repo/tooling facts: run tests with `.venv/bin/python -m unittest` (pytest-style
   files individually via `uv run --with pytest <file>`); macOS has no `timeout`
   command; never `git add -A` (untracked `checkpoints/`, dirty vendored submodule);
   do not commit smoke outputs or caches.

## Standard benchmark harness (use for every measurement)

```sh
OUT=$(mktemp -d /tmp/collect-bench-XXXX)   # CLEAN parent every run (cap-walk artifact)
POKEZERO_ARRAYS_TIMING=1 .venv/bin/python -m pokezero.rollout_cli collect-training-cache \
  --games 20 --out "$OUT/cache" --overwrite \
  --showdown-root "$SHOWDOWN_ROOT" \
  --p1-policy "neural:$CKPT" --p2-policy "neural:$CKPT" \
  --max-decision-rounds 250 --seed-start "$SEED"
```

- `$CKPT`: a local v2.2 checkpoint (e.g. `checkpoints/pz-v2-2-1m.pt` if present —
  any census-155 checkpoint works; record which one in every report).
- `$SHOWDOWN_ROOT`: your local pokemon-showdown checkout.
- **Protocol**: 3 runs at `--seed-start 1000/2000/3000`; report per-phase mean ± spread.
- **Determinism property (basis of all parity gates)**: with a fixed seed set and a
  fixed policy, trajectories are reproducible. Compare **per-step `action_index`
  sequences and terminal winners** extracted from the records — NOT raw record bytes
  (per-step timing fields embedded in records are nondeterministic until that is
  fixed; do not let them fake a parity failure).

## Measured baseline (2026-07-08, 20-game neural smoke, 10M v2.2 checkpoint, local CPU)

| phase | time | share |
|---|---|---|
| env_step (node sim + IPC + parse + encode) | 4.91 s | 45% |
| policy_select (NN forward, ~2.9 ms × 1,576 calls) | 4.61 s | 42% |
| cache flush_write (post cap-fix; materialize 0.69 s) | 0.71 s | 6.5% |
| cap_dir_walk (clean parent / post-fix) | ~0 | — |

Padding measurement (from the smoke cache's `attention_mask.npy`): mean real tokens
**103.5 / 151** (p50 = 110) → **31.5% of every forward is padding**.

If your re-measured baseline differs from this table by >±30% on any phase, STOP,
update this section in your PR, and re-derive the expected gains before proceeding.

---

## WS-0 (mandatory first): split `env_step` — instrument, don't guess

The 45% bucket has never been decomposed. Everything in WS-3+ depends on knowing
what's inside it.

**Implementation**: extend the existing env-guarded phase-timing pattern
(`POKEZERO_ARRAYS_TIMING` precedent) with `POKEZERO_ENV_TIMING=1`, off by default,
timing three sub-phases inside the env step path (locate in `src/pokezero/env.py` /
the local-showdown env implementation):
- `env_ipc` — write choice → read complete protocol chunk from the node process
- `env_parse` — protocol-line parsing / state update
- `env_encode` — observation encode (the 155+51-column build)

**Verification evidence (required in PR body)**:
- Standard harness ×3 seeds: sub-phase table; sub-phases must sum to ≥95% of
  `env_step` (else the instrumentation has a hole — find it).
- Zero overhead when the env var is unset: total wall within noise of pre-change.
- Determinism: fixed-seed action sequences identical before/after (instrumentation
  must not perturb behavior — see the record-timing caution above).

**Decision output**: which sub-phase dominates. `env_encode` >30% → flag encode
optimization as a follow-up workstream (cache static species/move lookups, avoid
rebuilding invariant rows). `env_ipc` >40% → flag node-process stream consolidation.
Do NOT implement either in this PR — report the numbers and the recommendation.

## WS-1: sequence-length trimming (measured 31.5% padding; expected ≥1.3× forward)

**Implementation**: in the collection-path policy forward only (the batch-1 inference
wrapper in `src/pokezero/neural_policy.py` used by `policy_select`): compute the real
token length L from the attention mask and slice all per-token inputs
(categorical/numeric/token-type/mask) to `[:L]` before the transformer. Positions
0..L-1 are unchanged, so positional embeddings are unaffected. Training and
benchmark forwards untouched.

**Verification evidence (in order; stop at first failure)**:
1. **Logit parity (the load-bearing gate)**: dump a fixed corpus of ≥500 encoded
   observations from a smoke cache; forward each with and without trimming;
   require max |logit delta| < 1e-5 over legal actions and identical argmax on
   100% of the corpus. *If this fails, padding is influencing outputs — that is a
   masking bug worth its own report; STOP.*
2. **End-to-end determinism**: 20-game fixed-seed collect, trimmed vs untrimmed →
   identical per-seed action sequences and winners.
3. **Speed**: standard harness ×3 seeds → `policy_select` reduced ≥25%
   (prediction: 30–45%; FF scales with L, attention with L²).

## WS-2: int8 dynamic quantization of the collection policy (expected ≥1.8×)

**Implementation**: `torch.ao.quantization.quantize_dynamic(model, {nn.Linear},
dtype=torch.qint8)` applied to the collection policy **behind a default-off flag**
(`--collection-int8` / `POKEZERO_COLLECT_INT8=1`). Never applied to eval, benchmark,
or training paths. Composes with WS-1 (quantize the trimmed-forward model).

**Semantics note the PR must state**: quantization slightly changes the *acting*
policy while training updates the fp32 weights — an off-policy shift on top of the
tolerated 1-iteration pipeline staleness. The gates below bound the shift locally;
**enabling the flag in production requires a matched training A/B run** (out of
scope here; deployment decision). The PR lands flag-off.

**Verification evidence**:
1. **Policy-shift bound** on the ≥500-obs corpus: argmax agreement ≥99%; mean total
   variation distance between action distributions <0.01. Report both numbers.
2. **Strength parity**: the repo's neural benchmark, quantized vs fp32, same
   checkpoint, ≥600 games vs each scripted opponent (random-legal, simple-legal,
   max-damage): every win-rate delta within 2 percentage points (≈1σ at n=600).
3. **Speed**: standard harness ×3 seeds → `policy_select` reduced ≥45% vs the WS-1
   baseline (int8 linears alone predict ~2–3× on the linear-dominated forward).

## WS-3: multi-env micro-batched inference (expected 2.5–5×; largest diff — LAST)

Start only after WS-0/1/2 are merged, and only if their combined measured
`policy_select` speedup is <3× — otherwise report and ask whether the added
complexity is still wanted (it remains strategically useful: it prototypes the
batched-serving machinery larger models will need).

**Implementation**: `--parallel-envs K` (default 1 = exactly today's code path) in
the collect loop: K envs advance concurrently (one node stream each, as today);
pending decisions are gathered and executed as one batched forward, actions routed
back. Per-env RNG/seed handling unchanged. Thread pool or asyncio — pick whatever
keeps the K=1 path byte-identical.

**Verification evidence**:
1. **Determinism across K**: a fixed 20-seed game set run at K=1 and K=8 →
   *identical per-seed action sequences and winners* (batching must not leak state
   or reorder RNG consumption within an env).
2. **Scaling curve**: games/min at K ∈ {1, 4, 8, 16} on the same machine, cores
   noted; require ≥2.5× at K=8 vs K=1.
3. K=1 regression check: standard harness within noise of pre-change.

---

## Explicit don't-do list (measured/analyzed this week — do not spend effort here)

- **Torch thread-count increases** — fleet-neutral: cores are the budget; 1-thread
  pod-packing is already optimal at fleet level.
- **fp16 CPU inference** — frequently slower without AVX512-FP16; int8 is the route.
- **Arrays-end-to-end refactor of record rows** — measured materialize is
  0.39 ms/example (~1–2% of a real run); not worth the churn.
- **Further cache-write optimization** — 6.5% and shrinking post cap-fix; the
  remaining NFS concerns are deployment-side, not code-side.

## Sequencing, reporting, stop rules

- Order: WS-0 → WS-1 → WS-2 → summary report → WS-3 (conditional, see its gate).
- Each PR body: the relevant evidence tables, the exact harness commands run, the
  checkpoint used, and machine/core count.
- After WS-2, post a summary comparing achieved vs predicted speedups and the WS-0
  attribution table — that report decides WS-3 and any encode/IPC follow-ups.
- **STOP and report instead of proceeding when**: any parity gate fails; the
  re-measured baseline deviates >±30% from this doc; a workstream exceeds ~2× its
  apparent scope; or a change would touch training/eval semantics to make a gate
  pass. A blocked-with-evidence stop is a successful outcome; a green build with a
  loosened gate is not.
