# In-crate TorchScript leaf evaluation (track D)

Status: 2026-07-18. Lands the model INSIDE `rust/pokezero-search` so MCTS
leaf evaluation never crosses the Python bridge per-leaf — the track D
deliverable of `test_time_search_plan_v3.md` ("Integration endgame") built on
the export path proven in `model_export_findings.md` (TorchScript via tch-rs,
per its recommendation). Everything here is CRATE-side; wiring into
`search.py`'s policy interface is the integration step that follows A+B+C.

## What landed

- **Cargo feature `model`** on `pokezero-search`: `tch = "=0.24.0"`
  (optional). Default build is unchanged and libtorch-free.
- **`TorchScriptLeafEval`** (`src/model.rs`): loads a `scripts/export_model.py`
  artifact via `CModule`, runs batched forward (obs batch + optional
  legal-mask batch → tanh values, policy logits, masked-softmax priors)
  behind the `BatchLeafEval` trait.
- **Virtual-loss batched one-ply PUCT** (`batched_search_core`): leaves are
  collected under a provisional side-one loss, evaluated as one model batch,
  then the real values replace the virtual ones. Engine-terminal leaves
  resolve from the outcome without a model call.
- **Python surface** (`NativeLeafModel`): artifact load once; `eval_obs_flat`
  (parity/debug), `bench_forward` (forward-only throughput),
  `search_batched` (model-in-the-loop search). All three release the GIL
  around search + model (`py.detach`).
- **Parity gate**: `tests/test_crate_model_leafeval.py` feeds identical
  pre-encoded observations to the venv's torch and to the crate on the same
  TorchScript artifact.

## Build recipe (the one that worked)

```sh
scripts/build_search_crate_model.sh <venv-python>
```

which is: `LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1
maturin build --release --features model` + wheel install. Specifics:

- **tch 0.24.0 against the venv's torch 2.12.1.** torch-sys 0.24.0 expects
  libtorch 2.11.0; the bypass acknowledges the minor-version skew. The C++
  shims compile against the venv torch's real headers, and the parity gate
  below is the machine-checkable compatibility proof. Bump tch and drop the
  bypass together when a matched release exists; re-run the gate after any
  bump.
- **No vendored libtorch, ever** — the crate links the torch the venv
  already ships (one runtime for Python-side and in-crate inference; also
  what makes bit-exact parity expected rather than hoped).
- **rpath embedding** (`build.rs`, model feature only): the extension gets an
  rpath to the venv's `torch/lib`, so `import pokezero_search` works without
  importing torch first.

## Artifacts (per-device, and what "checkpoint" means here)

- CPU: `python scripts/export_model.py --checkpoint <ckpt> --out-dir exports/
  --formats ts --validate` → `exports/model_ts.pt`.
- MPS: traces bake device constants (`model_export_findings.md` caveat), so
  the MPS artifact is re-traced on MPS (same shim, `.to("mps")` inputs) —
  the bench used `exports/model_ts_mps.pt`. A CUDA deployment likewise
  traces on CUDA at export time.
- **The measured artifact is a randomly-initialized v2.2-shaped model**
  (10.20M params, d=512, 3 layers, window=1, 151 tokens × 51 categorical +
  155 numeric — `exports/synthetic_v22_random.pt`, saved through the real
  checkpoint format). No real v2.2 checkpoint exists on this machine
  (`checkpoints/` is the v1 no-belief line, refused by the loader). Parity
  and throughput are weight-value-independent, so the numbers transfer; no
  strength claim is made or possible here.

## Parity (the machine-checkable claim)

`python -m unittest tests.test_crate_model_leafeval` — identical random
valid observations through (a) venv torch running the TorchScript artifact
and (b) the crate's `TorchScriptLeafEval` via `eval_obs_flat`:

| Artifact | Outputs compared | Max abs diff |
|---|---|---|
| tiny trace (1 layer, d=16), batches 1/4/64 | policy logits, value, priors | **0.0 (bit-exact)** |
| full-size 10.2M v2.2-shaped trace, batch 8 | policy logits, value | **0.0 (bit-exact)** |

Also gated: masked-prior parity vs `softmax(logits.masked_fill(~legal, -inf))`
with exactly zero mass on illegal actions, batched-search visit conservation,
seed determinism, and the batch=1 sequential-regime path.

## Batching design and the fidelity tradeoff

**Chosen: virtual-loss batching** (round of `batch` selections, each arm pair
provisionally scored as a side-one loss, one batched forward, then real
values swap in). At one ply, frontier collection without virtual loss is
degenerate — selection is deterministic given the stats, so all `batch`
leaves would be the same arm pair; the virtual loss is the minimal mechanism
that makes batched selection well-defined, and it reuses the existing
decoupled-PUCT stats untouched.

Fidelity cost at small sims/decision: one round of `batch` virtual-loss
selections explores wider than `batch` sequential PUCT steps (the loss only
discourages, never forbids, re-selection). At `batch >= sims` the search
degrades toward a prior-weighted sweep. Keep `batch <= sims/4`; the tables
below show the throughput knee sits at batch 16–64 anyway, so nothing is
paid for staying there. `batch=1` is the sequential regime by construction
— each round's single virtual loss is replaced by its real value before the
next selection, so no selection ever observes provisional stats. The gate
asserts the per-round mechanics and seed determinism at batch=1; it does
NOT compare visit distributions against an independent sequential
implementation (none exists — the batched core at batch=1 IS that path).

Two scoping facts, stated plainly:

- **This benchmark uses template leaf observations.** The per-leaf copy prices
  tensor marshaling, not encoding; the schema-bound Rust encoder now exists
  upstream of `BatchLeafEval`, but this historical throughput measurement does
  not exercise it. Throughput is real (forward cost is value-independent);
  leaf CONTENT — and therefore search strength — is not evaluated here.
- **Model priors are not yet wired into selection** (uniform priors remain).
  Priors come out of the evaluator per the contract, but mapping action
  indices onto poke-engine `MoveChoice`s is action-schema/encoder territory.

## Measured: model-in-the-loop searches/sec

Apple M5 Max laptop, minimal 1v1 gen3 fixture (`poke_engine_adapter`),
10.2M v2.2-shaped artifact, ≥3s measured per cell after one warm search
(`scripts/bench_crate_search.py`). "python-seq est." is the
`engine_search_poc.md` ladder's 3.2 ms/eval CPU rate × sims — the
Python-side sequential loop this replaces. Caveat: on this near-terminal
fixture ~30–35% of leaves resolve engine-terminal (no forward), which
flatters sims/s relative to a midgame state by roughly that fraction;
forward-only columns give the pure model ceiling. Column label, precisely:
"model-priced sims/s" = ALL sims per wall second — model-forwarded leaves
AND free engine-terminal ones (the per-cell `model_evals`/`terminal` split
is printed by the bench script).

### CPU (`exports/model_ts.pt`)

| batch | sims/decision | searches/s | ms/decision | model-priced sims/s | forward-only evals/s (b) | python-seq est. ms/decision (3.2ms/eval) |
|---|---|---|---|---|---|---|
| 1 | 64 | 3.61 | 277 | 231 | 180 | 205 |
| 16 | 64 | 6.02 | 166 | 386 | 273 | 205 |
| 64 | 64 | 6.03 | 166 | 386 | 306 | 205 |
| 256 | 64 | 6.23 | 161 | 399 | 320 | 205 |
| 1 | 256 | 0.79 | 1,271 | 201 | 180 | 819 |
| 16 | 256 | 1.49 | 672 | 381 | 273 | 819 |
| 64 | 256 | 1.72 | 582 | 440 | 306 | 819 |
| 256 | 256 | 1.60 | 625 | 409 | 320 | 819 |
| 1 | 1024 | 0.20 | 4,994 | 205 | 180 | 3,277 |
| 16 | 1024 | 0.38 | 2,626 | 390 | 273 | 3,277 |
| 64 | 1024 | 0.45 | 2,201 | 465 | 306 | 3,277 |
| 256 | 1024 | 0.34 | 2,899 | 353 | 320 | 3,277 |

### MPS (`exports/model_ts_mps.pt`)

| batch | sims/decision | searches/s | ms/decision | model-priced sims/s | forward-only evals/s (b) | python-seq est. ms/decision (3.2ms/eval) |
|---|---|---|---|---|---|---|
| 1 | 64 | 4.53 | 221 | 290 | 208 | 205 |
| 16 | 64 | 19.00 | 53 | 1,216 | 1,217 | 205 |
| 64 | 64 | 14.84 | 67 | 950 | 1,404 | 205 |
| 256 | 64 | 22.24 | 45 | 1,423 | 1,417 | 205 |
| 1 | 256 | 0.99 | 1,013 | 253 | 208 | 819 |
| 16 | 256 | 5.96 | 168 | 1,527 | 1,217 | 819 |
| 64 | 256 | 5.09 | 196 | 1,304 | 1,404 | 819 |
| 256 | 256 | 2.26 | 442 | 579 | 1,417 | 819 |
| 1 | 1024 | 0.26 | 3,800 | 269 | 208 | 3,277 |
| 16 | 1024 | 1.48 | 675 | 1,516 | 1,217 | 3,277 |
| 64 | 1024 | 0.84 | 1,190 | 861 | 1,404 | 3,277 |
| 256 | 1024 | 0.40 | 2,470 | 415 | 1,417 | 3,277 |

Reading:

- **The regime prediction holds: the model is the loop and batching+device
  is the lever.** CPU saturates at ~380–465 model-priced sims/s (vs ~310–330
  forward-only evals/s — terminal leaves account for the excess); MPS at
  batch 16 sustains ~1,200–1,530 sims/s, ~4–5× the Python sequential ladder
  and ~6× in-crate batch-1.
- **CPU batched search beats the Python-side sequential estimate ~1.4×**
  (e.g. 582 vs 819 ms at sims=256) — saturation plus everything around the
  forward being free. In-crate batch-1 ≈ the ladder, as expected: same
  forward, same single stream.
- **batch 16 is the MPS sweet spot in-search** even though forward-only
  peaks at 64–256. Cause: partial final batches. `narrow` to the leftover
  round size changes the input shape, and MPS recompiles its
  shape-specialized graph (visible as batch 256 collapsing to 415–579
  sims/s at sims 256/1024, where most rounds are partial). Known fix if
  batch ≥ 64 on MPS ever matters: pad the last round to the fixed batch
  shape and discard the padded outputs.
- Search-tree overhead is unmeasurable at this model size: the engine side
  of the loop runs at ~1M steps/s (crate README), 3–4 orders of magnitude
  under the forward.

## What's left before the paired FoulPlay read (per the v3 plan)

1. **Track B encoder into `ObsBatch`** — real leaf observations replace the
   template stub (the boundary was shaped for exactly that hand-off), fold
   state advanced per chance branch per the search-tree contract.
2. **Action mapping for priors** — policy-logit indices → `MoveChoice`s so
   the model prior replaces uniform in selection.
3. **World constructor at the root** (track A, landed) + `search.py`
   integration behind the existing search-policy interface (track D's
   `search.py` ownership; not touched here).
4. **Multi-ply + exact chance-node expectation** — the plan's search-tree
   contract; this one-ply loop is the batching substrate, not the final tree.
5. Then the single 200-seed paired read at fixed wall-clock, per the
   validation-gates doctrine.
