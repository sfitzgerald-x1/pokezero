# Model export + native-inference throughput (engine-swap stream S2)

> Follow-up landed: the recommended tch-rs path is now implemented in the
> crate — in-crate TorchScript leaf evaluation, parity gate (bit-exact), and
> model-in-the-loop batched-search benches live in
> [`crate_model_integration.md`](crate_model_integration.md).

Proves the export path the native search crate depends on
(docs/test_time_search_plan_v3.md, "Integration endgame": TorchScript via
tch-rs or ONNX via onnxruntime) and prices what each runtime buys over the
eager-torch numbers in docs/engine_search_poc.md's model-cost ladder.

Checkpoint: the v2.2 1M-games policy (10.2M params, d=512, 3 layers,
window=1, 151 tokens x 51 categorical + 155 numeric). Machine: Apple M5 Max
laptop (6P+12E), torch 2.12.1, onnxruntime 1.27.0. Inputs dummy-filled at
checkpoint shapes (timing is encoding-independent, same methodology as the
ladder). Only the expanded-observation forward path is exported; the
row-indexed training path is not needed at inference.

## Export status

| Format | Exporter | Artifact | Status |
|---|---|---|---|
| TorchScript | `torch.jit.trace` (positional-args shim) | `model_ts.pt` (40.8 MB) | works, dynamic batch verified |
| ONNX | `torch.onnx.export` dynamo (torch.export-based) | `model.onnx` + `model.onnx.data` (40.8 MB external weights) | works, dynamic batch axis verified |
| ONNX | legacy TorchScript-based exporter | — | **does NOT work**: fails on `aten::_transformer_encoder_layer_fwd` (fused encoder kernel has no ONNX lowering) |

`torch.onnx.export` did NOT choke: the model is a plain pre-norm
transformer encoder (`norm_first=True` disables the nested-tensor fastpath)
and both exporters handle the bool `src_key_padding_mask` and the embedding
clamps. The dynamo exporter requires `onnxscript` (now in the venv); the
ONNX export HARD-REQUIRES the dynamo exporter (`onnxscript` installed); the
legacy exporter fails on the fused `aten::_transformer_encoder_layer_fwd`
op, so there is no fallback — `--onnx-exporter auto` errors with guidance
when dynamo is unavailable.

## Parity (--validate: 64 random valid inputs + batch-1 re-check, vs eager)

| Format | policy_logits | value | opponent_action_logits | Verdict (tol 1e-4) |
|---|---|---|---|---|
| TorchScript | 0.0 | 0.0 | 0.0 | PASS (bit-exact) |
| ONNX (ORT CPU EP) | 2.4e-06 | 4.2e-07 | 3.3e-06 | PASS |

Validation batches use seeds distinct from the trace batch (batch 4), so a
trace that silently baked the batch dimension in cannot pass.

## Throughput (evals/s, batch rows per second; ms = per-batch latency)

| Runtime | batch 1 | batch 16 | batch 64 | batch 256 | Note |
|---|---|---|---|---|---|
| eager-cpu | 188/s (5.3ms) | 305/s (52.4ms) | 329/s (194.3ms) | 325/s (788.9ms) | |
| ts-cpu | 128/s (7.8ms) | 281/s (56.8ms) | 333/s (192.0ms) | 328/s (779.9ms) | |
| ort-cpu | 260/s (3.8ms) | 307/s (52.1ms) | 307/s (208.7ms) | 280/s (915.6ms) | best intra_op of {1,2,4,8}: 4-8 |
| eager-mps | 158/s (6.3ms) | 1,610/s (9.9ms) | 1,635/s (39.1ms) | 1,602/s (159.8ms) | |
| eager-mps-fp16 | 336/s (3.0ms) | 577/s (27.7ms) | 1,639/s (39.0ms) | 2,593/s (98.7ms) | torch.autocast(float16) |
| ts-mps | 388/s (2.6ms) | 1,617/s (9.9ms) | 1,655/s (38.7ms) | 1,603/s (159.7ms) | re-traced on MPS (see caveats) |
| ort-coreml | - | - | - | - | EP present but cannot compile this graph (see caveats) |

Variance: batched (>=16) rows repeat within ~5% across two full runs;
batch-1 rows are noisy (eager-cpu 173-188/s, ts-cpu 128-171/s, eager-mps
158-330/s observed) — treat batch-1 as latency-order-of-magnitude only.

What the table says:

- **CPU is compute-bound at ~310-330 evals/s for every runtime** —
  consistent with the ladder's 309/s (eager, batch 64). Swapping runtimes
  buys ~nothing at saturated batch on CPU; the native crate's CPU-side win
  remains the elimination of Python orchestration, not faster forwards.
- **ORT-CPU wins only at batch 1** (3.8ms vs 5.3-7.8ms, ~1.5-2x): tuned
  intra-op threading amortizes better on a single row. Relevant if the
  crate ever prices single root evals on CPU.
- **MPS is the ~5x lever on this box**: ~1,600-1,650 evals/s fp32 at any
  batch >= 16 (matches the ladder's ~1,700). TorchScript-on-MPS matches
  eager-MPS — no speedup, none expected; the model is a few large matmuls.
- **fp16 (autocast) pays only at large batch**: 1.6x at batch 256
  (2,593/s), a wash at 64, worse at small batch (per-layer cast overhead).
  Worth wiring for batched collection workloads; irrelevant for search at
  batch <= 64 until a real half-precision weight copy is used instead of
  autocast.

## Recommended runtime for the native crate: tch-rs (TorchScript), ONNX kept as validated fallback

- **Parity**: TorchScript is bit-exact against eager (0.0 max abs diff), so
  the golden-corpus discipline transfers to the crate without a tolerance
  budget. ONNX is 2.4e-06 — fine, but nonzero.
- **GPU reach**: tch-rs drives MPS today (the only native >1k evals/s path
  on the dev laptop, since ORT's CoreML EP rejects the graph) and CUDA on
  the cluster, which is the regime track B actually targets. ORT on this
  laptop is CPU-only in practice.
- **Cost accepted**: libtorch linkage in the crate, and per-device trace
  artifacts (below).
- **Keep ONNX anyway**: the export is proven and parity-validated, `ort`
  is the escape hatch if libtorch linkage becomes painful, and ORT-CPU has
  the best single-eval CPU latency. Publishing both artifacts costs one
  script invocation.

## Caveats

- **Traces bake device constants.** The eager forward calls
  `torch.arange(..., device=x.device)`; under `torch.jit.trace` the device
  is frozen into the graph, so a CPU trace loaded with `map_location="mps"`
  fails at runtime ("Placeholder storage has not been allocated on MPS
  device"). Produce one trace per target device (the bench re-traces on
  MPS; export_model.py emits the CPU artifact). Same applies to a future
  CUDA deployment via tch-rs: trace on CUDA in Python at export time.
- **ORT CoreML EP cannot run this graph** (ort 1.27.0, both exporters, both
  MLProgram and NeuralNetwork formats): the dynamic batch axis hits
  "unbounded dimension which is not supported [see correction below]" in CoreML's MIL compiler,
  and the value head's `squeeze(-1)` trips "Invalid tensor rank 0 inferred
  from: ios18.squeeze". A fixed-batch export might compile but forfeits the
  flexible batching the search loop needs; not pursued.
- **Dynamo-exported ONNX is two files** (`model.onnx` + `model.onnx.data`
  external weights). Ship both files together; there is no single-file
  fallback (the legacy exporter cannot lower the fused encoder op).
- **CPU batch-256 is past saturation** (ort-cpu regresses to 280/s there);
  64 remains the CPU sweet spot, as in the ladder.
- Throughput is forward-pass only — no v2.2 encoding, no tensor marshaling
  from the engine. Per-leaf encoding cost is tracked separately as the
  known next wall (v3 plan, "Integration endgame").

## Repro

```bash
python scripts/export_model.py --checkpoint <ckpt.pt> \
    --out-dir exports/ --formats ts,onnx --validate
python scripts/bench_inference.py --checkpoint <ckpt.pt> \
    --export-dir exports/ --batch-sizes 1,16,64,256 --out bench.md
python -m unittest tests.test_export_model
```

## Corrections after independent review (2026-07-18)

- **Legacy ONNX exporter does not work** on this model (fails on
  `aten::_transformer_encoder_layer_fwd`); ONNX requires the
  dynamo/onnxscript path. The table above is corrected accordingly.
- **CoreML failure mechanism**: with the shipped dynamo external-data
  artifact, the CoreML EP fails at session init (external-weights load:
  `model_path must not be empty`) BEFORE reaching the MIL compiler; the
  rank-0 squeeze/unbounded-dim errors were observed on a different artifact
  variant. Conclusion unchanged: CoreML EP unusable today.
- **fp16 numbers are throughput probes only** — fp16 changes outputs and has
  NO parity validation; do not wire fp16 into collection (it feeds the
  training distribution) without a dedicated parity/quality story.
- **Bench fairness**: ORT intra_op threads were tuned per batch; torch CPU
  used default threading. The asymmetry favors ORT, which still does not win
  at saturated batch, so the "CPU is compute-bound" conclusion is robust.
