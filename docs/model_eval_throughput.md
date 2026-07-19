# Real-checkpoint eval throughput: emeta-v2-2-lr3m vs metamon-m50 (engine-swap stream)

Status: 2026-07-19. First REAL-weights pass of the model-integration line
(`docs/model_export_findings.md` proved the export path on a synthetic
v2.2-shaped model; `docs/crate_model_integration.md` measured the in-crate
loop with random weights). This study fetches real cluster checkpoints,
re-proves export parity on them, prices device x dtype x batch throughput
for BOTH model scales, measures the MPS partial-final-batch recompile
regression with real shapes, and validates the documented-but-unbuilt fix
(pad-to-batch). Machine: Apple M5 Max laptop (CPU + MPS as the local GPU
proxy); the cluster-GPU rerun is specced at the end and is a config change,
not new code.

## Checkpoints (fetched 2026-07-19, sha256-verified; `checkpoints/curated/MANIFEST.json`)

| File | Run / role | Games | Params / config |
|---|---|---|---|
| `emeta-v2-2-lr3m-belief-final-3000k-iter0625.pt` | v22-lr3m lineage FINAL (3m leg iter 0625) | 3,000,000 | 10.18M, d=512, 3 layers |
| `emeta-v2-2-lr3m-belief-hifi-argmax-1900k-iter0563.pt` | v22-lr3m lineage HI-FI ARGMAX (2m leg iter 0563) | 1,900,800 | 10.18M, d=512, 3 layers |
| `metamon-m50-lr10m-ep7-latest-3574k-iter2234.pt` | metamon-m50-2m-lr10m-ep7 LATEST (still training) | 3,574,400 | 51.84M, d=1024, 4 layers |

Which emeta checkpoint is which matters (owner policy, 2026-07-17 tail-collapse
finding): **strength claims use the hi-fi argmax, never the final.** The
lineage-wide hi-fi timeline (50 milestones, 550k-3.0M) puts the foul-play
argmax at milestone 1,900,000 (win rate 0.388 over 1000 games; the
max-damage argmax is milestone 2,950,000 at 0.9365 — not fetched). The FINAL
checkpoint is the throughput/parity workhorse here; both emeta checkpoints
share one architecture, so every throughput number covers both.

## Export + parity on real weights (Part 2)

`scripts/export_model.py` grew a `--device {cpu,mps,cuda}` flag: traces bake
device constants (findings doc), so each device gets its own artifact
(`model_ts.pt`, `model_ts_mps.pt`, later `model_ts_cuda.pt`).

| Model | Artifact | Parity vs eager (policy+value, random inputs) |
|---|---|---|
| emeta final | `exports/emeta-final/model_ts.pt` (CPU) | 0.0 — bit-exact |
| emeta final | `exports/emeta-final/model_ts_mps.pt` (MPS) | 0.0 — bit-exact (vs eager-on-MPS) |
| m50 latest | `exports/m50-latest/model_ts.pt` (CPU) | 0.0 — bit-exact |
| m50 latest | `exports/m50-latest/model_ts_mps.pt` (MPS) | 0.0 — bit-exact (vs eager-on-MPS) |

**m50 is exportable by the same path** — it is a bigger config of the same
family (d=1024, 4 layers, window=1, same 151x51+155 token layout), so the
window=1 exporter guard and the positional-args shim apply unchanged.

Non-degeneracy on REAL observations (`scripts/verify_real_checkpoint_outputs.py`,
first 64 golden-corpus-v2 rows, both devices):

| Model | eager-vs-TS on real rows | value head | masked priors |
|---|---|---|---|
| emeta final | 0.0 (bit-exact) | min -0.331, max 0.901, std 0.263, 64/64 unique | rows sum to 1.000000, illegal mass exactly 0 |
| m50 latest | 0.0 (bit-exact) | min -0.696, max 0.892, std 0.319, 64/64 unique | rows sum to 1.000000, illegal mass exactly 0 |

(Provenance note: the 64-row battery is the full regenerated golden-corpus-v2
random battery — regenerate per docs/golden_corpus_notes.md to reproduce; the
committed 5-row sample gives the same parity/illegal-mass results but value-std
at n=5 is not comparable to the n=64 figures above.)

## Throughput grid (Part 3) — `scripts/bench_model_eval.py`

Method: fp32 cells are the TorchScript runtime (what the crate runs via
tch-rs), traced in-process on the target device — identical to the shipped
artifact (traces are bit-exact vs eager). Steady state: 3 warmup forwards
then >=1.5s / >=4 iterations measured, device-synchronized; trace and
compile time excluded. Random dummy-filled inputs at checkpoint shapes
(timing is encoding-independent — same methodology as the model-cost
ladder); per-leaf ENCODING cost is the separate track-B wall and is not
priced here.

Three fp16-on-MPS findings (torch 2.12.1, macOS 25.5), which shaped the
method:

1. **A REAL half-precision weight copy cannot run on MPS at all** — both
   `torch.jit.trace` of the half model AND the plain eager half forward
   hard-abort in Metal (`MPSNDArrayMatrixMultiplication`: "Destination
   NDArray and Accumulator NDArray cannot have different datatype"). The
   fused transformer-encoder kernel demands an fp32 accumulator with an
   fp32 destination. So the fp16 rows below are **eager autocast** (fp32
   weights, per-op fp16 casts — the same probe model_export_findings.md
   measured), selected by `--fp16-runtime auto`. Real-half TorchScript is
   expected to work on CUDA (`--fp16-runtime ts`) — part of the cluster
   experiment below.
2. **autocast-fp16 at batch 512 stalls >10 minutes on FIRST compile, then
   runs normally.** The first forward at that shape blocked so long
   (process alive, ~12% CPU) the run was initially declared wedged and
   killed — but a surviving instance eventually got through and finished
   the cell at a healthy 2,725/s, 187.9ms/batch
   (`exports/bench_emeta_full_earlier_run.json`). 512 rows x 151 tokens =
   77,312 flattened matmul rows — the first grid point past 2^16 —
   pointing at a pathological Metal fp16 shader-compile path above that
   dimension. One-time per shape (the OS shader cache then holds it), but
   operationally disqualifying for cold starts, so the split fp16 grids
   stop at batch 256 and the 512 number is quoted from the completed
   earlier run.
3. fp16 numbers remain throughput probes only: no parity story, never feed
   collection (model_export_findings.md corrections).

### emeta (10.18M, d=512, 3 layers) — evals/s (ms/batch)

| device/dtype | batch 1 | batch 16 | batch 64 | batch 128 | batch 256 | batch 512 |
|---|---|---|---|---|---|---|
| cpu/fp32 (ts) | 226/s (4.4ms) | 258/s (62.0ms) | 317/s (202.0ms) | 338/s (379.2ms) | 352/s (727.6ms) | 294/s (1742.1ms) |
| mps/fp32 (ts) | 448/s (2.2ms) | 1,753/s (9.1ms) | 1,815/s (35.3ms) | 1,794/s (71.4ms) | 1,758/s (145.6ms) | 1,729/s (296.1ms) |
| mps/fp16 (autocast) | 515/s (1.9ms) | 1,405/s (11.4ms) | 2,818/s (22.7ms) | 2,862/s (44.7ms) | 2,795/s (91.6ms) | 2,725/s (187.9ms)* |

\* from the completed earlier run (finding 2 above: >10-min first compile).

### m50 (51.84M, d=1024, 4 layers) — evals/s (ms/batch)

| device/dtype | batch 1 | batch 16 | batch 64 | batch 128 | batch 256 | batch 512 |
|---|---|---|---|---|---|---|
| cpu/fp32 (ts) | 62/s (16.2ms) | 86/s (185.3ms) | 84/s (764.3ms) | 86/s (1496.3ms) | 70/s (3663.9ms) | 80/s (6412.5ms) |
| mps/fp32 (ts) | 113/s (8.8ms) | 467/s (34.3ms) | 481/s (133.1ms) | 483/s (265.0ms) | 474/s (540.2ms) | 6/s (89,875ms)** |
| mps/fp16 (autocast) | 123/s (8.1ms) | 341/s (46.9ms) | 697/s (91.8ms) | 537/s (238.2ms) | 619/s (413.3ms) | - |

\** measured once (pre-rerun session, log-quoted): m50 fp32 falls off a
~70x-per-row cliff at batch 512 on MPS — same >2^16-flattened-rows regime
where emeta's fp16 pathology sits, but at d=1024 it hits fp32 too, and it
is the steady-state rate, not a one-time compile. That session's process
then died silently during the partial study (jetsam/Metal watchdog
suspected). m50 on MPS is a <=256-batch model; the batch-512 cell is
excluded from the rerun grids.

Reading the grids:

- **CPU is compute-bound and prices the models at their parameter ratio.**
  emeta saturates at ~350 evals/s, m50 at ~86 — a 4.1x cost ratio for a
  5.1x parameter ratio. Both models' CPU rates match their prior ladders
  (emeta ~330/s in `model_export_findings.md`), so nothing regressed with
  real weights.
- **MPS fp32: emeta ~1,750-1,815/s at any batch >=16; m50 ~467-483/s** —
  a ~3.8x gap, again tracking parameters. The MPS-vs-CPU lever is ~5.2x
  for emeta and ~5.6x for m50 (batch 64).
- **The throughput knee is batch 16 on MPS, 64 on CPU, for both models** —
  larger batches buy <=4%. This is good news for search fidelity: the
  batch<<sims discipline (`crate_search_design.md`) costs nothing; batch
  16-64 is both the fidelity-safe and the throughput-optimal region.
- **fp16 (autocast) is worth ~1.6x for emeta at batch >=64** (2,818-2,862/s
  vs 1,753-1,815/s) — same shape as the findings-doc probe. For m50 it is
  erratic (below) and cannot be trusted on MPS.
- **m50 autocast-fp16 on MPS is operationally unstable**: the grid is
  non-monotone (697/s at 64, 537/s at 128) and its partial study produced
  randomly catastrophic forwards (15-50s at warm shapes that also ran at
  86-360ms — `exports/bench_m50_fp16.json`). No such instability in fp32 or
  in emeta-fp16. m50's fp16 economics are a CUDA tensor-core question
  (cluster spec below), not an MPS one.

## MPS partial-final-batch regression + pad-to-batch fix

Setup (`--partial-study 256`): seven never-benched leftover sizes
(`sims % batch` stand-ins), each timed cold (first hit in this process),
warm (second hit), then the same rows padded to the fixed 256 shape via
`pad_obs_batch` (repeat-last-row filler, outputs sliced to the real rows),
first and second hit. emeta, MPS:

| leftover rows | fp32 cold | fp32 warm | fp32 padded 1st/2nd | fp16 cold | fp16 warm | fp16 padded 1st/2nd |
|---|---|---|---|---|---|---|
| 248 | 144.0 | 141.2 | 595.0 / 152.7 | 91.3 | 160.5 | 160.5 / 90.4 |
| 186 | 107.0 | 104.3 | 146.4 / 146.2 | 70.0 | 65.9 | 92.1 / 90.6 |
| 130 | 79.3 | 73.6 | 145.0 / 147.2 | 55.4 | 47.1 | 90.4 / 90.5 |
| 94 | 55.4 | 52.9 | 146.1 / 145.8 | 42.5 | 44.6 | 89.6 / 90.1 |
| 58 | 40.7 | 32.9 | 148.4 / 146.8 | 35.6 | 21.1 | 89.6 / 91.4 |
| 28 | 24.6 | 16.2 | 146.0 / 147.4 | 38.6 | 14.8 | 91.1 / 91.0 |
| 12 | 20.1 | 7.1 | 146.3 / 146.0 | 38.0 | 13.6 | 90.8 / 91.6 |

(ms per forward; steady-state full-256 batch: fp32 145.6ms, fp16 91.6ms.
Pad correctness: padded outputs match the direct partial forward to
5.96e-08 fp32 / 0.0 in the fp16 study — the padding is inert, as
`tests/test_bench_model_eval.py` also gates at unit level.)

m50, MPS fp32 (steady 540.2ms; pad diff 0.0):

| leftover rows | cold | warm | padded 1st/2nd |
|---|---|---|---|
| 248 | 528.7 | 1271.2 | 3483.1 / 664.3 |
| 186 | 393.0 | 410.7 | 561.5 / 541.9 |
| 130 | 277.7 | 597.7 | 1008.6 / 545.5 |
| 94 | 200.0 | 196.4 | 1540.2 / 545.1 |
| 58 | 124.5 | 122.2 | 963.4 / 572.4 |
| 28 | 66.6 | 59.1 | 550.4 / 544.2 |
| 12 | 33.3 | 25.3 | 1360.8 / 552.2 |

Same structure as emeta with more jitter (the 1271ms warm@248 outlier;
padded-1st spikes up to 3.5s) — d=1024 sits closer to whatever resource
edge the batch-512 cliff falls off. Padded-2nd again equals steady state
across the board.

The regression has three layers, and being precise about them matters:

1. **First-ever compile of a shape (cold OS shader cache) is the expensive
   layer** — observed on first-process runs: up to ~70ms extra per shape
   fp32 (12-row forward: 78.5ms first-process vs 7.1ms warm here) and up to
   ~554ms fp16 (smoke run), with the batch-512 fp16 pathology (>10 min,
   finding 2) as the extreme. These land on the first decisions a worker
   makes after boot.
2. **Per-process recompiles with a warm OS cache are mild**: the table's
   cold-vs-warm deltas are only ~3-25ms — macOS persists compiled shaders
   across processes, so a rebooted worker does not pay layer 1 again unless
   shapes are genuinely new.
3. **Warm partial forwards are proportionally CHEAP** — 12 rows cost 7.1ms
   vs 146ms for the full batch. Python/torch caches per-shape graphs, so a
   bounded leftover-shape space amortizes quickly.

**Pad-to-batch verdict: works exactly as designed, use it selectively.**
Padded forwards hold the fixed-shape steady state (padded-2nd == steady
within noise, both dtypes; the one 595ms padded-1st is the first-touch
allocation of the padded buffers) and eliminate shape churn entirely. But
padding buys full-batch COMPUTE for every leftover — 146ms for 12 real rows
that cost 7.1ms warm-direct. So: pad when shapes churn against a cold cache
(worker cold-start, wide `sims%batch` variety, and especially the tch-rs
in-crate loop where `crate_model_integration.md` measured batch-256 search
collapsing to 415-579 sims/s — the crate side does not enjoy this Python
shape-cache amortization), and skip it in long-lived Python workers whose
leftover shapes have warmed. `pad_obs_batch` in `scripts/bench_model_eval.py`
is the reusable reference implementation; the crate-side fix is the same
mechanism in `model.rs` (pad the final round's `ObsBatch` rows, drop the
padded outputs at unpack).

## Decision synthesis: sims/s at search batch sizes

At search batch sizes that respect batch<<sims (batch <= sims/4 per
`crate_search_design.md`; for a 1024-sim decision that allows up to 256,
and the knee means 16-64 is where you'd actually sit), sims/s ==
forward-only evals/s to first order (engine-side loop cost is 3-4 orders
below the forward; engine-terminal leaves only make these numbers
conservative). CAVEAT: these are forward-only UPPER BOUNDS, not end-to-end
guarantees — the crate's tch-rs loop measured 415-579 sims/s at batch 256 on
MPS (shape-churn overhead; the pad-to-batch fix above is what closes that
gap and must be applied crate-side before quoting these numbers for search).
Projection per 1024-sim decision:

| model | device/dtype | batch | evals/s (~sims/s) | s per 1024-sim decision | decisions/min |
|---|---|---|---|---|---|
| emeta | cpu/fp32 | 64 | 317 | 3.2 | 19 |
| emeta | cpu/fp32 | 256 | 352 | 2.9 | 21 |
| emeta | mps/fp32 | 16 | 1,753 | 0.58 | 103 |
| emeta | mps/fp32 | 64 | 1,815 | 0.56 | 106 |
| emeta | mps/fp16 | 64 | 2,818 | 0.36 | 165 |
| m50 | cpu/fp32 | 64 | 84 | 12.2 | 4.9 |
| m50 | mps/fp32 | 16 | 467 | 2.2 | 27 |
| m50 | mps/fp32 | 64 | 481 | 2.1 | 28 |
| m50 | mps/fp16 | 64 | 697* | 1.5 | 41 |

\* unstable on MPS (see above); listed for completeness only.

Where the tradeoff lands:

1. **emeta-on-CPU is ~70-75% of m50-on-GPU-proxy.** 317-352 evals/s on
   pure CPU vs m50's 467-483/s on MPS. On the cluster this is the
   decision-relevant shape: emeta search can run on CPU-only workers
   (zero GPU ask, horizontally scalable next to self-play collectors),
   while m50 search buys only ~1.4x over that at the cost of a GPU slot —
   before CUDA-vs-MPS differences, which the specced run below settles.
2. **emeta-on-GPU is the throughput play**: ~1,800/s fp32 (~0.56s per
   1024-sim decision, ~5.7x CPU emeta) and ~2,800/s under fp16 where a
   validated fp16 path exists. A single GPU worker sustains ~106
   1024-sim decisions/min — enough to run the 200-seed paired FoulPlay
   read at fixed wall-clock without sharding heroics.
3. **m50 search costs ~4x emeta everywhere** (parameter-ratio pricing on
   both devices), and on today's checkpoints buys no measured hi-fi
   strength: m50's lineage-best hi-fi foul-play is 0.386 (2.1M games;
   latest 3.55M sits at ~0.36-0.38) vs emeta's 0.388. Unless m50 pulls
   ahead by its 5M-game target, search experiments should default to
   emeta and spend the saved budget on sims. (Strength claims remain the
   paired-capstone harness's job, per the owner eval doctrine — this is a
   price-performance observation, not a crowning.)
4. **Batch sizing**: batch 16 already captures >=96% of MPS peak for both
   models, so the batch<<sims fidelity rule is free. Nothing supports
   batch >=512 anywhere locally: emeta-fp16 pays a >10-min first compile,
   m50-fp32 collapses ~70x, m50-fp16 destabilizes — cap local/MPS search
   batches at 256, prefer 16-64.

## What needs a real CUDA GPU (specced, NOT run — separate owner-approved step)

The open questions MPS cannot answer:

1. **m50 fp16 tensor-core throughput.** MPS half-rate says nothing about
   A100/H100/GB200 tensor cores; fp16/bf16 on CUDA is where the m50-at-search
   economics could flip.
2. **CUDA graph capture / `torch.compile` over the traced module** — MPS has
   no equivalent of CUDA graphs; small-batch launch overhead on CUDA is
   unknown here.
3. **Whether the partial-batch recompile regression exists on CUDA at all**
   (CUDA kernels are shape-agnostic where MPS specializes; expectation is
   "no regression", which would make pad-to-batch an MPS-only workaround —
   verify, don't assume).
4. **CPU-vs-GPU crossover under co-located self-play** (cluster CPUs differ
   from an M5 Max; the emeta-on-CPU option only matters if it frees GPUs).

Exact experiment (one job, <30 GPU-minutes):

- **Script (already cluster-ready):** for each of the two checkpoints:
  `python scripts/bench_model_eval.py --checkpoint <ckpt> --label <label>
  --devices cuda,cpu --dtypes fp32,fp16 --fp16-runtime ts
  --batch-sizes 1,16,64,128,256,512,1024 --partial-study 256
  --out-json /shared/.../bench_<label>_cuda.json`
  (`--fp16-runtime ts` = real half weights traced — the thing MPS cannot
  run; add `--batch-sizes ...,2048` for m50 fp16 — tensor-core saturation
  sits higher than MPS's). CUDA artifacts for the crate come from
  `scripts/export_model.py --device cuda --formats ts` on the same box.
- **Image:** the standard training image already used by this repo's cluster
  jobs (it carries the matching torch); no new image work.
- **Resource ask:** 1 GPU, 8 CPU, 32Gi, runAsJob with `ttlSecondsAfterFinished:
  600` and deletion as the last workflow step (owner rule). Checkpoints are
  already on /shared — no data movement.
- **Job manifest:** lives in the private deploy repo next to the other
  benchmark jobs; nothing cluster-specific belongs in this repo.
- **Not in scope for that job:** crate-side (tch-rs) CUDA bench with real
  weights — follow-up after the Python-side curves, since tch links the same
  libtorch and the findings doc showed TS==eager throughput on both CPU and
  MPS; expectation is the same on CUDA.

## Artifacts

- `checkpoints/curated/` (gitignored): the three checkpoints + `MANIFEST.json`
  (run, iteration, games, source path, sha256, fetch date).
- `exports/emeta-final/`, `exports/m50-latest/` (gitignored): per-device
  TorchScript artifacts + export manifests with parity results.
- `exports/bench_emeta_{fp32,fp16}.json`,
  `exports/bench_m50_fp32_{cpu,mps}.json`, `exports/bench_m50_fp16.json`
  (gitignored): raw grid + partial-study measurements (this doc's tables
  render from them; the splits exist because of the batch-512 pathologies
  above). `exports/bench_emeta_full_earlier_run.json` is the completed
  earlier single-invocation run — the source of the emeta fp16@512 cell
  and the reproducibility cross-check (fp32 cells agree within ~1%).
- `scripts/bench_model_eval.py` (committed): the grid runner (+
  `pad_obs_batch` helper, unit-tested in `tests/test_bench_model_eval.py`).
- `scripts/verify_real_checkpoint_outputs.py` (committed): real-row parity +
  non-degeneracy gate.
