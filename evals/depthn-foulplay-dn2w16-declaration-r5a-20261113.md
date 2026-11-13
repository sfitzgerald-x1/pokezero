# screen-r5a: screen-r4a3's configuration on the post-fidelity tree (declaration notes)

**Pre-registered panel declaration:** `depthn-foulplay-dn2w16-declaration-r5a-20261113.json`, sha256 `0b995d6e864ec430c9ebe8dc9694dd0b70683a6293024ab0e488eafe928ae96e`. It's built from screen-r4a3's committed declaration (`e2e39feb…`) by `build_declaration_r5a.py`, which replaces only the binding fields. **Adoption record:** `depthn-foulplay-dn2w16-amendment-screen-r5a-adoption-20261113.json`, sha256 `ceffcf858ef6b5b81ba5f48065b32abf6f52e3c6ea773e63fb3752550ec23494`.

## What screen-r5a answers

r4a3 met the bar at 3.0 s on the r4a3 tree: the conservative lower bound was +0.050 over 2,000 pairs, and the independent recomputation agreed.

r5a asks whether search still beats our raw arm on the post-fidelity tree, where far more roots go to search: about 91% of non-forced decisions with M2, or about 89% without, against 71% in r4a3. Many of those roots involve moves the engine only just started modelling.

The decision rules, the harm mirror and RULE_R3 stay exactly as in r4a3.

## What changes from screen-r4a3 (and only this)

1. **Server tree** `41faf832fd168841ce73913c5d8831376d635d95` (tree 97f074f9), the integration tip at the tree-fix (#1946, main's #1929 synced in):
   - the engine's A1, A2, R and M1, and M2 + R1 (f3e71663);
   - the constructor walls (#1929);
   - the Knock Off carry (#1935);
   - the effect-guard retirements;
   - the ability guard.

   The dispatch tree is the server tree plus this declaration JSON alone: `21b0a8f38eba3d61853c7aba57881a9742e2114c`.
2. **Effect guard (UNREPRESENTED_EFFECT).** `depthn_effect_guard.RETIRED` at the server tree holds these moves, each with the engine merge from which the engine models it:
   - destinybond (A2), hyperbeam (M1);
   - curse, aromatherapy, refresh, lovelykiss, charm, and the Sleep Clause guard on spore, sleeppowder, hypnosis (A1);
   - gigadrain, mirrorcoat (A2); counter, bonemerang (R);
   - trick and knockoff (#1935, 3f48093b; this also covers Sticky Hold holders);
   - wrap, yawn, perishsong (M2 + R1, f3e71663).

   White Herb + C-lite merged after the tree-fix (8823d7c5), so its four (superpower, psychoboost, sacredfire, facade) are **not** retired in r5a.

   Still guarded: transform and superpower (tier A); sacredfire, psychoboost and facade (tier C, under their conditions); and the live actor's own-recharge guard (OWN_RECHARGE_UNREPRESENTED). The union guard, the reach rule and the 5% threshold are unchanged.
3. **Ability guard** (`--ability-guard`; it has no declaration key, so this note and the adoption record state it, and the serve script checks every server's READY header):
   - WG: every arm;
   - PINCH: choice only, PINCH_ROOT third;
   - CONTACT: choice only.

   It adds about 0.93 pp of guarded decisions on the gate-2 replay; the per-family table is in the declaration draft.
4. **Re-measured on the new tree and worker:**
   - the pre-flight (`preflight-r5a-pf6`, 1 server per GPU on 8 GPUs, 8 lanes);
   - the worker calibration and move walls;
   - `expected_searched_share`, from the pre-flight, with `expected_searched_share_roster` null since there's no roster bench this round;
   - `bench_p95_wall_seconds`, from the pre-flight's client wall;
   - `provenance.engine_tables.digest` (see the erratum).
5. **Placement and lanes:** **1 server per GPU on 8 GPUs, 8 worker lanes** (`server_processes_per_gpu` 1, `worker_lanes` 8). How it came to this:
   - The declared rule started at 3 per GPU on 4 GPUs, 12 lanes, with the 0.5% switch to 2 per GPU.
   - pf1 and pf2 failed on GC stalls, which the remedy below removed.
   - pf4 (3 per GPU, 12 lanes) and pf5 (2 per GPU, 8 lanes) still timed out on co-located time-slicing: two servers pricing on one GPU.
   - So the lead approved 1 server per GPU on 8 GPUs (under the 16-GPU limit), beyond the declared rule's 2-per-GPU floor. pf6 passed on it.
   - Calibration, the move walls, the binding pre-flight and the screen run at the declared lane count and shape.
   - A reduced lane count only lengthens the screen; no rule or statistic changes.
   - Servers may span nodes behind the one proxy (the lead, 2026-11-12, with the cluster nearly full), but never on the worker's node or the Foul Play collection node.
   - The declared pre-flight ran on the same placement shape as the screen: the same per-GPU count and the same node split. That's 4 + 2 + 2 servers on three nodes, one per GPU, behind one proxy.

## Server environment: the GC remedy

The declaration has no key for this, so this note and the adoption record's `server_environment` state it.

**The problem:** each DepthN server keeps its warmup's traced programs alive, about 66M Python objects (jaxprs). Every full gen-2 collection rescanned them, taking about 13.3 s with the GIL held and stopping the request loop.
- That happened every ~100–200 requests on this tree.
- It caused pf1's and pf2's timeouts.
- r4a3's screen had the same, at about 10 s, roughly every 5,000 requests.

**The remedy:** the servers run with `sitecustomize.py` (sha256 `3232ac8d465e294a74463c416841945b0425df54bd573f8e149e39dc541c9d4c`) via `serve-r5a-gc.sh` (sha256 `be65b80c27887cd13d3634743afb55eed827240f237b1c938a12a3b95297b9ca`). The tree is unchanged.
- **Before any traffic,** each server runs `gc.collect()` and then `gc.freeze()`.
- **Every 30 s,** it repeats collect-then-freeze, so full collections scan only recent objects.
- **In pf6,** the longest post-warmup gen-2 collection was 19.6 ms on every server.
- The diagnostic instrumentation of pf3 is off.
- The real fix (the pricer keeping its traced jaxprs) is a tree change for after r5a.

## Identity certificate (gate 12)

- **Dry run** on 3bbadfc8: guards on against off, 526 replayed r4a3 decisions, 401 priced, Q tables bitwise equal on all of them, 0 mismatches. Every kind where the changed code is live had at least 100 decisions: retired_legal 129, WG 129, PINCH 208, CONTACT 232.
- **Binding run** on `41faf832`: IDENTICAL.
  - 526 decisions in each arm, 495 priced in both (31 refused against the dry run's 125, before #1929).
  - Q tables bitwise equal on all 495, 0 mismatches, 0 OFF trips.
  - Per kind: retired_legal 195, WG 142, PINCH 246, CONTACT 259.
  - The ON guard tripped on 51 decisions (21 ability, 29 with effect reasons); action differences 0.
- **With the GC remedy active** (below):
  - **gc-r1**, ON against OFF on a fresh cache: IDENTICAL.
  - **gc-r2**, the same on a copy of the binding run's cache: IDENTICAL. Against the binding run (no remedy) it gave 0 differences on all 495 priced decisions, in both arms, with the same refused set.
  - **gc-r1 against the binding run** differed by at most |ΔQ| 1.76e-7, with 0 actions changed. That came from separate autotuning, as gc-r2 proves.
  - The gate-12 claim is bitwise identity within one cache. ON and OFF share the cache by design. Cross-run differences at autotuning level are outside that claim and don't touch the certificate.

## Known warning: the cuBLAS < 13.2 concurrency advisory (gate 13(c))

XLA prints this advisory at startup on this image; r4a3's servers printed it too. The lead closed it on 2026-11-12:

"cuBLAS < 13.2 concurrency advisory: not exercised by the live servers. Evidence:
- nsys trace with per-node CUDA-graph tracing, at R1's engine with serve-r4a variant-B flags: 2,706 cuBLASLt launches with 0 overlaps against any other kernel, and 0 cross-stream overlaps;
- command buffers off is bitwise identical on 4 roots;
- MPS off on the server nodes (labels checked at pre-flight).

Re-check if any XLA stream, async or overlap flag changes, or if jax is upgraded."

**Evidence:** the engine session's Q1 run `v2e-gpu-r17`: engine R1 `c8521ed9` (M2's merged engine), jax and jaxlib 0.12.0, cuBLAS 13.1.1, one GB300, serve-r4a variant-B flags, roster roots 1–4 after a 2-root warmup, memory fraction 0.3. Its manifest `v2e-gpu-r17/q1-manifest.sha256` (sha256 `ccdcf97b1fd4ae1788ff306566052fa92a546723feb4c403aa3d9e33e6ff755b`) lists 12 files, each with its full sha256, among them:

| Evidence | File | sha256 |
|---|---|---|
| nsys trace (`cuda_gpu_trace` export; 2,706 cuBLASLt `nvjet_sm103_*` launches, 0 overlaps involving them, 0 cross-stream overlaps in the serving window) | `q1-trace_cuda_gpu_trace.csv` | `baa0f83b10bb24900bb8fa48f982e6b21dd7dfec2da7f1b789effe89433be350` |
| the raw report | `q1-nsys-default.nsys-rep` | `ade3548f255d3132ee941f7d4a14cfc50f8cd2700bdd0204cee02ab7ad3ef38f` |
| the analysis giving those counts (serving window: the final 30 s) | `q1_trace2.py` | `cd8305a45de9cf36a3d49e28ec3b6d1e2559b13edb45dda7a0df0020056afd80` |
| 4 roots priced with default XLA flags | `q1-default.json` | `251d3cb4a10b9e2dfea08a9e018277dc46cf6c25e6fac0dcacf921e743310730` |
| the same 4 roots with command buffers off (`--xla_gpu_enable_command_buffer=`); status, action, q_values, completed_q, policy_target, root_value and action_mean_q all bitwise equal | `q1-nocb.json` | `47c40dad9555adbc09a658fa300805bab856e2880b0ed6f17def8cc5a006c3e2` |

The manifest also lists the run and pricing scripts (`q1-run.sh`, `q1_price.py`), `q1.status` and the two run logs.

**MPS check at this pre-flight:** every server node (three nodes for pf6) shows mps.capable=false, gpu.sharing-strategy=none and gpu.replicas=1 (`mps_off_r5a.py`, MPS_OFF PASS).

## Pre-flight history (all runs, failures included)

- **pf1** (2026-11-13 02:47–03:00Z; seeds battle 2103402300 / Foul Play 2103502300 × 50; 48 games; 3 servers per GPU on 4 GPUs of one node, 12 lanes; calibration-r5a, MEASURED at 12 lanes): **placement SWITCH_TO_2_PER_GPU** (gate decision FAIL). Timeouts 9 of 998 non-forced decisions (0.90%, cp95 upper 1.57%; 8 of 9 in seat 0's 20+ band, four consecutive in one game); client wall p50 1.88 s, p95 2.33 s, p99 2.59 s, max 2.80 s; searched 949 of 998 non-forced (95.1%); the panel's own bounds within (search wall p95 within 2.6 s, refusals within, 0 reuse refusals). No other GPUs were free (03:01Z), so the 0.5% rule's second branch applied: the same 4 GPUs at 2 per GPU, 8 lanes, with re-calibration at 8 lanes.
- **pf2** (2026-11-13 03:54–04:14Z; seeds battle 2103402350 / Foul Play 2103502350 × 50; 48 games; 2 servers per GPU on the same 4 GPUs, 8 lanes; calibration-r5a-l8, MEASURED at 8 lanes, ratios 0.921 / 1.022): **placement ESCALATE** (gate decision FAIL). Timeouts 22 of 1,063 non-forced (2.07%, cp95 upper 2.94%), all from 4 process-level stalls of about 13.3 s and their cascades; client wall p50 1.82 s, p95 2.11 s, p99 2.30 s, max 2.75 s; searched 918 of 1,063 non-forced (86.4%). The diagnosis (pf1, pf2, pf3) is summarised under the server environment above and in the adoption record. Taken to the lead.
- **pf3, diagnostic** (2026-11-13 05:29–05:44Z; seeds battle 2103402400 / Foul Play 2103502400 × 50; 48 games; 2 servers per GPU, 8 lanes, calibration-r5a-l8; instrumented by an environment-only sitecustomize, not a candidate for binding): PASS, KEEP, 0 timeouts of 996 non-forced; client wall p50 1.80 s, p95 2.12 s, p99 2.32 s; searched 92.5%. It identified the stall: full Python gen-2 collections over about 66M retained jaxpr objects (13.6 s at READY), which no serving request met in this window.
- **pf4** (2026-11-13 07:50–08:02Z; seeds +450; the GC remedy; 3 servers per GPU on the same 4 GPUs, 12 lanes, calibration-r5a): **placement SWITCH_TO_2_PER_GPU** (gate FAIL). 14 timeouts in 1,024 non-forced decisions (1.37%, cp95 upper 2.13%), all genuine overruns: server time 2.81–3.06 s (price 1.39–1.63 s), early-game roots (turns 1–7) priced while the other busy server on the same GPU was also pricing. Queue 0, no stalls: the remedy held, and post-warmup gen-2 collections were at most 14 ms on all 12 servers. Client wall p50 1.88 s, p95 2.51 s, p99 2.69 s; searched 84.1%. Per the pre-stated rule: pf5 at 2 per GPU, 8 lanes.
- **pf5** (2026-11-13 08:24–08:39Z; seeds +500; the GC remedy; 2 servers per GPU, 8 lanes, calibration-r5a-l8): placement **KEEP** (4 of 943 non-forced, 0.42%, cp95 upper 0.97%) but **gate FAIL**: first-1000 0.42% against the 0.3% bound. The 4 are single genuine overruns (server 2.80–3.00 s, price about 1.5 s while the GPU's other server was pricing; 2 are turn-1 roots); post-warmup gen-2 collections at most 9.5 ms. Client wall p50 1.80 s, p95 2.24 s, p99 2.60 s; searched 90.9%. Escalated to the lead per the pre-stated rule.
- **pf6** (2026-11-13 09:07–09:22Z; seeds battle 2103402550 / Foul Play 2103502550 × 50; 48 games; the GC remedy; **1 server per GPU on 8 GPUs**: 4 on one node, 2 + 2 on two more, behind the one proxy; 8 lanes; calibration-r5a-l8): **PASS, KEEP**. 1 timeout in 1,059 non-forced decisions (0.094%, cp95 upper 0.45%; first-1000 1/1000), a turn-1 root; client wall p50 1.78 s, p95 2.12 s, p99 2.26 s, max 2.33 s; searched 91.4% (968/1,059). Every one of the 8 servers was used (95–186 requests each), at most 1 battle and 1 request at a time on any server; proxy hop p50 8.4 ms local, 9.2 ms cross-node (p99 12.8 / 13.5 ms); post-warmup gen-2 collections at most 19.6 ms on all 8. MPS off on all three server nodes. **The declared pre-flight.**


## Verdict rules

As r4a3. The adoption record adopts the same four rule files (unknown cells, harm mirror, both bounds, final fairness under RULE_R3) and the extension unchanged.

## Erratum: `provenance.engine_tables.digest` in earlier declarations

The digest `e349be50d83625ebd8222cfaa6f6f61d9424647625275211c45e85c0359c4882` first appeared in screen-r1's declaration (server tree 77235041) and was carried unchanged through screen-r2, r3, r4b, r4a2 and r4a3. It does not reproduce on any declared tree.

**The field was descriptive only:** `depthn_live_panel.validate` checks its format, and nothing compares it with the tree. So no past verdict depended on it, and the past archived declarations stay as they are.

**From r5a on,** the field is a binding value computed by a stated method. `engine_digest_r5a.py` compiles the live executor's tables exactly as `scripts/depthn_live_server.build` does (its `load_inputs`, then `constructor_worlds.engine_tables(metadata, catalog)`, then `constructor_worlds.engine_tables_digest`), on the constructor metadata `fad2db5a…`. Per tree, by that method:

| Tree | Digest |
|---|---|
| 77235041 (screen-r1) | not computable: the function did not exist yet |
| 04192f46 (R1-C, where `engine_tables_digest` began) | `2daf793abd3d075195ca391699fb8ba56e2608da308d0c0b145dfd7282988136` |
| 92d8af8c (screen-r4a3's server tree) | `22803a61f6ddfefdf6dfb9fa725acb3bed591a89ec738ff62b99bd776d65f2d1` |
| 3bbadfc8 (integration after #1940) | `1af44bde5509df493fc47ec5753e27de637b3a49392e8314e3084d6c9a351f24` |
| 41faf832 (r5a, declared) | `6536ab2bbcc298fdf26b724386f20a97167d2c615534b7c8595076848824a126` |

## Launch timing and what was known

r4a3's full 2,000-pair result was known when r5a was decided: BAR_MET, conservative +0.050, recomputation AGREE.

Nothing of r5a is computed before this record. The pre-flights, calibrations and identity checks above are outcome-blind and never scored.
