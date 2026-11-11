# screen-r4a: DN2W16 br_fp gated by NOISE_GATED_K2 against our RAW arm (declaration notes)

**Pre-registered panel declaration:** `depthn-foulplay-dn2w16-declaration-r4a-20261111.json`, sha256 `1697ae7ab209a3e0000351817112916998a63ad2485bffacbffd5131156801fa`.
- `panel.validate` and `check_preflight` pass from the server-tree checkout.
- The panel's schema is closed, so the conditions it cannot carry are recorded here.

## What screen-r4a asks

Does the reply-model hypothesis hold live? This is variant B, called by the MC ladder's run D under the lead's calibration rule:
- **br_fp:** averaging with each requested world's reply prior replaced by π_FP red3-full's logits on view b;
- **gated by NOISE_GATED_K2:** br_fp's choice is played only when it differs from greedy (the root logits' argmax) and the mean per-world gap exceeds 2 standard errors; otherwise greedy is played.
- **Offline (MC ladder, Foul Play gate roots, guards on):** R5fp_ng +0.0080 [+0.0034, +0.0126] vs the policy, override 24.9%.
  - R5 (r3's live configuration): −0.0037 [−0.0104, +0.0028].
  - The ladder ranks configurations (R5 < R5_ng < R5fp_ng offline, matching r3 −0.116 < r4b +0.010 live). It does not predict the game-level effect size.

## What changes from screen-r4b (and only this)

1. **The reply model** (`reply_prior`):
   - archive `d2cac5cf6599cddbabc8611ebe39918a81185a2bbde0d4c60dcb074b12d4d771`, checkpoint `739cef0db39a72000fe70f2346f55b05d34adab598d7e383126aeda7b9f0a32d`;
   - trained on view b (`--peer-view-b`), jit views, union guard.
   - Every br_fp response names it, and the proxy (`--expect-reply-archive-sha256 … --expect-peer-view-b on --expect-reply-views jit`) and the panel (probe and readout provenance) pin it (#1900).
   - The root operator stays NOISE_GATED_K2 (#1901), now gating br_fp's choice.
2. **The server tree:** `5af1a78d3d5cfc9aab4ba4f49620cdc1ba1bb774` (git tree `84da2803`) = integration `78105ee4` + #1887 (peer view b belief fix) + #1900 + #1901.
   - It deliberately predates A2 (#1890), so screen-r4a runs on r3/r4b's engine and changes one hypothesis.
   - A2 and the later fidelity changes, with the guard shrink, enter together with the next screen.
3. **The pre-flight** (`/out/preflight-s4a/preflight.json`, sha256 `e57425dc1a3596c1591e689de7785d33b1868f85aeacaab12a2570e2ad1e3c72`): 24 outcome-blind games on this placement.
   - Search wall p50 2.04 s, p95 2.53 s (r4b: 2.26 s), max 2.85 s.
   - 3 timeouts in about 540 searched decisions (0.56%), within the panel's rule.
   - Searched share 82.9% of non-timeout decisions.
   - Declared warm p95 (RAW): encode 0.042 s, greedy 0.239 s.

Everything else is screen-r4b's:
- arms, cells, deadline 3.0 s, reserve 0.15 s, RULE_R3 arm fairness;
- the calibration of worker pod cpu-r3 (MEASURED, ratios 2×1000 ms 1.044 and 4×500 ms 0.925), its move walls, and the journal-full base config;
- engine tables and provenance.

## Replay check (final tree)

- The 160 definitive RR battles (sample seed 20261108) were re-priced on the server tree with the declared flags, and held to the scorer's rule by `scripts/depthn_gated_replay_check.py --reply` (action, q, root operator and noise-gate record on br_fp's worlds).
- Partial check before declaration: 1,152 roots compared, **0 mismatches, 0 unscorable**, override rate 25.5%.
- The complete check is recorded in the screen's records before the 500-pair readout. A mismatch stops the screen.

## Extension (pre-registered)

- The screen-r1 extension amendment (`evals/depthn-foulplay-dn2w16-amendment-extension-20261108.json`, sha256 `8a708ebb…`) is adopted unchanged, with the four rule files, by `evals/depthn-foulplay-dn2w16-amendment-screen-r4a-adoption-20261111.json`. That record is pinned in `RUN_BINDINGS`, `RUN_FINAL_RULES` (RULE_R3) and `EXTENSION_BINDINGS`.
- If the 500-pair amended verdict is exactly CONTINUE_TO_2000 with no amended breakage, the screen continues to 2,000 pairs under the amendment's own gate, with no mid-run call (`scripts/depthn_extension_dispatch.py --run-binding`, #1903).
- **Extra compute:** the same 4-GPU server node for the extended run.

## Placement and dispatch

- **Servers:** one 4-GPU pod on its own node (10.140.65.122), 3 processes per GPU behind the proxy on :7000.
- **Workers:** cpu-r3 (pairs 96–119), on another node.
- **Dispatch** runs the declaration commit `44494b01228a72e03c90b254ebd743b2ac993af0`, which differs from the server tree by the declaration JSON alone (`server_tree_check`).
- The verdict and the extension run from an archive of the rules commit with `--tree-commit 44494b01`. Its `src/pokezero` and `gpu_engine` are the dispatch tree's.
- **A RULE_R3 fairness flag** stops the run and goes to the lead, with no amendment mid-run.
