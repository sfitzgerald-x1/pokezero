# screen-r4a2: screen-r4a restarted at a 5.0 s deadline after VOID_LATENCY (declaration notes)

**Pre-registered panel declaration:** `depthn-foulplay-dn2w16-declaration-r4a2-20261111.json`, sha256 `77e9c133d4641e12559a14c29618a0ade7f1929455c515165d66325faf53973b`.
- `panel.validate` and `check_preflight` pass from the server-tree checkout.
- The panel's schema is closed, so the conditions it cannot carry are recorded here.
- screen-r4a's notes (`depthn-foulplay-dn2w16-declaration-r4a-20261111.md`) hold for screen-r4a2 except for what follows.

## Why a restart

- screen-r4a was voided by the panel's latency rule (VOID_LATENCY) after 59 cells: 46 timeouts in its first 1,000 searched decisions (4.6% > 1%).
- Under continuous load the reply model's reduce stage pushed the end-to-end decision wall to p95 2.75 s and p99 3.13 s, against the 3.0 s deadline less the 0.15 s reserve. No server fault: every timeout was client-side.
- **Pre-registered rule:** the deadline is re-declared from the void readout's coverage table (candidate deadlines 5, 10 and 15 s; coverage ≥ 0.995) and the screen restarts fresh. At 5.0 s the coverage is 1.0 over 1,359 searched decisions (max wall 4.21 s), so the deadline is **5.0 s**.
- Only the void readout was read: latency and coverage, with no paired, score or outcome statistic. screen-r4a's 59 cells are not scored and do not enter screen-r4a2.
- Foul Play's budget is unchanged. The search does not spend its time budget, so only the cutoffs move.

## What screen-r4a2 answers

- screen-r4a2 answers **whether the operator helps** (the reply-model hypothesis, variant B), not deployability. Its search content is screen-r4a's.
- The ladder's bar is still a 3 s cap, and this configuration's tail runs over it under 12-lane load (p95 2.75 s, p99 3.13 s in screen-r4a).
- If screen-r4a2 is positive, a latency workstream follows: the reply stage's cost (p95 0.36 s), server packing, or the CUDA engine.
- screen-r4a's pre-flight (24 games) understated the tail under full load. Future screens run their pre-flight under full concurrent load.

## What changes from screen-r4a (and only this)

1. **The deadline:** `deadline_seconds` 5.0 and the breakage latency bound (`latency_p95_max_seconds`) 5.0. Both were 3.0.
2. **The pre-flight** (`/out/preflight-s4a2c/preflight.json`, sha256 `bcaf5f2f3a5bad068dfea1034946ad63c4ae112a642f9fca0535a367eb6dc84f`): 48 outcome-blind games at 5.0 s on fresh servers (server tree 982909e1), so the 12 lanes stay under sustained load (the lead: pre-flight under full concurrent load).
   - Search wall p50 2.00 s, p95 2.54 s, p99 2.67 s, max 3.31 s (bound 4.6 s). 0 timeouts in 1,084 search-arm decisions. Searched share 78.1% of non-timeout decisions (fallbacks: unrepresented effect 129, constructor refused 108).
   - **Load:** the servers saw screen-r4a's load. 117.5 requests per minute against screen-r4a's 115.9, and in-flight decisions mean 3.55, p90 5 against screen-r4a's 3.51, p90 5 (server telemetry, the middle 80% of each run).
   - **Server telemetry (the lead's conditions):** SEARCH_PREFIX_NOT_ORIGINAL_SUCCESSOR 0 of 1,084 server requests (required 0); constructor refusals 10.0% of server requests (r4a's pre-flight 8.4%, screen 10.4%); search wall p50 2.00 s, p95 2.54 s (r4a's pre-flight p50 2.04 s, p95 2.53 s).
   - Declared warm p95 (RAW): encode 0.042 s, greedy 0.307 s.
3. **The server tree:** `982909e10c141055164daa875860ec078e0817cd` = screen-r4a's server tree `5af1a78d3d5cfc9aab4ba4f49620cdc1ba1bb774` plus one panel change: the pre-flight search-wall bound now derives from the declared deadline (next section). Search and engine code are unchanged. It is still deliberately before A2.
4. **The run:** run directory `/out/screen-r4a2`, its own adoption record (`evals/depthn-foulplay-dn2w16-amendment-screen-r4a2-adoption-20261111.json`), and its own dispatch commit. screen-r4a's record stays as history.

The reply model and its pins, NOISE_GATED_K2, the arms, cells, reserve, RULE_R3 arm fairness, the worker pod's calibration and the base config are all screen-r4a's.

## The pre-flight wall bound (the server tree's one change)

- Through screen-r4a the panel bounded the pre-flight's search-arm end-to-end decision wall p95 at a fixed 2.6 s (`PREFLIGHT_WALL_P95_MAX`). That is 3.0 s deadline − 0.15 s reserve − 0.25 s.
- The pre-registered latency re-declaration moved the deadline to 5.0 s but not this bound. At 5.0 s the slow tail completes and enters the measured wall instead of timing out: screen-r4a's pre-flight cut 3 decisions at 3.0 s and read p95 2.53 s, while the first valid pre-flight at 5.0 s read p95 2.70 s with no timeouts (screen-r4a under load: 2.75 s). So `check_preflight` refused a pre-flight whose wall was screen-r4a's own.
- **The lead's call:** the bound is now `deadline − reserve − 0.25` (`preflight_wall_p95_max`). It is identical at 3.0 s, so every earlier screen and declaration is unchanged, and 4.6 s at 5.0 s. `check_preflight` also binds the pre-flight file's bound to the declaration's.
- Running a pre-flight again until it read under 2.6 s was rejected as fishing on a gate.

## Restarted servers (a pre-flight that measured nothing)

- The first pre-flight at 5.0 s ran on the still-running screen-r4a servers. It reused the calibration seed band of r4a's own pre-flight (`--battle-seed-start 2103402000`).
- Each live server keeps one session per (battle, owner, stream) for its whole life. A reused battle ID is refused (`SEARCH_PREFIX_NOT_ORIGINAL_SUCCESSOR`), so all 551 of its search requests fell back and the wall read 0.32 s.
- `check_preflight` bounds the search wall and the timeouts but not the refusal share, so it passed. That pre-flight and the declaration built from it are not used.
- The search servers were restarted with no carried sessions, and the pre-flight was rerun (preflight-s4a2b: 0 prefix refusals; its wall p95 2.70 s exceeded the fixed 2.6 s bound, see above). After the bound change the servers were restarted again on server tree 982909e1 for the declared pre-flight. This is the sequence screen-r4a's declaration came from.
- screen-r4a2's battle IDs are new to the restarted servers. screen-r4a itself had no prefix refusals.

## Replay check, extension, placement and dispatch

- **Final-tree replay:** screen-r4a's: the 160 definitive RR battles re-priced on server tree 5af1a78d with the declared flags. 982909e1 changes only `depthn_live_panel`, which no search, server or replay module imports, so the search code is the same. Partial check before declaration: 7,297 rows, 6,341 compared, **0 mismatches, 0 unscorable, 0 gate refusals**, override rate 27.9%. The complete check is recorded in the screen's records before the 500-pair readout. A mismatch stops the screen.
- **Extension (pre-registered):** the screen-r1 extension amendment is adopted unchanged, with the four rule files and RULE_R3, by screen-r4a2's adoption record. That record is pinned in `RUN_BINDINGS`, `RUN_FINAL_RULES` and `EXTENSION_BINDINGS`. A CONTINUE_TO_2000 with no amended breakage continues to 2,000 pairs with no mid-run call. Extra compute: the same 4-GPU server node.
- **Placement:** as screen-r4a: the search servers in their own pod with 4 GPUs on one machine (not the worker pod's or the Foul Play collection's), and the same calibrated worker pod.
- **Dispatch:** runs the declaration commit `0a157c8cacaeb48a7ea850aed9dc15b506742c61`, which differs from the server tree by the declaration JSON alone (`server_tree_check`). The verdict and the extension run from an archive of the rules commit with `--tree-commit 0a157c8c`.
- **A RULE_R3 fairness flag** stops the run and goes to the lead, with no amendment mid-run.
