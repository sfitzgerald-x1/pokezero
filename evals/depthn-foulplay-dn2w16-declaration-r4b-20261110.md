# screen-r4b: DN2W16 with the noise-gated root operator against our RAW arm (declaration notes)

**Pre-registered panel declaration:** `depthn-foulplay-dn2w16-declaration-r4b-20261110.json`, sha256 `902516bc4b13588ffc469adaaf2add9636c2d6de3228d3fb5a33468da8f8c1f8`. The panel validates it with its own rules (`panel.validate`, `check_preflight`). The panel's schema is closed, so the conditions below that it cannot carry are recorded here.

## What screen-r4b asks
Is screen-r3's loss (−0.116, STOP_HARM) the winner's curse, the search overriding greedy on noisy evidence? screen-r4b changes **one thing**: the root operator. Everything else is screen-r3's. It runs beside screen-r4a (br_fp, the reply-model hypothesis); the lead's reading of the pair:
- if r4a ≥ 0 and r4b < 0, the reply model was the problem;
- if r4b ≥ 0, overriding too often was;
- if both lose, the value head or engine fidelity is the problem, and the engine fidelity patch comes before any more screens.

## The operator: NOISE_GATED_K2
- It plays averaging's choice a* only when a* differs from greedy **and** the per-world gap clears a noise gate: mean_w gap > 2 · sd_w(gap) / √W, where gap_w = Q_w(a*) − Q_w(greedy), Q_w(a) = Σ_b π_w(b) G_w[a][b] (the same per-world values and reply prior averaging uses), sample sd, a strict `>`, and W = 16. Otherwise it plays greedy.
- It is exactly the RR scorer's `noise_gated:k=2`. The function lives once in `src/pokezero/depthn_root_operator.py`, and the scorer imports it.
- **Greedy for the gate:** the live search gates against the server's greedy, the argmax of its bf16 `root_logits`. The scorer gates against the actor's FP32 greedy, which the server never sees. On the 160 definitive battles the two greedies differ on 16 of 6,440 roots (0.25%), and that changes the choice on 7 roots (0.11%). The lead accepted this difference, with no protocol change.
- **Guards:** unchanged. The reach guard's arm (c) applies to the played action.
- **Why not target_argmax:** on the fixed tree it overrides greedy on only 1.0% of roots. The definitive RR's 18% read the live run's broken-tree `policy_target` (`depthn_rr_score.py` reads telemetry `search.policy_target`).

## Replay check (160 definitive battles, re-priced on the server tree with NOISE_GATED_K2)
- `/out/rr-r4b-noisegated-s160/replay-check.json`, sha256 `f2233fa916fa776cbb0513c1f620788b85836a94274c79881a670b95de101262`: **6,440 roots compared, 0 mismatches** between the live played choice and the scorer's rule on each row's own re-priced values, guards included. The server-greedy differences are reported separately (`rr_greedy`).
- **Override rates on the fixed tree:**

  | Population | Operator | Override rate against greedy |
  |---|---|---|
  | All roots | noise_gated k=2 | 24.7% played (31.9% raw); gate passes 31.8% |
  | Reach-unguarded roots | noise_gated k=2 | 29.6% |
  | Reach-unguarded roots | averaging | 54.8% |

  That is the dose-response contrast the lead asked for. The RR's 34% was on the old any-cell guard's unguarded population.

## What changes from screen-r3 (and only this)
1. **The root operator:** `NOISE_GATED_K2` (declared `root_operator`), on the server tree `79b1e51592a666b6a51b2ee15f1a78b7b5369330`. That tree is screen-r3's tree `3bd60120` plus the operator code (the TARGET_ARGMAX plumbing `3e16fc2e`, a replay-check hardening `85e681d2`, and NOISE_GATED_K2 `79b1e515`). The search, guard, actor and fairness code is otherwise screen-r3's, and a cross-tree dump shows the default path byte-identical to it.
2. **A second worker pod,** on its own machine, so r4b can run beside r4a. Its outcome-blind RAW calibration is MEASURED, ratios 2×1000 ms 1.044 and 4×500 ms 0.925 (`calibration.json` sha256 `23d871a8…`), and its move walls are sha256 `d9fdd9cf…`. The calibration uses the same seed band and the same journal-full base config.
3. **The search servers** run in their own pod (4 GPUs on one machine), on a machine different from both worker pods and from the Foul Play collection's.
4. **Pre-flight** on this tree and placement (`/out/preflight-s4b/preflight.json`, sha256 `3ea279aaa0f7f27663a08177cd881b1152681134ad4e3f61142ed4bcf55e43c2`): 24 games, outcome-blind, on this placement. Search wall p50 1.82 s, p95 2.26 s (screen-r3: 2.34 s); timeouts 0; searched share 83.1% of non-timeout decisions (screen-r3: 82.0%); constructor refusals 9.0%. The declared warm p95 is the RAW arm's (as `check_preflight` requires): encode 0.041 s, greedy 0.286 s; `panel.init` checks it matches the pre-flight and it feeds no decision rule.
5. **Dispatch** runs the declaration commit `1338fb5abc020054f037152e0803939ef2140a39`, which differs from the server tree by the declaration JSON alone (`server_tree_check`).

## Unchanged from screen-r3
- arms;
- interleave;
- the same 500 cells;
- deadline 3 s and reserve 0.15 s;
- the journal-full base config (`49a8a21f…`);
- engine tables;
- the exact reach guard and its conditions;
- arm fairness `RULE_R3` (exact alive bands, minimum support 120, tolerance 5%), in the checkpoints and the final gate;
- breakage and latency rules;
- confidence 0.9;
- the one-sided Pressure caveat: no view b in averaging-family screens.

## Decision rules
The official readout is the panel's, computed from a clean checkout of the dispatch commit. The verdict is `scripts/depthn_screen_amended_verdict.py --run-binding`, with screen-r1's four committed amendments, adopted by `evals/depthn-foulplay-dn2w16-amendment-screen-r4b-adoption-20261110.json` (sha256 `434369aefd7b0b8f7fdcb556798ff87b1f3964698103e20e91978f77663ab713`, pinned in `RUN_BINDINGS` and `RUN_FINAL_RULES`). That record binds them to this declaration, `/out/screen-r4b` and the dispatch tree, and declares `final_fairness_rule: RULE_R3`.
- No run amendment is adopted or allowed.
- No extension to 2,000 cells is declared.
- Any fairness flag that fails `RULE_R3` stops the run and goes to the lead, with no amendment mid-run.
- The verdict refuses to run when any COMPLETE cell's owner capture is missing.

## Readout plan
- the result with its seed-clustered interval;
- the live override rate against greedy and the gate pass rate, beside screen-r3's averaging;
- the guarded share;
- p95 latency;
- RULE_R3 fairness with supported shares;
- confirmation that servers and workers ran on separate machines.
