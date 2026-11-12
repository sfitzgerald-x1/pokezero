# screen-r2: DN2W16 against our RAW arm on the fixed search tree (declaration notes)

**Pre-registered panel declaration:** `depthn-foulplay-dn2w16-declaration-r2-20261109.json`, sha256 `45cbb471c07ccf9059fa15b519a0e6f9f2b3424a2c4338a4dfdd39d951b7ad41`. The panel validates it with the exact same keys and rules as screen-r1's (`depthn-foulplay-dn2w16-declaration-20261108.json`). The panel's schema is closed, so it can't carry the extra conditions below; this note records them.

## What screen-r2 asks
Was screen-r1's result caused by the broken search inputs, or does search lose to Foul Play anyway? screen-r2 runs the same operator (DN2W16, ARGMAX_MEAN_COMPLETED_Q, 3 s cap) on the corrected tree, against Foul Play, on screen-r1's seeds.

## The arms and the tree
- **Arms:** RAW and DN2W16, interleaved as before (`ALTERNATE_ARMS_BY_PAIR_ORDINAL_LEAD_AT_MOST_CONCURRENCY`). 500 paired cells from the ladder declaration's deterministic cell order, the same seeds as screen-r1.
- **RAW is re-run, not reused.** The live actor module changed with the guards, so the harness hashes differ. A RAW baseline from the same time is also cleaner.
- **Servers** run `server_tree_commit` = `c7808ad90f16d1338112e364487508f2042f198b`. Dispatch runs the declaration commit `712cf20b683d69c98a290040bb98120ed6372b1e`, which differs from the server tree by the declaration JSON alone, as the panel's `server_tree_check` requires. This follows screen-r1, where the servers ran 77235041 and dispatch ran its declaration commit 1aaa75f9. This note and the amendment adoption record are pre-registered in later commits, pushed before dispatch; neither is part of the dispatch tree.
- **The first dispatch commit was superseded before any cell ran.** `5ca04a4b` (declaration sha256 `f065fee3…`) declared the actor's warm p95 from the pre-flight's DN2W16 arm; `panel.init`'s `check_preflight` requires the RAW arm's, so init refused it. No cell ran and `/out/screen-r2` was never created. `712cf20b` changes only `actor.warm_p95_seconds` (to the RAW arm's encode 0.04189746100018965 s and greedy 0.3835921959980624 s).
- **The tree contains:**
  - the root-view and constructor-carry fixes (#1865);
  - the `UNREPRESENTED_EFFECT` guard (#1867);
  - the reach-weighted scope (#1873);
  - the live reach guard (#1874).

## Conditions beyond the JSON
1. **`opponent_journal: full`.** The base config is the journal-full variant. It is pinned by `provenance.base_config_sha256` = `49a8a21ff8ec87aeedd420cee3bd3bb83c78e15bd32fce70140f96336f9e1b60`, and it differs from screen-r1's base config only in that key. Calibration and pre-flight both ran with it.
2. **The server and actor deploy together.** The live actor treats a COMPLETE search response without the `UNREPRESENTED_EFFECT` record as `server_error`.
3. **The live guard is the exact reach rule.** A root plays greedy when any of these holds:
   - **turn-1 (b):** either side's turn-1 prior puts at least 5% on a lane whose move is guarded for that side, in any world;
   - **reach:** either side's probability under the joint turn-1 priors of playing a guarded move at turn 2 is at least 5%,
     M_s = Σ_w p_w Σ_(a,b) π_own(a)·π_peer,w(b)·q_(w,ab),s, computed exactly from each turn-1 child's own guard decision;
   - **(c):** the search's choice is a guarded move.

   **Guarded moves** (`src/pokezero/depthn_effect_guard.py`):
   - **Tier A**, the GPU engine's unmodeled effects: Hyper Beam, Destiny Bond, Transform, Yawn, Wrap, Perish Song, Trick, Counter, Mirror Coat, Curse, Superpower, Knock Off, Charm, Lovely Kiss, Bonemerang, Giga Drain.
   - **Tier C**, guarded only while its condition holds for the side (at the root, and at each turn-2 child with that child's own statuses):
     - **Spore, Sleep Powder, Hypnosis** under Sleep Clause: the target's side has a non-fainted Pokémon asleep from any source but Rest (a Rest sleep does not count, and Rest itself is not guarded);
     - **Refresh** when its user is brn, par, psn or tox;
     - **Sacred Fire** when its user is frozen;
     - **Psycho Boost** while its user holds an unconsumed White Herb;
     - **Facade** when its user has Guts and is brn, par, psn or tox;
     - **Aromatherapy** when the user's party has a non-fainted member with a status.

     Our own items and abilities come from our request. The opponent's White Herb comes from the public lines and the pool facts: Deoxys and Deoxys-Attack hold White Herb until an item line says otherwise. The opponent's Guts is always the pool fact, never read from the lines: every pool Facade user has Guts.
   - **Fail-closed trips:** a root also plays greedy when its turn-2 capture is unavailable (cells expanded without labels) or when its guard inputs could not be read (an error record).

   **Unchanged guards:** the opponent-recharge refusal and the own-recharge guard.

   **Equivalence:** on screen-r1's 160 RR battles, the live rule agrees with the report-only reach bounds on every root that isn't ambiguous: lower trips 641/641, upper passes 4,816/4,816, and M lies within its bounds on 6,440/6,440. It trips on 17.2% of priced roots.
4. **Guard latency:** at the live shape (bucket 1, W16), 306 decisions on the server tree, the guard adds p50 30.5 ms, p95 59.7 ms and p99 71.9 ms. The bar is p95 ≤ 100 ms. The 3 s cap holds with everything on.
5. **Decision rules are screen-r1's, as amended.** The official readout is the panel's. The verdict is `scripts/depthn_screen_amended_verdict.py --run-binding` applied to it, with screen-r1's four committed amendments, which the CLI pins by sha256. They are adopted unchanged for screen-r2 by `evals/depthn-foulplay-dn2w16-amendment-screen-r2-adoption-20261109.json` (sha256 `5e53654231abc940461decde4cdfa6b85ce89f22e20ff0ea7f9e626ea1855a75`, also pinned by the CLI), which binds them to this declaration's sha256, the run directory `/out/screen-r2` and the dispatch tree `712cf20b`, and which is committed before any screen-r2 cell exists:

   | amendment | sha256 |
   |---|---|
   | unknown-cells | `c4ff30a25a573f90fdd02957b40fb8307d8b87b56dfbfe60d85ea662b18e8b3b` |
   | harm mirror | `dbce9a54223969a213a3e584fd2e01bbe259ebad7530678fbba7cd38ecd9539f` |
   | both bounds | `98e6120f30565c0b40d423504aa0d07a2929b2207551a5c901adb7579d361e13` |
   | final fairness | `b2fccc253136b2bce3125cc773973c7186a0dff6a5ff8e6825a5618d6268050e` |

   The screen rule, bar, breakage, latency rule, confidence 0.9 and arm fairness are unchanged.

   The official readout is computed from a clean checkout of the dispatch commit (`712cf20b`), because the panel holds a readout's
   tree to the run's tree. The verdict then runs from a checkout of this branch, from the repository root, with the root on the path:
   ```
   PYTHONPATH=src:. python scripts/depthn_screen_amended_verdict.py \
     --readout /out/screen-r2/<readout>.json --readout-sha256 <its sha256> \
     --amendment evals/depthn-foulplay-dn2w16-amendment-unknown-cells-20261108.json --amendment-sha256 c4ff30a25a573f90fdd02957b40fb8307d8b87b56dfbfe60d85ea662b18e8b3b \
     --addendum evals/depthn-foulplay-dn2w16-amendment-harm-mirror-20261108.json --addendum-sha256 dbce9a54223969a213a3e584fd2e01bbe259ebad7530678fbba7cd38ecd9539f \
     --both evals/depthn-foulplay-dn2w16-amendment-both-bounds-20261108.json --both-sha256 98e6120f30565c0b40d423504aa0d07a2929b2207551a5c901adb7579d361e13 \
     --final evals/depthn-foulplay-dn2w16-amendment-final-fairness-20261108.json --final-sha256 b2fccc253136b2bce3125cc773973c7186a0dff6a5ff8e6825a5618d6268050e \
     --run-binding evals/depthn-foulplay-dn2w16-amendment-screen-r2-adoption-20261109.json 5e53654231abc940461decde4cdfa6b85ce89f22e20ff0ea7f9e626ea1855a75 \
     --run-amendment evals/depthn-foulplay-dn2w16-amendment-screen-r2-alive-bands-20261109.json fa7c32a51db058526e436a925ef2156b52ab9debd54881fae6829febcfaced53 \
     --run-directory /out/screen-r2 --output /out/screen-r2/amended-verdict.json
   ```
   **Alive-bands amendment (made during the run, outcome-blind).** At fairness flag 2 (150 pairs, drift), the lead amended the final-fairness gate and the drift diagnosis for screen-r2 only: they stratify by exact alive bands {1}, {2}, {3-4}, {5-6} per side × budget stratum × turn band, with the official sparse-cell merge; the 5% tolerance and the panel's in-run checkpoint are unchanged. The record, `evals/depthn-foulplay-dn2w16-amendment-screen-r2-alive-bands-20261109.json` (sha256 `fa7c32a51db058526e436a925ef2156b52ab9debd54881fae6829febcfaced53`), attests that no outcome statistic of screen-r2 had been computed or inspected, cites the flags and diagnoses by sha256 and hashes the panel state. The screen-r2 binding requires it (`--run-amendment`, pinned by the CLI).
   **Its acknowledgement rule** (`rule.acknowledgement`): flag 2 is acknowledged benign and dispatch resumes only if `scripts/depthn_fairness_final.py window --flag 2 --alive-bands exact` reports `flag.reproduced` (the declared reading over the window equals flag 2's recorded 1.112 exactly) and `finer.within` (exact-band finer difference within 5%). Otherwise the panel stays stopped and the decision goes back to the lead, with no further amendment. Revision 2 of the record (this sha) states that rule; revision 1 (`6b8a5b3d…`) left it out and was never used.
   **No extension.** screen-r2 is a 500-cell screen; no extension to 2,000 cells is declared. screen-r1's extension amendment (`8a708ebb…`) is specific to screen-r1 and is not adopted. A `CONTINUE_TO_2000` verdict goes to the lead for a separate, pre-registered decision.
6. **Footprint:** the same as screen-r1.
   - 4 GPUs, with 3 DN2W16 server processes per GPU (12) behind the sticky proxy;
   - 12 worker lanes on calibrated core pairs;
   - servers and workers pinned to disjoint cores.

## Evidence
| item | sha256 | summary |
|---|---|---|
| Calibration | `5f3ef007c59aefa7…` | MEASURED; ratios 2×1000 ms 1.130 and 4×500 ms 0.902 (screen-r1: 1.058 and 0.924). Foul Play's iteration rates match screen-r1's within a few percent. |
| Move walls | `46750c78…` | from the calibration games |
| Pre-flight on the server tree | `389b71b5…` | 24 games, outcome-blind. Search wall p50 1.87 s, p95 2.52 s; first decisions within the warm p99. Searched share 75.0% of non-timeout decisions; timeouts 0.44%. Fallbacks: guard 70, constructor 99. The declared warm p95 is the pre-flight's RAW arm's (as `check_preflight` requires): encode 0.042 s, greedy 0.384 s, against screen-r1's greedy 0.269 s. `panel.init` checks that it matches the pre-flight; it feeds no decision rule. |

## Known caveat (one-sided)
Showdown sends Gen 3 Pressure's start line only to the holder's own side (`data/mods/gen3/abilities.ts` `addSplit`). Our wire carries it for our own Pressure Pokémon, and the constructor's peer view inherits it as a revealed ability. Live Foul Play never sees it.

This touches only the search arm's reply prior; the RAW arm models no reply. Measured on the 63 RR battles with such lines, by re-pricing with the flagged redaction:
- the reply prior's KL p95 is 6.1e-4;
- the top reply flips in 0.34% of worlds;
- **averaging's played move flips on 4 of 2,170 searchable roots (0.18%)**, under the 1% rule;
- control roots are bit-identical.

The redaction lands later, with the π_FP retrain.

## Readout plan
- the result with its seed-clustered interval;
- the guarded share live;
- the override rate;
- p95 latency;
- a comparison with screen-r1 restricted to the positions both screens searched.
