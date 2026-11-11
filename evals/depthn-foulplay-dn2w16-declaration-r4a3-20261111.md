# screen-r4a3: screen-r4a2's configuration at a 3.0 s deadline (declaration notes)

**Pre-registered panel declaration:** `depthn-foulplay-dn2w16-declaration-r4a3-20261111.json`, sha256 `e2e39feb7e6c1000b4ea0f090bebc2303d75788d8fea9ea5014ed803c96ce422`.
- `panel.validate` and `check_preflight` pass from the server-tree checkout `92d8af8c`.
- The dispatch commit is `594a0ee2`: the server tree plus this JSON alone, as `server_tree_check` requires.
- The panel's schema is closed, so the conditions it cannot carry are recorded here.

## What screen-r4a3 answers

screen-r4a2's operator (variant B: br_fp on π_FP red3-full, view b, gated by NOISE_GATED_K2, on r4a2's pre-A2 engine and guard) **at the 3.0 s deadline the ladder's bar uses**.
- screen-r4a voided at 3.0 s on latency, and screen-r4a2 re-declared 5.0 s.
- screen-r4a3 removes the latency causes instead, with changes that leave every prepared and priced output byte-identical.
- The engine and guard pair (tier A and B fidelity, guard shrink) belongs to a later screen.

## What changes from screen-r4a2 (and only this)

1. **The deadline:** `deadline_seconds` 3.0, and the breakage latency bound with it. Both were 5.0.
2. **The server tree:** `92d8af8cb51b6c889fd359193275b3f63b05490b`. This is r4a2's `982909e1` plus:
   - **Latency changes,** each identity-tested and server-side behind its own flag (both flags on in this screen):
     - incremental public roots, once-per-root line scans and a once-per-request catalog check (`--incremental-roots`);
     - the eager peer fold (`--eager-peer-fold`), which advances each battle's peer wire fold on every root so the first private reveal no longer folds the whole battle (1–2 s spikes).
   - **Routing-only proxy placement:** a worker's battle end frame stops the battle counting toward its GPU's load, and new battles go to the GPU with the fewest live battles, then fewest in flight, then least recently placed. The proxy runs with `--sticky-idle 20`. No module of the live server imports the changed client or panel code.
   - **The pre-flight seat balance** (pre-flights only).
3. **The calibration:** a new worker's.
   - RAW only, solo then at the planned 12 lanes, before any timed game; seed band 2103402000 / 2103502000 × 200.
   - Status MEASURED; ratios 2×1000 ms 0.947 and 4×500 ms 0.915.
   - Its move walls and fairness evidence come from the same directory.
   - Every pre-flight ran on the same worker instance.
   - The RAW game modules are byte-identical between the calibration's tree and the server tree.
4. **The pre-flight:** run 4 (`preflight-s4a3d/preflight.json`), described below.
5. **The run:** run directory `/out/screen-r4a3`, its own adoption record (`depthn-foulplay-dn2w16-amendment-screen-r4a3-adoption-20261111.json`), and its own dispatch commit.

Everything else is screen-r4a2's:
- the reply model and its pins, NOISE_GATED_K2, arms, cells, reserve, RULE_R3 arm fairness, base config and engine tables;
- 12 lanes, and 3 servers per GPU on one 4-GPU node.

## Identity certificates

**On server tree `73ba01fe`** (the search code of `92d8af8c`; only the proxy, the search client, the panel and tests differ after it):
- **Prepare, classify, root and peer-belief digests:** 1,043 of 1,043 decisions identical, against `982909e1` on the frozen r4a3 pre-flight battles. 270 decisions carry a private-reveal belief, and all 270 match.
- **B2 fold check:** 341 of 341 equal (incremental feeds against one-shot folds).
- **Error paths:** the eager feed was made to fail partway, before or on the first reveal, in 6 battles. 0 mismatches.
- **Priced-output identity on GPU,** at the live servers' arguments and matmul precision 'highest':
  - 325 decisions: action, status, cause and refusal code exact; gate and guard records exact.
  - Q tables bitwise equal on all 294 priced decisions.

**Precision:** both screens' servers report `matmul_precision` 'highest', its pin 'highest' and `jax_enable_x64` false.

## Pre-flight history (all runs, failures included)

Every run: 48 outcome-blind games, 12 lanes, 3.0 s deadline and 0.15 s reserve, the same gate. Each failure was fixed by removing its identified cause, never by re-rolling seeds.

| Run | Seeds | First 1,000 timeouts | Result | Cause | Fix |
|---|---|---|---|---|---|
| 1 | +0 | 4 (0.40%) | FAIL | Prepare spikes at a battle's first private reveal: the peer fold rebuilt from scratch | Eager peer fold |
| 2 | +100 | 7 (0.70%) | FAIL | Three battles priced at once on one GPU: the proxy counted finished battles as live for 60 s | `--sticky-idle 20` |
| 3 | +150 | 0 | Placement check FAIL | One GPU reached 3 running battles from a stale tie | Battle end frames, and ties by in-flight then least recent |
| 4 | +200 | **0** | **PASS** | | |

**Run 4** (the declared pre-flight):
- **Timeouts:** first 1,000 searched decisions 0 (one-sided 95% upper bound 0.30%); decisions 20+ 0 of 546 (upper bound 0.55%).
- **Seats:** 12 games per arm per seat, 0 timeouts in either seat.
- **Search wall:** p50 1.75 s, p95 2.16 s, max 2.63 s, against the 2.6 s p95 bound.
- **Placement:** every new battle went to a GPU at the minimum true running count (0 violations). At most 2 battles ran on one GPU, with up to 7 running at once.
- **Routing:** end frames for all 24 battles; 0 off-server routes.

## Verdict rules

The declaration's `paired()` applies through screen-r4a2's adopted amendments, unchanged, through this run's adoption record:
- **Unknown cells:** an unknown share above 5% of an arm's declared cells counts as breakage, replacing `unknown_cells`. The conservative lower bound counts UNKNOWN search cells as losses and UNKNOWN RAW cells as wins.
- **Harm mirror.**
- **Both bounds:** the bar needs the known-pairs **and** the conservative lower bound above 0; harm needs both upper bounds below 0.
- **Final fairness:** under RULE_R3.
- **Extension:** the pre-registered extension to 2,000.

A flag that fails RULE_R3 stops the run and goes to the lead, with no amendment mid-run. The readout also reports UNKNOWN cells by arm and cause, and a drop / loss / win sensitivity line (descriptive).

**Launch timing:** screen-r4a3 launches on the lead's approval of this declaration, before screen-r4a2's outcome is known. None of screen-r4a2's paired, score or outcome statistics were read for this decision.
