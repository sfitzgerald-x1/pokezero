# Dynamic search budget — implementation plan

Written 2026-08-12.

> **PROVENANCE, stated first because it changes how this document should be
> read.** The goal that commissioned this work names `docs/dynamic-search-budget-plan-20260812.md`
> as the definition of the behaviour to implement. **No such file existed** — not in
> this checkout, not in the deploy repo, not anywhere on the filesystem (searched by
> name and by pattern). The two preceding goals each had their plan placed in `docs/`
> beforehand; this one did not. Rather than stall, this operator authored the plan
> below from two things that are not guesswork: the **latent mechanism already present
> in the crate**, and the **measurements from `mcts_value_gap_findings_20260812.md`**
> that say what a dynamic budget should be expected to buy. If the intended meaning of
> "dynamic MCTS behavior" was something else — adaptive `c_puct`, adaptive depth,
> wall-clock-budgeted search, progressive widening — then §1 is the wrong target and
> should be corrected before §3 is trusted.

---

## 0b. WHAT THE OWNER ACTUALLY SPECIFIED — this supersedes §1's target

§1 below guessed the axis wrong. It is kept because its measurements still stand and
because the reasoning error is instructive, but the design that shipped is this:

* **`sims` is the TOTAL budget for a decision**, divided across belief worlds:
  16,384 over 4 worlds is 4,096 each. Not per-world.
* **Worlds step DOWN one at a time** (4 → 3 → 2 → 1), sims per world rising to hold
  the total constant. These rungs are compute-neutral: they trade belief breadth for
  depth-fill. A world is dropped once the per-world leaders agree.
* **Depth marches forward SOLELY on saturation pressure**, at a near-full threshold
  (0.9), never on how contested a decision looks. A deeper search that did not fill
  the shallower depth explores its new plies too thinly to back them up and can be
  WORSE than the depth beneath it.
* **depth_min is never below 2.** A one-ply search is no better than the raw policy.
* Defaults: **depth 2→8, worlds 4→1** at 16,384 total.

§1's own target — the sim-budget early stop — was built and is committed
(`--engine-early-stop`), but it is a different axis and is not what these variants
measure.

## 0c. WHAT AN INDEPENDENT REVIEW FOUND, and what changed because of it

Recorded here rather than only in the PR, because three of these are things the
FIRST implementation got wrong in ways a green test suite did not notice, and the
lesson is about the tests as much as the code.

| # | finding | why it mattered | resolution |
|---|---|---|---|
| F1 | the no-op branch divided the budget by the world count | every **fixed** `worlds>1` cell silently ran at `budget//worlds` — 4,096 of 16,384 at the default `worlds=4` — under an *unchanged* `config_id`, so it would have pooled with banked shards that ran the full budget | the branch returns `sims=None` ("untouched"); a test asserts the native positional list element-for-element against the pre-ladder call |
| F2 | the fallback guard tested `decision.policy_id != self.policy_id` | `_fallback` returns *this* policy's id, so the guard was dead code: the ladder marched past failed rungs on **stale** signals and returned the failed rung's decision even when an earlier rung had a good one — making escalation a strength regression | detect via the `fallback_decisions` counter delta; keep `last_good`; reset both signals per rung |
| F3 | the early-stop replay ran at the total budget and the depth cap | it then fed the saturation test a licence to deepen that the rung had not earned | the replay receives the rung's `sims` and `depth` — but the FIRST version of this fix also multiplied by the collapse multiplicity, which was wrong; see **N1** in §0d |
| F4 | five per-decision rates were per-**rung**; `world_search_abort_rate` was **−1.75** | `searched_decisions` counts rungs. Reading a rung-denominated rate as per-decision is exactly the error that once reported a measured 2× cost regression as a 23% saving | new `world_search_attempts` (per world **per rung**) carries the abort rate; `iterations_per_ladder_decision` / `search_wall_per_ladder_decision` are the cross-cell cost denominators; `ladder_rungs_per_decision` is emitted so a reader can see which denominator applies |
| F6 | `_search_ladder` had **zero** coverage — 7 of 9 seeded mutants survived 408 tests | the suite could not distinguish a working ladder from a ladder that was silently a fixed cell | `LadderDriveTest` drives the wrapper with a recording `_search_model`; 11 of 11 mutants died. **Not enough** — stubbing `_search_model` left the signals' COMPUTATION unpinned; see **N2** in §0d |
| F7 | `early_stop_min_sims` was validated against the total | a floor of 64 against a 128 total across 4 worlds is a floor *above* the 32-sim rung, so the stop rule could never fire on the rungs that need it | clamped to the rung's per-world budget |
| F8 | `--engine-override-telemetry` had no refusal outside `engine-mcts` | worse than a no-op: the flag is deliberately **excluded** from `config_id`, so the shard's own `"override_telemetry": true` is the only record the instrument ran — and under `raw` that record was false | refused, like every sibling flag |
| F11 | the bridge accepted `depth_min=1`, the engine refused it | the refusal arrived after the pod had claimed its GPUs | mirrored, with the "one-ply" reason in the message |
| F5, F9, F10, F12 | duplicated `ladder_escalations += 1`; `ladder_margin` assigned and never read (with tests pinning it); comments describing the removed margin mechanism; `worlds × sims` overspending when `search_sims < worlds` | — | all removed/corrected, though F12's clamp introduced **N6** and was replaced by a refusal; a test refuses `ladder_margin` by `TypeError` so a cell key carrying it fails at config time rather than banking a mislabelled result |

## 0d. WHAT A SECOND REVIEW ROUND FOUND — including a defect in a §0c FIX

The first round's fixes were verified and 11 of 12 held. The 12th did not, and the
lesson is that **a fix needs its own adversarial pass**: F3's patch was itself wrong,
and the test written alongside it could not see the defect because the test used
uncollapsed worlds, where the wrong factor is ×1.

| # | finding | why it mattered | resolution |
|---|---|---|---|
| N1 | F3's replay multiplied by the **collapse multiplicity** | the ambiguous-stop replay walks RECORDS, so an N-group is replayed N times — the multiplicity is already spent by the iteration. Scaling each replay by it too cost **N² × rung_sims**: at a 4,096 rung and a 3-group, 36,864 sims for one group against an intended 12,288 — 2.25× the whole decision's budget — and the oversized trees then re-forged the exact saturation licence F3 removed | pass `rung_sims` alone; the test now uses a **real** collapsed group, since at multiplicity 1 the defect is invisible |
| N2 | the two **signal producers** had zero coverage | `LadderDriveTest` stubs `_search_model` out, so it pinned how the wrapper *consumes* the signals and nothing about how they are *computed*. Turning the saturation threshold into `>= 0.0` (depth advances every rung — the one thing this design forbids) and the agreement test into `True` (every world drops immediately) both survived **637** tests. F6 had been closed on wrapper coverage alone | `LadderSignalProducerTest` enters at `_search_model` with the rung overrides set, and pins the D−1 ceiling, the near-full threshold against a 0.5 case, unreadable reports not reading as saturated, and per-world leaders over an aggregate |
| N3 | the 20 s/turn latency gate read the **per-rung** wall | `latency_of` reached for `search_wall_per_searched_decision` even though `seat_block` hoists the per-decision figure precisely so it would not. A cell at 2.1 rungs/decision reports 5.7 s while its true per-decision wall is 12 s, so the cap silently stops gating on exactly the cells this feature produces | the gate prefers `search_wall_per_ladder_decision` and emits `gate_denominator` so a reader never has to guess |
| N4 | `fallback_rate` charged **recovered** fallbacks | `_fallback` charges per RUNG. A rung failing after an earlier rung succeeded is not a fallen-back decision — the engine returns the earlier rung's real search — yet the rate read **1.0** for a decision that searched fine, and the power report's health gate would drop a healthy cell on it | `ladder_recovered_fallbacks`, netted out of the rate; both numbers emitted so the taxonomy stays whole |
| N6 | the F12 clamp **broke the `worlds_min` floor** | at `search_sims=2, worlds=4, worlds_min=3` it ran 2 worlds while the floor said 3 — breaking the very floor that turns the axis dynamic — and left rung 0 asking for fewer worlds than `decide()` had charged, putting a **false 0.25** into `world_search_abort_rate`, the mirror of the −1.75 | `search_sims < worlds` is **refused** on a dynamic cell; the clamp and the dedupe it needed are deleted |
| N8 | the **override rate** was rung-weighted | `override_measured_decisions` is charged per `_search_model` call, so an escalating decision voted once per rung and the headline rate came out weighted by how far each decision climbed — incomparable to a fixed cell's, which is the only comparison the override study is about. `search_config_id` pools telemetry-on with telemetry-off, so nothing else would have caught it | `_search_ladder` rewinds the ledger to the **winning rung's delta**, so one decision casts one vote; the documented identity now states both forms |
| N5 | the renamed payload key was unpinned | the rename test asserted the *attribute*; `to_dict` was free to emit the old key and every consumer reads the payload | pinned on the emitted key |
| N7 | the stop-floor clamp's *direction* was undocumented | not a defect — clamping the floor to the rung sets it to that rung's **entire** budget, so a stop can fire no earlier than the rung's last batch. The alternative is early-stop being silently dead on every rung below the floor. But the validator's message still described the old contract | message and comment corrected; a fixed cell is provably untouched (`_ladder_sims_override` is `None`) |

**Mutation screen: 25 seeded mutants, 25 killed** (previous rounds: 7 of 9 surviving, then
4 of 20). The four that survived the last round were N1, N2a, N2b and N3 — every one a
place where a *rule* rather than its plumbing was unguarded.

## 0e. WHAT A THIRD REVIEW ROUND FOUND — the same defect on its third and fourth surface

Round 2's fixes held. Round 3 found nine more, and the pattern is the one worth
carrying out of this exercise: **the N8 override defect had four surfaces, and each
round fixed one more of them.** Counters (round 2), rows (found while briefing round
3), addresses (round 3). A fix aimed at one surface of a defect is not a fix.

| # | finding | why it mattered | resolution |
|---|---|---|---|
| NEW-1 | the rewind left the override **ADDRESSES** | `override_disagreements` is a forkable `(battle_id, round, seat)` replay handle. One left by a discarded rung makes the shard report zero overrides while carrying an address claiming one — and a fork probe replaying it lands on a decision the engine did **not** override | superseded addresses are dropped and counted in `override_addresses_superseded`; the module now states ONE rule for all four surfaces — a count or an address is a claim about the returned decision and is rewound, a cause taxonomy is a claim about the run and is not |
| NEW-2 | `mcts/analyze_seltune.py` was a **stale consumer** of both N3 and N4 | it computes `s/dec` from `searched_decisions` (the per-rung field N3's own docstring says must never be read on a dynamic cell) and recomputes `fallback_rate` from raw counters, re-charging recovered fallbacks. `analyze_value_gap.py` and `foulplay_power_report.py` were updated; this one was missed | `ladder_sdec` and `rungs/dec` printed beside `s/dec`, blank on a fixed cell; `fallback_rate` nets `ladder_recovered_fallbacks`; the three ladder counters are aggregated |
| NEW-3 | N3's fix **re-opened** "UNEVALUABLE is never a PASS" | `ladder_decisions` is charged BEFORE the first rung, so a cell whose every decision fell back at rung 0 still emitted a wall — the *fallback* wall — and the cap read it as `PASS - mean 0.30s` where it used to read UNEVALUABLE | the ladder wall is emitted only when `searched_decisions > 0` |
| NEW-4 | **FIXED cells were being stamped** | `_search_ladder` is the dispatch for every model decision, so the unconditional stamp put `ladder_rung`/`ladder_superseded` on a fixed cell's rows — changing the banked row schema on the one branch whose central promise is that a fixed cell is untouched, and contradicting two docstrings consumers are told to rely on | stamped only when the cell is dynamic; **both directions** pinned, since the mutant that stamped everything and the one that stamped nothing had both survived |
| NEW-5 | the recovered-fallback **producer** was untested | moving the increment outside its `last_good is not None` guard survived all 574 tests: a run where every decision fell back at rung 0 would emit `fallback_rate 0.0` with `fallback_decisions == decisions` — a fully contaminated cell reading perfectly clean, the inverse of the failure the counter was added for. The only test set the fields by hand. **N2's lesson repeating on a counter introduced by the N2 round** | tests drive `_search_ladder`; the emitted counter is pinned as well as the netted rate |
| NEW-6 | the documented ladder identity was **false**, and this campaign's JSON asserted it was tested | `ladder_decisions` counts a decision that fell back at rung 0 and the override ledger cannot, so `measured + unmeasured == ladder_decisions` fails and a consumer deriving `unmeasured := ladder_decisions − measured` overcounts by exactly the rung-0 fallbacks | `ladder_unsearched_decisions` emitted as the third term; both forms pinned; the JSON claim corrected |
| NEW-7 | `decision_rows()` **failed open** | `row.get("ladder_superseded")` → `None` → falsy → kept. Every shard banked before the stamp existed has a non-zero `ladder_decisions` and unstamped rows, so re-analysing one silently restored full rung-weighting with no diagnostic | refuses a dynamic shard whose rows carry no stamp |
| NEW-8 | the row cap can **delete the straddling decision** | variant 2 runs up to 19 rungs per decision against a 4,096-row cap, so at the boundary a decision can have rows in the shard and none passing the filter — a systematic bias toward escalating decisions | counted as `ladder_decision_rows_lost` |
| NEW-9 / M23 | two of four override surfaces rewound, two not, with no stated rule; and `winning_row_span`'s `None` default rests on an equivalence in *another* function | — | the rule is stated once; the equivalence is recorded where it is relied on |

**Mutation screen: 36 seeded across three rounds, 36 killed.** The honest progression
is what matters: 7 of 9 surviving → 0 of 15 → **4 of 20** → 0 of 25 → **4 more found by
round 3** → 0 of 36. Twice a round closed at zero survivors and the next round found
more, both times because the screen was aimed at plumbing rather than at rules, or at
one surface of a defect rather than all of them.

## 1. What "dynamic" means here, and why this target

**Today the per-decision budget is fixed.** Every decision spends `search_sims`
simulations regardless of whether the choice was settled after 64 or genuinely
contested at 16,384.

**The mechanism to make it dynamic already exists and has never been switched on.**

| piece | where | state |
|---|---|---|
| visit-lock stop rule | `rust/pokezero-search/src/tree.rs:160` `root_visit_lock` | implemented, unit-tested (`tree.rs:1780`) |
| per-world stop | `rust/pokezero-search/src/model.rs:1204` | implemented, gated on `early_stop_min_sims > 0` |
| cross-world acceptance | `src/pokezero/engine_search.py` `_locked_aggregate_choice` | implemented, fail-open |
| full-budget replay on ambiguity | same | implemented |
| telemetry | `early_stop_triggered_worlds`, `_accepted_decisions`, `_full_budget_replays`, `_sims_saved` | implemented |
| config surface | `EngineMctsConfig.early_stop` (default **False**), `early_stop_min_sims` (default 64) | **latent** |
| reachable from bridge / driver / launcher | — | **NO. Not plumbed anywhere.** |

`events.rs:2388` says it outright: *"`early_stop` defaults off, so this is latent, not
live."* So the implementation work is **not** inventing a stop rule. It is (a) plumbing
the existing one end to end so a campaign can ask for it, and (b) measuring whether it
saves budget without costing strength.

**The stop rule, stated exactly.** Stop once `completed >= early_stop_min_sims` and the
root visit leader cannot be overtaken: `top - runner_up > remaining`. That is sound by
construction for a visit-argmax policy — no further simulation can change which arm has
most visits. It is computed on the **acting** seat's arms (`record["side_key"] ==
"side_one"` is passed per record), not the opponent's.

### Why this is worth doing now — from measurement, not intuition

`mcts_value_gap_findings_20260812.md`, same build, same seeds:

- **Budget past s2048 is not buying strength.** Opponent priors ON at **s2048** beat
  the raw policy by **+13.81 pp** (p = 0.0011, n = 210); the same config at **s16384**
  beat raw by **+10.00 pp** (p = 0.021, n = 240). Eight times the budget, 7.4× the
  per-decision wall (13.59 s against 1.83 s), and no more strength.
- **Most decisions are not close.** The top-1/top-2 root Q gap by quartile is 0.0015 /
  0.0061 / 0.0135 / 0.0511. In the smallest quartile the two best arms differ by 0.15 pp
  of win probability — those decisions are settled, and spending 16,384 sims on them is
  waste by construction.
- **The value ceiling on discrimination is ~1.8 pp**, stable across every cell
  regardless of belief quality or calibration. So there is no hidden upside in searching
  settled positions harder.

A dynamic budget is therefore the one lever the findings actually endorse: keep
s16384-class depth for the contested tail, pay s2048-class cost on the settled majority.

---

## 2. Success criteria, pre-registered

Accept the feature iff **all** of:

1. **It engages.** `early_stop_accepted_decisions > 0` and
   `early_stop_sims_saved > 0` on a scored cell. A flag that is on and never fires is
   reported as CANNOT RUN, not as a null.
2. **It is applied, not merely requested.** Report
   `early_stop_accepted_decisions` against `early_stop_full_budget_replays`. **A stop
   that gets replayed at full budget saves nothing** — the replay path is fail-open by
   design, so `triggered` is not the applied denominator. This is the same
   requested-vs-applied trap opponent priors set in the previous campaign.
3. **Strength does not regress.** Seed-paired win delta against the same config with
   `early_stop` off, above the standing **−2 pp** guardrail, McNemar on discordant
   pairs. n = 125 seeds × 2 seats = 250 games per arm.
4. **Cost falls materially.** `search_wall_per_searched_decision` drops, and the drop is
   consistent with `early_stop_sims_saved / total_iterations`. If wall does not fall
   while sims_saved is large, the saving is being eaten elsewhere (encode dominates this
   workload at 71–80% of search wall) and that must be said rather than absorbed.

Registered prediction, so it cannot be retrofitted: given the Q-gap distribution,
expect **sims saved on the order of half the budget** and a strength delta **within
±2 pp** — i.e. the feature should be roughly free in strength and large in cost.

Registered failure mode: if `early_stop_full_budget_replays` dominates
`accepted_decisions`, the cross-world aggregate is rarely lock-clean, the feature saves
little at w > 1, and the honest conclusion is "works at w1, not at w4" rather than
"works".

---

## 3. Implementation

**3a. Plumbing (the actual work).** Follow the chain proven three times in this
programme (`--engine-override-telemetry`, `--engine-oracle-belief`,
`--opponent-journal`):

`cell key → mcts/foulplay-power-k8s.sh → scripts/foulplay_paired_eval.py →
python -m pokezero.foulplay_bridge → ControlledFoulPlayConfig → EngineMctsConfig`

- `--engine-early-stop` (store_true) and `--engine-early-stop-min-sims` (int).
- Forwarded **only when set**, so an unset cell renders the byte-identical child argv it
  rendered before the flags existed and stays poolable with every banked shard.
- **In `config_id`**: this one changes the search — it changes how many simulations a
  decision gets. A `+early-stop` fragment (with the min-sims value when non-default), or
  two cells that differ in budget policy will pool into one and strand the control.
- Refuse outside `policy_mode=engine-mcts`; refuse when `leaf_eval != model`, which the
  config validator already enforces (`engine_search.py:457`).

**3b. Tests.** Mirror `tests/test_oracle_belief_arm.py`'s shape:
- the flag reaches the bridge when set and is absent when unset;
- the driver's choices come from the bridge's own declared surface, not retyped;
- it DOES enter `config_id`, and two min-sims values do not pool;
- the crate's stop rule is exercised at a known tree (the existing `tree.rs:1780` test
  already pins `root_visit_lock`; add the config-level path);
- an applied counter moves, and a replayed stop does **not** count as saved.

**3c. Measurement.** Two cells, same 125-seed band, same build, same node:

| cell | config |
|---|---|
| `d8-s16384-b64-w1-opon-es` | + `early_stop`, `min_sims` 64 |
| `d8-s16384-b64-w1-opon` | banked control from the value-gap campaign, same build |

w1 deliberately: at w1 the cross-world normalisation that drives the fail-open replay
cannot bite, so this isolates the stop rule itself. If it works at w1, a follow-up asks
whether it survives w4 — that is the second experiment, not this one.

Canary one shard first and confirm criterion 1 before the fleet: this programme has
twice built an image whose flag could not be reached, and once shipped a wave whose
measurement was empty.

---

## 4. What this plan does NOT do

- It does not make **depth** dynamic, nor `c_puct`, nor the world count. Those are
  separate levers and each needs its own control.
- It does not implement a **wall-clock** budget. Simulation count is the unit the crate
  already stops on; a time budget would make results non-reproducible across hosts,
  which this programme's cost measurements cannot afford.
- It does not touch the **opponent-priors default**, which the value-gap findings name
  as the actual binding constraint and which is worth +10 to +14 pp on its own. Dynamic
  budget is a **cost** optimisation, not a strength lever, and must not be sold as one.
