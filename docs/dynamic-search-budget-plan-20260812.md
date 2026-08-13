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
