# MCTS design: test-time search + fpdistill-seeded bootstrap

Status: **planning / draft.** Detailed design for the test-time search policy-improvement operator
and a proposed search-teacher-seeded bootstrap track. Sits under
[`selfplay_mcts_roadmap.md`](selfplay_mcts_roadmap.md) (WS-C forking, WS-D search, WS-E value).

> **Reality check (2026-07):** more of this is already built than the roadmap's "to-do" framing
> implies — see **Current implementation state** below. This doc is therefore about *extending +
> validating* the existing search, not building it from scratch.

## Why now / motivating evidence

Measured on the current 1.5M checkpoints (hazard/behavior probes): the searchless policy is **myopic
on delayed/positional lines** — never sets Spikes (0/243 argmax), blind to the self-hazard feature
(ΔP(Rapid Spin) ≈ 0 across 0→3 injected layers), under-uses setup. This is a **credit-assignment**
limit: the value head can't credit payoffs many turns out. Search computes that payoff *at decision
time* by simulating forward. And **fpdistill** (BC-distilled foul-play + RL) already encodes the deep
lines (~17% foul-play, ~80% max-damage), so it's a strong search prior.

## Current implementation state (what already exists)

| Piece | Status | Where |
|---|---|---|
| Forking = **replay-from-root** | **built** | `replay_branching.py`, `search.py` |
| **1-ply PUCT branch search** (prior-weighted, value-head leaf eval, optional leaf rollouts) | **built** | `search.py::puct_branch_search`, `value_branch_search`, `flat_branch_search` |
| Opponent modeling (greedy / top-k prior / policy planners, weighted scenarios) | **built** | `search_policy.py` |
| Net+search **Policy adapter** (`select_action` = net+search) | **built** | `search_policy.py` |
| Search **behavior benchmark** (action-change rate, candidate count, per-move cost) | **built** | `search_benchmark.py` |
| Value-**calibration** tooling (ECE, affine/isotonic fit + transform) | **built** | `value_calibration.py`, `neural_policy.py` |
| poke-engine reversible backend (apply/reverse, doctor, smoke) | **wired + installed** | `engine_cli.py`, `poke_engine_backend.py` |
| Unit tests for search / search_policy / benchmark | **built** | `tests/test_search*.py` |

So forking (WS-C) and a first-cut search (WS-D) are **done**. The scaffolding is real.

## What's actually missing before search is *viable* (beats net-alone, ladder-ready)

In priority order:

1. **Value-head search-readiness (WS-E) — the hard prerequisite.** Calibration *tooling* exists, but
   there's no evidence the value head is calibrated/ordered well enough to guide search. A myopic or
   miscalibrated leaf value makes 1-ply search **no better (or worse) than the net**. This gates
   everything → measure ECE + leaf ranking on held-out data for the candidate nets; apply a fit
   transform if it helps.
2. **Strength validation — the go/no-go we don't have.** `search_benchmark.py` measures *how often
   search changes the net's action* and *what it costs* — **not win rate vs net-alone**. So the
   roadmap acceptance gate ("net+search beats net-alone by a clear margin") has **never been
   demonstrated**. We need a head-to-head strength benchmark: net+search vs net-alone (and vs
   foul-play). Until this exists we don't actually know the current search helps at all.
3. **Search depth.** It's **1-ply** (branch each root action, evaluate the immediate result). For the
   delayed-value lines that motivated this (setup sequences, hazard payoff over many switch-ins),
   1-ply + a myopic value head can't see the payoff. Levers: **deeper leaf rollouts** (already
   supported via `leaf_rollout_*`) with a decent rollout policy, or extend to a **multi-ply tree**.
   Deciding 1-ply-with-rollouts vs a real tree is the core design question.
4. **Determinization for hidden-info ladder play.** Search currently has **no belief-set sampling**
   (`grep` finds none in `search.py`/`search_policy.py`). Fine for perfect-info self-play/benchmark;
   required for real ladder play (hidden opponent set). Wire `belief.py`/`Gen3RandbatSource` to sample
   K opponent worlds at the root and average.

Not blocking: chance handling (replay re-simulates real RNG; averaging rollouts approximates the
distribution — explicit KO/no-KO grouping is an optimization), simultaneous moves (handled by the
opponent planners for 1-ply; DUCT only matters for a deep tree).

## Design principles / hard constraints

- **Showdown is ground truth**; poke-engine only after a Gen-3 equivalence spike
  ([`poke_engine_assessment.md`](poke_engine_assessment.md)).
- **foul-play is GPL** → benchmark / behavior source only; never imported.
- **Value quality bounds search quality** — WS-E is a prerequisite, not a nicety.
- **North-star lanes.** Searchless self-play + **test-time** search = recipe-faithful **flagship**.
  fpdistill-seeded search and any **in-loop** search (AlphaZero-style) = **sanctioned parallel arm**,
  kept off the flagship so the clean baseline stays measurable ([`goals.md`](goals.md)).

## Component design (extending what exists)

- **Forking (WS-C):** keep **replay-from-root** (built); validate per-move cost against the target
  search budget; escalate to snapshot or poke-engine only if the budget demands.
- **Determinization:** add root belief-set sampling (K worlds, average) — the main *new* wiring.
- **Search core (WS-D):** the extension question — is **1-ply + deeper leaf rollouts** enough to
  recover delayed-value lines, or do we build a **multi-ply PUCT tree**? Prototype rollout-depth
  first (cheap, already supported), escalate to a tree if depth-1 underperforms.
- **Net integration:** prior = policy-head softmax with a **temperature** (soft, so PUCT can override
  a peaked prior); leaf value = the (calibrated) value head via the pluggable `value_fn`.
- **Value head (WS-E):** measure calibration + ranking; improve targets / ranking loss / transform
  until search-ready. Verify the candidate net's head (incl. fpdistill's).

## The fpdistill-seeded bootstrap / expert-iteration track (parallel arm)

Use fpdistill — itself a BC distillation of the foul-play MCTS agent — as the **prior + value** of the
search. Expert iteration warm-started from a search teacher; a prior that already knows the deep lines
should explore them immediately instead of relying on a myopic net's exploration term. Expected
ordering: **search(fpdistill) > fpdistill-alone > search(self-play net)** on foul-play.

- **Test-time only** (E0 below): runnable **now** — `search_policy` with fpdistill as prior+value_fn.
  No training change.
- **Expert iteration**: `search(net) → data → re-distill → repeat` — in-loop search (the expensive
  paradigm MIT avoids) → strictly a parallel arm, measured against the flagship.
- Caveats: value-head accuracy is the linchpin; keep the prior temperature soft; imitation-seeded, so
  it lives beside the from-scratch line (the roadmap's north-star note cautions against cloning
  foul-play as a teacher).

## Experiment sequence

- **E0 — search(fpdistill) strength benchmark (cheapest, runnable now).** Wire fpdistill as
  prior+value into the *existing* `search_policy`; **build the missing strength benchmark** (win rate
  vs net-alone, vs fpdistill-alone, vs foul-play). Re-run `hazard_probe`/`behavior_probe` on the
  search-augmented policy to see if search now sets Spikes / uses setup. This simultaneously (a) fills
  gap #2, and (b) tests the whole search bet.
- **E1 — value-head search-readiness (WS-E, gap #1).** Calibration + ranking audit on self-play-1.5M
  and fpdistill; apply transform if it helps. Gates E0's interpretation.
- **E2 — depth (gap #3).** Sweep leaf-rollout depth / policy; if 1-ply+rollouts underperforms on deep
  lines, prototype a multi-ply tree.
- **E3 — determinization (gap #4).** Root belief-set sampling for hidden-info play; needed before
  ladder.
- **E4 — (optional parallel arm) expert-iteration loop.** Only if E0 shows a large search win and the
  forking budget supports in-loop generation.

**Gate:** net+search beats net-alone by a clear margin (fixed yardstick + head-to-head). E0 is the
first time we'd actually measure this.

## Files (mostly *extend*, not new)

- `search.py` — extend depth (rollout config or a tree) if 1-ply underperforms.
- `search_policy.py` — fpdistill prior+value wiring; belief-determinization hook.
- `search_benchmark.py` — **add a win-rate-vs-net strength mode** (currently behavior/cost only).
- `belief.py` — opponent-set determinization sampler for the searcher.
- `value_calibration.py` / `neural_policy.py` — apply/verify a calibration transform for leaf eval.
- `poke_engine_backend.py` — only if adopted as a fast backend (post equivalence spike).

## Open questions / risks

1. **Value-head search-readiness** — the single biggest risk and gate (E1).
2. **Does 1-ply+rollouts recover the delayed-value lines, or do we need a tree?** (E2.)
3. **Forking cost** at useful rollout depth (replay-from-root vs snapshot/poke-engine).
4. **fpdistill prior over-narrowness** — tune temperature so PUCT explores off-teacher lines.
5. **poke-engine Gen-3 equivalence** — unproven; blocks its use as a fast backend.
6. **In-loop compute** for expert iteration — likely too slow at scale; keep E4 gated on E0.

## Next step

**E0 + E1 together:** measure value-head calibration (E1), then run the *missing* strength benchmark
of search(fpdistill) vs net-alone / fpdistill-alone / foul-play using the existing `search_policy`.
This reuses nearly all existing machinery, needs only a strength-benchmark harness + fpdistill wiring,
and is independent of the runs currently training — the fastest way to learn whether search is viable.
