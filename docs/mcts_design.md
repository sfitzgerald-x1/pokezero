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

| Piece | Status | Where / caveat |
|---|---|---|
| Forking = **replay-from-root** | **built** | `replay_branching.py`, `search.py` |
| **1-ply prior-weighted value *ranking*** (value-head leaf eval, optional leaf rollouts) | **built, but NOT a tree** | `search.py::puct_branch_search` — a **single-pass** scorer: each candidate is hard-coded `visits=1` (`_puct_candidate`), no selection→expansion→backup loop, no visit accumulation. "PUCT" is generous; it's 1-ply prior-weighted value ranking, not iterated MCTS. |
| Opponent modeling (greedy / top-k prior / policy planners, weighted scenarios) | **built** | `search_policy.py` |
| Net+search **Policy adapter** | **built** | `RootPUCTSearchPolicy` via **`select_action_with_context`**; plain `select_action` only runs the *fallback* (no context → no search). |
| Controlled foul-play **strength harness** | **built, smoke-verified** | `foulplay_bridge.py`, `scripts/root_puct_vs_foulplay.py` — runs foul-play as a **separate process** over a fake Showdown websocket while PokeZero owns a seeded BattleStream, so root-PUCT gets the replay seed + trajectory context it needs. Full-game smoke: 1.5M checkpoint root-PUCT vs foul-play, 24 searched decisions, 0 fallbacks. This is a harness validation, **not** a strength result. |
| Search **behavior benchmark** (action-change rate, candidate count, per-move cost) | **built** | `search_benchmark.py` — **behavior/cost only, no win rate**; and the counterfactual harness replays branches against the **recorded** opponent action (`search_benchmark.py:345`) → oracle leakage (see E0). |
| Value-**calibration** tooling (ECE, affine/isotonic fit + transform) | **built** | `value_calibration.py`, `neural_policy.py` |
| **Belief determinizer** `sample_opponent_determinizations` | **built, but NOT wired into search** | `belief.py` — emits concrete opponent realizations from the belief view; nothing injects them into the branch env yet (see "missing" #4). |
| poke-engine reversible backend | **probe only** | `engine_cli.py`, `poke_engine_backend.py` — apply/reverse smoke exists, but **Gen-3 outcome equivalence is unproven** (`poke_engine_assessment.md`); not a usable backend yet. |
| Unit tests for search / search_policy / benchmark | **built** | `tests/test_search*.py` |

So forking (WS-C) and a first-cut *1-ply* scorer exist. The scaffolding is real, but it is **not yet
an iterated tree search**, and none of it is validated for strength.

## What's actually missing before search is *viable* (beats net-alone, ladder-ready)

In priority order:

1. **Value-head search-readiness (WS-E) — the hard prerequisite.** Calibration *tooling* exists, but
   there's no evidence the value head is calibrated/ordered well enough to guide search. A myopic or
   miscalibrated leaf value makes 1-ply search **no better (or worse) than the net**. This gates
   everything → measure ECE + leaf ranking on held-out data for the candidate nets; apply a fit
   transform if it helps.
2. **Strength validation — the go/no-go still pending.** `search_benchmark.py` measures *how often
   search changes the net's action* and *what it costs* — **not win rate vs net-alone**. A new
   controlled foul-play harness now exists for full-game external-opponent reads, but only smoke
   evidence exists so far. The roadmap acceptance gate ("net+search beats net-alone by a clear
   margin") still requires an adequately powered raw-vs-search read (≥300 games, fixed seeds).
   Until that exists we don't actually know the current search helps.
3. **Search depth.** It's **1-ply** (branch each root action, evaluate the immediate result). For the
   delayed-value lines that motivated this (setup sequences, hazard payoff over many switch-ins),
   1-ply + a myopic value head can't see the payoff. Levers: **deeper leaf rollouts** (already
   supported via `leaf_rollout_*`) with a decent rollout policy, or extend to a **multi-ply tree**.
   Deciding 1-ply-with-rollouts vs a real tree is the core design question.
4. **Determinization *injection* for hidden-info ladder play.** The searcher never substitutes a
   hidden opponent set: replay-from-root re-runs the *real* recorded battle (perfect-info by
   construction in benchmarks), and leaf rollouts use the branch's real observations. The **sampler
   already exists** — `belief.py::sample_opponent_determinizations` emits concrete opponent
   realizations from the public belief view (invents no probabilities; unresolved stays unresolved),
   and the roadmap already calls it "bounded player-relative opponent determinizations for search."
   So the missing piece is **not** a sampler and **not** re-implementing MIT's randbats rejection
   sampling — it is the **injection seam**: materialize a sampled determinization into the
   branch/replay env so rollouts run against a concrete team, re-sampling **per rollout**. Prefer the
   **belief-based** determinizer (the project's stated, better-founded basis) over MIT's
   randbats-prior rejection sampling; note the divergence from the literal recipe and why. Required
   for the ladder; not needed for perfect-info benchmarking.

Only "not blocking" *once real multi-rollout averaging exists*: with today's single-pass `visits=1`
scorer there is **no averaging over multiple rollouts**, so chance handling (damage rolls) and
simultaneous-move handling are **not yet** adequately covered — each branch is scored against one RNG
draw and one opponent action (itself a strategy-fusion at the opponent node). These become real
design work the moment we move past the 1-ply single-pass scorer.

## Design principles / hard constraints

- **Showdown is ground truth**; poke-engine only after a Gen-3 equivalence spike
  ([`poke_engine_assessment.md`](poke_engine_assessment.md)).
- **foul-play is GPL** → benchmark / behavior source only; never imported.
- **Value quality bounds search quality** — WS-E is a prerequisite, not a nicety.
- **Benchmark search against an *independent* opponent.** If the net is *both* the search's opponent
  model and the eval opponent, net+search-vs-net-alone is **inflated** (the thesis explicitly flags
  this). The honest strength read is vs an independent opponent (**foul-play**); net-vs-net is
  diagnostic only.
- **North-star lanes.** Searchless self-play + **test-time** search = recipe-faithful **flagship**.
  fpdistill-seeded search and any **in-loop** search (AlphaZero-style) = **sanctioned parallel arm**,
  kept off the flagship so the clean baseline stays measurable ([`goals.md`](goals.md)).

## Component design (extending what exists)

- **Forking (WS-C) + compute budget (the real viability gate):** keep **replay-from-root** (built).
  MIT's budget is **1000–2000 rollouts/move at 10 s/move, and the env step — not GPU inference — is
  the bottleneck** (`mit_thesis_reference_config.md:78-80`). Replay-from-root re-simulates a full line
  per rollout; the roadmap's measured search throughput is only **~2–3 decisions/s**, which at
  MIT-scale rollout counts implies **minutes/move** — likely infeasible without snapshot/poke-engine
  or a much smaller rollout budget. **Design decision to make explicit:** pick a target
  rollouts-or-leaf-depth budget tied to a measured per-move cost (from `search_benchmark`'s
  `average_elapsed_seconds`), and treat forking cost as the gating constraint, not a checkbox.
- **Determinization:** wire the **existing** `belief.sample_opponent_determinizations` into the branch
  env and re-sample **per rollout** — the *new* work is the **injection seam** (materialize a sampled
  team into replay/branch state), not a sampler. See the theory note below.
- **Search core (WS-D):** two extension questions — (a) go from the 1-ply single-pass scorer to a real
  iterated loop (visit accumulation / a multi-ply tree), and (b) is **1-ply + deeper leaf rollouts**
  enough to recover delayed-value lines? Note the roadmap's own leaf-depth results are **non-monotonic**
  (leaf-2 sometimes worse than leaf-1), so "prototype rollout-depth first" is a real experiment, not a
  clean win.
- **Net integration:** leaf value = the (calibrated) value head via the pluggable `value_fn`. A prior
  **temperature** knob is desirable (soft prior so exploration can override a peaked prior) but is
  **not built** — the current path just renormalizes the prior over legal actions
  (`_normalized_legal_priors`); `cpuct` defaults to 1.25. Both are unbuilt/untuned levers.
- **Value head (WS-E):** measure calibration + ranking; improve targets / ranking loss / transform
  until search-ready. Verify the candidate net's head (incl. fpdistill's).

## Determinization: recipe fidelity + theory (PIMC vs ISMCTS)

**Recipe (MIT §3.2).** Determinize **per rollout**: sample one concrete possibility for all unknown
opponent info, restore the **Markov property** via multi-turn-effect duration encodings, and let the
net policy play the opponent inside the tree. MIT sampled from **Showdown's randbats generator +
rejection sampling** (≈10 attempts to match revealed traits, then force known ones). **We diverge
deliberately:** we sample from the **belief engine** (`sample_opponent_determinizations`) instead —
the project's stated, better-founded basis (it narrows the hidden set from observed facts rather than
rejection-sampling the raw prior). Per-rollout re-sampling averages over more worlds than a fixed set
of K root worlds.

**Precision (correcting an earlier overclaim):** per-rollout re-sampling **is still PIMC** — it
reduces per-world overfitting but does **not** build an information-set-keyed tree, so **strategy
fusion persists**. It is *not* "closer to ISMCTS" in the sense that matters. (And note: with today's
single-pass `visits=1` scorer there is no per-rollout loop yet, so per-rollout averaging is
aspirational until the iterated loop exists.)

**Theory / why this shape.** MIT's method is textbook **PIMC** (Perfect-Information Monte Carlo):
determinize → search the perfect-information game → average. PIMC works well in practice (bridge,
Skat) despite two known flaws:
- **Strategy fusion** — the search implicitly "cheats" by choosing *different* actions in worlds it
  actually cannot distinguish (they share an information set), because it solves each determinized
  world independently.
- **Non-locality** — a node's true value depends on the opponent's beliefs/strategy elsewhere in the
  tree, which determinization ignores.

The principled alternative is **Information-Set MCTS** (Cowling, Powley & Whitehouse, 2012): keep one
tree keyed on *information sets* and re-determinize per iteration, so the policy can't fuse strategies
across indistinguishable worlds. **Plan:** ship **PIMC with per-rollout averaging first**
(recipe-faithful, simple, proven), and escalate to **ISMCTS only if determinization artifacts
measurably bite**.

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

- **E1 first — value-head search-readiness (WS-E, gap #1).** Measure leaf-value **ranking** (Pearson)
  + calibration (ECE) on held-out data for self-play-1.5M and fpdistill; the tooling already supports
  `--min-pearson-correlation` / `--max-expected-calibration-error` gates. Set a concrete bar
  (e.g. Pearson ≥ ~0.3) before trusting a head as a leaf evaluator. **Honesty note:** the roadmap
  records current independent Pearson ~0.12 — likely **not search-ready** — so E0 below is a *plumbing*
  run, not a strength read, until a head clears the bar. This gates everything.
- **E0 — search strength benchmark (harness built; result pending).** The full-game external-opponent
  harness is now `scripts/root_puct_vs_foulplay.py`: foul-play stays out-of-process over a websocket,
  while PokeZero owns a seeded BattleStream and can build the context required by
  `select_action_with_context`. **Do not use the existing `search_benchmark` counterfactual mode for
  strength** — it replays branches against the *recorded* opponent action (`search_benchmark.py:345`),
  which leaks the opponent's real move (oracle info). Headline metric remains search-agent vs
  **foul-play (independent opponent)**; raw checkpoint vs foul-play is the comparison. Fix
  games/seeds/variance up front (≥300 games; the roadmap has been burned by 8–16-game reads). Re-run
  `hazard_probe`/`behavior_probe` on the search-augmented policy to see if search now sets Spikes /
  uses setup. Fills gap #2 and tests the whole bet.
- **E2 — depth (gap #3).** Sweep leaf-rollout depth / policy; if 1-ply+rollouts underperforms on deep
  lines, prototype a multi-ply tree.
- **E3 — determinization (gap #4).** Per-rollout opponent-set sampling (rejection-sampled randbats,
  force known traits) for hidden-info play; needed before ladder. ISMCTS only if PIMC artifacts bite.
- **E4 — (optional parallel arm) expert-iteration loop.** Only if E0 shows a large search win and the
  forking budget supports in-loop generation.

**Gate:** net+search beats net-alone by a clear margin (fixed yardstick + head-to-head). E0 is the
first time we'd actually measure this.

## Files (mostly *extend*, not new)

- `search.py` — from the single-pass `visits=1` scorer toward an iterated loop / deeper rollouts.
- `search_policy.py` — fpdistill prior+value into `select_action_with_context`; determinization-
  injection hook; a prior-**temperature** knob (currently absent).
- `foulplay_bridge.py`, `scripts/root_puct_vs_foulplay.py` — controlled full-game head-to-head strength
  mode vs foul-play. The existing `search_benchmark.py` counterfactual mode replays branches against
  the *recorded* opponent action (`:345`) → don't use it for strength.
- `belief.py` — **already has** `sample_opponent_determinizations`; no new sampler. The seam is the
  branch/replay env accepting an injected determinization (likely `local_showdown.py` / `replay_branching.py`).
- `value_calibration.py` / `neural_policy.py` — measure the search-readiness gate; apply a transform.
- `poke_engine_backend.py` — only if adopted as a fast backend (post equivalence spike).

## Validation & failure modes (was missing)

- **Determinization consistency test:** sampled opponent teams must be consistent with all revealed
  facts (never contradict a shown move/item/ability). Assert this before trusting rollouts.
- **Benchmark variance/seed control:** fix seeds + ≥300 games; report CIs. Do not read strength off
  small (8–16-game) samples.
- **Anti-leakage:** the strength harness must not feed the searcher the opponent's real action or set.
- **Fallback behavior:** `RootPUCTSearchPolicy` can fall back to a base policy; `allow_fallback`
  defaults to raising. Decide + log behavior on search failure so a "fallback storm" can't masquerade
  as search strength.

## Open questions / risks

1. **Value-head search-readiness** — the single biggest risk and gate (E1); current heads look *not*
   ready (Pearson ~0.12).
2. **Forking cost vs budget** — replay-from-root at MIT rollout counts may be minutes/move; this is the
   viability gate, not a checkbox.
3. **1-ply single-pass → iterated loop / tree** — needed for genuine multi-rollout averaging (chance,
   simultaneous moves) and for depth on delayed-value lines; leaf-depth results are non-monotonic.
4. **Determinization injection seam** — wiring the existing belief sampler into the branch env; PIMC
   strategy-fusion persists (ISMCTS only if it bites).
5. **fpdistill prior over-narrowness** — needs a temperature knob that isn't built yet.
6. **poke-engine Gen-3 equivalence** — unproven; blocks its use as a fast backend.
7. **In-loop compute** for expert iteration — likely too slow at scale; keep E4 gated on E0.

## Next step

**E1 gates E0.** First measure leaf-value ranking/calibration on self-play-1.5M + fpdistill against a
concrete bar; if a head clears it, wire fpdistill into `select_action_with_context` and build a **new
full-game head-to-head harness** (vs foul-play headline; vs net-alone/fpdistill-alone diagnostic;
≥300 games, fixed seeds) — **not** the existing counterfactual benchmark, which leaks the recorded
opponent action. This reuses most existing machinery (forking, scorer, calibration tooling), needs the
new strength harness + fpdistill wiring, and is independent of the runs currently training — the
fastest honest read on whether search is viable. If no head is search-ready, E1's outcome is itself
the finding: value work (WS-E) precedes any strength claim.
