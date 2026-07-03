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
| Forking = **replay-from-root plus accepted-prefix snapshot restore** | **built, still materialization-limited** | `replay_branching.py`, `search.py`, `local_showdown.py`, `scripts/battle_bridge.mjs` — search can still replay a sampled battle prefix from root, but `LocalShowdownEnv` now exposes a Showdown `Battle.toJSON()` / `Battle.fromJSON()` snapshot seam. When available, `value_branch_search` replays/materializes the prefix once, snapshots the accepted branch-point state, and restores it before each candidate branch/repeated root visit. This reduces repeated prefix replay cost for accepted worlds; it does **not** make live hidden-state snapshots acceptable for hidden-info search, because those contain oracle private state. |
| Root PUCT with optional visit/time budget (value-head leaf eval, optional leaf rollouts) | **built, but root-only** | `search.py::puct_branch_search` supports an optional root visit budget and wall-clock budget: legal root actions are evaluated once, then PUCT selection/backup accumulates additional root visits until the visit cap or time budget is exhausted. The mandatory first legal-action sweep is always completed and can exceed the configured time budget; the budget suppresses additional post-sweep visits once exhausted. Multi-scenario opponent planning uses a per-scenario visit budget, while time-budgeted searches receive the remaining decision budget when each scenario is searched. Repeated leaf rollouts use visit-specific rollout seeds, and the controlled foul-play harness can use sampled checkpoint policies inside leaf rollouts. The policy adapter now defaults to a **16-visit root budget** and final root action selection by **most visits**, with equal-visit ties resolved by policy prior before value; final `Q+U` PUCT-score selection remains available only as a diagnostic mode because the exploration bonus is for traversal, not deployment move choice. This is still **not a multi-ply tree**: selection/backup happens at the root only, and each visit replays/evaluates a root branch leaf. |
| Opponent modeling (greedy / top-k prior / policy planners, weighted scenarios) | **built** | `search_policy.py` |
| Net+search **Policy adapter** | **built** | `RootPUCTSearchPolicy` via **`select_action_with_context`**; plain `select_action` only runs the *fallback* (no context → no search). |
| Controlled foul-play **strength harness** | **built, smoke-verified** | `foulplay_bridge.py`, `scripts/root_puct_vs_foulplay.py`, `scripts/compare_root_puct_vs_foulplay.py` — runs foul-play as a **separate process** over a fake Showdown websocket while PokeZero owns a seeded BattleStream, so root-PUCT gets the replay seed + trajectory context it needs. Default mode withholds the opponent's private legal-action mask; `--opponent-legal-mask-mode privileged` is diagnostic-only. Full-game hidden-mode smokes now report root searches, total visits, fallbacks, fallback reasons/categories, opponent-action skip categories, replay-illegal opponent-scenario skip counts, unsearched reserve scenario counts, how often the selected root-PUCT action differs from the checkpoint prior's greedy legal action, and per-decision details for those changes. The controlled harness now generates reserve opponent-action candidates by default (`--root-opponent-action-candidate-scenarios 9`) and stops after the configured accepted scenario count (`--root-opponent-action-scenarios`, default 1), reducing hidden-mode fallback storms without always averaging every reserve candidate. When the opponent legal mask is hidden, unrevealed switch slots are treated as exchangeable: move candidates remain distinct, but switch-slot priors are collapsed into one summed switch candidate with a concrete representative action for replay. The paired comparison wrapper defaults to `--comparison-mode per-seed`: for each seed it runs raw, then root-PUCT, restarting foul-play with a matching per-seed startup seed before advancing. It writes one combined summary, matches completed games by seed, reports marginal per-arm Wilson 95% intervals plus discordant same-seed counts, and marks reads below 300 paired games as diagnostic-only. `--comparison-mode per-arm` preserves the older raw-all-then-search-all order when process startup overhead matters more than useful partial progress. When `--summary-out` is supplied, the harness writes partial progress after every completed game so slow MCTS reads are inspectable instead of all-or-nothing; partial `win_rate` is a completed-prefix read, not a complete benchmark result. The harness seeds foul-play's Python random/hash startup state (default: `--seed-start`) and records the seed in summaries, but foul-play still uses an unseeded, time-budgeted, multi-process/threaded poke-engine MCTS, so this is **not** a deterministic opponent or a perfect per-game paired counterfactual. The comparison artifact records that the opponent is not deterministic and the win-rate delta is descriptive rather than a paired statistical test. Replay-illegal skips are replay-legality probes against the real branch state, so they improve harness robustness but are still **not oracle-free hidden-info strength evidence**. |
| Search **behavior benchmark** (action-change rate, candidate count, per-move cost) | **built** | `search_benchmark.py` — **behavior/cost only, no win rate**; and the counterfactual harness replays branches against the **recorded** opponent action (`search_benchmark.py:345`) → oracle leakage (see E0). |
| Value-**calibration** tooling (ECE, affine/isotonic fit + transform) | **built** | `value_calibration.py`, `neural_policy.py` |
| **Belief determinizer / start overrides** | **partial, replay-brittle** | `belief.py` emits concrete opponent realizations from the belief view, and replay/search can now accept a `BattleStartOverride` or callable start-override source. Overrides are deliberately restricted to complete `p1`/`p2` packed teams in `gen3customgame`, because arbitrary teams are not honored by `gen3randombattle`; generic replay audits can still check the searched player's pre-action observation features before recorded prefix actions are submitted, excluding instance metadata and opponent-private POV. Root search now validates the sampled world at the branch-point observation instead of every intermediate prefix observation, and reports the first mismatching observation field when sampled worlds drift. The Gen 3 randbat start-override planner can now materialize our request-known team plus sampled public-belief opponent teams, filter only request-known absolute HP-compatible variants, use Gen 3 source metadata, constrain already-public opponent moves into harness-recorded replay slots without reading opponent-private request moves, and retry sampled worlds per opponent-action scenario. Root-PUCT can also split each opponent-action scenario across multiple sampled belief start overrides (`--belief-start-override-samples` in the foul-play harness), so one opponent switch action can be scored against multiple plausible hidden backline worlds instead of a single sampled team. Latest corrected hidden-mode smoke still falls back often on deeper histories, so this is not a strength path yet. |
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
3. **Search depth.** Current search has root visit accumulation, but it is still **root-only**: branch
   each root action, evaluate the immediate/leaf-rolled result, back up to root visits. For the
   delayed-value lines that motivated this (setup sequences, hazard payoff over many switch-ins),
   root-only search + a myopic value head may still miss the payoff. Levers: **deeper leaf rollouts**
   (already supported via `leaf_rollout_*`) with a decent rollout policy, or extend to a **multi-ply
   tree**. Deciding root-only-with-rollouts vs a real tree remains the core design question.
4. **Current-state materialization for hidden-info ladder play.** The searcher has the first
   payload/replay seam and a Gen 3 randbat belief-to-packed-team planner, but replay-from-root is
   still too brittle after public history has accumulated. Replay/search can pass a
   `BattleStartOverride` into `LocalShowdownEnv`, so branches can start from sampled packed teams
   instead of the default random battle root. Root-PUCT can now expand each accepted opponent-action
   scenario into multiple weighted belief-world samples before aggregation; in the foul-play harness,
   `--belief-start-override-samples N` keeps the opponent-action cap fixed and searches up to `N`
   sampled worlds inside each accepted opponent action before advancing to the next reserve action.
   This is still PIMC-style averaging, not an information-set tree, but it makes hidden
   backline/switch-in uncertainty observable to search instead of locking each action scenario to one
   sampled team. This seam is intentionally strict: arbitrary packed
   teams are only materialized through `gen3customgame`, both players' teams must be supplied, and
   strict replay audits can check the searched player's prefix observation features so a sampled
   world that no longer reproduces that player's recorded prefix fails loudly. Root search uses the
   less brittle current-state contract instead: after replaying the recorded prefix, it requires the
   searched player's branch-point observation to match before scoring any root action. Both paths
   deliberately exclude instance metadata and the opponent's private POV, and mismatch reports now
   include the first differing observation field/token. If the belief planner cannot provide a
   sampled world, or if a provided sampled-world source materializes to nothing, that sample is
   skipped rather than searched against the seeded default randbat world, because that default replay
   can reconstruct the opponent's actual hidden team from the live battle seed. The existing planner now materializes our
   known team from the root request snapshot, samples public-belief opponent variants, fills hidden
   backline candidates, applies only public/request-known filters, reorders only already-public
   opponent moves into harness-recorded replay slots, and can retry multiple sampled worlds per
   opponent-action scenario. The replay slot index comes from the
   controlled harness' recorded opponent action, while the move identity comes only from public event
   lines; this is a replay-fidelity mechanism, not a true ladder-time hidden-info signal. The
   remaining missing piece is proving this stays stable across many seeds and deeper histories, or
   adding a hidden-info-safe way to materialize the **current public battle state** without relying on
   a sampled full-team root replay to reproduce every prior damage roll, volatile, item interaction,
   and status transition. The first snapshot seam now exists for accepted prefixes: after a sampled
   world has already passed branch-point validation, search can restore that exact simulator state
   before each candidate branch instead of replaying the prefix again. That is a forking/cost fix,
   not a hidden-info materialization fix.
   Prefer the **belief-based** determinizer (the project's stated, better-founded basis) over MIT's
   randbats-prior rejection sampling; note the divergence from the literal recipe and why. Required
   for the ladder; controlled perfect-info benchmarks still need separate raw-vs-search strength
   reads.

Current root visit accumulation allows multiple root visits, and sampled leaf rollouts now get
visit-specific rollout seeds. Without determinization injection, however, repeated visits still run
through the same revealed branch state, so chance handling, simultaneous-move uncertainty, and
opponent hidden legal-action uncertainty are **not yet** adequately covered. Hidden-mode foul-play
smokes make this concrete: when the
opponent private legal mask is withheld, the opponent-action prior can still propose illegal opponent
actions. The searcher now skips replay-illegal opponent-action scenarios and reports skip counts, but
that skip is learned by probing replay legality against the real branch state. It is useful for
diagnosing and avoiding fallback storms, but it is **not** an oracle-free belief substitute. If every
scenario is illegal for a decision it must still fall back. Privileged legal-mask mode remains useful
as a diagnostic safety guard but not a headline hidden-info result.

Latest hidden-mode smoke against `foul-play` using the 1.5M checkpoint remains a plumbing diagnostic,
not strength evidence. A same-seed raw checkpoint read over seeds `941001..941005` won `0/5`. The
root-PUCT read on the same seed band, with visits selection, 16 root visits, root prior temperature
`4.0`, one opponent-action scenario, no leaf rollouts, and no score gate, also won `0/5`; it searched
131 decisions, changed the checkpoint prior action 21 times, and fell back 21 times because the
single hidden opponent-action scenario was replay-illegal. Adding a `0.0` score gate reduced the
selected changes to 14 and fallbacks to 11, but still won `0/5`. After adding reserve opponent-action
candidates with a one-scenario accepted cap, a two-game diagnostic over seeds `941001..941002` won
`0/2`, but eliminated fallbacks entirely (`0/76` searches) while skipping 3 replay-illegal reserve
candidates and leaving 198 reserve candidates unsearched after the cap was filled. Average
PokeZero-side search cost was `0.90s` per searched decision, much closer to single-scenario cost than
full four-scenario averaging. A follow-up full-reserve diagnostic over seeds `944001..944003`, with
`--root-opponent-action-candidate-scenarios 9`, `--root-opponent-action-scenarios 2`,
`--minimum-score-improvement 0.0`, and `--minimum-override-prior-ratio 0.75`, won `0/3`, eliminated
fallbacks (`0/163` searches), and reduced selected prior-action changes to 7, but raised average
search cost to `1.90s` per searched decision. That is useful coverage evidence, not a strength
signal: it shows the search can avoid the single-scenario fallback storm, but it does not yet show
net+MCTS beating the raw checkpoint. No >=300-game MCTS strength claim is meaningful until both
coverage and a plausible search-selection configuration are in place.

A one-game multi-belief start-override smoke over seed `952001` with
`--belief-start-overrides`, `--belief-start-override-samples 2`, and
`--start-override-attempts 2` completed end-to-end, but exposed the expected replay brittleness:
raw and root-PUCT both won `0/1`; root-PUCT searched 12 decisions, used 15 start-override sources,
spent 379 override attempts, changed no selected prior actions, and fell back 11 times because
sampled worlds or sampled opponent-action scenarios failed replay validation. This proves the
multi-belief path is wired, not that it is strong enough for a headline read.

After tightening belief mode so a missing sampled world is skipped instead of searched against the
seeded default randbat world, the same one-game smoke remained diagnostic-only: raw and root-PUCT
both won `0/1`; root-PUCT searched 20 decisions, used 22 start-override sources, spent 391 override
attempts, changed the selected prior action once, and fell back 8 times. This is the current
hidden-info-safe behavior: replay brittleness remains visible rather than being hidden behind an
oracle default-world search.

After adding aggregate fallback/skip categories and collapsing hidden opponent switch-slot priors into
one exchangeable switch candidate, a same-seed `961001` diagnostic still had raw and root-PUCT both
win `0/1`; root-PUCT searched 2 decisions, fell back 42 times, used 32 total visits, and changed the
selected prior action 0 times. The change reduced hidden opponent scenario volume/override attempts
(`opponent_action_scenarios_generated=400`, `start_override_attempts_used=782`) while the dominant
skip categories remained replay materialization failures
(`replay_request_unexpected_player=323`, `start_override_observation_mismatch=66`). This is a
semantic and cost cleanup, not strength evidence.

The Gen 3 belief start-override planner is now marked scenario-independent: sampled hidden worlds are
validated once per decision and then crossed with opponent-action scenarios, rather than resampling a
new hidden world for every immediate opponent action. This keeps hidden-world uncertainty separate
from action uncertainty, adds explicit shared-sample diagnostics, and avoids reattempting the same bad
belief sample for each opponent-action branch. It does not remove the replay-from-root brittleness:
sampled worlds still have to reproduce the branch-point observation before they are searched.

On the same seed `961001` one-game diagnostic, shared belief-world samples kept raw and root-PUCT at
`0/1`, searched only 2 decisions, and changed the prior action 0 times. The useful change was cost and
diagnostic clarity: start-override attempts are now counted at the shared-sample prevalidation layer
instead of per opponent-action/sample pair. One rerun yielded `146` attempts, `74` shared samples
prevalidated, `3` accepted, and `71` rejected. Replay materialization remains the blocker
(`replay_request_unexpected_player=270`, `start_override_observation_mismatch=57` in that artifact);
these exact one-game counts are descriptive because foul-play is not fully deterministic.

The next replay-localization diagnostic keeps raw and root-PUCT at `0/1` on seed `961001`, with
root-PUCT searching only 2 decisions, falling back 42 times, and skipping 389 opponent-action
scenarios. It now records replay rejection rounds split by request-shape mismatch vs start-override
observation mismatch, plus the first mismatching observation path. In one run the first failing replay
rounds were often early or mid-prefix (`2`, `3`, and `12`), and the first observed mismatch paths
included opponent belief/item and HP features (`categorical_ids/opponent_pokemon[8][11]`,
`numeric_features/opponent_pokemon[8][0]`) plus our own active HP
(`numeric_features/self_pokemon[1][0]`). These are first-divergence diagnostics, not a full inventory
of every mismatching feature, but they are enough to rule out pure PUCT selection tuning as the next
fix. The next materialization fix should either narrow sampled worlds from public damage/effect
evidence or replace replay-from-root branch-point validation with a true current-state/snapshot
contract.

Showdown snapshot/restore is now exposed through the local bridge and used to fork accepted replay
prefixes inside branch search. This should reduce repeated replay cost for root candidate evaluation,
but it does not by itself resolve the diagnostic above: rejected sampled worlds still fail before
there is a valid prefix to snapshot.

Replay validation now allows a narrow, opt-in HP-fraction tolerance for sampled start overrides
(`--start-override-hp-fraction-tolerance`, default `0.02`). This is deliberately scoped to HP
fraction numeric cells on self/opponent Pokémon tokens only; request shape, legal masks,
action-candidate tokens, categorical state, status, and all non-HP numeric features still match
exactly. On a seed `961001` one-game diagnostic before the request-shape fields existed, that
tolerance kept raw and root-PUCT at `0/1` but improved hidden-world coverage: searches rose from
`2` to `7`, accepted shared samples from `3` to `9`, skipped opponent-action scenarios fell from
`429` to `391`, and fallbacks fell from `46` to `41`. This is useful coverage evidence, not
strength evidence. The mechanism covers small current-HP-fraction drift only; max-HP/stat drift and
switch action-candidate HP cells still reject sampled worlds. The remaining blocker is still replay
materialization: in that earlier artifact, request-shape divergence remained dominant
(`replay_request_unexpected_player=321`, `replay_request_missing_player=20`), while larger HP drift
and belief/item-token mismatches continued to reject sampled worlds.

Request-shape diagnostics now also report the missing/unexpected player side (`missing:p1`,
`unexpected:p2`, etc.) and the full shape (`requested:<players>|actions:<players>`). A newer
same-seed `961001` one-game artifact showed `unexpected:p2=388` and
`requested:p1|actions:p1,p2=388`, meaning sampled hidden-world replay reached positions where only
PokeZero was requested while the recorded prefix still expected both players to act.

Hidden-info-safe foul-play validation must pass `--belief-start-overrides`. Non-belief root search
uses default seeded randbat replay and can reconstruct the opponent's actual hidden team, so those
runs are useful only as oracle diagnostics and must not be reported as real net+MCTS strength.

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

- **Forking (WS-C) + compute budget (the real viability gate):** keep **replay-from-root** as the
  materialization path for now, but use **Showdown snapshot/restore** after an accepted prefix.
  MIT's budget is **1000–2000 rollouts/move at 10 s/move, and the env step — not GPU inference — is
  the bottleneck** (`mit_thesis_reference_config.md:78-80`). Replay-from-root re-simulates a full line
  per accepted sampled world, while candidate branches can now restore the accepted prefix snapshot
  instead of replaying the full prefix again. This is still likely infeasible at MIT-scale rollout
  counts without a stronger current-state materializer, poke-engine, or a much smaller rollout budget.
  **Design decision to make explicit:** pick a target
  rollouts-or-leaf-depth budget tied to a measured per-move cost (from `search_benchmark`'s
  `average_elapsed_seconds`), and treat forking cost as the gating constraint, not a checkbox.
- **Determinization:** finish wiring the **existing** `belief.sample_opponent_determinizations` into
  the branch env and re-sample **per branch replay/visit**. The generic **payload/replay seam** now
  exists (`BattleStartOverride`, replay/search `start_override`, callable start-override sources, and
  `RootPUCTSearchPolicy.start_override_planner`), but it only accepts complete two-sided
  `gen3customgame` packed-team materializations and verifies the searched player's replay-prefix
  observation features. The remaining work is the belief-world-to-packed-team materializer,
  especially fully hidden backline slots. See the theory note below.
- **Search core (WS-D):** two extension questions — (a) go from the 1-ply single-pass scorer to a real
  iterated loop (visit accumulation / a multi-ply tree), and (b) is **1-ply + deeper leaf rollouts**
  enough to recover delayed-value lines? Note the roadmap's own leaf-depth results are **non-monotonic**
  (leaf-2 sometimes worse than leaf-1), so "prototype rollout-depth first" is a real experiment, not a
  clean win.
- **Net integration:** leaf value = the (calibrated) value head via the pluggable `value_fn`. Root
  PUCT now has a root-only prior-**temperature** knob, exposed as `--root-prior-temperature` on the
  full-game neural/foul-play harnesses. When omitted it preserves old behavior by falling back to
  `--temperature`; when set it softens only root action traversal priors, not opponent-action priors
  or the raw fallback policy. The foul-play harness also supports reserve opponent-action candidates
  (`--root-opponent-action-candidate-scenarios`, defaulting to the full 9-action space) that are
  replay-tested until the accepted scenario cap is filled. `cpuct` defaults to 1.25. These are still
  tuning levers, not proven strength improvements.
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
- **E0 — search strength benchmark (harness built; result pending).** The preferred full-game
  external-opponent read is now `scripts/compare_root_puct_vs_foulplay.py`: it runs raw checkpoint and
  root-PUCT against foul-play over the same BattleStream seed band and emits a single paired summary
  with marginal per-arm Wilson intervals, discordant same-seed counts, machine-readable non-determinism
  caveats, and a diagnostic-only marker for reads below 300 paired games. Default `per-seed` ordering
  produces paired partial evidence after each raw/root-PUCT seed pair; `per-arm` ordering exists only
  as a throughput-oriented fallback. The lower-level
  single-policy harness remains `scripts/root_puct_vs_foulplay.py`; foul-play stays out-of-process over
  a websocket, while PokeZero owns a seeded BattleStream and can build the context required by
  `select_action_with_context`. **Do not use the existing `search_benchmark` counterfactual mode for
  strength** — it replays branches against the *recorded* opponent action (`search_benchmark.py:345`),
  which leaks the opponent's real move (oracle info). Also keep the controlled bridge in its default
  hidden-info mode for headline reads: `--opponent-legal-mask-mode privileged` feeds the search the
  opponent's true legal-action mask, which is useful as a diagnostic safety guard but is still hidden
  information. Headline metric remains search-agent vs **foul-play (independent opponent)**; raw
  checkpoint vs foul-play is the comparison. Fix games/seeds/variance up front (≥300 games; the
  roadmap has been burned by 8–16-game reads) and log foul-play's `--search-time-ms` because compute
  asymmetry can inflate a search result. Re-run `hazard_probe`/`behavior_probe` on the
  search-augmented policy to see if search now sets Spikes / uses setup. Fills gap #2 and tests the
  whole bet.
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
- `search_policy.py` — fpdistill prior+value into `select_action_with_context`; determinization
  planner hook; multi-sample start-override expansion for belief-world averaging; a root
  prior-**temperature** knob for decoupling traversal softness from raw-policy and opponent-action
  temperatures.
- `foulplay_bridge.py`, `scripts/root_puct_vs_foulplay.py`, `scripts/compare_root_puct_vs_foulplay.py`
  — controlled full-game head-to-head strength mode vs foul-play. Use the comparison wrapper for raw
  vs root-PUCT reads so tiny samples are explicitly labeled diagnostic and larger runs share a seed
  band. The existing `search_benchmark.py` counterfactual mode replays branches against the
  *recorded* opponent action (`:345`) → don't use it for strength.
- `env.py`, `local_showdown.py`, `replay_branching.py`, `search.py` — `BattleStartOverride` path for
  branch/replay env start-state overrides, restricted to complete two-sided `gen3customgame` packed
  teams with searched-player replay-prefix feature checks.
- `belief.py` — **already has** `sample_opponent_determinizations`; the remaining seam is turning
  sampled belief worlds into packed teams that can be passed through `BattleStartOverride`.
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
4. **Belief materialization seam** — the custom-game branch start override exists, but the existing
   belief sampler still needs conversion into complete packed teams, including our known team and
   plausible unrevealed opponent backline Pokémon; PIMC strategy-fusion persists (ISMCTS only if it
   bites).
5. **fpdistill prior over-narrowness** — root prior temperature is now built; it still needs a sweep
   against foul-play to determine whether softening the prior gives search enough room to improve on
   the raw checkpoint.
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
