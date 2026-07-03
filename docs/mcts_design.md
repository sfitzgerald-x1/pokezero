# MCTS design: test-time search + fpdistill-seeded bootstrap

Status: **planning / draft.** Detailed design for the test-time MCTS policy-improvement operator and
a proposed search-teacher-seeded bootstrap track. This is the concrete design layer under the
workstream summaries in [`selfplay_mcts_roadmap.md`](selfplay_mcts_roadmap.md) (WS-C forking, WS-D
MCTS, WS-E value calibration). Read that first for the why; this doc is the how.

## Why now / motivating evidence

Measured on the current 1.5M checkpoints (see the hazard/behavior probes):
- The searchless self-play policy is **myopic on delayed/positional lines** — it never sets Spikes
  (0/243 argmax), is blind to the self-hazard feature (ΔP(Rapid Spin) ≈ 0 across 0→3 injected
  layers), and under-uses setup. This is a **credit-assignment** limit, not a capacity or coverage
  one — the value head can't credit payoffs that arrive many turns later, and self-play sits at a
  weak-signal fixed point.
- The **fpdistill** net (BC-distilled from foul-play + RL) is our strongest on foul-play (~17% hi-fi)
  and max-damage (~80%), i.e. it already encodes the deep lines the self-play net is blind to.

Search is the principled fix: MCTS computes the delayed payoff *at decision time* by simulating the
future, independent of whether the value head learned it. This doc designs that operator and a way
to seed it well.

## Design principles / hard constraints

- **Showdown is ground truth.** `poke-engine` may be adopted as a fast reversible backend only after
  a Gen-3 random-battle mechanics-equivalence spike passes ([`poke_engine_assessment.md`](poke_engine_assessment.md)).
- **foul-play is GPL** → external benchmark / behavior source only; never imported.
- **Value quality bounds search quality.** A noisy value head makes MCTS *worse* than the raw policy.
  WS-E (value calibration) is a hard prerequisite, not a nicety.
- **North-star lanes.** Searchless self-play from scratch + **test-time** MCTS is the recipe-faithful
  **flagship** (MIT). Anything that seeds search from an imitation teacher (fpdistill) or puts search
  **in the training loop** (AlphaZero-style) is a **sanctioned parallel arm**, kept off the flagship
  so the clean self-play baseline stays measurable. See [`goals.md`](goals.md).

## Architecture overview

```
                       ┌─────────────────────────────────────────┐
   battle position ──▶ │  belief determinizer (belief.py)         │  sample K opponent worlds
                       │  hidden set → (near) perfect-info state   │
                       └───────────────┬─────────────────────────┘
                                       ▼
   net (policy prior + value) ──▶ ┌──────────────┐   fork/rollout   ┌──────────────────┐
   neural_policy.py               │  PUCT search  │ ◀──────────────▶ │ forkable sim     │
                                  │  (search/mcts)│                  │ (Showdown / p-e) │
                                  └──────┬───────┘                   └──────────────────┘
                                         ▼
                        search-policy adapter (policy.py) ──▶ select_action(net+MCTS)
```

Five components, each mapping to a roadmap workstream: **forking backend** (WS-C), **determinizer**
(WS-E), **search core** (WS-D), **net integration** (WS-A/E), **value head** (WS-E).

## Component 1 — Forking / rollout backend (WS-C)

Search needs to explore alternative lines from a position. Options, cheapest-to-validate first:

1. **Replay-from-root (preferred default).** With determinization each rollout re-simulates from the
   recorded line + a sampled opponent set. Warm sim (~0.4 ms/turn) may make this fast enough for
   shallow search and avoids all state serialization. **Validate per-move cost against the search
   budget before anything else** — this is the highest-leverage de-risking step.
2. **Snapshot/restore.** Only if replay-from-root is too slow: `Battle.toJSON()` / restart-from-state
   at our pinned commit; extend `battle_bridge.mjs` with `snapshot`/`fork`; expose
   `LocalShowdownEnv.snapshot()/fork()`.
3. **poke-engine reversible apply/reverse.** MIT-licensed, exposes `apply_instructions` /
   `reverse_instructions` — the make/unmake primitive that makes deep search cheap. Gated on the
   Gen-3 equivalence spike; adopt as a *fast backend*, not a ground-truth replacement.

**Decision:** implement replay-from-root, measure, and only escalate to (2)/(3) if the budget demands.
Acceptance: divergent lines are byte-identical to from-scratch replays that took the same actions.

## Component 2 — Determinization (hidden info → perfect info)

Gen-3 randbats are imperfect-information (opponent set hidden). Determinize at the search root:
sample the opponent's hidden set from the belief engine (`belief.py` / `Gen3RandbatSource`, already
built and wired), search a (near) perfect-information instance, and **average over K sampled worlds**.

- **Root determinization** (sample K worlds, run an independent search in each, average action
  values) — simpler, embarrassingly parallel. **Start here.**
- **Per-simulation re-determinization** (information-set MCTS) — more faithful but complex; a later
  upgrade.
- Open knob: K (worlds) vs sims-per-world under a fixed budget.

## Component 3 — Search core (PUCT) (WS-D)

- **Node** = determinized state; **edges** = legal actions (reuse the action space + legal mask).
- **Selection:** PUCT — `argmax_a [ Q(s,a) + c_puct · P(s,a) · √ΣN / (1 + N(s,a)) ]`, prior `P` from
  the policy head, `Q` backed up from leaf **value-head** evaluations.
- **Simultaneous moves** (gen mons pick concurrently, then resolve by speed):
  1. **Opponent plays the net policy** (thesis approach) — simplest, start here.
  2. **DUCT** (decoupled UCB) — proper simultaneous-move handling; leave a hook.
  3. max^n / *-minimax — later.
- **Chance nodes** (damage rolls, accuracy, secondary effects): collapse the 16 damage rolls to
  **KO / no-KO branches** (foul-play's grouping) rather than expanding all rolls; accuracy and
  secondaries as chance outcomes. Optional best/worst/avg-case aggregation as a knob.
- **Backup:** average over children (expectation over chance); minimax-over-chance as an option.
- **Budget:** wall-clock or sim-count per move; integrate as an alternate `select_action` (net+MCTS).

## Component 4 — Net integration (prior + value)

- **Prior** = policy-head softmax, with a **temperature** so PUCT keeps exploring off-prior lines
  (a too-peaked prior starves search of good moves the net under-rates).
- **Leaf value** = value head (calibrated — see Component 5).
- **Which net drives search?** This is the key experimental fork:
  - **self-play net** — recipe-faithful flagship, but a *myopic prior* on deep lines (search must
    discover hazards/setup from scratch via the exploration term).
  - **fpdistill net** — a prior that *already knows* the foul-play lines → search explores the correct
    lines immediately. This is the seeded-bootstrap track below.

## Component 5 — Value head: the hard prerequisite (WS-E)

MCTS leaf selection depends on value **ordering** more than absolute calibration; both matter.

- Measure **calibration** (predicted vs realized outcome) on held-out data; compare raw / affine /
  isotonic transforms (isotonic is a calibration lever, not a fix for bad ranking).
- Model-side levers: improved value targets (terminal / discounted / capped-game), the opt-in
  **pairwise value-ranking loss**, the recurrent temporal aggregator, optional clipped shaped-return
  targets (ablate before adopting).
- **Verify the candidate net's value head is search-ready before trusting it as a leaf evaluator** —
  including fpdistill's RL value head (fits outcomes on the common distribution ≠ accurate on the deep
  lines search will probe).

## The fpdistill-seeded bootstrap / expert-iteration track (parallel arm)

The idea (from the "distill an MCTS agent, then use it *for* MCTS" discussion): use fpdistill — itself
a BC distillation of the foul-play MCTS agent — as the **prior + value** of the search. This is
expert iteration warm-started from a search teacher, and it should explore the correct lines far
faster than a myopic self-play prior.

**Why it should work:** MCTS quality is dominated by prior + value. A prior that concentrates
simulations on plausible lines (hazards when appropriate, setup, phazing) recovers ceiling that both
(a) BC threw away — it copies the teacher's *move*, not its *search* — and (b) the self-play net never
had. Expected ordering: **MCTS(fpdistill) > fpdistill-alone > MCTS(self-play net)** on foul-play.

**Two modes:**
1. **Test-time only** — MCTS(fpdistill) at inference. No training change. Cheapest; the first
   experiment (E0 below).
2. **Expert iteration** — `MCTS(net) → generate data → re-distill/train → repeat`. Seeding the first
   net from foul-play-BC skips AlphaZero's painful weak-cold-start iterations. Powerful but this is
   **in-loop search** (the expensive paradigm the MIT recipe deliberately avoids) → strictly a
   parallel arm, and it must be measured against the pure self-play flagship.

**Caveats:** value-head accuracy is still the linchpin; keep the prior temperature soft so PUCT can
override the teacher on off-distribution states; and this is imitation-seeded — the roadmap's
north-star note cautions against treating foul-play as a teacher to clone, so it lives beside, not
inside, the from-scratch line.

## Experiment sequence / milestones

- **E0 — MCTS(fpdistill), test-time, no training (highest leverage, cheapest).** Wire the existing
  fpdistill net as prior+value into the search; benchmark vs foul-play, vs fpdistill-alone, and vs
  MCTS(self-play net). Also probe whether search now sets Spikes / uses setup / spins correctly
  (reuse `hazard_probe`/`behavior_probe` on the search-augmented policy). Confirms the core thesis in
  one shot and tells us whether search alone repairs the blind spots.
- **E1 — Forking backend validation (WS-C).** Replay-from-root per-move cost vs a realistic search
  budget; equivalence tests. Escalate to snapshot / poke-engine only if needed.
- **E2 — Value-head search-readiness (WS-E).** Calibration + ranking audit on the candidate nets
  (self-play 1.5M, fpdistill); apply calibration transform if it helps.
- **E3 — Search tuning.** K determinization worlds × sims/world; damage-roll grouping; opponent model
  (net-policy → DUCT); c_puct + prior temperature.
- **E4 — (optional, parallel arm) expert-iteration loop.** Only if E0 shows a large search win and the
  forking budget supports in-loop generation.

**Gate (roadmap acceptance):** net+MCTS beats net-alone by a clear margin on the fixed yardstick and
head-to-head. If E0 with fpdistill clears it, that's the go signal for the search track.

## Interfaces / files touched

- **new** `search/mcts.py` — PUCT core, chance grouping, determinization loop.
- `local_showdown.py` — fork/replay API (WS-C).
- `belief.py` — opponent-set determinization sampler (mostly exists).
- `neural_policy.py` — expose policy-prior + value access for leaf eval.
- `policy.py` — a search-policy adapter (`select_action` = net+MCTS).
- `engine_cli.py` / optional `poke-engine` path — only if adopted as a fast backend.

## Open questions / risks

1. **Forking cost.** Is replay-from-root fast enough for a useful per-move sim budget, or do we need
   snapshot/poke-engine? (E1 answers; gates everything.)
2. **Value-head search-readiness.** The single biggest risk — a miscalibrated/ mis-ordered value head
   makes search worse than the net. (E2.)
3. **Simultaneous moves.** How much does "opponent plays net policy" bias search vs DUCT? Start simple,
   measure.
4. **fpdistill prior over-narrowness.** A peaked BC prior may starve exploration of good off-teacher
   lines; tune temperature.
5. **poke-engine Gen-3 equivalence.** Unproven; blocks its use as a fast backend until the spike passes.
6. **In-loop compute.** Expert iteration needs search-in-generation; likely too slow at scale (the
   reason MIT keeps search at test time). Keep E4 gated on E0's payoff.

## Next step

Scope E0 concretely: what minimal `search/mcts.py` + fork API + net-prior/value wiring is needed to
run MCTS(fpdistill) vs foul-play on a handful of games. This is independent of the runs currently
training and is the fastest way to validate the whole search bet.
