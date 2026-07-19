# Engine-MCTS POC — results, tradeoffs, and limitations

Status: 2026-07-18, revised after independent review (which falsified the
first draft's fallback attribution — see "Fallback anatomy") and extended
with belief-alignment fixes and the model-cost ladder.

POC for the v3 plan's speed target: MCTS at FoulPlay-class throughput,
built from merged tracks A (world constructor) and C (fidelity gate +
residual-order patch). `pokezero.engine_search` runs poke-engine's native
multi-ply MCTS over K belief-sampled determinized worlds per decision —
FoulPlay's architecture on pokezero's belief engine — as a
`ContextAwarePolicy` playable through the standard rollout driver.

Repro:

```bash
python -m pokezero.engine_search --showdown-root <showdown> \
  --games 10 --opponent simple-legal --out bench.json
```

## Speed result (10 games vs simple-legal, M-series laptop, 1 thread)

| Metric | POC (no learned model) | FoulPlay (default op point) | Current root-PUCT |
|---|---|---|---|
| Simulations per searched decision | **~457,000** | ~190,000 | 129 visits |
| Search wall per searched decision | **0.44s** | ~0.2s | ~8s CPU (leaf-calibrated read); **~37s** on the newer instrumented CPU read, ~34s of it transformer forwards |
| Simulations/sec (single thread) | ~1.04M | ~0.95M | ~16 visits/sec |

**Read the table honestly:** the POC column contains no learned-model
forwards; the root-PUCT column's wall is *dominated* by them (per the v2
plan's same-day instrumented read). The apples-to-apples statement is
narrower and still strong: the simulation substrate now sustains
FoulPlay-class volume (~2.5× its per-decision sims at `worlds=4` — a config
choice, not an efficiency claim) through our belief engine end to end, and
the per-world engine throughput matches FoulPlay's because it is the same
engine. What the learned model costs on top is quantified below. `threads=4`
multiplies engine throughput ~1.8× (`total_visits` verified to aggregate
across threads). Win rate 24/30 vs simple-legal across three bench runs
(non-binding smoke; the raw checkpoint scores ~99% — see tradeoff 1).

## The model-cost ladder (10.2M-param v2.2 checkpoint, measured)

"How much does our model slow the loop": measured with the real 1M-games
checkpoint (10.2M params, d=512) doing forward passes at v2.2 shapes
(151×51 categorical + 151×155 numeric), dummy-filled (timing is
encoding-independent):

| Rung | Throughput |
|---|---|
| Native Rust MCTS (handcrafted eval, in-engine) | ~580k–1.08M sims/s |
| Python-driven loop over engine primitives (FFI, no NN) | ~33–46k sims/s |
| Loop + model pricing every leaf, CPU batch=1 | **168 evals/s** |
| Loop + model, CPU batch=64 (saturated) | **309 evals/s** |
| Model alone, Apple MPS batch=64 | ~1,700 evals/s |

The model does not slow the loop — **it becomes the loop**: at CPU batch-64
the forward passes are ~99.7% of the budget, a ~1,900× per-simulation
premium over the handcrafted eval (3.2ms vs 1.7µs batched). Consequences:

- A model-priced engine search on this laptop gets ~130 evals into 0.44s —
  the same visit count as today's root-PUCT, ~20× cheaper on wall. On MPS
  ~750; on a cluster GPU with real batching, thousands. The engine swap's
  win for *model-priced* search is the 10–100× from batching+GPU plus the
  elimination of sim/bridge overhead — not the raw million-sims regime,
  which only the handcrafted eval can afford.
- The design space this opens: hybrid pricing (model at the root/shallow
  nodes, handcrafted eval deep), model-guided priors over engine rollouts,
  or straight batched-leaf PUCT at ~10³–10⁴ visits — all gated on track B's
  encoder for correctness, none on more speed work.

## Fallback anatomy (fifth revision: 47% -> 0.8% -> 0.45% -> 0.0%)

> Revision note (2026-07-19, PR #737; attribution corrected same day): the
> 0.0% figure below is specific to the original bench seed set and was
> seed-lucky — fresh seeds (7000-7014) read 15.2%, identically under the
> HpFraction control. Actual wall composition from the bench logs: a genuine
> Trick swap (by-design fail-close, 48/60), request-state flags (7/60), and
> the unseeded flashfire volatile (5/60). No Transform or Knock-Off-removal
> wall occurred on these seeds. The trajectory below remains accurate for
> the walls it removed.

Same-seed bench trajectory: 55% -> 47% (alignment wave 1) -> **0.8%**
(235/237 decisions searched) after the dead-end hunt:

- **Stale-window slot pins**: rounds where the opponent produced no fresh
  public move (asleep, full-para) pinned an older move at the newly-chosen
  slot; Sleep-Talk-called moves pinned at the caller's slot. A move pinned
  at two slots excludes every candidate variant — permanent dead-end.
  Fixed: cross-slot contradiction sanitizer drops the unreliable move's
  pins entirely (`_sanitized_move_slot_constraints`).
- **Catalog-enumeration gaps** (e.g. Shedinja whose four publicly-witnessed
  moves matched no enumerated variant): fixed by the witnessed-set
  fallback — when catalog reconciliation fails, build the world from the
  publicly witnessed moves and fill from the unfiltered movepool. Witnessed
  facts are exact; only the fill is sampled; anti-leakage unchanged
  (independently verified). **Gated:** the fallback is opt-in
  (`witnessed_fallback=True`, engine search only) because the planner is
  shared with the production Node/root-PUCT stack — fills can produce
  role-inconsistent sets, so the production distribution stays unchanged
  until a paired A/B validates enabling it there.
- **Wish** now constructs exactly: the caster is the sampled world's unique
  Wish carrier (amount = maxhp/2, turns from the public set turn);
  ambiguous carriers fail closed.
- **Edge-case wave (2026-07-18, owner-directed): Transform detection +
  Encore support.** A transformed Ditto previously constructed as a
  SILENTLY WRONG world (base Ditto stats + [transform] moveset vs the real
  copied set) — no fallback fired; confirmed by live probe. Now: the
  belief engine's public transform tracking feeds `blocked_slots`
  (opponent side, `public_effect_blocked`), and a general self-moveset
  consistency guard (`self_moveset_mismatch`: every request-known move
  must exist in the sampled set) catches the self side plus any
  Mimic-class desync. Covered by a live end-to-end Ditto test. **Encore**
  (endgame-critical) now constructs instead of failing closed: the engine
  restricts the side to `last_used_move` under the ENCORE volatile
  (empirically pinned); the lock is modeled as horizon-long (the engine
  does not decrement encore duration — conservative vs gen3's 3-8 turns).
  Self side derives the locked move from request disabled flags; opponent
  side from the publicly-observed last move; ambiguity fails closed
  (`encore_move_unknown`). Bench: fallback 0.8% -> **0.45%** (223/224),
  the remaining decision being pending Baton Pass.
- **Baton Pass boundary (2026-07-18): the last recurring class, now modeled.**
  From our info set the opponent's committed-but-hidden action is exactly
  what determinization samples over: the passer side constructs with the
  engine's `baton_passing` + `force_switch` (recipient choice only, boosts
  pass), and the opponent side carries a per-world sampled commitment via the
  engine's saved-move field — which review probes show the gen3 build does
  NOT actually resolve after the pass (fail-soft optimistic under-model;
  field kept for forward compatibility). The fallback win is the boundary
  itself searching: recipient choice with boosts passing. Unsupported pending shapes
  (opponent-side pending) still fail closed. 15-game bench: **0.0% fallback
  (329/329 searched) on the original seed set** (seed-lucky — see the
  revision note above; fresh seeds read ~15% via the remaining walls);
  residue is per-attempt catalog two-HP-variant rejects that never cost a
  decision.
- Remaining (both principled fail-closes, ~1 decision each per 10 games):
  **encore** (meaningless without `last_used_move` wiring) and **pending
  Baton Pass** (needs deferred opponent-action semantics). Plus a rare
  per-attempt guard catch (`hidden_power_iv_mismatch`, 4/976 attempts,
  decisions still searched via other worlds) — root-caused by review: the
  randbats CATALOG itself enumerates 11 variants pairing two Hidden Power
  types (Unown bug+fighting, Forretress bug+steel); the spread fixes IVs
  for the first and the guard correctly rejects the second. Pre-existing
  catalog inconsistency, engine-path-only, fail-closed.

## Tradeoffs and limitations

1. **No learned model in the search loop — speed ≠ strength.** Leaves are
   priced by poke-engine's handcrafted evaluation; the tree has no policy
   priors. Our paired +10pt vs FoulPlay came from the learned value at 129
   visits. 24/30 vs simple-legal (raw checkpoint: ~99%) makes it concrete.
   Track B is the strength path; the model-cost ladder above prices it.
2. **Sleep approximation flattens Rest** (review finding): a publicly
   RESTED mon (deterministic 2-turn sleep in gen3) is modeled as fresh
   ordinary sleep (random 1–4 wake) because the public payload carries no
   rest/sleep counters. On by default in the POC config only; `engine_world`
   defaults stay strict. Exact fix: public sleep/rest counter tracking in
   the replay state.
3. **Substitute health is an upper bound** (fresh maxhp/4); public sub-hit
   tracking would tighten it.
4. **Uniform world weights** (no sample-likelihood weighting yet) and
   **FoulPlay-style visit aggregation** rather than shared-root PUCT —
   revisit both when the learned model lands.
5. **Decisions are not seed-reproducible**: the MCTS budget is wall-clock,
   so visit counts (and rare near-tie argmaxes) vary run to run. Matters
   only if RL data collection assumes determinism.
6. **Fidelity scope inherited from track C**: 15/15 curated one-turn
   mechanics on the patched build; multi-turn mechanics, rare branches, and
   screen durations lean on the engine's untested tail (tier-2
   prerequisites in `engine_fidelity_findings.md`).

## What this changes in the plan

- The speed question is settled: the engine stack sustains ~10⁶
  handcrafted-eval simulations/sec through our belief engine, world
  constructor, and rollout harness end to end.
- The model-cost ladder converts track B's payoff from a hope into a
  number: batched GPU leaf eval at ~10³–10⁴ evals/s is the regime where
  learned-value search inherits this speed.
- The isolated strength lever is the belief sampler's deterministic
  dead-ends (~95% of remaining fallbacks), upstream of both search stacks.
