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

## Fallback anatomy (revised — the review falsified the first draft)

Current bench (all alignment fixes in): **47% of decisions fall back**
(searched 198/376). Taxonomy after the fix wave:

- **Belief-sampler dead-ends: ~95% of remaining world failures.**
  Deterministic per position: a revealed opponent (Regice, Blastoise,
  Kyogre, Shedinja in current seeds) whose revealed moves match no catalog
  set under current constraints burns every retry. Same family as W1's
  `missing_sampled_world`; upstream of BOTH search stacks; now the single
  isolated lever.
- Fixed this revision: **Unown letter formes** (review finding — cosmetic
  formes failed `species_unknown` and silently zeroed search for entire
  games; now collapsed to base species), **force-switch boundaries** (the
  constructor now sets the engine's `force_switch` flag; the whole decision
  class searches), **substitute** (fresh-sub maxhp/4 approximation, opt-in
  flag like sleep).
- Remaining small classes: pending Wish (needs a payload field for the
  wisher's identity), non-substitute volatiles (confusion durations), rare
  `hidden_power_iv_mismatch` samples.
- Caveat on the 47% headline: the denominator includes one-legal-action
  and trivial decisions where fallback costs nothing; the rate is not yet
  sliced by addressable harm.

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
