# Engine-MCTS POC — results, tradeoffs, and limitations

Status: 2026-07-18. POC for the v3 plan's speed target: MCTS at FoulPlay-class
throughput, built from merged tracks A (world constructor) and C (fidelity
gate + residual-order patch). `pokezero.engine_search` runs poke-engine's
native multi-ply MCTS over K belief-sampled determinized worlds per decision
— FoulPlay's architecture on pokezero's belief engine — as a
`ContextAwarePolicy` playable through the standard rollout driver.

Repro:

```bash
python -m pokezero.engine_search --showdown-root <showdown> \
  --games 10 --opponent simple-legal --out bench.json
```

## Speed result (10 games vs simple-legal, M-series laptop, 1 thread)

| Metric | POC | FoulPlay (its default op point) | Current root-PUCT (Tier 1) |
|---|---|---|---|
| Simulations per searched decision | **~475,000** | ~190,000 | ~129 visits |
| Search wall per searched decision | **0.44s** | ~0.2s | ~8s |
| Simulations/sec (single thread) | ~1.08M | ~0.95M | ~16 visits/sec |

The speed goal is met: per-world throughput matches FoulPlay's engine
(same engine, same regime), total per-decision simulation volume is ~2.5×
FoulPlay's, and the wall clock is ~18× faster than the current Tier-1
root-PUCT at ~3,700× the simulation count. `threads=4` adds another ~1.8×
if wanted. Win rate in the bench: 16/20 vs simple-legal across the two runs
(non-binding smoke, not a strength claim).

## Tradeoffs and limitations (the honest half of the POC)

1. **No learned model in the loop — speed ≠ strength.** Leaves are priced by
   poke-engine's handcrafted evaluation; the tree has no policy priors. Our
   paired +10pt vs FoulPlay came from the learned value function at 129
   visits. This POC is FoulPlay's brain at FoulPlay's speed, reached through
   our belief engine — it is NOT expected to beat the raw checkpoint. The
   16/20 smoke vs simple-legal (the raw 1M checkpoint scores ~99%) makes the
   point concretely. Track B (the bit-identical encoder) is what puts our
   model onto this fast path; until then speed and strength live on
   different branches.
2. **Fallback rate ~55% of decisions, dominated by belief-sampler
   dead-ends, not engine construction.** With per-reason telemetry and
   retries (4× budget), the residual failures are *deterministic*: specific
   revealed opponents (e.g. a Snorlax/Blastoise whose revealed move
   combination matches no catalog set under the current constraints) burn
   every retry. This is the same family as W1's `missing_sampled_world`
   wall, lives in belief/determinization upstream of BOTH search stacks, and
   is now precisely attributed per species. Fixing the sampler (reserve
   candidates on this path, constraint relaxation) is the single highest-
   leverage strength lever for any determinized search, engine-backed or
   not.
3. **Sleep turn counts are approximated (opt-in, on by default here).**
   Public state does not track sleep/rest counters, so publicly-asleep mons
   are modeled as freshly asleep (`approximate_sleep_turns`). Without it,
   sleep alone fail-closes ~60% of decisions; with it, wake-up odds are
   biased late in a sleep. The exact fix is public sleep-counter tracking in
   the replay state — plumbing, not research.
4. **Force-switch boundaries never search** (`boundary_not_move_request`,
   ~10% of world failures): the constructor only expresses move-request
   boundaries. poke-engine has a `force_switch` flag; wiring it is
   mechanical but unvalidated, so it fails closed today.
5. **Remaining construction fail-closed classes:** substitute/confusion
   volatiles (no public sub-health/duration bookkeeping), pending
   Wish/Baton Pass, occasional `hidden_power_iv_mismatch` samples. Each is
   small (<7% of world attempts) and individually addressable.
6. **Uniform world weights.** FoulPlay weights determinizations by sample
   likelihood; our planner does not expose one. Aggregation quality loses a
   little sharpness in high-uncertainty positions.
7. **Fidelity scope inherited from track C:** 15/15 curated one-turn
   mechanics clean on the patched engine build, but multi-turn mechanics,
   rare branches, and screen durations are not yet differentially validated
   (tier-2 prerequisites in `engine_fidelity_findings.md`). Search results
   over long horizons lean on the engine's untested tail.
8. **Aggregation semantics differ from root-PUCT.** Visit-weighted argmax
   across worlds (FoulPlay-style), not PUCT over a shared root with priors.
   When the learned model lands (track B), the aggregation design should be
   revisited rather than inherited.

## What this changes in the plan

- The speed question is settled in the strongest form: the engine stack
  sustains ~10⁶ simulations/sec through our belief engine and world
  constructor end to end, inside the standard rollout harness.
- The critical path to *strength* at this speed is unchanged — track B —
  plus one new, sharply-scoped lever surfaced by the POC telemetry: the
  deterministic belief-sampler dead-ends (item 2), which cap ANY
  determinized search at ~45% of decisions today.
