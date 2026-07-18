# Test-time search: working plan v3 — the engine swap

Status: 2026-07-18. Successor to `test_time_search_plan_v2.md`. v2 is left
untouched (it is actively read and written by in-flight agents finishing the
W5 Tier-1/Tier-2 validation recorded there); this doc is the fresh append
point and the working plan going forward. The measurement doctrine carries
forward unchanged: **paired seeds, both arms, the compare harness
(`scripts/compare_root_puct_vs_foulplay.py`), provenance logged but never
gating.**

## Where v2 ended (carried results)

- **Search works: paired +10.0 pts vs FoulPlay** (43.0% vs 33.0%, 200 seeds,
  McNemar p≈0.02) at root-PUCT-120 with the 1M v2.2 checkpoint and frozen
  isotonic leaf. Search never loses to its own prior.
- **W5 Tier 1 (snapshot-per-decision) landed and default-on**, probe-verified:
  `prefix_replay_count = 0`, world materialization ~5% of search wall, ~8s
  per decision at extra-120 (vs 19.2s baseline). Fallback trajectory:
  19.4% → ~15.3% after the W1 fix wave → 8.2% in the Tier-1 probe (small n).
- **W5 Tier 2 on the Node sim (direct materialization) is implemented**
  (`prepare_direct_materialization_prefix`, `materialize_public_world` —
  fail-closed on unsupported public effects) with validation in progress on
  the ~10-game telemetry probe. Whatever it delivers, it inherits the Node
  sim's per-step cost.
- **The remaining cost is not the sim mechanics but the loop around it:**
  ~72% of search wall is per-visit orchestration (bridge round-trips, Python
  loop). Instrumentation to attribute it landed; a guarded one-round-trip
  snapshot candidate is staged behind the probe's verdict.
- **Search cost is scale-independent** (~5–6.5 visits/sec at 50M and 200M on
  GPU) — the model is not the bottleneck at any scale we run.

## The ceiling, measured (2026-07-18 profile)

FoulPlay's engine (poke-engine, the Rust reimplementation of the showdown
engine, vendored via `third_party/foul-play`, gen3 feature build):

- **~0.86M MCTS iterations/sec midgame, ~1.2M/sec lategame, single thread**
  (~1.2µs per iteration all-in; sub-microsecond state transitions in-loop).
  FoulPlay's whole design rests on this: ~190k simulations per decision at
  its 100ms operating point — against our ~130 visits at ~8s.
- A single FFI-crossed call into that same engine costs ~20µs — 20–100× the
  in-loop per-node budget. Conclusion: no bridge- or IPC-based loop can ever
  approach this regime; the loop must be in-process with batched boundary
  crossings.
- `generate_instructions` returns exact chance distributions per joint
  action — strictly better chance-node handling than sampling outcomes
  through a stochastic sim.

**The plan: adopt poke-engine as the search branch simulator** and remove
the real sim from the search loop entirely. The real sim stays ground truth
everywhere else — training, evaluation games, the paired harness. Landing
zone: sim cost vanishes; the floor becomes batched NN leaf evaluation
(~10⁴+ evals/sec on GPU for the s-model) → **10³–10⁴ visits/sec vs today's
~16**. FoulPlay-class volume with a learned prior and value function; makes
multi-ply meaningful; extends the budget→value curve two decades right.

**Not chosen:** a Python engine reimplementation (~100µs–1ms/step — 10–100×
over the bridge but 100–1000× short of Rust, GIL-bound, and we own every
mechanics bug) and a from-scratch Rust engine (months to re-arrive at
poke-engine, which is battle-hardened by years of ladder play). poke-engine
is an approximation of the real engine — that risk is measured (track C
below), not inherited as an assumption.

## Contracts (freeze before any track forks)

1. **World contract.** Exactly which fields a belief-sampled world supplies
   to the poke-engine `State` constructor: teams, HP, statuses, boosts, PP,
   items/abilities as sampled, side conditions, field, trick-state flags.
   Worlds are constructed ONLY from public information plus belief-sampled
   hypotheses — never from the live battle's hidden state. The P-1
   anti-leakage checksum gate must pass unchanged.
2. **Encoder contract.** Input: (poke-engine state, instruction list since
   the last decision). Output: a v2.2 observation **bit-identical** to what
   the production protocol-stream encoder emits for the same position.
   Identity is defined by the golden corpus (below), not by code review.

## Tracks (parallel; only meet at integration)

| Track | Deliverable | Depends on | Owns |
|---|---|---|---|
| A | World constructor: belief world → poke-engine `State` (adapt foul-play's `battler_to_poke_engine_side` mapping to source from our belief engine) + anti-leakage tests | World contract | new module only |
| B | v2.2 encoder from engine state + instructions, developed against the golden corpus | Encoder contract | new module only |
| C | Fidelity differential harness: a few thousand (state, joint-action) cases stepped in both engines, outcome distributions compared | nothing | new script only |
| D | Batched NN leaf evaluation in the search loop (pays off on the Tier-1 path today; required to cash in visit volume later) | the v2 residual-probe verdict | `search.py` (sole owner) |

File-ownership rule: track D is the only track that touches `search.py`;
A/B/C land as new modules with their own tests. This is what makes
concurrent agents safe. Track D additionally waits for the in-flight v2
residual-attribution probe so batching targets the measured bucket.

Track C reports first by design — it is the go/no-go gate on poke-engine's
gen3 accuracy. If it fails badly, A and B pivot before they are deep. (One
gen3 bug is already known and patched locally: Rest/Sleep Talk PP
underflow.)

## The golden corpus (track B's definition of done)

The network only understands positions in the exact encoding it was trained
on, and "exact" cannot be verified by reading code — only against a
reference. The corpus is that reference: a few thousand decision points from
real games storing (a) the position in poke-engine representation (input to
the new encoder) and (b) the observation tensor the production encoder
emitted for that position (golden output). Track B is done when every
stored tensor is reproduced bit-for-bit.

It guards against the failure mode that never crashes: encoding drift.
A transition token ordered differently, an HP fraction scaled off a
different base, a status categorical resolved through the wrong vocab
priority — every one produces a valid-looking tensor, no fallback fires,
and search strength quietly sags while the symptom points at the search
logic. (Precedents: the `include_turn_merged` capture-flag miss — the loud
version; the pipeline parity-lineage bug — the quiet version.)

Three roles for one artifact: definition of done for the encoder agent, a
binary ship gate for integration, and a permanent regression net for future
vocab/schema changes.

**Exemption rule:** fields the protocol stream knows that a reconstructed
engine state cannot (or vice versa) may be legitimately unequal. Each such
field gets an explicit documented exemption with a justification — never a
global loosening of the comparison. The exemption list IS the enumerated
residual domain-shift risk being accepted.

Corpus generation does not need track A: convert positions from real game
transcripts via foul-play's existing helpers.

## Validation gates (right-sized per the 2026-07-17 owner directive)

Minutes, not hours, per track: A = constructor unit tests + P-1 checksum;
B = golden corpus pass; C = differential harness summary; D = a 1–2 game
mechanics smoke with batching on. The expensive read runs once: after
integration, a single 200-seed paired FoulPlay comparison (both arms,
shared seeds, the standard harness). Success is more winrate at fixed
wall-clock, not more visits.

## POC checkpoint (2026-07-18)

The speed target is demonstrated end to end:
[`engine_search_poc.md`](engine_search_poc.md) — poke-engine MCTS over
belief-sampled worlds as a standard rollout policy, ~475k simulations per
searched decision at 0.44s (FoulPlay-class throughput, ~18× faster than
Tier-1 root-PUCT), with tradeoffs enumerated. Two findings feed back into
the plan: the belief sampler's deterministic dead-ends cap any determinized
search at ~45% of decisions searched (highest-leverage strength lever,
upstream of both search stacks), and speed remains decoupled from strength
until track B puts the learned model on this path.

## Integration (serial, single owner)

After A+B land and C passes: swap the branch simulator behind the existing
search-policy interface, leaf evaluation batched via D. Then the one paired
read. If the corpus and fidelity gates pass, everything downstream — leaf
calibration, the W2 budget→value curve, W3 frontier reads — transfers
unchanged and re-runs cheaply on the fast path.

Carried from v2 unchanged: the W2 value curve and W3 frontier paired reads
remain wanted and become near-free after the swap; the owner criteria for
any binding claim (+3 md / +5 fp, paired CI > 0, never losing to the prior)
still stand. Deferred until after the swap: multi-ply, tree reuse, early
termination — all re-priced on the fast path.
