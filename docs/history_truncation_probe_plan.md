# History-truncation probe — plan

Status: 2026-07-22, owner-directed. Companion to
`observation_v3_layout_cutover_plan.md` (the probe's verdict feeds the
cutover's history-region-size decision) and downstream of the input-
efficiency thread (typed adapters / history compression are SIBLING
explorations, out of scope here).

## Question

128 of the 151 observation tokens are turn-merged transition history —
~85% of the sequence and the dominant share of transformer cost at every
consumer (training, benchmarks, and the engine-search leaf path, where
per-eval encode+forward is the entire budget). Are the DEEP history slots
earning their attention, or does the policy effectively condition only on
a recent prefix?

## What is already measured (2026-07-22 census, corpus games, n=1,028)

| statistic | history slots used (of 128) |
|---|---|
| mean / median | 30.7 / 28 |
| p90 / p99 / max | 57 / 90 / 96 |
| decisions exceeding 32 | 42.8% |
| decisions exceeding 64 | 6.5% |

Conclusions so far: the region is NOT padding (a 32-cap truncates 43% of
windows — long forced-switch/multi-turn windows legitimately stack deep);
pure right-sizing to observed max saves only ~12–20% of sequence; and the
question therefore shifts from FILL to USAGE — fill proves content
exists, not that attention on it changes decisions. Caveats to close
during the probe: (a) this census ran on corpus games (near-random play);
trained-policy games have different window statistics — re-census from a
production training cache before the smoke arms' caches are cleaned;
(b) `--transition-token-budget 32` evidently bounds something narrower
than the region (fills reach 96) — document its exact semantics in the
same pass; it may be a tunable half of this question.

## The probe (zero training)

1. **Harness:** a small eval-side switch that, at encode time, masks the
   history region beyond the most recent k tokens (attention-mask edit +
   matching token zeroing; nothing else changes). Implemented in the
   benchmark path only — production encoding untouched.
2. **Grid:** k ∈ {16, 32, 64, full(128)} on the m50 5M checkpoint
   (iteration-3125) and the S 3M frontier (emeta iteration-0625) — two
   model classes guard against a capacity-dependent answer.
3. **Reads per (checkpoint, k):** the standard ladder (max-damage /
   simple-legal / random-legal) at n=1000 plus foul-play at n=1000
   (SE ≈ 1.6%). Fixed seed bands shared across k so the comparison is
   paired within checkpoint.
4. **Cost:** an afternoon — harness (agent task) + a few cluster
   benchmark jobs. Results land before the layout-cutover window closes.

## Interpretation rules (pre-registered — the asymmetry is the point)

- **Flat curve down to some k\*** (deltas within 2×SE of full, all
  opponents, both checkpoints): STRONG evidence the deeper slots are
  decorative — the trained model had them and ignored them. The cutover
  adopts a k\*-sized history region (with margin, e.g. k\*→next power of
  two) as an evidence-backed layout decision. Sequence shrinks
  accordingly (151 → 23+k\*); every consumer gets faster.
- **Degradation at small k:** usage PROVEN — but mildly confounded (a
  model trained with full history may lean on deep tokens it could have
  learned to live without). If the efficiency prize still matters, ONE
  S-scale trained variant at the candidate length (500k games, ~24h at
  fleet speeds) is the tiebreaker; otherwise keep 128 and close the
  question permanently — the attention is earning its cost.
- **Mixed/ambiguous** (flat on ladder, degraded vs foul-play, or
  class-dependent): treat as usage-proven for the cutover (keep 128);
  record the pattern — it localizes WHERE deep history matters and
  becomes input to the sibling history-compression study rather than a
  reason to truncate.

## Explicit non-goals

- Not the typed-adapter/frontend experiment (model-side, decoupled from
  schema, post-cutover).
- Not the history-compression architecture study (learned summarization
  of the history region — a real study with its own risk budget; this
  probe's k\*/usage map is an INPUT to it, not a substitute).
- No schema change happens in this plan: the probe only produces the
  evidence; the layout decision executes in the cutover PR.
