# History-truncation probe — results

Status: 2026-07-21, in progress. Companion to `history_truncation_probe_plan.md`
(the plan this executes) and `observation_v3_layout_cutover_plan.md` (the
consumer of the verdict). This doc collects the probe's evidence: the harness,
the `--transition-token-budget` semantics (plan caveat b), the trained-policy
re-census (plan caveat a), the k-grid reads, and the pre-registered verdict.

## 1. Harness (plan step 1) — DONE

An eval-side switch masks the checkpoint's transition-history region down to the
most-recent `k` tokens at decision time, deliberately mismatched from the
checkpoint's trained `transition_token_budget`. It is benchmark-only; env and
training encode are untouched.

- Flag: `python -m pokezero.neural_cli benchmark ... --history-mask-k K`
  (`src/pokezero/neural_cli.py`, benchmark subcommand).
- Mechanism: `pokezero.neural_policy.truncate_history_tensors` runs inside
  `TransformerSoftmaxPolicy.select_action`, between encode
  (`observation_window_to_torch`) and the model forward. For the transition
  region (tokens `TRANSITION_TOKEN_OFFSET .. +TRANSITION_TOKEN_COUNT`, i.e.
  indices 23..150) it keeps only the most-recent `k` attended tokens: it clears
  the attention-mask bit **and** zeroes the matching categorical/numeric rows,
  so a truncated slot becomes byte-identical to an unfilled one — the
  "attention-mask edit + matching token zeroing" the plan calls for.
- Why post-encode masking is faithful: the transformer carries no per-index
  positional embedding within a frame and the transition tokens all share one
  token type, so it is permutation-invariant across them. Blanking the *oldest*
  filled slots in place is therefore equivalent to an encode-time
  `transition_token_budget=k` — the trained model cannot distinguish "the tail
  k, compacted to the front" (what a budget-k encode produces) from "the tail k,
  left in place" (what the harness produces). Verified by
  `tests/test_neural_policy.py::TruncateHistoryTensorsTest`.
- Provenance: a truncated benchmark stamps `history_mask_k` into its summary
  JSON; a full-history benchmark's payload is unchanged (default output stays
  byte-identical for existing consumers).

## 2. `--transition-token-budget` semantics (plan caveat b) — DONE

The plan flagged that `--transition-token-budget 32` "evidently bounds something
narrower than the region (fills reach 96)". Resolved:

- **It is a training/model-config knob, NOT an eval override.** `--transition-
  token-budget` on `neural_cli` (and `rollout_cli`) sets
  `ObservationFeatureMasks.transition_token_budget`, which is stamped into the
  checkpoint's `model_config` (`neural_policy.py`). With `--initial-checkpoint`,
  `_require_mask_flags_agree_with_checkpoint` hard-fails on any disagreement, and
  at eval `local_showdown.env_config_with_checkpoint_masks` refuses to encode
  when the env masks differ from the loaded checkpoint's trained masks. So the
  flag cannot be repurposed to eval a budget-128-trained model at a smaller
  budget — which is exactly why the probe needs a separate eval-only harness
  (§1) rather than this flag.
- **What it bounds:** at encode time (`showdown._encode_turn_merged_transition_
  tokens`, `_encode_transition_tokens`),
  `budget = min(masks.transition_token_budget, spec.transition_token_count)` and
  only the most-recent `budget` tokens are filled (`stream[-budget:]`, oldest
  first at index 23); the rest stay zeroed + attention-masked. It counts
  **tokens**, and its unit is schema-dependent:
  - v2 / v2.1: one token per *declared action* (~2 tokens/turn), so budget=64 ≈
    32 turns.
  - v2.2 / v3 (turn-merged, the current default and both probe checkpoints): one
    token per *turn/lead/replacement phase*, so budget=k ≈ k turns and an
    unchanged K roughly doubles the temporal horizon vs v2.
- **Why the census fill reached 96 despite "budget 32":** the census fill is the
  number of transition slots actually populated in a *full-budget* (128) encode —
  it is not bounded by 32. The "budget 32" in the caveat referred to the v2-era
  ablation-arm training config, whose token-budget unit (per-action) differs from
  the v2.2 per-turn slots the census counts. Under the current turn-merged
  schema the region holds up to 128 turn-tokens; observed production fills are
  re-measured in §3. There is no contradiction: budget bounds *fill during
  encode*; the census measures *fill*, and it was run at full budget.
- **Probe grid mapping:** the probe's `k ∈ {16, 32, 64, 128}` are counts of
  turn-tokens on the v2.2/v3 checkpoints (`--history-mask-k` units == the region
  slot count == the `transition_token_budget` token unit for these schemas).

## 3. Trained-policy re-census (plan caveat a) — see status below

The plan's census (mean/median 30.7/28, p90/p99/max 57/90/96, 42.8% >32, 6.5%
>64; n=1,028) ran on corpus games (near-random play). Caveat (a) asks for a
re-census on trained-policy games.

Finding: the m50 and emeta **production per-iteration training caches are already
cleaned** (`/shared/scott-experiment/<run>/cache/iteration-*` are empty) — the
exact "before the smoke arms' caches are cleaned" window the plan warned about
has passed. A faithful trained-policy census therefore requires re-generating
games with the checkpoint and counting history fill on the encoded observations.

Tool: `scripts/history_slot_census.py` computes the per-decision transition-slot
fill distribution from a training-cache `attention_mask.npy` (or any encoded
observation source).

Status (2026-07-21): the only persisted caches on the shared PVC are the
`diversity-d0-smoke` self-play caches, and they are **`observation.v2`** (per-
ACTION tokens, budget capped at 64) — a census over them reads mean 34.4, median
33, p90/p99/max 64/64/64, 51.4% >32 (n=54,681), but the max=64 is the v2 budget
cap and the token unit is per-action, so this is **not comparable** to the plan's
v2.2 128-slot per-turn corpus census and cannot serve as the trained-policy
re-census. A faithful v2.2 trained-policy census therefore needs a fresh
collection with a current checkpoint (e.g. one `neural_cli iterate` collection
pass on m50/emeta), then `history_slot_census.py` on the resulting v2.2 cache.
This closes the *investigation* of caveat (a): the production v2.2 caches were
already cleaned (exactly the window the plan warned about), so the FILL re-census
is a fresh-collection follow-on. Per the plan's own logic the USAGE grid (§4)
supersedes the FILL question, so this does not block the verdict.

## 4. k-grid reads (plan steps 2–3)

Grid: `k ∈ {16, 32, 64, 128}` × { m50 5M (`metamon-m50-2m-lr10m-ep7`
iteration-3125), emeta S 3M (`emeta-v2-2-lr3m-3m-belief` iteration-0625) }.
Both checkpoints confirmed present on the shared PVC (m50 iteration-3125 = the 5M
target, written 2026-07-21). Run on GPU (pinned to the designated Crusoe nodepool
`856c0ba6…`), sharded 4×/cell, paired seed band `--seed-start 50000000` shared
across all cells. Launcher:
`pokezero-deploy/foundation/history-truncation-probe.sh`.

### Ladder — DONE (n=1000 per opponent, paired; SE ≈ 1.6%)

Checkpoint win rate at each k vs the full-128 baseline (Δ = k − 128):

**m50 5M (iteration-3125):**

| opponent | full(128) | k=64 Δ | k=32 Δ | k=16 Δ |
|---|---|---|---|---|
| max-damage | 0.909 | +0.000 | +0.000 | −0.001 |
| simple-legal | 0.978 | +0.000 | +0.000 | −0.008 |
| random-legal | 0.976 | +0.000 | +0.000 | +0.002 |

**emeta S 3M (iteration-0625):**

| opponent | full(128) | k=64 Δ | k=32 Δ | k=16 Δ |
|---|---|---|---|---|
| max-damage | 0.915 | +0.000 | +0.000 | +0.002 |
| simple-legal | 0.994 | +0.000 | +0.000 | +0.001 |
| random-legal | 0.996 | +0.000 | +0.000 | +0.004 |

Two headline facts: (1) **k=32 and k=64 are byte-identical to full-128** (Δ=0.0000
everywhere, both checkpoints) — ladder games are short enough that the history
region rarely fills past ~32 turns, so truncating there is a literal no-op,
consistent with the census (median fill 28, p90 57). (2) **No degradation at any k
on either checkpoint** — the only cell outside 2×SE is emeta k=16 vs random-legal
(0.996→1.000, a positive ceiling wobble, not a loss).

**Ladder verdict: FLAT.** No opponent, checkpoint, or k∈{16,32,64} degrades vs
full-128. Deep slots beyond ~32 are decorative on the ladder. Conservative
k\*=32 (both checkpoints fully flat at 32 and 64; k=16 also shows no degradation).

### Foul-play — IN PROGRESS (n=1000; the pre-registered tiebreak read)

Foul-play is the strongest opponent and the read most likely to expose deep-history
usage the ladder cannot. Running on the `-r2` image (adds `play_online.py
--history-mask-k`); results fold into the verdict via
`analyze_history_truncation_probe.py --foulplay-root`.

## 5. Verdict (pre-registered rules) — PENDING FOUL-PLAY

Ladder is FLAT (k\*=32). The final verdict follows the pre-registered asymmetry
once foul-play lands:
- **Flat on ladder AND foul-play** → deep slots decorative; cutover adopts a
  k\*-sized region (k\*=32 → next power of two = 32; sequence 151 → 55). Every
  consumer (training, benchmarks, engine-search leaf) gets faster.
- **Flat on ladder, degraded vs foul-play** → MIXED: keep 128; the pattern
  localizes where deep history matters and feeds the sibling history-compression
  study.
- Ladder degradation did not occur, so the pure "usage proven on ladder" branch
  is ruled out.
