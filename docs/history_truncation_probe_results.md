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
fill distribution from either (a) a training-cache `attention_mask.npy` or (b) a
freshly generated benchmark trajectory capture. Run + numbers: PENDING (requires
the eval image; see §4).

## 4. k-grid reads (plan steps 2–3) — PENDING

Grid: `k ∈ {16, 32, 64, 128}` × { m50 5M (`metamon-m50-2m-lr10m-ep7`
iteration-3125), emeta S 3M (`emeta-v2-2-lr3m-3m-belief` iteration-0625) }.
Reads per cell: ladder (max-damage / simple-legal / random-legal) n=1000 +
foul-play n=1000 (SE ≈ 1.6%), fixed shared seed bands (paired within
checkpoint). Both checkpoints confirmed present on the shared PVC (m50
iteration-3125 = the 5M target, written 2026-07-21).

Launcher: `pokezero-deploy/foundation/history-truncation-probe.sh` (private
deploy repo). Results collected to
`/shared/scott-experiment/history-truncation-probe-<date>/`.

## 5. Verdict (pre-registered rules from the plan) — PENDING

Applied by `scripts/analyze_history_truncation_probe.py`:
- **Flat down to k\*** (all deltas within 2×SE of full, all opponents, both
  checkpoints) → deep slots decorative; cutover adopts a k\*-sized region
  (k\* → next power of two).
- **Degradation at small k** → usage proven; keep 128 (or one S-scale trained
  variant at the candidate length as tiebreaker).
- **Mixed / class-dependent** → usage-proven for the cutover (keep 128); record
  the pattern as input to the sibling history-compression study.
