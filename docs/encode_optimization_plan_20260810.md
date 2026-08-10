# Encode optimization plan — retire the per-leaf JSON round-trip, behind a byte-identity gate

Written 2026-08-10. Owner-approved direction; this document is the execution contract.

## 1. The finding, verified from source

Search is encode-dominated, and the hot path does two things per leaf that do not
need doing per leaf:

- `leaf_row_inputs` — **17.2% of decision wall** — starts from
  `self.root.clone()`: a deep clone of a `serde_json::Value` tree, per leaf, to
  mutate a handful of fields.
- `encode_row_value` — **24.3% of decision wall** — reads the row back through
  ~55 string-keyed `get()` lookups against a schema that is **fixed for the run**.

Together: **41.5% of decision wall in two functions**, consistent with the earlier
~275µs/leaf encode attribution. The shape exists for a good reason — the JSON row
mirrors the Python `showdown.py` encoder, and encoder agreement is load-bearing for
every fidelity result — which is why this plan is staged and gated rather than a
rewrite.

## 2. Sequencing constraint: no encode change lands mid-campaign

Wall-clock comparisons are valid only within one build. The MCTS axis study
(`docs/mcts_axis_cost_strength_study_20260809.md`, authored in the eval workspace
and landing separately) runs **first, on the current build**. It serves this plan
twice, for free:

1. Its per-shard phase walls (`row_input`, `tensor`, `products`, `row_write`,
   `encode`, `fold_clone`) are the **profile** this plan currently lacks — they
   size which encode component dominates at production config before anyone writes
   Rust. The 41.5% is a ceiling claim until then, not a promise.
2. Its cells are the **before** baseline. The after-measurement (§5) re-runs a
   subset of the same cells, same config, same node class, post-change.

## 3. Staged changes, cheapest first, each behind the same gate

**Stage 1 — pre-resolve the key lookups.** At table load, resolve the ~55 string
paths to positional indices once; read positionally per leaf. Mechanical, no data
shape change, no Python-side change. Expected to reclaim a large fraction of the
24.3% on its own.

**Stage 2 (conditional) — retire the per-leaf clone / typed Row.** A typed struct
holding the invariant root plus a per-leaf delta (or copy-on-write on the mutated
subtrees), and ultimately a typed Row consumed directly by encode, subsuming the
JSON round-trip. **Proceed only if the post-Stage-1 phase walls still show the
round-trip dominating.** Two reasons not to presume it: Stage 1 may capture most of
the win, and world collapse (#1009) already cut the number of searches issued, so
the clone's share may have shrunk since the 17.2% was measured.

No stage changes the Python encoder, the observation schema, or any emitted value.

## 4. The acceptance gate — byte-identity, all stages, no exceptions

A stage merges only with all four of:

1. **Golden-corpus bit-exactness** — the existing corpus comparison, unchanged.
2. **`assert_vocab_alignment`** — root == checkpoint == leaf pins, unchanged.
3. **The dev/validation differential windows** — nothing opened, per the standing
   "nothing opened" falsifier convention.
4. **Live A/B shard-byte identity**: the same seed block run on the pre-change and
   post-change builds must produce **byte-identical shard encoder output**. This is
   the strongest available check, costs minutes, and is the one that catches what
   fixture-based gates cannot — a divergence only reachable through live world
   construction.

Verification hygiene per the standing harness rules: per-arm identity witness from
the loaded module (`__file__` + content fingerprint), absolute per-arm paths, no
inferring the arm from the command line.

**Known tax, planned rather than discovered:** any Rust source edit moves the
source-bytes fingerprint and trips the pinned citation artifacts (the c155
mechanism — a comment-only delta is sufficient). Each stage PR includes the
surgical citation re-resolution in the same change, and never a cold full regen.

## 5. After-measurement and success criteria

Post-merge of each stage: re-run 2–3 axis-study cells (baseline config plus the two
highest-encode cells) on the same node class and seed block, and report the phase
walls side by side with the study's originals. Success for Stage 1 is a measured
reduction in `encode`-side wall at production config with zero bytes changed in
output; the go/no-go for Stage 2 is written from those numbers, not from this
document.

## 6. Non-goals and scope fences

- **Does not gate the v4 run.** Collection is showdown-backend; no search in the
  training loop. This is eval-economics work.
- No opportunistic refactors riding along: the diff per stage is the optimization,
  its tests, and the citation re-resolution — nothing else.
- If any gate in §4 reddens, the stage stops and the divergence is root-caused
  before any retry; a performance win is never traded against encoder agreement.
