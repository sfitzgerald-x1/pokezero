# Crate search design: multi-ply decision/chance tree

Status: 2026-07-18. Implements the search-tree contract of
[`test_time_search_plan_v3.md`](test_time_search_plan_v3.md) (owner-aligned
2026-07-18) in the native `pokezero-search` crate:
`rust/pokezero-search/src/tree.rs` (core), `model.rs` (virtual-loss batched
model pricing), on top of the one-ply skeleton in `lib.rs` (which stays
untouched and exported). Correctness gates: `cargo test` in the crate,
`tests/test_multiply_chance_search.py`, and the multi-ply classes in
`tests/test_crate_model_leafeval.py`.

## Node types

- **Decision node** — one per reachable simultaneous-move state. Runs the
  same decoupled per-side PUCT as the one-ply core: each side independently
  maximizes its own PUCT score over its legal options (side two on
  `1 - value`), uniform priors until the action-mapping stream lands. Root
  options come from `root_get_all_options` (force-trapped / slow-uturn
  aware), interior nodes from `get_all_options` (force-switch / saved-move
  aware) — mirroring the vendored engine's own MCTS.
- **Chance node** — one per expanded joint-action edge. Children are the
  engine's OWN enumerated outcome list from
  `generate_instructions_from_move_pair`, each carrying the engine's exact
  `percentage`, normalized to sum to 1. Probability conservation is
  `debug_assert`ed at creation and on every expectation read, and re-checked
  tree-wide by tests. A branch stores only the engine instruction list
  (apply/reverse on the single shared `State` during traversal) plus running
  value stats — tree nodes carry engine state only, no observation tokens
  (owner decision: frozen root observations at leaves are internally
  inconsistent and OOD; per-outcome fold-state advance is track B's seam).
- **Terminal branch** — an outcome whose instructions end the battle. Holds
  the exact value ({0, 1} side-one win probability) and never grows a child.

Structure-of-arrays arenas (`Vec<DecisionNode>`, `Vec<ChanceNode>`) with
index handles; traversal advances/reverses the one `State` exactly like the
engine's own MCTS (no state clones in the loop).

## Chance policy: implemented vs contract ideal

Contract ideal (plan v3): explicit chance nodes with the engine's exact
enumerated probabilities; damage-roll branching at plies 1-2 (the engine's
own `root || parent.root` policy, `mcts.rs`); deeper plies collapse damage
EXCEPT keep KO-threshold splits when the exact roll list straddles a
KO/berry/Substitute threshold; sampling only past a depth/branch-product
cutoff.

What the vendored API (poke-engine 0.0.47, gen3 feature) actually offers,
and what was implemented with it:

| Contract item | Vendored API surface | Implemented |
|---|---|---|
| Exact enumerated outcomes | `generate_instructions_from_move_pair(state, s1, s2, branch_on_damage) -> Vec<StateInstructions>` with exact `percentage` per branch (sums to 100) | Used as-is; normalized weights, conservation asserted |
| Plies 1-2 damage branching | `branch_on_damage: bool` (turn-global) | `true` for expansions at decision depth < 2, exactly the engine's `root \|\| parent.root` |
| Deep KO-threshold splits | KO-straddle branching is built into `branch_on_damage=true` (gen3 `generate_instructions.rs`: kill-roll branch with exact probability `(1-crit)*k/16 + crit` when `max >= hp > min`, min = 0.85·max); `calculate_both_damage_rolls` (pub) returns `[max_damage, crit_damage]` per side for cheap detection | At depth ≥ 2, a straddle detector (`deep_ko_straddle`) prices both sides' rolls via `calculate_both_damage_rolls` and enables `branch_on_damage` for that expansion only (`deep_ko_split=True` default) |
| Berry/Substitute threshold splits | Not exposed as branch points by the gen3 API (no berries in gen3; Substitute damage is deterministic given the collapsed roll) | Not implemented — nothing to consume; KO is the only threshold the vendored engine branches on |
| Sampling past a cutoff | n/a | Not needed in this shape: expansion prices ALL branches once (exact); every revisit descends exactly ONE sampled branch, so per-visit cost never scales with the branch product. The cutoff concern applies to eval fan-out at expansion; observed branch counts stay small (≤ ~12 with speed-tie × crit × secondary on the bench positions) |

Two honest deviations from the ideal, both inherent to the vendored API:

1. **The `branch_on_damage` flag is turn-global.** When the deep straddle
   detector fires for one side's move, the other side's non-straddling move
   also gets the engine's crit branch — a superset of KO-only splitting.
   Branches remain exactly weighted; the cost is a ≤2× child multiplier on
   triggered turns.
2. **Move-order input to the detector is a raw-speed heuristic.**
   `calculate_both_damage_rolls` wants `side_one_moves_first`; we pass a raw
   active-speed comparison. This gates only WHETHER the engine's exact
   branching is enabled, never the branch probabilities — a wrong guess
   degrades to the engine's default collapsing (or a harmless extra split).

## Backup math (exact expectation)

The value head is a win probability, so by the law of total expectation the
optimal policy maximizes plain expected value — no risk adjustment anywhere.

- On **expansion** of joint edge (i, j): every enumerated branch k is priced
  once — exact terminal value, or the leaf seam (`LeafEval` /
  `BatchLeafEval`) — initializing branch mean `m_k`. The edge's first
  backed-up sample is `E = Σ_k p_k · m_k`: the exact expectation over the
  engine's enumerated distribution, never a sampled outcome.
- On **revisit**: traversal descends one branch (weighted sample), obtains a
  deeper sample v (recursively, the child edge's expectation), folds it into
  that branch's running mean, then backs up the RECOMPUTED
  `E = Σ_k p_k · m_k`. The chance layer therefore contributes zero sampling
  variance to every backed-up value; sampling only chooses which branch gets
  refined.
- **Decision marginals**: each visit adds the traversed edge's `E` to both
  sides' per-arm stats (side two accounted as `1 - value` at read time), as
  in the one-ply core — decoupled-PUCT marginalization over the opponent's
  traversed arms.
- **Depth cap / pseudo-branches**: a capped branch's sample is its own
  current mean (leaf estimate refined by nothing); terminal branches back up
  their exact value forever. An empty instruction list from the engine (e.g.
  both sides forced to `None`) becomes a single certain pseudo-branch that
  never grows a child — the engine MCTS's no-expand case.

Verified analytically (Rust + Python twins): gen3 toxic vs a splash-locked
Chansey — outcomes {85% hit: 6 residual damage, 15% miss}, HpFraction leaf →
root edge Q must equal 0.85·0.53 + 0.15·0.5 = 0.5255, and does (tolerance
1e-4, which also absorbs the engine's f32 percentage wobble); a guaranteed
seismic-toss KO edge reads exactly 1.0 at every depth.

## Batching through chance nodes (virtual loss)

Documented choice: **sample-by-weight for traversal, exact expectation for
backup** — the standard resolution, with one refinement: because expansion
prices all branches, every model row is generated at an expansion. A
collection round:

1. Up to `batch_size` traversals are collected (stopping early once
   `batch_size` leaf rows are pending; an expansion never splits across
   rounds, so the last expansion may overshoot by its branch count).
   Virtual loss along each path: decision arms take a provisional side-one
   loss (identical convention to the one-ply batched core, PR #716),
   traversed branches take a provisional visit, and freshly expanded
   branches carry `pending_row` markers.
2. One `BatchLeafEval` call prices all pending rows (observation batch sized
   to the exact row count; rows are template-stub encoded until the track-B
   encoder replaces the row write — same seam as the one-ply batched core).
3. `finalize` replays each traversal in collection order, replacing
   provisionals with real values and backing up expectations as above. A
   same-round traversal that bottoms out on a sibling's still-pending branch
   resolves against that branch's batch row (`TraversalEnd::Row`), not the
   provisional zero.

`batch_size=1` is the sequential regime by construction (every provisional
is replaced before the next selection). Residual virtual-loss distortion at
`batch > 1`: same-round expectation reads can see siblings' provisional
means; bounded by round size and gone at round end — keep
`batch << iterations`, exactly as the one-ply core documents.

The sequential HpFraction driver (`puct_search_multi`) runs the identical
traverse/finalize core with inline (`Ready`) pricing — one code path for
both regimes.

## Bench (HpFractionEval, Apple M-series laptop, single thread, release)

`scripts/bench_multiply_search.py`, 2026-07-18. Three positions: the
standard minimal 1v1 fixture, a curated 3v3 gen3-OU-style midgame, a 1v1
endgame where tackle straddles the KO (deep-split reachable). Cells are
sims/s and ms per decision (min-time 0.8s/cell, warm):

| position | depth | sims=256 | sims=1024 | decision/chance nodes @1024 | leaf evals @1024 | deep-KO triggers |
|---|---|---|---|---|---|---|
| minimal_1v1 | 1 | 6.4M/s, 0.04ms | 12.1M/s, 0.08ms | 1 / 4 | 36 | 0 |
| minimal_1v1 | 2 | 2.7M/s, 0.09ms | 5.5M/s, 0.19ms | 27 / 82 | 136 | 0 |
| minimal_1v1 | 3 | 2.5M/s, 0.10ms | 4.5M/s, 0.23ms | 35 / 85 | 132 | 0 |
| minimal_1v1 | 4 | 2.2M/s, 0.11ms | 4.2M/s, 0.24ms | 38 / 95 | 154 | 0 |
| midgame_3v3 | 1 | 3.2M/s, 0.08ms | 7.8M/s, 0.13ms | 1 / 36 | 184 | 0 |
| midgame_3v3 | 2 | 0.70M/s, 0.37ms | 1.21M/s, 0.85ms | 103 / 663 | 3,303 | 0 |
| midgame_3v3 | 3 | 0.64M/s, 0.40ms | 0.63M/s, 1.62ms | 260 / 1,015 | 4,198 | 15 |
| midgame_3v3 | 4 | 0.66M/s, 0.39ms | 0.61M/s, 1.68ms | 284 / 1,024 | 4,065 | 15 |
| endgame_straddle | 1 | 7.5M/s, 0.03ms | 13.8M/s, 0.07ms | 1 / 4 | 6 | 0 |
| endgame_straddle | 2 | 5.3M/s, 0.05ms | 9.4M/s, 0.11ms | 7 / 27 | 24 | 0 |
| endgame_straddle | 3 | 2.6M/s, 0.10ms | 5.3M/s, 0.19ms | 21 / 69 | 48 | 10 |
| endgame_straddle | 4 | 2.4M/s, 0.11ms | 4.5M/s, 0.23ms | 27 / 86 | 54 | 16 |

Readings:

- **Branching-factor blowup.** On the 3v3 midgame, one extra ply (1→2)
  multiplies chance nodes ~18× and leaf evals ~18× at 1024 sims
  (~30 joint arms/node × ~5-6 exactly-weighted outcomes/edge). Leaf evals
  per sim at depth 2-3 run ~3-4× — expansion prices every enumerated
  outcome, so early sims are eval-heavy; the ratio amortizes as revisits
  dominate at larger budgets.
- **The practical depth wall at these budgets is visit dilution, not
  wall-clock: depth 3 at ≤1024 sims.** Depth 4 adds ~9% decision nodes and
  no leaf-eval growth on the midgame — sims thin out through the
  decision×chance fan-out before they reach ply 4. Wall-clock stays in the
  0.6M sims/s band with the trivial eval; under model pricing the binding
  cost is leaf evals (≈4.1 evals/sim at midgame depth 3), which is what the
  batched `search_batched_multi` path amortizes.
- **Argmax stability (seeds 0-4, sims=1024):** midgame (earthquake) and
  endgame (tackle) are seed-stable at every depth and agree with depth 1.
  The minimal 1v1 flips seed-to-seed at depth ≥ 2 (ember/tackle): the
  position is LOST for side one under deeper search (Water Gun is a
  near-guaranteed 2-ply KO; root value ≈ 0.02), so the argmax is a
  near-tie between equally losing moves — depth surfacing the loss is the
  search working, the flip is noise among equivalent arms.

## Seams and non-goals

- **Leaf pricing** stays behind `LeafEval` (sequential) / `BatchLeafEval`
  (batched, `model` feature); `HpFractionEval` for correctness gates, the
  TorchScript evaluator compiles and runs against the identical tree core
  (`search_batched_multi`). Priors remain uniform until the action-index →
  `MoveChoice` mapping lands (encoder stream).
- **Per-outcome fold-state advance** (track B) is NOT yet available in Rust:
  leaves are priced from engine state only (trivial eval) or template-stub
  observations (model path). Root observations are deliberately NOT frozen
  onto leaves. The encoder plugs in at the batch-row write in
  `multiply_batched_core` and needs nothing from the tree beyond what
  branches already carry (the per-outcome instruction/event list).
- Epistemic variance (belief worlds) stays a separate axis above this crate,
  aggregated at the root by the Python orchestration.
