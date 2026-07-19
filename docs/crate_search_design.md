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
- **Per-outcome fold-state advance + native leaf encode (LANDED)**: the
  instruction→event mapping (`src/events.rs`) renders each branch's engine
  instruction list as protocol lines, the Rust `FoldState` advances a clone
  of the PARENT branch's fold state over them (fold states chain through
  the tree via the `BranchSeam` parent key — depth-k leaves carry
  root-prefix + k simulated plies of real history), and the in-crate
  encoder (`src/leaf.rs` `LeafEncoder` + `src/encoder.rs`
  `write_history_cells`) writes the REAL leaf observation at the batch-row
  write (`multiply_batched_encoded_core`, `search_batched_multi_encoded`).
  Contract + validation: docs/leaf_observation_column_map.md — root-parity
  gate 1015/1015 + 235/235 driven corpus rows byte-exact at depth 0;
  full-surface fold-product consumption byte-exact over all 1318 rows
  (`validate_rust_encoder.py --backend rust-fold`); real-observation
  overhead ~115–190µs/eval vs the template stub
  (`scripts/bench_leaf_search.py`). The template-stub paths remain exported
  for benching.
- Epistemic variance (belief worlds) stays a separate axis above this crate,
  aggregated at the root by the Python orchestration.

## Instruction→event mapping (track B seam, `src/events.rs`)

The engine's instruction list is an unlabeled state-delta stream: no line
says which move produced an instruction, who moved first, or where the
end-of-turn residual phase begins. The mapper recovers phase structure
EXACTLY by re-running the engine's own public per-move generator
(`generate_instructions_from_move`) and prefix-matching the branch: phase 1
(first mover) must be a prefix, phase 2 (second mover, phase-1 list as
`incoming`) must extend it, and the remaining tail is the end-of-turn
segment. Generation is deterministic, so segmentation is exact; a branch
that fails to segment is flagged `lossy` and rendered fold-safe, never
silently mis-attributed. Python surface: `pokezero_search.branch_events(
state, s1, s2, ctx_json, branch_on_damage, include_post_state)`; context is
`{"p1": [display species in engine party order], "p2": [...], "turn": N}`
(see `EngineWorld.party_species`).

### Coverage (fold-consumed line classes)

| Fold-consumed class | Rendered | Notes |
|---|---|---|
| `\|move\|` w/ target | ✅ | opponent-target explicit (+`[miss]`); self-target explicit on success, blank+`[still]` on failure (corpus-measured rule); Curse-by-non-Ghost self-target special case; `[from]lockedmove` continuations; Sleep Talk renders both lines with the called move recovered by candidate re-generation |
| `\|switch\|/\|drag\|` | ✅ | display details from ctx; `[from] Baton Pass`; spikes chip `[from] Spikes`; drag branches render `\|drag\|` |
| `\|cant\|` | ✅ | slp / frz / par / flinch / recharge / `ability: Truant` / `move: Taunt` |
| `\|-damage\|/\|-heal\|/\|-sethp\|` | ✅ (sethp: n/a — engine models Pain Split as Damage+Heal) | cur/max from live engine HP (plain ASCII); recoil `[from] Recoil\|[of]`; drain `[from] drain`; Rest heal `[silent]`; crash `[from] <move>`; confusion self-hit `[from] confusion`; residual damage/heals carry best-effort `[from]` tags (windows are closed there, so tags are belt-and-braces) |
| `\|faint\|` | ✅ | deferred to end of the move phase (real ordering: after recoil/drain lines) |
| `\|-status\|` | ✅ | NONE→X transitions only; Rest `[from] move: Rest` |
| `\|-boost\|/-unboost\|` | ✅ | move/intimidate boosts bare; end-of-turn (berry/ability) boosts `[from] item:`-tagged; capped boost moves emit the real 0-amount lines |
| `\|-sidestart\|/-sideend\|` | ✅ | sets vs. expiry distinguished from counter state; Rapid Spin `[from]` |
| `\|-weather\|` | ✅ | set / `[from] ability:`+`[of]` on switch-in / `[upkeep]` on decrement / `none` on dissipation |
| `\|-prepare\|` | ✅ | charge volatiles (SolarBeam class) → move-name payload (fold `pending_charge`) |
| `\|-crit\|` | ◐ | labeled by exact match against `calculate_both_damage_rolls`' collapsed crit value; NOT labelable on the KO-straddle branch (engine conflates kill-roll and crit) |
| `\|-miss\|` | ◐ | inferred (empty delta + acc<100 + deterministic causes ruled out); merged with full-para / move-fail branches where deltas coincide (below) |
| `\|-supereffective\|/-resisted\|/-immune\|` | ✅ | from the engine type chart on the mutated choice; suppressed for fixed-damage moves (real protocol rule); `-immune` covers type immunity, type-status immunity (Steel/psn etc.), and the modeled ability immunities (Levitate, Wonder Guard, absorb trio, Immunity, Insomnia/Vital Spirit, Limber, Water Veil, Magma Armor) |
| `\|-hitcount\|` | ✅ | count of rendered hits (the engine collapses 2-5-hit moves to 3 — its model, rendered faithfully) |
| `\|-activate\|` Protect/Sub, `\|-end\|` Sub, absorb `\|-heal/-immune [from] ability:` | ✅ | Blocked / hit-sub / broke-sub / Absorbed outcomes |
| `\|-transform\|` | ✅ | single line; internals silent |
| `\|turn\|/\|upkeep\|/blank` | ✅ | ply-shape aware: end-of-turn plies emit residuals+upkeep (+turn when no replacement pending); faint-replacement plies emit switch+turn; pivot follow-ups emit upkeep+turn with no residuals (the engine never runs pivot-turn residuals — its model) |

Deliberately omitted (fold provably ignores them — `fold.rs process_line`):
`|-singleturn|`, `|-curestatus|`, `|-fail|`, `|-ability|`, `|-enditem|`,
`|-mustrecharge|`, `|-start|` (except absorb signatures), `|-anim|`,
`|debug|`, chat/meta lines.

### Insufficiency findings (instruction stream ↔ event stream)

1. **The engine merges semantically distinct outcomes with identical
   deltas** (`combine_duplicate_instructions`): a fully-paralyzed turn, a
   missed move, and a failed move can be ONE branch. No mapper can split
   them — the branch is rendered as the highest-probability cause
   (deterministic causes first, then full-para over miss, fail over miss on
   already-statused targets), and the residual mass is a documented,
   measured ambiguity (the dominant class-(c) family below).
2. **The KO-straddle branch conflates kill-roll and crit** (single branch
   with combined probability): `|-crit|` is never emitted for it.
3. **Sleep Talk's called move id is not in the delta**; it is recovered by
   re-generating each `get_sleep_talk_choices` candidate and exact-matching
   the tail — unique in practice; ambiguous matches flag `lossy`.
4. **The engine has no turn counter**: `|turn|N` numbering is context
   (`ctx.turn`) + ply-shape bookkeeping (`turn_completed` in the result).

### Fidelity gate (scripts/fidelity_gate_events.py, 2026-07-18)

For every same-seat row pair of the corpus v2 fold sidecar: construct the
engine state from the recorded public payload + TRUE teams
(`engine_world.battle_spec_from_payload`, no belief sampling), step the
rounds between the boundaries with the joint actions the players actually
took, select the enumerated branch consistent with the realized outcome
(post-state actives/HP/status/boosts; HP tolerance scaled by the engine's
damage-roll collapse; post-state ties resolved on the realized action
order), advance the RECORDED row-n Rust fold state over the synthesized
lines, apply row n+1's annotation overlay, and compare fold PRODUCTS
(tokens + tendencies — the encoder-visible surface) against row n+1's
recorded products. Classes: (a) byte-identical canonical JSON, (b) equal
modulo documented equivalences (damage-roll floats within the collapse
envelope), (c) real divergence.

| corpus | boundaries | driven | (a) | (b) | (c) | skipped |
|---|---|---|---|---|---|---|
| golden-v2-scenarios | 270 | 183 | 77 (42%) | 87 (48%) | 19 (10%) | 87 |
| golden-v2 (random) | 1008 | 775 | 378 (49%) | 291 (37%) | 106 (14%) | 233 |

Class (c) decomposition (every case examined and attributed):

- **merged no-op branches** (full-para vs miss vs fail; 2 scenario + ~34
  random): insufficiency #1 — the realized minority outcome renders as the
  majority one.
- **move-line target minutiae** (~24 random): per-move `[still]`/target
  blanking details of the real sim not fully replicated (affects
  `defender_species` on failed/self-target moves only).
- **`|-crit|` on KO-capped rolls** (~9 random): insufficiency #2.
- **residual immunity classes** (~12 random): ability/clause immunities not
  in the modeled set (e.g. Soundproof, sleep clause).
- **ENGINE-MODEL deviations, not mapper defects** (fidelity findings for
  track C): (i) fixed-damage special effects (Seismic Toss class,
  `choice_special_effect`) IGNORE Protect — the engine deals damage through
  it (16 scenario + 3 random class-c; the wish_boundary scenario hits it
  every protect turn); (ii) a recharge is consumed by the engine during a
  faint-replacement ply (one ply early vs. the real game); (iii) a pivot's
  saved move never resolves after the replacement (known fail-soft,
  belief_edge_case_matrix) — these also drive most `no_branch_match` skips
  (baton_pass chains).

Skips are counted, not hidden: world construction fail-closed reasons
(encore/transform/request-state — `EngineWorldUnsupported`), and
`no_branch_match` where no enumerated branch reproduces the realized
post-state (dominated by the engine deviations above plus the documented
world approximations: sleep/rest turn counts unknown publicly, 2-5-hit
collapse to 3, screens-boundary damage).

Regression gates: `cargo test` (events unit test: render + state
restoration), `tests/test_instruction_event_mapping.py` (mapper contract +
the END-TO-END LEAF DEMO: root fold state → branch → synthesized events →
Rust fold advance → per-outcome products, hit/miss histories diverging),
and `scripts/validate_corpus_v2.py --backend rust` unchanged
(1028+290 boundaries byte-exact — the fold itself is untouched).

Review hardening (PR #727 adversarial review, both LOWs landed):
attacker-side damage renders through an attribution ladder — Rough Skin
contact punishment `[from] ability: Rough Skin|[of]` (engine order: before
recoil, exact-amount matched), Destiny Bond `[from] move: Destiny Bond`,
recoil `[from] Recoil`, genuine self-costs (Substitute / Belly Drum /
Curse / Pain Split) bare, anything unexplained bare + `lossy`
(`unattributed_self_damage`) — a bare line charges the fold's
`self_hp_cost`, so opponent-inflicted damage is never mis-read as a self
cost; and an ambiguous Sleep Talk call is flagged `lossy` even on an empty
delta (the never-mis-attribute invariant holds universally).

### Forward caveats for the in-crate-encoder PR (RESOLVED 2026-07-19)

1. **Nicknames.** Synthesized idents use display SPECIES (`p1a: Slaking`).
   Correct for randbats/local games (no nicknames) and for fold semantics
   (occupants come from switch DETAILS, which are species either way).
   Resolution: rely on the fold's details-based occupant tracking; ident
   mismatch on nicknamed ladder games is documented as cosmetic
   (docs/leaf_observation_column_map.md). OUT OF SCOPE (owner decision
   2026-07-19): this is a randbats-only project — no nicknamed-ladder
   integration is planned, so the caveat is closed rather than deferred.
2. **Opponent HP base reconciliation.** Evidence: the local harness feeds
   the OMNISCIENT stream — exact HP for both sides (corpus slices confirm)
   — so in the training/eval/paired-read domain the mapper's true-base
   rendering already matches the root fold's base; the /100 regime exists
   only on ladder (player-view) streams. Resolution: default exact;
   `EventContext.hp_percent` opts a side into Showdown's exact HP
   Percentage Mod rendering (`ceil(100*hp/maxhp)`, 99-cap; unit-tested)
   so ladder-rooted leaf fractions land on the /100 grid the root fold
   consumed. Grid distinction pinned by
   `tests/test_leaf_encoder.py::HpPercentGridTest`.

## Review caveats (PR #721, non-blocking)

- **f32 value accumulation drifts at very high sim counts.** `MoveStats.total_value`
  sums f32 `visits` times before dividing; at a constant 0.85 backup the reported
  Q reads 0.84941 at 200k sims and 0.85676 at 1M (deterministic, reproduced with a
  pure-f32 accumulator). Negligible (<1e-4) at the intended <=8192-sim budgets and
  pre-existing in the one-ply core, but the exact-expectation backup makes it the
  dominant error at extreme sim counts — switch to f64 accumulation before any
  high-sim regime.
- **Keep batch << iterations on the throughput path.** `search_batched_multi`
  root VALUE fidelity degrades as batch approaches iterations (0.934 -> 0.569 at
  batch 1 -> 64 with iters=512; argmax stays stable) — the documented virtual-loss
  tradeoff. Whoever wires the FoulPlay eval path must size batch well below the
  per-decision sim budget.
