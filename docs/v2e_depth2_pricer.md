# Depth-2 D pricing: interface contract (draft for agreement)

The depth-2 pricer extends today's D recipe (`gpu_engine/device_public_teacher_batch.py`, `FramePricer(vectorized=True)`) by one more ply. It prices the same `Request`s in the same two posterior worlds, with the same evaluator heads. It returns a dense array surface. Backups are pure functions over that surface.

## Semantics (per root, per world)

1. **Turn 1** is exactly D's grid:
   - every legal learner action in `recipe.pricing_order`, crossed with every legal opponent reply (column 0 if the opponent is waiting);
   - stepped with D's chance capsule `transition.sample_chance_inputs(world_state)`.
2. **At each turn-1 child:**
   - **terminal:** exact value: +1 win, −1 loss, 0 draw, from the learner's side;
   - **non-terminal:** read both seats' request flags and legal masks from the child state, as in `executor.legal`.
     - A seat that isn't requested (it is waiting) gets the no-action column/row 0 with unit mass, as `WorldPrices` does for a waiting opponent.
     - **Forced replacement after a faint** is therefore a turn-2 ply for whichever side must switch: only that seat is requested and only switches are legal. The engine resumes the deferred turn with the residual draws it saved.
3. **Turn 2:** every enumerated pair (learner action × opponent action) is stepped once more. Each leaf is then scored:
   - terminal leaves get the exact value;
   - all other leaves get the learner's value head on its public view, the same view and encoder D uses for its leaves.

## Chance at turn 2

Turn-2 chance is the engine's own draw from each child: `transition.sample_chance_inputs(child_state)`. This is exactly what the engine does when it steps the world twice with no chance override.
- The engine splits the world key once per ordinary step, independent of the actions taken. So **all ordinary turn-1 children of a world share one turn-2 chance capsule**: common random numbers across children, as across pairs at turn 1.
- A child with a pending residual (a mid-turn faint) resumes with the turn-1 draws it saved, per the engine's replacement rule. It does not draw a new block.
- Worlds stay independent. `kernel_seed` and the posterior seed are unchanged from D.

## Width

- **Turn 2:** `k_own` and `k_opp` (1–9) keep the top-k legal actions per seat, ranked by that seat's prior at the child (logit descending, lower action index on ties).
- **Turn 1 (optional):** `k1_own` keeps a prefix of `pricing_order` (the Gumbel order), and `k1_opp` keeps the top-k replies by the root opponent prior.
- **Full width is the reference:** all four at 9. The turn-2 leaf count per world is ≤ 81 × k_own × k_opp.

## Output surface (`Depth2Frame`, numpy, leading axes `[root, world]`, world ∈ {0, 1})

Action axes are action indices 0–8, not ordinals. Values are from the learner's side, in [−1, 1].

| Field | Shape | Meaning |
| --- | --- | --- |
| `pre` | `[R,2]` int32 | D's pre-refusal code per world; 0 = usable. Materialization refusals are in `world_reports`. |
| `root_peer_requested`, `root_peer_legal`, `root_peer_logits` | `[R,2]`, `[R,2,9]`, `[R,2,9]` | The opponent's prior at the root, as in `WorldPrices` |
| `t1_status` | `[R,2,9,9]` int8 | 0 = not enumerated (illegal or pruned), 1 = terminal, 2 = expanded, 3 = error |
| `t1_value` | `[R,2,9,9]` f32 | Exact value if terminal; otherwise the learner's head value at the child (equal to D's `joint_values`) |
| `t1_error` | `[R,2,9,9]` int8 | D's transition error code (1 invalid action, 2 terminal winner missing, 3 encoder error, 4 actor output refused) |
| `t1_own_requested`, `t1_own_legal`, `t1_own_logits` | `[R,2,9,9]`, `[…,9]`, `[…,9]` | The learner's prior at each child (from the same forward as `t1_value`) |
| `t1_peer_requested`, `t1_peer_legal`, `t1_peer_logits` | same | The opponent's prior at each child (its own sampled-party view, as at the root) |
| `t2_status` | `[R,2,9,9,9,9]` int8 | Indexed `[a1][r1][a2][r2]`: 0 = not enumerated, 1 = terminal, 4 = head-valued leaf, 3 = error |
| `t2_value` | `[R,2,9,9,9,9]` f32 | The leaf value: exact if terminal |
| `t2_error` | `[R,2,9,9,9,9]` int8 | Error code, as for turn 1 |
| `world_reports` | per root, per world | Seeds, materialization refusals and the reply-history approximation, as in D |
| `timing` | dict | Device and host seconds; leaves evaluated and padded |

Turn-2 entries are 0 or NaN wherever `t1_status != 2`.

## Backups (pure functions over `Depth2Frame`)

Agreed with the lead on November 5, with two amendments: the learner-max reference, and the refusal accounting.

- **`backup_learner_max(frame, root) -> (WorldPrices, WorldPrices)`** (the reference): D's treatment of the root action, carried down one turn. Average over the opponent's reply, then choose.
  - Each expanded turn-1 cell gets `max over enumerated a2 of Σ_r2 π_opp(r2)·v(a2,r2)`.
    - `π_opp` is a masked softmax of the opponent's child prior over the enumerated legal replies, renormalized under top-k.
    - Ties go to the lower index.
    - A waiting seat contributes only row/column 0, with unit mass.
  - A terminal cell keeps its exact value.
  - Any error under an enumerated cell makes the cell `None`, so the root refuses, as in D.

  The result is today's depth-1 `WorldPrices` surface, so `v5_gumbel.reduce_completed_q` gives the root target unchanged.
- **`backup_average(frame, root)`** (a diagnostic): each expanded cell gets `Σ π_own(a2)·π_opp(r2)·v(a2,r2)`, with both priors as above. It evaluates the learner's own prior play at turn 2, so it can't find a line that prior doesn't already play. It returns the same `WorldPrices` pair.
- **Other backups** (the measurement agent's hedge and others) take the same `Depth2Frame` and return the same `WorldPrices` pair, or their own reduction.

## Refusal accounting

- **The reference rule** is D's: any error under an enumerated cell refuses the root.
- **`refusal_summary(frame)`** reports:
  - the refused roots and the causes;
  - turn-1 and turn-2 error counts by code;
  - the turn-2 errors in each refused root.
- **Gate and measurement runs report:**
  - the depth-2 root refusal share next to depth 1's (D3 `FramePricer`) on the same requests;
  - turn-2 error counts by code;
  - under top-k, the share of roots whose only full-width depth-2 errors sit in cells top-k prunes away (a full-width and a top-k run on the same requests).
- **Conditional cell mask** (not built unless triggered): if depth 2 refuses more than 2 points more roots than depth 1, an opt-in mask drops an erroring turn-2 cell and renormalizes the priors over the remaining enumerated actions. It refuses only if a whole turn-2 row empties. That is a **semantic deviation from D**, and it must be flagged wherever it's used.

## Worlds beyond 2

W = 2K is K requests per root, one per independent `Recipe` seed, as in U1's `world_block_seeds`. The pricer prices them in one frame. Backups or reductions combine the blocks.

## Evaluators

The pricer runs v5 heads (`v5_model.forward`, bf16): B1-i99 and B0-i99, with D's archive grouping for learner and opponent actors. The champion 9375 is a different model type. What porting it would take is a separate note.
