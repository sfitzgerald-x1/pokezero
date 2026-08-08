//! Model-prior GATHER and APPLY: the seam between a model's flat per-action
//! prior rows and a decision node's own per-arm PUCT priors.
//!
//! Deliberately OUTSIDE the `model` cargo feature even though its only caller
//! is `crate::model`. The logic here is pure arithmetic over `&[f32]` and
//! `&[Option<usize>]` — it has no tch dependency — and behind the feature gate
//! it could only be tested on a host with a working libtorch, which is exactly
//! the "unrun in the environment where it matters" failure the opponent-priors
//! work is trying not to repeat. Keeping it here makes the whole path testable
//! under a plain `cargo test`.
//!
//! Two rules hold everywhere in this module, and every test below is written
//! to fail if either is broken:
//!
//! * **SLOT ORDER.** A gathered vector is in the DECISION NODE's option order,
//!   and the value at option `i` is the model's prior for action-block slot
//!   `map[i]`. The map is not sorted, not contiguous, and not an identity —
//!   `map[i]` is wherever the leaf encoder's action surface put that option.
//! * **NO REFLECTION AT THE SEAT BOUNDARY.** Priors are per-seat action
//!   distributions applied to the seat that OWNS the actions. The opponent
//!   head is gathered through the OPPONENT's map and written to the
//!   OPPONENT's stat vector. Only *values* flip between seats, and that
//!   happens elsewhere (`multiply_batched_encoded_core`). Nothing in this file
//!   negates a side flag or computes `1 - p`.

use crate::tree::{apply_self_priors, Tree};

/// Gather a leaf's action-block priors onto a seat's option list.
///
/// `priors_row` is one row of the UNMASKED softmax; the gathered subset is
/// renormalized over the mapped options — mathematically identical to the
/// masked softmax restricted to those actions (exp(l_i)/Σ_mapped exp(l_j)).
/// Returns `None` — leaving the node uniform — when any option lacks an
/// action-block slot (the whole node falls back rather than zeroing arms the
/// model cannot see) or the mapped mass underflows.
#[cfg_attr(not(feature = "model"), allow(dead_code))]
pub(crate) fn gather_self_priors(priors_row: &[f32], map: &[Option<usize>]) -> Option<Vec<f32>> {
    if map.is_empty() {
        return None;
    }
    let mut gathered = Vec::with_capacity(map.len());
    let mut sum = 0.0f32;
    for entry in map {
        let index = (*entry)?;
        let prior = *priors_row.get(index)?;
        sum += prior;
        gathered.push(prior);
    }
    // NaN comparisons are false, so a non-finite logit would slip past the
    // underflow guard alone and propagate into stat.prior.
    if !sum.is_finite() || sum <= 1e-8 {
        return None;
    }
    for prior in &mut gathered {
        *prior /= sum;
    }
    Some(gathered)
}

/// One row of a flat `[n_rows, action_count]` prior block.
///
/// `None` rather than a slice panic: a short/ragged block is a model-contract
/// break, and the callers already have a documented uniform fallback for "no
/// priors for this branch". Failing into that fallback keeps a shape bug from
/// taking down a whole search — it is still counted as a prior fallback, which
/// is the telemetry that surfaces it.
#[cfg_attr(not(feature = "model"), allow(dead_code))]
pub(crate) fn prior_row(flat: &[f32], row: usize, action_count: usize) -> Option<&[f32]> {
    if action_count == 0 {
        return None;
    }
    let base = row.checked_mul(action_count)?;
    flat.get(base..base.checked_add(action_count)?)
}

/// Which seat's prior slot a resolved vector belongs in on the chance branch.
///
/// The seat that OWNS the arms is carried separately (`side_one`); this only
/// picks the storage field, which exists in two copies precisely so the two
/// seats' vectors can never be confused for one another.
#[cfg_attr(not(feature = "model"), allow(dead_code))]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PriorSeat {
    /// The searching seat's own policy head (`LeafBatchOutput::priors`).
    Acting,
    /// The opponent action head (`LeafBatchOutput::opponent_priors`).
    Opponent,
}

/// Number of branches whose priors were gathered AND landed, and the number
/// that fell back to uniform.
#[cfg_attr(not(feature = "model"), allow(dead_code))]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) struct PriorResolution {
    pub applied: usize,
    pub fallbacks: usize,
}

/// Resolve one batch round's pending prior maps against the batch's own prior
/// rows: gather per branch, apply onto a child decision node that already
/// exists, and store on the branch for the child that does not exist yet.
///
/// `pending` entries are `(chance-branch key, BATCH ROW, option→action-slot
/// map)`. The row is the branch's index in THIS round's batch, so the prior
/// row is `flat[row * action_count .. +action_count]` — a branch reading any
/// other row would price its arms from a different leaf's policy.
///
/// `side_one` is the seat that owns the mapped arms: `self_side_one` for
/// [`PriorSeat::Acting`], `!self_side_one` for [`PriorSeat::Opponent`]. It is
/// stored on the branch verbatim so the lazy application at child creation
/// ([`crate::tree::apply_branch_child_priors`]) is a plain write to that
/// seat's stats with no negation anywhere.
#[cfg_attr(not(feature = "model"), allow(dead_code))]
pub(crate) fn resolve_pending_priors(
    tree: &mut Tree,
    pending: &[((usize, usize), usize, Vec<Option<usize>>)],
    flat: &[f32],
    action_count: usize,
    side_one: bool,
    seat: PriorSeat,
) -> PriorResolution {
    let mut resolution = PriorResolution::default();
    for (key, row, map) in pending {
        let priors = prior_row(flat, *row, action_count)
            .and_then(|priors_row| gather_self_priors(priors_row, map));
        let Some(priors) = priors else {
            resolution.fallbacks += 1;
            continue;
        };
        let child = tree.chances[key.0].branches[key.1].child;
        let applied = match child {
            Some(child) => apply_self_priors(&mut tree.decisions[child], side_one, &priors),
            None => true,
        };
        // Parked on the branch whether or not the child exists yet: a child
        // created LATER in this search takes its priors from here (batch > 1
        // can descend a sibling before its priors resolve), and an arity
        // mismatch with TODAY's child is not necessarily one with a later one.
        let branch = &mut tree.chances[key.0].branches[key.1];
        let slot = match seat {
            PriorSeat::Acting => &mut branch.child_self_priors,
            PriorSeat::Opponent => &mut branch.child_opponent_priors,
        };
        *slot = Some((side_one, priors));
        if applied {
            resolution.applied += 1;
        } else {
            resolution.fallbacks += 1;
        }
    }
    resolution
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::{ChanceBranch, ChanceNode, DecisionNode, Tree};
    use crate::MoveStats;
    use poke_engine::engine::state::MoveChoice;
    use std::collections::HashMap;

    /// A recognizable non-uniform, non-monotone row over 9 action slots
    /// (schema v1's action count). Every entry is distinct, so ANY wrong slot
    /// choice changes the gathered vector.
    const ROW: [f32; 9] = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.05, 0.03, 0.29];

    fn approx(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "length: {actual:?}");
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() < 1e-6,
                "slot {i}: got {a}, want {e} (full: {actual:?})"
            );
        }
    }

    // -----------------------------------------------------------------
    // gather: slot order and indexing
    // -----------------------------------------------------------------

    /// THE slot-order pin. The map is a permutation with a gap, so the
    /// gathered vector equals the row read THROUGH the map and nothing else:
    /// identity (`row[i]`), sorted-slot order, and any ±1 index shift all
    /// produce a different vector.
    #[test]
    fn gather_reads_the_row_through_the_map_in_option_order() {
        let map = vec![Some(5), Some(1), Some(8)];
        let gathered = gather_self_priors(&ROW, &map).expect("mapped mass is well above underflow");
        let sum = ROW[5] + ROW[1] + ROW[8];
        approx(&gathered, &[ROW[5] / sum, ROW[1] / sum, ROW[8] / sum]);

        // Explicitly NOT the identity read, and explicitly NOT the same
        // multiset in slot-sorted order: both are the mutants this pins.
        let identity_sum = ROW[0] + ROW[1] + ROW[2];
        assert!(
            (gathered[0] - ROW[0] / identity_sum).abs() > 1e-3,
            "gather must not ignore the map"
        );
        let sorted = [ROW[1] / sum, ROW[5] / sum, ROW[8] / sum];
        assert!(
            (gathered[0] - sorted[0]).abs() > 1e-3,
            "gather must preserve OPTION order, not action-slot order"
        );
    }

    /// An off-by-one on the action index is the cheapest way to silently
    /// mis-price every arm, and a uniform row would hide it. This row is
    /// strictly increasing over the mapped window, so a +1 or -1 shift is
    /// order-detectable, not just value-detectable.
    #[test]
    fn gather_is_off_by_one_sensitive() {
        let row = [0.1, 0.2, 0.3, 0.4];
        let map = vec![Some(0), Some(1)];
        let gathered = gather_self_priors(&row, &map).expect("mass present");
        approx(&gathered, &[1.0 / 3.0, 2.0 / 3.0]);
        // What the +1 mutant would produce, spelled out so the difference is
        // asserted rather than assumed.
        let shifted = gather_self_priors(&row, &vec![Some(1), Some(2)]).expect("mass present");
        approx(&shifted, &[0.4, 0.6]);
        assert!((gathered[0] - shifted[0]).abs() > 1e-3);
    }

    #[test]
    fn gather_renormalizes_over_the_mapped_subset_only() {
        let map = vec![Some(0), Some(2)];
        let gathered = gather_self_priors(&ROW, &map).expect("mass present");
        let total: f32 = gathered.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "gathered mass {total} != 1");
        // Renormalized over the MAPPED subset (0.01 + 0.04), not over the whole
        // row (which sums to 1.0 and would leave the raw values untouched).
        approx(&gathered, &[0.2, 0.8]);
    }

    /// Single mapped option: the renormalization must produce 1.0, not the raw
    /// row value. A no-op "renormalizer" leaves 0.16 here.
    #[test]
    fn gather_of_a_single_mapped_option_is_certainty() {
        let gathered = gather_self_priors(&ROW, &vec![Some(4)]).expect("mass present");
        approx(&gathered, &[1.0]);
    }

    /// Order of the map is the order of the output, including when the map is
    /// the reverse permutation. Pins that no sort sneaks in.
    #[test]
    fn gather_output_order_follows_the_map_exactly() {
        let forward = gather_self_priors(&ROW, &vec![Some(0), Some(5)]).expect("mass");
        let reverse = gather_self_priors(&ROW, &vec![Some(5), Some(0)]).expect("mass");
        approx(&forward, &[reverse[1], reverse[0]]);
        assert!(
            (forward[0] - reverse[0]).abs() > 1e-3,
            "reversing the map must reverse the output"
        );
    }

    // -----------------------------------------------------------------
    // gather: the fallback contract (whole node, never partial)
    // -----------------------------------------------------------------

    /// One unmapped option fails the WHOLE node. A per-option fallback (0.0 or
    /// uniform for the unmapped arm) would zero or invent an arm the model
    /// cannot see; the design says the node goes uniform instead.
    #[test]
    fn one_unmapped_option_falls_the_whole_node_back() {
        assert_eq!(
            gather_self_priors(&ROW, &vec![Some(0), None, Some(2)]),
            None
        );
        // ... including when the unmapped option is last, i.e. after real
        // values have already been accumulated.
        assert_eq!(
            gather_self_priors(&ROW, &vec![Some(0), Some(2), None]),
            None
        );
    }

    #[test]
    fn an_out_of_range_action_slot_falls_back_rather_than_panicking() {
        assert_eq!(gather_self_priors(&ROW, &vec![Some(0), Some(9)]), None);
        assert_eq!(gather_self_priors(&[], &vec![Some(0)]), None);
    }

    #[test]
    fn an_empty_option_list_falls_back() {
        assert_eq!(gather_self_priors(&ROW, &[]), None);
    }

    #[test]
    fn underflowing_mapped_mass_falls_back() {
        let row = [1e-12f32, 1e-12, 1.0];
        assert_eq!(gather_self_priors(&row, &vec![Some(0), Some(1)]), None);
        // The guard is on the MAPPED mass, so the same row with the big slot
        // mapped in must succeed — otherwise the test above would also pass
        // against a "never gather anything" mutant.
        assert!(gather_self_priors(&row, &vec![Some(0), Some(2)]).is_some());
    }

    #[test]
    fn non_finite_priors_fall_back_instead_of_poisoning_stats() {
        let nan = [f32::NAN, 0.5, 0.5];
        assert_eq!(gather_self_priors(&nan, &vec![Some(0), Some(1)]), None);
        let inf = [f32::INFINITY, 0.5, 0.5];
        assert_eq!(gather_self_priors(&inf, &vec![Some(0), Some(1)]), None);
        // A negative row that cancels to ~0 would divide by ~0; the finite +
        // underflow guard has to catch it before it reaches stat.prior.
        let cancelling = [1.0f32, -1.0, 0.5];
        assert_eq!(
            gather_self_priors(&cancelling, &vec![Some(0), Some(1)]),
            None
        );
    }

    // -----------------------------------------------------------------
    // prior_row: batch row indexing
    // -----------------------------------------------------------------

    #[test]
    fn prior_row_slices_the_requested_row_and_only_that_row() {
        let flat: Vec<f32> = (0..12).map(|v| v as f32).collect();
        assert_eq!(prior_row(&flat, 0, 4), Some(&[0.0, 1.0, 2.0, 3.0][..]));
        assert_eq!(prior_row(&flat, 1, 4), Some(&[4.0, 5.0, 6.0, 7.0][..]));
        assert_eq!(prior_row(&flat, 2, 4), Some(&[8.0, 9.0, 10.0, 11.0][..]));
        // Past the end -> fallback, not a panic and not a wrapped read.
        assert_eq!(prior_row(&flat, 3, 4), None);
        assert_eq!(prior_row(&flat, 0, 13), None);
        assert_eq!(prior_row(&flat, 0, 0), None);
        assert_eq!(prior_row(&flat, usize::MAX, 4), None);
    }

    // -----------------------------------------------------------------
    // resolve: seat routing, row routing, storage, telemetry
    // -----------------------------------------------------------------

    fn stats(n: usize) -> Vec<MoveStats> {
        (0..n)
            .map(|i| MoveStats {
                display: format!("arm{i}"),
                prior: 1.0 / n as f32,
                visits: 0,
                total_value: 0.0,
            })
            .collect()
    }

    fn decision(s1_arms: usize, s2_arms: usize) -> DecisionNode {
        DecisionNode {
            visits: 0,
            depth: 1,
            s1_options: vec![MoveChoice::None; s1_arms],
            s2_options: vec![MoveChoice::None; s2_arms],
            s1_stats: stats(s1_arms),
            s2_stats: stats(s2_arms),
            children: HashMap::new(),
        }
    }

    fn branch(child: Option<usize>) -> ChanceBranch {
        ChanceBranch {
            probability: 1.0,
            instructions: Vec::new(),
            value_sum: 0.5,
            visits: 1,
            terminal: None,
            no_expand: false,
            pending_row: None,
            child,
            child_self_priors: None,
            child_opponent_priors: None,
        }
    }

    /// Root decision + one chance node whose single branch already has a child
    /// decision node with `s1_arms`/`s2_arms` arms.
    fn tree_with_child(s1_arms: usize, s2_arms: usize) -> Tree {
        Tree {
            decisions: vec![decision(2, 2), decision(s1_arms, s2_arms)],
            chances: vec![ChanceNode {
                branches: vec![branch(Some(1))],
            }],
        }
    }

    /// The orientation pin. The opponent head is gathered through the
    /// OPPONENT's map and written to the OPPONENT's stats: with the searching
    /// seat on side one, opponent priors land on `s2_stats` and side one is
    /// left uniform. A mutant that passes `self_side_one` here (or negates the
    /// flag anywhere downstream) writes the opponent's distribution onto the
    /// searching seat's arms, and both assertions below catch it.
    #[test]
    fn opponent_priors_land_on_the_opponent_seat_and_leave_the_actor_uniform() {
        let self_side_one = true;
        let mut tree = tree_with_child(3, 2);
        let pending = vec![((0usize, 0usize), 0usize, vec![Some(5), Some(1)])];
        let resolution = resolve_pending_priors(
            &mut tree,
            &pending,
            &ROW,
            ROW.len(),
            !self_side_one,
            PriorSeat::Opponent,
        );
        assert_eq!(
            resolution,
            PriorResolution {
                applied: 1,
                fallbacks: 0
            }
        );
        let sum = ROW[5] + ROW[1];
        let child = &tree.decisions[1];
        approx(
            &child.s2_stats.iter().map(|s| s.prior).collect::<Vec<_>>(),
            &[ROW[5] / sum, ROW[1] / sum],
        );
        // The acting seat is untouched: still the uniform 1/3 from make_stats.
        approx(
            &child.s1_stats.iter().map(|s| s.prior).collect::<Vec<_>>(),
            &[1.0 / 3.0; 3],
        );
    }

    /// The same call with the searching seat on side TWO. The opponent is then
    /// side one, so the identical vector must land on `s1_stats` instead. A
    /// hard-coded seat (always s2 for "opponent") passes the test above and
    /// fails this one.
    #[test]
    fn opponent_seat_follows_the_searching_seat_rather_than_a_fixed_side() {
        let self_side_one = false;
        let mut tree = tree_with_child(2, 3);
        let pending = vec![((0usize, 0usize), 0usize, vec![Some(5), Some(1)])];
        resolve_pending_priors(
            &mut tree,
            &pending,
            &ROW,
            ROW.len(),
            !self_side_one,
            PriorSeat::Opponent,
        );
        let sum = ROW[5] + ROW[1];
        let child = &tree.decisions[1];
        approx(
            &child.s1_stats.iter().map(|s| s.prior).collect::<Vec<_>>(),
            &[ROW[5] / sum, ROW[1] / sum],
        );
        approx(
            &child.s2_stats.iter().map(|s| s.prior).collect::<Vec<_>>(),
            &[1.0 / 3.0; 3],
        );
    }

    /// Both heads resolved against the SAME batch, as the search does. The two
    /// seats must end up with their own head's distribution — a mutant that
    /// feeds one array to both loops leaves the two stat vectors equal.
    #[test]
    fn the_two_heads_stay_separated_when_resolved_off_one_batch() {
        let self_side_one = true;
        let mut tree = tree_with_child(2, 2);
        let self_row = [0.7f32, 0.1, 0.1, 0.1];
        let opponent_row = [0.1f32, 0.1, 0.1, 0.7];
        let map = vec![Some(0), Some(3)];
        let pending = vec![((0usize, 0usize), 0usize, map)];
        resolve_pending_priors(
            &mut tree,
            &pending,
            &self_row,
            4,
            self_side_one,
            PriorSeat::Acting,
        );
        resolve_pending_priors(
            &mut tree,
            &pending,
            &opponent_row,
            4,
            !self_side_one,
            PriorSeat::Opponent,
        );
        let child = &tree.decisions[1];
        approx(
            &child.s1_stats.iter().map(|s| s.prior).collect::<Vec<_>>(),
            &[0.875, 0.125],
        );
        approx(
            &child.s2_stats.iter().map(|s| s.prior).collect::<Vec<_>>(),
            &[0.125, 0.875],
        );
        // And each head is stored in its OWN branch slot, tagged with the seat
        // that owns the arms.
        let branch = &tree.chances[0].branches[0];
        assert_eq!(
            branch.child_self_priors.as_ref().map(|(s, _)| *s),
            Some(true)
        );
        assert_eq!(
            branch.child_opponent_priors.as_ref().map(|(s, _)| *s),
            Some(false)
        );
        approx(
            &branch.child_self_priors.as_ref().unwrap().1,
            &[0.875, 0.125],
        );
        approx(
            &branch.child_opponent_priors.as_ref().unwrap().1,
            &[0.125, 0.875],
        );
    }

    /// Each pending entry reads ITS OWN batch row. Two branches in one round,
    /// with rows whose gathers are distinguishable: a mutant that always reads
    /// row 0 (or `row` without the `* action_count` stride) gives both
    /// branches the same priors.
    #[test]
    fn each_branch_reads_its_own_batch_row() {
        let mut tree = Tree {
            decisions: vec![decision(2, 2), decision(2, 2), decision(2, 2)],
            chances: vec![ChanceNode {
                branches: vec![branch(Some(1)), branch(Some(2))],
            }],
        };
        // row 0 favours slot 0, row 1 favours slot 1.
        let flat = vec![0.9f32, 0.1, 0.0, 0.1, 0.9, 0.0];
        let map = vec![Some(0), Some(1)];
        let pending = vec![
            ((0usize, 0usize), 0usize, map.clone()),
            ((0usize, 1usize), 1usize, map),
        ];
        let resolution =
            resolve_pending_priors(&mut tree, &pending, &flat, 3, false, PriorSeat::Opponent);
        assert_eq!(resolution.applied, 2);
        approx(
            &tree.decisions[1]
                .s2_stats
                .iter()
                .map(|s| s.prior)
                .collect::<Vec<_>>(),
            &[0.9, 0.1],
        );
        approx(
            &tree.decisions[2]
                .s2_stats
                .iter()
                .map(|s| s.prior)
                .collect::<Vec<_>>(),
            &[0.1, 0.9],
        );
    }

    /// A branch whose child does not exist yet is NOT a fallback: the priors
    /// are parked on the branch and consumed when the child is created.
    #[test]
    fn a_childless_branch_stores_its_priors_for_later_and_counts_as_applied() {
        let mut tree = Tree {
            decisions: vec![decision(2, 2)],
            chances: vec![ChanceNode {
                branches: vec![branch(None)],
            }],
        };
        let pending = vec![((0usize, 0usize), 0usize, vec![Some(0), Some(5)])];
        let resolution = resolve_pending_priors(
            &mut tree,
            &pending,
            &ROW,
            ROW.len(),
            false,
            PriorSeat::Opponent,
        );
        assert_eq!(
            resolution,
            PriorResolution {
                applied: 1,
                fallbacks: 0
            }
        );
        let stored = tree.chances[0].branches[0]
            .child_opponent_priors
            .as_ref()
            .expect("priors parked on the branch");
        assert_eq!(stored.0, false, "the stored flag is the OWNING seat");
        let sum = ROW[0] + ROW[5];
        approx(&stored.1, &[ROW[0] / sum, ROW[5] / sum]);
        assert!(
            tree.chances[0].branches[0].child_self_priors.is_none(),
            "the opponent head must not write the acting seat's slot"
        );
    }

    /// An arity disagreement between the map and the node's arms is counted as
    /// a fallback and leaves the node's uniform priors intact — and it must
    /// still be STORED, because the mismatch is with the child that exists
    /// now, not necessarily with a later one.
    #[test]
    fn an_arity_mismatch_falls_back_without_touching_the_stats() {
        let mut tree = tree_with_child(2, 3);
        // Two mapped options for a three-arm opponent node.
        let pending = vec![((0usize, 0usize), 0usize, vec![Some(0), Some(5)])];
        let resolution = resolve_pending_priors(
            &mut tree,
            &pending,
            &ROW,
            ROW.len(),
            false,
            PriorSeat::Opponent,
        );
        assert_eq!(
            resolution,
            PriorResolution {
                applied: 0,
                fallbacks: 1
            }
        );
        approx(
            &tree.decisions[1]
                .s2_stats
                .iter()
                .map(|s| s.prior)
                .collect::<Vec<_>>(),
            &[1.0 / 3.0; 3],
        );
    }

    /// An unmapped option makes the branch a fallback and writes NOTHING —
    /// neither the stats nor the branch slot, so a later child does not pick
    /// up a half-built vector.
    #[test]
    fn a_gather_fallback_writes_nothing_at_all() {
        let mut tree = tree_with_child(2, 2);
        let pending = vec![((0usize, 0usize), 0usize, vec![Some(0), None])];
        let resolution = resolve_pending_priors(
            &mut tree,
            &pending,
            &ROW,
            ROW.len(),
            false,
            PriorSeat::Opponent,
        );
        assert_eq!(
            resolution,
            PriorResolution {
                applied: 0,
                fallbacks: 1
            }
        );
        assert!(tree.chances[0].branches[0].child_opponent_priors.is_none());
        approx(
            &tree.decisions[1]
                .s2_stats
                .iter()
                .map(|s| s.prior)
                .collect::<Vec<_>>(),
            &[0.5, 0.5],
        );
    }

    /// The acting-seat path routes to the acting seat's slot and stats, so the
    /// seat-routing tests above cannot pass by writing everything to s2.
    #[test]
    fn the_acting_head_writes_the_acting_seat() {
        let mut tree = tree_with_child(2, 2);
        let pending = vec![((0usize, 0usize), 0usize, vec![Some(0), Some(5)])];
        resolve_pending_priors(
            &mut tree,
            &pending,
            &ROW,
            ROW.len(),
            true,
            PriorSeat::Acting,
        );
        let sum = ROW[0] + ROW[5];
        approx(
            &tree.decisions[1]
                .s1_stats
                .iter()
                .map(|s| s.prior)
                .collect::<Vec<_>>(),
            &[ROW[0] / sum, ROW[5] / sum],
        );
        approx(
            &tree.decisions[1]
                .s2_stats
                .iter()
                .map(|s| s.prior)
                .collect::<Vec<_>>(),
            &[0.5, 0.5],
        );
        assert!(tree.chances[0].branches[0].child_opponent_priors.is_none());
    }
}
