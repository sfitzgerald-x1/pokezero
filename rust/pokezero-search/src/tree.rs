//! Multi-ply decision/chance search tree (engine-swap stream S1, search-tree
//! contract of docs/test_time_search_plan_v3.md).
//!
//! Node types:
//! - **Decision nodes** run decoupled per-side PUCT over the engine's legal
//!   options (simultaneous-move handling identical to the one-ply core in
//!   `lib.rs`: each side independently maximizes its own PUCT score, side two
//!   on `1 - value`).
//! - **Chance nodes** sit under every joint-action edge and carry the engine's
//!   own enumerated outcome list from `generate_instructions_from_move_pair`,
//!   with each branch weighted by the engine's exact `percentage` (normalized
//!   to sum to 1; conservation is `debug_assert`ed at every chance node).
//!
//! Backup is EXACT EXPECTATION over chance outcomes (law of total
//! expectation; the value head is a win probability, so no risk adjustment):
//! on expansion every enumerated branch is priced (terminal outcome or leaf
//! eval), and every backed-up sample through a chance node is
//! `sum_k p_k * mean_k` over the CURRENT branch means — the chance layer
//! contributes zero sampling variance to the estimate. Sampling appears in
//! exactly one place: which branch a later traversal descends to refine
//! (weighted by the exact probabilities). See `docs/crate_search_design.md`.
//!
//! Damage-roll branching mirrors the engine's own MCTS policy
//! (`third_party/poke-engine-src/src/mcts.rs`: `root || parent.root`, i.e.
//! plies 1-2): expansions at decision depth < 2 pass `branch_on_damage=true`.
//! Deeper plies use the engine's default collapsing EXCEPT when the
//! `calculate_damage_rolls`-based detector sees a damage roll straddling a KO
//! threshold, in which case the engine's exact KO-split branching is enabled
//! for that expansion (`deep_ko_split`).
//!
//! Tree nodes carry ENGINE STATE ONLY (instruction lists; the single `State`
//! is advanced/reversed during traversal). Leaves are priced through the
//! [`crate::LeafEval`] seam — per-outcome fold-state/encoder advance is a
//! separate in-flight stream (track B) that plugs in at exactly that seam.

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

use poke_engine::choices::Choice;
use poke_engine::engine::generate_instructions::{
    calculate_both_damage_rolls, generate_instructions_from_move_pair,
};
use poke_engine::engine::state::MoveChoice;
use poke_engine::instruction::Instruction;
use poke_engine::state::{Side, State};

use crate::{make_stats, parse_state, select, stats_to_json, HpFractionEval, LeafEval, MoveStats};

/// Engine damage-branching horizon: the vendored MCTS branches damage rolls
/// when expanding the root or a child of the root (`root || parent.root`),
/// i.e. at decision depths 0 and 1 (action plies 1-2).
pub(crate) const DAMAGE_BRANCH_DEPTH: u8 = 2;

/// Absolute tolerance on the engine's branch percentages summing to 100.
const PERCENT_SUM_TOL: f32 = 0.5;

// ---------------------------------------------------------------------------
// Tree arenas
// ---------------------------------------------------------------------------

pub(crate) struct DecisionNode {
    pub visits: u32,
    pub depth: u8,
    pub s1_options: Vec<MoveChoice>,
    pub s2_options: Vec<MoveChoice>,
    pub s1_stats: Vec<MoveStats>,
    pub s2_stats: Vec<MoveStats>,
    /// Joint-action edge -> chance node (arena index into `Tree::chances`).
    pub children: HashMap<(u16, u16), usize>,
}

pub(crate) struct ChanceBranch {
    /// Normalized branch probability (engine `percentage` / branch-list sum).
    pub probability: f32,
    /// The engine instructions realizing this outcome (applied/reversed on
    /// the shared `State` during traversal — engine state only, no tokens).
    pub instructions: Vec<Instruction>,
    /// Running value estimate: mean = value_sum / visits. Initialized at
    /// expansion with the branch's own price (terminal or leaf eval), so the
    /// chance-node expectation is defined from the first backup on.
    pub value_sum: f32,
    pub visits: u32,
    /// Exact terminal value when the branch ends the battle (side-one win
    /// probability, {0, 1}); terminal branches never grow children.
    pub terminal: Option<f32>,
    /// Pseudo-branch marker (empty instruction list from the engine, e.g.
    /// both sides forced to None): never grows a child.
    pub no_expand: bool,
    /// Row id while this branch's leaf value is deferred to a batched
    /// evaluator (virtual-loss batching); cleared by `finalize`.
    pub pending_row: Option<usize>,
    /// Child decision node (arena index), created lazily on first descent.
    pub child: Option<usize>,
    /// Model policy priors for the ACTING seat's options at this branch's
    /// child decision node: (acting side is side one, per-option prior in the
    /// child node's own option order, masked-renormalized). Written by the
    /// encoded model core from the branch's own leaf observation; `None`
    /// keeps the uniform priors of `make_stats` (existing cores unchanged).
    /// Applied when the child decision node is created — priors reweight
    /// exploration only, never values.
    pub child_self_priors: Option<(bool, Vec<f32>)>,
}

impl ChanceBranch {
    pub(crate) fn mean(&self) -> f32 {
        debug_assert!(self.visits > 0, "chance branch read before initialization");
        self.value_sum / self.visits as f32
    }
}

pub(crate) struct ChanceNode {
    pub branches: Vec<ChanceBranch>,
}

impl ChanceNode {
    /// Exact expectation over the enumerated outcomes' current means.
    pub(crate) fn expectation(&self) -> f32 {
        debug_assert_probability_conservation(&self.branches);
        self.branches
            .iter()
            .map(|b| b.probability * b.mean())
            .sum()
    }
}

fn debug_assert_probability_conservation(branches: &[ChanceBranch]) {
    if cfg!(debug_assertions) {
        let total: f32 = branches.iter().map(|b| b.probability).sum();
        debug_assert!(
            (total - 1.0).abs() < 1e-4,
            "chance-node probability mass {total} != 1"
        );
    }
}

pub(crate) struct Tree {
    pub decisions: Vec<DecisionNode>,
    pub chances: Vec<ChanceNode>,
}

/// Whether the acting side's most-visited root arm is mathematically
/// uncatchable within the remaining simulation budget.
///
/// The strict inequality preserves the full-budget argmax without relying on
/// tie-breaking. A single legal arm is locked as soon as the caller's minimum
/// simulation floor has been met.
#[cfg_attr(not(feature = "model"), allow(dead_code))]
pub(crate) fn root_visit_lock(
    tree: &Tree,
    side_one: bool,
    remaining: usize,
) -> Option<(bool, u32, u32)> {
    let root = tree.decisions.first()?;
    let stats = if side_one {
        &root.s1_stats
    } else {
        &root.s2_stats
    };
    if stats.is_empty() {
        return None;
    }
    let mut top = 0u32;
    let mut runner_up = 0u32;
    for stat in stats {
        if stat.visits > top {
            runner_up = top;
            top = stat.visits;
        } else if stat.visits > runner_up {
            runner_up = stat.visits;
        }
    }
    let locked = stats.len() == 1 || usize::try_from(top - runner_up).ok()? > remaining;
    Some((locked, top, runner_up))
}

impl Tree {
    /// Root decision node from the ROOT option surface (`root_get_all_options`
    /// — force-trapped / slow-uturn aware, matching the one-ply core).
    pub(crate) fn from_root(state: &State) -> PyResult<Self> {
        let (s1_options, s2_options) = state.root_get_all_options();
        if s1_options.is_empty() || s2_options.is_empty() {
            return Err(PyValueError::new_err(
                "no legal root options for one or both sides",
            ));
        }
        let root = DecisionNode {
            visits: 0,
            depth: 0,
            s1_stats: make_stats(&state.side_one, &s1_options),
            s2_stats: make_stats(&state.side_two, &s2_options),
            s1_options,
            s2_options,
            children: HashMap::new(),
        };
        Ok(Tree {
            decisions: vec![root],
            chances: Vec::new(),
        })
    }
}

// ---------------------------------------------------------------------------
// Search configuration and counters
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub(crate) struct MultiPlyConfig {
    /// Maximum decision plies (joint actions) along any path; 1 = the
    /// one-ply regime (root chance nodes never grow children).
    pub max_depth: u8,
    pub c_puct: f32,
    /// Enable KO-threshold damage splits past the engine's ply-1/2 horizon
    /// (straddle-triggered `branch_on_damage`; see `deep_ko_straddle`).
    pub deep_ko_split: bool,
}

#[derive(Default)]
pub(crate) struct SearchCounters {
    pub leaf_evals: usize,
    pub expansions: usize,
    pub deep_ko_triggers: usize,
    pub terminal_branches: usize,
    pub max_depth_reached: u8,
}

// ---------------------------------------------------------------------------
// Leaf pricing seam
// ---------------------------------------------------------------------------

/// Price of one leaf state at expansion time. `Ready` carries an immediate
/// value (sequential mode / terminal outcomes); `Deferred` names a batch row
/// whose value arrives at `finalize` (virtual-loss batched mode — used by
/// the `model`-feature batched core).
#[cfg_attr(not(feature = "model"), allow(dead_code))]
pub(crate) enum LeafPrice {
    Ready(f32),
    Deferred(usize),
}

/// Per-branch context handed to the leaf pricing seam alongside the leaf
/// state: the joint move pair of the expanded edge plus the branch's own
/// instruction list. This is exactly what the instruction→event mapping
/// (`events::render_branch_events`) needs — the track-B encoder integration
/// renders the branch's events from (pre-branch state, moves, instructions),
/// advances a clone of the root fold state, and encodes the leaf's REAL
/// observation at the batch-row write (docs/crate_search_design.md). NOTE:
/// the `&State` passed with this context has the branch's instructions
/// APPLIED (it is the leaf state); the mapper wants the pre-branch state,
/// which the consumer reconstructs via `reverse_instructions` on the same
/// shared state or by rendering before descent.
pub(crate) struct BranchSeam<'a> {
    #[allow(dead_code)]
    pub s1: &'a MoveChoice,
    #[allow(dead_code)]
    pub s2: &'a MoveChoice,
    #[allow(dead_code)]
    pub instructions: &'a [Instruction],
    /// The ancestor (chance node, branch index) whose child decision node is
    /// being expanded; `None` when expanding a root edge. The consumer's
    /// fold-state cache is keyed on this: the parent branch's advanced fold
    /// is the prefix this branch's events extend.
    #[allow(dead_code)]
    pub parent: Option<(usize, usize)>,
    /// Arena index the expanding chance node will occupy.
    #[allow(dead_code)]
    pub chance: usize,
    /// This outcome's index within the expanding chance node.
    #[allow(dead_code)]
    pub branch_index: usize,
    /// The damage-branching flag this expansion was generated with (the
    /// instruction→event mapper re-generates with the same flag).
    #[allow(dead_code)]
    pub branch_on_damage: bool,
    /// Decision depth of the EXPANDING node; this branch's child decision
    /// node (if it ever grows) sits at `depth + 1`. Lets the prior consumer
    /// skip action-map work for branches that can never grow a child
    /// (`depth + 1 >= max_depth`).
    #[allow(dead_code)]
    pub depth: u8,
}

// ---------------------------------------------------------------------------
// Traversal (selection + expansion) with virtual loss
// ---------------------------------------------------------------------------

pub(crate) struct PathStep {
    pub decision: usize,
    pub i: usize,
    pub j: usize,
    pub chance: usize,
    /// `Some(k)` = descended existing branch k; `None` = this step expanded
    /// the chance node (all branches freshly priced).
    pub branch: Option<usize>,
}

pub(crate) enum TraversalEnd {
    /// Bottom sample known at traversal time (terminal branch, depth cap on a
    /// resolved branch, or pseudo-branch).
    Ready(f32),
    /// Bottom sample is a batch row still awaiting its value (depth cap on a
    /// branch whose expansion price is deferred in the same round).
    Row(usize),
    /// Bottom is the freshly expanded chance node of the final path step; its
    /// branch prices are in place (or pending via `ChanceBranch::pending_row`).
    Expanded,
}

pub(crate) struct Traversal {
    pub path: Vec<PathStep>,
    pub end: TraversalEnd,
}

/// One selection pass from the root: descend decision nodes by decoupled
/// per-side PUCT and chance nodes by weighted sampling over the exact branch
/// probabilities, expanding the first untried joint edge. Applies VIRTUAL
/// LOSS along the way (decision arms: provisional side-one loss, identical to
/// the one-ply batched core; traversed branches: provisional visit) so
/// batched collection stays well-defined; `finalize` replaces provisionals
/// with real values. The shared `State` is restored before returning.
pub(crate) fn traverse<F: FnMut(&State, &BranchSeam) -> LeafPrice>(
    tree: &mut Tree,
    state: &mut State,
    rng: &mut StdRng,
    cfg: &MultiPlyConfig,
    counters: &mut SearchCounters,
    price: &mut F,
) -> Traversal {
    let mut path: Vec<PathStep> = Vec::with_capacity(cfg.max_depth as usize + 1);
    let mut node_idx = 0usize;
    let end = loop {
        let depth = tree.decisions[node_idx].depth;
        counters.max_depth_reached = counters.max_depth_reached.max(depth);
        // --- decision node: decoupled per-side PUCT + virtual loss ---
        let (i, j) = {
            let node = &tree.decisions[node_idx];
            let i = select(&node.s1_stats, node.visits, cfg.c_puct, true);
            let j = select(&node.s2_stats, node.visits, cfg.c_puct, false);
            (i, j)
        };
        {
            let node = &mut tree.decisions[node_idx];
            node.visits += 1;
            node.s1_stats[i].visits += 1;
            node.s2_stats[j].visits += 1;
            node.s2_stats[j].total_value += 1.0; // provisional side-one loss
        }
        let key = (i as u16, j as u16);
        match tree.decisions[node_idx].children.get(&key).copied() {
            None => {
                // --- expansion: enumerate the engine's chance outcomes ---
                let parent = path
                    .last()
                    .map(|step| (step.chance, step.branch.expect("descended steps carry a branch")));
                let chance_idx =
                    expand_edge(tree, state, node_idx, i, j, cfg, counters, price, parent);
                tree.decisions[node_idx].children.insert(key, chance_idx);
                path.push(PathStep {
                    decision: node_idx,
                    i,
                    j,
                    chance: chance_idx,
                    branch: None,
                });
                break TraversalEnd::Expanded;
            }
            Some(chance_idx) => {
                // --- chance node: weighted sample for traversal only ---
                let k = sample_branch_index(rng, &tree.chances[chance_idx].branches);
                path.push(PathStep {
                    decision: node_idx,
                    i,
                    j,
                    chance: chance_idx,
                    branch: Some(k),
                });
                let branch = &mut tree.chances[chance_idx].branches[k];
                let pre_mean = branch.mean();
                let pending = branch.pending_row;
                branch.visits += 1; // provisional; finalize adds the sample
                if let Some(v) = branch.terminal {
                    break TraversalEnd::Ready(v);
                }
                if branch.no_expand || depth + 1 >= cfg.max_depth {
                    // Depth cap / pseudo-branch: the sample is the branch's
                    // own (pre-provisional) estimate — or its batch row when
                    // that estimate is still pending in this round.
                    break match pending {
                        Some(row) => TraversalEnd::Row(row),
                        None => TraversalEnd::Ready(pre_mean),
                    };
                }
                state.apply_instructions(&tree.chances[chance_idx].branches[k].instructions);
                let child = match tree.chances[chance_idx].branches[k].child {
                    Some(child) => child,
                    None => {
                        let child = new_decision_node(tree, state, depth + 1);
                        // Model priors for the acting seat, when the encoded
                        // core priced this branch's observation (uniform
                        // otherwise — including a same-round pending eval,
                        // which the core re-applies after its batch returns).
                        if let Some((side_one, priors)) =
                            tree.chances[chance_idx].branches[k].child_self_priors.clone()
                        {
                            apply_self_priors(&mut tree.decisions[child], side_one, &priors);
                        }
                        tree.chances[chance_idx].branches[k].child = Some(child);
                        child
                    }
                };
                node_idx = child;
            }
        }
    };
    // Restore the shared state (reverse the instructions applied downward).
    unapply_path(tree, state, &path);
    Traversal { path, end }
}

/// Reverse the instruction lists applied while descending `path`.
/// Only steps that actually descended past their branch (grew/entered a
/// child) applied instructions; the traversal-ending step never did.
fn unapply_path(tree: &Tree, state: &mut State, path: &[PathStep]) {
    for (idx, step) in path.iter().enumerate().rev() {
        if idx == path.len() - 1 {
            continue; // the ending step never applies instructions
        }
        if let Some(k) = step.branch {
            state.reverse_instructions(&tree.chances[step.chance].branches[k].instructions);
        }
    }
}

/// Overwrite one side's PUCT priors with model priors (the prior term the
/// selection formula already carries; `make_stats` seeds it uniform). Returns
/// false — leaving the node's uniform priors intact — when the prior vector
/// does not align with the node's option count (callers count mismatches;
/// values are never touched, priors reweight exploration only).
pub(crate) fn apply_self_priors(node: &mut DecisionNode, side_one: bool, priors: &[f32]) -> bool {
    let stats = if side_one {
        &mut node.s1_stats
    } else {
        &mut node.s2_stats
    };
    if stats.len() != priors.len() {
        return false;
    }
    for (stat, prior) in stats.iter_mut().zip(priors) {
        stat.prior = *prior;
    }
    true
}

fn new_decision_node(tree: &mut Tree, state: &State, depth: u8) -> usize {
    let (mut s1_options, mut s2_options) = state.get_all_options();
    // Defensive: a decision node must always offer at least one arm.
    if s1_options.is_empty() {
        s1_options.push(MoveChoice::None);
    }
    if s2_options.is_empty() {
        s2_options.push(MoveChoice::None);
    }
    let node = DecisionNode {
        visits: 0,
        depth,
        s1_stats: make_stats(&state.side_one, &s1_options),
        s2_stats: make_stats(&state.side_two, &s2_options),
        s1_options,
        s2_options,
        children: HashMap::new(),
    };
    tree.decisions.push(node);
    tree.decisions.len() - 1
}

/// Expand joint edge (i, j) of `node_idx`: enumerate the engine's outcome
/// branches (exact percentages), price every branch (terminal or leaf), and
/// return the new chance node's arena index.
#[allow(clippy::too_many_arguments)]
fn expand_edge<F: FnMut(&State, &BranchSeam) -> LeafPrice>(
    tree: &mut Tree,
    state: &mut State,
    node_idx: usize,
    i: usize,
    j: usize,
    cfg: &MultiPlyConfig,
    counters: &mut SearchCounters,
    price: &mut F,
    parent: Option<(usize, usize)>,
) -> usize {
    counters.expansions += 1;
    let depth = tree.decisions[node_idx].depth;
    let node = &tree.decisions[node_idx];
    let s1_move = node.s1_options[i];
    let s2_move = node.s2_options[j];
    // The arena index this chance node will occupy (pushed at the end).
    let chance_idx = tree.chances.len();

    // Engine damage-branch policy (plies 1-2) + deep KO-threshold splits.
    let mut branch_on_damage = depth < DAMAGE_BRANCH_DEPTH;
    if !branch_on_damage && cfg.deep_ko_split && deep_ko_straddle(state, &s1_move, &s2_move) {
        branch_on_damage = true;
        counters.deep_ko_triggers += 1;
    }

    let generated =
        generate_instructions_from_move_pair(state, &s1_move, &s2_move, branch_on_damage);

    let mut branches: Vec<ChanceBranch> = Vec::with_capacity(generated.len().max(1));
    if generated.is_empty() {
        // No instructions (e.g. both sides forced to None): a single certain
        // pseudo-outcome pricing the current state, never expanded further.
        let outcome = state.battle_is_over();
        let seam = BranchSeam {
            s1: &s1_move,
            s2: &s2_move,
            instructions: &[],
            parent,
            chance: chance_idx,
            branch_index: 0,
            branch_on_damage,
            depth,
        };
        let (value_sum, visits, terminal, pending_row) =
            price_outcome(outcome, state, counters, price, &seam);
        branches.push(ChanceBranch {
            probability: 1.0,
            instructions: Vec::new(),
            value_sum,
            visits,
            terminal,
            no_expand: true,
            pending_row,
            child: None,
            child_self_priors: None,
        });
    } else {
        let total: f32 = generated.iter().map(|b| b.percentage).sum();
        debug_assert!(
            (total - 100.0).abs() < PERCENT_SUM_TOL,
            "engine branch percentages sum to {total}, expected 100"
        );
        let norm = if total > 0.0 { total } else { 100.0 };
        for (branch_index, state_instructions) in generated.into_iter().enumerate() {
            let probability = state_instructions.percentage / norm;
            let instructions = state_instructions.instruction_list;
            state.apply_instructions(&instructions);
            let outcome = state.battle_is_over();
            let seam = BranchSeam {
                s1: &s1_move,
                s2: &s2_move,
                instructions: &instructions,
                parent,
                chance: chance_idx,
                branch_index,
                branch_on_damage,
                depth,
            };
            let (value_sum, visits, terminal, pending_row) =
                price_outcome(outcome, state, counters, price, &seam);
            state.reverse_instructions(&instructions);
            branches.push(ChanceBranch {
                probability,
                instructions,
                value_sum,
                visits,
                terminal,
                no_expand: false,
                pending_row,
                child: None,
                child_self_priors: None,
            });
        }
    }
    debug_assert_probability_conservation(&branches);
    tree.chances.push(ChanceNode { branches });
    tree.chances.len() - 1
}

/// Price one enumerated outcome: exact terminal value when the battle ended,
/// else the leaf seam. Returns (value_sum, visits, terminal, pending_row).
fn price_outcome<F: FnMut(&State, &BranchSeam) -> LeafPrice>(
    outcome: f32,
    state: &State,
    counters: &mut SearchCounters,
    price: &mut F,
    seam: &BranchSeam,
) -> (f32, u32, Option<f32>, Option<usize>) {
    if outcome != 0.0 {
        counters.terminal_branches += 1;
        let v = if outcome > 0.0 { 1.0 } else { 0.0 };
        return (v, 1, Some(v), None);
    }
    counters.leaf_evals += 1;
    match price(state, seam) {
        LeafPrice::Ready(v) => (v, 1, None, None),
        LeafPrice::Deferred(row) => (0.0, 1, None, Some(row)),
    }
}

fn sample_branch_index(rng: &mut StdRng, branches: &[ChanceBranch]) -> usize {
    if branches.len() == 1 {
        return 0;
    }
    let mut roll: f32 = rng.random_range(0.0..1.0);
    for (k, branch) in branches.iter().enumerate() {
        if roll < branch.probability {
            return k;
        }
        roll -= branch.probability;
    }
    branches.len() - 1
}

// ---------------------------------------------------------------------------
// Deep KO-threshold split detector
// ---------------------------------------------------------------------------

fn move_choice_to_choice(side: &Side, mc: &MoveChoice) -> Option<Choice> {
    match mc {
        MoveChoice::Move(index) => Some(side.get_active_immutable().moves[index].choice.clone()),
        _ => None,
    }
}

/// True when either side's chosen move has a damage-roll span straddling the
/// defender's remaining HP (max roll KOs, min roll does not) — the exact
/// condition under which the engine's `branch_on_damage` produces its
/// KO-threshold split (see gen3 generate_instructions: `max_damage_dealt >=
/// defender.hp && min_damage_dealt < defender.hp`, min = 0.85 * max).
///
/// Uses the engine's public `calculate_both_damage_rolls` (returns
/// `[max_damage, crit_damage]` per side). Move order is a raw-speed
/// heuristic: it only gates WHETHER to enable the engine's exact branching,
/// never the branch probabilities themselves.
fn deep_ko_straddle(state: &State, s1_move: &MoveChoice, s2_move: &MoveChoice) -> bool {
    let c1 = move_choice_to_choice(&state.side_one, s1_move);
    let c2 = move_choice_to_choice(&state.side_two, s2_move);
    if c1.is_none() && c2.is_none() {
        return false;
    }
    let s1_first = state.side_one.get_active_immutable().speed
        >= state.side_two.get_active_immutable().speed;
    let (rolls_s1, rolls_s2) = calculate_both_damage_rolls(
        state,
        c1.clone().unwrap_or_default(),
        c2.clone().unwrap_or_default(),
        s1_first,
    );
    (c1.is_some() && straddles_ko(&rolls_s1, state.side_two.get_active_immutable().hp))
        || (c2.is_some() && straddles_ko(&rolls_s2, state.side_one.get_active_immutable().hp))
}

fn straddles_ko(rolls: &Option<Vec<i16>>, defender_hp: i16) -> bool {
    if defender_hp <= 0 {
        return false;
    }
    match rolls {
        Some(values) if !values.is_empty() => {
            let max_damage = values[0];
            max_damage >= defender_hp && ((max_damage as f32 * 0.85) as i16) < defender_hp
        }
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Backup: exact expectation through chance nodes
// ---------------------------------------------------------------------------

/// Replace the traversal's virtual losses with real values and back up.
///
/// `row_values` resolves `LeafPrice::Deferred` rows (empty in sequential
/// mode). At each chance node the value backed to the decision edge above is
/// the node's EXACT EXPECTATION over current branch means — never the sampled
/// branch's raw value. Returns the sample backed into the root.
pub(crate) fn finalize(tree: &mut Tree, traversal: &Traversal, row_values: &[f32]) -> f32 {
    // Resolve deferred branch prices on the expanded chance node (if any).
    if let TraversalEnd::Expanded = traversal.end {
        let step = traversal.path.last().expect("expanded traversal has a path");
        for branch in &mut tree.chances[step.chance].branches {
            if let Some(row) = branch.pending_row.take() {
                branch.value_sum += row_values[row];
            }
        }
    }
    let mut value = match traversal.end {
        TraversalEnd::Ready(v) => v,
        TraversalEnd::Row(row) => row_values[row],
        TraversalEnd::Expanded => f32::NAN, // set by the expansion step below
    };
    for (idx, step) in traversal.path.iter().enumerate().rev() {
        let is_ending_step = idx == traversal.path.len() - 1;
        if let Some(k) = step.branch {
            // Traversed branch: the deeper sample lands in its running mean.
            let branch = &mut tree.chances[step.chance].branches[k];
            debug_assert!(is_ending_step || branch.pending_row.is_none());
            branch.value_sum += value;
        } else {
            debug_assert!(is_ending_step, "expansion can only end a traversal");
        }
        // Exact-expectation resolution of the joint edge.
        let expectation = tree.chances[step.chance].expectation();
        let node = &mut tree.decisions[step.decision];
        node.s1_stats[step.i].total_value += expectation;
        node.s2_stats[step.j].total_value += expectation - 1.0; // replace virtual loss
        value = expectation;
    }
    value
}

// ---------------------------------------------------------------------------
// Sequential driver (inline leaf pricing) + Python surface
// ---------------------------------------------------------------------------

pub(crate) struct MultiPlyOutcome {
    pub tree: Tree,
    pub counters: SearchCounters,
    pub elapsed_s: f64,
}

pub(crate) fn multiply_search_with_eval<E: LeafEval>(
    state: &mut State,
    iterations: usize,
    cfg: &MultiPlyConfig,
    seed: u64,
    evaluator: &E,
) -> PyResult<MultiPlyOutcome> {
    if state.battle_is_over() != 0.0 {
        return Err(PyValueError::new_err("battle is already over at the root"));
    }
    let mut tree = Tree::from_root(state)?;
    let mut counters = SearchCounters::default();
    let mut rng = StdRng::seed_from_u64(seed);
    let start = Instant::now();
    for _ in 0..iterations {
        let traversal = traverse(
            &mut tree,
            state,
            &mut rng,
            cfg,
            &mut counters,
            &mut |leaf: &State, _seam: &BranchSeam| LeafPrice::Ready(evaluator.eval(leaf)),
        );
        finalize(&mut tree, &traversal, &[]);
    }
    Ok(MultiPlyOutcome {
        tree,
        counters,
        elapsed_s: start.elapsed().as_secs_f64(),
    })
}

pub(crate) fn multiply_report_json(
    outcome: &MultiPlyOutcome,
    iterations: usize,
    cfg: &MultiPlyConfig,
    seed: u64,
    evaluator_name: &str,
    extra_fields: &str,
) -> String {
    let root = &outcome.tree.decisions[0];
    let root_visits: u32 = root.s1_stats.iter().map(|s| s.visits).sum();
    let root_total: f32 = root.s1_stats.iter().map(|s| s.total_value).sum();
    let root_value = if root_visits > 0 {
        root_total / root_visits as f32
    } else {
        0.5
    };
    let iterations_per_s = if outcome.elapsed_s > 0.0 {
        iterations as f64 / outcome.elapsed_s
    } else {
        f64::INFINITY
    };
    format!(
        "{{\"iterations\":{},\"search\":\"multi_ply\",\"max_depth\":{},\"evaluator\":\"{}\",\
         \"c_puct\":{},\"seed\":{},\"deep_ko_split\":{},\
         \"elapsed_s\":{:.6},\"iterations_per_s\":{:.1},\
         \"leaf_evals\":{},\"expansions\":{},\"deep_ko_triggers\":{},\
         \"terminal_branches\":{},\"decision_nodes\":{},\"chance_nodes\":{},\
         \"max_depth_reached\":{},\"root_value\":{:.6}{}{}\
         ,\"side_one\":{},\"side_two\":{}}}",
        iterations,
        cfg.max_depth,
        evaluator_name,
        cfg.c_puct,
        seed,
        cfg.deep_ko_split,
        outcome.elapsed_s,
        iterations_per_s,
        outcome.counters.leaf_evals,
        outcome.counters.expansions,
        outcome.counters.deep_ko_triggers,
        outcome.counters.terminal_branches,
        outcome.tree.decisions.len(),
        outcome.tree.chances.len(),
        outcome.counters.max_depth_reached,
        root_value,
        if extra_fields.is_empty() { "" } else { "," },
        extra_fields,
        stats_to_json(&root.s1_stats),
        stats_to_json(&root.s2_stats),
    )
}

/// Multi-ply decision/chance PUCT with the trivial HP-fraction leaf evaluator
/// (`docs/crate_search_design.md`). `max_depth=1` is the one-ply regime with
/// exact-expectation chance resolution; damage-roll branching follows the
/// engine's own plies-1-2 policy, plus KO-threshold splits at deeper plies
/// while `deep_ko_split` is set. Deterministic for a fixed seed.
#[pyfunction]
#[pyo3(signature = (state_str, iterations, max_depth = 2, c_puct = 1.4, seed = 0, deep_ko_split = true))]
pub(crate) fn puct_search_multi(
    state_str: &str,
    iterations: usize,
    max_depth: u8,
    c_puct: f32,
    seed: u64,
    deep_ko_split: bool,
) -> PyResult<String> {
    if iterations == 0 {
        return Err(PyValueError::new_err("iterations must be > 0"));
    }
    if max_depth == 0 || max_depth > 32 {
        return Err(PyValueError::new_err("max_depth must be in 1..=32"));
    }
    let mut state = parse_state(state_str)?;
    let cfg = MultiPlyConfig {
        max_depth,
        c_puct,
        deep_ko_split,
    };
    let evaluator = HpFractionEval;
    let outcome = multiply_search_with_eval(&mut state, iterations, &cfg, seed, &evaluator)?;
    Ok(multiply_report_json(
        &outcome,
        iterations,
        &cfg,
        seed,
        "hp_fraction",
        "",
    ))
}

// ---------------------------------------------------------------------------
// Tests (fixture states generated by src/pokezero/poke_engine_adapter.py —
// see tests/test_multiply_chance_search.py for the Python-side gates)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Charmander (ember/tackle) vs Squirtle (watergun/tackle), 100 HP each —
    /// the crate's standard minimal fixture (`minimal_gen3_fixture`).
    const MINIMAL: &str = include_str!("test_fixtures/minimal.state");
    /// Rattata (toxic/seismictoss, faster) vs Chansey (splash only), 1v1
    /// 100 HP: analytically solvable chance structure (gen3 toxic = 85% hit
    /// applying 6 residual damage on a 100-max-HP target; seismic toss =
    /// level damage 100 = guaranteed KO).
    const ANALYTIC_TOXIC: &str = include_str!("test_fixtures/analytic_toxic.state");
    /// Rattata (splash/seismictoss, faster) vs Chansey (splash only): the
    /// win is exactly one ply past the root, so depth 2 must lift splash's
    /// value while depth 1 cannot.
    const DEPTH_BENEFIT: &str = include_str!("test_fixtures/depth_benefit.state");
    /// Rattata (splash/tackle, faster) vs Chansey (splash/tackle) at 50/100
    /// HP: tackle's damage rolls (max 52, min 44) straddle Chansey's HP, so
    /// KO-threshold splits are reachable at every ply.
    const STRADDLE: &str = include_str!("test_fixtures/straddle.state");
    /// `minimal.state` with side two replaced by a verbatim copy of side one:
    /// the two seats are IDENTICAL (same species, moves, stats, HP, speed), so
    /// the position is exactly mirror-symmetric. Used by the depth-parity
    /// pins below — see `depth_parity_invariance_on_a_mirrored_position`.
    const SYMMETRIC: &str = include_str!("test_fixtures/symmetric.state");

    /// Swap the two seats of a serialized engine state. The wire format is
    /// `side_one/side_two/weather/...`, so mirroring a position is a swap of
    /// the first two fields — a construction-level mirror, not an engine
    /// round trip.
    fn mirrored(state_str: &str) -> String {
        let mut parts: Vec<&str> = state_str.trim().split('/').collect();
        assert!(parts.len() >= 2, "state has both side fields");
        parts.swap(0, 1);
        parts.join("/")
    }

    fn run(
        state_str: &str,
        iterations: usize,
        max_depth: u8,
        seed: u64,
        deep_ko_split: bool,
    ) -> MultiPlyOutcome {
        let mut state = parse_state(state_str.trim()).expect("fixture state parses");
        let cfg = MultiPlyConfig {
            max_depth,
            c_puct: 1.4,
            deep_ko_split,
        };
        multiply_search_with_eval(&mut state, iterations, &cfg, seed, &HpFractionEval)
            .expect("search runs")
    }

    fn arm_q(outcome: &MultiPlyOutcome, display: &str) -> f32 {
        let root = &outcome.tree.decisions[0];
        let stat = root
            .s1_stats
            .iter()
            .find(|s| s.display == display)
            .unwrap_or_else(|| panic!("no side-one arm {display}"));
        assert!(stat.visits > 0, "arm {display} was never visited");
        stat.mean()
    }

    fn side_one_argmax(outcome: &MultiPlyOutcome) -> String {
        let root = &outcome.tree.decisions[0];
        root.s1_stats
            .iter()
            .max_by_key(|s| s.visits)
            .expect("root has arms")
            .display
            .clone()
    }

    #[test]
    fn root_visit_lock_is_strict_about_remaining_simulations() {
        let state = parse_state(MINIMAL.trim()).expect("fixture parses");
        let mut tree = Tree::from_root(&state).expect("root builds");
        assert!(tree.decisions[0].s1_stats.len() >= 2);
        tree.decisions[0].s1_stats[0].visits = 70;
        tree.decisions[0].s1_stats[1].visits = 10;
        tree.decisions[0].s2_stats[0].visits = 15;
        tree.decisions[0].s2_stats[1].visits = 65;

        assert_eq!(root_visit_lock(&tree, true, 59), Some((true, 70, 10)));
        assert_eq!(root_visit_lock(&tree, true, 60), Some((false, 70, 10)));
        assert_eq!(root_visit_lock(&tree, false, 49), Some((true, 65, 15)));
        assert_eq!(root_visit_lock(&tree, false, 50), Some((false, 65, 15)));
    }

    /// (a) Analytical fixture: the root edge value equals the hand-computed
    /// exact expectation over the engine's enumerated outcomes.
    #[test]
    fn analytic_expectation_depth1() {
        let outcome = run(ANALYTIC_TOXIC, 400, 1, 0, true);
        // Hand math (gen3, HpFractionEval): toxic hits 85% for 6 residual
        // damage on a 100/100 target -> value 0.5 + 0.5 * (1 - 94/100);
        // miss 15% -> 0.5. Exact expectation:
        let hit = 0.5 + 0.5 * (1.0 - 94.0 / 100.0_f32);
        let expected_toxic = 0.85 * hit + 0.15 * 0.5;
        let q_toxic = arm_q(&outcome, "toxic");
        assert!(
            (q_toxic - expected_toxic).abs() < 1e-4,
            "toxic Q {q_toxic} != analytic expectation {expected_toxic}"
        );
        // Seismic toss (level damage 100) always KOs the last opposing mon:
        // a single terminal branch of exact value 1.
        let q_toss = arm_q(&outcome, "seismictoss");
        assert!(
            (q_toss - 1.0).abs() < 1e-6,
            "seismictoss Q {q_toss} != terminal 1.0"
        );
        assert_eq!(side_one_argmax(&outcome), "seismictoss");
        // Depth 1: the tree never grows past the root decision node.
        assert_eq!(outcome.tree.decisions.len(), 1);
        assert_eq!(outcome.counters.max_depth_reached, 0);
    }

    /// The exact expectation holds at depth 2 as well: terminal branches stay
    /// exact and the toxic edge can only improve once its subtree sees the
    /// guaranteed seismic-toss KO a ply later.
    #[test]
    fn analytic_terminal_stable_at_depth2() {
        let outcome = run(ANALYTIC_TOXIC, 2_000, 2, 0, true);
        let q_toss = arm_q(&outcome, "seismictoss");
        assert!((q_toss - 1.0).abs() < 1e-6);
        let hit = 0.5 + 0.5 * (1.0 - 94.0 / 100.0_f32);
        let depth1_toxic = 0.85 * hit + 0.15 * 0.5;
        assert!(
            arm_q(&outcome, "toxic") > depth1_toxic,
            "depth-2 toxic Q should exceed its one-ply expectation"
        );
    }

    /// Depth benefit: a win exactly one ply past the root lifts the passive
    /// arm's value at depth 2; at depth 1 it stays at the leaf estimate.
    #[test]
    fn depth2_sees_one_ply_ahead() {
        let shallow = run(DEPTH_BENEFIT, 600, 1, 0, true);
        let q_splash_d1 = arm_q(&shallow, "splash");
        assert!(
            (q_splash_d1 - 0.5).abs() < 1e-4,
            "depth-1 splash Q {q_splash_d1} != HP-fraction 0.5"
        );
        let deep = run(DEPTH_BENEFIT, 600, 2, 0, true);
        let q_splash_d2 = arm_q(&deep, "splash");
        assert!(
            q_splash_d2 > 0.8,
            "depth-2 splash Q {q_splash_d2} should approach the KO next ply"
        );
    }

    /// (c) Determinism: identical seeds give identical trees and stats.
    #[test]
    fn deterministic_for_fixed_seed() {
        let a = run(STRADDLE, 3_000, 3, 11, true);
        let b = run(STRADDLE, 3_000, 3, 11, true);
        assert_eq!(a.tree.decisions.len(), b.tree.decisions.len());
        assert_eq!(a.tree.chances.len(), b.tree.chances.len());
        for (x, y) in a.tree.decisions[0]
            .s1_stats
            .iter()
            .zip(&b.tree.decisions[0].s1_stats)
        {
            assert_eq!(x.visits, y.visits);
            assert_eq!(x.total_value.to_bits(), y.total_value.to_bits());
        }
        for (x, y) in a.tree.decisions[0]
            .s2_stats
            .iter()
            .zip(&b.tree.decisions[0].s2_stats)
        {
            assert_eq!(x.visits, y.visits);
            assert_eq!(x.total_value.to_bits(), y.total_value.to_bits());
        }
    }

    /// (d) Probability conservation at every chance node (also enforced by
    /// `debug_assert`s during every expansion/backup in this debug build).
    #[test]
    fn probability_conservation_across_tree() {
        let outcome = run(STRADDLE, 3_000, 3, 0, true);
        assert!(!outcome.tree.chances.is_empty());
        for chance in &outcome.tree.chances {
            let total: f32 = chance.branches.iter().map(|b| b.probability).sum();
            assert!(
                (total - 1.0).abs() < 1e-4,
                "chance node probability mass {total} != 1"
            );
        }
    }

    /// Deep KO-threshold splits: past the engine's plies-1-2 horizon the
    /// straddle detector must enable the engine's exact KO split when (and
    /// only when) `deep_ko_split` is set.
    #[test]
    fn deep_ko_split_toggle() {
        let with_split = run(STRADDLE, 3_000, 3, 0, true);
        assert!(
            with_split.counters.deep_ko_triggers > 0,
            "straddle fixture at depth 3 must trigger deep KO splits"
        );
        let without = run(STRADDLE, 3_000, 3, 0, false);
        assert_eq!(without.counters.deep_ko_triggers, 0);
    }

    /// (b) Depth=1 regression against the one-ply core: identical option
    /// surfaces and the same argmax on the standard minimal fixture (the
    /// semantic difference — exact expectation instead of sampled-branch
    /// backup — must not move the decision).
    #[test]
    fn depth1_matches_oneply_argmax() {
        let mut state = parse_state(MINIMAL.trim()).expect("fixture parses");
        let (s1_old, s2_old, _) =
            crate::puct_search_with_eval(&mut state, 2_000, 1.4, 0, &HpFractionEval)
                .expect("one-ply search runs");
        let outcome = run(MINIMAL, 2_000, 1, 0, true);
        let root = &outcome.tree.decisions[0];
        let old_moves: Vec<&str> = s1_old.iter().map(|s| s.display.as_str()).collect();
        let new_moves: Vec<&str> = root.s1_stats.iter().map(|s| s.display.as_str()).collect();
        assert_eq!(old_moves, new_moves);
        assert_eq!(
            s2_old.iter().map(|s| s.display.as_str()).collect::<Vec<_>>(),
            root.s2_stats.iter().map(|s| s.display.as_str()).collect::<Vec<_>>(),
        );
        let old_argmax = s1_old
            .iter()
            .max_by_key(|s| s.visits)
            .expect("one-ply root has arms")
            .display
            .clone();
        assert_eq!(old_argmax, side_one_argmax(&outcome));
        // Visit conservation, as in the one-ply report contract.
        let visits: u32 = root.s1_stats.iter().map(|s| s.visits).sum();
        assert_eq!(visits, 2_000);
    }

    /// Model priors reweight EXPLORATION only, never values: on the analytic
    /// fixture, skewing the root priors hard toward toxic moves visits toward
    /// it but leaves both arms' Q at their exact analytic values — the
    /// guaranteed seismic-toss KO still reads exactly 1.0.
    #[test]
    fn priors_reweight_exploration_not_values() {
        fn run_with_priors(skew: bool) -> (u32, u32, f32, f32) {
            let mut state = parse_state(ANALYTIC_TOXIC.trim()).expect("fixture parses");
            let cfg = MultiPlyConfig {
                max_depth: 1,
                c_puct: 1.4,
                deep_ko_split: true,
            };
            let mut tree = Tree::from_root(&state).expect("root builds");
            let toxic = tree.decisions[0]
                .s1_stats
                .iter()
                .position(|s| s.display == "toxic")
                .expect("toxic arm");
            let toss = tree.decisions[0]
                .s1_stats
                .iter()
                .position(|s| s.display == "seismictoss")
                .expect("seismictoss arm");
            if skew {
                let mut priors = vec![0.0f32; tree.decisions[0].s1_stats.len()];
                priors[toxic] = 0.99;
                priors[toss] = 0.01;
                assert!(apply_self_priors(&mut tree.decisions[0], true, &priors));
                // Length mismatch must refuse and leave priors untouched.
                assert!(!apply_self_priors(&mut tree.decisions[0], true, &[1.0]));
            }
            let mut counters = SearchCounters::default();
            let mut rng = StdRng::seed_from_u64(0);
            let evaluator = HpFractionEval;
            for _ in 0..400 {
                let traversal = traverse(
                    &mut tree,
                    &mut state,
                    &mut rng,
                    &cfg,
                    &mut counters,
                    &mut |leaf: &State, _seam: &BranchSeam| LeafPrice::Ready(evaluator.eval(leaf)),
                );
                finalize(&mut tree, &traversal, &[]);
            }
            let root = &tree.decisions[0];
            (
                root.s1_stats[toxic].visits,
                root.s1_stats[toss].visits,
                root.s1_stats[toxic].mean(),
                root.s1_stats[toss].mean(),
            )
        }
        let (toxic_uniform, _, q_toxic_uniform, q_toss_uniform) = run_with_priors(false);
        let (toxic_skewed, toss_skewed, q_toxic_skewed, q_toss_skewed) = run_with_priors(true);
        // Values: exact under both prior settings.
        let hit = 0.5 + 0.5 * (1.0 - 94.0 / 100.0_f32);
        let expected_toxic = 0.85 * hit + 0.15 * 0.5;
        for (q_toxic, q_toss) in [
            (q_toxic_uniform, q_toss_uniform),
            (q_toxic_skewed, q_toss_skewed),
        ] {
            assert!(
                (q_toxic - expected_toxic).abs() < 1e-4,
                "toxic Q {q_toxic} != analytic expectation {expected_toxic}"
            );
            assert!((q_toss - 1.0).abs() < 1e-6, "seismictoss Q {q_toss} != 1.0");
        }
        // Exploration: the high-prior arm collects strictly more visits than
        // under uniform priors; the guaranteed KO still wins the argmax.
        assert!(
            toxic_skewed > toxic_uniform,
            "prior skew must raise toxic visits ({toxic_skewed} <= {toxic_uniform})"
        );
        assert!(toss_skewed > toxic_skewed, "argmax must stay on the KO");
    }

    // -----------------------------------------------------------------
    // Value-orientation pins (docs/mcts_degradation_findings.md §10).
    //
    // These exist to keep the "ply-parity-dependent value orientation"
    // hypothesis dead. The tree's value is SIDE-ONE win probability at every
    // level: terminal branches come from `battle_is_over` (+1 = side one won),
    // backup adds the chance-node expectation to BOTH seats' stats unchanged,
    // and the only seat flip lives in `MoveStats::puct` (`1 - mean` for side
    // two). Nothing negates or re-seats per ply. Every expectation below is
    // derived from that construction or from the fixture's symmetry, never
    // from a recorded engine number.
    // -----------------------------------------------------------------

    /// A leaf value and its mirror reflect about 0.5, at any state.
    ///
    /// This is the crate-level form of "encode a mirrored state pair at a leaf
    /// and assert v01 reflects about 0.5": `HpFractionEval` is the crate's
    /// side-one-oriented leaf contract (lib.rs: `0.5 + 0.5 * (s1 - s2)`), so
    /// swapping the seats must map v -> 1 - v exactly.
    #[test]
    fn leaf_value_reflects_about_half_under_seat_mirror() {
        for fixture in [MINIMAL, ANALYTIC_TOXIC, DEPTH_BENEFIT, STRADDLE, SYMMETRIC] {
            let state = parse_state(fixture.trim()).expect("fixture parses");
            let flipped = parse_state(mirrored(fixture).trim()).expect("mirror parses");
            let v = HpFractionEval.eval(&state);
            let v_mirror = HpFractionEval.eval(&flipped);
            assert!(
                (v + v_mirror - 1.0).abs() < 1e-6,
                "leaf value {v} and its mirror {v_mirror} do not reflect about 0.5"
            );
        }
        // A perfectly mirrored position is its own mirror: exactly 0.5.
        let symmetric = parse_state(SYMMETRIC.trim()).expect("fixture parses");
        assert!((HpFractionEval.eval(&symmetric) - 0.5).abs() < 1e-6);
    }

    /// Backed-up root value on a mirror-symmetric position is 0.5 at EVERY
    /// depth, with no even/odd split.
    ///
    /// The fixture's two seats are byte-identical, so side one's exact win
    /// probability is 0.5 by symmetry, independently of the model, the depth
    /// or the engine's branch structure. A per-level sign or perspective error
    /// — the mechanism that would leave depth 1 clean and corrupt deeper plies
    /// — moves this off 0.5 in a depth-parity-dependent way. It does not.
    #[test]
    fn depth_parity_invariance_on_a_mirrored_position() {
        let mut by_depth: Vec<(u8, f32)> = Vec::new();
        for depth in 1..=6u8 {
            let outcome = run(SYMMETRIC, 4_000, depth, 20_260_729, true);
            let root = &outcome.tree.decisions[0];
            let visits: u32 = root.s1_stats.iter().map(|s| s.visits).sum();
            let total: f32 = root.s1_stats.iter().map(|s| s.total_value).sum();
            let root_value = total / visits as f32;
            by_depth.push((depth, root_value));
            assert!(
                (root_value - 0.5).abs() < 0.02,
                "depth {depth}: mirrored root value {root_value} is not 0.5"
            );
            // Both seats read the SAME side-one-oriented quantity: side two's
            // stats are not stored on a flipped or shifted scale.
            let s2_total: f32 = root.s2_stats.iter().map(|s| s.total_value).sum();
            let s2_visits: u32 = root.s2_stats.iter().map(|s| s.visits).sum();
            assert!(
                ((s2_total / s2_visits as f32) - root_value).abs() < 1e-4,
                "depth {depth}: side-two mean {} != side-one mean {root_value}",
                s2_total / s2_visits as f32
            );
            for stat in root.s1_stats.iter().chain(root.s2_stats.iter()) {
                assert!(
                    (0.0..=1.0).contains(&stat.mean()),
                    "depth {depth}: arm {} mean {} left [0, 1]",
                    stat.display,
                    stat.mean()
                );
            }
        }
        // No even/odd structure: the deviations from 0.5 must not separate by
        // depth parity (a per-ply sign error is exactly such a separation).
        let even: Vec<f32> = by_depth
            .iter()
            .filter(|(d, _)| d % 2 == 0)
            .map(|(_, v)| v - 0.5)
            .collect();
        let odd: Vec<f32> = by_depth
            .iter()
            .filter(|(d, _)| d % 2 == 1)
            .map(|(_, v)| v - 0.5)
            .collect();
        let mean = |xs: &[f32]| xs.iter().sum::<f32>() / xs.len() as f32;
        assert!(
            (mean(&even) - mean(&odd)).abs() < 0.02,
            "even-depth deviation {:?} separates from odd-depth deviation {:?}",
            mean(&even),
            mean(&odd)
        );
    }

    /// The whole tree is mirror-equivariant: mirroring the ROOT swaps the two
    /// seats' statistics and maps the root value to `1 - v`, at every depth.
    ///
    /// A per-level perspective error would break this at deep plies while
    /// leaving depth 1 intact; a seat-constant one breaks it everywhere.
    #[test]
    fn seat_mirror_maps_root_value_to_its_complement_at_every_depth() {
        let flipped_fixture = mirrored(MINIMAL);
        for depth in 1..=4u8 {
            let straight = run(MINIMAL, 3_000, depth, 4, true);
            let flipped = run(&flipped_fixture, 3_000, depth, 4, true);
            let value = |outcome: &MultiPlyOutcome| {
                let root = &outcome.tree.decisions[0];
                let visits: u32 = root.s1_stats.iter().map(|s| s.visits).sum();
                root.s1_stats.iter().map(|s| s.total_value).sum::<f32>() / visits as f32
            };
            let v = value(&straight);
            let v_mirror = value(&flipped);
            assert!(
                (v + v_mirror - 1.0).abs() < 0.03,
                "depth {depth}: root value {v} and mirrored root value {v_mirror} \
                 do not reflect about 0.5"
            );
        }
    }

    /// The `- 1.0` on `s2_stats` in `finalize` is virtual-loss REPLACEMENT, not
    /// a seat convention: over one traversal side two's arm accumulates exactly
    /// the chance-node expectation, the same [0, 1] quantity side one's arm
    /// gets. (Q ranges are therefore seat-symmetric and `c_puct` weighs the
    /// exploration term against the same scale on both sides.)
    #[test]
    fn side_two_stats_accumulate_the_same_expectation_as_side_one() {
        let mut state = parse_state(STRADDLE.trim()).expect("fixture parses");
        let cfg = MultiPlyConfig {
            max_depth: 3,
            c_puct: 1.4,
            deep_ko_split: true,
        };
        let mut tree = Tree::from_root(&state).expect("root builds");
        let mut counters = SearchCounters::default();
        let mut rng = StdRng::seed_from_u64(99);
        let evaluator = HpFractionEval;
        for _ in 0..500 {
            let traversal = traverse(
                &mut tree,
                &mut state,
                &mut rng,
                &cfg,
                &mut counters,
                &mut |leaf: &State, _seam: &BranchSeam| LeafPrice::Ready(evaluator.eval(leaf)),
            );
            let backed = finalize(&mut tree, &traversal, &[]);
            assert!(
                (0.0..=1.0).contains(&backed),
                "backed-up root sample {backed} left [0, 1]"
            );
        }
        for node in &tree.decisions {
            for stat in node.s1_stats.iter().chain(node.s2_stats.iter()) {
                assert!(
                    (0.0..=1.0).contains(&stat.mean()),
                    "arm {} mean {} left [0, 1] — side two is NOT on a [-1, 0] scale",
                    stat.display,
                    stat.mean()
                );
            }
        }
    }

    /// Terminal orientation is absolute, not relative to whoever is acting: a
    /// guaranteed side-TWO win prices at 0.0, the exact complement of the
    /// side-one KO that `analytic_expectation_depth1` pins at 1.0.
    #[test]
    fn terminal_orientation_is_absolute_across_the_seat_mirror() {
        let flipped = mirrored(ANALYTIC_TOXIC);
        let outcome = run(&flipped, 400, 1, 0, true);
        let root = &outcome.tree.decisions[0];
        let toss = root
            .s2_stats
            .iter()
            .find(|s| s.display == "seismictoss")
            .expect("mirrored fixture puts seismictoss on side two");
        assert!(toss.visits > 0);
        assert!(
            toss.mean().abs() < 1e-6,
            "side-two guaranteed KO priced at {} (side-one win probability must be 0.0)",
            toss.mean()
        );
    }

    /// Terminal branches never grow children and keep their exact value.
    #[test]
    fn terminal_branches_stay_exact() {
        let outcome = run(ANALYTIC_TOXIC, 3_000, 4, 3, true);
        for chance in &outcome.tree.chances {
            for branch in &chance.branches {
                if let Some(v) = branch.terminal {
                    assert!(branch.child.is_none());
                    assert!((branch.mean() - v).abs() < 1e-6);
                }
            }
        }
    }
}
