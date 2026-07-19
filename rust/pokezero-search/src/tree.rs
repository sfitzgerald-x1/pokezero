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
                let chance_idx =
                    expand_edge(tree, state, node_idx, i, j, cfg, counters, price);
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
) -> usize {
    counters.expansions += 1;
    let depth = tree.decisions[node_idx].depth;
    let node = &tree.decisions[node_idx];
    let s1_move = node.s1_options[i];
    let s2_move = node.s2_options[j];

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
        });
    } else {
        let total: f32 = generated.iter().map(|b| b.percentage).sum();
        debug_assert!(
            (total - 100.0).abs() < PERCENT_SUM_TOL,
            "engine branch percentages sum to {total}, expected 100"
        );
        let norm = if total > 0.0 { total } else { 100.0 };
        for state_instructions in generated {
            let probability = state_instructions.percentage / norm;
            let instructions = state_instructions.instruction_list;
            state.apply_instructions(&instructions);
            let outcome = state.battle_is_over();
            let seam = BranchSeam {
                s1: &s1_move,
                s2: &s2_move,
                instructions: &instructions,
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
