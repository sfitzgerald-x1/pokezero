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
    /// Same, for the NON-acting (opponent) seat's options at this branch's
    /// child, from the model's opponent action head. Written only when
    /// `MultiPlyConfig::use_opponent_priors` is set; `None` keeps that seat
    /// uniform, which is the historical behaviour.
    ///
    /// Stored SEPARATELY from `child_self_priors` rather than replacing it:
    /// the two describe different seats' arms and are applied to different
    /// stat vectors. Sharing one field would force a reflection at exactly
    /// the boundary #937 warns about.
    pub child_opponent_priors: Option<(bool, Vec<f32>)>,
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
        self.branches.iter().map(|b| b.probability * b.mean()).sum()
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
    /// Seed the OPPONENT seat's PUCT priors from the model's opponent action
    /// head instead of leaving them uniform. Default OFF: with it off the
    /// search is behaviourally identical to the uniform-opponent design that
    /// every recorded result was produced under.
    pub use_opponent_priors: bool,
    /// First-play-urgency reduction for UNVISITED arms (`crate::fpu_value`).
    /// `None` is the legacy flat 0.5 that `MoveStats::mean` returns at zero
    /// visits — the value every recorded result was produced under, and the
    /// setting the bit-identity gate is asserted at. `Some(r)` prices an
    /// unvisited arm at `clamp(parent mean in that seat's frame - r, 0, 1)`.
    pub fpu_reduction: Option<f32>,
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
// Within-batch selection collisions
// ---------------------------------------------------------------------------

/// How often a batched round's selections land where the SAME round has already
/// been, kept PER SEAT.
///
/// Virtual loss is the only thing keeping a 64-selection round spread out, so
/// the rate at which it fails to is the batch axis's own mechanism metric — and
/// it is the only instrument that can test the deferred-leaf theory (the
/// selection-tuning plan's finding 3, explicitly unverified).
///
/// The provisionals a round leaves behind are SIDE-ONE ABSOLUTE. A deferred leaf
/// is priced `(value_sum 0.0, visits 1)` and a traversed branch takes a bare
/// `visits += 1`; both drag a chance node's expectation toward 0 until the
/// owning traversal's `finalize` reconciles them, and `finalize` runs in
/// collection order, so an early traversal's backup reads the depression a later
/// one has not paid off yet. That expectation reaches the side-one arm as itself
/// (near 0: a loss, as intended) and the side-two arm as `expectation - 1.0`,
/// which that seat reads back through `1 - mean` as a WIN. So the same
/// unreconciled placeholder that repels one seat may ATTRACT the other — and a
/// single pooled repeat count cannot see it, because the two effects land in one
/// total. Hence a tally per seat, not one tally.
///
/// NOTE the asymmetry is SIDE-absolute, not seat-relative: it falls on
/// `s2_stats` whichever seat is searching. Consumers that want the self/opponent
/// view must condition on which side the searching seat sat; the encoded core
/// ships that alongside these counts.
///
/// Recorded from the FINISHED [`Traversal`], never from inside `traverse`: the
/// path is already there, so the hot selection loop is not touched at all and
/// the counter cannot perturb what it measures. The four sets are allocated once
/// per search and `clear`ed per round, so steady state allocates nothing either.
#[derive(Default)]
#[cfg_attr(not(feature = "model"), allow(dead_code))]
pub(crate) struct CollisionLedger {
    /// (decision node, i, j) joint cells this round has already taken.
    joint_seen: std::collections::HashSet<(usize, usize, usize)>,
    /// (decision node, arm), per seat.
    side_one_seen: std::collections::HashSet<(usize, usize)>,
    side_two_seen: std::collections::HashSet<(usize, usize)>,
    /// (chance node, descended branch) a traversal has already bottomed out on.
    leaf_seen: std::collections::HashSet<(usize, Option<usize>)>,
    pub counts: CollisionCounts,
}

/// Whole-search totals over every round's [`CollisionLedger`]. Rates are left to
/// the consumer: the denominators are here so a caller pooling several worlds
/// into one decision can pool numerator and denominator together instead of
/// averaging ratios.
#[derive(Default, Clone, Copy)]
#[cfg_attr(not(feature = "model"), allow(dead_code))]
pub(crate) struct CollisionCounts {
    pub rounds: usize,
    /// Rounds that actually deferred at least one leaf. The deferred-leaf theory
    /// is a claim about PENDING batches specifically, so the audit needs to know
    /// what share of rounds could exhibit it; a round that expanded nothing has
    /// no placeholder in it to be asymmetric.
    pub pending_rounds: usize,
    /// Decision-node visits recorded — the denominator for the three
    /// per-selection repeat counts below (one traversal contributes one per ply
    /// it descended).
    pub selections: usize,
    pub joint_repeats: usize,
    pub side_one_repeats: usize,
    pub side_two_repeats: usize,
    /// Traversals recorded — the denominator for `leaf_repeats`.
    pub traversals: usize,
    pub leaf_repeats: usize,
}

#[cfg_attr(not(feature = "model"), allow(dead_code))]
impl CollisionLedger {
    /// Forget the previous round's cells, keeping their allocation.
    pub(crate) fn begin_round(&mut self) {
        self.joint_seen.clear();
        self.side_one_seen.clear();
        self.side_two_seen.clear();
        self.leaf_seen.clear();
    }

    /// Fold one collected traversal into this round's cells.
    pub(crate) fn record(&mut self, traversal: &Traversal) {
        for step in &traversal.path {
            self.counts.selections += 1;
            if !self.joint_seen.insert((step.decision, step.i, step.j)) {
                self.counts.joint_repeats += 1;
            }
            if !self.side_one_seen.insert((step.decision, step.i)) {
                self.counts.side_one_repeats += 1;
            }
            if !self.side_two_seen.insert((step.decision, step.j)) {
                self.counts.side_two_repeats += 1;
            }
        }
        self.counts.traversals += 1;
        if let Some(step) = traversal.path.last() {
            // An EXPANSION can never repeat — `traverse` inserts the edge into
            // `children` before the round's next selection, so no second
            // traversal can expand it — but it is still recorded, under its own
            // fresh chance index, so `traversals` stays the honest denominator
            // rather than one that quietly excludes the productive selections.
            if !self.leaf_seen.insert((step.chance, step.branch)) {
                self.counts.leaf_repeats += 1;
            }
        }
    }

    /// Close a round. `pending_rows` is that round's deferred-leaf count, which
    /// is what makes the round eligible for the deferred-leaf reading.
    pub(crate) fn end_round(&mut self, pending_rows: usize) {
        self.counts.rounds += 1;
        if pending_rows > 0 {
            self.counts.pending_rounds += 1;
        }
    }
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

impl BranchSeam<'_> {
    /// Keep unsafe renderer branches out of the tree's fold/encoder seam.
    /// Returning an error aborts the entire native world so the Python caller
    /// takes its established whole-world fallback instead of dropping one
    /// chance outcome from an expectation.
    #[cfg_attr(not(feature = "model"), allow(dead_code))]
    pub(crate) fn reject_attribution_unsafe(
        &self,
        rendered: &crate::events::RenderedEvents,
    ) -> PyResult<()> {
        crate::events::reject_attribution_unsafe(rendered, "tree/model fold")
    }
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
            let i = select(
                &node.s1_stats,
                node.visits,
                cfg.c_puct,
                true,
                cfg.fpu_reduction,
            );
            let j = select(
                &node.s2_stats,
                node.visits,
                cfg.c_puct,
                false,
                cfg.fpu_reduction,
            );
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
                let parent = path.last().map(|step| {
                    (
                        step.chance,
                        step.branch.expect("descended steps carry a branch"),
                    )
                });
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
                        apply_branch_child_priors(tree, chance_idx, k, child);
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

/// Apply whatever model priors the encoded core parked on a chance branch to
/// the child decision node just created under it.
///
/// Both stored vectors carry the side flag of the seat that OWNS their arms —
/// `child_self_priors` the acting seat, `child_opponent_priors` the other one —
/// so BOTH are applied verbatim to that seat's stat vector. There is no
/// negation of the flag and no `1 - p` on the values: reflection at the seat
/// boundary applies to VALUES only, and happens in
/// `multiply_batched_encoded_core`, not here. Branches with no stored priors
/// keep the uniform priors `make_stats` seeded (the historical behaviour, and
/// what a same-round pending eval leaves behind until the core re-applies it).
pub(crate) fn apply_branch_child_priors(
    tree: &mut Tree,
    chance_idx: usize,
    branch_idx: usize,
    child: usize,
) {
    if let Some((side_one, priors)) = tree.chances[chance_idx].branches[branch_idx]
        .child_self_priors
        .clone()
    {
        apply_self_priors(&mut tree.decisions[child], side_one, &priors);
    }
    if let Some((side_one, priors)) = tree.chances[chance_idx].branches[branch_idx]
        .child_opponent_priors
        .clone()
    {
        apply_self_priors(&mut tree.decisions[child], side_one, &priors);
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
child_opponent_priors: None,
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
child_opponent_priors: None,
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
    let s1_first =
        state.side_one.get_active_immutable().speed >= state.side_two.get_active_immutable().speed;
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
        let step = traversal
            .path
            .last()
            .expect("expanded traversal has a path");
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

/// `arm_priors` is threaded rather than read off `cfg`: it changes only what the
/// report SAYS, and putting it in `MultiPlyConfig` would put a reporting knob in
/// the struct whose fields are the search's semantics -- where a future
/// `cfg`-derived cache key or equality check would treat two identical searches
/// as different ones.
pub(crate) fn multiply_report_json(
    outcome: &MultiPlyOutcome,
    iterations: usize,
    cfg: &MultiPlyConfig,
    seed: u64,
    evaluator_name: &str,
    extra_fields: &str,
    arm_priors: bool,
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
    // Emitted only when SET, so a legacy run's report keeps the bytes it has
    // always had (the bit-identity gate reads these reports) while a tuned run
    // still carries the knob it was tuned at. A field spelled `null` on every
    // legacy report would have made every one of them a new artifact.
    let fpu_field = match cfg.fpu_reduction {
        Some(r) => format!(",\"fpu_reduction\":{r}"),
        None => String::new(),
    };
    format!(
        "{{\"iterations\":{},\"search\":\"multi_ply\",\"max_depth\":{},\"evaluator\":\"{}\",\
         \"c_puct\":{},\"seed\":{},\"deep_ko_split\":{}{},\
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
        fpu_field,
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
        stats_to_json(&root.s1_stats, arm_priors),
        stats_to_json(&root.s2_stats, arm_priors),
    )
}

/// Reject an out-of-range first-play-urgency reduction at the Python boundary.
///
/// Q lives in `[0, 1]` here (the value head is a win probability), so a
/// reduction outside it cannot mean anything the clamp would not already have
/// flattened, and a NEGATIVE one is a first-play BONUS — the opposite of the
/// mechanism, silently. Same standing as the `max_depth` bound above it.
pub(crate) fn validate_fpu_reduction(fpu_reduction: Option<f32>) -> PyResult<Option<f32>> {
    match fpu_reduction {
        Some(r) if !(0.0..=1.0).contains(&r) => Err(PyValueError::new_err(format!(
            "fpu_reduction must be in 0.0..=1.0, got {r}"
        ))),
        other => Ok(other),
    }
}

/// Multi-ply decision/chance PUCT with the trivial HP-fraction leaf evaluator
/// (`docs/crate_search_design.md`). `max_depth=1` is the one-ply regime with
/// exact-expectation chance resolution; damage-roll branching follows the
/// engine's own plies-1-2 policy, plus KO-threshold splits at deeper plies
/// while `deep_ko_split` is set. Deterministic for a fixed seed.
#[pyfunction]
#[pyo3(signature = (
    state_str,
    iterations,
    max_depth = 2,
    c_puct = 1.4,
    seed = 0,
    deep_ko_split = true,
    // Default None, and last in the signature, so every existing caller is
    // byte-for-byte unaffected: flag-off must be the flat-0.5 first-play
    // urgency every recorded result was produced under.
    fpu_reduction = None,
))]
pub(crate) fn puct_search_multi(
    state_str: &str,
    iterations: usize,
    max_depth: u8,
    c_puct: f32,
    seed: u64,
    deep_ko_split: bool,
    fpu_reduction: Option<f32>,
) -> PyResult<String> {
    if iterations == 0 {
        return Err(PyValueError::new_err("iterations must be > 0"));
    }
    if max_depth == 0 || max_depth > 32 {
        return Err(PyValueError::new_err("max_depth must be in 1..=32"));
    }
    let fpu_reduction = validate_fpu_reduction(fpu_reduction)?;
    let mut state = parse_state(state_str)?;
    let cfg = MultiPlyConfig {
        max_depth,
        c_puct,
        deep_ko_split,
        // No model in this core, so there is no opponent head to gather.
        use_opponent_priors: false,
        fpu_reduction,
    };
    let evaluator = HpFractionEval;
    // Contain poke-engine's panics here too. This is the hp_fraction path, and
    // its caller (engine_search.py, the `crate_search_hp` handler) wraps it in
    // `except Exception` with the same "count the bad world, keep the others"
    // intent as the model path -- an intent a PanicException defeats, because it
    // derives from BaseException. Review demonstrated the real
    // `Invalid rest_turns value: 32` panic firing through THIS function, so
    // guarding only the model entry point left the identical hole open.
    let outcome = crate::panic_guard::catch_native_panic(|| {
        multiply_search_with_eval(&mut state, iterations, &cfg, seed, &evaluator)
    })?;
    Ok(multiply_report_json(
        &outcome,
        iterations,
        &cfg,
        seed,
        "hp_fraction",
        "",
        // Uniform by construction on this path -- there is no model to price the
        // arms -- so the column would be a constant 1/n on every entry.
        false,
    ))
}

// ---------------------------------------------------------------------------
// Tests (fixture states generated by src/pokezero/poke_engine_adapter.py —
// see tests/test_multiply_chance_search.py for the Python-side gates)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Model-prior APPLY: seat routing, arity refusal, and the guarantee
    // that priors reweight exploration WITHOUT touching values.
    // Gather-side coverage lives in `crate::priors`.
    // -----------------------------------------------------------------

    fn prior_stats(priors: &[f32]) -> Vec<MoveStats> {
        priors
            .iter()
            .enumerate()
            .map(|(i, prior)| MoveStats {
                display: format!("arm{i}"),
                prior: *prior,
                visits: 0,
                total_value: 0.0,
            })
            .collect()
    }

    fn prior_node(s1: &[f32], s2: &[f32]) -> DecisionNode {
        DecisionNode {
            visits: 0,
            depth: 1,
            s1_options: vec![MoveChoice::None; s1.len()],
            s2_options: vec![MoveChoice::None; s2.len()],
            s1_stats: prior_stats(s1),
            s2_stats: prior_stats(s2),
            children: HashMap::new(),
        }
    }

    fn priors_of(stats: &[MoveStats]) -> Vec<f32> {
        stats.iter().map(|s| s.prior).collect()
    }

    #[test]
    fn apply_self_priors_writes_exactly_one_seat() {
        let mut node = prior_node(&[0.5, 0.5], &[0.5, 0.5]);
        assert!(apply_self_priors(&mut node, true, &[0.8, 0.2]));
        assert_eq!(priors_of(&node.s1_stats), vec![0.8, 0.2]);
        assert_eq!(
            priors_of(&node.s2_stats),
            vec![0.5, 0.5],
            "side one's priors must not leak onto side two"
        );

        let mut node = prior_node(&[0.5, 0.5], &[0.5, 0.5]);
        assert!(apply_self_priors(&mut node, false, &[0.8, 0.2]));
        assert_eq!(priors_of(&node.s2_stats), vec![0.8, 0.2]);
        assert_eq!(priors_of(&node.s1_stats), vec![0.5, 0.5]);
    }

    #[test]
    fn apply_self_priors_refuses_an_arity_mismatch_and_changes_nothing() {
        let mut node = prior_node(&[0.5, 0.5], &[0.5, 0.5]);
        assert!(!apply_self_priors(&mut node, true, &[0.4, 0.4, 0.2]));
        assert_eq!(priors_of(&node.s1_stats), vec![0.5, 0.5]);
        assert!(!apply_self_priors(&mut node, true, &[1.0]));
        assert_eq!(priors_of(&node.s1_stats), vec![0.5, 0.5]);
        // A partial write (zip stops at the shorter side) would leave 0.4 in
        // arm 0 above; assert the full vector, not just the length.
    }

    /// Priors reweight EXPLORATION only. A prior write that also disturbed
    /// visits or accumulated value would corrupt the backup, and the
    /// exact-expectation contract is the crate's whole quality story.
    #[test]
    fn apply_self_priors_leaves_visits_and_values_alone() {
        let mut node = prior_node(&[0.5, 0.5], &[0.5, 0.5]);
        node.s1_stats[0].visits = 7;
        node.s1_stats[0].total_value = 3.5;
        node.visits = 9;
        assert!(apply_self_priors(&mut node, true, &[0.9, 0.1]));
        assert_eq!(node.s1_stats[0].visits, 7);
        assert_eq!(node.s1_stats[0].total_value, 3.5);
        assert_eq!(node.s1_stats[0].mean(), 0.5);
        assert_eq!(node.visits, 9);
    }

    /// The applied prior has to actually STEER selection, otherwise the whole
    /// opponent-priors path could be wired correctly and inert. Same stats,
    /// same visits, only the prior differs.
    #[test]
    fn an_applied_prior_changes_which_arm_puct_picks() {
        let mut node = prior_node(&[0.5, 0.5], &[0.5, 0.5]);
        assert_eq!(
            crate::select(&node.s2_stats, 16, 1.4, false, None),
            0,
            "uniform priors leave the first arm winning the tie"
        );
        assert!(apply_self_priors(&mut node, false, &[0.1, 0.9]));
        assert_eq!(crate::select(&node.s2_stats, 16, 1.4, false, None), 1);
    }

    // -----------------------------------------------------------------
    // First-play urgency (`MultiPlyConfig::fpu_reduction`)
    // -----------------------------------------------------------------

    /// `(prior, visits, total_value)` per arm. `total_value` is SIDE-ONE
    /// ABSOLUTE in both seats' vectors — that is the tree's convention
    /// (`finalize` adds the same chance expectation to both) and the whole point
    /// of the seat-frame tests below.
    fn valued_stats(arms: &[(f32, u32, f32)]) -> Vec<MoveStats> {
        arms.iter()
            .enumerate()
            .map(|(i, (prior, visits, total_value))| MoveStats {
                display: format!("arm{i}"),
                prior: *prior,
                visits: *visits,
                total_value: *total_value,
            })
            .collect()
    }

    /// The flag has to change WHICH ARM IS SELECTED, not merely which number is
    /// stored — a reduction that never reaches the argmax is the failure this
    /// programme keeps hitting.
    ///
    /// Constructed so the two settings disagree: one visited arm at Q 0.2 (16
    /// visits, prior 0.99) and one unvisited arm at prior 0.01.
    ///   * legacy: unvisited Q = 0.5, score 0.5 + 1.4*0.01*4/1 = 0.556, and the
    ///     visited arm scores 0.2 + 1.4*0.99*4/17 = 0.526 — the untried arm wins
    ///     purely on the flat 0.5, which is the defect the plan names;
    ///   * `Some(0.2)`: unvisited Q = clamp(0.2 - 0.2) = 0.0, score 0.056 — the
    ///     visited arm keeps the node.
    ///
    /// The SKEWED PRIOR is load-bearing and is itself the mechanism's shape:
    /// PUCT's `u` term for an unvisited arm is `c_puct * prior * sqrt(N)`, which
    /// grows without bound in `N`, so under UNIFORM priors every unvisited
    /// sibling is selected eventually no matter what Q it is priced at. FPU can
    /// only bite where the policy head has concentrated mass — which is exactly
    /// the regime the opponent-priors stage puts the search into.
    #[test]
    fn fpu_reduction_prices_an_unvisited_arm_off_the_parent_mean() {
        let stats = valued_stats(&[(0.99, 16, 3.2), (0.01, 0, 0.0)]);
        assert_eq!(
            crate::select(&stats, 16, 1.4, true, None),
            1,
            "flat 0.5 first-play urgency must hand the node to the untried arm"
        );
        assert_eq!(
            crate::select(&stats, 16, 1.4, true, Some(0.2)),
            0,
            "an unvisited arm priced at the parent mean minus 0.2 must lose"
        );
    }

    /// THE SEAT-FRAME PIN. Selection is decoupled per-side PUCT and side two
    /// scores on `1 - value`, so the reduction has to be subtracted AFTER the
    /// reflection, in the seat's own frame.
    ///
    /// Mirror of the test above: the visited arm holds side-one-absolute mean
    /// 0.8, which side two reads as 0.2, and the arithmetic is then identical —
    /// correct frame gives the unvisited arm `clamp(0.2 - 0.2) = 0.0` and the
    /// visited arm keeps the node.
    ///
    /// A build that subtracted in the side-ONE frame would price the unvisited
    /// arm at `clamp(0.8 - 0.2) = 0.6`, i.e. HIGHER than the legacy 0.5 it
    /// replaces: the reduction would arrive at this seat as a first-play BONUS
    /// and the untried arm would win by MORE than before the flag existed. That
    /// mutant scores 0.656 against the visited arm's 0.526 and fails here.
    #[test]
    fn fpu_reduction_uses_the_selecting_seats_own_frame() {
        let stats = valued_stats(&[(0.99, 16, 12.8), (0.01, 0, 0.0)]);
        assert_eq!(
            crate::select(&stats, 16, 1.4, false, None),
            1,
            "flat 0.5 first-play urgency must hand the node to the untried arm"
        );
        assert_eq!(
            crate::select(&stats, 16, 1.4, false, Some(0.2)),
            0,
            "side two's baseline is 1 - 0.8 = 0.2; the side-one frame would read 0.8"
        );
    }

    /// A node with no visits has no running value to reduce, and the flag is a
    /// documented no-op there: every arm is unvisited, every `u` term is
    /// `c_puct * prior * sqrt(0) / 1 == 0`, so the argmax is index 0 for any
    /// constant Q. Pinned because the alternative — inventing a baseline for a
    /// node that has none — would make the flag's FIRST touch on every node
    /// depend on a number the node never produced.
    #[test]
    fn fpu_reduction_is_a_no_op_at_a_node_with_no_visits() {
        let stats = valued_stats(&[(0.1, 0, 0.0), (0.9, 0, 0.0)]);
        for reduction in [None, Some(0.0), Some(0.3), Some(1.0)] {
            assert_eq!(crate::select(&stats, 0, 1.4, true, reduction), 0);
            assert_eq!(crate::select(&stats, 0, 1.4, false, reduction), 0);
        }
    }

    /// End to end through `MultiPlyConfig`, not just `select`: the knob has to
    /// survive the config, the traversal and the backup and land in the root's
    /// visit distribution. Same fixture, same seed, same budget — only the flag
    /// differs.
    ///
    /// This is the companion to the bit-identity gate. That gate proves `None`
    /// changes nothing; without this one, a flag that was silently dropped on
    /// the way into `traverse` would pass it perfectly.
    #[test]
    fn fpu_reduction_changes_a_real_multi_ply_search() {
        let visits = |outcome: &MultiPlyOutcome| -> Vec<u32> {
            outcome.tree.decisions[0]
                .s1_stats
                .iter()
                .map(|s| s.visits)
                .collect()
        };
        let legacy = run_with_fpu(STRADDLE, 512, 4, 11, true, None);
        let reduced = run_with_fpu(STRADDLE, 512, 4, 11, true, Some(0.3));
        assert_ne!(
            visits(&legacy),
            visits(&reduced),
            "fpu_reduction reached the config but not the search"
        );
        // Sanity: the flag steers the budget, it does not lose any of it.
        assert_eq!(
            visits(&legacy).iter().sum::<u32>(),
            visits(&reduced).iter().sum::<u32>()
        );
    }

    #[test]
    fn an_out_of_range_fpu_reduction_is_refused_at_the_python_boundary() {
        for bad in [-0.1f32, 1.5] {
            let error = validate_fpu_reduction(Some(bad))
                .expect_err("a reduction outside [0, 1] must not reach the search");
            assert!(error.to_string().contains("fpu_reduction"), "{error}");
        }
        assert_eq!(validate_fpu_reduction(Some(0.0)).unwrap(), Some(0.0));
        assert_eq!(validate_fpu_reduction(None).unwrap(), None);
    }

    // -----------------------------------------------------------------
    // Override telemetry (`arm_priors`) — REPORTING ONLY
    // -----------------------------------------------------------------

    /// Remove every `,"prior":<number>` that `stats_to_json` inserts.
    fn strip_prior_columns(report: &str) -> String {
        let mut out = String::with_capacity(report.len());
        let mut rest = report;
        while let Some(at) = rest.find(",\"prior\":") {
            out.push_str(&rest[..at]);
            let tail = &rest[at..];
            let end = tail
                .find('}')
                .expect("a prior column must end inside its own arm entry");
            rest = &tail[end..];
        }
        out.push_str(rest);
        out
    }

    /// `arm_priors` adds a column and changes NOTHING else.
    ///
    /// The value-gap plan's §2 override measurement is switched on per shard, and
    /// `scripts/foulplay_paired_eval.py` deliberately keeps the flag OUT of
    /// `config_id` on the strength of this claim: telemetry-on and telemetry-off
    /// are one cell, so a shard that measured an override rate pools with a banked
    /// shard that did not. If the flag perturbed selection, §2 would be measuring a
    /// different engine than every other stage under the same cell id — a wrong
    /// number, not an error.
    ///
    /// The STRUCTURAL half of the argument is visible in the signatures rather than
    /// here: `arm_priors` is not an input to the search at all. It is threaded to
    /// `multiply_report_json`, which takes a FINISHED [`MultiPlyOutcome`], and
    /// [`MultiPlyConfig`] — the struct whose fields ARE the search's semantics —
    /// has no such field, for the reason stated above `multiply_report_json`.
    ///
    /// The MEASURED half is this: one search, both renderings, and the on-report
    /// with its added key removed is byte-identical to the off-report. Rendering
    /// the SAME outcome twice is the point — it removes run-to-run variation from
    /// the comparison entirely, so a difference could only come from the flag.
    #[test]
    fn arm_priors_only_adds_a_reported_column() {
        let cfg = MultiPlyConfig {
            max_depth: 4,
            c_puct: 1.4,
            deep_ko_split: true,
            use_opponent_priors: false,
            fpu_reduction: None,
        };
        let outcome = run_with_fpu(STRADDLE, 256, cfg.max_depth, 11, cfg.deep_ko_split, None);
        let off = multiply_report_json(&outcome, 256, &cfg, 11, "hp_fraction", "", false);
        let on = multiply_report_json(&outcome, 256, &cfg, 11, "hp_fraction", "", true);
        // Positive control on the query: if `prior` never appeared, the strip
        // below would trivially "prove" identity on two identical strings.
        assert!(on.contains(",\"prior\":"), "the flag added no column: {on}");
        assert!(!off.contains("\"prior\":"), "flag-off must add none: {off}");
        assert_ne!(off, on);
        assert_eq!(
            strip_prior_columns(&on),
            off,
            "arm_priors changed something other than the arm-prior column"
        );
    }

    /// The complement: a knob that DOES belong in `config_id` must be visible in
    /// the same rendering, so the test above is a statement about `arm_priors` and
    /// not about a report that ignores its config.
    #[test]
    fn fpu_reduction_by_contrast_reaches_the_report_and_the_search() {
        let base = MultiPlyConfig {
            max_depth: 4,
            c_puct: 1.4,
            deep_ko_split: true,
            use_opponent_priors: false,
            fpu_reduction: None,
        };
        let tuned = MultiPlyConfig {
            fpu_reduction: Some(0.3),
            ..base
        };
        let outcome = run_with_fpu(STRADDLE, 256, base.max_depth, 11, base.deep_ko_split, None);
        let plain = multiply_report_json(&outcome, 256, &base, 11, "hp_fraction", "", false);
        let with_fpu = multiply_report_json(&outcome, 256, &tuned, 11, "hp_fraction", "", false);
        assert!(!plain.contains("fpu_reduction"));
        assert!(with_fpu.contains("\"fpu_reduction\":0.3"));
    }

    // -----------------------------------------------------------------
    // Within-batch collision ledger
    // -----------------------------------------------------------------

    fn step(decision: usize, i: usize, j: usize, chance: usize, branch: Option<usize>) -> PathStep {
        PathStep {
            decision,
            i,
            j,
            chance,
            branch,
        }
    }

    fn traversal(path: Vec<PathStep>) -> Traversal {
        Traversal {
            path,
            end: TraversalEnd::Ready(0.5),
        }
    }

    /// The per-seat split is the whole instrument. A round in which side one
    /// takes a fresh arm every time while side two keeps returning to arm 0 is
    /// exactly the shape the deferred-leaf theory predicts (the provisional loss
    /// is a provisional WIN for that seat), and a pooled counter reports the
    /// same total whichever seat is doing the repeating.
    ///
    /// Three selections at one node: (0,0), (1,0), (2,0).
    ///   joint cells   — all distinct, 0 repeats;
    ///   side one arms — 0, 1, 2, all distinct, 0 repeats;
    ///   side two arms — 0, 0, 0, so 2 repeats.
    #[test]
    fn the_collision_ledger_separates_the_two_seats() {
        let mut ledger = CollisionLedger::default();
        ledger.begin_round();
        for i in 0..3 {
            ledger.record(&traversal(vec![step(0, i, 0, i, Some(0))]));
        }
        ledger.end_round(3);
        assert_eq!(ledger.counts.selections, 3);
        assert_eq!(ledger.counts.joint_repeats, 0);
        assert_eq!(ledger.counts.side_one_repeats, 0);
        assert_eq!(
            ledger.counts.side_two_repeats, 2,
            "a pooled counter cannot tell this round from its mirror image"
        );
        assert_eq!(ledger.counts.rounds, 1);
        assert_eq!(ledger.counts.pending_rounds, 1);
    }

    /// Joint cells and leaves are counted per ROUND, and a round is where the
    /// batch's virtual loss lives: cells repeated ACROSS rounds are ordinary
    /// tree refinement, not a collision, because `finalize` has already replaced
    /// the provisionals in between.
    #[test]
    fn the_collision_ledger_forgets_its_cells_between_rounds() {
        let mut ledger = CollisionLedger::default();
        for _ in 0..2 {
            ledger.begin_round();
            ledger.record(&traversal(vec![step(0, 1, 1, 4, Some(2))]));
            ledger.record(&traversal(vec![step(0, 1, 1, 4, Some(2))]));
            ledger.end_round(0);
        }
        assert_eq!(ledger.counts.selections, 4);
        assert_eq!(ledger.counts.joint_repeats, 2, "one per round, not three");
        assert_eq!(ledger.counts.leaf_repeats, 2);
        assert_eq!(ledger.counts.traversals, 4);
        assert_eq!(ledger.counts.rounds, 2);
        assert_eq!(
            ledger.counts.pending_rounds, 0,
            "a round that deferred no leaf cannot exhibit the placeholder"
        );
    }

    /// An EXPANSION (`branch: None`) is a distinct leaf per chance node and can
    /// never repeat inside a round, but it still counts toward `traversals` —
    /// otherwise the repeat RATE would be taken over a denominator that quietly
    /// excludes the productive selections and would read far too high.
    #[test]
    fn expansions_count_as_traversals_and_never_as_leaf_repeats() {
        let mut ledger = CollisionLedger::default();
        ledger.begin_round();
        ledger.record(&traversal(vec![step(0, 0, 0, 7, None)]));
        ledger.record(&traversal(vec![step(0, 0, 1, 8, None)]));
        ledger.record(&traversal(vec![step(0, 0, 2, 3, Some(1))]));
        ledger.record(&traversal(vec![step(0, 0, 3, 3, Some(1))]));
        ledger.end_round(9);
        assert_eq!(ledger.counts.traversals, 4);
        assert_eq!(ledger.counts.leaf_repeats, 1, "only the two Some(1) share a leaf");
    }

    /// Multi-ply paths contribute one selection PER PLY, and cells are keyed on
    /// the decision node — the same arm index at two different nodes is not a
    /// collision.
    #[test]
    fn collisions_are_keyed_on_the_decision_node_not_the_arm_index() {
        let mut ledger = CollisionLedger::default();
        ledger.begin_round();
        ledger.record(&traversal(vec![
            step(0, 0, 0, 1, Some(0)),
            step(5, 0, 0, 2, Some(0)),
        ]));
        ledger.end_round(1);
        assert_eq!(ledger.counts.selections, 2);
        assert_eq!(ledger.counts.joint_repeats, 0);
        assert_eq!(ledger.counts.side_one_repeats, 0);
        assert_eq!(ledger.counts.side_two_repeats, 0);
    }

    fn priored_branch(
        self_priors: Option<(bool, Vec<f32>)>,
        opponent_priors: Option<(bool, Vec<f32>)>,
    ) -> ChanceBranch {
        ChanceBranch {
            probability: 1.0,
            instructions: Vec::new(),
            value_sum: 0.5,
            visits: 1,
            terminal: None,
            no_expand: false,
            pending_row: None,
            child: None,
            child_self_priors: self_priors,
            child_opponent_priors: opponent_priors,
        }
    }

    /// The lazy-apply orientation pin. The searching seat is side one, so the
    /// branch carries `(true, self)` and `(false, opponent)`; each vector must
    /// land on its own seat's stats. A mutant that negates either stored flag,
    /// or that applies both vectors to one seat, fails here.
    #[test]
    fn both_seats_priors_land_on_their_own_stats_at_child_creation() {
        let mut tree = Tree {
            decisions: vec![prior_node(&[0.5, 0.5], &[0.5, 0.5])],
            chances: vec![ChanceNode {
                branches: vec![priored_branch(
                    Some((true, vec![0.9, 0.1])),
                    Some((false, vec![0.2, 0.8])),
                )],
            }],
        };
        apply_branch_child_priors(&mut tree, 0, 0, 0);
        assert_eq!(priors_of(&tree.decisions[0].s1_stats), vec![0.9, 0.1]);
        assert_eq!(priors_of(&tree.decisions[0].s2_stats), vec![0.2, 0.8]);
    }

    /// The same branch with the searching seat on side TWO: the stored flags
    /// invert, and so must the destinations. Together with the test above this
    /// rules out a hard-coded "self = s1, opponent = s2" apply.
    #[test]
    fn stored_side_flags_route_the_apply_when_the_searching_seat_is_side_two() {
        let mut tree = Tree {
            decisions: vec![prior_node(&[0.5, 0.5], &[0.5, 0.5])],
            chances: vec![ChanceNode {
                branches: vec![priored_branch(
                    Some((false, vec![0.9, 0.1])),
                    Some((true, vec![0.2, 0.8])),
                )],
            }],
        };
        apply_branch_child_priors(&mut tree, 0, 0, 0);
        assert_eq!(priors_of(&tree.decisions[0].s2_stats), vec![0.9, 0.1]);
        assert_eq!(priors_of(&tree.decisions[0].s1_stats), vec![0.2, 0.8]);
    }

    /// Opponent priors alone must not disturb the acting seat — this is the
    /// flag-off/flag-on containment property at the apply site.
    #[test]
    fn an_opponent_only_branch_leaves_the_acting_seat_uniform() {
        let mut tree = Tree {
            decisions: vec![prior_node(&[0.5, 0.5], &[0.5, 0.5])],
            chances: vec![ChanceNode {
                branches: vec![priored_branch(None, Some((false, vec![0.2, 0.8])))],
            }],
        };
        apply_branch_child_priors(&mut tree, 0, 0, 0);
        assert_eq!(priors_of(&tree.decisions[0].s1_stats), vec![0.5, 0.5]);
        assert_eq!(priors_of(&tree.decisions[0].s2_stats), vec![0.2, 0.8]);
    }

    #[test]
    fn a_branch_with_no_stored_priors_leaves_the_child_uniform() {
        let mut tree = Tree {
            decisions: vec![prior_node(&[0.5, 0.5], &[0.5, 0.5])],
            chances: vec![ChanceNode {
                branches: vec![priored_branch(None, None)],
            }],
        };
        apply_branch_child_priors(&mut tree, 0, 0, 0);
        assert_eq!(priors_of(&tree.decisions[0].s1_stats), vec![0.5, 0.5]);
        assert_eq!(priors_of(&tree.decisions[0].s2_stats), vec![0.5, 0.5]);
    }

    #[test]
    fn renderer_unsafe_branch_is_rejected_at_the_tree_fold_seam() {
        Python::initialize();
        let s1 = MoveChoice::None;
        let s2 = MoveChoice::None;
        let instructions: [Instruction; 0] = [];
        let seam = BranchSeam {
            s1: &s1,
            s2: &s2,
            instructions: &instructions,
            parent: None,
            chance: 0,
            branch_index: 0,
            branch_on_damage: false,
            depth: 0,
        };
        let rendered = crate::events::RenderedEvents {
            lines: Vec::new(),
            turn_completed: false,
            lossy: vec!["synthetic_test_ambiguity".to_string()],
            attribution_unsafe: vec!["synthetic_test_ambiguity".to_string()],
            lossy_subcases: Vec::new(),
            active_status_transitions: Vec::new(),
        };
        let error = seam
            .reject_attribution_unsafe(&rendered)
            .expect_err("unsafe event text must not reach fold/encoder");
        assert!(error
            .to_string()
            .contains("attribution-unsafe renderer branch rejected before tree/model fold"));
    }

    /// Charmander (ember/tackle) vs Squirtle (watergun/tackle), 100 HP each —
    /// the crate's standard minimal fixture (`minimal_gen3_fixture`).
    const MINIMAL: &str = include_str!("test_fixtures/minimal.state");

    /// The REAL panic, contained, end to end -- and the process survives it.
    ///
    /// poke-engine rejects an out-of-range Rest counter with a panic. Across the
    /// pyo3 boundary that is a PanicException, which derives from BaseException,
    /// so the caller's `except Exception` cannot catch it and one bad belief
    /// world kills the whole shard process. Measured in a campaign probe: it
    /// took out shards deterministically by seed.
    ///
    /// This drives the real panic through a real entry point. The guard's own
    /// unit tests use a synthetic `panic!`; this one proves the containment
    /// holds for an engine panic raised deep inside the search.
    #[test]
    fn a_real_engine_panic_is_contained_and_the_next_search_still_works() {
        pyo3::Python::initialize();
        let mut poisoned = parse_state(MINIMAL.trim()).expect("fixture parses");
        poisoned.side_one.get_active().status = poke_engine::state::PokemonStatus::SLEEP;
        poisoned.side_one.get_active().rest_turns = 32;
        let poisoned_str = poisoned.serialize();

        let error = puct_search_multi(&poisoned_str, 32, 2, 1.4, 7, true, None)
            .expect_err("an out-of-range rest counter must surface as an error");
        pyo3::Python::attach(|py| {
            assert!(
                error.is_instance_of::<pyo3::exceptions::PyValueError>(py),
                "must be catchable by `except Exception`, or one world kills the shard"
            );
            let message = error.value(py).to_string();
            assert!(
                message.contains("rest_turns"),
                "the engine's own reason must survive into world_failure_reasons: {message}"
            );
        });

        // The point of containing rather than crashing: the NEXT world still runs.
        let healthy = parse_state(MINIMAL.trim()).expect("fixture parses").serialize();
        let report = puct_search_multi(&healthy, 32, 2, 1.4, 7, true, None)
            .expect("a good state must still search after a contained panic");
        assert!(report.contains("visits"), "{report}");
    }
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
        run_with_fpu(state_str, iterations, max_depth, seed, deep_ko_split, None)
    }

    fn run_with_fpu(
        state_str: &str,
        iterations: usize,
        max_depth: u8,
        seed: u64,
        deep_ko_split: bool,
        fpu_reduction: Option<f32>,
    ) -> MultiPlyOutcome {
        let mut state = parse_state(state_str.trim()).expect("fixture state parses");
        let cfg = MultiPlyConfig {
            max_depth,
            c_puct: 1.4,
            deep_ko_split,
            use_opponent_priors: false,
            fpu_reduction,
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
            s2_old
                .iter()
                .map(|s| s.display.as_str())
                .collect::<Vec<_>>(),
            root.s2_stats
                .iter()
                .map(|s| s.display.as_str())
                .collect::<Vec<_>>(),
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
                use_opponent_priors: false,
                fpu_reduction: None,
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
            use_opponent_priors: false,
            fpu_reduction: None,
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
