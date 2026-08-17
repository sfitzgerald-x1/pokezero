//! Oracle-leaf batching seam: R-rollout Monte-Carlo leaf values inside the
//! SAME `traverse`/`finalize` tree the production crate search uses.
//!
//! # Why this module exists
//!
//! The search-ceiling program's instrument 2 (the arbiter) runs the production
//! search config with leaf values replaced by R-rollout terminal estimates.
//! The plan states the engineering risk plainly: "'same search, same
//! everything' is a design requirement to be verified, not a property the code
//! gives for free". This module is built so that requirement is *checkable*:
//!
//! - It never re-implements selection, expansion, or backup. It calls
//!   [`crate::tree::traverse`] and [`crate::tree::finalize`] with the same
//!   [`crate::tree::MultiPlyConfig`] the production path builds.
//! - The search RNG stream and the rollout RNG stream are DISJOINT. Rollouts
//!   draw from per-leaf RNGs seeded by `splitmix64(rollout_seed, ordinal,
//!   trial)`, never from the `&mut StdRng` that `traverse` uses to sample
//!   chance branches.
//!
//!   THE EXACT CLAIM, because a looser wording of it was measurably false and
//!   shipped in review. Disjointness buys this and only this: **at a fixed
//!   leaf-value vector** the selection sequence is bit-identical to
//!   production's, so nothing about the tree changes for a reason other than
//!   the leaf values. It does NOT buy invariance to `rollout_seed` or to `R`.
//!   Those knobs reprice the leaves; repriced leaves change selection; changed
//!   selection descends elsewhere and therefore samples different chance
//!   branches. Measured on `symmetric.state` with the search seed held and only
//!   `rollout_seed` moved 1 -> 2: `chance_nodes` 344 -> 348,
//!   `decision_nodes` 139 -> 149, `depth_occupancy`
//!   [600, 596, 512] -> [600, 596, 509]. That is the instrument working as
//!   designed, not a leak -- but "changing R cannot perturb which branches were
//!   sampled" was the wrong sentence for it, on the one knob whose whole
//!   purpose is redrawing labels.
//!
//!   The narrow claim is what the fidelity gate actually tests, and it is
//!   verified over 1440 configurations (5 fixtures x max_depth {1,2,3,5,8,12} x
//!   seeds x iterations x `deep_ko_split` x `fpu_reduction`), 0 mismatches.
//! - The leaf pricer is a PARAMETER ([`LeafMode`]). With
//!   [`LeafMode::HpFraction`] this driver prices leaves exactly as
//!   `multiply_search_with_eval` does, which is what the fidelity gate
//!   compares against (`rollout_batch1_matches_sequential_report`): same
//!   driver, same batching, only the row pricer swapped.
//!
//! # The batching seam
//!
//! Leaves are DEFERRED, not priced inline: the pricing closure clones the leaf
//! state into a row buffer and returns [`LeafPrice::Deferred`]. A round
//! collects traversals until it holds `batch` traversals or `batch` pending
//! rows (an expansion never splits across rounds), prices every row, then
//! `finalize`s the round's traversals in collection order. This is the same
//! round structure the `model` feature's `multiply_batched_core` uses, and it
//! is what lets one round's rows be priced together — in parallel threads
//! here, and (the estimand-faithful variant) in one batched call out to a
//! Python row pricer.
//!
//! `batch = 1` is the sequential regime BY CONSTRUCTION (the round's single
//! virtual loss is replaced before the next selection observes the stats), and
//! it is what the arbiter arm runs: an expansion still yields ~4 rows, so
//! `4 * R` independent rollouts are available to parallelise WITHOUT trading
//! any selection fidelity for throughput. `batch > 1` exists for pricers with
//! a per-call cost worth amortising (a GPU forward, a Python callback) and is
//! a measured fidelity LOSS — the fidelity gate demonstrates exactly that.

use std::sync::atomic::{AtomicU64, Ordering};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::MoveChoice;
use poke_engine::state::State;

use crate::tree::{
    finalize, multiply_report_json, traverse, BranchSeam, LeafPrice, MultiPlyConfig,
    MultiPlyOutcome, SearchCounters, Traversal,
};
use crate::{parse_state, HpFractionEval, LeafEval};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Which value function prices a leaf. The tree is identical under both; this
/// is the ONLY thing the arbiter arm changes relative to production search,
/// and making it an enum on one driver is what makes that claim testable
/// rather than asserted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LeafMode {
    /// Production's handcrafted leaf (`HpFractionEval`). Used by the fidelity
    /// gate: same driver, same batching, production's leaf values.
    HpFraction,
    /// R terminal rollouts under [`RolloutPolicy`], mean win indicator.
    Rollout,
}

/// The action distribution both seats play during a rollout.
///
/// `Uniform` is the only variant implemented in-crate, and its estimand must
/// be stated wherever a result from it is quoted: it prices
/// `P(side one wins | both seats play uniformly at random from here)`, which
/// is NOT the vhprobe shards' `true_*` (terminal win probability under POLICY
/// continuation). The estimand-faithful pricer is the Python row callback
/// (`row_pricer`), which hands whole leaf states out to the same
/// `continue_rollout_from_current_state` machinery the shards were built with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RolloutPolicy {
    Uniform,
}

impl RolloutPolicy {
    pub(crate) fn parse(name: &str) -> PyResult<Self> {
        match name {
            "uniform" => Ok(Self::Uniform),
            other => Err(PyValueError::new_err(format!(
                "unknown rollout_policy {other:?}; supported: 'uniform'"
            ))),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RolloutConfig {
    /// R: rollouts per leaf. `0` is rejected at the boundary — a zero-rollout
    /// "Monte-Carlo" leaf would silently price every leaf at the cap fallback.
    pub rollouts: u32,
    /// Ply cap per rollout. A rollout that hits it is NOT a terminal
    /// observation; it falls back to the handcrafted leaf and is counted, so
    /// the report can say what fraction of the "oracle" was actually
    /// HP-fraction.
    pub max_plies: u32,
    pub policy: RolloutPolicy,
    /// Damage-roll branching inside rollouts. Rollouts sample ONE outcome, so
    /// enabling it only refines the sampled damage distribution; the search's
    /// own `branch_on_damage` policy is untouched by this field.
    pub branch_on_damage: bool,
    /// Root of the rollout RNG stream. Disjoint from the search seed by
    /// construction (see module docs).
    pub seed: u64,
    /// Worker threads for pricing one round's rows. Rollout seeds are derived
    /// from (ordinal, trial), never from thread identity, so the priced values
    /// are independent of this number — asserted by
    /// `thread_count_does_not_change_values`.
    pub threads: usize,
}

/// What the rollouts actually did, so nobody has to assume it.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RolloutStats {
    pub leaves_priced: u64,
    pub rollouts_run: u64,
    pub plies_stepped: u64,
    /// Rollouts that reached `battle_is_over() != 0` — the real terminal
    /// observations.
    pub terminal_hits: u64,
    /// Rollouts stopped by `max_plies` (priced by the HP-fraction fallback).
    pub cap_hits: u64,
    /// Rollouts stopped because the engine offered no legal continuation
    /// (empty option vector or empty instruction list) without the battle
    /// reading as over. Also priced by the fallback, counted separately
    /// because it means something different from a cap hit.
    pub dead_ends: u64,
}

impl RolloutStats {
    fn merge(&mut self, other: &RolloutStats) {
        self.leaves_priced += other.leaves_priced;
        self.rollouts_run += other.rollouts_run;
        self.plies_stepped += other.plies_stepped;
        self.terminal_hits += other.terminal_hits;
        self.cap_hits += other.cap_hits;
        self.dead_ends += other.dead_ends;
    }

    pub(crate) fn to_json_fields(self, cfg: &RolloutConfig, batch: usize, rounds: usize) -> String {
        let denom = self.rollouts_run.max(1) as f64;
        format!(
            "\"rollouts\":{},\"rollout_policy\":\"{}\",\"rollout_max_plies\":{},\
             \"rollout_seed\":{},\"rollout_threads\":{},\"leaf_batch\":{},\"rounds\":{},\
             \"leaves_priced\":{},\"rollouts_run\":{},\"rollout_plies\":{},\
             \"rollout_terminal_hits\":{},\"rollout_cap_hits\":{},\"rollout_dead_ends\":{},\
             \"rollout_terminal_fraction\":{:.6},\"rollout_fallback_fraction\":{:.6},\
             \"rollout_mean_plies\":{:.3}",
            cfg.rollouts,
            cfg.policy.name(),
            cfg.max_plies,
            cfg.seed,
            cfg.threads,
            batch,
            rounds,
            self.leaves_priced,
            self.rollouts_run,
            self.plies_stepped,
            self.terminal_hits,
            self.cap_hits,
            self.dead_ends,
            self.terminal_hits as f64 / denom,
            (self.cap_hits + self.dead_ends) as f64 / denom,
            self.plies_stepped as f64 / denom,
        )
    }
}

// ---------------------------------------------------------------------------
// Rollout RNG: order-independent by construction
// ---------------------------------------------------------------------------

/// SplitMix64. Used to derive a per-(leaf, trial) seed so a rollout's draws
/// depend on WHICH leaf and WHICH trial it is, never on thread scheduling or
/// on how many rows shared its round.
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = z;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn rollout_seed(root: u64, ordinal: u64, trial: u32) -> u64 {
    splitmix64(
        root ^ splitmix64(ordinal.wrapping_mul(0x2545_F491_4F6C_DD1D))
            ^ splitmix64(u64::from(trial).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
    )
}

// ---------------------------------------------------------------------------
// One rollout
// ---------------------------------------------------------------------------

fn pick(policy: RolloutPolicy, rng: &mut StdRng, options: &[MoveChoice]) -> MoveChoice {
    match policy {
        RolloutPolicy::Uniform => options[rng.random_range(0..options.len())],
    }
}

/// Play one rollout to terminal (or to the ply cap) from `state`.
///
/// `state` is mutated and RESTORED: every applied instruction list is reversed
/// in reverse order before returning, so a caller may hand the same `&mut
/// State` to successive rollouts. This mirrors what `expand_edge` does around
/// the pricing seam, and it is why the leaf clone can be reused across all R
/// trials of one leaf instead of cloned R times.
fn rollout_once(
    state: &mut State,
    cfg: &RolloutConfig,
    rng: &mut StdRng,
    stats: &mut RolloutStats,
) -> f32 {
    let mut applied: Vec<Vec<poke_engine::instruction::Instruction>> = Vec::new();
    let mut value: Option<f32> = None;
    for _ in 0..cfg.max_plies {
        let over = state.battle_is_over();
        if over != 0.0 {
            stats.terminal_hits += 1;
            value = Some(if over > 0.0 { 1.0 } else { 0.0 });
            break;
        }
        let (s1_options, s2_options) = state.get_all_options();
        if s1_options.is_empty() || s2_options.is_empty() {
            stats.dead_ends += 1;
            break;
        }
        let s1 = pick(cfg.policy, rng, &s1_options);
        let s2 = pick(cfg.policy, rng, &s2_options);
        let branches =
            generate_instructions_from_move_pair(state, &s1, &s2, cfg.branch_on_damage);
        if branches.is_empty() {
            stats.dead_ends += 1;
            break;
        }
        let index = crate::sample_branch_index(rng, &branches);
        let instructions = branches[index].instruction_list.clone();
        state.apply_instructions(&instructions);
        applied.push(instructions);
        stats.plies_stepped += 1;
    }
    if value.is_none() {
        // Not a terminal observation. Either the ply cap ran out (the loop
        // completed `max_plies` steps) or the engine offered no legal
        // continuation — the dead-end branches already counted themselves, so
        // the applied-ply count is the discriminator, and it cannot reach
        // `max_plies` on a dead end because that path breaks before applying.
        if applied.len() == cfg.max_plies as usize {
            stats.cap_hits += 1;
        }
        value = Some(HpFractionEval.eval(state));
    }
    for instructions in applied.iter().rev() {
        state.reverse_instructions(instructions);
    }
    value.expect("value set on every exit path")
}

/// One trial's outcome for one leaf. Stored per (row, trial) rather than
/// accumulated, so the per-leaf mean is always summed in TRIAL ORDER: f32/f64
/// addition is not associative, and letting threads accumulate partial sums
/// would make the priced value depend on the thread count.
fn rollout_trial(
    leaf: &State,
    ordinal: u64,
    trial: u32,
    cfg: &RolloutConfig,
    scratch: &mut State,
    stats: &mut RolloutStats,
) -> f32 {
    scratch.clone_from(leaf);
    let mut rng = StdRng::seed_from_u64(rollout_seed(cfg.seed, ordinal, trial));
    let value = rollout_once(scratch, cfg, &mut rng, stats);
    stats.rollouts_run += 1;
    value
}

// ---------------------------------------------------------------------------
// Row pricing: the batching seam's payload
// ---------------------------------------------------------------------------

/// Canonical reduction of a `[row][trial]` matrix to one value per row, and
/// the refusal that makes an unpriced slot an error instead of a NaN.
///
/// Split out of `price_rows` for one reason: it is the piece that has to be
/// callable with a doctored matrix. A guard whose only demonstration is "someone
/// mutated the writer and rebuilt" is a guard nobody can regression-test, and
/// this program's rule is that every guard ships with a demonstrated failing
/// input -- so the failing input has to be reachable from a test.
///
/// Row-major, trial order, f64 accumulator: f32 addition is not associative, so
/// this order is what makes the priced value independent of how the scheduler
/// split the work.
///
/// Returns `Result<_, String>` rather than `PyResult`: `PyErr::to_string`
/// requires an initialised interpreter, so a `PyResult` here would force the
/// guard's own test to boot Python just to read the refusal it is asserting on.
/// The caller wraps it.
fn reduce_trials(trials: &[f32], rows: usize, r: usize) -> Result<Vec<f32>, String> {
    if let Some(task) = trials.iter().position(|v| !v.is_finite()) {
        return Err(format!(
            "rollout trial {} of {} was never priced (row {}, trial {}); refusing to \
             average a non-finite slot into a leaf value",
            task,
            trials.len(),
            task / r.max(1),
            task % r.max(1),
        ));
    }
    Ok((0..rows)
        .map(|row| {
            let total: f64 = trials[row * r..(row + 1) * r]
                .iter()
                .map(|v| f64::from(*v))
                .sum();
            (total / r as f64) as f32
        })
        .collect())
}

/// Price one round's rows: `rows.len() * R` independent rollouts.
///
/// Threading is a THROUGHPUT knob only, and two design choices are what make
/// that true rather than hoped:
///
/// - The unit of work is a `(row, trial)` PAIR, not a row. At `leaf_batch = 1`
///   a round holds only the ~2-4 rows of one expansion, so row-granular
///   parallelism caps out at 2-4 threads no matter how many are configured —
///   measured: 8 threads bought 2.2x. Pair-granular work exposes
///   `rows * R` tasks instead.
/// - Trials are WRITTEN to a `[row][trial]` matrix and reduced afterwards in
///   trial order. Per-thread partial sums would make the priced value depend
///   on how the scheduler split the work, because floating-point addition is
///   not associative. `thread_count_does_not_change_values` is the gate.
///
/// Tasks are handed out by a shared atomic cursor rather than split into
/// contiguous blocks: rollout length varies by an order of magnitude with how
/// close the leaf is to terminal, so static splits idle.
fn price_rows(
    rows: &[State],
    ordinals: &[u64],
    cfg: &RolloutConfig,
) -> PyResult<(Vec<f32>, RolloutStats)> {
    let n = rows.len();
    let r = cfg.rollouts as usize;
    let mut stats = RolloutStats::default();
    if n == 0 {
        return Ok((Vec::new(), stats));
    }
    stats.leaves_priced = n as u64;
    let tasks = n * r;
    let mut trials = vec![f32::NAN; tasks];
    let threads = cfg.threads.max(1).min(tasks);
    if threads == 1 {
        let mut scratch = rows[0].clone();
        for task in 0..tasks {
            let (row, trial) = (task / r, task % r);
            trials[task] =
                rollout_trial(&rows[row], ordinals[row], trial as u32, cfg, &mut scratch, &mut stats);
        }
    } else {
        let cursor = AtomicU64::new(0);
        let collected: Vec<(Vec<(usize, f32)>, RolloutStats)> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    let cursor = &cursor;
                    scope.spawn(move || {
                        let mut local: Vec<(usize, f32)> = Vec::new();
                        let mut local_stats = RolloutStats::default();
                        let mut scratch = rows[0].clone();
                        loop {
                            let task = cursor.fetch_add(1, Ordering::Relaxed) as usize;
                            if task >= tasks {
                                break;
                            }
                            let (row, trial) = (task / r, task % r);
                            let value = rollout_trial(
                                &rows[row],
                                ordinals[row],
                                trial as u32,
                                cfg,
                                &mut scratch,
                                &mut local_stats,
                            );
                            local.push((task, value));
                        }
                        (local, local_stats)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("rollout worker panicked"))
                .collect()
        });
        for (local, local_stats) in &collected {
            for (task, value) in local {
                trials[*task] = *value;
            }
            // `leaves_priced` is set once above from the row count; workers
            // count rollouts, not leaves, so nothing double-counts here.
            stats.merge(local_stats);
        }
    }
    // NOT a `debug_assert!`. This crate ships `--release` (Cargo.toml sets
    // `[profile.release]` and `scripts/build_search_crate_model.sh` builds
    // `--release`), where `debug_assert!` is compiled OUT -- so the guard that
    // used to stand here did not exist in any wheel anyone ran.
    //
    // Demonstrated, by deliberately skipping the write of task 0 and rebuilding
    // with the exact shipping command: `root_value` and every root arm's `q`
    // came back `nan`, with NO exception, and the report string carried a bare
    // `NaN` token -- which is invalid JSON that Python's non-strict
    // `json.loads` accepts, so `_search_rollout_crate` would have taken a
    // decision off a degenerate tree and banked it as a measurement. An
    // unpriced slot is a lost rollout, and a lost rollout must be an error, not
    // a quiet NaN that propagates into a win-rate number.
    //
    // `events.rs` already wrote this rule down for this crate; the seam simply
    // has to obey it.
    let values = reduce_trials(&trials, n, r).map_err(PyValueError::new_err)?;
    Ok((values, stats))
}

// ---------------------------------------------------------------------------
// The driver
// ---------------------------------------------------------------------------

pub(crate) struct RolloutOutcome {
    pub outcome: MultiPlyOutcome,
    pub stats: RolloutStats,
    pub rounds: usize,
}

/// Deferred-row multi-ply search. Identical tree mechanics to
/// `multiply_search_with_eval`; the leaf pricer and the batching are the only
/// differences, and `LeafMode::HpFraction` + `batch = 1` reduces it to that
/// function's exact behaviour (the fidelity gate).
pub(crate) fn multiply_search_rollout(
    state: &mut State,
    iterations: usize,
    cfg: &MultiPlyConfig,
    seed: u64,
    batch: usize,
    leaf_mode: LeafMode,
    rcfg: &RolloutConfig,
) -> PyResult<RolloutOutcome> {
    if state.battle_is_over() != 0.0 {
        return Err(PyValueError::new_err("battle is already over at the root"));
    }
    let batch = batch.max(1);
    let mut tree = crate::tree::Tree::from_root(state)?;
    let mut counters = SearchCounters::default();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut stats = RolloutStats::default();
    let mut ordinal_next: u64 = 0;
    let mut rounds = 0usize;
    let mut completed = 0usize;
    let start = std::time::Instant::now();
    while completed < iterations {
        let traversal_budget = batch.min(iterations - completed);
        let mut traversals: Vec<Traversal> = Vec::with_capacity(traversal_budget);
        let mut rows: Vec<State> = Vec::new();
        let mut ordinals: Vec<u64> = Vec::new();
        while traversals.len() < traversal_budget && rows.len() < batch {
            let traversal = traverse(
                &mut tree,
                state,
                &mut rng,
                cfg,
                &mut counters,
                &mut |leaf: &State, _seam: &BranchSeam| {
                    let row = rows.len();
                    rows.push(leaf.clone());
                    ordinals.push(ordinal_next);
                    ordinal_next += 1;
                    LeafPrice::Deferred(row)
                },
            );
            traversals.push(traversal);
        }
        let row_values = match leaf_mode {
            LeafMode::HpFraction => {
                stats.leaves_priced += rows.len() as u64;
                rows.iter().map(|leaf| HpFractionEval.eval(leaf)).collect()
            }
            LeafMode::Rollout => {
                let (values, round_stats) = price_rows(&rows, &ordinals, rcfg)?;
                stats.merge(&round_stats);
                values
            }
        };
        for traversal in &traversals {
            finalize(&mut tree, traversal, &row_values);
        }
        completed += traversals.len();
        rounds += 1;
    }
    Ok(RolloutOutcome {
        outcome: MultiPlyOutcome {
            tree,
            counters,
            elapsed_s: start.elapsed().as_secs_f64(),
        },
        stats,
        rounds,
    })
}

// ---------------------------------------------------------------------------
// Python surface
// ---------------------------------------------------------------------------

/// Multi-ply PUCT with R-rollout Monte-Carlo leaf values (search-ceiling
/// program, Phase 1 instrument 2 — the arbiter).
///
/// Every tree-shaping argument carries EXACTLY the meaning it carries in
/// `puct_search_multi`, and is threaded into the same `MultiPlyConfig`. The
/// new arguments only touch leaf valuation and how rows are grouped:
///
/// * `rollouts` — R per leaf. Per-leaf label SE near 0.5 is `0.5/sqrt(R)`.
/// * `rollout_max_plies` — ply cap. Report carries
///   `rollout_fallback_fraction`: the share of rollouts that did NOT reach a
///   terminal and were priced by the handcrafted fallback instead. Read a
///   result with a high fallback fraction as a blend, not as an oracle.
/// * `leaf_batch` — rows priced per round. `1` (default) is the sequential
///   selection regime; larger values trade selection fidelity for pricer
///   amortisation.
/// * `leaf_mode` — `"rollout"` or `"hp_fraction"`. The second exists so the
///   fidelity gate can run THIS driver against production's leaf values.
/// * `rollout_threads` — throughput only; values are thread-count invariant.
#[pyfunction]
#[pyo3(signature = (
    state_str,
    iterations,
    max_depth = 2,
    c_puct = 1.4,
    seed = 0,
    deep_ko_split = true,
    fpu_reduction = None,
    rollouts = 32,
    rollout_max_plies = 200,
    rollout_policy = "uniform",
    rollout_seed = 0,
    rollout_threads = 1,
    rollout_branch_on_damage = false,
    leaf_batch = 1,
    leaf_mode = "rollout",
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn puct_search_multi_rollout(
    state_str: &str,
    iterations: usize,
    max_depth: u8,
    c_puct: f32,
    seed: u64,
    deep_ko_split: bool,
    fpu_reduction: Option<f32>,
    rollouts: u32,
    rollout_max_plies: u32,
    rollout_policy: &str,
    rollout_seed: u64,
    rollout_threads: usize,
    rollout_branch_on_damage: bool,
    leaf_batch: usize,
    leaf_mode: &str,
) -> PyResult<String> {
    if iterations == 0 {
        return Err(PyValueError::new_err("iterations must be > 0"));
    }
    if max_depth == 0 || max_depth > 32 {
        return Err(PyValueError::new_err("max_depth must be in 1..=32"));
    }
    // A zero-rollout Monte-Carlo leaf is not a cheap Monte-Carlo leaf: it is
    // the HP-fraction fallback wearing the oracle's name in the report.
    if rollouts == 0 {
        return Err(PyValueError::new_err(
            "rollouts must be > 0 (a zero-rollout leaf would silently be the fallback evaluator)",
        ));
    }
    // Likewise a zero ply cap: every rollout would fall back at ply zero.
    if rollout_max_plies == 0 {
        return Err(PyValueError::new_err(
            "rollout_max_plies must be > 0 (every rollout would fall back at ply zero)",
        ));
    }
    if leaf_batch == 0 {
        return Err(PyValueError::new_err("leaf_batch must be > 0"));
    }
    if rollout_threads == 0 {
        return Err(PyValueError::new_err("rollout_threads must be > 0"));
    }
    let leaf_mode_parsed = match leaf_mode {
        "rollout" => LeafMode::Rollout,
        "hp_fraction" => LeafMode::HpFraction,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown leaf_mode {other:?}; supported: 'rollout', 'hp_fraction'"
            )))
        }
    };
    let policy = RolloutPolicy::parse(rollout_policy)?;
    let fpu_reduction = crate::tree::validate_fpu_reduction(fpu_reduction)?;
    let mut state = parse_state(state_str)?;
    let cfg = MultiPlyConfig {
        max_depth,
        c_puct,
        deep_ko_split,
        use_opponent_priors: false,
        fpu_reduction,
    };
    let rcfg = RolloutConfig {
        rollouts,
        max_plies: rollout_max_plies,
        policy,
        branch_on_damage: rollout_branch_on_damage,
        seed: rollout_seed,
        threads: rollout_threads,
    };
    // Same containment as `puct_search_multi`: poke-engine panics reach this
    // path through the identical `generate_instructions_from_move_pair` calls,
    // and rollouts drive far MORE engine plies than search does, so the guard
    // matters more here, not less.
    let result = crate::panic_guard::catch_native_panic(|| {
        multiply_search_rollout(
            &mut state,
            iterations,
            &cfg,
            seed,
            leaf_batch,
            leaf_mode_parsed,
            &rcfg,
        )
    })?;
    let extra = result.stats.to_json_fields(&rcfg, leaf_batch, result.rounds);
    let evaluator_name = match leaf_mode_parsed {
        LeafMode::Rollout => "rollout",
        LeafMode::HpFraction => "hp_fraction",
    };
    Ok(multiply_report_json(
        &result.outcome,
        iterations,
        &cfg,
        seed,
        evaluator_name,
        &extra,
        false,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Same fixture files `tree.rs`'s own tests use, included directly rather
    // than re-exported through that module's private test scope.
    const MINIMAL: &str = include_str!("test_fixtures/minimal.state");
    const ANALYTIC_TOXIC: &str = include_str!("test_fixtures/analytic_toxic.state");

    /// Strip the fields that are allowed to differ between two runs of the
    /// same search: wall-clock timings and the rollout-specific extras.
    fn normalize(report: &str) -> String {
        let value: serde_json::Value = serde_json::from_str(report).expect("report parses");
        let mut map = value.as_object().expect("report is an object").clone();
        for key in [
            "elapsed_s",
            "iterations_per_s",
            "evaluator",
            "rollouts",
            "rollout_policy",
            "rollout_max_plies",
            "rollout_seed",
            "rollout_threads",
            "leaf_batch",
            "rounds",
            "leaves_priced",
            "rollouts_run",
            "rollout_plies",
            "rollout_terminal_hits",
            "rollout_cap_hits",
            "rollout_dead_ends",
            "rollout_terminal_fraction",
            "rollout_fallback_fraction",
            "rollout_mean_plies",
        ] {
            map.remove(key);
        }
        serde_json::to_string(&serde_json::Value::Object(map)).expect("re-serializes")
    }

    /// THE FIDELITY GATE. The rollout driver at `leaf_batch = 1` with
    /// production's leaf values reproduces `puct_search_multi`'s report field
    /// for field: identical visits, identical Q, identical depth occupancy,
    /// identical expansion and leaf-eval counts, identical root value.
    ///
    /// That is the "same search, same everything" requirement discharged as a
    /// measurement: whatever the batching seam does, it does not change
    /// selection, expansion, backup, or the RNG stream.
    #[test]
    fn rollout_batch1_matches_sequential_report() {
        for fixture in [MINIMAL, ANALYTIC_TOXIC] {
            for (iterations, depth, seed) in [(400usize, 1u8, 7u64), (1200, 3, 99), (800, 4, 5)] {
                let production = crate::tree::puct_search_multi(
                    fixture.trim(),
                    iterations,
                    depth,
                    1.4,
                    seed,
                    true,
                    None,
                )
                .expect("production search runs");
                let arm = puct_search_multi_rollout(
                    fixture.trim(),
                    iterations,
                    depth,
                    1.4,
                    seed,
                    true,
                    None,
                    // Rollout knobs are inert under leaf_mode="hp_fraction";
                    // set them to non-defaults so the gate would catch a
                    // pricer that leaked into the tree.
                    17,
                    13,
                    "uniform",
                    12345,
                    4,
                    true,
                    1,
                    "hp_fraction",
                )
                .expect("rollout driver runs");
                assert_eq!(
                    normalize(&production),
                    normalize(&arm),
                    "batch=1 hp_fraction must reproduce the sequential report \
                     (iterations {iterations}, depth {depth}, seed {seed})"
                );
            }
        }
    }

    /// THE FIDELITY GATE'S DEMONSTRATED FAILING INPUT (program-wide rule: a
    /// check that cannot read False certifies nothing).
    ///
    /// Two independent mutations must break the comparison the gate above
    /// asserts:
    ///
    /// 1. `leaf_batch > 1` — the virtual-loss round is a real fidelity loss,
    ///    so the gate must NOT pass at batch 8. If this assertion ever fails,
    ///    the gate above is comparing something insensitive to selection.
    /// 2. `leaf_mode = "rollout"` — swapping the leaf pricer must change the
    ///    report, or the "only leaf valuation differs" claim is vacuous
    ///    because nothing downstream reads the leaf value.
    #[test]
    fn fidelity_gate_reads_false_on_batch_and_on_pricer() {
        let baseline = crate::tree::puct_search_multi(
            MINIMAL.trim(),
            1200,
            3,
            1.4,
            99,
            true,
            None,
        )
        .expect("production search runs");
        let batched = puct_search_multi_rollout(
            MINIMAL.trim(),
            1200,
            3,
            1.4,
            99,
            true,
            None,
            4,
            50,
            "uniform",
            1,
            1,
            false,
            8, // <-- the mutation
            "hp_fraction",
        )
        .expect("rollout driver runs");
        assert_ne!(
            normalize(&baseline),
            normalize(&batched),
            "leaf_batch=8 must NOT reproduce the sequential report; if it does, \
             the fidelity gate is blind to selection changes"
        );
        let rollout_priced = puct_search_multi_rollout(
            MINIMAL.trim(),
            1200,
            3,
            1.4,
            99,
            true,
            None,
            4,
            50,
            "uniform",
            1,
            1,
            false,
            1,
            "rollout", // <-- the mutation
        )
        .expect("rollout driver runs");
        assert_ne!(
            normalize(&baseline),
            normalize(&rollout_priced),
            "swapping the leaf pricer must change the report; if it does not, \
             leaf values are not reaching the tree at all"
        );
    }

    /// Rollout values do not depend on how many threads priced them, and do
    /// not depend on the search seed's stream. Both are load-bearing: the
    /// first makes results reproducible off the cluster, the second is what
    /// makes the fidelity gate meaningful.
    #[test]
    fn an_unpriced_trial_slot_is_refused_rather_than_averaged() {
        // THE DEMONSTRATED FAILING INPUT for the guard that replaced a
        // `debug_assert!`. That assert was compiled out of every shipped
        // `--release` wheel, and a skipped write produced `root_value: nan` with
        // no exception and a bare `NaN` token in the report -- invalid JSON that
        // Python's non-strict `json.loads` accepts, so a decision came off a
        // degenerate tree and would have been banked as a measurement.
        //
        // Reachable from a test because the reduction is its own function: a
        // guard whose only demonstration is "mutate the writer and rebuild" is
        // one nobody can regression-test.
        let clean = [0.0_f32, 1.0, 0.5, 0.5];
        let values = reduce_trials(&clean, 2, 2).expect("a fully priced matrix reduces");
        assert_eq!(values, vec![0.5, 0.5]);

        for (label, doctored) in [
            ("never written", [f32::NAN, 1.0, 0.5, 0.5]),
            ("infinite", [0.0, f32::INFINITY, 0.5, 0.5]),
            ("last slot", [0.0, 1.0, 0.5, f32::NAN]),
        ] {
            let message = reduce_trials(&doctored, 2, 2)
                .expect_err(&format!("{label}: a non-finite slot must be refused"));
            assert!(
                message.contains("was never priced"),
                "{label}: refusal must name the cause, got {message}"
            );
            assert!(
                message.contains("row") && message.contains("trial"),
                "{label}: refusal must locate the slot, got {message}"
            );
        }
    }

    #[test]
    fn thread_count_does_not_change_values() {
        let one = puct_search_multi_rollout(
            MINIMAL.trim(), 200, 2, 1.4, 3, true, None,
            8, 60, "uniform", 4242, 1, false, 1, "rollout",
        )
        .expect("runs");
        let many = puct_search_multi_rollout(
            MINIMAL.trim(), 200, 2, 1.4, 3, true, None,
            8, 60, "uniform", 4242, 6, false, 1, "rollout",
        )
        .expect("runs");
        let strip = |r: &str| {
            let v: serde_json::Value = serde_json::from_str(r).unwrap();
            let mut m = v.as_object().unwrap().clone();
            for k in ["elapsed_s", "iterations_per_s", "rollout_threads"] {
                m.remove(k);
            }
            serde_json::to_string(&serde_json::Value::Object(m)).unwrap()
        };
        assert_eq!(strip(&one), strip(&many), "thread count changed the search");
    }

    /// A different `rollout_seed` moves the rollout labels (so the seed is
    /// wired), while `rollouts` raising R shrinks the label's spread. The
    /// first half is the guard's failing input for "rollout_seed is ignored".
    #[test]
    fn rollout_seed_is_wired_and_r_reduces_spread() {
        let root_value = |rollout_seed: u64, r: u32| -> f64 {
            let report = puct_search_multi_rollout(
                MINIMAL.trim(), 300, 2, 1.4, 11, true, None,
                r, 80, "uniform", rollout_seed, 4, false, 1, "rollout",
            )
            .expect("runs");
            let v: serde_json::Value = serde_json::from_str(&report).unwrap();
            v["root_value"].as_f64().unwrap()
        };
        let a = root_value(1, 4);
        let b = root_value(2, 4);
        assert!(
            (a - b).abs() > 1e-9,
            "rollout_seed made no difference ({a} vs {b}) — the seed is not wired"
        );
        let spread = |r: u32| {
            let vals: Vec<f64> = (1..=6).map(|s| root_value(s, r)).collect();
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            (vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64).sqrt()
        };
        let low_r = spread(2);
        let high_r = spread(32);
        assert!(
            high_r < low_r,
            "raising R from 2 to 32 did not shrink the root-value spread \
             ({high_r} >= {low_r}) — rollouts are not averaging"
        );
    }

    /// Rejections at the Python boundary, each with the input that trips it.
    #[test]
    fn boundary_rejections_read_false() {
        let call = |rollouts: u32, max_plies: u32, policy: &str, batch: usize, mode: &str| {
            puct_search_multi_rollout(
                MINIMAL.trim(), 10, 2, 1.4, 0, true, None,
                rollouts, max_plies, policy, 0, 1, false, batch, mode,
            )
        };
        assert!(call(0, 10, "uniform", 1, "rollout").is_err(), "rollouts=0 accepted");
        assert!(call(4, 0, "uniform", 1, "rollout").is_err(), "max_plies=0 accepted");
        assert!(call(4, 10, "greedy", 1, "rollout").is_err(), "unknown policy accepted");
        assert!(call(4, 10, "uniform", 0, "rollout").is_err(), "leaf_batch=0 accepted");
        assert!(call(4, 10, "uniform", 1, "oracle").is_err(), "unknown leaf_mode accepted");
        assert!(call(4, 10, "uniform", 1, "rollout").is_ok(), "valid call rejected");
    }

    /// The cap-hit accounting is honest: a 1-ply cap makes essentially every
    /// rollout a fallback, and the report must SAY so rather than presenting
    /// the blend as an oracle.
    #[test]
    fn fallback_fraction_reports_the_blend() {
        let report = puct_search_multi_rollout(
            MINIMAL.trim(), 100, 2, 1.4, 1, true, None,
            4, 1, "uniform", 7, 1, false, 1, "rollout",
        )
        .expect("runs");
        let v: serde_json::Value = serde_json::from_str(&report).unwrap();
        let fallback = v["rollout_fallback_fraction"].as_f64().unwrap();
        assert!(
            fallback > 0.9,
            "a 1-ply cap must report a near-total fallback fraction, got {fallback}"
        );
        assert!(v["rollouts_run"].as_u64().unwrap() > 0);
    }

    /// The oracle's terminal reading is exact where the truth is known: from
    /// the analytic fixture a guaranteed side-one KO must price at 1.0 under
    /// ANY rollout policy, because the rollout's first `battle_is_over` check
    /// fires before any random action is taken.
    #[test]
    fn terminal_leaves_price_exactly_under_rollouts() {
        let report = puct_search_multi_rollout(
            ANALYTIC_TOXIC.trim(), 400, 1, 1.4, 0, true, None,
            8, 100, "uniform", 5, 2, false, 1, "rollout",
        )
        .expect("runs");
        let v: serde_json::Value = serde_json::from_str(&report).unwrap();
        let toss = v["side_one"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["move"] == "seismictoss")
            .expect("fixture has seismictoss");
        let q = toss["q"].as_f64().unwrap();
        assert!(
            (q - 1.0).abs() < 1e-6,
            "guaranteed KO priced at {q} under rollout leaves (must stay exactly 1.0 — \
             terminal branches never reach the rollout pricer)"
        );
    }
}
