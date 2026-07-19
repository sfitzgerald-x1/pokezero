//! pokezero-search: native search crate skeleton (engine-swap stream S1).
//!
//! Proves the native-loop throughput regime over the vendored, gen3-patched
//! poke-engine (`third_party/poke-engine-src/`, fetched by
//! `scripts/vendor_poke_engine_src.sh`) and establishes the eval-hook shape
//! that poke-engine's built-in MCTS lacks: leaf values flow through the
//! pluggable [`LeafEval`] trait, which is where the learned model (batched
//! TorchScript/ONNX per docs/test_time_search_plan_v3.md) plugs in later.
//! Search quality is explicitly NOT the goal of this skeleton.

pub mod encoder;
pub mod events;
pub mod fold;
#[cfg(feature = "model")]
pub mod model;
pub mod tree;

use std::hint::black_box;
use std::time::Instant;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::MoveChoice;
use poke_engine::instruction::StateInstructions;
use poke_engine::pokemon::PokemonName;
use poke_engine::state::{Side, State};

// ---------------------------------------------------------------------------
// Leaf evaluation hook
// ---------------------------------------------------------------------------

/// Pluggable leaf evaluator — the hook poke-engine's native MCTS lacks.
///
/// Implementations return a value in `[0, 1]` from side one's perspective.
/// The trivial [`HpFractionEval`] stands in until the learned model (batched
/// leaf inference) is wired behind this trait.
pub trait LeafEval {
    fn eval(&self, state: &State) -> f32;
}

/// Trivial placeholder evaluator: team HP-fraction differential, mapped to
/// `[0, 1]` from side one's perspective.
pub struct HpFractionEval;

fn side_hp_fraction(side: &Side) -> f32 {
    let mut hp = 0.0f32;
    let mut maxhp = 0.0f32;
    for p in side.pokemon.into_iter() {
        if p.id == PokemonName::NONE || p.maxhp <= 0 {
            continue;
        }
        hp += p.hp.max(0) as f32;
        maxhp += p.maxhp as f32;
    }
    if maxhp <= 0.0 {
        0.5
    } else {
        hp / maxhp
    }
}

impl LeafEval for HpFractionEval {
    fn eval(&self, state: &State) -> f32 {
        0.5 + 0.5 * (side_hp_fraction(&state.side_one) - side_hp_fraction(&state.side_two))
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn parse_move(name: &str, side: &Side, label: &str) -> PyResult<MoveChoice> {
    MoveChoice::from_string(name, side)
        .ok_or_else(|| PyValueError::new_err(format!("invalid move for {label}: {name}")))
}

/// Match the poke-engine Python binding's display convention: switches render
/// as `switch <name>` so returned strings round-trip through
/// `MoveChoice::from_string`.
fn move_display(side: &Side, choice: &MoveChoice) -> String {
    match choice {
        MoveChoice::Switch(_) => format!("switch {}", choice.to_string(side)),
        _ => choice.to_string(side),
    }
}

fn sample_branch<'a>(
    rng: &mut StdRng,
    branches: &'a [StateInstructions],
) -> &'a StateInstructions {
    let total: f32 = branches.iter().map(|b| b.percentage).sum();
    if total <= 0.0 {
        return &branches[0];
    }
    let mut roll = rng.random_range(0.0..total);
    for branch in branches {
        if roll < branch.percentage {
            return branch;
        }
        roll -= branch.percentage;
    }
    branches.last().expect("non-empty branches")
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// bench_apply_reverse: the throughput probe
// ---------------------------------------------------------------------------

/// Measure the in-Rust simulation-step rate: parse `state_str` once, then loop
/// `iterations` times over generate_instructions -> apply -> cheap value read
/// -> reverse (the first generated branch each iteration, deterministic).
/// Returns iterations per second measured entirely inside Rust — no Python/FFI
/// crossings inside the loop.
///
/// `branch_on_damage=true` matches what poke-engine's own MCTS does for the
/// top two plies (mcts.rs root||parent.root; threaded MCTS_DAMAGE_BRANCH_DEPTH=2);
/// depth >= 2 runs with it off, so pass `false` to price the deep-tree regime.
/// Parse a serialized engine state, converting poke-engine's internal panics
/// on malformed input into a catchable `ValueError` instead of a
/// `PanicException` (which `except Exception` does not catch).
fn parse_state(state_str: &str) -> PyResult<State> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| State::deserialize(state_str)))
        .map_err(|_| PyValueError::new_err("state_str is not a valid poke-engine state string"))
}

#[pyfunction]
#[pyo3(signature = (state_str, s1_move, s2_move, iterations, branch_on_damage = true))]
fn bench_apply_reverse(
    state_str: &str,
    s1_move: &str,
    s2_move: &str,
    iterations: usize,
    branch_on_damage: bool,
) -> PyResult<f64> {
    if iterations == 0 {
        return Err(PyValueError::new_err("iterations must be > 0"));
    }
    let mut state = parse_state(state_str)?;
    let s1 = parse_move(s1_move, &state.side_one, "side one")?;
    let s2 = parse_move(s2_move, &state.side_two, "side two")?;
    let evaluator = HpFractionEval;

    let mut acc = 0.0f64;
    let start = Instant::now();
    for _ in 0..iterations {
        let branches = generate_instructions_from_move_pair(&mut state, &s1, &s2, branch_on_damage);
        if let Some(branch) = branches.first() {
            state.apply_instructions(&branch.instruction_list);
            acc += f64::from(evaluator.eval(&state));
            state.reverse_instructions(&branch.instruction_list);
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    black_box(acc);
    if elapsed <= 0.0 {
        return Err(PyValueError::new_err("elapsed time was zero; raise iterations"));
    }
    Ok(iterations as f64 / elapsed)
}

// ---------------------------------------------------------------------------
// puct_search: one-ply PUCT skeleton over engine primitives
// ---------------------------------------------------------------------------

struct MoveStats {
    display: String,
    prior: f32,
    visits: u32,
    total_value: f32,
}

impl MoveStats {
    fn mean(&self) -> f32 {
        if self.visits == 0 {
            0.5 // first-play urgency: neutral value
        } else {
            self.total_value / self.visits as f32
        }
    }

    /// PUCT score. `for_side_one` flips Q so side two minimizes side one's value.
    fn puct(&self, parent_visits: u32, c_puct: f32, for_side_one: bool) -> f32 {
        let q = if for_side_one {
            self.mean()
        } else {
            1.0 - self.mean()
        };
        let u = c_puct * self.prior * ((parent_visits as f32).sqrt() / (1.0 + self.visits as f32));
        q + u
    }
}

fn make_stats(side: &Side, options: &[MoveChoice]) -> Vec<MoveStats> {
    let prior = 1.0 / options.len() as f32;
    options
        .iter()
        .map(|choice| MoveStats {
            display: move_display(side, choice),
            prior,
            visits: 0,
            total_value: 0.0,
        })
        .collect()
}

fn select(stats: &[MoveStats], parent_visits: u32, c_puct: f32, for_side_one: bool) -> usize {
    let mut best = 0;
    let mut best_score = f32::MIN;
    for (index, stat) in stats.iter().enumerate() {
        let score = stat.puct(parent_visits, c_puct, for_side_one);
        if score > best_score {
            best_score = score;
            best = index;
        }
    }
    best
}

/// Run the core loop with a caller-supplied leaf evaluator. Kept generic over
/// [`LeafEval`] so the learned-model evaluator slots in without touching the
/// tree logic.
pub(crate) fn puct_search_with_eval<E: LeafEval>(
    state: &mut State,
    iterations: usize,
    c_puct: f32,
    seed: u64,
    evaluator: &E,
) -> PyResult<(Vec<MoveStats>, Vec<MoveStats>, f64)> {
    let (s1_options, s2_options) = state.root_get_all_options();
    if s1_options.is_empty() || s2_options.is_empty() {
        return Err(PyValueError::new_err("no legal root options for one or both sides"));
    }
    let mut s1_stats = make_stats(&state.side_one, &s1_options);
    let mut s2_stats = make_stats(&state.side_two, &s2_options);
    let mut rng = StdRng::seed_from_u64(seed);

    let start = Instant::now();
    for parent_visits in 0..iterations as u32 {
        let i = select(&s1_stats, parent_visits, c_puct, true);
        let j = select(&s2_stats, parent_visits, c_puct, false);
        let branches =
            generate_instructions_from_move_pair(state, &s1_options[i], &s2_options[j], true);
        let value = if branches.is_empty() {
            evaluator.eval(state)
        } else {
            let branch = sample_branch(&mut rng, &branches);
            state.apply_instructions(&branch.instruction_list);
            let outcome = state.battle_is_over();
            let value = if outcome == 0.0 {
                evaluator.eval(state)
            } else if outcome > 0.0 {
                1.0
            } else {
                0.0
            };
            state.reverse_instructions(&branch.instruction_list);
            value
        };
        s1_stats[i].visits += 1;
        s1_stats[i].total_value += value;
        s2_stats[j].visits += 1;
        s2_stats[j].total_value += value;
    }
    let elapsed = start.elapsed().as_secs_f64();
    Ok((s1_stats, s2_stats, elapsed))
}

fn stats_to_json(stats: &[MoveStats]) -> String {
    let mut order: Vec<usize> = (0..stats.len()).collect();
    order.sort_by(|&a, &b| stats[b].visits.cmp(&stats[a].visits));
    let entries: Vec<String> = order
        .iter()
        .map(|&index| {
            let stat = &stats[index];
            format!(
                "{{\"move\":\"{}\",\"visits\":{},\"q\":{:.6}}}",
                json_escape(&stat.display),
                stat.visits,
                stat.mean()
            )
        })
        .collect();
    format!("[{}]", entries.join(","))
}

/// Minimal one-ply PUCT over engine primitives with the trivial HP-fraction
/// leaf evaluator. Decoupled selection: each side independently maximizes its
/// own PUCT score (side two on `1 - value`), matching the simultaneous-move
/// shape of poke-engine's own MCTS but with the leaf priced through
/// [`LeafEval`]. Returns a JSON report with visit counts per root move.
///
/// This is an architecture skeleton (eval hook + tree stats + stochastic
/// branch sampling), not a strength claim: one ply deep, uniform priors.
#[pyfunction]
#[pyo3(signature = (state_str, iterations, c_puct = 1.4, seed = 0))]
fn puct_search(state_str: &str, iterations: usize, c_puct: f32, seed: u64) -> PyResult<String> {
    if iterations == 0 {
        return Err(PyValueError::new_err("iterations must be > 0"));
    }
    let mut state = parse_state(state_str)?;
    let evaluator = HpFractionEval;
    let (s1_stats, s2_stats, elapsed) =
        puct_search_with_eval(&mut state, iterations, c_puct, seed, &evaluator)?;
    let iterations_per_s = if elapsed > 0.0 {
        iterations as f64 / elapsed
    } else {
        return Err(PyValueError::new_err("elapsed time was zero; raise iterations"))
    };
    Ok(format!(
        "{{\"iterations\":{},\"evaluator\":\"hp_fraction\",\"c_puct\":{},\"seed\":{},\
         \"elapsed_s\":{:.6},\"iterations_per_s\":{:.1},\
         \"side_one\":{},\"side_two\":{}}}",
        iterations,
        c_puct,
        seed,
        elapsed,
        iterations_per_s,
        stats_to_json(&s1_stats),
        stats_to_json(&s2_stats),
    ))
}

#[pymodule]
fn pokezero_search(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("ENGINE_FEATURES", "gen3 (residual-order patched)")?;
    // True when the crate was built with `--features model` (tch-rs linked).
    m.add("MODEL_FEATURE_ENABLED", cfg!(feature = "model"))?;
    m.add_function(wrap_pyfunction!(bench_apply_reverse, m)?)?;
    m.add_function(wrap_pyfunction!(puct_search, m)?)?;
    m.add_function(wrap_pyfunction!(tree::puct_search_multi, m)?)?;
    m.add_function(wrap_pyfunction!(events::branch_events, m)?)?;
    m.add_function(wrap_pyfunction!(encoder::encode_decision, m)?)?;
    m.add_class::<encoder::NativeEncoder>()?;
    m.add_class::<fold::PyFoldState>()?;
    #[cfg(feature = "model")]
    m.add_class::<model::NativeLeafModel>()?;
    Ok(())
}
