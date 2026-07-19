//! In-crate TorchScript leaf evaluation (track D of the engine swap).
//!
//! Loads the TorchScript artifact produced by `scripts/export_model.py`
//! (positional-args shim: 5 inputs, 3 outputs — see the export manifest) and
//! runs batched leaf evaluation entirely inside the crate, so MCTS leaf
//! pricing never crosses the Python bridge per-leaf.
//!
//! Input boundary (deliberate): [`TorchScriptLeafEval`] consumes PRE-ENCODED
//! observation tensors. The v2.2 encoder in Rust is a separate in-flight
//! stream (fold-state refactor, track B); until it lands, callers supply
//! encoded observations from Python (golden-corpus rows / template tensors)
//! and the search loop stub-encodes leaves by copying a caller-supplied
//! template row. Forward cost is value-independent, so throughput numbers
//! are real; leaf OBSERVATION CONTENT is not (values/priors are placeholders
//! until the Rust encoder plugs into exactly this boundary).
//!
//! Value contract: the checkpoint's value head is tanh-activated ([-1, 1],
//! searching seat = side one). The tree operates on [0, 1] side-one win
//! probability, so values are mapped v01 = (v + 1) / 2 before backprop.
//! Priors are masked softmax over the policy logits under a caller-supplied
//! legal-action mask (action schema v1: 9 actions). Mapping action indices
//! onto poke-engine `MoveChoice`s is encoder-stream territory; until then the
//! batched search consumes model VALUES and keeps uniform priors.

use std::time::Instant;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tch::{CModule, Device, IValue, Kind, TchError, Tensor};

use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::MoveChoice;
use poke_engine::instruction::Instruction;
use poke_engine::state::State;

use crate::tree::{
    apply_self_priors, finalize, multiply_report_json, traverse, BranchSeam, LeafPrice,
    MultiPlyConfig, MultiPlyOutcome, SearchCounters, Traversal, Tree,
};
use crate::{make_stats, parse_state, sample_branch, select, stats_to_json};

/// Observation tensor shape (v2.2 export contract; see export_manifest.json).
#[derive(Clone, Copy, Debug)]
pub struct ObsSpec {
    pub window: i64,
    pub tokens: i64,
    pub categorical_features: i64,
    pub numeric_features: i64,
}

impl ObsSpec {
    fn per_row(&self) -> (usize, usize, usize, usize, usize) {
        let wt = (self.window * self.tokens) as usize;
        (
            wt * self.categorical_features as usize, // categorical_ids
            wt * self.numeric_features as usize,     // numeric_features
            wt,                                      // token_type_ids
            wt,                                      // attention_mask
            self.window as usize,                    // history_mask
        )
    }
}

fn tch_err(error: TchError) -> PyErr {
    PyValueError::new_err(format!("tch: {error}"))
}

fn parse_device(device: &str) -> PyResult<Device> {
    match device {
        "cpu" => Ok(Device::Cpu),
        "mps" => Ok(Device::Mps),
        "cuda" => Ok(Device::Cuda(0)),
        other => Err(PyValueError::new_err(format!(
            "unsupported device {other:?}: expected cpu, mps, or cuda"
        ))),
    }
}

/// One pre-encoded observation batch (CPU tensors, batch-major).
pub struct ObsBatch {
    pub categorical_ids: Tensor,  // [n, w, t, c] int64
    pub numeric_features: Tensor, // [n, w, t, f] float32
    pub token_type_ids: Tensor,   // [n, w, t] int64
    pub attention_mask: Tensor,   // [n, w, t] bool
    pub history_mask: Tensor,     // [n, w] bool
}

impl ObsBatch {
    fn from_flat(
        spec: &ObsSpec,
        batch: i64,
        categorical_ids: &[i64],
        numeric_features: &[f32],
        token_type_ids: &[i64],
        attention_mask: &[bool],
        history_mask: &[bool],
    ) -> PyResult<Self> {
        let (cat_len, num_len, tok_len, attn_len, hist_len) = spec.per_row();
        let n = batch as usize;
        let check = |name: &str, got: usize, want: usize| -> PyResult<()> {
            if got != want {
                return Err(PyValueError::new_err(format!(
                    "{name}: expected {want} elements for batch {batch} at spec {spec:?}, got {got}"
                )));
            }
            Ok(())
        };
        check("categorical_ids", categorical_ids.len(), n * cat_len)?;
        check("numeric_features", numeric_features.len(), n * num_len)?;
        check("token_type_ids", token_type_ids.len(), n * tok_len)?;
        check("attention_mask", attention_mask.len(), n * attn_len)?;
        check("history_mask", history_mask.len(), n * hist_len)?;
        let (w, t) = (spec.window, spec.tokens);
        Ok(Self {
            categorical_ids: Tensor::from_slice(categorical_ids).reshape([
                batch,
                w,
                t,
                spec.categorical_features,
            ]),
            numeric_features: Tensor::from_slice(numeric_features).reshape([
                batch,
                w,
                t,
                spec.numeric_features,
            ]),
            token_type_ids: Tensor::from_slice(token_type_ids).reshape([batch, w, t]),
            attention_mask: Tensor::from_slice(attention_mask).reshape([batch, w, t]),
            history_mask: Tensor::from_slice(history_mask).reshape([batch, w]),
        })
    }

    /// Preallocate an all-zeros batch (first attention/history entries true so
    /// no row is fully masked even before a template lands in it).
    fn zeros(spec: &ObsSpec, batch: i64) -> Self {
        let (w, t) = (spec.window, spec.tokens);
        let attention_mask = Tensor::zeros([batch, w, t], (Kind::Bool, Device::Cpu));
        let _ = attention_mask.narrow(2, 0, 1).fill_(1);
        Self {
            categorical_ids: Tensor::zeros(
                [batch, w, t, spec.categorical_features],
                (Kind::Int64, Device::Cpu),
            ),
            numeric_features: Tensor::zeros(
                [batch, w, t, spec.numeric_features],
                (Kind::Float, Device::Cpu),
            ),
            token_type_ids: Tensor::zeros([batch, w, t], (Kind::Int64, Device::Cpu)),
            attention_mask,
            history_mask: Tensor::ones([batch, w], (Kind::Bool, Device::Cpu)),
        }
    }

    /// Copy `template` (batch 1) into row `row` — the stub-encode step the
    /// Rust encoder will replace with a real per-leaf encode.
    fn write_row(&mut self, row: i64, template: &ObsBatch) {
        self.categorical_ids
            .narrow(0, row, 1)
            .copy_(&template.categorical_ids);
        self.numeric_features
            .narrow(0, row, 1)
            .copy_(&template.numeric_features);
        self.token_type_ids
            .narrow(0, row, 1)
            .copy_(&template.token_type_ids);
        self.attention_mask
            .narrow(0, row, 1)
            .copy_(&template.attention_mask);
        self.history_mask
            .narrow(0, row, 1)
            .copy_(&template.history_mask);
    }

    fn narrow_rows(&self, n: i64) -> ObsBatch {
        ObsBatch {
            categorical_ids: self.categorical_ids.narrow(0, 0, n),
            numeric_features: self.numeric_features.narrow(0, 0, n),
            token_type_ids: self.token_type_ids.narrow(0, 0, n),
            attention_mask: self.attention_mask.narrow(0, 0, n),
            history_mask: self.history_mask.narrow(0, 0, n),
        }
    }

    /// Build a batch from REAL encoded leaf observations (the track-B leaf
    /// encoder's output; window is 1 on the v2.2 contract, history all-true).
    fn from_encoded(spec: &ObsSpec, rows: &[crate::encoder::EncodedArrays]) -> PyResult<ObsBatch> {
        let n = rows.len();
        let (cat_len, num_len, tok_len, _attn_len, hist_len) = spec.per_row();
        let mut categorical: Vec<i64> = Vec::with_capacity(n * cat_len);
        let mut numeric: Vec<f32> = Vec::with_capacity(n * num_len);
        let mut token_types: Vec<i64> = Vec::with_capacity(n * tok_len);
        let mut attention: Vec<bool> = Vec::with_capacity(n * tok_len);
        let history: Vec<bool> = vec![true; n * hist_len];
        for row in rows {
            if row.categorical.len() != cat_len || row.numeric.len() != num_len {
                return Err(PyValueError::new_err(format!(
                    "encoded leaf shape mismatch: got {}x cat / {}x num, spec wants {cat_len}/{num_len}",
                    row.categorical.len(),
                    row.numeric.len()
                )));
            }
            categorical.extend(row.categorical.iter().map(|v| *v as i64));
            numeric.extend(row.numeric.iter().map(|v| *v as f32));
            token_types.extend(row.token_types.iter().map(|v| *v as i64));
            attention.extend(row.attention.iter().map(|v| *v != 0));
        }
        ObsBatch::from_flat(
            spec,
            n as i64,
            &categorical,
            &numeric,
            &token_types,
            &attention,
            &history,
        )
    }
}

/// Raw model outputs for one evaluated batch (CPU, float32).
pub struct LeafBatchOutput {
    /// tanh value head output, [-1, 1], one per row.
    pub values_tanh: Vec<f32>,
    /// Tree-space values (v + 1) / 2, [0, 1] side-one win probability.
    pub values01: Vec<f32>,
    /// Flat [n, action_count] policy logits.
    pub policy_logits: Vec<f32>,
    /// Flat [n, action_count] priors: softmax over logits, masked to the
    /// caller's legal actions when a mask is supplied.
    pub priors: Vec<f32>,
    pub action_count: usize,
}

/// Batched leaf evaluator boundary: pre-encoded observations in, values +
/// priors out. The learned model implements it via TorchScript; the Rust
/// encoder plugs in UPSTREAM of this trait (it produces the `ObsBatch`).
pub trait BatchLeafEval {
    fn eval_batch(&self, obs: &ObsBatch, legal_mask: Option<&Tensor>) -> PyResult<LeafBatchOutput>;
}

/// TorchScript-backed leaf evaluator (tch-rs `CModule`).
///
/// Artifacts are PER-DEVICE: `torch.jit.trace` bakes device constants
/// (docs/model_export_findings.md), so a CPU artifact must run on CPU and an
/// MPS/CUDA deployment needs an artifact traced on that device.
pub struct TorchScriptLeafEval {
    module: CModule,
    device: Device,
    spec: ObsSpec,
}

impl TorchScriptLeafEval {
    pub fn load(path: &str, device: Device, spec: ObsSpec) -> PyResult<Self> {
        let mut module = CModule::load_on_device(path, device).map_err(tch_err)?;
        module.set_eval();
        Ok(Self {
            module,
            device,
            spec,
        })
    }

    pub fn spec(&self) -> ObsSpec {
        self.spec
    }

    /// Run the traced forward on `obs`, returning (policy_logits, value) on CPU.
    fn forward(&self, obs: &ObsBatch) -> PyResult<(Tensor, Tensor)> {
        let outputs = tch::no_grad(|| {
            self.module.forward_is(&[
                IValue::Tensor(obs.categorical_ids.to_device(self.device)),
                IValue::Tensor(obs.numeric_features.to_device(self.device)),
                IValue::Tensor(obs.token_type_ids.to_device(self.device)),
                IValue::Tensor(obs.attention_mask.to_device(self.device)),
                IValue::Tensor(obs.history_mask.to_device(self.device)),
            ])
        })
        .map_err(tch_err)?;
        let IValue::Tuple(mut outputs) = outputs else {
            return Err(PyValueError::new_err(
                "TorchScript artifact did not return a tuple (wrong artifact? re-export via scripts/export_model.py)",
            ));
        };
        if outputs.len() != 3 {
            return Err(PyValueError::new_err(format!(
                "TorchScript artifact returned {} outputs, expected 3 (policy_logits, value, opponent_action_logits)",
                outputs.len()
            )));
        }
        // Field order matches scripts/export_model.py OUTPUT_NAMES.
        let _opponent = outputs.pop();
        let value = match outputs.pop() {
            Some(IValue::Tensor(tensor)) => tensor,
            _ => return Err(PyValueError::new_err("value output is not a tensor")),
        };
        let policy_logits = match outputs.pop() {
            Some(IValue::Tensor(tensor)) => tensor,
            _ => {
                return Err(PyValueError::new_err(
                    "policy_logits output is not a tensor",
                ))
            }
        };
        Ok((
            policy_logits.to_device(Device::Cpu).to_kind(Kind::Float),
            value.to_device(Device::Cpu).to_kind(Kind::Float),
        ))
    }
}

impl BatchLeafEval for TorchScriptLeafEval {
    fn eval_batch(&self, obs: &ObsBatch, legal_mask: Option<&Tensor>) -> PyResult<LeafBatchOutput> {
        let (policy_logits, value) = self.forward(obs)?;
        let action_count = *policy_logits
            .size()
            .last()
            .ok_or_else(|| PyValueError::new_err("policy_logits has no dimensions"))?
            as usize;
        let masked = match legal_mask {
            Some(mask) => {
                // Fail loudly on a fully-illegal row: softmax over all -inf
                // is NaN, and a row with zero legal actions is a caller bug
                // (real decisions always offer at least one legal action).
                // Erroring — not a uniform fallback — keeps a broken mask
                // construction from silently shaping priors once priors are
                // wired into selection.
                let rows_with_legal = mask.any_dim(-1, false);
                if rows_with_legal.all().int64_value(&[]) == 0 {
                    let row = rows_with_legal.logical_not().nonzero().int64_value(&[0, 0]);
                    return Err(PyValueError::new_err(format!(
                        "legal_mask row {row} has zero legal actions; priors would be NaN \
                         (fix the caller's mask construction)"
                    )));
                }
                policy_logits.masked_fill(&mask.logical_not(), f64::NEG_INFINITY)
            }
            None => policy_logits.shallow_clone(),
        };
        let priors = masked.softmax(-1, Kind::Float);
        let values_tanh: Vec<f32> = Vec::try_from(value.flatten(0, -1)).map_err(tch_err)?;
        let values01 = values_tanh.iter().map(|v| 0.5 * (v + 1.0)).collect();
        Ok(LeafBatchOutput {
            values_tanh,
            values01,
            policy_logits: Vec::try_from(policy_logits.flatten(0, -1)).map_err(tch_err)?,
            priors: Vec::try_from(priors.flatten(0, -1)).map_err(tch_err)?,
            action_count,
        })
    }
}

// ---------------------------------------------------------------------------
// Batched one-ply PUCT with model leaf pricing (virtual-loss batching)
// ---------------------------------------------------------------------------

/// Virtual-loss batched search core.
///
/// Design choice (documented tradeoff): leaves are collected with a VIRTUAL
/// LOSS — each selected (s1, s2) arm pair is provisionally scored as a
/// side-one loss (s1: +visit, +0 value; s2: +visit, +1 value) until its model
/// value replaces the provisional one after the batched forward. At one ply,
/// frontier collection WITHOUT virtual loss is degenerate (selection is
/// deterministic given the stats, so all `batch` leaves would be the same arm
/// pair); the virtual loss is the minimal mechanism that makes batched
/// selection well-defined. Fidelity cost: one round of `batch` selections
/// explores wider than `batch` sequential PUCT steps — at `batch >=
/// iterations` the search degrades toward a uniform sweep. Keep `batch <<
/// iterations` (the bench sweeps the throughput side of the tradeoff).
/// `batch = 1` is the sequential regime BY CONSTRUCTION: each round's single
/// virtual loss is replaced by its real value before the next selection, so
/// no selection ever observes provisional stats.
#[allow(clippy::too_many_arguments)]
fn batched_search_core<E: BatchLeafEval>(
    state_str: &str,
    iterations: usize,
    batch_size: usize,
    template: &ObsBatch,
    evaluator: &E,
    spec: &ObsSpec,
    c_puct: f32,
    seed: u64,
) -> PyResult<String> {
    let mut state = parse_state(state_str)?;
    let (s1_options, s2_options) = state.root_get_all_options();
    if s1_options.is_empty() || s2_options.is_empty() {
        return Err(PyValueError::new_err(
            "no legal root options for one or both sides",
        ));
    }
    let mut s1_stats = make_stats(&state.side_one, &s1_options);
    let mut s2_stats = make_stats(&state.side_two, &s2_options);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut batch = ObsBatch::zeros(spec, batch_size as i64);

    let mut completed = 0usize;
    let mut rounds = 0usize;
    let mut model_evals = 0usize;
    let mut terminal_leaves = 0usize;
    let start = Instant::now();
    while completed < iterations {
        let round_size = batch_size.min(iterations - completed);
        // (s1 arm, s2 arm) awaiting a model value, in batch-row order.
        let mut pending: Vec<(usize, usize)> = Vec::with_capacity(round_size);
        // Terminal leaves resolved by the engine outcome, no model call.
        let mut resolved: Vec<(usize, usize, f32)> = Vec::new();
        for offset in 0..round_size {
            let parent_visits = (completed + offset) as u32;
            let i = select(&s1_stats, parent_visits, c_puct, true);
            let j = select(&s2_stats, parent_visits, c_puct, false);
            // Virtual loss: provisional side-one loss until the real value lands.
            s1_stats[i].visits += 1;
            s2_stats[j].visits += 1;
            s2_stats[j].total_value += 1.0;
            let branches = generate_instructions_from_move_pair(
                &mut state,
                &s1_options[i],
                &s2_options[j],
                true,
            );
            if branches.is_empty() {
                // No instructions (e.g. double-switch edge): price the root
                // observation itself — same template stub either way.
                batch.write_row(pending.len() as i64, template);
                pending.push((i, j));
                continue;
            }
            let branch = sample_branch(&mut rng, &branches);
            state.apply_instructions(&branch.instruction_list);
            let outcome = state.battle_is_over();
            if outcome != 0.0 {
                resolved.push((i, j, if outcome > 0.0 { 1.0 } else { 0.0 }));
                terminal_leaves += 1;
            } else {
                // Stub encode: the Rust encoder (track B) will write the real
                // leaf observation here; the copy prices the marshaling.
                batch.write_row(pending.len() as i64, template);
                pending.push((i, j));
            }
            state.reverse_instructions(&branch.instruction_list);
        }
        if !pending.is_empty() {
            let n = pending.len();
            let output = evaluator.eval_batch(&batch.narrow_rows(n as i64), None)?;
            model_evals += n;
            for (row, &(i, j)) in pending.iter().enumerate() {
                let value = output.values01[row];
                s1_stats[i].total_value += value;
                s2_stats[j].total_value += value - 1.0; // replace the virtual loss
            }
        }
        for &(i, j, value) in &resolved {
            s1_stats[i].total_value += value;
            s2_stats[j].total_value += value - 1.0;
        }
        completed += round_size;
        rounds += 1;
    }
    let elapsed = start.elapsed().as_secs_f64();
    if elapsed <= 0.0 {
        return Err(PyValueError::new_err(
            "elapsed time was zero; raise iterations",
        ));
    }
    Ok(format!(
        "{{\"iterations\":{},\"evaluator\":\"torchscript\",\"batch_size\":{},\"rounds\":{},\
         \"model_evals\":{},\"terminal_leaves\":{},\"c_puct\":{},\"seed\":{},\
         \"elapsed_s\":{:.6},\"sims_per_s\":{:.1},\"searches_per_s\":{:.4},\
         \"side_one\":{},\"side_two\":{}}}",
        iterations,
        batch_size,
        rounds,
        model_evals,
        terminal_leaves,
        c_puct,
        seed,
        elapsed,
        completed as f64 / elapsed,
        1.0 / elapsed,
        stats_to_json(&s1_stats),
        stats_to_json(&s2_stats),
    ))
}

// ---------------------------------------------------------------------------
// Batched MULTI-PLY search with model leaf pricing (virtual-loss batching
// through chance nodes)
// ---------------------------------------------------------------------------

/// Multi-ply virtual-loss batched search core (tree mechanics in `tree.rs`).
///
/// Batching-through-chance design (documented choice): traversal resolves a
/// chance node by WEIGHTED SAMPLING over the engine's exact branch
/// probabilities; backup resolves it by EXACT EXPECTATION over the current
/// branch means. All model rows come from EXPANSIONS: expanding a joint edge
/// prices every enumerated outcome (terminal branches exactly, the rest as
/// deferred batch rows), so the edge's first backed-up value is already the
/// exact expectation over model values. Virtual loss (provisional side-one
/// loss on decision arms, provisional visit on traversed branches, deferred
/// rows carried via `ChanceBranch::pending_row`) keeps a round's selections
/// diverse; `finalize` replaces every provisional in collection order.
///
/// A round collects traversals until either `batch_size` traversals or
/// `batch_size` pending rows accumulate (an expansion never splits across
/// rounds, so the final expansion may overshoot the row budget by its branch
/// count); the observation batch is then sized to the exact pending-row
/// count. Rows are template-stub encoded — the Rust encoder (track B) will
/// replace the row write with a real per-leaf encode at the same seam.
#[allow(clippy::too_many_arguments)]
fn multiply_batched_core<E: BatchLeafEval>(
    state_str: &str,
    iterations: usize,
    batch_size: usize,
    max_depth: u8,
    deep_ko_split: bool,
    template: &ObsBatch,
    evaluator: &E,
    spec: &ObsSpec,
    c_puct: f32,
    seed: u64,
) -> PyResult<String> {
    let mut state = parse_state(state_str)?;
    if state.battle_is_over() != 0.0 {
        return Err(PyValueError::new_err("battle is already over at the root"));
    }
    let cfg = MultiPlyConfig {
        max_depth,
        c_puct,
        deep_ko_split,
    };
    let mut tree = Tree::from_root(&state)?;
    let mut counters = SearchCounters::default();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut completed = 0usize;
    let mut rounds = 0usize;
    let mut model_evals = 0usize;
    let start = Instant::now();
    while completed < iterations {
        let traversal_budget = batch_size.min(iterations - completed);
        let mut traversals: Vec<Traversal> = Vec::with_capacity(traversal_budget);
        let mut pending_rows = 0usize;
        while traversals.len() < traversal_budget && pending_rows < batch_size {
            let traversal = traverse(
                &mut tree,
                &mut state,
                &mut rng,
                &cfg,
                &mut counters,
                &mut |_leaf: &State, _seam: &BranchSeam| {
                    // Template-stub encode happens after collection (content
                    // is state-independent until the track-B encoder lands);
                    // this closure only claims the batch row. `_seam` carries
                    // the joint move pair + the branch's instruction list —
                    // the in-crate encoder integration renders the branch's
                    // events (events::render_branch_events), advances a clone
                    // of the root fold state, and writes the REAL leaf
                    // observation for this row at exactly this seam.
                    let row = pending_rows;
                    pending_rows += 1;
                    LeafPrice::Deferred(row)
                },
            );
            traversals.push(traversal);
        }
        let row_values = if pending_rows > 0 {
            let mut batch = ObsBatch::zeros(spec, pending_rows as i64);
            for row in 0..pending_rows {
                batch.write_row(row as i64, template);
            }
            let output = evaluator.eval_batch(&batch, None)?;
            model_evals += pending_rows;
            output.values01
        } else {
            Vec::new()
        };
        for traversal in &traversals {
            finalize(&mut tree, traversal, &row_values);
        }
        completed += traversals.len();
        rounds += 1;
    }
    let elapsed_s = start.elapsed().as_secs_f64();
    let outcome = MultiPlyOutcome {
        tree,
        counters,
        elapsed_s,
    };
    let extra = format!(
        "\"batch_size\":{},\"rounds\":{},\"model_evals\":{}",
        batch_size, rounds, model_evals
    );
    Ok(multiply_report_json(
        &outcome,
        iterations,
        &cfg,
        seed,
        "torchscript",
        &extra,
    ))
}

// ---------------------------------------------------------------------------
// Batched MULTI-PLY search with REAL leaf observations (the capstone loop:
// root state -> tree -> branch -> synthesized events -> fold advance ->
// native encode -> batched model eval -> exact-expectation backup)
// ---------------------------------------------------------------------------

/// Per-branch leaf record: the branch's advanced fold state, battle turn,
/// and evolved self-team display order (Showdown switch-swap semantics).
/// NOTE: entries are never pruned within one search — memory grows with the
/// priced-branch count (fold state ~O(100KB) worst case each); fine at the
/// intended <=8192-sim budgets, revisit before very large searches.
struct BranchFold {
    fold: crate::fold::FoldStateInner,
    turn: i64,
    self_order: Vec<String>,
    meta: crate::leaf::LeafMeta,
}

/// Gather a leaf's action-block priors onto the acting seat's option list.
///
/// `priors_row` is one row of the UNMASKED softmax; the gathered subset is
/// renormalized over the mapped options — mathematically identical to the
/// masked softmax restricted to those actions (exp(l_i)/Σ_mapped exp(l_j)).
/// Returns `None` — leaving the node uniform — when any option lacks an
/// action-block slot (the whole node falls back rather than zeroing arms the
/// model cannot see) or the mapped mass underflows.
fn gather_self_priors(priors_row: &[f32], map: &[Option<usize>]) -> Option<Vec<f32>> {
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
    if sum <= 1e-8 {
        return None;
    }
    for prior in &mut gathered {
        *prior /= sum;
    }
    Some(gathered)
}

/// The acting seat's option list at a decision node, mirroring
/// `new_decision_node`'s defensive empty-side handling so prior vectors align
/// with the node's own arm order.
fn self_options_at(state: &State, self_side_one: bool) -> Vec<MoveChoice> {
    let (s1_options, s2_options) = state.get_all_options();
    let mut options = if self_side_one { s1_options } else { s2_options };
    if options.is_empty() {
        options.push(MoveChoice::None);
    }
    options
}

fn is_single_none(options: &[MoveChoice]) -> bool {
    options.len() == 1 && matches!(options[0], MoveChoice::None)
}

/// Multi-ply batched search where every model row is a REAL leaf observation:
/// each priced branch renders its instruction list as protocol lines
/// (`events::render_branch_events`), advances a CLONE of its parent's fold
/// state over them (per-outcome history — the crit branch's history shows the
/// crit), and encodes the leaf natively (engine-state recompute +
/// world-constant carry + fold products). Fold states chain through the tree
/// via the `BranchSeam` parent key, so depth-k leaves carry the full
/// root-prefix + k simulated plies of history.
///
/// With `model_priors` set (the default), the ACTING seat's decision arms use
/// the model's policy priors instead of uniform: the root node from one extra
/// forward on the root observation, each interior node from the priors of the
/// branch observation it grows under (mapped option-by-option via
/// `LeafContext::self_action_map`, the leaf encoder's own action-block
/// correspondence). The OPPONENT side stays uniform in this integration —
/// opponent-perspective observations from a determinized world need a belief
/// design decision first (see docs/crate_search_design.md, "Model priors").
/// Priors reweight exploration only; values and the exact-expectation backup
/// are untouched.
#[allow(clippy::too_many_arguments)]
fn multiply_batched_encoded_core<E: BatchLeafEval>(
    state_str: &str,
    iterations: usize,
    batch_size: usize,
    max_depth: u8,
    deep_ko_split: bool,
    model_priors: bool,
    leaf_ctx: &crate::leaf::LeafContext,
    event_ctx: &crate::events::EventContext,
    root_fold: &crate::fold::FoldStateInner,
    evaluator: &E,
    spec: &ObsSpec,
    c_puct: f32,
    seed: u64,
) -> PyResult<String> {
    let mut state = parse_state(state_str)?;
    if state.battle_is_over() != 0.0 {
        return Err(PyValueError::new_err("battle is already over at the root"));
    }
    let cfg = MultiPlyConfig {
        max_depth,
        c_puct,
        deep_ko_split,
    };
    let mut tree = Tree::from_root(&state)?;
    let mut counters = SearchCounters::default();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut completed = 0usize;
    let mut rounds = 0usize;
    let mut model_evals = 0usize;
    let mut lossy_renders = 0usize;
    let mut fold_by_branch: std::collections::HashMap<(usize, usize), BranchFold> =
        std::collections::HashMap::new();
    let root_turn = event_ctx.turn;
    let self_side_one = leaf_ctx.self_is_side_one();
    // Prior wiring telemetry: nodes whose acting-side priors came from the
    // model vs fallbacks to uniform (unmapped option / underflow / mismatch).
    let mut prior_branches = 0usize;
    let mut prior_fallbacks = 0usize;
    let mut root_priors: Option<Vec<f32>> = None;
    if model_priors {
        let root_options = if self_side_one {
            tree.decisions[0].s1_options.clone()
        } else {
            tree.decisions[0].s2_options.clone()
        };
        if !is_single_none(&root_options) {
            // One extra forward on the ROOT observation prices the root
            // node's priors (the root is never a chance-branch leaf).
            let encoded = leaf_ctx.encode_leaf(&state, root_fold, root_turn, None, None)?;
            let batch = ObsBatch::from_encoded(spec, std::slice::from_ref(&encoded))?;
            let output = evaluator.eval_batch(&batch, None)?;
            model_evals += 1;
            let map = leaf_ctx.self_action_map(&state, &root_options, None, None)?;
            match gather_self_priors(&output.priors[..output.action_count], &map) {
                Some(priors)
                    if apply_self_priors(&mut tree.decisions[0], self_side_one, &priors) =>
                {
                    root_priors = Some(priors);
                }
                _ => prior_fallbacks += 1,
            }
        }
    }
    let start = Instant::now();
    while completed < iterations {
        let traversal_budget = batch_size.min(iterations - completed);
        let mut traversals: Vec<Traversal> = Vec::with_capacity(traversal_budget);
        let mut pending: Vec<crate::encoder::EncodedArrays> = Vec::new();
        // Per-round prior work: (branch key, batch row, option→action map)
        // for every priced branch whose child decision node can exist.
        let mut pending_maps: Vec<((usize, usize), usize, Vec<Option<usize>>)> = Vec::new();
        let mut leaf_error: Option<PyErr> = None;
        while traversals.len() < traversal_budget && pending.len() < batch_size {
            let traversal = traverse(
                &mut tree,
                &mut state,
                &mut rng,
                &cfg,
                &mut counters,
                &mut |leaf: &State, seam: &BranchSeam| {
                    if leaf_error.is_some() {
                        return LeafPrice::Ready(0.5);
                    }
                    // Parent fold prefix: the root fold for root edges, the
                    // ancestor branch's advanced fold otherwise.
                    let (mut fold, mut turn, parent_order, parent_meta) = match seam.parent {
                        None => (
                            root_fold.clone(),
                            root_turn,
                            leaf_ctx.root_self_order().to_vec(),
                            leaf_ctx.root_meta().clone(),
                        ),
                        Some(key) => match fold_by_branch.get(&key) {
                            Some(rec) => (
                                rec.fold.clone(),
                                rec.turn,
                                rec.self_order.clone(),
                                rec.meta.clone(),
                            ),
                            None => {
                                leaf_error = Some(PyValueError::new_err(
                                    "parent branch fold missing (traversal order violated)",
                                ));
                                return LeafPrice::Ready(0.5);
                            }
                        },
                    };
                    // The mapper wants the PRE-branch state: rewind a clone.
                    let mut pre = leaf.clone();
                    let instructions: Vec<Instruction> = seam.instructions.to_vec();
                    pre.reverse_instructions(&instructions);
                    let mut ctx = event_ctx.clone();
                    ctx.turn = turn;
                    let rendered = crate::events::render_branch_events(
                        &mut pre,
                        seam.s1,
                        seam.s2,
                        seam.instructions,
                        seam.branch_on_damage,
                        &ctx,
                    );
                    if !rendered.lossy.is_empty() {
                        lossy_renders += 1;
                    }
                    if let Err(error) = fold.advance_in_place(&rendered.lines) {
                        leaf_error = Some(error);
                        return LeafPrice::Ready(0.5);
                    }
                    if rendered.turn_completed {
                        turn += 1;
                    }
                    let self_order = crate::leaf::evolve_self_order(
                        &parent_order,
                        &rendered.lines,
                        leaf_ctx.self_prefix(),
                    );
                    let meta = crate::leaf::evolve_leaf_meta(&parent_meta, &rendered.lines);
                    let encoded = match leaf_ctx.encode_leaf(
                        leaf,
                        &fold,
                        turn,
                        Some(&self_order),
                        Some(&meta),
                    ) {
                        Ok(encoded) => encoded,
                        Err(error) => {
                            leaf_error = Some(error);
                            return LeafPrice::Ready(0.5);
                        }
                    };
                    let row = pending.len();
                    // Prior map for this branch's future child decision node
                    // — only where a child can exist (depth cap) and the
                    // acting seat has a real choice.
                    if model_priors && seam.depth + 1 < cfg.max_depth {
                        let options = self_options_at(leaf, self_side_one);
                        if !is_single_none(&options) {
                            match leaf_ctx.self_action_map(
                                leaf,
                                &options,
                                Some(&self_order),
                                Some(&meta),
                            ) {
                                Ok(map) => pending_maps.push((
                                    (seam.chance, seam.branch_index),
                                    row,
                                    map,
                                )),
                                Err(error) => {
                                    leaf_error = Some(error);
                                    return LeafPrice::Ready(0.5);
                                }
                            }
                        }
                    }
                    fold_by_branch.insert(
                        (seam.chance, seam.branch_index),
                        BranchFold {
                            fold,
                            turn,
                            self_order,
                            meta,
                        },
                    );
                    pending.push(encoded);
                    LeafPrice::Deferred(row)
                },
            );
            traversals.push(traversal);
        }
        if let Some(error) = leaf_error {
            return Err(error);
        }
        let row_values = if pending.is_empty() {
            Vec::new()
        } else {
            let batch = ObsBatch::from_encoded(spec, &pending)?;
            let output = evaluator.eval_batch(&batch, None)?;
            model_evals += pending.len();
            // Resolve this round's prior maps against the batch's policy
            // rows: store on the branch (consumed at child creation) and
            // re-apply onto a child that was already created THIS round
            // (batch > 1 can descend a sibling before its priors resolved).
            for (key, row, map) in &pending_maps {
                let base = row * output.action_count;
                let priors =
                    gather_self_priors(&output.priors[base..base + output.action_count], map);
                match priors {
                    Some(priors) => {
                        let child = tree.chances[key.0].branches[key.1].child;
                        let applied = match child {
                            Some(child) => apply_self_priors(
                                &mut tree.decisions[child],
                                self_side_one,
                                &priors,
                            ),
                            None => true,
                        };
                        tree.chances[key.0].branches[key.1].child_self_priors =
                            Some((self_side_one, priors));
                        if applied {
                            prior_branches += 1;
                        } else {
                            prior_fallbacks += 1;
                        }
                    }
                    None => prior_fallbacks += 1,
                }
            }
            output.values01
        };
        for traversal in &traversals {
            finalize(&mut tree, traversal, &row_values);
        }
        completed += traversals.len();
        rounds += 1;
    }
    let elapsed_s = start.elapsed().as_secs_f64();
    let outcome = MultiPlyOutcome {
        tree,
        counters,
        elapsed_s,
    };
    let root_priors_json = match &root_priors {
        None => "null".to_string(),
        Some(priors) => format!(
            "[{}]",
            priors
                .iter()
                .map(|p| format!("{p:.6}"))
                .collect::<Vec<_>>()
                .join(",")
        ),
    };
    let extra = format!(
        "\"batch_size\":{},\"rounds\":{},\"model_evals\":{},\"encoder\":\"native_leaf\",\
         \"lossy_renders\":{},\"branch_folds\":{},\"model_priors\":{},\"prior_branches\":{},\
         \"prior_fallbacks\":{},\"root_priors\":{}",
        batch_size,
        rounds,
        model_evals,
        lossy_renders,
        fold_by_branch.len(),
        model_priors,
        prior_branches,
        prior_fallbacks,
        root_priors_json
    );
    Ok(multiply_report_json(
        &outcome,
        iterations,
        &cfg,
        seed,
        "torchscript",
        &extra,
    ))
}

// ---------------------------------------------------------------------------
// Python surface
// ---------------------------------------------------------------------------

/// `eval_obs_flat` result: (values_tanh, policy_logits_flat, priors_flat, action_count).
type EvalObsFlatResult = (Vec<f32>, Vec<f32>, Vec<f32>, usize);

/// TorchScript leaf model handle for Python: loads the artifact once, then
/// serves parity probes, forward benches, and full model-in-the-loop batched
/// searches. All heavy entrypoints release the GIL around search + model.
#[pyclass]
pub struct NativeLeafModel {
    evaluator: TorchScriptLeafEval,
}

#[pymethods]
impl NativeLeafModel {
    #[new]
    #[pyo3(signature = (
        model_path,
        device = "cpu",
        window = 1,
        tokens = 151,
        categorical_features = 51,
        numeric_features = 155,
    ))]
    fn new(
        model_path: &str,
        device: &str,
        window: i64,
        tokens: i64,
        categorical_features: i64,
        numeric_features: i64,
    ) -> PyResult<Self> {
        let spec = ObsSpec {
            window,
            tokens,
            categorical_features,
            numeric_features,
        };
        Ok(Self {
            evaluator: TorchScriptLeafEval::load(model_path, parse_device(device)?, spec)?,
        })
    }

    #[getter]
    fn device(&self) -> String {
        format!("{:?}", self.evaluator.device)
    }

    /// Debug/parity entrypoint: evaluate `batch` pre-encoded observations
    /// supplied as flat row-major buffers. Returns
    /// `(values_tanh, policy_logits_flat, priors_flat, action_count)` — the
    /// exact tensors the Python-side TorchScript run must reproduce.
    #[pyo3(signature = (
        batch,
        categorical_ids,
        numeric_features,
        token_type_ids,
        attention_mask,
        history_mask,
        legal_mask = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn eval_obs_flat(
        &self,
        py: Python<'_>,
        batch: usize,
        categorical_ids: Vec<i64>,
        numeric_features: Vec<f32>,
        token_type_ids: Vec<i64>,
        attention_mask: Vec<bool>,
        history_mask: Vec<bool>,
        legal_mask: Option<Vec<bool>>,
    ) -> PyResult<EvalObsFlatResult> {
        if batch == 0 {
            return Err(PyValueError::new_err("batch must be > 0"));
        }
        py.detach(|| {
            let spec = self.evaluator.spec();
            let obs = ObsBatch::from_flat(
                &spec,
                batch as i64,
                &categorical_ids,
                &numeric_features,
                &token_type_ids,
                &attention_mask,
                &history_mask,
            )?;
            let mask_tensor = match &legal_mask {
                Some(flat) => {
                    if flat.len() % batch != 0 {
                        return Err(PyValueError::new_err(format!(
                            "legal_mask length {} is not a multiple of batch {batch}",
                            flat.len()
                        )));
                    }
                    let width = (flat.len() / batch) as i64;
                    Some(Tensor::from_slice(flat).reshape([batch as i64, width]))
                }
                None => None,
            };
            let output = self.evaluator.eval_batch(&obs, mask_tensor.as_ref())?;
            Ok((
                output.values_tanh,
                output.policy_logits,
                output.priors,
                output.action_count,
            ))
        })
    }

    /// Forward-only throughput probe: evals/s at `batch_size` (deterministic
    /// synthetic inputs built once — timing is encoding-independent, same
    /// methodology as scripts/bench_inference.py).
    #[pyo3(signature = (batch_size, iters = 20, warmup = 3, seed = 0))]
    fn bench_forward(
        &self,
        py: Python<'_>,
        batch_size: i64,
        iters: usize,
        warmup: usize,
        seed: i64,
    ) -> PyResult<f64> {
        if batch_size <= 0 || iters == 0 {
            return Err(PyValueError::new_err("batch_size and iters must be > 0"));
        }
        py.detach(|| {
            let spec = self.evaluator.spec();
            tch::manual_seed(seed);
            let (w, t) = (spec.window, spec.tokens);
            let obs = ObsBatch {
                categorical_ids: Tensor::randint(
                    2,
                    [batch_size, w, t, spec.categorical_features],
                    (Kind::Int64, Device::Cpu),
                ),
                numeric_features: Tensor::randn(
                    [batch_size, w, t, spec.numeric_features],
                    (Kind::Float, Device::Cpu),
                ),
                token_type_ids: Tensor::zeros([batch_size, w, t], (Kind::Int64, Device::Cpu)),
                attention_mask: {
                    let mask = Tensor::rand([batch_size, w, t], (Kind::Float, Device::Cpu)).gt(0.2);
                    let _ = mask.narrow(2, 0, 1).fill_(1);
                    mask
                },
                history_mask: Tensor::ones([batch_size, w], (Kind::Bool, Device::Cpu)),
            };
            for _ in 0..warmup {
                self.evaluator.eval_batch(&obs, None)?;
            }
            let start = Instant::now();
            for _ in 0..iters {
                self.evaluator.eval_batch(&obs, None)?;
            }
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed <= 0.0 {
                return Err(PyValueError::new_err("elapsed time was zero; raise iters"));
            }
            Ok((batch_size as usize * iters) as f64 / elapsed)
        })
    }

    /// Model-in-the-loop batched one-ply PUCT (virtual-loss batching — see
    /// `batched_search_core` for the fidelity tradeoff). The template
    /// observation (flat buffers, batch 1) stands in for per-leaf encoding
    /// until the Rust encoder (track B) plugs in. Returns the search report
    /// as JSON. Runs with the GIL released.
    #[pyo3(signature = (
        state_str,
        iterations,
        batch_size,
        categorical_ids,
        numeric_features,
        token_type_ids,
        attention_mask,
        history_mask,
        c_puct = 1.4,
        seed = 0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn search_batched(
        &self,
        py: Python<'_>,
        state_str: &str,
        iterations: usize,
        batch_size: usize,
        categorical_ids: Vec<i64>,
        numeric_features: Vec<f32>,
        token_type_ids: Vec<i64>,
        attention_mask: Vec<bool>,
        history_mask: Vec<bool>,
        c_puct: f32,
        seed: u64,
    ) -> PyResult<String> {
        if iterations == 0 || batch_size == 0 {
            return Err(PyValueError::new_err(
                "iterations and batch_size must be > 0",
            ));
        }
        py.detach(|| {
            let spec = self.evaluator.spec();
            let template = ObsBatch::from_flat(
                &spec,
                1,
                &categorical_ids,
                &numeric_features,
                &token_type_ids,
                &attention_mask,
                &history_mask,
            )?;
            batched_search_core(
                state_str,
                iterations,
                batch_size,
                &template,
                &self.evaluator,
                &spec,
                c_puct,
                seed,
            )
        })
    }

    /// Model-in-the-loop MULTI-PLY batched search (decision/chance tree with
    /// exact-expectation backup; see `multiply_batched_core` for the
    /// batching-through-chance design). The template observation stands in
    /// for per-leaf encoding until the Rust encoder (track B) plugs in.
    /// Returns the search report as JSON. Runs with the GIL released.
    #[pyo3(signature = (
        state_str,
        iterations,
        batch_size,
        categorical_ids,
        numeric_features,
        token_type_ids,
        attention_mask,
        history_mask,
        max_depth = 2,
        c_puct = 1.4,
        seed = 0,
        deep_ko_split = true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn search_batched_multi(
        &self,
        py: Python<'_>,
        state_str: &str,
        iterations: usize,
        batch_size: usize,
        categorical_ids: Vec<i64>,
        numeric_features: Vec<f32>,
        token_type_ids: Vec<i64>,
        attention_mask: Vec<bool>,
        history_mask: Vec<bool>,
        max_depth: u8,
        c_puct: f32,
        seed: u64,
        deep_ko_split: bool,
    ) -> PyResult<String> {
        if iterations == 0 || batch_size == 0 {
            return Err(PyValueError::new_err(
                "iterations and batch_size must be > 0",
            ));
        }
        if max_depth == 0 || max_depth > 32 {
            return Err(PyValueError::new_err("max_depth must be in 1..=32"));
        }
        py.detach(|| {
            let spec = self.evaluator.spec();
            let template = ObsBatch::from_flat(
                &spec,
                1,
                &categorical_ids,
                &numeric_features,
                &token_type_ids,
                &attention_mask,
                &history_mask,
            )?;
            multiply_batched_core(
                state_str,
                iterations,
                batch_size,
                max_depth,
                deep_ko_split,
                &template,
                &self.evaluator,
                &spec,
                c_puct,
                seed,
            )
        })
    }

    /// Model-in-the-loop MULTI-PLY batched search with REAL leaf
    /// observations (the engine-swap capstone loop): every model row is the
    /// branch's own evolved observation — synthesized events, per-outcome
    /// fold advance, native encode — instead of the template stub.
    ///
    /// `root_inputs_json` is the root decision's sanctioned input surface,
    /// `ctx_json` the event context (`{"p1": [...], "p2": [...], "turn": N,
    /// "hp_percent": [...]}` — species in engine party order), `root_fold`
    /// the root fold state. With `model_priors` (default) the ACTING seat's
    /// decision arms use the model's masked policy priors (root: one extra
    /// forward on the root observation; interior nodes: the priors of the
    /// branch observation they grow under); the opponent side stays uniform.
    /// Returns the search report as JSON (with `encoder: "native_leaf"`,
    /// lossy-render accounting, and `root_priors`/`prior_branches`/
    /// `prior_fallbacks` telemetry). Runs with the GIL released.
    #[pyo3(signature = (
        state_str,
        iterations,
        batch_size,
        tables_json,
        root_inputs_json,
        ctx_json,
        root_fold,
        max_depth = 2,
        c_puct = 1.4,
        seed = 0,
        deep_ko_split = true,
        model_priors = true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn search_batched_multi_encoded(
        &self,
        py: Python<'_>,
        state_str: &str,
        iterations: usize,
        batch_size: usize,
        tables_json: &str,
        root_inputs_json: &str,
        ctx_json: &str,
        root_fold: PyRef<'_, crate::fold::PyFoldState>,
        max_depth: u8,
        c_puct: f32,
        seed: u64,
        deep_ko_split: bool,
        model_priors: bool,
    ) -> PyResult<String> {
        if iterations == 0 || batch_size == 0 {
            return Err(PyValueError::new_err(
                "iterations and batch_size must be > 0",
            ));
        }
        if max_depth == 0 || max_depth > 32 {
            return Err(PyValueError::new_err("max_depth must be in 1..=32"));
        }
        let root_state = parse_state(state_str)?;
        let leaf_ctx = crate::leaf::LeafContext::new(
            tables_json,
            root_inputs_json,
            ctx_json,
            &root_state,
        )?;
        let event_ctx = crate::events::EventContext::from_json(ctx_json)
            .map_err(PyValueError::new_err)?;
        let fold = root_fold.inner().clone();
        drop(root_fold);
        py.detach(|| {
            let spec = self.evaluator.spec();
            multiply_batched_encoded_core(
                state_str,
                iterations,
                batch_size,
                max_depth,
                deep_ko_split,
                model_priors,
                &leaf_ctx,
                &event_ctx,
                &fold,
                &self.evaluator,
                &spec,
                c_puct,
                seed,
            )
        })
    }
}
