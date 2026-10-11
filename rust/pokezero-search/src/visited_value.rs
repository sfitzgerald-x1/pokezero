//! Actual encoded-tree evaluation-event capture, with owned native frontiers.
//! This is an opt-in ENGINEERING diagnostic, not a Showdown calibration label.
//! Searched-line fidelity and producer integration remain separate gates.
//! Private states reach only an explicitly opted-in, post-selection fidelity
//! controller; never a policy callback or a published receipt. No diagnostic
//! label or fidelity result feeds backups.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};
use poke_engine::state::State;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use crate::events::EventContext;
use crate::leaf::{LeafMeta, LeafMetaCtx};
use crate::raw_policy_leaf::{label_price, RawPolicyLeaf, RawPolicyLeafStats, TerminalLabel};
use crate::tree::BranchSeam;

fn refuse(message: &str) -> PyErr { PyValueError::new_err(message.to_owned()) }
fn hash(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }

/// Opt-in ancestry of the ACTUAL priced branches, not regenerated support.
/// All retained serialized ancestry shares the existing private payload cap.
#[derive(Default)]
struct SearchedLines {
    root: String,
    steps: BTreeMap<(usize, usize), Value>,
    bytes: usize,
}

impl SearchedLines {
    fn path(&self, key: (usize, usize)) -> PyResult<Vec<Value>> {
        let mut cursor = Some(key);
        let mut reverse = Vec::new();
        while let Some(key) = cursor {
            if reverse.len() >= 64 { return Err(refuse("searched line: ancestry cap/cycle")); }
            let step = self.steps.get(&key).ok_or_else(|| refuse("searched line: missing actual ancestor"))?;
            cursor = serde_json::from_value(step["parent"].clone())
                .map_err(|_| refuse("searched line: invalid parent identity"))?;
            reverse.push(step.clone());
        }
        reverse.reverse();
        Ok(reverse)
    }
}
fn seeded(namespace: &str, seed: u64, worker: usize, root: &str, ordinal: usize,
    replicate: Option<usize>) -> u64 {
    let mut parts = vec![json!(namespace), json!(seed), json!(worker), json!(root), json!(ordinal)];
    if let Some(replicate) = replicate { parts.push(json!(replicate)); }
    let bytes = Sha256::digest(serde_json::to_vec(&parts).expect("finite identity"));
    u64::from_be_bytes(bytes[..8].try_into().unwrap())
}

struct Frontier {
    state: State,
    key: (usize, usize),
    ctx: EventContext,
    meta: LeafMeta,
    meta_ctx: LeafMetaCtx,
    raw: RawPolicyLeaf,
    snapshot_sha256: String,
    state_sha256: String,
    payload_bytes: usize,
    prediction: Option<f32>,
}

impl Frontier {
    fn payload(&self) -> PyResult<Vec<u8>> {
        Ok(serde_json::to_vec(&json!({
            "native_state": self.state.serialize(), "key": self.key,
            "ctx": format!("{:?}", self.ctx), "meta": format!("{:?}", self.meta),
            "meta_ctx": format!("{:?}", self.meta_ctx),
            "p1_private_bridge": self.raw.bridges[0].snapshot_payload(self.key)?,
            "p2_private_bridge": self.raw.bridges[1].snapshot_payload(self.key)?,
            "branch_on_damage": self.raw.branch_on_damage,
        })).expect("private finite strings"))
    }
}

pub(crate) struct VisitedValueBank {
    seed: u64,
    worker: usize,
    root: String,
    keep: usize,
    maximum_payload_bytes: usize,
    deadline_at: f64,
    deadline: Instant,
    self_side_one: Option<bool>,
    root_native_state_sha256: Option<String>,
    selection_report_sha256: Option<String>,
    seen: usize,
    bytes: usize,
    rows: BTreeMap<(u64, usize), Frontier>,
    searched_lines: Option<SearchedLines>,
}

impl VisitedValueBank {
    pub(crate) fn expired(&self) -> bool { Instant::now() >= self.deadline }
    pub(crate) fn deadline(&self) -> Instant { self.deadline }
    pub(crate) fn bind_root(&mut self, state_str: &str) -> PyResult<()> {
        self.root_native_state_sha256 = Some(hash(state_str.as_bytes()));
        if let Some(lines) = &mut self.searched_lines {
            if state_str.len() > self.maximum_payload_bytes { return Err(refuse("searched line: private payload cap reached")); }
            lines.root = state_str.to_owned();
            lines.bytes = state_str.len();
        }
        Ok(())
    }
    fn new(seed: u64, worker: usize, root: String, keep: usize,
        maximum_payload_bytes: usize, deadline_at: f64, deadline: Instant) -> PyResult<Self> {
        if root.is_empty() || root.len() % 2 != 0 || !root.bytes().all(|b| b.is_ascii_hexdigit())
            || root != root.to_ascii_lowercase() || !(1..=8).contains(&keep)
            || maximum_payload_bytes == 0 || maximum_payload_bytes > 8 * 1024 * 1024
            || !deadline_at.is_finite() {
            return Err(refuse("visited value: invalid root/sampling/deadline/payload contract"));
        }
        Ok(Self { seed, worker, root, keep, maximum_payload_bytes, deadline_at, deadline,
            self_side_one: None, root_native_state_sha256: None, selection_report_sha256: None,
            seen: 0, bytes: 0, rows: BTreeMap::new(), searched_lines: None })
    }

    pub(crate) fn record_searched_branch(&mut self, state: &State, seam: &BranchSeam) -> PyResult<()> {
        let Some(lines) = &mut self.searched_lines else { return Ok(()) };
        let mut pre = state.clone();
        pre.reverse_instructions(&seam.instructions.to_vec());
        let before = pre.serialize();
        let expected = match seam.parent {
            None => lines.root.as_str(),
            Some(parent) => lines.steps.get(&parent)
                .and_then(|v| v["after"].as_str())
                .ok_or_else(|| refuse("searched line: missing priced parent"))?,
        };
        if before != expected { return Err(refuse("searched line: actual prestate does not equal parent frontier")); }
        let choice = |action: &poke_engine::engine::state::MoveChoice, side: &poke_engine::state::Side| {
            use poke_engine::engine::state::MoveChoice;
            let kind = match action { MoveChoice::Move(_) => "move", MoveChoice::Switch(_) => "switch",
                MoveChoice::Struggle => "struggle", MoveChoice::None => "none" };
            json!({ "kind": kind, "engine_id": action.to_string(side) })
        };
        let key = (seam.chance, seam.branch_index);
        let step = json!({ "key": key, "parent": seam.parent, "depth": seam.depth,
            "before": before, "after": state.serialize(), "branch_on_damage": seam.branch_on_damage,
            "side_one_choice": choice(seam.s1, &pre.side_one),
            "side_two_choice": choice(seam.s2, &pre.side_two) });
        if lines.steps.contains_key(&key) { return Err(refuse("searched line: duplicate branch identity")); }
        let size = serde_json::to_vec(&step).expect("finite private strings").len();
        let total = lines.bytes.checked_add(size).and_then(|v| v.checked_add(self.bytes))
            .ok_or_else(|| refuse("searched line: payload accounting overflow"))?;
        if total > self.maximum_payload_bytes { return Err(refuse("searched line: private payload cap reached")); }
        lines.bytes += size;
        lines.steps.insert(key, step);
        Ok(())
    }

    /// Stage only priority-selected candidates before the actual forward, while
    /// the traversal still owns its precise frontier. A failed forward poisons
    /// the entire handle; predictions are attached only after successful eval.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn stage(&mut self, state: &State, key: (usize, usize), row: usize,
        ctx: &EventContext, meta: &LeafMeta, meta_ctx: &LeafMetaCtx,
        raw: &RawPolicyLeaf, self_side_one: bool) -> PyResult<()> {
        if self.self_side_one.is_some_and(|seat| seat != self_side_one) {
            return Err(refuse("visited value: searching seat changed within one invocation"));
        }
        self.self_side_one = Some(self_side_one);
        let ordinal = self.seen.checked_add(row).ok_or_else(|| refuse("visited ordinal overflow"))?;
        let priority = seeded("visited-model-sample.v1", self.seed, self.worker, &self.root, ordinal, None);
        let identity = (priority, ordinal);
        if self.rows.len() == self.keep && identity >= *self.rows.last_key_value().unwrap().0 {
            return Ok(());
        }
        let mut item = Frontier {
            state: state.clone(), key, ctx: ctx.clone(), meta: meta.clone(), meta_ctx: meta_ctx.clone(),
            raw: RawPolicyLeaf { bridges: [raw.bridges[0].fork_at(key)?, raw.bridges[1].fork_at(key)?],
                max_plies: 250, seed: self.seed, branch_on_damage: raw.branch_on_damage },
            snapshot_sha256: String::new(), state_sha256: hash(state.serialize().as_bytes()),
            payload_bytes: 0, prediction: None,
        };
        let payload = item.payload()?;
        item.snapshot_sha256 = hash(&payload);
        item.payload_bytes = payload.len();
        let victim_bytes = if self.rows.len() == self.keep {
            self.rows.last_key_value().unwrap().1.payload_bytes
        } else { 0 };
        let bytes = self.bytes.checked_add(item.payload_bytes).and_then(|n| n.checked_sub(victim_bytes))
            .ok_or_else(|| refuse("visited payload accounting overflow"))?;
        if bytes + self.searched_lines.as_ref().map_or(0, |lines| lines.bytes) > self.maximum_payload_bytes {
            return Err(refuse("visited value: retained private payload cap reached"));
        }
        if self.rows.len() == self.keep { self.rows.pop_last(); }
        self.rows.insert(identity, item);
        self.bytes = bytes;
        Ok(())
    }

    pub(crate) fn commit(&mut self, self_values01: &[f32]) -> PyResult<()> {
        if self_values01.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v)) {
            return Err(refuse("visited value: invalid actual model prediction"));
        }
        for ((_, ordinal), item) in &mut self.rows {
            if *ordinal >= self.seen {
                let value = self_values01.get(*ordinal - self.seen)
                    .ok_or_else(|| refuse("visited value: prediction/frontier row mismatch"))?;
                item.prediction = Some(2.0 * value - 1.0); // SELF frame, before tree reflection.
            }
        }
        self.seen += self_values01.len();
        if self.rows.len() != self.keep.min(self.seen) || self.rows.values().any(|item| item.prediction.is_none()) {
            return Err(refuse("visited value: incomplete successful evaluation reservoir"));
        }
        Ok(())
    }

    fn label(self, lossy: &mut crate::abort_telemetry::LossySubcaseLedger) -> PyResult<Value> {
        if self.searched_lines.is_some() { return Err(refuse("searched line: use fidelity controller, not native labels")); }
        if self.selection_report_sha256.is_none() || self.root_native_state_sha256.is_none() {
            return Err(refuse("visited value: actual search root and successful selection receipt required"));
        }
        let mut leaves = Vec::new();
        for ((priority, ordinal), item) in &self.rows {
            if hash(&item.payload()?) != item.snapshot_sha256 {
                return Err(refuse("visited value: captured private frontier mutated"));
            }
            let mut labels = Vec::new();
            for replicate in 0..8 {
                let seed = seeded("native-visited-label.v1", self.seed, self.worker, &self.root,
                    *ordinal, Some(replicate));
                let cfg = RawPolicyLeaf { bridges: [item.raw.bridges[0].fork_at(item.key)?,
                    item.raw.bridges[1].fork_at(item.key)?], max_plies: 250, seed,
                    branch_on_damage: item.raw.branch_on_damage };
                let mut stats = RawPolicyLeafStats::default();
                let terminal = label_price(&item.state, item.key, 0, &item.ctx, &item.meta,
                    &item.meta_ctx, &cfg, Some(self.deadline), &mut stats, lossy)?;
                let (status, outcome) = match terminal {
                    TerminalLabel::Terminal(value) => ("COMPLETE", Some(if self.self_side_one.unwrap() {
                        2.0 * value - 1.0
                    } else { 1.0 - 2.0 * value })),
                    TerminalLabel::Capped => ("CAPPED_UNCERTAIN", None),
                    TerminalLabel::Deadline => ("DEADLINE_UNCERTAIN", None),
                };
                labels.push(json!({ "replicate": replicate, "status": status,
                    "signed_outcome": outcome, "boundaries": stats.plies,
                    "continuation_policy": "raw_argmax_both_seats", "seed": seed }));
            }
            leaves.push(json!({ "evaluation_ordinal": ordinal, "priority": priority,
                "native_state_sha256": item.state_sha256, "snapshot_sha256": item.snapshot_sha256,
                "model_signed_value": item.prediction.unwrap(), "labels": labels }));
        }
        Ok(json!({ "schema": "pokezero.native-visited-value.engineering.v1",
            "worker": self.worker, "root_information_key": self.root,
            "sample_seed": self.seed, "original_deadline_at": self.deadline_at,
            "root_native_state_sha256": self.root_native_state_sha256,
            "selection_report_sha256": self.selection_report_sha256,
            "leaves_per_native_invocation": self.keep, "replicates": 8, "maximum_boundaries": 250,
            "model_evaluations_seen": self.seen, "sampling_unit": "successful_model_evaluation_event",
            "sampling": "smallest_seeded_priorities_per_native_invocation", "value_frame": "self_relative",
            "retained_payload_bytes": self.bytes, "maximum_payload_bytes": self.maximum_payload_bytes,
            "payload_cap_is_not_rss_qualification": true, "leaves": leaves,
            "label_lossy_subcases": serde_json::from_str::<Value>(&lossy.json_object()).expect("bounded subcase vocabulary"),
            "label_engine": "poke_engine_unqualified_against_showdown",
            "searched_line_showdown_fidelity": false, "scientific_calibration_evidence": false,
            "scientific_strength_evidence": false, "labels_used_for_backup": false,
            "private_frontiers_exported": false, "instrumentation_can_change_selection": true }))
    }
}

/// Exactly one native invocation, then exactly one post-selection label call.
/// Taking the bank consumes it before callbacks/FFI; errors cannot be retried.
#[pyclass(unsendable)]
pub(crate) struct NativeVisitedValueBank {
    bank: Option<VisitedValueBank>,
    ready: bool,
    taken: bool,
    closed: bool,
}

impl NativeVisitedValueBank {
    pub(crate) fn begin(&mut self) -> PyResult<VisitedValueBank> {
        if self.taken || self.closed { return Err(refuse("visited value: capture invocation already consumed")); }
        self.taken = true;
        self.bank.take().ok_or_else(|| refuse("visited value: missing private bank"))
    }
    pub(crate) fn finish(&mut self, mut bank: VisitedValueBank, result: &PyResult<String>) {
        self.ready = result.is_ok() && !self.closed;
        if let Ok(report) = result { bank.selection_report_sha256 = Some(hash(report.as_bytes())); }
        self.bank = if self.ready { Some(bank) } else { None };
    }
}

#[pymethods]
impl NativeVisitedValueBank {
    #[new]
    #[pyo3(signature = (sample_seed, worker, root_information_key, original_deadline_at,
        leaves_per_native_invocation=2, maximum_payload_bytes=8388608, capture_searched_lines=false))]
    fn py_new(py: Python<'_>, sample_seed: u64, worker: usize, root_information_key: String,
        original_deadline_at: f64, leaves_per_native_invocation: usize,
        maximum_payload_bytes: usize, capture_searched_lines: bool) -> PyResult<Self> {
        let now: f64 = py.import("time")?.getattr("perf_counter")?.call0()?.extract()?;
        let remaining = original_deadline_at - now;
        if !original_deadline_at.is_finite() || !remaining.is_finite() || remaining > 14400.0 {
            return Err(refuse("visited value: finite original producer deadline within four hours required"));
        }
        let deadline = Instant::now().checked_add(Duration::from_secs_f64(remaining.max(0.0)))
            .ok_or_else(|| refuse("visited value: deadline overflow"))?;
        let mut bank = VisitedValueBank::new(sample_seed, worker, root_information_key,
            leaves_per_native_invocation, maximum_payload_bytes, original_deadline_at, deadline)?;
        if capture_searched_lines { bank.searched_lines = Some(SearchedLines::default()); }
        Ok(Self { bank: Some(bank),
            ready: false, taken: false, closed: false })
    }

    fn label(&mut self, py: Python<'_>) -> PyResult<String> {
        if !self.ready { return Err(refuse("visited value: successful selection required before labels")); }
        self.ready = false;
        let bank = self.bank.take().ok_or_else(|| refuse("visited value: labels already consumed"))?;
        py.detach(move || crate::abort_telemetry::guarded_search_with_ledger(|lossy| bank.label(lossy).map(|v| v.to_string())))
    }

    /// Trusted PRIVATE diagnostic controller, not a raw-policy provider. Takes
    /// the bank before invoking Python, including on exceptions or reentry.
    /// The only returned evidence is a bounded, whitelisted engineering result.
    fn check_searched_lines(&mut self, py: Python<'_>, controller: Py<PyAny>) -> PyResult<String> {
        if !self.ready { return Err(refuse("searched line: successful selection required")); }
        self.ready = false;
        let bank = self.bank.take().ok_or_else(|| refuse("searched line: already consumed"))?;
        let lines = bank.searched_lines.as_ref().ok_or_else(|| refuse("searched line: capture was not enabled"))?;
        let mut results = Vec::new();
        for ((priority, ordinal), item) in &bank.rows {
            if bank.expired() { return Err(refuse("searched line: original deadline expired")); }
            if hash(&item.payload()?) != item.snapshot_sha256 { return Err(refuse("searched line: frontier mutated")); }
            let path = lines.path(item.key)?;
            if path.last().and_then(|v| v["after"].as_str()) != Some(item.state.serialize().as_str()) {
                return Err(refuse("searched line: path endpoint does not equal predicted frontier"));
            }
            let private = json!({ "root_native_state": lines.root, "steps": path,
                "native_frontier": item.state.serialize(), "model_signed_value": item.prediction,
                "evaluation_ordinal": ordinal, "private_controller_only": true });
            let reply: String = controller.call1(py, (private.to_string(),))?.extract(py)?;
            if bank.expired() { return Err(refuse("searched line: controller exceeded original deadline")); }
            let result: Value = serde_json::from_str(&reply).map_err(|_| refuse("searched line: invalid controller reply"))?;
            let status = result["status"].as_str().unwrap_or("");
            let count = result["checked_transitions"].as_u64().unwrap_or(u64::MAX);
            let fields = result["mismatch_components"].as_array().ok_or_else(|| refuse("searched line: invalid mismatch fields"))?;
            const COMPONENTS: &[&str] = &["root", "prestate", "hp", "status", "weather", "boosts", "native_pp", "request", "engine_serialization", "unsupported_action", "chance_coupling"];
            if !matches!(status, "MATCHED_PROJECTION_NOT_FULL_STATE" | "MISMATCH" | "UNSUPPORTED")
                || count > path.len() as u64 || fields.len() > COMPONENTS.len()
                || fields.iter().any(|v| !v.as_str().is_some_and(|s| COMPONENTS.contains(&s)))
                || result.as_object().map_or(0, |v| v.len()) != 3
                || (status == "MATCHED_PROJECTION_NOT_FULL_STATE" && (count != path.len() as u64 || !fields.is_empty())) {
                return Err(refuse("searched line: non-sanitized or unsupported controller verdict"));
            }
            results.push(json!({ "evaluation_ordinal": ordinal, "priority": priority,
                "native_state_sha256": item.state_sha256, "model_signed_value": item.prediction,
                "path_sha256": hash(serde_json::to_string(&path).unwrap().as_bytes()),
                "path_boundaries": path.len(), "result": result }));
        }
        Ok(json!({ "schema": "pokezero.native-searched-line.excluded-engineering.v1",
            "root_native_state_sha256": bank.root_native_state_sha256,
            "selection_report_sha256": bank.selection_report_sha256, "model_evaluations_seen": bank.seen,
            "retained_private_payload_bytes": bank.bytes + lines.bytes,
            "maximum_payload_bytes": bank.maximum_payload_bytes, "leaves": results,
            "payload_cap_is_not_rss_qualification": true,
            "scope": "excluded_deterministic_fixture_only", "private_states_in_receipt": false,
            "full_state_correspondence": false, "searched_line_showdown_fidelity": false,
            "scientific_calibration_evidence": false, "scientific_strength_evidence": false,
            "labels_used_for_backup": false, "instrumentation_can_change_selection": true }).to_string())
    }

    fn close(&mut self) { self.bank = None; self.ready = false; self.taken = true; self.closed = true; }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw_policy_leaf::tests::fixture;

    fn bank(keep: usize) -> VisitedValueBank {
        let mut bank = VisitedValueBank::new(17, 0, "aa".into(), keep, 8*1024*1024,
            100.0, Instant::now() + Duration::from_secs(120)).unwrap();
        // These unit fixtures never claim to be a production/search receipt.
        bank.bind_root("excluded-engineering-unit-fixture").unwrap();
        bank.selection_report_sha256 = Some(hash(b"excluded-fixture-selection"));
        bank
    }

    #[test]
    fn priority_matches_the_python_law_and_not_predictions() {
        // Golden value computed with the reference's SHA256 JSON identity law.
        assert_eq!(seeded("visited-model-sample.v1", 17, 0, "aa", 3, None), 8939513739264610312);
        let (state, ctx, meta, raw) = fixture(true, 2);
        let mut first = bank(2);
        let mut second = bank(2);
        for offset in [0, 10] {
            for row in 0..10 {
                let mut leaf = state.clone();
                leaf.side_one.get_active().hp -= (offset + row) as i16;
                for bank in [&mut first, &mut second] {
                    bank.stage(&leaf, (0, 0), row, &ctx, &meta, &Default::default(), &raw, true).unwrap();
                }
            }
            let values: Vec<_> = (0..10).map(|row| (offset + row) as f32 / 20.0).collect();
            first.commit(&values).unwrap();
            second.commit(&values.iter().map(|v| 1.0-v).collect::<Vec<_>>()).unwrap();
        }
        let mut expected: Vec<_> = (0..20).map(|i|
            (seeded("visited-model-sample.v1", 17, 0, "aa", i, None), i)).collect();
        expected.sort();
        expected.truncate(2);
        assert_eq!(first.rows.keys().copied().collect::<Vec<_>>(), expected);
        assert_eq!(first.rows.keys().collect::<Vec<_>>(), second.rows.keys().collect::<Vec<_>>());
        assert_eq!(first.seen, 20);
        for ((_, ordinal), item) in &first.rows {
            assert_eq!(item.prediction, Some(2.0 * (*ordinal as f32 / 20.0) - 1.0));
        }
        assert_eq!(first.bytes, first.rows.values().map(|r| r.payload_bytes).sum::<usize>());
    }

    #[test]
    fn eight_owned_labels_are_self_relative_and_do_not_touch_original_state_or_bridges() {
        for searching_side_one in [true, false] {
            let (state, ctx, meta, raw) = fixture(true, 2);
            let before = state.serialize();
            let mut bank = bank(2);
            bank.stage(&state, (0, 0), 0, &ctx, &meta, &Default::default(), &raw, searching_side_one).unwrap();
            bank.commit(&[0.625]).unwrap();
            let evidence = bank.label(&mut Default::default()).unwrap();
            let row = &evidence["leaves"][0];
            assert_eq!(row["model_signed_value"], 0.25);
            assert_eq!(row["native_state_sha256"], hash(before.as_bytes()));
            let labels = row["labels"].as_array().unwrap();
            assert_eq!(labels.len(), 8);
            for label in labels {
                assert_eq!(label["status"], "COMPLETE");
                assert_eq!(label["signed_outcome"], if searching_side_one { 1.0 } else { -1.0 });
                assert_eq!(label["boundaries"], 2);
            }
            assert_eq!(state.serialize(), before);
            assert_eq!(raw.bridges[0].provider_calls.get(), 0);
            assert_eq!(raw.bridges[1].provider_calls.get(), 0);
            assert_eq!(evidence["scientific_calibration_evidence"], false);
            assert_eq!(evidence["labels_used_for_backup"], false);
            assert!(!evidence.to_string().contains(&before));
        }
    }

    #[test]
    fn original_deadline_keeps_all_eight_labels_uncertain() {
        let (state, ctx, meta, raw) = fixture(true, 2);
        let mut bank = bank(2);
        bank.deadline = Instant::now();
        bank.stage(&state, (0, 0), 0, &ctx, &meta, &Default::default(), &raw, true).unwrap();
        bank.commit(&[0.25]).unwrap();
        let evidence = bank.label(&mut Default::default()).unwrap();
        for label in evidence["leaves"][0]["labels"].as_array().unwrap() {
            assert_eq!(label["status"], "DEADLINE_UNCERTAIN");
            assert!(label["signed_outcome"].is_null());
            assert_eq!(label["boundaries"], 0);
        }
        assert_eq!(raw.bridges[0].provider_calls.get(), 0);
    }

    #[test]
    fn changed_frontier_is_refused_not_labeled_as_the_original_prediction() {
        let (state, ctx, meta, raw) = fixture(true, 2);
        let mut bank = bank(2);
        bank.stage(&state, (0, 0), 0, &ctx, &meta, &Default::default(), &raw, true).unwrap();
        bank.commit(&[0.25]).unwrap();
        bank.rows.first_entry().unwrap().get_mut().state.side_one.get_active().hp -= 1;
        assert!(bank.label(&mut Default::default()).unwrap_err().to_string().contains("mutated"));
        assert_eq!(raw.bridges[0].provider_calls.get(), 0);
    }

    #[test]
    fn payload_limit_refuses_instead_of_sampling_smaller_frontiers() {
        let (state, ctx, meta, raw) = fixture(true, 2);
        let mut bank = bank(2);
        bank.maximum_payload_bytes = 1;
        assert!(bank.stage(&state, (0, 0), 0, &ctx, &meta, &Default::default(), &raw, true)
            .unwrap_err().to_string().contains("payload cap"));
        assert_eq!(bank.seen, 0);
        assert!(bank.rows.is_empty());
    }

    #[test]
    fn invalid_predictions_and_missing_rows_do_not_admit_a_reservoir() {
        let (state, ctx, meta, raw) = fixture(true, 2);
        for prediction in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
            let mut bank = bank(2);
            bank.stage(&state, (0, 0), 0, &ctx, &meta, &Default::default(), &raw, true).unwrap();
            assert!(bank.commit(&[prediction]).is_err());
        }
        let mut bank = bank(2);
        bank.stage(&state, (0, 0), 1, &ctx, &meta, &Default::default(), &raw, true).unwrap();
        assert!(bank.commit(&[0.5]).is_err());
    }

    #[test]
    fn handle_is_consumed_on_failed_search_successful_label_and_close() {
        Python::initialize();
        Python::attach(|py| {
            let mut handle = NativeVisitedValueBank { bank: Some(bank(2)), ready: false, taken: false, closed: false };
            assert!(handle.label(py).is_err());
            let captured = handle.begin().unwrap();
            assert!(handle.begin().is_err());
            handle.finish(captured, &Err(refuse("excluded search failure")));
            assert!(handle.label(py).is_err());
            assert!(handle.bank.is_none());
            let mut handle = NativeVisitedValueBank { bank: Some(bank(2)), ready: false, taken: false, closed: false };
            let captured = handle.begin().unwrap();
            handle.finish(captured, &Ok("excluded successful selection".into()));
            assert!(handle.label(py).is_ok()); // zero actual forwards, explicitly zero leaves.
            assert!(handle.label(py).is_err());
            assert!(handle.bank.is_none());
            let mut handle = NativeVisitedValueBank { bank: Some(bank(2)), ready: false, taken: false, closed: false };
            handle.close();
            assert!(handle.begin().is_err());
            let mut handle = NativeVisitedValueBank { bank: Some(bank(2)), ready: false, taken: false, closed: false };
            let captured = handle.begin().unwrap();
            handle.close();
            handle.finish(captured, &Ok("excluded successful selection".into()));
            assert!(handle.bank.is_none());
            assert!(handle.label(py).is_err());
        });
    }
}
