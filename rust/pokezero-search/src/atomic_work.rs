//! Opt-in observation of actual encoded-core batch commits. Never search input.
//! Bit patterns preserve float32 backup order for a separate consistency reader.
//! These receipts do not independently validate evaluator or terminal labels.
use pyo3::prelude::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use crate::tree::{Tree, Traversal, TraversalEnd, finalize};

pub(crate) const SCHEMA: &str = "native-atomic-backups-v1";
pub(crate) const COMPACT_SCHEMA: &str = "native-atomic-backups-compact-v2";

pub(crate) fn snapshot(tree: &Tree) -> Value {
    json!({
        "decisions": tree.decisions.iter().map(|node| {
            let stats = |rows: &[crate::MoveStats]| rows.iter().map(|s|
                json!({"visits": s.visits, "total_bits": s.total_value.to_bits()}))
                .collect::<Vec<_>>();
            let mut children = node.children.iter().map(|(&(i,j), &chance)|
                (i,j,chance)).collect::<Vec<_>>();
            children.sort_unstable();
            json!({"visits": node.visits, "depth": node.depth,
                "side_one": stats(&node.s1_stats), "side_two": stats(&node.s2_stats),
                "children": children})
        }).collect::<Vec<_>>(),
        "chances": tree.chances.iter().map(|chance| chance.branches.iter().map(|b|
            json!({"probability_bits": b.probability.to_bits(),
                "value_sum_bits": b.value_sum.to_bits(), "visits": b.visits,
                "pending_row": b.pending_row, "child": b.child,
                "terminal_bits": b.terminal.map(f32::to_bits), "no_expand": b.no_expand}))
            .collect::<Vec<_>>()).collect::<Vec<_>>()
    })
}

fn path(traversal: &Traversal) -> Value {
    json!(traversal.path.iter().map(|s| json!({"decision": s.decision,
        "side_one": s.i, "side_two": s.j, "chance": s.chance, "branch": s.branch}))
        .collect::<Vec<_>>())
}

/// Append only after every actual finalize has returned. A panic or enclosing
/// search error invalidates the whole call; no accepted prefix is returned.
pub(crate) fn commit_round(tree: &mut Tree, traversals: &[Traversal], values: &[f32],
    before_collection: Value, rounds: &mut Vec<Value>) {
    let after_collection = snapshot(tree);
    let mut commits = Vec::with_capacity(traversals.len());
    for traversal in traversals {
        let before = snapshot(tree);
        let end = match traversal.end {
            TraversalEnd::Ready(v) => json!({"kind": "ready", "value_bits": v.to_bits()}),
            TraversalEnd::Row(row) => json!({"kind": "row", "row": row}),
            TraversalEnd::Expanded => json!({"kind": "expanded"}),
        };
        let value = finalize(tree, traversal, values);
        commits.push(json!({"path": path(traversal), "end": end, "before": before,
            "after": snapshot(tree), "root_value_bits": value.to_bits()}));
    }
    rounds.push(json!({"ordinal": rounds.len(), "status": "COMMITTED",
        "before_collection": before_collection, "after_collection": after_collection,
        "row_value_bits": values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "commits": commits, "after": snapshot(tree), "accepted": traversals.len()}));
}

pub(crate) fn cancel_round(tree: &Tree, traversals: &[Traversal], before: Value, collected: Value,
    rounds: &mut Vec<Value>) {
    rounds.push(json!({"ordinal": rounds.len(), "status": "CANCELLED",
        "before_collection": before, "after_collection": collected, "after": snapshot(tree), "accepted": 0,
        "attempted_paths": traversals.iter().map(path).collect::<Vec<_>>()}));
}

pub(crate) fn report(tree: &Tree, rounds: Vec<Value>, completed: usize,
    state: &str, requested: usize, batch_size: usize, seed: u64) -> PyResult<String> {
    // An independent finalized-path invariant, never a reservation count.
    let joint = tree.joint_action_witness(completed)?.ok_or_else(||
        pyo3::exceptions::PyValueError::new_err("atomic work requires finalized joint ledger"))?;
    Ok(json!({"schema": SCHEMA, "scope": "observed_float32_backup_consistency",
        "independent_terminal_labels": false, "runtime_qualified": false,
        "input_state_utf8_sha256": format!("{:x}", Sha256::digest(state.as_bytes())),
        "requested_iterations": requested, "batch_size": batch_size, "chance_seed": seed,
        "rounds": rounds, "final": snapshot(tree),
        "joint_action_witness": serde_json::from_str::<Value>(&joint).expect("own JSON")}).to_string())
}

/// Python's digest uses sorted keys and compact JSON. Tree snapshots contain
/// only ASCII field names, integers, booleans and null: no float formatting or
/// Unicode escaping choice enters their fingerprint. Sort explicitly rather
/// than relying on serde_json's feature-unified map implementation.
fn canonical_bytes(value: &Value) -> Vec<u8> {
    fn write(value: &Value, bytes: &mut Vec<u8>) {
        match value {
            Value::Object(fields) => {
                bytes.push(b'{');
                let mut keys = fields.keys().collect::<Vec<_>>();
                keys.sort_unstable();
                for (index, key) in keys.into_iter().enumerate() {
                    if index > 0 { bytes.push(b','); }
                    serde_json::to_writer(&mut *bytes, key).expect("ASCII snapshot field");
                    bytes.push(b':');
                    write(&fields[key], bytes);
                }
                bytes.push(b'}');
            }
            Value::Array(rows) => {
                bytes.push(b'[');
                for (index, row) in rows.iter().enumerate() {
                    if index > 0 { bytes.push(b','); }
                    write(row, bytes);
                }
                bytes.push(b']');
            }
            _ => serde_json::to_writer(bytes, value).expect("integer/bool/null snapshot scalar"),
        }
    }
    let mut bytes = Vec::new();
    write(value, &mut bytes);
    bytes
}

fn fingerprint(value: &Value) -> String {
    format!("{:x}", Sha256::digest(canonical_bytes(value)))
}

/// Encode only actual collection differences. This does not predict backups.
/// Existing containers retain identity and shape; array tails are append-only.
fn collection_edits(before: &Value, after: &Value) -> PyResult<Vec<Value>> {
    fn walk(before: &Value, after: &Value, path: &mut Vec<Value>, edits: &mut Vec<Value>) -> PyResult<()> {
        let drift = || pyo3::exceptions::PyValueError::new_err("compact atomic collection structure drift");
        match (before, after) {
            (Value::Object(a), Value::Object(b)) => {
                if a.len() != b.len() || a.keys().any(|key| !b.contains_key(key)) { return Err(drift()); }
                let mut keys = a.keys().collect::<Vec<_>>(); keys.sort_unstable();
                for key in keys {
                    path.push(json!(key));
                    walk(&a[key], &b[key], path, edits)?;
                    path.pop();
                }
            }
            (Value::Array(a), Value::Array(b)) => {
                if b.len() < a.len() { return Err(drift()); }
                for (index, old) in a.iter().enumerate() {
                    path.push(json!(index));
                    walk(old, &b[index], path, edits)?;
                    path.pop();
                }
                if b.len() > a.len() {
                    edits.push(json!({"kind":"append", "path":path,
                        "before_length":a.len(), "rows":&b[a.len()..]}));
                }
            }
            (Value::Null, Value::Null) => (),
            (Value::Null, Value::Number(b)) if b.is_u64() =>
                edits.push(json!({"kind":"set", "path":path, "before":before, "after":after})),
            (Value::Number(a), Value::Null) if a.is_u64() =>
                edits.push(json!({"kind":"set", "path":path, "before":before, "after":after})),
            (Value::Number(a), Value::Number(b)) if a.is_u64() && b.is_u64() => {
                if a != b { edits.push(json!({"kind":"set", "path":path, "before":before, "after":after})); }
            }
            (Value::Bool(a), Value::Bool(b)) => {
                if a != b { edits.push(json!({"kind":"set", "path":path, "before":before, "after":after})); }
            }
            _ => return Err(drift()),
        }
        Ok(())
    }
    let mut edits = Vec::new();
    walk(before, after, &mut Vec::new(), &mut edits)?;
    Ok(edits)
}

/// Hash the REAL tree immediately around each REAL finalize. Retained output
/// is paths/hashes only; full transient snapshots are dropped each traversal.
pub(crate) fn commit_compact_round(tree: &mut Tree, traversals: &[Traversal], values: &[f32],
    before_collection: Value, rounds: &mut Vec<Value>) -> PyResult<()> {
    let collected = snapshot(tree);
    let edits = collection_edits(&before_collection, &collected)?;
    let mut commits = Vec::with_capacity(traversals.len());
    for traversal in traversals {
        let before_sha256 = fingerprint(&snapshot(tree));
        let end = match traversal.end {
            TraversalEnd::Ready(v) => json!({"kind":"ready", "value_bits":v.to_bits()}),
            TraversalEnd::Row(row) => json!({"kind":"row", "row":row}),
            TraversalEnd::Expanded => json!({"kind":"expanded"}),
        };
        let value = finalize(tree, traversal, values);
        let after_sha256 = fingerprint(&snapshot(tree));
        commits.push(json!({"path":path(traversal), "end":end,
            "before_sha256":before_sha256, "after_sha256":after_sha256,
            "root_value_bits":value.to_bits()}));
    }
    rounds.push(json!({"ordinal":rounds.len(), "status":"COMMITTED",
        "before_sha256":fingerprint(&before_collection), "collection_edits":edits,
        "collection_sha256":fingerprint(&collected),
        "row_value_bits":values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "commits":commits, "after_sha256":fingerprint(&snapshot(tree)), "accepted":traversals.len()}));
    Ok(())
}

pub(crate) fn cancel_compact_round(tree: &Tree, traversals: &[Traversal], before: Value,
    collected: Value, rounds: &mut Vec<Value>) -> PyResult<()> {
    // Caller has already applied the original reservation journal's cancel.
    // Observe its actual restored tree, never synthesize the expected rollback.
    rounds.push(json!({"ordinal":rounds.len(), "status":"CANCELLED",
        "before_sha256":fingerprint(&before), "collection_edits":collection_edits(&before, &collected)?,
        "collection_sha256":fingerprint(&collected), "after_sha256":fingerprint(&snapshot(tree)),
        "accepted":0, "attempted_paths":traversals.iter().map(path).collect::<Vec<_>>()}));
    Ok(())
}

pub(crate) fn compact_report(tree: &Tree, initial: Value, rounds: Vec<Value>, completed: usize,
    state: &str, requested: usize, batch_size: usize, seed: u64) -> PyResult<String> {
    let joint = tree.joint_action_witness(completed)?.ok_or_else(||
        pyo3::exceptions::PyValueError::new_err("atomic work requires finalized joint ledger"))?;
    Ok(json!({"schema":COMPACT_SCHEMA, "scope":"observed_float32_backup_consistency",
        "independent_terminal_labels":false, "runtime_qualified":false,
        "input_state_utf8_sha256":format!("{:x}", Sha256::digest(state.as_bytes())),
        "requested_iterations":requested, "batch_size":batch_size, "chance_seed":seed,
        "initial":initial, "rounds":rounds, "final":snapshot(tree),
        "joint_action_witness":serde_json::from_str::<Value>(&joint).expect("own JSON")}).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::{traverse, MultiPlyConfig, LeafPrice, SearchCounters, RoundReservation};
    use rand::{SeedableRng, rngs::StdRng};
    const STATE: &str = include_str!("test_fixtures/minimal.state");

    #[test]
    fn compact_canonical_fingerprint_matches_python_golden() {
        // Golden computed by Python's sorted-key compact JSON digest; integer
        // bit patterns avoid float-format differences between serializers.
        let tree = json!({"decisions":[{"visits":0, "depth":0, "children":[],
            "side_one":[{"visits":0, "total_bits":0}],
            "side_two":[{"visits":0, "total_bits":0}]}], "chances":[]});
        assert_eq!(fingerprint(&tree),
            "7f1b432957f4e885752f735d69678f154839165ab0122a3f8b03a1b0d8728746");
        assert_eq!(String::from_utf8(canonical_bytes(&tree)).unwrap(),
            r#"{"chances":[],"decisions":[{"children":[],"depth":0,"side_one":[{"total_bits":0,"visits":0}],"side_two":[{"total_bits":0,"visits":0}],"visits":0}]}"#);
        let before = json!({"chances":[{"child":null, "pending_row":0}], "decisions":[{"visits":0}]});
        let after = json!({"chances":[{"child":1, "pending_row":null}, {"child":null}],
            "decisions":[{"visits":1}]});
        assert_eq!(collection_edits(&before, &after).unwrap(), vec![
            json!({"kind":"set", "path":["chances",0,"child"], "before":null, "after":1}),
            json!({"kind":"set", "path":["chances",0,"pending_row"], "before":0, "after":null}),
            json!({"kind":"append", "path":["chances"], "before_length":1, "rows":[{"child":null}]}),
            json!({"kind":"set", "path":["decisions",0,"visits"], "before":0, "after":1}),
        ]);
        assert!(collection_edits(&after, &before).is_err());
        assert!(collection_edits(&before, &json!({"chances":{}, "decisions":[]})).is_err());
    }

    /// Actual native toy production, not a converted v1 record. This test is
    /// intentionally unexecuted until an isolated native test build is allowed.
    #[test]
    fn compact_native_deferred_reader_fixture_and_choice_parity() {
        pyo3::Python::initialize();
        let cfg = MultiPlyConfig { max_depth:3, c_puct:1.4, deep_ko_split:true,
            use_opponent_priors:false, fpu_reduction:None };
        let run = |encoding| {
            let mut state = crate::parse_state(STATE.trim()).unwrap();
            let mut tree = Tree::from_root(&state).unwrap();
            tree.joint_action_visits = Some(Default::default());
            let initial = snapshot(&tree);
            let mut rng = StdRng::seed_from_u64(17);
            let mut counters = SearchCounters::default();
            let mut rounds = Vec::new();
            let mut observed_paths = Vec::new();
            for _ in 0..4 {
                let before = snapshot(&tree);
                let mut values = Vec::new();
                let traversals = (0..4).map(|_| traverse(&mut tree, &mut state, &mut rng,
                    &cfg, &mut counters, &mut |_s, _b| {
                        let row = values.len();
                        values.push(if row % 2 == 0 {0.2} else {0.7});
                        LeafPrice::Deferred(row)
                    })).collect::<Vec<_>>();
                observed_paths.extend(traversals.iter().map(path));
                match encoding {
                    0 => { for traversal in &traversals { finalize(&mut tree, traversal, &values); } }
                    1 => commit_round(&mut tree, &traversals, &values, before, &mut rounds),
                    2 => commit_compact_round(&mut tree, &traversals, &values, before, &mut rounds).unwrap(),
                    _ => unreachable!(),
                }
            }
            (snapshot(&tree), observed_paths, initial, rounds, tree, counters)
        };
        let (ordinary, paths, _, _, _, _) = run(0);
        let (full, full_paths, _, _, _, _) = run(1);
        let (compact, compact_paths, initial, rounds, tree, counters) = run(2);
        assert_eq!(ordinary, full); assert_eq!(ordinary, compact);
        assert_eq!(paths, full_paths); assert_eq!(paths, compact_paths);
        for round in &rounds {
            assert!(round.get("before_collection").is_none());
            for commit in round["commits"].as_array().unwrap() {
                assert!(commit.get("before").is_none()); assert!(commit.get("after").is_none());
            }
        }
        let work:Value = serde_json::from_str(&compact_report(&tree, initial, rounds, 16,
            STATE.trim(), 16, 4, 17).unwrap()).unwrap();
        let outcome = crate::tree::MultiPlyOutcome {tree, counters, elapsed_s:1.0};
        let extra = format!("\"joint_action_witness\":{},\"atomic_work\":{}", work["joint_action_witness"], work);
        println!("ATOMIC_COMPACT_READER_FIXTURE:{}", crate::tree::multiply_report_json(
            &outcome, 16, &cfg, 17, "synthetic", &extra, false));
    }

    #[test]
    fn compact_cancelled_native_fixture_records_actual_restoration() {
        pyo3::Python::initialize();
        let mut state = crate::parse_state(STATE.trim()).unwrap();
        let mut tree = Tree::from_root(&state).unwrap(); tree.joint_action_visits = Some(Default::default());
        let cfg = MultiPlyConfig {max_depth:2, c_puct:1.4, deep_ko_split:true,
            use_opponent_priors:false, fpu_reduction:None};
        let mut rng = StdRng::seed_from_u64(17); let mut counters = SearchCounters::default();
        let initial = snapshot(&tree); let mut reservation = RoundReservation::new(&tree);
        let traversals = (0..4).map(|_| crate::tree::traverse_cancellable(&mut tree, &mut state,
            &mut rng, &cfg, &mut counters, &mut |_s,_b| LeafPrice::Ready(0.5), &mut reservation))
            .collect::<Vec<_>>();
        let collected = snapshot(&tree);
        assert_ne!(fingerprint(&collected), fingerprint(&initial));
        reservation.cancel(&mut tree, &traversals).unwrap();
        let mut rounds = Vec::new();
        cancel_compact_round(&tree, &traversals, initial.clone(), collected, &mut rounds).unwrap();
        assert_eq!(rounds[0]["after_sha256"], json!(fingerprint(&snapshot(&tree))));
        assert_eq!(rounds[0]["before_sha256"], rounds[0]["after_sha256"]);
        assert_eq!(rounds[0]["accepted"], 0);
        let work:Value = serde_json::from_str(&compact_report(&tree, initial, rounds, 0,
            STATE.trim(), 16, 4, 17).unwrap()).unwrap();
        let outcome = crate::tree::MultiPlyOutcome {tree, counters, elapsed_s:1.0};
        let extra = format!("\"time_budget_exhausted\":true,\"joint_action_witness\":{},\"atomic_work\":{}",
            work["joint_action_witness"], work);
        println!("ATOMIC_COMPACT_CANCELLED_FIXTURE:{}", crate::tree::multiply_report_json(
            &outcome, 0, &cfg, 17, "synthetic", &extra, false));
    }

    #[test]
    fn observed_batches_preserve_actual_choices_and_values() {
        pyo3::Python::initialize();
        let cfg = MultiPlyConfig { max_depth: 3, c_puct: 1.4, deep_ko_split: true,
            use_opponent_priors: false, fpu_reduction: None };
        let run = |observed| {
            let mut state = crate::parse_state(STATE.trim()).unwrap();
            let mut tree = Tree::from_root(&state).unwrap();
            tree.joint_action_visits = Some(Default::default());
            let mut rng = StdRng::seed_from_u64(17);
            let mut counters = SearchCounters::default();
            let mut rounds = Vec::new();
            for _ in 0..4 {
                let before = snapshot(&tree);
                let paths: Vec<_> = (0..4).map(|_| traverse(&mut tree, &mut state,
                    &mut rng, &cfg, &mut counters, &mut |_s, _b| LeafPrice::Ready(0.5))).collect();
                if observed { commit_round(&mut tree, &paths, &[], before, &mut rounds); }
                else { for p in &paths { finalize(&mut tree, p, &[]); } }
            }
            (snapshot(&tree), rounds, tree)
        };
        let (ordinary, _, _) = run(false);
        let (measured, rounds, tree) = run(true);
        assert_eq!(ordinary, measured);
        assert_eq!(rounds.len(), 4);
        for round in &rounds { assert_eq!(round["accepted"], 4); assert_eq!(round["commits"].as_array().unwrap().len(), 4); }
        assert!(report(&tree, rounds, 16, STATE.trim(), 16, 4, 17).is_ok());
    }

    #[test]
    fn cancelled_reservations_are_not_commits() {
        pyo3::Python::initialize();
        let mut state = crate::parse_state(STATE.trim()).unwrap();
        let mut tree = Tree::from_root(&state).unwrap();
        tree.joint_action_visits = Some(Default::default());
        let cfg = MultiPlyConfig { max_depth: 2, c_puct: 1.4, deep_ko_split: true,
            use_opponent_priors: false, fpu_reduction: None };
        let mut rng = StdRng::seed_from_u64(17);
        let mut counters = SearchCounters::default();
        let before = snapshot(&tree);
        let mut reservation = RoundReservation::new(&tree);
        let paths: Vec<_> = (0..4).map(|_| crate::tree::traverse_cancellable(&mut tree, &mut state,
            &mut rng, &cfg, &mut counters, &mut |_s, _b| LeafPrice::Ready(0.5), &mut reservation)).collect();
        assert!(tree.joint_action_witness(0).is_err());
        let collected = snapshot(&tree);
        reservation.cancel(&mut tree, &paths).unwrap();
        let mut rounds = Vec::new();
        cancel_round(&tree, &paths, before.clone(), collected, &mut rounds);
        assert_eq!(snapshot(&tree), before);
        assert_eq!(rounds[0]["accepted"], 0);
        let work: Value = serde_json::from_str(&report(&tree, rounds, 0, STATE.trim(), 16, 4, 17).unwrap()).unwrap();
        let outcome = crate::tree::MultiPlyOutcome { tree, counters, elapsed_s: 1.0 };
        let extra = format!("\"time_budget_exhausted\":true,\"joint_action_witness\":{},\"atomic_work\":{}",
            work["joint_action_witness"], work);
        println!("ATOMIC_CANCELLED_FIXTURE:{}", crate::tree::multiply_report_json(
            &outcome, 0, &cfg, 17, "synthetic", &extra, false));
    }

    /// Transport in stdout only for the Python reader integration test. Uses
    /// the existing synthetic state, not fresh sources or a model checkpoint.
    #[test]
    fn independent_reader_fixture() {
        pyo3::Python::initialize();
        let cfg = MultiPlyConfig { max_depth: 3, c_puct: 1.4, deep_ko_split: true,
            use_opponent_priors: false, fpu_reduction: None };
        let mut state = crate::parse_state(STATE.trim()).unwrap();
        let mut tree = Tree::from_root(&state).unwrap();
        tree.joint_action_visits = Some(Default::default());
        let mut rng = StdRng::seed_from_u64(17);
        let mut counters = SearchCounters::default();
        let mut rounds = Vec::new();
        for _ in 0..4 {
            let before = snapshot(&tree);
            let mut values = Vec::new();
            let paths: Vec<_> = (0..4).map(|_| traverse(&mut tree, &mut state,
                &mut rng, &cfg, &mut counters, &mut |_s, _b| {
                    let row = values.len();
                    values.push(if row % 2 == 0 { 0.2 } else { 0.7 });
                    LeafPrice::Deferred(row)
                })).collect();
            commit_round(&mut tree, &paths, &values, before, &mut rounds);
        }
        let work: Value = serde_json::from_str(&report(&tree, rounds, 16, STATE.trim(), 16, 4, 17).unwrap()).unwrap();
        let outcome = crate::tree::MultiPlyOutcome { tree, counters, elapsed_s: 1.0 };
        let extra = format!("\"joint_action_witness\":{},\"atomic_work\":{}",
            work["joint_action_witness"], work);
        println!("ATOMIC_READER_FIXTURE:{}", crate::tree::multiply_report_json(
            &outcome, 16, &cfg, 17, "synthetic", &extra, false));
    }
}
