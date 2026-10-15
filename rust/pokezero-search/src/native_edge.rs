//! Independent regeneration of a captured native edge's transition support.
//! No search, evaluator, Showdown, RNG coupling or scientific admission.
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::MoveChoice;
use poke_engine::state::{Side, State};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub(crate) const EDGE_SCHEMA: &str = "pokezero.native-captured-edge.v1";
const SUPPORT_SCHEMA: &str = "pokezero.native-edge-support.v1";
fn hash(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))
}
fn refuse(text: &str) -> PyErr {
    PyValueError::new_err(text.to_owned())
}

fn validate_branch_masses(masses: impl Iterator<Item = f32>) -> PyResult<()> {
    let mut total = 0.0f32;
    for mass in masses {
        if !mass.is_finite() || mass < 0.0 {
            return Err(refuse("native edge: branch mass invalid"));
        }
        total += mass;
    }
    // Match expand_edge's engine-mass invariant; not independent physics.
    if !total.is_finite() || (total - 100.0).abs() >= 0.5 {
        return Err(refuse(
            "native edge: branch mass total differs from tree invariant",
        ));
    }
    Ok(())
}

pub(crate) fn choice(side: &Side, action: &MoveChoice) -> Value {
    let (kind, slot) = match action {
        MoveChoice::Move(i) => ("move", Some(*i as u8)),
        MoveChoice::Switch(i) => ("switch", Some(*i as u8)),
        MoveChoice::Struggle => ("struggle", None),
        MoveChoice::None => ("none", None),
    };
    json!({"kind":kind,"slot":slot,"engine_id":action.to_string(side)})
}

/// Called at the actual applied BranchSeam, not reconstructed from identities.
pub(crate) fn capture(state: &State, seam: &crate::tree::BranchSeam<'_>) -> Value {
    let mut pre = state.clone();
    pre.reverse_instructions(&seam.instructions.to_vec());
    let instructions = format!("{:?}", seam.instructions);
    json!({"schema":EDGE_SCHEMA,"root":seam.depth == 0,
        "branch_index":seam.branch_index,"branch_on_damage":seam.branch_on_damage,
        "actions":{"p1":choice(&pre.side_one,seam.s1),"p2":choice(&pre.side_two,seam.s2)},
        "instruction_count":seam.instructions.len(),"instructions_debug":instructions,
        "instructions_debug_sha256":hash(&instructions)})
}

fn exact_keys(value: &Value, keys: &[&str]) -> bool {
    value.as_object().is_some_and(|object| {
        object.len() == keys.len() && keys.iter().all(|key| object.contains_key(*key))
    })
}

fn resolve(side: &Side, options: &[MoveChoice], requested: &Value) -> PyResult<Option<MoveChoice>> {
    if !exact_keys(requested, &["kind", "slot", "engine_id"]) || !requested["engine_id"].is_string()
    {
        return Err(refuse("native edge: exact slot-preserving action required"));
    }
    let valid = match requested["kind"].as_str() {
        Some("move") => requested["slot"].as_u64().is_some_and(|v| v < 4),
        Some("switch") => requested["slot"].as_u64().is_some_and(|v| v < 6),
        Some("struggle") => requested["slot"].is_null() && requested["engine_id"] == "struggle",
        Some("none") => requested["slot"].is_null() && requested["engine_id"] == "No Move",
        _ => false,
    };
    if !valid {
        return Err(refuse("native edge: malformed action slot/kind"));
    }
    let matching = options
        .iter()
        .filter(|v| choice(side, v) == *requested)
        .collect::<Vec<_>>();
    if matching.len() > 1 {
        return Err(refuse("native edge: duplicate exact option"));
    }
    Ok(matching.first().map(|v| (*v).clone()))
}

fn support(before: &str, after: &str, edge: &Value) -> PyResult<Value> {
    support_details(before, after, edge, false).map(|v| v.0)
}

// The old export remains byte/schema-compatible. The separate prospective
// export retains ALL positive-mass successors, never a selected subset.
fn support_details(before: &str, after: &str, edge: &Value, project: bool) -> PyResult<(Value, Value)> {
    if before.len() + after.len() > 8 * 1024 * 1024
        || !exact_keys(
            edge,
            &[
                "schema",
                "root",
                "branch_index",
                "branch_on_damage",
                "actions",
                "instruction_count",
                "instructions_debug",
                "instructions_debug_sha256",
            ],
        )
        || edge["schema"] != EDGE_SCHEMA
        || !edge["root"].is_boolean()
        || !edge["branch_on_damage"].is_boolean()
        || edge["branch_index"].as_u64().is_none()
        || !edge["instruction_count"]
            .as_u64()
            .is_some_and(|v| v <= 65536)
        || !exact_keys(&edge["actions"], &["p1", "p2"])
        || !edge["instructions_debug"]
            .as_str()
            .is_some_and(|v| !v.is_empty() && v.len() <= 8 * 1024 * 1024)
        || edge["instructions_debug_sha256"] != hash(edge["instructions_debug"].as_str().unwrap())
    {
        return Err(refuse(
            "native edge: bounded captured law/instruction binding required",
        ));
    }
    let mut state = crate::parse_state(before)?;
    // Reject noncanonical successor serialization too, not just equal digests.
    let successor = crate::parse_state(after)?;
    if state.serialize() != before || successor.serialize() != after {
        return Err(refuse(
            "native edge: actual canonical serialized states required",
        ));
    }
    let root = edge["root"].as_bool().unwrap();
    let (mut one, mut two) = if root {
        state.root_get_all_options()
    } else {
        state.get_all_options()
    };
    let terminal = state.battle_is_over() != 0.0;
    // Match the tree's interior-only empty-option fallback. Terminal states
    // cannot produce a new searched edge, even if residual options exist.
    if !root && !terminal {
        if one.is_empty() {
            one.push(MoveChoice::None);
        }
        if two.is_empty() {
            two.push(MoveChoice::None);
        }
    }
    let s1 = resolve(&state.side_one, &one, &edge["actions"]["p1"])?;
    let s2 = resolve(&state.side_two, &two, &edge["actions"]["p2"])?;
    let admissible = !terminal && s1.is_some() && s2.is_some();
    let mut matching = Vec::new();
    let mut recorded_matches = false;
    let mut instructions_match = false;
    let mut branches = 0;
    let mut masses = Vec::new();
    let mut successors = Vec::new();
    let mut payload_bytes = before.len() + after.len();
    let pre_projection = if project {
        crate::scoped_boundary::projection(&state, before, root)
    } else { Value::Null };
    if admissible {
        let generated = generate_instructions_from_move_pair(
            &mut state,
            &s1.unwrap(),
            &s2.unwrap(),
            edge["branch_on_damage"].as_bool().unwrap(),
        );
        if state.serialize() != before {
            return Err(refuse("native edge: generation changed prestate"));
        }
        if generated.len() > 4096 {
            return Err(refuse("native edge: regeneration branch cap reached"));
        }
        let rows = if generated.is_empty() {
            vec![(Vec::new(), 100.0)]
        } else {
            generated
                .into_iter()
                .map(|v| (v.instruction_list, v.percentage))
                .collect()
        };
        validate_branch_masses(rows.iter().map(|(_, mass)| *mass))?;
        branches = rows.len();
        for (index, (instructions, percentage)) in rows.into_iter().enumerate() {
            if !percentage.is_finite() || percentage < 0.0 {
                return Err(refuse("native edge: branch mass invalid"));
            }
            let text = format!("{:?}", instructions);
            let mut candidate = state.clone();
            candidate.apply_instructions(&instructions);
            let candidate_text = candidate.serialize();
            let matches = percentage > 0.0 && candidate_text == after;
            if project {
                masses.push(json!({"branch_index":index,"percentage":percentage}));
                if percentage > 0.0 {
                    let boundary = crate::scoped_boundary::projection(&candidate, &candidate_text, false);
                    payload_bytes += candidate_text.len() + boundary.to_string().len();
                    if payload_bytes > 8 * 1024 * 1024 {
                        return Err(refuse("native successors: full positive roster payload cap reached"));
                    }
                    successors.push(json!({"branch_index":index,"percentage":percentage,
                        "state_utf8":candidate_text,"boundary":boundary}));
                }
            }
            if matches {
                matching.push(index);
            }
            if edge["branch_index"].as_u64() == Some(index as u64) {
                instructions_match = edge["instruction_count"].as_u64()
                    == Some(instructions.len() as u64)
                    && edge["instructions_debug"] == text;
                recorded_matches = matches && instructions_match;
            }
        }
    }
    if state.serialize() != before {
        return Err(refuse("native edge: prestate changed during support read"));
    }
    let report = json!({"schema":SUPPORT_SCHEMA,"before_utf8_sha256":hash(before),"after_utf8_sha256":hash(after),
        "captured_edge":edge,"prestate_terminal":terminal,"joint_action_admissible":admissible,
        "regenerated_branches":branches,"matching_successor_branch_indices":matching,
        "successor_in_native_support":!matching.is_empty(),"recorded_branch_instructions_match":instructions_match,
        "recorded_edge_supported":recorded_matches,"private_controller_only":true,
        "search_or_evaluator_run":false,"independent_physics_validated":false,
        "showdown_transition_validated":false,"equal_rng_coupling":false,
        "runtime_qualified":false,"scientific_collection_authorized":false});
    let projected = if project { json!({"schema":"pokezero.native-scoped-successor-support.v1",
        "captured_support":report,"prestate_boundary":pre_projection,"branch_masses":masses,
        "positive_successors":successors,"complete_positive_mass_roster":true,
        "scope":"captured native prestate, exact joint action and original branch_on_damage law",
        "private_controller_only":true,"independent_physics_validated":false,
        "showdown_transition_validated":false,"equal_rng_coupling":false,"runtime_qualified":false,
        "scientific_collection_authorized":false}) } else { Value::Null };
    Ok((report,projected))
}

#[pyfunction]
pub fn native_edge_support(
    before: &str,
    after: &str,
    captured_edge_json: &str,
) -> PyResult<String> {
    if captured_edge_json.len() > 8 * 1024 * 1024 {
        return Err(refuse("native edge: capture JSON cap reached"));
    }
    let edge: Value = serde_json::from_str(captured_edge_json)
        .map_err(|_| refuse("native edge: invalid capture JSON"))?;
    support(before, after, &edge).map(|v| v.to_string())
}

/// Prospective private diagnostic; no model/search/Showdown or RNG coupling.
#[pyfunction]
pub fn native_scoped_successor_support(before: &str, after: &str, captured_edge_json: &str) -> PyResult<String> {
    if captured_edge_json.len() > 8 * 1024 * 1024 {
        return Err(refuse("native successors: capture JSON cap reached"));
    }
    let edge: Value = serde_json::from_str(captured_edge_json)
        .map_err(|_| refuse("native successors: invalid capture JSON"))?;
    let value = support_details(before, after, &edge, true)?.1.to_string();
    if value.len() > 8 * 1024 * 1024 {
        return Err(refuse("native successors: complete export payload cap reached"));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use poke_engine::state::PokemonIndex;
    fn toy() -> State {
        State::deserialize(include_str!("test_fixtures/minimal.state").trim())
    }
    #[test]
    fn malformed_mass_totals_refuse_regeneration_evidence() {
        for masses in [
            vec![],
            vec![0.0],
            vec![99.5],
            vec![-1.0, 101.0],
            vec![f32::NAN],
            vec![f32::INFINITY],
        ] {
            assert!(validate_branch_masses(masses.into_iter()).is_err());
        }
        assert!(validate_branch_masses([49.9, 50.1].into_iter()).is_ok());
    }
    fn actual(
        state: &State,
        one: MoveChoice,
        two: MoveChoice,
        law: bool,
        index: usize,
        root: bool,
    ) -> (String, Value) {
        let mut pre = state.clone();
        let generated = generate_instructions_from_move_pair(&mut pre, &one, &two, law);
        let instructions = generated[index].instruction_list.clone();
        let mut after = state.clone();
        after.apply_instructions(&instructions);
        let seam = crate::tree::BranchSeam {
            s1: &one,
            s2: &two,
            instructions: &instructions,
            parent: None,
            chance: 0,
            branch_index: index,
            branch_on_damage: law,
            depth: if root { 0 } else { 1 },
        };
        (after.serialize(), capture(&after, &seam))
    }
    #[test]
    fn actual_edge_regenerates_without_mutating_state() {
        let state = toy();
        let before = state.serialize();
        let (one, two) = state.root_get_all_options();
        let (after, edge) = actual(&state, one[0].clone(), two[0].clone(), true, 0, true);
        let read = support(&before, &after, &edge).unwrap();
        assert_eq!(read["recorded_edge_supported"], true);
        assert_eq!(state.serialize(), before);
        let mut altered = State::deserialize(&after);
        altered.side_one.get_active().hp = 1;
        assert_eq!(
            support(&before, &altered.serialize(), &edge).unwrap()["recorded_edge_supported"],
            false
        );
        let mut changed = edge.clone();
        changed["instructions_debug"] = json!("[]");
        assert!(support(&before, &after, &changed).is_err());
        changed["instructions_debug_sha256"] = json!(hash("[]"));
        assert_eq!(
            support(&before, &after, &changed).unwrap()["recorded_edge_supported"],
            false
        );
    }
    #[test]
    fn duplicate_move_ids_retain_exact_slots_and_unavailable_slots_refuse_support() {
        let mut state = toy();
        state.side_one.get_active().moves.m1 =
            state.side_one.get_active_immutable().moves.m0.clone();
        let (one, two) = state.root_get_all_options();
        let (after, edge) = actual(&state, one[1].clone(), two[0].clone(), false, 0, true);
        assert_eq!(edge["actions"]["p1"]["slot"], 1);
        assert_eq!(
            support(&state.serialize(), &after, &edge).unwrap()["recorded_edge_supported"],
            true
        );
        let mut bad = edge.clone();
        bad["actions"]["p1"]["slot"] = json!(3);
        assert_eq!(
            support(&state.serialize(), &after, &bad).unwrap()["joint_action_admissible"],
            false
        );
        bad["actions"]["p1"]["slot"] = json!(true);
        assert!(support(&state.serialize(), &after, &bad).is_err());
    }
    #[test]
    fn forced_replacement_and_terminal_prestate() {
        let mut state = toy();
        state.side_one.pokemon[PokemonIndex::P1] = state.side_one.get_active_immutable().clone();
        state.side_one.get_active().hp = 0;
        state.side_one.force_switch = true;
        let (one, two) = state.get_all_options();
        let (after, edge) = actual(&state, one[0].clone(), two[0].clone(), true, 0, false);
        assert_eq!(edge["actions"]["p1"]["kind"], "switch");
        assert_eq!(
            support(&state.serialize(), &after, &edge).unwrap()["recorded_edge_supported"],
            true
        );
        let mut terminal = toy();
        terminal.side_one.get_active().hp = 0;
        let read = support(&terminal.serialize(), &terminal.serialize(), &edge).unwrap();
        assert_eq!(read["prestate_terminal"], true);
        assert_eq!(read["joint_action_admissible"], false);
    }
    #[test]
    fn actual_tree_seams_regenerate_root_and_interior_edges() {
        use crate::tree::{finalize, traverse, LeafPrice, MultiPlyConfig, SearchCounters, Tree};
        use rand::{rngs::StdRng, SeedableRng};
        let mut state = toy();
        let original = state.serialize();
        let mut tree = Tree::from_root(&state).unwrap();
        let cfg = MultiPlyConfig {
            max_depth: 4,
            c_puct: 1.4,
            deep_ko_split: true,
            use_opponent_priors: false,
            fpu_reduction: None,
        };
        let mut rng = StdRng::seed_from_u64(42);
        let mut counters = SearchCounters::default();
        let mut roots = 0;
        let mut interiors = 0;
        let mut price = |leaf: &State, seam: &crate::tree::BranchSeam| {
            let edge = capture(leaf, seam);
            let mut pre = leaf.clone();
            pre.reverse_instructions(&seam.instructions.to_vec());
            assert_eq!(
                support(&pre.serialize(), &leaf.serialize(), &edge).unwrap()
                    ["recorded_edge_supported"],
                true
            );
            if seam.depth == 0 {
                roots += 1;
            } else {
                interiors += 1;
            }
            LeafPrice::Ready(0.5)
        };
        for _ in 0..64 {
            let path = traverse(
                &mut tree,
                &mut state,
                &mut rng,
                &cfg,
                &mut counters,
                &mut price,
            );
            finalize(&mut tree, &path, &[]);
            assert_eq!(state.serialize(), original);
        }
        assert!(roots > 0 && interiors > 0);
    }
    #[test]
    fn stochastic_alternative_and_wrong_branch_law_are_not_recorded_edges() {
        let state = toy();
        let (one, two) = state.root_get_all_options();
        let mut pre = state.clone();
        let generated = generate_instructions_from_move_pair(&mut pre, &one[0], &two[0], true);
        assert!(generated.len() > 1);
        let (after, edge) = actual(&state, one[0].clone(), two[0].clone(), true, 0, true);
        let mut alternative = state.clone();
        alternative.apply_instructions(&generated[1].instruction_list);
        assert_ne!(after, alternative.serialize());
        let read = support(&state.serialize(), &alternative.serialize(), &edge).unwrap();
        assert_eq!(read["successor_in_native_support"], true);
        assert_eq!(read["recorded_edge_supported"], false);
        let mut wrong_law = edge;
        wrong_law["branch_on_damage"] = json!(false);
        assert_eq!(
            support(&state.serialize(), &after, &wrong_law).unwrap()["recorded_edge_supported"],
            false
        );
    }
    #[test]
    fn terminal_arrival_and_explicit_wait() {
        let mut state = toy();
        state.side_two.get_active().hp = 1;
        let (one, two) = state.root_get_all_options();
        let (after, edge) = actual(&state, one[0].clone(), two[0].clone(), true, 0, true);
        assert_ne!(State::deserialize(&after).battle_is_over(), 0.0);
        assert_eq!(
            support(&state.serialize(), &after, &edge).unwrap()["recorded_edge_supported"],
            true
        );
        let mut replacement = toy();
        replacement.side_one.pokemon[PokemonIndex::P1] =
            replacement.side_one.get_active_immutable().clone();
        replacement.side_one.get_active().hp = 0;
        replacement.side_one.force_switch = true;
        let (one, two) = replacement.get_all_options();
        assert!(matches!(two[0], MoveChoice::None));
        let (after, edge) = actual(
            &replacement,
            one[0].clone(),
            two[0].clone(),
            false,
            0,
            false,
        );
        assert_eq!(edge["actions"]["p2"]["kind"], "none");
        assert_eq!(
            support(&replacement.serialize(), &after, &edge).unwrap()["recorded_edge_supported"],
            true
        );
    }
    #[test]
    fn complete_projected_roster_preserves_alternative_and_original_export() {
        let state=toy();let before=state.serialize();
        let (one,two)=state.root_get_all_options();
        let (after,edge)=actual(&state,one[0].clone(),two[0].clone(),true,0,true);
        let (old,full)=support_details(&before,&after,&edge,true).unwrap();
        assert_eq!(old,support(&before,&after,&edge).unwrap());
        assert_eq!(old,full["captured_support"]);
        let masses=full["branch_masses"].as_array().unwrap();
        let rows=full["positive_successors"].as_array().unwrap();
        assert_eq!(masses.len(),old["regenerated_branches"].as_u64().unwrap() as usize);
        assert_eq!(rows.len(),masses.iter().filter(|r|r["percentage"].as_f64().unwrap()>0.).count());
        assert!(rows.len()>1);
        assert!(rows.iter().any(|r|r["state_utf8"]!=after));
        for row in rows {
            let text=row["state_utf8"].as_str().unwrap();let native=State::deserialize(text);
            assert_eq!(row["boundary"],crate::scoped_boundary::projection(&native,text,false));
            assert_eq!(row["percentage"],masses[row["branch_index"].as_u64().unwrap() as usize]["percentage"]);
        }
        assert_eq!(state.serialize(),before);
        let legacy:String=native_edge_support(&before,&after,&edge.to_string()).unwrap();
        assert_eq!(serde_json::from_str::<Value>(&legacy).unwrap(),old);
    }

    #[test]
    fn projected_terminal_prestate_has_explicit_empty_support() {
        let state=toy();let (one,two)=state.root_get_all_options();
        let (_,edge)=actual(&state,one[0].clone(),two[0].clone(),true,0,true);
        let mut terminal=state;terminal.side_one.get_active().hp=0;
        let text=terminal.serialize();let full=support_details(&text,&text,&edge,true).unwrap().1;
        assert_eq!(full["branch_masses"],json!([]));assert_eq!(full["positive_successors"],json!([]));
        assert_eq!(full["captured_support"]["joint_action_admissible"],false);
    }

    #[test]
    fn successor_reader_fixture() {
        let state=toy();let (one,two)=state.root_get_all_options();
        let (after,edge)=actual(&state,one[0].clone(),two[0].clone(),true,0,true);
        let step=json!({"key":[0,0],"parent":null,"depth":0,"before":state.serialize(),"after":after,
            "branch_on_damage":true,"side_one_choice":{"kind":edge["actions"]["p1"]["kind"],"engine_id":edge["actions"]["p1"]["engine_id"]},
            "side_two_choice":{"kind":edge["actions"]["p2"]["kind"],"engine_id":edge["actions"]["p2"]["engine_id"]},"native_edge":edge});
        let full=native_scoped_successor_support(step["before"].as_str().unwrap(),step["after"].as_str().unwrap(),
            &step["native_edge"].to_string()).unwrap();
        println!("NATIVE_SUCCESSOR_READER_FIXTURE:{}",json!({"step":step,"support":serde_json::from_str::<Value>(&full).unwrap()}));
    }
    #[test]
    fn reader_fixture() {
        let state = toy();
        let (one, two) = state.root_get_all_options();
        let (after, edge) = actual(&state, one[0].clone(), two[0].clone(), true, 0, true);
        let step = json!({"key":[0,0],"parent":null,"depth":0,"before":state.serialize(),"after":after,
            "branch_on_damage":true,"side_one_choice":{"kind":edge["actions"]["p1"]["kind"],
                "engine_id":edge["actions"]["p1"]["engine_id"]},
            "side_two_choice":{"kind":edge["actions"]["p2"]["kind"],"engine_id":edge["actions"]["p2"]["engine_id"]},
            "native_edge":edge});
        let checked = support(
            step["before"].as_str().unwrap(),
            step["after"].as_str().unwrap(),
            &step["native_edge"],
        )
        .unwrap();
        println!(
            "NATIVE_EDGE_READER_FIXTURE:{}",
            json!({"step":step,"support":checked})
        );
    }
}
