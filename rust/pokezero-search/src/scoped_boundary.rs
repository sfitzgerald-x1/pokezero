//! Read-only projection of an ACTUAL captured native State. No branches, search,
//! model or simulator are run. Identity is local to this engine party order;
//! joining it to Showdown requires an independently validated party mapping.
use poke_engine::engine::state::MoveChoice;
use poke_engine::pokemon::PokemonName;
use poke_engine::state::{Side, State};
use pyo3::prelude::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

fn choice(side: &Side, action: &MoveChoice) -> Value {
    match action {
        MoveChoice::Move(index) => json!({"kind":"move", "slot":*index as u8,
            "engine_id":action.to_string(side)}),
        MoveChoice::Switch(index) => json!({"kind":"switch", "slot":*index as u8,
            "engine_id":action.to_string(side)}),
        MoveChoice::Struggle => json!({"kind":"struggle", "slot":null, "engine_id":"struggle"}),
        MoveChoice::None => json!({"kind":"none", "slot":null, "engine_id":"No Move"}),
    }
}

fn side_projection(side: &Side, options: &[MoveChoice], root: bool, terminal: bool) -> Value {
    let active = side.get_active_immutable();
    let raw = options.iter().map(|v| choice(side,v)).collect::<Vec<_>>();
    let mut arms = raw.clone();
    // Mirrors tree::new_decision_node, NOT the rejecting root constructor.
    let fallback = !root && arms.is_empty();
    if fallback { arms.push(choice(side, &MoveChoice::None)); }
    let moves = active.moves.into_iter().enumerate().map(|(slot,mv)| json!({
        "slot":slot, "engine_id":mv.id.to_string().to_lowercase(),
        "pp":mv.pp, "disabled":mv.disabled})).collect::<Vec<_>>();
    let party = side.pokemon.into_iter().enumerate().filter(|(_,p)| p.id != PokemonName::NONE)
        .map(|(slot,p)| json!({"slot":slot,"engine_id":p.id.to_string().to_lowercase()}))
        .collect::<Vec<_>>();
    json!({"active_identity":{"slot":side.active_index as u8,
            "engine_id":active.id.to_string().to_lowercase()},
        "hp":[active.hp, active.maxhp], "fainted":active.hp <= 0,
        "status":active.status.to_string().to_lowercase(), "active_moves":moves,
        "party_identities":party, "engine_options":raw, "search_arms":arms,
        "interior_empty_fallback":fallback,
        // Declared projection law: a terminal boundary offers no NEW choices.
        // Retain the engine/search option vectors as distinct diagnostic facts.
        "effective_legal_choices":if terminal { Vec::<Value>::new() } else { raw },
        "force_switch":side.force_switch, "force_trapped":side.force_trapped,
        "slow_uturn_move":side.slow_uturn_move})
}

pub(crate) fn projection(state: &State, state_str: &str, root: bool) -> Value {
    let (one,two) = if root { state.root_get_all_options() } else { state.get_all_options() };
    let outcome = state.battle_is_over();
    json!({"schema":"pokezero.search-over-raw.native-scoped-boundary.v1",
        "state_utf8_sha256":format!("{:x}",Sha256::digest(state_str.as_bytes())),
        "root":root, "team_preview":state.team_preview, "native_outcome":outcome,
        "terminal":outcome != 0.0,
        "terminal_choice_law":"no_new_choices_at_terminal",
        "root_search_admissible":!root || (!one.is_empty() && !two.is_empty()),
        "p1":side_projection(&state.side_one,&one,root,outcome != 0.0),
        "p2":side_projection(&state.side_two,&two,root,outcome != 0.0),
        "native_pp_class":"engine_pp_model", "party_mapping_validated":false,
        "showdown_requests_inferred":false, "branch_transition_validated":false,
        "runtime_qualified":false, "scientific_collection_authorized":false})
}

/// Private controller utility only. Caller must keep state/projection private.
/// Reports all option slots; does not use species-first action parsing. Raw
/// native outcome is not an independent terminal label (including double KO).
#[pyfunction]
pub fn native_scoped_boundary(state_str: &str, root: bool) -> PyResult<String> {
    Ok(projection(&crate::parse_state(state_str)?,state_str,root).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use poke_engine::engine::state::PokemonVolatileStatus;
    use poke_engine::state::PokemonIndex;
    const TOY: &str = include_str!("test_fixtures/minimal.state");
    fn toy() -> State { State::deserialize(TOY.trim()) }
    fn view(state: &State, root: bool) -> Value { projection(state,&state.serialize(),root) }

    #[test]
    fn actual_serialized_projection_preserves_state_and_all_move_slots() {
        let state=toy(); let before=state.serialize();
        let report:Value=serde_json::from_str(&native_scoped_boundary(&before,true).unwrap()).unwrap();
        assert_eq!(before,state.serialize());
        assert_eq!(report["p1"]["active_identity"],json!({"slot":0,"engine_id":"charmander"}));
        assert_eq!(report["p1"]["hp"],json!([100,100]));
        assert_eq!(report["p1"]["engine_options"].as_array().unwrap().len(),2);
        assert_eq!(report["p1"]["active_moves"].as_array().unwrap().len(),4);
        assert_eq!(report["p1"]["engine_options"][1]["slot"],1);
        assert_eq!(report["state_utf8_sha256"],format!("{:x}",Sha256::digest(before.as_bytes())));
        assert_eq!(report["showdown_requests_inferred"],false);
    }
    #[test]
    fn duplicate_species_and_moves_never_collapse_slots() {
        let mut state=toy();
        state.side_one.pokemon[PokemonIndex::P1]=state.side_one.get_active_immutable().clone();
        state.side_one.pokemon[PokemonIndex::P2]=state.side_one.get_active_immutable().clone();
        state.side_one.get_active().moves.m1=state.side_one.get_active_immutable().moves.m0.clone();
        let report=view(&state,true); let options=report["p1"]["engine_options"].as_array().unwrap();
        assert_eq!(options.len(),4);
        assert_eq!(options[0]["engine_id"],options[1]["engine_id"]);
        assert_ne!(options[0]["slot"],options[1]["slot"]);
        assert_eq!(options[2]["engine_id"],options[3]["engine_id"]);
        assert_ne!(options[2]["slot"],options[3]["slot"]);
    }
    #[test]
    fn root_trap_and_slow_uturn_are_not_interior_options() {
        let mut state=toy();
        state.side_one.pokemon[PokemonIndex::P1]=state.side_one.get_active_immutable().clone();
        state.side_one.force_trapped=true;
        assert_eq!(view(&state,true)["p1"]["engine_options"].as_array().unwrap().len(),2);
        assert_eq!(view(&state,false)["p1"]["engine_options"].as_array().unwrap().len(),3);
        state.side_one.force_trapped=false; state.side_one.force_switch=true;
        state.side_one.slow_uturn_move=true;
        assert_eq!(view(&state,true)["p1"]["engine_options"][0]["kind"],"move");
        assert_eq!(view(&state,false)["p1"]["engine_options"][0]["kind"],"switch");
    }
    #[test]
    fn recharge_struggle_and_saved_commitment_are_distinct() {
        let mut state=toy(); state.side_one.volatile_statuses.insert(PokemonVolatileStatus::MUSTRECHARGE);
        assert_eq!(view(&state,false)["p1"]["engine_options"][0]["kind"],"none");
        state.side_one.volatile_statuses.remove(&PokemonVolatileStatus::MUSTRECHARGE);
        state.side_one.get_active().moves.m0.pp=0; state.side_one.get_active().moves.m1.pp=0;
        assert_eq!(view(&state,false)["p1"]["engine_options"][0]["kind"],"struggle");
        state.side_one.force_switch=true;
        state.side_two.switch_out_move_second_saved_move=state.side_two.get_active_immutable().moves.m0.id;
        let report=view(&state,false);
        assert_eq!(report["p2"]["engine_options"][0]["kind"],"move");
        assert_eq!(report["showdown_requests_inferred"],false);
    }
    #[test]
    fn simultaneous_replacement_and_fainted_active() {
        let mut state=toy();
        for side in [&mut state.side_one,&mut state.side_two] {
            side.pokemon[PokemonIndex::P1]=side.get_active_immutable().clone();
            side.get_active().hp=0; side.force_switch=true;
        }
        let report=view(&state,false);
        assert_eq!(report["terminal"],false);
        for seat in ["p1","p2"] {
            assert_eq!(report[seat]["fainted"],true);
            assert_eq!(report[seat]["engine_options"],json!([{"kind":"switch","slot":1,
                "engine_id":if seat=="p1" {"charmander"} else {"squirtle"}}]));
        }
    }
    #[test]
    fn terminal_is_not_residual_options_or_independent_double_ko_label() {
        let mut state=toy(); state.side_one.get_active().hp=0; state.side_two.get_active().hp=0;
        let report=view(&state,false);
        assert_eq!(report["native_outcome"],-1.0); // Native precedence, NOT Showdown tie inference.
        assert_eq!(report["terminal"],true);
        assert!(!report["p1"]["engine_options"].as_array().unwrap().is_empty());
        assert_eq!(report["p1"]["effective_legal_choices"],json!([]));
        assert_eq!(report["branch_transition_validated"],false);
    }
    #[test]
    fn empty_team_preview_root_is_rejected_not_interior_fallback() {
        let mut state=toy(); state.team_preview=true; state.side_one.get_active().hp=0;
        let report=view(&state,true);
        assert_eq!(report["p1"]["engine_options"],json!([]));
        assert_eq!(report["p1"]["search_arms"],json!([]));
        assert_eq!(report["root_search_admissible"],false);
        assert_eq!(report["p1"]["interior_empty_fallback"],false);
        // Exercise fallback law independently of the engine's current None guards.
        let side=side_projection(&state.side_one,&[],false,false);
        assert_eq!(side["engine_options"],json!([]));
        assert_eq!(side["search_arms"][0]["kind"],"none");
        assert_eq!(side["effective_legal_choices"],json!([]));
    }
    #[test]
    fn reader_fixture() {
        let state=toy(); println!("NATIVE_BOUNDARY_FIXTURE:{}",native_scoped_boundary(TOY.trim(),true).unwrap());
        assert_eq!(state.serialize(),toy().serialize());
        let mut preview=state.clone(); preview.team_preview=true;
        println!("NATIVE_PREVIEW_FIXTURE:{}",native_scoped_boundary(&preview.serialize(),true).unwrap());
        println!("NATIVE_PREVIEW_STATE:{}",preview.serialize());
        assert_eq!(view(&preview,true)["p1"]["engine_options"][0]["slot"],0);
    }
}
