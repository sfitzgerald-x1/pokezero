//! A sampled seat's PRIVATE request. The core accepts only that seat's Side;
//! it cannot inspect the other seat's unrevealed ability, item, moves or stats.
//! Native legal options are checked against this own-information surface, not
//! used to disclose hidden trapping abilities through the policy's legal mask.

use std::collections::{HashMap, HashSet};

use poke_engine::choices::Choices;
use poke_engine::engine::state::{MoveChoice, PokemonVolatileStatus};
use poke_engine::state::{Pokemon, PokemonGender, PokemonStatus, Side};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::{json, Value};

fn refusal(message: &str) -> PyErr {
    PyValueError::new_err(format!("policy opponent request: {message}"))
}

fn key(name: &str) -> String {
    name.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

pub(crate) fn move_names(choice: Choices) -> PyResult<(String, String, String)> {
    let engine = format!("{choice:?}").to_lowercase();
    if let Some(suffix) = engine.strip_prefix("hiddenpower") {
        let kind = suffix.trim_end_matches(|c: char| c.is_ascii_digit());
        if ![
            "fighting", "flying", "poison", "ground", "rock", "bug", "ghost", "steel", "fire",
            "water", "grass", "electric", "psychic", "ice", "dragon", "dark",
        ]
        .contains(&kind)
        {
            return Err(refusal("unknown sampled Hidden Power type"));
        }
        return Ok((
            "hiddenpower".into(),
            format!("Hidden Power {kind}"),
            format!("hiddenpower{kind}"),
        ));
    }
    Ok((engine.clone(), engine.clone(), engine))
}

fn condition(mon: &Pokemon) -> String {
    if mon.hp <= 0 {
        return "0 fnt".into();
    }
    let status = match mon.status {
        PokemonStatus::NONE => "",
        PokemonStatus::BURN => " brn",
        PokemonStatus::FREEZE => " frz",
        PokemonStatus::PARALYZE => " par",
        PokemonStatus::POISON => " psn",
        PokemonStatus::TOXIC => " tox",
        PokemonStatus::SLEEP => " slp",
    };
    format!("{}/{}{}", mon.hp, mon.maxhp, status)
}

/// Return a request, complete own per-mon PP state and native→policy map.
/// Request order is explicit and evolved by the caller's public switch fold.
/// The max-PP table must be the registered engine/dex table, not guessed PP.
pub(crate) fn sampled_side_request(
    side: &Side,
    slot: &str,
    species: &[String],
    order: &[String],
    options: &[MoveChoice],
    max_pp: &HashMap<String, i64>,
    root: bool,
) -> PyResult<Value> {
    sampled_side_request_in_phase(side, slot, species, order, options, max_pp, root, false)
}

/// The replacement phase is public request-boundary information, not an
/// opposing ability/item/legal mask. A saved move resumes after Baton Pass;
/// its owner is NOT requested to choose that move again at the switch boundary.
pub(crate) fn sampled_side_request_in_phase(
    side: &Side,
    slot: &str,
    species: &[String],
    order: &[String],
    options: &[MoveChoice],
    max_pp: &HashMap<String, i64>,
    root: bool,
    opponent_replacing: bool,
) -> PyResult<Value> {
    sampled_side_request_with_pp(
        side,
        slot,
        species,
        order,
        options,
        max_pp,
        root,
        opponent_replacing,
        None,
    )
}

/// Base PP is needed only after Transform: Gen 3 gives copied slots five
/// CURRENT PP, but MAX PP uses the transformer's ORIGINAL slot PP Ups. Empty
/// original slots have no PP Ups. Never infer this from remaining PP.
pub(crate) fn sampled_side_request_with_pp(
    side: &Side,
    slot: &str,
    species: &[String],
    order: &[String],
    options: &[MoveChoice],
    max_pp: &HashMap<String, i64>,
    root: bool,
    opponent_replacing: bool,
    base_pp: Option<&HashMap<String, i64>>,
) -> PyResult<Value> {
    if slot != "p1" && slot != "p2" {
        return Err(refusal("unknown seat"));
    }
    let party: Vec<_> = side
        .pokemon
        .into_iter()
        .enumerate()
        .filter(|(_, mon)| format!("{:?}", mon.id) != "NONE")
        .collect();
    if party.is_empty() || species.len() != party.len() || order.len() != party.len() {
        return Err(refusal("incomplete sampled party/order"));
    }
    if party
        .iter()
        .enumerate()
        .any(|(expected, (actual, _))| expected != *actual)
    {
        return Err(refusal("noncontiguous sampled party"));
    }
    let normalized: Vec<_> = species.iter().map(|s| key(s)).collect();
    let ordered: Vec<_> = order.iter().map(|s| key(s)).collect();
    if normalized.iter().any(String::is_empty)
        || normalized.iter().collect::<HashSet<_>>().len() != party.len()
        || ordered.iter().collect::<HashSet<_>>() != normalized.iter().collect::<HashSet<_>>()
    {
        return Err(refusal("ambiguous sampled party/order identity"));
    }
    let active = side
        .active_index
        .serialize()
        .parse::<usize>()
        .map_err(|_| refusal("invalid active party index"))?;
    if active >= party.len() || ordered[0] != normalized[active] {
        return Err(refusal("request order must identify current active first"));
    }
    // Check display identity against the engine instead of trusting an arbitrary
    // label. Gen 3 Unown cosmetic forms intentionally share the engine's base id.
    // Transform changes the effective native id, NOT the private party's ident
    // or details. Its own saved snapshot certifies that original identity; no
    // opposing Side or target's unrevealed request is needed here.
    for ((physical, mon), name) in party.iter().zip(&normalized) {
        let transformed = mon.pre_transform.is_some();
        if transformed
            != (*physical == active
                && side
                    .volatile_statuses
                    .contains(&PokemonVolatileStatus::TRANSFORMED))
        {
            return Err(refusal("inconsistent own Transform snapshot/volatile"));
        }
        let original_id = mon
            .pre_transform
            .as_ref()
            .map(|snapshot| snapshot.id)
            .unwrap_or(mon.id);
        let engine = key(&format!("{:?}", original_id));
        let cosmetic_unown = engine == "unown"
            && (name == "unown"
                || name == "unownexclamation"
                || name == "unownquestion"
                || (name.len() == 6
                    && name.starts_with("unown")
                    && name.as_bytes()[5].is_ascii_alphabetic()));
        if *name != engine && !cosmetic_unown {
            return Err(refusal("sampled species identity mismatch"));
        }
        if mon.maxhp <= 0 || mon.hp < 0 || mon.hp > mon.maxhp {
            return Err(refusal("invalid sampled HP"));
        }
    }
    if options.is_empty() {
        return Err(refusal("empty native option surface"));
    }
    let force_switch =
        side.force_switch || side.baton_passing || side.get_active_immutable().hp <= 0;
    let recharging = side
        .volatile_statuses
        .contains(&PokemonVolatileStatus::MUSTRECHARGE);
    if opponent_replacing && !force_switch {
        let commitment_matches = options.len() == 1
            && match options[0] {
                MoveChoice::None => true,
                MoveChoice::Move(index) => {
                    side.get_active_immutable().moves[&index].id
                        == side.switch_out_move_second_saved_move
                        && side.switch_out_move_second_saved_move != Choices::NONE
                }
                MoveChoice::Struggle => side.switch_out_move_second_saved_move == Choices::STRUGGLE,
                _ => false,
            };
        if !commitment_matches {
            return Err(refusal("replacement wait does not match saved commitment"));
        }
    }
    let waiting = !force_switch
        && (opponent_replacing
            || (!recharging && options.len() == 1 && matches!(options[0], MoveChoice::None)));
    let mut own_move_options = Vec::new();
    if !force_switch && !recharging && !waiting {
        if let Some(index) = side.active_is_charging_move() {
            own_move_options.push(MoveChoice::Move(index));
        } else {
            side.get_active_immutable().add_available_moves(
                &mut own_move_options,
                &side.last_used_move,
                side.volatile_statuses
                    .contains(&PokemonVolatileStatus::ENCORE),
                side.volatile_statuses
                    .contains(&PokemonVolatileStatus::TAUNT),
            );
        }
    }
    let mut rows = Vec::new();
    let mut move_states = serde_json::Map::new();
    let mut active_moves = Vec::new();
    for name in &ordered {
        let physical = normalized
            .iter()
            .position(|s| s == name)
            .expect("validated order");
        let mon = party[physical].1;
        let mut moves = Vec::new();
        let mut known_moves = Vec::new();
        for (index, mv) in mon.moves.into_iter().enumerate() {
            if mv.id == Choices::NONE {
                moves.push(json!({"id": format!("slot{}", index + 1), "move": format!("slot:{}", index + 1),
                    "pp": 0, "maxpp": 0, "disabled": true}));
                continue;
            }
            let (id, display, known) = move_names(mv.id)?;
            let registered_maximum = *max_pp
                .get(&id)
                .ok_or_else(|| refusal("missing registered move max PP"))?;
            let maximum = if let Some(snapshot) = &mon.pre_transform {
                let base =
                    base_pp.ok_or_else(|| refusal("missing registered Transform base PP"))?;
                let copied_base = *base
                    .get(&id)
                    .filter(|pp| **pp > 0)
                    .ok_or_else(|| refusal("missing copied move base PP"))?;
                if i64::from(mv.pp) > copied_base.min(5) {
                    return Err(refusal("invalid copied move PP"));
                }
                let original = snapshot.moves[index].0;
                let pp_ups = if original == Choices::NONE {
                    false
                } else {
                    let (original_id, _, _) = move_names(original)?;
                    let original_base = *base
                        .get(&original_id)
                        .filter(|pp| **pp > 0)
                        .ok_or_else(|| refusal("missing original move base PP"))?;
                    let original_max = *max_pp
                        .get(&original_id)
                        .ok_or_else(|| refusal("missing original move max PP"))?;
                    if original_max == original_base {
                        false
                    } else if original_max == original_base * 8 / 5 {
                        true
                    } else {
                        return Err(refusal("unsupported original slot PP Ups"));
                    }
                };
                if pp_ups {
                    registered_maximum
                } else {
                    copied_base
                }
            } else {
                registered_maximum
            };
            if maximum <= 0 || i64::from(mv.pp) < 0 || i64::from(mv.pp) > maximum {
                return Err(refusal("invalid sampled move PP"));
            }
            let enabled = physical == active
                && own_move_options.iter().any(|option| {
                    matches!(option,
                MoveChoice::Move(slot) if slot.serialize().parse::<usize>().ok() == Some(index))
                });
            moves.push(
                json!({"id": id, "move": display, "pp": mv.pp, "maxpp": maximum,
                "disabled": if physical == active { !enabled } else { mv.disabled }}),
            );
            known_moves.push(known);
        }
        let states: Vec<_> = moves
            .iter()
            .filter(|m| m["maxpp"].as_i64().unwrap_or(0) > 0)
            .cloned()
            .collect();
        move_states.insert(species[physical].trim().to_lowercase(), json!(states));
        if physical == active {
            active_moves = moves;
        }
        let ability = format!("{:?}", mon.ability).to_lowercase();
        let item = format!("{:?}", mon.item).to_lowercase();
        let gender = match mon.gender {
            PokemonGender::NONE => "",
            PokemonGender::MALE => ", M",
            PokemonGender::FEMALE => ", F",
        };
        rows.push(json!({"ident": format!("{slot}: {}", species[physical]),
            "details": format!("{}, L{}{}", species[physical], mon.level, gender), "active": physical == active,
            "condition": condition(mon), "moves": known_moves,
            "ability": if ability == "none" { "" } else { &ability },
            "item": if item == "none" { "" } else { &item },
            "stats": {"atk": mon.attack, "def": mon.defense, "spa": mon.special_attack,
                "spd": mon.special_defense, "spe": mon.speed}}));
    }
    let mut request = json!({"side": {"id": slot, "pokemon": rows}});
    if waiting {
        request["wait"] = json!(true);
    } else if force_switch {
        request["forceSwitch"] = json!([true]);
    } else {
        // Only known own volatiles contribute. Never ask Side::trapped with
        // the real opposing Pokemon (that would reveal hidden trap abilities).
        let trapped = recharging
            || side.active_is_charging_move().is_some()
            || (root && side.force_trapped)
            || side.trapped(&Pokemon::default());
        if recharging {
            active_moves = vec![json!({"id": "recharge", "move": "recharge", "disabled": false})];
        } else if options.contains(&MoveChoice::Struggle) && !waiting {
            active_moves = vec![json!({"id": "struggle", "move": "Struggle", "target": "randomNormal", "disabled": false})];
        }
        request["active"] = json!([{"moves": active_moves, "trapped": trapped}]);
    }
    let mut map = Vec::with_capacity(options.len());
    let mut native_legal = HashSet::new();
    for option in options {
        let index = match option {
            MoveChoice::Struggle if waiting => None,
            MoveChoice::Struggle => Some(0),
            MoveChoice::Move(_) | MoveChoice::None if waiting => None,
            MoveChoice::Move(index) => Some(
                index
                    .serialize()
                    .parse::<usize>()
                    .map_err(|_| refusal("invalid move slot"))?,
            ),
            MoveChoice::Switch(index) => {
                let physical = index
                    .serialize()
                    .parse::<usize>()
                    .map_err(|_| refusal("invalid switch slot"))?;
                let name = normalized
                    .get(physical)
                    .ok_or_else(|| refusal("unknown switch identity"))?;
                let request_position = ordered
                    .iter()
                    .position(|s| s == name)
                    .expect("validated order");
                if request_position == 0 {
                    return Err(refusal("native switch points to active"));
                }
                Some(4 + request_position - 1)
            }
            MoveChoice::None if recharging && !force_switch => Some(0),
            MoveChoice::None => None,
        };
        if let Some(index) = index {
            if !native_legal.insert(index) {
                return Err(refusal("duplicate native policy action"));
            }
        }
        map.push(index);
    }
    let mut private_legal = HashSet::new();
    if let Some(moves) = request.pointer("/active/0/moves").and_then(Value::as_array) {
        for (index, mv) in moves.iter().enumerate() {
            if mv["disabled"] == false {
                private_legal.insert(index);
            }
        }
    }
    if force_switch || (!waiting && request.pointer("/active/0/trapped") == Some(&json!(false))) {
        for (position, name) in ordered.iter().enumerate().skip(1) {
            let physical = normalized
                .iter()
                .position(|s| s == name)
                .expect("validated order");
            if party[physical].1.hp > 0 {
                private_legal.insert(4 + position - 1);
            }
        }
    }
    if native_legal != private_legal {
        // A refusal is not a policy input. Retain the two already-derived
        // action surfaces for diagnosis without sending opposing hidden state
        // to the provider or silently changing either legal mask.
        let mut native_labels: Vec<_> = native_legal.iter().copied().collect();
        let mut private_labels: Vec<_> = private_legal.iter().copied().collect();
        native_labels.sort_unstable();
        private_labels.sort_unstable();
        let diagnostic = json!({"schema": "policy-opponent-refusal-v1",
            "kind": "legal_surface_mismatch", "seat": slot, "root": root,
            "native_action_indices": native_labels, "private_action_indices": private_labels,
            "native_options": options.iter().map(|option| format!("{option:?}")).collect::<Vec<_>>(),
            "request_order": ordered, "active_index": active,
            "force_switch": force_switch, "waiting": waiting, "recharging": recharging,
            "own_volatile_statuses": format!("{:?}", side.volatile_statuses),
            "own_last_used_move": format!("{:?}", side.last_used_move),
            "diagnostic_only_not_policy_input": true});
        let error = refusal("native and private-knowledge legal surfaces differ");
        Python::attach(|py| {
            error
                .value(py)
                .setattr("policy_opponent_diagnostic", diagnostic.to_string())
        })?;
        return Err(error);
    }
    Ok(json!({"request": request, "self_move_states": move_states, "native_action_indices": map}))
}

/// Diagnostic bridge. Production uses the same Side-only constructor in the
/// reached-node provider; full state/other-side context never enters Python.
#[pyfunction]
#[pyo3(signature = (state_str, slot, species, request_order, max_pp, root = true, base_pp = None))]
pub fn sampled_policy_request(
    state_str: &str,
    slot: &str,
    species: Vec<String>,
    request_order: Vec<String>,
    max_pp: HashMap<String, i64>,
    root: bool,
    base_pp: Option<HashMap<String, i64>>,
) -> PyResult<String> {
    let state = crate::parse_state(state_str)?;
    let (one, two) = if root {
        state.root_get_all_options()
    } else {
        state.get_all_options()
    };
    let (side, options, opponent_replacing) = match slot {
        "p1" => (&state.side_one, one, state.side_two.force_switch),
        "p2" => (&state.side_two, two, state.side_one.force_switch),
        _ => return Err(refusal("unknown seat")),
    };
    sampled_side_request_with_pp(
        side,
        slot,
        &species,
        &request_order,
        &options,
        &max_pp,
        root,
        opponent_replacing,
        base_pp.as_ref(),
    )
    .map(|v| v.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use poke_engine::engine::abilities::Abilities;
    use poke_engine::state::{PokemonIndex, PokemonMoveIndex, PreTransform, State};
    use std::str::FromStr;

    fn fixture() -> State {
        let mut state =
            crate::parse_state(include_str!("test_fixtures/minimal.state").trim()).unwrap();
        state
            .side_one
            .get_active()
            .replace_move(PokemonMoveIndex::M1, Choices::NONE);
        let active = state.side_one.get_active().clone();
        for (position, id) in [(1, "BULBASAUR"), (2, "SQUIRTLE")] {
            state.side_one.pokemon.pkmn[position] = active.clone();
            state.side_one.pokemon.pkmn[position].id = FromStr::from_str(id).unwrap();
        }
        state
    }

    fn names() -> Vec<String> {
        ["Charmander", "Bulbasaur", "Squirtle"]
            .map(String::from)
            .to_vec()
    }
    fn pp() -> HashMap<String, i64> {
        [("ember".into(), 40), ("watergun".into(), 40)].into()
    }

    #[test]
    fn synthetic_struggle_request_keeps_private_pp_banks_and_live_switches() {
        let mut state = fixture();
        state.side_one.get_active().moves.m0.pp = 0;
        let options = state.root_get_all_options().0;
        assert!(options.contains(&MoveChoice::Struggle));
        let result = sampled_side_request(
            &state.side_one, "p1", &names(), &names(), &options, &pp(), true,
        ).unwrap();
        assert_eq!(result["request"]["active"][0]["moves"], json!([
            {"id": "struggle", "move": "Struggle", "target": "randomNormal", "disabled": false}
        ]));
        let indices = result["native_action_indices"].as_array().unwrap();
        assert!(indices.contains(&json!(0)));
        assert!(indices.iter().any(|i| i.as_u64().is_some_and(|i| i >= 4)));
        assert!(indices.iter().all(|i| !i.is_null()));
        assert_eq!(result["self_move_states"]["charmander"][0]["id"], "ember");
        assert_eq!(result["self_move_states"]["charmander"][0]["pp"], 0);
    }
    fn bundle(state: &State, order: &[String]) -> PyResult<Value> {
        sampled_side_request(
            &state.side_one,
            "p1",
            &names(),
            order,
            &state.root_get_all_options().0,
            &pp(),
            true,
        )
    }

    #[test]
    fn own_private_party_and_complete_bench_pp_are_preserved() {
        let mut state = fixture();
        state.side_one.get_active().moves.m0.pp = 13;
        state.side_one.pokemon.pkmn[1].moves.m0.pp = 7;
        let result = bundle(&state, &names()).unwrap();
        assert_eq!(result["request"]["side"]["id"], "p1");
        assert_eq!(
            result["request"]["side"]["pokemon"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
        assert_eq!(result["request"]["active"][0]["moves"][0]["pp"], 13);
        assert_eq!(result["self_move_states"]["bulbasaur"][0]["pp"], 7);
        assert_eq!(result["native_action_indices"], json!([0, 4, 5]));
    }

    #[test]
    fn known_gender_and_status_are_not_dropped() {
        let mut state = fixture();
        state.side_one.get_active().gender = PokemonGender::FEMALE;
        state.side_one.get_active().status = PokemonStatus::TOXIC;
        let result = bundle(&state, &names()).unwrap();
        assert_eq!(
            result["request"]["side"]["pokemon"][0]["details"],
            "Charmander, L100, F"
        );
        assert_eq!(
            result["request"]["side"]["pokemon"][0]["condition"],
            "100/100 tox"
        );
    }

    #[test]
    fn changing_unrevealed_other_side_stats_moves_item_and_team_is_invisible() {
        let state = fixture();
        let expected = bundle(&state, &names()).unwrap();
        let mut other = state.clone();
        other.side_two.get_active().attack += 123;
        other.side_two.get_active().maxhp += 200;
        other.side_two.get_active().hp += 200;
        other
            .side_two
            .get_active()
            .replace_move(PokemonMoveIndex::M0, Choices::THUNDERBOLT);
        other.side_two.get_active().item = FromStr::from_str("CHOICEBAND").unwrap();
        other.side_two.pokemon.pkmn[1] = other.side_two.get_active().clone();
        assert_eq!(bundle(&other, &names()).unwrap(), expected);
    }

    #[test]
    fn hidden_trapping_ability_cannot_be_encoded_as_private_knowledge() {
        pyo3::Python::initialize();
        let mut state = fixture();
        assert!(bundle(&state, &names()).is_ok());
        state.side_two.get_active().ability = Abilities::SHADOWTAG;
        let error = bundle(&state, &names())
            .expect_err("native trap masking must refuse rather than disclose truth");
        pyo3::Python::attach(|py| {
            assert!(error
                .value(py)
                .to_string()
                .contains("legal surfaces differ"))
        });
    }

    #[test]
    fn sparse_move_slots_do_not_compact() {
        let mut state = fixture();
        state
            .side_one
            .get_active()
            .replace_move(PokemonMoveIndex::M0, Choices::NONE);
        state
            .side_one
            .get_active()
            .replace_move(PokemonMoveIndex::M2, Choices::EMBER);
        state.side_one.get_active().moves.m2.disabled = false;
        state.side_one.get_active().moves.m2.pp = 32;
        let result = bundle(&state, &names()).unwrap();
        assert_eq!(result["native_action_indices"], json!([2, 4, 5]));
        assert_eq!(result["request"]["active"][0]["moves"][0]["disabled"], true);
        assert_eq!(result["request"]["active"][0]["moves"][2]["id"], "ember");
    }

    #[test]
    fn evolved_request_order_maps_physical_switches_to_actual_targets() {
        let mut state = fixture();
        state.side_one.active_index = PokemonIndex::P2;
        let order = ["Squirtle", "Bulbasaur", "Charmander"].map(String::from);
        let result = bundle(&state, &order).unwrap();
        assert_eq!(result["native_action_indices"], json!([0, 5, 4]));
        assert_eq!(
            result["request"]["side"]["pokemon"][0]["ident"],
            "p1: Squirtle"
        );
    }

    #[test]
    fn recharge_and_forced_replacement_have_distinct_masks() {
        let mut state = fixture();
        state
            .side_one
            .volatile_statuses
            .insert(PokemonVolatileStatus::MUSTRECHARGE);
        let recharge = bundle(&state, &names()).unwrap();
        assert_eq!(recharge["native_action_indices"], json!([0]));
        assert_eq!(
            recharge["request"]["active"][0]["moves"][0]["id"],
            "recharge"
        );
        assert_eq!(recharge["request"]["active"][0]["trapped"], true);
        state.side_one.get_active().hp = 0;
        state.side_one.force_switch = true;
        let forced = bundle(&state, &names()).unwrap();
        assert_eq!(forced["native_action_indices"], json!([4, 5]));
        assert_eq!(forced["request"]["forceSwitch"], json!([true]));
        assert!(forced["request"].get("active").is_none());
    }

    #[test]
    fn public_trap_and_self_charge_are_known_without_reading_other_side() {
        let mut state = fixture();
        state
            .side_one
            .volatile_statuses
            .insert(PokemonVolatileStatus::TRAPPED);
        assert_eq!(
            bundle(&state, &names()).unwrap()["native_action_indices"],
            json!([0])
        );
        state
            .side_one
            .volatile_statuses
            .remove(&PokemonVolatileStatus::TRAPPED);
        state
            .side_one
            .get_active()
            .replace_move(PokemonMoveIndex::M1, Choices::FLY);
        state
            .side_one
            .volatile_statuses
            .insert(PokemonVolatileStatus::FLY);
        let mut max_pp = pp();
        max_pp.insert("fly".into(), 24);
        state.side_one.get_active().moves.m1.pp = 23;
        let result = sampled_side_request(
            &state.side_one,
            "p1",
            &names(),
            &names(),
            &state.root_get_all_options().0,
            &max_pp,
            true,
        )
        .unwrap();
        assert_eq!(result["native_action_indices"], json!([1]));
    }

    #[test]
    fn replacement_phase_waits_for_a_saved_move_without_resampling_it() {
        let mut state = fixture();
        state.side_one.switch_out_move_second_saved_move = Choices::EMBER;
        let options = [MoveChoice::Move(PokemonMoveIndex::M0)];
        let result = sampled_side_request_in_phase(
            &state.side_one,
            "p1",
            &names(),
            &names(),
            &options,
            &pp(),
            false,
            true,
        )
        .unwrap();
        assert_eq!(result["request"]["wait"], true);
        assert_eq!(result["native_action_indices"], json!([null]));
        assert!(result["request"].get("active").is_none());
    }

    #[test]
    fn replacement_phase_does_not_accept_an_uncommitted_move() {
        pyo3::Python::initialize();
        let state = fixture();
        assert!(sampled_side_request_in_phase(
            &state.side_one,
            "p1",
            &names(),
            &names(),
            &[MoveChoice::Move(PokemonMoveIndex::M0)],
            &pp(),
            false,
            true,
        )
        .is_err());
        let wait = sampled_side_request_in_phase(
            &state.side_one,
            "p1",
            &names(),
            &names(),
            &[MoveChoice::None],
            &pp(),
            false,
            true,
        )
        .unwrap();
        assert_eq!(wait["native_action_indices"], json!([null]));
        let mut forced = state;
        forced.side_one.force_switch = true;
        let replacing = sampled_side_request_in_phase(
            &forced.side_one,
            "p1",
            &names(),
            &names(),
            &[
                MoveChoice::Switch(PokemonIndex::P1),
                MoveChoice::Switch(PokemonIndex::P2),
            ],
            &pp(),
            false,
            true,
        )
        .unwrap();
        assert_eq!(replacing["native_action_indices"], json!([4, 5]));
        assert_eq!(replacing["request"]["forceSwitch"], json!([true]));
    }

    #[test]
    fn root_only_trapping_latch_does_not_leak_into_reached_children() {
        let mut state = fixture();
        state.side_one.force_trapped = true;
        assert_eq!(
            bundle(&state, &names()).unwrap()["native_action_indices"],
            json!([0])
        );
        let child = sampled_side_request(
            &state.side_one,
            "p1",
            &names(),
            &names(),
            &state.get_all_options().0,
            &pp(),
            false,
        )
        .unwrap();
        assert_eq!(child["native_action_indices"], json!([0, 4, 5]));
        assert_eq!(child["request"]["active"][0]["trapped"], false);
    }

    #[test]
    fn typed_hidden_power_private_identity_is_retained_with_base_action_id() {
        let state = fixture();
        let choice = FromStr::from_str("HIDDENPOWERWATER70").unwrap();
        let mut sampled = state.side_one.clone();
        sampled
            .get_active()
            .replace_move(PokemonMoveIndex::M0, choice);
        sampled.get_active().moves.m0.pp = 15;
        let maximum = [("hiddenpower".into(), 24), ("ember".into(), 40)].into();
        let result = sampled_side_request(
            &sampled,
            "p1",
            &names(),
            &names(),
            &[
                MoveChoice::Move(PokemonMoveIndex::M0),
                MoveChoice::Switch(PokemonIndex::P1),
                MoveChoice::Switch(PokemonIndex::P2),
            ],
            &maximum,
            true,
        )
        .unwrap();
        assert_eq!(
            result["request"]["active"][0]["moves"][0]["id"],
            "hiddenpower"
        );
        assert_eq!(
            result["request"]["active"][0]["moves"][0]["move"],
            "Hidden Power water"
        );
        assert_eq!(
            result["request"]["side"]["pokemon"][0]["moves"],
            json!(["hiddenpowerwater"])
        );
    }

    #[test]
    fn incomplete_order_pp_and_inconsistent_transform_refuse() {
        pyo3::Python::initialize();
        let mut state = fixture();
        assert!(bundle(&state, &names()[..2]).is_err());
        assert!(bundle(
            &state,
            &["Charmander".into(), "Bulbasaur".into(), "Bulbasaur".into()]
        )
        .is_err());
        assert!(sampled_side_request(
            &state.side_one,
            "p1",
            &names(),
            &names(),
            &state.root_get_all_options().0,
            &HashMap::new(),
            true
        )
        .is_err());
        let before_transform = PreTransform::capture(state.side_one.get_active_immutable());
        state.side_one.get_active().pre_transform = Some(Box::new(before_transform));
        assert!(bundle(&state, &names()).is_err());
    }

    #[test]
    fn transformed_request_keeps_original_party_identity_and_original_slot_pp_ups() {
        let mut state = fixture();
        let snapshot = PreTransform::capture(state.side_one.get_active_immutable());
        let original_hp = state.side_one.get_active_immutable().hp;
        state.side_one.get_active().pre_transform = Some(Box::new(snapshot));
        state
            .side_one
            .volatile_statuses
            .insert(PokemonVolatileStatus::TRANSFORMED);
        state.side_one.get_active().id = FromStr::from_str("SQUIRTLE").unwrap();
        state
            .side_one
            .get_active()
            .replace_move(PokemonMoveIndex::M0, Choices::WATERGUN);
        state.side_one.get_active().moves.m0.pp = 3;
        // The original second slot is empty: its copied move starts at five
        // current PP but receives no PP Ups in Gen 3's MAX-PP field.
        state
            .side_one
            .get_active()
            .replace_move(PokemonMoveIndex::M1, Choices::EMBER);
        state.side_one.get_active().moves.m1.pp = 5;
        state.side_one.get_active().attack = 87;
        let base_pp = HashMap::from([("ember".into(), 25), ("watergun".into(), 25)]);
        let value = sampled_side_request_with_pp(
            &state.side_one,
            "p1",
            &names(),
            &names(),
            &state.root_get_all_options().0,
            &pp(),
            true,
            false,
            Some(&base_pp),
        )
        .unwrap();
        let row = &value["request"]["side"]["pokemon"][0];
        assert_eq!(row["ident"], "p1: Charmander");
        assert!(row["details"]
            .as_str()
            .unwrap()
            .starts_with("Charmander, L"));
        assert!(row["condition"]
            .as_str()
            .unwrap()
            .starts_with(&format!("{original_hp}/")));
        assert_eq!(row["moves"][0], "watergun");
        assert_eq!(row["stats"]["atk"], 87);
        assert_eq!(value["request"]["active"][0]["moves"][0]["pp"], 3);
        assert_eq!(value["request"]["active"][0]["moves"][0]["maxpp"], 40);
        assert_eq!(value["request"]["active"][0]["moves"][1]["maxpp"], 25);
        assert_eq!(value["self_move_states"]["charmander"][0]["maxpp"], 40);
        assert_eq!(value["self_move_states"]["bulbasaur"][0]["maxpp"], 40);
        // A copied bank cannot claim boosted/original PP, or relabel the
        // private party as the Transform target (already present on this team).
        Python::initialize();
        state.side_one.get_active().moves.m0.pp = 6;
        let error = sampled_side_request_with_pp(
            &state.side_one,
            "p1",
            &names(),
            &names(),
            &state.root_get_all_options().0,
            &pp(),
            true,
            false,
            Some(&base_pp),
        )
        .unwrap_err();
        assert!(error.to_string().contains("invalid copied move PP"));
        state.side_one.get_active().moves.m0.pp = 3;
        let mut wrong = names();
        wrong[0] = "Squirtle".into();
        assert!(sampled_side_request(
            &state.side_one,
            "p1",
            &wrong,
            &wrong,
            &state.root_get_all_options().0,
            &pp(),
            true
        )
        .is_err());
    }
}
