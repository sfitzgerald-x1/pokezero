//! SERVER-private observation, not public ability knowledge or a new policy.
//! The only opposing sampled fact exported is stock Showdown's maybeTrapped
//! bit. No opposing ability label, team, moves, item or stats enter Python.
//! Source: pinned f76228a sim/battle.ts request loop, pokemon.ts getMoveRequestData,
//! data/abilities.ts Arena Trap and data/mods/gen3/abilities.ts Magnet Pull.

use poke_engine::engine::abilities::Abilities;
use poke_engine::engine::state::PokemonVolatileStatus;
use poke_engine::state::{Pokemon, PokemonType, Side};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub(crate) struct MaybeTrap {
    // Private constructor: callers cannot turn an arbitrary mask mismatch into
    // an observer certificate. Every invocation rechecks the complete history.
    slot: String,
    binding: Value,
    own_state: String,
    own_id: String,
    own_ability: String,
}

fn refusal(message: &str) -> PyErr {
    PyValueError::new_err(format!("policy opponent request observer: {message}"))
}

fn id(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn digest(value: &Value, domain: &[u8]) -> String {
    // serde_json's default Map is sorted; compact UTF-8 JSON matches Python's
    // ensure_ascii=False/sort_keys=True contract for these integer/string rows.
    let mut hash = Sha256::new();
    hash.update(domain);
    hash.update([0]);
    hash.update(serde_json::to_vec(value).expect("finite JSON"));
    format!("{:x}", hash.finalize())
}

pub(crate) fn public_binding(lines: &[String]) -> Value {
    let mut entries = [0usize; 2];
    let mut actives = [String::new(), String::new()];
    let mut turn = 0usize;
    for line in lines {
        let f: Vec<_> = line.split('|').collect();
        if f.len() > 3 && matches!(f[1], "switch" | "drag") {
            let seat = if f[2].starts_with("p1") {
                Some(0)
            } else if f[2].starts_with("p2") {
                Some(1)
            } else {
                None
            };
            if let Some(seat) = seat {
                entries[seat] += 1;
                actives[seat] = f[2].into();
            }
        }
        if f.len() > 2 && f[1] == "turn" {
            turn = f[2].parse().unwrap_or(0);
        }
    }
    json!({"sha256": digest(&json!(lines), b"pokezero.gen3-maybe-trap-public-history.v1"),
        "public_line_count": lines.len(), "turn": turn,
        "entries": {"p1": entries[0], "p2": entries[1]},
        "actives": {"p1": actives[0], "p2": actives[1]}})
}

/// Normal Gen3 singles only. Arena Trap is speculative over the species'
/// normal abilities even in a Sand Veil world. Gen3 Magnet Pull has no
/// FoeMaybeTrapPokemon handler: its actual AnyMaybeTrapPokemon handler supplies
/// the bit only in a Magnet Pull world. Treating both as species-uniform leaks
/// a false observation and changes the frozen checkpoint's action support.
pub(crate) fn observe(
    own: &Side,
    foe: &Side,
    slot: &str,
    lines: &[String],
    format: Option<&str>,
    root: bool,
) -> PyResult<Option<MaybeTrap>> {
    let current = own.get_active_immutable();
    let holder = foe.get_active_immutable();
    let active_index = own
        .active_index
        .serialize()
        .parse::<usize>()
        .expect("native active index");
    if !own
        .pokemon
        .into_iter()
        .enumerate()
        .any(|(index, mon)| index != active_index && mon.hp > 0)
    {
        return Ok(None); // getMoveRequestData: no canSwitchIn, no maybeTrapped
    }
    if own.force_switch
        || own.baton_passing
        || foe.force_switch
        || current.hp <= 0
        || holder.hp <= 0
    {
        return Ok(None); // forced/WAIT boundaries retain the established constructor
    }
    if (root && own.force_trapped)
        || own.trapped(&Pokemon::default())
        || own.active_is_charging_move().is_some()
        || own
            .volatile_statuses
            .contains(&PokemonVolatileStatus::MUSTRECHARGE)
    {
        return Ok(None); // getMoveRequestData hardLocked clears maybeTrapped
    }
    let foe_id = id(&format!("{:?}", holder.id));
    let arena_species = matches!(foe_id.as_str(), "diglett" | "dugtrio");
    let magnet_species = matches!(foe_id.as_str(), "magnemite" | "magneton" | "nosepass");
    let actual_trap = matches!(holder.ability, Abilities::ARENATRAP | Abilities::MAGNETPULL);
    if !(arena_species || magnet_species || actual_trap) {
        return Ok(None);
    }
    // Transform makes Showdown's knownType false. Its speculative trapping
    // predicates may then differ even for native Flying/non-Steel types, so
    // reject this unsupported context BEFORE deciding that no flag is needed.
    if current.pre_transform.is_some() || holder.pre_transform.is_some() {
        return Err(refusal("unsupported transformed trapping context"));
    }
    let maybe = (arena_species && current.is_grounded())
        || (holder.ability == Abilities::ARENATRAP && current.is_grounded())
        || (holder.ability == Abilities::MAGNETPULL && current.has_type(&PokemonType::STEEL));
    if !maybe {
        return Ok(None);
    } // no observation extension; do not poison ordinary choices
      // A missing context is NOT permission to synthesize the private bit. The
      // legacy Side-only path remains strict and will diagnose native mismatch.
    if format.is_none() && lines.is_empty() {
        return Ok(None);
    }
    if format != Some("gen3randombattle") || !matches!(slot, "p1" | "p2") {
        return Err(refusal("requires normal Gen3 singles provenance"));
    }
    if (arena_species && !matches!(holder.ability, Abilities::ARENATRAP | Abilities::SANDVEIL))
        || (magnet_species && !matches!(holder.ability, Abilities::MAGNETPULL | Abilities::STURDY))
        || (!arena_species && !magnet_species)
    {
        return Err(refusal("unsupported changed trapping ability"));
    }
    let own_seat = usize::from(slot == "p2");
    let mut active = [String::new(), String::new()];
    let mut species = [String::new(), String::new()];
    let mut alive = [false; 2];
    let mut tainted = [false; 2];
    let (mut gen3, mut singles, mut tier, mut started) = (false, false, false, false);
    let mut turn = 0u32;
    for line in lines {
        let f: Vec<_> = line.split('|').collect();
        let field = |n| f.get(n).copied().unwrap_or("");
        let event = field(1);
        if !line.starts_with('|') || matches!(event, "request" | "split" | "win" | "tie") {
            return Err(refusal("nonpublic or terminal history"));
        }
        match event {
            "gen" if !started => gen3 = field(2) == "3",
            "gametype" if !started => singles = field(2) == "singles",
            "tier" if !started => tier = id(field(2)) == "gen3randombattle",
            "gen" | "gametype" | "tier" => return Err(refusal("late format metadata")),
            "start" => {
                if started || !(gen3 && singles && tier) {
                    return Err(refusal("incomplete battle start"));
                }
                started = true;
            }
            _ => {}
        }
        if !started {
            continue;
        }
        let seat = if field(2).starts_with("p1a: ") {
            Some(0)
        } else if field(2).starts_with("p2a: ") {
            Some(1)
        } else {
            None
        };
        match event {
            "turn" => {
                let next = field(2)
                    .parse::<u32>()
                    .map_err(|_| refusal("invalid turn"))?;
                if next != turn + 1 {
                    return Err(refusal("noncontiguous turn history"));
                }
                turn = next;
            }
            "switch" | "drag" => {
                let seat = seat.ok_or_else(|| refusal("unknown active entry"))?;
                active[seat] = field(2).into();
                tainted[seat] = false;
                species[seat] = id(field(3).split(',').next().unwrap_or(""));
                alive[seat] = field(4)
                    .split('/')
                    .next()
                    .and_then(|s| s.parse::<u32>().ok())
                    .is_some_and(|hp| hp > 0);
            }
            "replace" | "detailschange" | "-formechange" | "-transform" | "-endability" => {
                tainted[seat.ok_or_else(|| refusal("unknown identity/ability owner"))?] = true;
            }
            "-ability" => {
                let seat = seat.ok_or_else(|| refusal("unknown ability owner"))?;
                if active[seat] != field(2) {
                    return Err(refusal("stale ability owner"));
                }
                let expected = if seat == own_seat {
                    current.ability
                } else {
                    holder.ability
                };
                // Ordinary current ability reveals are not changes. Past entry
                // changes cannot contaminate a later switch-in or a sibling.
                if id(field(3)) != id(&format!("{expected:?}")) || line.contains("[from]") {
                    tainted[seat] = true;
                }
            }
            "move"
                if matches!(
                    id(field(3)).as_str(),
                    "skillswap"
                        | "roleplay"
                        | "transform"
                        | "conversion"
                        | "conversion2"
                        | "camouflage"
                ) =>
            {
                if id(field(3)) == "skillswap" {
                    tainted = [true; 2];
                } else {
                    tainted[seat.ok_or_else(|| refusal("unknown ability/type actor"))?] = true;
                }
            }
            "-start" if matches!(id(field(3)).as_str(), "typechange" | "typeadd") => {
                tainted[seat.ok_or_else(|| refusal("unknown type owner"))?] = true;
            }
            "-activate"
                if line.contains("ability:")
                    || line.contains("Skill Swap")
                    || line.contains("Role Play") =>
            {
                tainted = [true; 2];
            }
            "faint" | "-damage" | "-sethp" => {
                let seat = seat.ok_or_else(|| refusal("unknown HP owner"))?;
                if active[seat] != field(2) {
                    return Err(refusal("stale active HP owner"));
                }
                if event == "faint" || field(3).starts_with("0 ") {
                    alive[seat] = false;
                }
                if event == "-sethp" && f.len() > 4 && !field(4).starts_with('[') {
                    return Err(refusal("unsupported multi-owner HP update"));
                }
            }
            _ => {}
        }
    }
    if !started
        || turn == 0
        || !alive.iter().all(|b| *b)
        || species[own_seat] != id(&format!("{:?}", current.id))
        || species[1 - own_seat] != foe_id
        || current.hp <= 0
        || holder.hp <= 0
        || tainted.iter().any(|b| *b)
    {
        return Err(refusal("public/node active identity or life mismatch"));
    }
    Ok(Some(MaybeTrap {
        slot: slot.into(),
        binding: public_binding(lines),
        own_state: own.serialize(),
        own_id: id(&format!("{:?}", current.id)),
        own_ability: id(&format!("{:?}", current.ability)),
    }))
}

impl MaybeTrap {
    /// Exactly the frozen policy's moves, or the complete protocol surface.
    /// A native subset, a new move or only some live switches remains refused.
    pub(crate) fn certify(&self, own: &Side, request: &Value, native: &[usize]) -> PyResult<Value> {
        let moves = request
            .pointer("/active/0/moves")
            .and_then(Value::as_array)
            .ok_or_else(|| refusal("missing observer moves"))?;
        let policy: Vec<_> = moves
            .iter()
            .enumerate()
            .filter(|(_, mv)| mv["disabled"] == false)
            .map(|(index, _)| index)
            .collect();
        if moves.is_empty() || moves.len() > 4 || policy.is_empty() {
            return Err(refusal("invalid observer moves"));
        }
        let rows = request
            .pointer("/side/pokemon")
            .and_then(Value::as_array)
            .ok_or_else(|| refusal("missing observer party"))?;
        let mut protocol = policy.clone();
        for (index, mon) in rows.iter().enumerate().skip(1) {
            if mon["condition"]
                .as_str()
                .unwrap_or("")
                .split('/')
                .next()
                .and_then(|hp| hp.parse::<u32>().ok())
                .is_some_and(|hp| hp > 0)
            {
                protocol.push(4 + index - 1);
            }
        }
        if protocol.len() == policy.len() {
            return Err(refusal("observer requires a living bench"));
        }
        let mut sorted = native.to_vec();
        sorted.sort_unstable();
        if sorted != policy && sorted != protocol {
            return Err(refusal(
                "unexplained native/protocol/policy action difference",
            ));
        }
        self.receipt(own, request, &protocol, &policy)
    }

    fn receipt(
        &self,
        own: &Side,
        request: &Value,
        protocol: &[usize],
        policy: &[usize],
    ) -> PyResult<Value> {
        if request.pointer("/side/id").and_then(Value::as_str) != Some(self.slot.as_str()) {
            return Err(refusal("own request seat mismatch"));
        }
        let row = request
            .pointer("/side/pokemon/0")
            .ok_or_else(|| refusal("missing observed own active"))?;
        let own_ident = row["ident"].as_str().unwrap_or("");
        if own.serialize() != self.own_state
            || own_ident.split_once(": ").map(|(_, name)| id(name)) != Some(self.own_id.clone())
            || id(row["ability"].as_str().unwrap_or("")) != self.own_ability
            || row["active"] != true
            || request.pointer("/active/0/maybeTrapped") != Some(&json!(true))
            || request.pointer("/active/0/trapped") == Some(&json!(true))
        {
            return Err(refusal("observed own state/request eligibility changed"));
        }
        Ok(
            json!({"schema": "pokezero.gen3-maybe-trap-request.v1", "seat": self.slot,
            "format_id": "gen3randombattle", "public_binding": self.binding,
            "request_sha256": digest(request, b"pokezero.gen3-maybe-trap-own-request.v1"),
            "protocol_action_indices": protocol, "policy_action_indices": policy}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use poke_engine::engine::state::MoveChoice;
    use poke_engine::state::{PokemonIndex, PreTransform};
    use std::str::FromStr;
    #[test]
    fn unsupported_transformed_unknown_types_refuse_before_the_free_native_type_path() {
        for (foe, ability, own, kind) in [
            (
                "MAGNETON",
                Abilities::MAGNETPULL,
                "PIKACHU",
                PokemonType::ELECTRIC,
            ),
            (
                "DUGTRIO",
                Abilities::SANDVEIL,
                "CHARIZARD",
                PokemonType::FLYING,
            ),
        ] {
            let (mut state, lines) = fixture(foe, ability, own, kind, Abilities::STATIC);
            let snapshot = PreTransform::capture(state.side_one.get_active_immutable());
            state.side_one.get_active().pre_transform = Some(Box::new(snapshot));
            assert!(observe(
                &state.side_one,
                &state.side_two,
                "p1",
                &lines,
                Some("gen3randombattle"),
                true
            )
            .is_err());
        }
    }
    #[test]
    fn real_constructor_accepts_exact_policy_or_protocol_never_partial_native_sets() {
        let (mut state, lines) = fixture(
            "DUGTRIO",
            Abilities::ARENATRAP,
            "PIKACHU",
            PokemonType::ELECTRIC,
            Abilities::STATIC,
        );
        state.side_one.pokemon.pkmn[2] = state.side_one.get_active_immutable().clone();
        state.side_one.pokemon.pkmn[2].id = FromStr::from_str("SQUIRTLE").unwrap();
        let proof = observe(
            &state.side_one,
            &state.side_two,
            "p1",
            &lines,
            Some("gen3randombattle"),
            true,
        )
        .unwrap()
        .unwrap();
        let names = ["Pikachu", "Snorlax", "Squirtle"].map(String::from);
        let pp = [("ember".into(), 40), ("tackle".into(), 56)].into();
        let policy = state.root_get_all_options().0;
        let build = |options: &[MoveChoice]| {
            crate::policy_request::sampled_side_request_with_observation(
                &state.side_one,
                "p1",
                &names,
                &names,
                options,
                &pp,
                true,
                false,
                None,
                &lines,
                Some("gen3randombattle"),
                Some(&proof),
            )
        };
        assert!(build(&policy).is_ok());
        let mut partial = policy.clone();
        partial.push(MoveChoice::Switch(PokemonIndex::P1));
        assert!(build(&partial).is_err());
        partial.push(MoveChoice::Switch(PokemonIndex::P2));
        assert!(build(&partial).is_ok());
        assert!(build(&policy[..1]).is_err());
    }
    #[test]
    fn canonical_utf8_domain_digests_match_python_including_unicode_and_escaping() {
        let lines = ["|player|p1|Pokémon 🐉".into(), "|turn|1".into()];
        assert_eq!(
            public_binding(&lines)["sha256"],
            "3b8104753c70846260997045cb96e19e82504644078bef1348726b733278b78f"
        );
        assert_eq!(
            digest(
                &json!({"z":[true,0,"Pokémon 🐉"],"a":{"β":"\n"}}),
                b"pokezero.gen3-maybe-trap-own-request.v1"
            ),
            "a5ced8b361a933cac9d207d20c8f566c64e93939b833656bc0d1077d98293ab4"
        );
    }
    fn fixture(
        foe: &str,
        ability: Abilities,
        own: &str,
        kind: PokemonType,
        own_ability: Abilities,
    ) -> (poke_engine::state::State, Vec<String>) {
        let mut state =
            crate::parse_state(include_str!("test_fixtures/minimal.state").trim()).unwrap();
        let mon = state.side_one.get_active();
        mon.id = FromStr::from_str(own).unwrap();
        mon.types = (kind, PokemonType::TYPELESS);
        mon.ability = own_ability;
        state.side_one.pokemon.pkmn[1] = state.side_one.get_active_immutable().clone();
        state.side_one.pokemon.pkmn[1].id = FromStr::from_str("SNORLAX").unwrap();
        let mon = state.side_two.get_active();
        mon.id = FromStr::from_str(foe).unwrap();
        mon.ability = ability;
        let lines = [
            "|gametype|singles".into(),
            "|gen|3".into(),
            "|tier|[Gen 3] Random Battle".into(),
            "|start".into(),
            format!("|switch|p1a: {own}|{own}, L100|100/100"),
            format!("|switch|p2a: {foe}|{foe}, L100|100/100"),
            "|turn|1".into(),
        ]
        .to_vec();
        (state, lines)
    }
    #[test]
    fn server_maybe_flag_respects_gen3_arena_speculation_and_magnet_actual_handler() {
        for (foe, ability, own, kind, own_ability, expected) in [
            (
                "DUGTRIO",
                Abilities::ARENATRAP,
                "PIKACHU",
                PokemonType::ELECTRIC,
                Abilities::STATIC,
                true,
            ),
            (
                "DUGTRIO",
                Abilities::SANDVEIL,
                "PIKACHU",
                PokemonType::ELECTRIC,
                Abilities::STATIC,
                true,
            ),
            (
                "DUGTRIO",
                Abilities::ARENATRAP,
                "CHARIZARD",
                PokemonType::FLYING,
                Abilities::BLAZE,
                false,
            ),
            (
                "DUGTRIO",
                Abilities::SANDVEIL,
                "GENGAR",
                PokemonType::GHOST,
                Abilities::LEVITATE,
                false,
            ),
            (
                "MAGNETON",
                Abilities::MAGNETPULL,
                "METAGROSS",
                PokemonType::STEEL,
                Abilities::CLEARBODY,
                true,
            ),
            (
                "MAGNETON",
                Abilities::STURDY,
                "METAGROSS",
                PokemonType::STEEL,
                Abilities::CLEARBODY,
                false,
            ),
            (
                "MAGNETON",
                Abilities::MAGNETPULL,
                "PIKACHU",
                PokemonType::ELECTRIC,
                Abilities::STATIC,
                false,
            ),
        ] {
            let (state, lines) = fixture(foe, ability, own, kind, own_ability);
            let seen = observe(
                &state.side_one,
                &state.side_two,
                "p1",
                &lines,
                Some("gen3randombattle"),
                true,
            )
            .unwrap();
            assert_eq!(seen.is_some(), expected, "{foe} {ability:?} {own}");
            if let Some(seen) = seen {
                let receipt = seen
                    .receipt(
                        &state.side_one,
                        &json!({"side":{"id":"p1", "pokemon":[{"ident": format!("p1: {own}"), "ability":format!("{own_ability:?}").to_lowercase(), "active":true}]},
                            "active":[{"maybeTrapped":true}], "name":"Pokémon 🐉"}),
                        &[0, 4],
                        &[0],
                    )
                    .unwrap();
                let text = receipt.to_string();
                assert!(!text.contains("MAGNETPULL") && !text.contains("ARENATRAP"));
                assert_eq!(receipt["public_binding"], public_binding(&lines));
            }
        }
    }
    #[test]
    fn unsupported_public_context_fails_closed_and_root_latch_does_not_survive_as_a_child_fact() {
        let (mut state, lines) = fixture(
            "DUGTRIO",
            Abilities::ARENATRAP,
            "PIKACHU",
            PokemonType::ELECTRIC,
            Abilities::STATIC,
        );
        for extra in [
            "|move|p1a: PIKACHU|Skill Swap|p2a: DUGTRIO",
            "|-ability|p2a: DUGTRIO|Sand Veil",
            "|turn|3",
            "|-start|p1a: PIKACHU|typechange|Flying",
            "|faint|p2a: DUGTRIO",
        ] {
            let mut changed = lines.clone();
            changed.push(extra.into());
            assert!(
                observe(
                    &state.side_one,
                    &state.side_two,
                    "p1",
                    &changed,
                    Some("gen3randombattle"),
                    true
                )
                .is_err(),
                "{extra}"
            );
        }
        state.side_one.force_trapped = true;
        assert!(observe(
            &state.side_one,
            &state.side_two,
            "p1",
            &lines,
            Some("gen3randombattle"),
            true
        )
        .unwrap()
        .is_none());
        assert!(observe(
            &state.side_one,
            &state.side_two,
            "p1",
            &lines,
            Some("gen3randombattle"),
            false
        )
        .unwrap()
        .is_some());
        state
            .side_one
            .volatile_statuses
            .insert(PokemonVolatileStatus::MUSTRECHARGE);
        assert!(observe(
            &state.side_one,
            &state.side_two,
            "p1",
            &lines,
            Some("gen3randombattle"),
            false
        )
        .unwrap()
        .is_none());
    }
    #[test]
    fn proof_binds_own_eligibility_and_current_entry_but_never_emits_a_no_bench_flag() {
        let (mut state, lines) = fixture(
            "DUGTRIO",
            Abilities::ARENATRAP,
            "PIKACHU",
            PokemonType::ELECTRIC,
            Abilities::STATIC,
        );
        let proof = observe(
            &state.side_one,
            &state.side_two,
            "p1",
            &lines,
            Some("gen3randombattle"),
            true,
        )
        .unwrap()
        .unwrap();
        let request = json!({"side":{"id":"p1","pokemon":[{"ident":"p1: PIKACHU","active":true,"ability":"static"}]},"active":[{"maybeTrapped":true}]});
        assert!(proof
            .receipt(&state.side_one, &request, &[0, 4], &[0])
            .is_ok());
        let mut wrong = request.clone();
        wrong["side"]["pokemon"][0]["ident"] = json!("p1: GENGAR");
        assert!(proof
            .receipt(&state.side_one, &wrong, &[0, 4], &[0])
            .is_err());
        wrong = request.clone();
        wrong["side"]["pokemon"][0]["ability"] = json!("levitate");
        assert!(proof
            .receipt(&state.side_one, &wrong, &[0, 4], &[0])
            .is_err());
        state.side_one.get_active().ability = Abilities::LEVITATE;
        assert!(proof
            .receipt(&state.side_one, &request, &[0, 4], &[0])
            .is_err());
        state.side_one.get_active().ability = Abilities::STATIC;
        let mut later = lines.clone();
        later.push("|-ability|p2a: DUGTRIO|Arena Trap".into());
        assert!(observe(
            &state.side_one,
            &state.side_two,
            "p1",
            &later,
            Some("gen3randombattle"),
            true
        )
        .unwrap()
        .is_some());
        later.push("|move|p2a: DUGTRIO|Role Play|p1a: PIKACHU".into());
        assert!(observe(
            &state.side_one,
            &state.side_two,
            "p1",
            &later,
            Some("gen3randombattle"),
            true
        )
        .is_err());
        later.push("|switch|p2a: DUGTRIO|DUGTRIO, L100|100/100".into());
        assert!(observe(
            &state.side_one,
            &state.side_two,
            "p1",
            &later,
            Some("gen3randombattle"),
            true
        )
        .unwrap()
        .is_some());
        for (foe, ability, kind) in [
            ("DUGTRIO", Abilities::ARENATRAP, PokemonType::ELECTRIC),
            ("DUGTRIO", Abilities::SANDVEIL, PokemonType::ELECTRIC),
            ("MAGNETON", Abilities::MAGNETPULL, PokemonType::STEEL),
            ("MAGNETON", Abilities::STURDY, PokemonType::STEEL),
        ] {
            let (mut empty, lines) = fixture(foe, ability, "PIKACHU", kind, Abilities::STATIC);
            empty.side_one.pokemon.pkmn[1].hp = 0;
            assert!(observe(
                &empty.side_one,
                &empty.side_two,
                "p1",
                &lines,
                Some("gen3randombattle"),
                true
            )
            .unwrap()
            .is_none());
        }
    }
}
