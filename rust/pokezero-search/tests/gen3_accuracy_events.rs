//! Authored full successors; no panel replay or strength qualification.
use poke_engine::choices::Choices;
use poke_engine::engine::abilities::Abilities;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::items::Items;
use poke_engine::engine::state::{MoveChoice, Weather};
use poke_engine::state::{PokemonIndex, PokemonMoveIndex, PokemonType, SideReference, State};

fn fixture(actor: SideReference) -> State {
    let mut state = State::default();
    state.weather.weather_type = Weather::SAND;
    state.weather.turns_remaining = -1;
    for seat in [SideReference::SideOne, SideReference::SideTwo] {
        let team = state.get_side(&seat);
        team.accuracy_boost = if seat == actor { -6 } else { 0 };
        team.evasion_boost = if seat == actor { 0 } else { 6 };
        for index in [PokemonIndex::P0, PokemonIndex::P1, PokemonIndex::P2,
            PokemonIndex::P3, PokemonIndex::P4, PokemonIndex::P5] {
            let p = &mut team.pokemon[index];
            p.hp = 1000; p.maxhp = 1000; p.level = 73;
            p.types = (PokemonType::NORMAL, PokemonType::TYPELESS);
            p.base_types = p.types;
            p.special_attack = 103; p.special_defense = 101;
            p.speed = if seat == actor { 300 } else { 100 };
            p.ability = if seat == actor { Abilities::NONE } else { Abilities::SANDVEIL };
            p.base_ability = p.ability; p.item = Items::NONE;
            for (slot,id) in [(PokemonMoveIndex::M0,Choices::PURSUIT),
                (PokemonMoveIndex::M1,Choices::SPLASH),(PokemonMoveIndex::M2,Choices::TOXIC),
                (PokemonMoveIndex::M3,Choices::AERIALACE)] {
                p.replace_move(slot,id);p.moves[&slot].pp=16;p.moves[&slot].disabled=false;
            }
        }
    }
    state
}

fn departing_target_hit_mass(state: &State, actor: SideReference, switching: bool) -> f64 {
    let target = actor.get_other_side();
    let opponent = if switching { MoveChoice::Switch(PokemonIndex::P1) }
        else { MoveChoice::Move(PokemonMoveIndex::M1) };
    let own = MoveChoice::Move(PokemonMoveIndex::M0);
    let actions = if actor == SideReference::SideOne { (own,opponent) } else { (opponent,own) };
    let before=format!("{:?}",state);let mut parent=state.clone();
    let branches=generate_instructions_from_move_pair(&mut parent,&actions.0,&actions.1,false);
    assert_eq!(format!("{:?}",parent),before,"parent mutation");
    let mut total=0.0;let mut hit=0.0;
    for branch in branches {
        let mass=branch.percentage as f64/100.0;assert!(mass>=0.0);total+=mass;
        let mut child=state.clone();child.apply_instructions(&branch.instruction_list);
        let team=child.get_side_immutable(&target);
        if team.pokemon[&PokemonIndex::P0].hp<1000 {hit+=mass;}
        if switching {assert_eq!(team.active_index,PokemonIndex::P1);}
        child.reverse_instructions(&branch.instruction_list);
        assert_eq!(format!("{:?}",child),before,"reversal mutation");
    }
    assert!((total-1.0).abs()<1e-6,"mass {total}");hit
}

#[test]
fn switching_pursuit_bypasses_stages_and_sand_veil_both_seats() {
    for actor in [SideReference::SideOne,SideReference::SideTwo] {
        assert!((departing_target_hit_mass(&fixture(actor),actor,true)-1.0).abs()<1e-6);
    }
}

#[test]
fn ordinary_pursuit_retains_numeric_accuracy_event_both_seats() {
    for actor in [SideReference::SideOne,SideReference::SideTwo] {
        // Source100/3/3 is fractional, so Q12 Sand Veil is skipped. Exactly
        // twelve integer draws in0..99 land, versus interception's hundred.
        assert!((departing_target_hit_mass(&fixture(actor),actor,false)-0.12).abs()<1e-6);
    }
}
