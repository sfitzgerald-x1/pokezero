use poke_engine::choices::Choices;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::items::Items;
use poke_engine::engine::state::{MoveChoice, PokemonVolatileStatus};
use poke_engine::instruction::Instruction;
use poke_engine::state::{LastUsedMove, PokemonIndex, PokemonMoveIndex, PokemonType, SideReference, State};

fn exhausted() -> State {
    let mut state = State::default();
    for side in [&mut state.side_one, &mut state.side_two] {
        side.get_active().replace_move(PokemonMoveIndex::M0, Choices::TACKLE);
        side.get_active().moves[&PokemonMoveIndex::M0].pp = 0;
    }
    state
}

#[test]
fn exhausted_moves_offer_struggle_and_live_switches_on_both_seats() {
    let state = exhausted();
    let (a, b) = state.get_all_options();
    for options in [a, b] {
        assert_eq!(options.iter().filter(|v| **v == MoveChoice::Struggle).count(), 1);
        assert!(options.iter().any(|v| matches!(v, MoveChoice::Switch(_))));
        assert!(!options.iter().any(|v| matches!(v, MoveChoice::Move(_) | MoveChoice::None)));
    }
    assert_eq!(state.root_get_all_options(), state.get_all_options());
}

#[test]
fn disabled_moves_with_positive_pp_offer_struggle() {
    let mut state = exhausted();
    state.side_one.get_active().moves[&PokemonMoveIndex::M0].pp = 7;
    state.side_one.get_active().moves[&PokemonMoveIndex::M0].disabled = true;
    assert!(state.get_all_options().0.contains(&MoveChoice::Struggle));
}

#[test]
fn taunted_status_only_actor_can_struggle() {
    let mut state = exhausted();
    state.side_one.get_active().replace_move(PokemonMoveIndex::M0, Choices::GROWTH);
    state.side_one.volatile_statuses.insert(PokemonVolatileStatus::TAUNT);
    assert!(state.get_all_options().0.contains(&MoveChoice::Struggle));
}

#[test]
fn enabled_real_move_suppresses_struggle() {
    let mut state = exhausted();
    state.side_one.get_active().moves[&PokemonMoveIndex::M0].pp = 1;
    let options = state.get_all_options().0;
    assert!(options.contains(&MoveChoice::Move(PokemonMoveIndex::M0)));
    assert!(!options.contains(&MoveChoice::Struggle));
}

#[test]
fn trapped_root_retains_struggle_not_a_noop() {
    let mut state = exhausted();
    state.side_one.force_trapped = true;
    assert_eq!(state.root_get_all_options().0, vec![MoveChoice::Struggle]);
}

#[test]
fn no_bench_actor_retains_struggle() {
    let mut state = exhausted();
    for i in [PokemonIndex::P1, PokemonIndex::P2, PokemonIndex::P3, PokemonIndex::P4, PokemonIndex::P5] {
        state.side_one.pokemon[i].hp = 0;
    }
    assert_eq!(state.get_all_options().0, vec![MoveChoice::Struggle]);
}

#[test]
fn recharge_and_replacement_waits_remain_noops() {
    let mut state = exhausted();
    state.side_one.volatile_statuses.insert(PokemonVolatileStatus::MUSTRECHARGE);
    assert_eq!(state.get_all_options().0, vec![MoveChoice::None]);
    state.side_one.volatile_statuses.remove(&PokemonVolatileStatus::MUSTRECHARGE);
    state.side_two.force_switch = true;
    assert_eq!(state.get_all_options().0, vec![MoveChoice::None]);
    assert!(!state.get_all_options().1.contains(&MoveChoice::Struggle));
}

#[test]
fn fainted_actor_never_gets_struggle() {
    let mut state = exhausted();
    state.side_one.get_active().hp = 0;
    assert!(!state.get_all_options().0.contains(&MoveChoice::Struggle));
}

#[test]
fn saved_struggle_commitment_survives_replacement() {
    let mut state = exhausted();
    state.side_two.force_switch = true;
    state.side_one.switch_out_move_second_saved_move = Choices::STRUGGLE;
    assert_eq!(state.get_all_options().0, vec![MoveChoice::Struggle]);
}

#[test]
fn synthetic_action_parses_without_replacing_any_slot() {
    let state = exhausted();
    assert_eq!(MoveChoice::from_string("struggle", &state.side_one), Some(MoveChoice::Struggle));
    assert_eq!(MoveChoice::Struggle.to_string(&state.side_one), "struggle");
    assert_eq!(LastUsedMove::deserialize(&LastUsedMove::Struggle.serialize()), LastUsedMove::Struggle);
}

#[test]
fn struggle_hits_ghost_recoils_and_never_charges_pp_or_choice_locks() {
    let mut state = exhausted();
    state.use_last_used_move = true;
    state.side_one.get_active().item = Items::CHOICEBAND;
    state.side_two.get_active().types = (PokemonType::GHOST, PokemonType::GHOST);
    let before = state.serialize();
    let branches = generate_instructions_from_move_pair(&mut state, &MoveChoice::Struggle, &MoveChoice::None, false);
    assert_eq!(state.serialize(), before, "generation must restore every input");
    assert!(!branches.is_empty());
    for branch in branches {
        assert!(!branch.instruction_list.iter().any(|i| matches!(i, Instruction::DecrementPP(_) | Instruction::DisableMove(_))));
        assert!(branch.instruction_list.iter().any(|i| matches!(i, Instruction::Damage(d) if d.side_ref == SideReference::SideTwo && d.damage_amount > 0)));
        assert!(branch.instruction_list.iter().any(|i| matches!(i, Instruction::Damage(d) if d.side_ref == SideReference::SideOne && d.damage_amount > 0)));
        state.apply_instructions(&branch.instruction_list);
        assert_eq!(state.side_one.last_used_move, LastUsedMove::Struggle);
        assert_eq!(state.side_one.get_active_immutable().moves[&PokemonMoveIndex::M0].id, Choices::TACKLE);
        assert_eq!(state.side_one.get_active_immutable().moves[&PokemonMoveIndex::M0].pp, 0);
        let encoded = state.serialize();
        assert_eq!(State::deserialize(&encoded).serialize(), encoded);
        state.reverse_instructions(&branch.instruction_list);
        assert_eq!(state.serialize(), before);
    }
}

#[test]
fn encore_cannot_redirect_struggle_to_exhausted_real_slot() {
    let mut state = exhausted();
    state.use_last_used_move = true;
    state.side_one.last_used_move = LastUsedMove::Move(PokemonMoveIndex::M0);
    state.side_one.volatile_statuses.insert(PokemonVolatileStatus::ENCORE);
    let branches = generate_instructions_from_move_pair(&mut state, &MoveChoice::Struggle, &MoveChoice::None, false);
    assert!(branches.iter().all(|b| b.instruction_list.iter().any(|i| matches!(i, Instruction::Damage(d) if d.side_ref == SideReference::SideTwo && d.damage_amount > 0))));
}
