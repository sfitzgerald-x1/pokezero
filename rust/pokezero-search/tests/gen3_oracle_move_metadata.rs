//! Pinned Gen3 move data required by the GPU transition oracle.
use poke_engine::choices::{Choices, MOVES};
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::MoveChoice;
use poke_engine::state::{PokemonMoveIndex, PokemonType, State};

#[test]
fn gen3_extremespeed_and_quickattack_share_priority_one() {
    assert_eq!(MOVES[&Choices::EXTREMESPEED].priority, 1);
    assert_eq!(MOVES[&Choices::QUICKATTACK].priority, 1);
}

#[test]
fn gen3_protect_priority_is_three() {
    assert_eq!(MOVES[&Choices::PROTECT].priority, 3);
}

#[test]
fn gen3_moonlight_and_charm_are_normal_not_fairy() {
    assert_eq!(MOVES[&Choices::MOONLIGHT].move_type, PokemonType::NORMAL);
    assert_eq!(MOVES[&Choices::CHARM].move_type, PokemonType::NORMAL);
}

#[test]
fn gen3_curse_is_typeless() {
    assert_eq!(MOVES[&Choices::CURSE].move_type, PokemonType::TYPELESS);
}

#[test]
fn faster_quickattack_beats_slower_extremespeed_on_both_seats() {
    for fast_side_one in [false, true] {
        let mut state = State::default();
        for (side, fast) in [(&mut state.side_one, fast_side_one),
                             (&mut state.side_two, !fast_side_one)] {
            let active = side.get_active();
            active.hp = 1;
            active.maxhp = 100;
            active.speed = if fast { 200 } else { 100 };
            active.replace_move(PokemonMoveIndex::M0,
                if fast { Choices::QUICKATTACK } else { Choices::EXTREMESPEED });
        }
        let before = state.serialize();
        let action = MoveChoice::Move(PokemonMoveIndex::M0);
        let branches = generate_instructions_from_move_pair(&mut state, &action, &action, true);
        assert!(!branches.is_empty());
        for branch in branches {
            if branch.percentage <= 0.0 { continue; }
            state.apply_instructions(&branch.instruction_list);
            assert_eq!(state.side_one.get_active_immutable().hp, if fast_side_one { 1 } else { 0 });
            assert_eq!(state.side_two.get_active_immutable().hp, if fast_side_one { 0 } else { 1 });
            state.reverse_instructions(&branch.instruction_list);
            assert_eq!(state.serialize(), before);
        }
    }
}
