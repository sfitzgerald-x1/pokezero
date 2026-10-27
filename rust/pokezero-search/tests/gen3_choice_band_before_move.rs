//! Source-adjudicated fresh Choice Band restrictions follow an accepted move.
use poke_engine::choices::Choices;
use poke_engine::engine::abilities::Abilities;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::items::Items;
use poke_engine::engine::state::{MoveChoice, PokemonVolatileStatus};
use poke_engine::state::{
    LastUsedMove, PokemonMoveIndex, PokemonStatus, SideReference, State,
};

const SLOTS: [PokemonMoveIndex; 4] = [
    PokemonMoveIndex::M0,
    PokemonMoveIndex::M1,
    PokemonMoveIndex::M2,
    PokemonMoveIndex::M3,
];
fn fixture(side: SideReference) -> State {
    let mut state = State::default();
    state.use_last_used_move = true;
    for seat in [SideReference::SideOne, SideReference::SideTwo] {
        let team = state.get_side(&seat);
        team.last_used_move = LastUsedMove::None;
        let p = team.get_active();
        p.hp = 1000;
        p.maxhp = 1000;
        p.speed = if seat == side { 300 } else { 100 };
        p.ability = Abilities::NONE;
        p.base_ability = Abilities::NONE;
        p.item = if seat == side {
            Items::CHOICEBAND
        } else {
            Items::NONE
        };
        for (slot, id) in SLOTS.into_iter().zip([
            Choices::RETURN,
            Choices::SPLASH,
            Choices::REST,
            Choices::PROTECT,
        ]) {
            p.replace_move(slot, id);
            p.moves[&slot].pp = 16;
            p.moves[&slot].disabled = false;
        }
    }
    state
}
fn successors(state: &State, side: SideReference, own: PokemonMoveIndex) -> Vec<(f64, State)> {
    let before = format!("{:?}", state);
    let mut parent = state.clone();
    let mut actions = (
        MoveChoice::Move(own),
        MoveChoice::Move(PokemonMoveIndex::M1),
    );
    if side == SideReference::SideTwo {
        actions = (actions.1, actions.0);
    }
    let branches = generate_instructions_from_move_pair(&mut parent, &actions.0, &actions.1, false);
    assert_eq!(format!("{:?}", parent), before);
    let mut mass = 0.;
    let mut out = Vec::new();
    for branch in branches {
        let probability = branch.percentage as f64 / 100.;
        mass += probability;
        if probability == 0. {
            continue;
        }
        let mut child = state.clone();
        child.apply_instructions(&branch.instruction_list);
        let mut reverse = child.clone();
        reverse.reverse_instructions(&branch.instruction_list);
        assert_eq!(format!("{:?}", reverse), before, "apply/reverse drift");
        out.push((probability, child));
    }
    assert!((mass - 1.).abs() < 1e-6);
    out
}
fn flags(state: &State, side: SideReference) -> [bool; 4] {
    let p = state.get_side_immutable(&side).get_active_immutable();
    SLOTS.map(|i| p.moves[&i].disabled)
}
#[test]
fn first_blocked_attempt_does_not_create_choice_restrictions_in_either_seat() {
    for side in [SideReference::SideOne, SideReference::SideTwo] {
        for kind in 0..4 {
            let mut state = fixture(side);
            let team = state.get_side(&side);
            match kind {
                0 => team.get_active().status = PokemonStatus::FREEZE,
                1 => team.get_active().status = PokemonStatus::PARALYZE,
                2 => {
                    team.get_active().status = PokemonStatus::SLEEP;
                    team.get_active().rest_turns = 3;
                }
                _ => {
                    team.volatile_statuses.insert(PokemonVolatileStatus::FLINCH);
                }
            }
            let mut accepted = 0.;
            let mut blocked = 0.;
            for (mass, next) in successors(&state, side, PokemonMoveIndex::M0) {
                let team = next.get_side_immutable(&side);
                let used = team.last_used_move == LastUsedMove::Move(PokemonMoveIndex::M0);
                if used {
                    accepted += mass;
                    assert_eq!(flags(&next, side), [false, true, true, true]);
                } else {
                    blocked += mass;
                    assert_eq!(team.last_used_move, LastUsedMove::None);
                    assert_eq!(flags(&next, side), [false; 4], "kind {}", kind);
                }
                assert_eq!(
                    team.get_active_immutable().moves[&PokemonMoveIndex::M0].pp,
                    if used && cfg!(feature = "oracle-full-pp") {
                        15
                    } else {
                        16
                    }
                );
            }
            let expected = match kind {
                0 => 0.2,
                1 => 0.75,
                _ => 0.,
            };
            assert!((accepted - expected).abs() < 1e-6);
            assert!((blocked - (1. - expected)).abs() < 1e-6);
        }
    }
}
#[test]
fn existing_restrictions_survive_a_blocked_turn() {
    for side in [SideReference::SideOne, SideReference::SideTwo] {
        let mut state = fixture(side);
        let team = state.get_side(&side);
        team.last_used_move = LastUsedMove::Move(PokemonMoveIndex::M0);
        team.get_active().status = PokemonStatus::SLEEP;
        team.get_active().rest_turns = 3;
        for i in [
            PokemonMoveIndex::M1,
            PokemonMoveIndex::M2,
            PokemonMoveIndex::M3,
        ] {
            team.get_active().moves[&i].disabled = true;
        }
        for (_, next) in successors(&state, side, PokemonMoveIndex::M0) {
            assert_eq!(flags(&next, side), [false, true, true, true]);
        }
    }
}
#[test]
fn accepted_failed_protect_still_creates_the_choice_restriction() {
    for side in [SideReference::SideOne, SideReference::SideTwo] {
        let mut state = fixture(side);
        state.get_side(&side).side_conditions.protect = 1;
        let mut success = 0.;
        let mut fail = 0.;
        for (mass, next) in successors(&state, side, PokemonMoveIndex::M3) {
            assert_eq!(flags(&next, side), [true, true, true, false]);
            let team = next.get_side_immutable(&side);
            assert_eq!(
                team.last_used_move,
                LastUsedMove::Move(PokemonMoveIndex::M3)
            );
            if team.side_conditions.protect == 0 {
                fail += mass;
            } else {
                success += mass;
            }
        }
        assert!((success - 0.5).abs() < 1e-6);
        assert!((fail - 0.5).abs() < 1e-6);
    }
}
#[test]
fn sleep_talk_callees_never_relock_the_caller_to_a_called_move() {
    for side in [SideReference::SideOne, SideReference::SideTwo] {
        let mut state = fixture(side);
        let team = state.get_side(&side);
        let p = team.get_active();
        p.replace_move(PokemonMoveIndex::M1, Choices::SLEEPTALK);
        p.moves[&PokemonMoveIndex::M1].pp = 16;
        p.status = PokemonStatus::SLEEP;
        p.rest_turns = 3;
        for (_, next) in successors(&state, side, PokemonMoveIndex::M1) {
            assert_eq!(flags(&next, side), [true, false, true, true]);
            let team = next.get_side_immutable(&side);
            assert_eq!(
                team.last_used_move,
                LastUsedMove::Move(PokemonMoveIndex::M1)
            );
            let p = team.get_active_immutable();
            assert_eq!(
                p.moves[&PokemonMoveIndex::M1].pp,
                if cfg!(feature = "oracle-full-pp") {
                    15
                } else {
                    16
                }
            );
            for i in [
                PokemonMoveIndex::M0,
                PokemonMoveIndex::M2,
                PokemonMoveIndex::M3,
            ] {
                assert_eq!(p.moves[&i].pp, 16);
            }
        }
    }
}
#[test]
fn awake_sleep_talk_failure_is_an_accepted_move_and_locks_its_caller() {
    for side in [SideReference::SideOne, SideReference::SideTwo] {
        let mut state = fixture(side);
        state
            .get_side(&side)
            .get_active()
            .replace_move(PokemonMoveIndex::M1, Choices::SLEEPTALK);
        for (_, next) in successors(&state, side, PokemonMoveIndex::M1) {
            assert_eq!(flags(&next, side), [true, false, true, true]);
        }
    }
}
#[test]
fn no_choice_item_never_introduces_choice_restrictions() {
    for side in [SideReference::SideOne, SideReference::SideTwo] {
        let mut state = fixture(side);
        state.get_side(&side).get_active().item = Items::NONE;
        for (_, next) in successors(&state, side, PokemonMoveIndex::M0) {
            assert_eq!(flags(&next, side), [false; 4]);
        }
    }
}
