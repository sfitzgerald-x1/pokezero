//! Authored regression for Rest -> Natural Cure -> new primary sleep.
//! Pinned simulator durations2..5 give awake masses0,1/4,1/2 after attempts1..3.
//! Component correctness only; no scientific roster or coverage credit.

use poke_engine::choices::Choices;
use poke_engine::engine::abilities::Abilities;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::MoveChoice;
use poke_engine::state::{PokemonIndex, PokemonMoveIndex, PokemonStatus, SideReference, State};

fn successors(
    state: &State,
    side: SideReference,
    own: MoveChoice,
    other: MoveChoice,
) -> Vec<(f64, State)> {
    let before = format!("{:?}", state);
    let mut source = state.clone();
    let (one, two) = if side == SideReference::SideOne {
        (own, other)
    } else {
        (other, own)
    };
    let branches = generate_instructions_from_move_pair(&mut source, &one, &two, false);
    assert_eq!(format!("{:?}", source), before, "generator mutated source");
    let total: f64 = branches.iter().map(|b| b.percentage as f64 / 100.0).sum();
    assert!((total - 1.0).abs() < 1e-6, "lost mass: {}", total);
    branches
        .into_iter()
        .map(|b| {
            let mut next = state.clone();
            next.apply_instructions(&b.instruction_list);
            let mut reverse = next.clone();
            reverse.reverse_instructions(&b.instruction_list);
            assert_eq!(
                format!("{:?}", reverse),
                before,
                "branch failed exact apply/reverse"
            );
            (b.percentage as f64 / 100.0, next)
        })
        .collect()
}

fn only(state: &State, side: SideReference, own: MoveChoice, other: MoveChoice) -> State {
    let mut out = successors(state, side, own, other);
    assert_eq!(out.len(), 1);
    assert!((out[0].0 - 1.0).abs() < 1e-6);
    out.remove(0).1
}

fn fixture(side: SideReference) -> State {
    let mut state = State::default();
    for seat in [SideReference::SideOne, SideReference::SideTwo] {
        let active = state.get_side(&seat).get_active();
        active.ability = Abilities::NONE;
        active.base_ability = Abilities::NONE;
        active.hp = 1000;
        active.maxhp = 1000;
        active.speed = if seat == side { 300 } else { 100 };
        active.replace_move(PokemonMoveIndex::M0, Choices::SPORE);
        active.replace_move(PokemonMoveIndex::M1, Choices::SPLASH);
        let bench = &mut state.get_side(&seat).pokemon[PokemonIndex::P1];
        bench.ability = Abilities::NONE;
        bench.base_ability = Abilities::NONE;
        bench.hp = 1000;
        bench.maxhp = 1000;
        bench.speed = 200;
    }
    let mon = state.get_side(&side).get_active();
    mon.ability = Abilities::NATURALCURE;
    mon.base_ability = Abilities::NATURALCURE;
    mon.hp = 50;
    mon.replace_move(PokemonMoveIndex::M0, Choices::REST);
    state
}

#[test]
fn departure_clears_real_rest_and_refund_and_natural_sleep_in_both_seats() {
    for side in [SideReference::SideOne, SideReference::SideTwo] {
        for (rest, natural, refund) in [(3, 0, 0), (2, 0, 1), (0, 2, 0)] {
            let mut state = fixture(side);
            let mon = state.get_side(&side).get_active();
            mon.status = PokemonStatus::SLEEP;
            mon.rest_turns = rest;
            mon.sleep_turns = natural;
            mon.rest_sleep_pending_refund = refund;
            let out = only(
                &state,
                side,
                MoveChoice::Switch(PokemonIndex::P1),
                MoveChoice::None,
            );
            let mon = &out.get_side_immutable(&side).pokemon[PokemonIndex::P0];
            assert_eq!(mon.status, PokemonStatus::NONE);
            assert_eq!(
                (
                    mon.rest_turns,
                    mon.sleep_turns,
                    mon.rest_sleep_pending_refund
                ),
                (0, 0, 0)
            );
        }
    }
}

#[test]
fn current_ability_controls_cure_before_base_restoration() {
    for side in [SideReference::SideOne, SideReference::SideTwo] {
        for (current, base, cured) in [
            (Abilities::NATURALCURE, Abilities::TRACE, true),
            (Abilities::NONE, Abilities::NATURALCURE, false),
        ] {
            let mut state = fixture(side);
            let mon = state.get_side(&side).get_active();
            mon.ability = current;
            mon.base_ability = base;
            mon.status = PokemonStatus::SLEEP;
            mon.rest_turns = 3;
            let out = only(
                &state,
                side,
                MoveChoice::Switch(PokemonIndex::P1),
                MoveChoice::None,
            );
            let mon = &out.get_side_immutable(&side).pokemon[PokemonIndex::P0];
            assert_eq!(mon.ability, base);
            assert_eq!(
                mon.status,
                if cured {
                    PokemonStatus::NONE
                } else {
                    PokemonStatus::SLEEP
                }
            );
            assert_eq!(mon.rest_turns, if cured { 0 } else { 3 });
        }
    }
}

#[test]
fn poison_and_toxic_departure_remain_reversible_without_duplicate_reset() {
    for side in [SideReference::SideOne, SideReference::SideTwo] {
        for status in [PokemonStatus::POISON, PokemonStatus::TOXIC] {
            let mut state = fixture(side);
            state.get_side(&side).get_active().status = status;
            if status == PokemonStatus::TOXIC {
                state.get_side(&side).side_conditions.toxic_count = 3;
            }
            let out = only(
                &state,
                side,
                MoveChoice::Switch(PokemonIndex::P1),
                MoveChoice::None,
            );
            assert_eq!(
                out.get_side_immutable(&side).pokemon[PokemonIndex::P0].status,
                PokemonStatus::NONE
            );
            assert_eq!(out.get_side_immutable(&side).side_conditions.toxic_count, 0);
        }
    }
}

#[test]
fn new_primary_sleep_after_rest_cure_has_the_full_natural_sleep_law() {
    for side in [SideReference::SideOne, SideReference::SideTwo] {
        let state = fixture(side);
        let rested = only(
            &state,
            side,
            MoveChoice::Move(PokemonMoveIndex::M0),
            MoveChoice::Move(PokemonMoveIndex::M1),
        );
        assert_eq!(
            rested
                .get_side_immutable(&side)
                .get_active_immutable()
                .rest_turns,
            3
        );
        let departed = only(
            &rested,
            side,
            MoveChoice::Switch(PokemonIndex::P1),
            MoveChoice::Move(PokemonMoveIndex::M1),
        );
        let fresh = only(
            &departed,
            side,
            MoveChoice::Switch(PokemonIndex::P0),
            MoveChoice::Move(PokemonMoveIndex::M0),
        );
        let mon = fresh.get_side_immutable(&side).get_active_immutable();
        assert_eq!(mon.status, PokemonStatus::SLEEP);
        assert_eq!(
            mon.rest_turns, 0,
            "fresh natural sleep must not inherit Rest origin"
        );
        let mut law = vec![(1.0, fresh)];
        for expected in [0.0, 0.25, 0.5, 0.75, 1.0] {
            law = law
                .into_iter()
                .flat_map(|(mass, state)| {
                    successors(
                        &state,
                        side,
                        MoveChoice::Move(PokemonMoveIndex::M1),
                        MoveChoice::Move(PokemonMoveIndex::M1),
                    )
                    .into_iter()
                    .map(move |(p, next)| (mass * p, next))
                })
                .collect();
            let total: f64 = law.iter().map(|(p, _)| *p).sum();
            let awake: f64 = law
                .iter()
                .filter(|(_, s)| {
                    s.get_side_immutable(&side).get_active_immutable().status == PokemonStatus::NONE
                })
                .map(|(p, _)| *p)
                .sum();
            assert!((total - 1.0).abs() < 1e-6);
            assert!(
                (awake - expected).abs() < 1e-6,
                "expected {}, actual {}",
                expected,
                awake
            );
        }
    }
}
