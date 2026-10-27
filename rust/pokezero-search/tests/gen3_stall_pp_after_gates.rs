//! PP/history are paid after BeforeMove, before Protect/Detect/Endure can fail.
//! Authored source controls only; no game/roster or coverage credit.
use poke_engine::choices::Choices;
use poke_engine::engine::abilities::Abilities;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::{MoveChoice, PokemonVolatileStatus};
use poke_engine::state::{
    LastUsedMove, PokemonIndex, PokemonMoveIndex, PokemonStatus, SideReference, State,
};

fn fixture(side: SideReference, move_id: Choices, pp: i8, chain: i8) -> State {
    let mut state = State::default();
    state.use_last_used_move = true;
    for seat in [SideReference::SideOne, SideReference::SideTwo] {
        let team = state.get_side(&seat);
        team.last_used_move = LastUsedMove::Move(PokemonMoveIndex::M1);
        for p in [PokemonIndex::P0, PokemonIndex::P1] {
            let mon = &mut team.pokemon[p];
            mon.hp = 1000;
            mon.maxhp = 1000;
            mon.speed = if seat == side { 300 } else { 100 };
            mon.ability = Abilities::NONE;
            mon.base_ability = Abilities::NONE;
            mon.replace_move(PokemonMoveIndex::M0, Choices::SPLASH);
            mon.replace_move(PokemonMoveIndex::M1, Choices::SPLASH);
            mon.replace_move(PokemonMoveIndex::M2, Choices::NONE);
            mon.replace_move(PokemonMoveIndex::M3, Choices::NONE);
        }
    }
    let team = state.get_side(&side);
    team.side_conditions.protect = chain;
    team.get_active()
        .replace_move(PokemonMoveIndex::M0, move_id);
    team.get_active().moves[&PokemonMoveIndex::M0].pp = pp;
    state
}

fn successors(
    state: &State,
    side: SideReference,
    own: MoveChoice,
    other: MoveChoice,
) -> Vec<(f64, State)> {
    let before = format!("{:?}", state);
    let mut parent = state.clone();
    let (one, two) = if side == SideReference::SideOne {
        (own, other)
    } else {
        (other, own)
    };
    let branches = generate_instructions_from_move_pair(&mut parent, &one, &two, false);
    assert_eq!(format!("{:?}", parent), before, "parent mutated");
    let mut mass = 0.;
    let mut result = Vec::new();
    for branch in branches {
        let p = branch.percentage as f64 / 100.;
        mass += p;
        if p == 0. {
            continue;
        }
        let mut next = state.clone();
        next.apply_instructions(&branch.instruction_list);
        let mut reverse = next.clone();
        reverse.reverse_instructions(&branch.instruction_list);
        assert_eq!(format!("{:?}", reverse), before, "apply/reverse drift");
        result.push((p, next));
    }
    assert!((mass - 1.).abs() < 1e-6, "mass {}", mass);
    result
}

#[test]
fn direct_stall_failures_spend_pp_and_record_history_in_both_profiles_and_seats() {
    for side in [SideReference::SideOne, SideReference::SideTwo] {
        for move_id in [Choices::PROTECT, Choices::DETECT, Choices::ENDURE] {
            for pp in [1, 9, 16] {
                for chain in [1, 2, 3, 5] {
                    for switched in [false, true] {
                        let state = fixture(side, move_id, pp, chain);
                        let other = if switched {
                            MoveChoice::Switch(PokemonIndex::P1)
                        } else {
                            MoveChoice::Move(PokemonMoveIndex::M1)
                        };
                        for (_, next) in
                            successors(&state, side, MoveChoice::Move(PokemonMoveIndex::M0), other)
                        {
                            let team = next.get_side_immutable(&side);
                            let expected = if cfg!(feature = "oracle-full-pp") || pp < 10 {
                                pp - 1
                            } else {
                                pp
                            };
                            assert_eq!(
                                team.get_active_immutable().moves[&PokemonMoveIndex::M0].pp,
                                expected,
                                "{:?}, chain {}, switch {}",
                                move_id,
                                chain,
                                switched
                            );
                            assert_eq!(
                                team.last_used_move,
                                LastUsedMove::Move(PokemonMoveIndex::M0)
                            );
                            if switched {
                                assert_eq!(team.side_conditions.protect, 0);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn accepted_paralysis_prefix_pays_both_stall_branches_but_blocked_prefix_is_free() {
    for side in [SideReference::SideOne, SideReference::SideTwo] {
        let mut state = fixture(side, Choices::PROTECT, 9, 1);
        state.get_side(&side).get_active().status = PokemonStatus::PARALYZE;
        let mut blocked = 0.;
        let mut paid = 0.;
        for (mass, next) in successors(
            &state,
            side,
            MoveChoice::Move(PokemonMoveIndex::M0),
            MoveChoice::Move(PokemonMoveIndex::M1),
        ) {
            let team = next.get_side_immutable(&side);
            match team.get_active_immutable().moves[&PokemonMoveIndex::M0].pp {
                9 => {
                    blocked += mass;
                    assert_eq!(
                        team.last_used_move,
                        LastUsedMove::Move(PokemonMoveIndex::M1)
                    );
                }
                8 => {
                    paid += mass;
                    assert_eq!(
                        team.last_used_move,
                        LastUsedMove::Move(PokemonMoveIndex::M0)
                    );
                }
                pp => panic!("unexpected PP {}", pp),
            }
        }
        assert!((blocked - 0.25).abs() < 1e-6);
        assert!((paid - 0.75).abs() < 1e-6);
    }
}

#[test]
fn sleep_and_flinch_before_move_gates_remain_free() {
    for side in [SideReference::SideOne, SideReference::SideTwo] {
        for asleep in [false, true] {
            let mut state = fixture(side, Choices::PROTECT, 9, 1);
            let team = state.get_side(&side);
            if asleep {
                team.get_active().status = PokemonStatus::SLEEP;
                team.get_active().rest_turns = 3;
            } else {
                team.volatile_statuses.insert(PokemonVolatileStatus::FLINCH);
            }
            for (_, next) in successors(
                &state,
                side,
                MoveChoice::Move(PokemonMoveIndex::M0),
                MoveChoice::Move(PokemonMoveIndex::M1),
            ) {
                let team = next.get_side_immutable(&side);
                assert_eq!(
                    team.get_active_immutable().moves[&PokemonMoveIndex::M0].pp,
                    9
                );
                assert_eq!(
                    team.last_used_move,
                    LastUsedMove::Move(PokemonMoveIndex::M1)
                );
            }
        }
    }
}

#[test]
fn sleep_talk_called_protect_rolls_stall_without_charging_called_pp_or_history() {
    for side in [SideReference::SideOne, SideReference::SideTwo] {
        let mut state = fixture(side, Choices::PROTECT, 9, 1);
        let team = state.get_side(&side);
        team.last_used_move = LastUsedMove::Move(PokemonMoveIndex::M0);
        let mon = team.get_active();
        mon.replace_move(PokemonMoveIndex::M1, Choices::SLEEPTALK);
        mon.moves[&PokemonMoveIndex::M1].pp = 16;
        mon.status = PokemonStatus::SLEEP;
        mon.rest_turns = 3;
        let mut failed = 0.;
        let mut succeeded = 0.;
        for (mass, next) in successors(
            &state,
            side,
            MoveChoice::Move(PokemonMoveIndex::M1),
            MoveChoice::Move(PokemonMoveIndex::M1),
        ) {
            let team = next.get_side_immutable(&side);
            let mon = team.get_active_immutable();
            assert_eq!(mon.moves[&PokemonMoveIndex::M0].pp, 9);
            assert_eq!(
                mon.moves[&PokemonMoveIndex::M1].pp,
                if cfg!(feature = "oracle-full-pp") {
                    15
                } else {
                    16
                }
            );
            assert_eq!(
                team.last_used_move,
                LastUsedMove::Move(PokemonMoveIndex::M1)
            );
            match team.side_conditions.protect {
                0 => failed += mass,
                2 => succeeded += mass,
                x => panic!("counter {}", x),
            }
        }
        assert!((failed - 0.5).abs() < 1e-6);
        assert!((succeeded - 0.5).abs() < 1e-6);
    }
}

#[test]
fn rest_wake_prefix_reaches_stall_only_after_pp_and_history() {
    for side in [SideReference::SideOne, SideReference::SideTwo] {
        let mut state = fixture(side, Choices::PROTECT, 9, 1);
        state.get_side(&side).get_active().status = PokemonStatus::SLEEP;
        state.get_side(&side).get_active().rest_turns = 1;
        for (_, next) in successors(
            &state,
            side,
            MoveChoice::Move(PokemonMoveIndex::M0),
            MoveChoice::Move(PokemonMoveIndex::M1),
        ) {
            let team = next.get_side_immutable(&side);
            let mon = team.get_active_immutable();
            assert_eq!(mon.status, PokemonStatus::NONE);
            assert_eq!(mon.moves[&PokemonMoveIndex::M0].pp, 8);
            assert_eq!(
                team.last_used_move,
                LastUsedMove::Move(PokemonMoveIndex::M0)
            );
        }
    }
}
