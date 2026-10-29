//! Authored source unit. No recovered battle, fixture, replay, MCTS, or model.
//! Fresh default-state transition generation supplies a complete renderer input.
//! Preserves the observed refusal class only through external immutable lineage.
use poke_engine::choices::Choices;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::MoveChoice;
use poke_engine::instruction::{Instruction, SleepTalkCalledInstruction};
use poke_engine::state::{PokemonIndex, PokemonMoveIndex, PokemonStatus, SideReference, State};
use pokezero_search::events::{render_branch_events, EventContext};

#[test]
fn a4_fresh_sleep_talk_marker_identity_and_malformed_lineage() {
    for sleeper in [SideReference::SideOne, SideReference::SideTwo] {
        let mut state = State::default();
        for side in [SideReference::SideOne, SideReference::SideTwo] {
            let active = match side {
                SideReference::SideOne => state.side_one.get_active(),
                SideReference::SideTwo => state.side_two.get_active(),
            };
            active.replace_move(PokemonMoveIndex::M0, Choices::SPLASH);
        }
        let active = match sleeper {
            SideReference::SideOne => state.side_one.get_active(),
            SideReference::SideTwo => state.side_two.get_active(),
        };
        active.status = PokemonStatus::SLEEP;
        active.rest_turns = 2;
        active.replace_move(PokemonMoveIndex::M0, Choices::SLEEPTALK);
        active.replace_move(PokemonMoveIndex::M1, Choices::SPLASH);
        active.replace_move(PokemonMoveIndex::M2, Choices::REST);
        active.replace_move(PokemonMoveIndex::M3, Choices::PROTECT);
        let ctx = EventContext {
            species: [
                vec!["SourceToyOne".into(); 6],
                vec!["SourceToyTwo".into(); 6],
            ],
            turn: 1,
            hp_percent: [false, false],
        };
        let opponent = match sleeper {
            SideReference::SideOne => &mut state.side_two,
            SideReference::SideTwo => &mut state.side_one,
        };
        opponent.pokemon[PokemonIndex::P1] = opponent.get_active_immutable().clone();
        let before = state.serialize();
        let (one, two) = if sleeper == SideReference::SideOne {
            (
                MoveChoice::Move(PokemonMoveIndex::M0),
                MoveChoice::Switch(PokemonIndex::P1),
            )
        } else {
            (
                MoveChoice::Switch(PokemonIndex::P1),
                MoveChoice::Move(PokemonMoveIndex::M0),
            )
        };
        let branches = generate_instructions_from_move_pair(&mut state, &one, &two, false);
        assert_eq!(state.serialize(), before);
        let base = &branches.iter().find(|branch| branch.instruction_list.iter().any(|instruction|
            matches!(instruction, Instruction::SleepTalkCalled(marker) if marker.side_ref == sleeper && marker.move_id == Choices::SPLASH)
        )).expect("fresh toy must reach a Splash callee").instruction_list;
        let index = base
            .iter()
            .position(|instruction| matches!(instruction, Instruction::SleepTalkCalled(_)))
            .unwrap();
        for corruption in 0..4 {
            let mut instructions = base.clone();
            match corruption {
                0 => {}
                1 => {
                    instructions.remove(index);
                }
                2 => {
                    instructions.insert(index, instructions[index].clone());
                }
                _ => {
                    instructions[index] =
                        Instruction::SleepTalkCalled(SleepTalkCalledInstruction {
                            side_ref: if sleeper == SideReference::SideOne {
                                SideReference::SideTwo
                            } else {
                                SideReference::SideOne
                            },
                            move_id: Choices::SPLASH,
                        });
                }
            }
            let rendered = render_branch_events(&mut state, &one, &two, &instructions, false, &ctx);
            assert_eq!(
                state.serialize(),
                before,
                "source renderer mutated toy state"
            );
            if corruption == 0 {
                assert!(!rendered.is_attribution_unsafe(), "{rendered:?}");
                assert!(rendered
                    .lines
                    .iter()
                    .any(|line| line.contains("|splash|") && line.contains("[from] Sleep Talk")));
            } else {
                assert!(
                    rendered.is_attribution_unsafe(),
                    "corruption {corruption}: {rendered:?}"
                );
                // A complete fresh branch may refuse earlier at segmentation
                // once its producer's mandatory marker is corrupted. Neither
                // refusal may become an accepted trace.
                assert!(
                    rendered
                        .attribution_unsafe
                        .iter()
                        .any(|reason| reason == "sleeptalk_marker_missing_or_conflicting"
                            || reason == "segmentation_failed"),
                    "{sleeper:?}/{corruption}: {rendered:?}"
                );
            }
        }
    }
}

#[test]
fn a4_two_sleep_talk_actions_allow_one_marker_per_seat() {
    let mut state = State::default();
    for side in [&mut state.side_one, &mut state.side_two] {
        let pokemon = side.get_active();
        pokemon.status = PokemonStatus::SLEEP;
        pokemon.rest_turns = 2;
        pokemon.replace_move(PokemonMoveIndex::M0, Choices::SLEEPTALK);
        pokemon.replace_move(PokemonMoveIndex::M1, Choices::SPLASH);
        pokemon.replace_move(PokemonMoveIndex::M2, Choices::NONE);
        pokemon.replace_move(PokemonMoveIndex::M3, Choices::NONE);
    }
    state.side_one.get_active().speed = 500;
    state.side_two.get_active().speed = 1;
    let choice = MoveChoice::Move(PokemonMoveIndex::M0);
    let before = state.serialize();
    let branches = generate_instructions_from_move_pair(&mut state, &choice, &choice, false);
    let ctx = EventContext {
        species: [vec!["ToyOne".into(); 6], vec!["ToyTwo".into(); 6]],
        turn: 1,
        hp_percent: [false, false],
    };
    assert!(!branches.is_empty());
    for branch in branches {
        assert_eq!(
            branch
                .instruction_list
                .iter()
                .filter(|instruction| matches!(instruction, Instruction::SleepTalkCalled(_)))
                .count(),
            2
        );
        let rendered = render_branch_events(
            &mut state,
            &choice,
            &choice,
            &branch.instruction_list,
            false,
            &ctx,
        );
        assert!(!rendered.is_attribution_unsafe(), "{rendered:?}");
        assert_eq!(
            rendered
                .lines
                .iter()
                .filter(|line| line.contains("[from] Sleep Talk"))
                .count(),
            2
        );
        assert_eq!(state.serialize(), before);
    }
}
