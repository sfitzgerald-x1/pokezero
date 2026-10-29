//! Authored source unit. No recovered battle, fixture, replay, MCTS, or model.
//! Fresh default-state transition generation supplies a complete renderer input.
//! Preserves the observed refusal class only through external immutable lineage.
use poke_engine::choices::Choices;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::{MoveChoice, PokemonVolatileStatus};
use poke_engine::instruction::{Instruction, SleepTalkCalledInstruction};
use poke_engine::state::{
    LastUsedMove, PokemonIndex, PokemonMoveIndex, PokemonStatus, PreTransform, SideReference, State,
};
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

// Additional fresh controls from independent source review.
fn ctx() -> EventContext {
    EventContext {
        species: [vec!["FreshOne".into(); 6], vec!["FreshTwo".into(); 6]],
        turn: 1,
        hp_percent: [false, false],
    }
}
fn toy() -> State {
    let mut s = State::default();
    for side in [&mut s.side_one, &mut s.side_two] {
        let p = side.get_active();
        p.replace_move(PokemonMoveIndex::M0, Choices::SPLASH);
        p.replace_move(PokemonMoveIndex::M1, Choices::NONE);
        p.replace_move(PokemonMoveIndex::M2, Choices::NONE);
        p.replace_move(PokemonMoveIndex::M3, Choices::NONE);
    }
    s
}
fn encore_redirect_probe(sleeper: SideReference) {
    let mut s = toy();
    s.use_last_used_move = true;
    let (asleep, encorer) = match sleeper {
        SideReference::SideOne => (&mut s.side_one, &mut s.side_two),
        SideReference::SideTwo => (&mut s.side_two, &mut s.side_one),
    };
    asleep.last_used_move = LastUsedMove::Move(PokemonMoveIndex::M0);
    let p = asleep.get_active();
    p.status = PokemonStatus::SLEEP;
    p.rest_turns = 2;
    p.speed = 1;
    p.replace_move(PokemonMoveIndex::M0, Choices::SLEEPTALK);
    p.replace_move(PokemonMoveIndex::M1, Choices::SPLASH);
    let p = encorer.get_active();
    p.speed = 500;
    p.replace_move(PokemonMoveIndex::M0, Choices::ENCORE);
    let (one, two) = match sleeper {
        SideReference::SideOne => (
            MoveChoice::Move(PokemonMoveIndex::M1),
            MoveChoice::Move(PokemonMoveIndex::M0),
        ),
        SideReference::SideTwo => (
            MoveChoice::Move(PokemonMoveIndex::M0),
            MoveChoice::Move(PokemonMoveIndex::M1),
        ),
    };
    let before = s.serialize();
    let branches = generate_instructions_from_move_pair(&mut s, &one, &two, false);
    assert_eq!(before, s.serialize());
    let b = branches
        .iter()
        .find(|b| {
            b.instruction_list
                .iter()
                .any(|i| matches!(i,Instruction::SleepTalkCalled(m) if m.side_ref==sleeper))
        })
        .expect("fresh Encore must redirect selected Splash to Sleep Talk");
    let rendered = render_branch_events(&mut s, &one, &two, &b.instruction_list, false, &ctx());
    assert_eq!(before, s.serialize());
    assert!(
        !rendered.is_attribution_unsafe(),
        "valid fresh Encore redirect rejected {sleeper:?}: {rendered:?}; instructions={:?}",
        b.instruction_list
    );
    assert!(rendered
        .lines
        .iter()
        .any(|line| line.contains("[from] Sleep Talk")));
}
#[test]
fn fresh_encore_redirect_side_one() {
    encore_redirect_probe(SideReference::SideOne);
}
#[test]
fn fresh_encore_redirect_side_two() {
    encore_redirect_probe(SideReference::SideTwo);
}
#[test]
fn fresh_struggle_and_ordinary_move_reject_injected_markers() {
    for action in [MoveChoice::Struggle, MoveChoice::Move(PokemonMoveIndex::M0)] {
        let mut s = toy();
        let opponent = MoveChoice::Move(PokemonMoveIndex::M0);
        let before = s.serialize();
        let branches = generate_instructions_from_move_pair(&mut s, &action, &opponent, false);
        for b in branches {
            let base = render_branch_events(
                &mut s,
                &action,
                &opponent,
                &b.instruction_list,
                false,
                &ctx(),
            );
            assert!(
                !base.is_attribution_unsafe(),
                "fresh control rejected: {base:?}"
            );
            let mut ins = b.instruction_list;
            ins.push(Instruction::SleepTalkCalled(SleepTalkCalledInstruction {
                side_ref: SideReference::SideOne,
                move_id: Choices::SPLASH,
            }));
            let bad = render_branch_events(&mut s, &action, &opponent, &ins, false, &ctx());
            assert!(
                bad.attribution_unsafe
                    .iter()
                    .any(|s| s == "sleeptalk_marker_missing_or_conflicting"),
                "{bad:?}"
            );
            assert_eq!(before, s.serialize());
        }
    }
}
#[test]
fn fresh_transformed_sleep_talk_uses_current_copied_moves() {
    let mut s = toy();
    let p = s.side_one.get_active();
    p.replace_move(PokemonMoveIndex::M0, Choices::TRANSFORM);
    p.pre_transform = Some(Box::new(PreTransform::capture(p)));
    p.replace_move(PokemonMoveIndex::M0, Choices::SLEEPTALK);
    p.replace_move(PokemonMoveIndex::M1, Choices::SPLASH);
    p.status = PokemonStatus::SLEEP;
    p.rest_turns = 2;
    s.side_one
        .volatile_statuses
        .insert(PokemonVolatileStatus::TRANSFORMED);
    let one = MoveChoice::Move(PokemonMoveIndex::M0);
    let two = MoveChoice::Move(PokemonMoveIndex::M0);
    let before = s.serialize();
    let branches = generate_instructions_from_move_pair(&mut s, &one, &two, false);
    assert!(!branches.is_empty());
    for b in branches {
        let rendered = render_branch_events(&mut s, &one, &two, &b.instruction_list, false, &ctx());
        assert!(!rendered.is_attribution_unsafe(), "{rendered:?}");
        assert!(rendered
            .lines
            .iter()
            .any(|line| line.contains("[from] Sleep Talk")));
        assert_eq!(before, s.serialize());
    }
}

fn awake_probe(caller: SideReference) {
    let mut s = toy();
    match caller {
        SideReference::SideOne => s.side_one.get_active(),
        SideReference::SideTwo => s.side_two.get_active(),
    }
    .replace_move(PokemonMoveIndex::M0, Choices::SLEEPTALK);
    let action = MoveChoice::Move(PokemonMoveIndex::M0);
    let before = s.serialize();
    let branches = generate_instructions_from_move_pair(&mut s, &action, &action, false);
    assert!(!branches.is_empty());
    for b in branches {
        assert!(!b
            .instruction_list
            .iter()
            .any(|i| matches!(i, Instruction::SleepTalkCalled(_))));
        let base =
            render_branch_events(&mut s, &action, &action, &b.instruction_list, false, &ctx());
        assert!(
            !base.is_attribution_unsafe(),
            "awake fresh control rejected {base:?}"
        );
        let mut corrupted = b.instruction_list;
        corrupted.push(Instruction::SleepTalkCalled(SleepTalkCalledInstruction {
            side_ref: caller,
            move_id: Choices::SPLASH,
        }));
        let bad = render_branch_events(&mut s, &action, &action, &corrupted, false, &ctx());
        assert_eq!(before, s.serialize());
        assert!(
            bad.is_attribution_unsafe(),
            "awake Sleep Talk never calls Splash; spurious marker accepted {caller:?}: {bad:?}"
        );
    }
}
#[test]
fn fresh_awake_side_one_rejects_spurious_marker() {
    awake_probe(SideReference::SideOne);
}
#[test]
fn fresh_awake_side_two_rejects_spurious_marker() {
    awake_probe(SideReference::SideTwo);
}
#[test]
fn fresh_immobilized_sleep_talk_rejects_spurious_marker() {
    for caller in [SideReference::SideOne, SideReference::SideTwo] {
        let mut s = toy();
        let side = match caller {
            SideReference::SideOne => &mut s.side_one,
            SideReference::SideTwo => &mut s.side_two,
        };
        side.volatile_statuses
            .insert(PokemonVolatileStatus::ATTRACT);
        let p = side.get_active();
        p.status = PokemonStatus::SLEEP;
        p.rest_turns = 2;
        p.replace_move(PokemonMoveIndex::M0, Choices::SLEEPTALK);
        p.replace_move(PokemonMoveIndex::M1, Choices::SPLASH);
        let action = MoveChoice::Move(PokemonMoveIndex::M0);
        let before = s.serialize();
        let branches = generate_instructions_from_move_pair(&mut s, &action, &action, false);
        let b = branches
            .iter()
            .find(|b| {
                b.instruction_list
                    .iter()
                    .any(|i| matches!(i,Instruction::MoveImmobilized(m) if m.side_ref==caller))
            })
            .expect("fresh Attract toy must have immobilized branch");
        assert!(!b
            .instruction_list
            .iter()
            .any(|i| matches!(i, Instruction::SleepTalkCalled(_))));
        let base =
            render_branch_events(&mut s, &action, &action, &b.instruction_list, false, &ctx());
        assert!(
            !base.is_attribution_unsafe(),
            "fresh immobilized control rejected {base:?}"
        );
        let mut corrupted = b.instruction_list.clone();
        corrupted.push(Instruction::SleepTalkCalled(SleepTalkCalledInstruction {
            side_ref: caller,
            move_id: Choices::SPLASH,
        }));
        let bad = render_branch_events(&mut s, &action, &action, &corrupted, false, &ctx());
        assert_eq!(before, s.serialize());
        assert!(bad.is_attribution_unsafe(),"immobilized Sleep Talk never calls Splash; spurious marker accepted {caller:?}: {bad:?}");
    }
}
