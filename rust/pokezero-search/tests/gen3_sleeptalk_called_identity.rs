//! Bundle-derived transition-only regression. This binary has its own exact-roll latch.
use poke_engine::choices::Choices;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::MoveChoice;
use poke_engine::instruction::Instruction;
use poke_engine::state::{PokemonStatus, SideReference, State};
use pokezero_search::events::{render_branch_events, EventContext};

fn move_choice_from_str(name: &str, state: &State, side: SideReference) -> Option<MoveChoice> {
    MoveChoice::from_string(name, state.get_side_immutable(&side))
}

fn recovered_sleep_talk_root() -> (State, EventContext, serde_json::Value) {
    std::env::set_var("POKEZERO_ENUMERATE_ROLLS", "1");
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/native_refusals/sleeptalk_empty_callee_r2.json"
    ))
    .unwrap();
    let state = State::deserialize(fixture["root_state"].as_str().unwrap());
    let ctx = EventContext::from_json(&fixture["ctx"].to_string()).unwrap();
    (state, ctx, fixture)
}

#[test]
fn bundle_derived_sleep_talk_history_preserves_aggregate_state_mass_both_seats() {
    use sha2::{Digest, Sha256};
    use std::collections::BTreeMap;
    for mirrored in [false, true] {
        let (mut state, mut ctx, fixture) = recovered_sleep_talk_root();
        if mirrored {
            std::mem::swap(&mut state.side_one, &mut state.side_two);
            ctx.species.swap(0, 1);
        }
        let original = state.serialize();
        let (a, b) = if mirrored {
            ("sleeptalk", "murkrow")
        } else {
            ("murkrow", "sleeptalk")
        };
        let s1 = move_choice_from_str(a, &state, SideReference::SideOne).unwrap();
        let s2 = move_choice_from_str(b, &state, SideReference::SideTwo).unwrap();
        let branches = generate_instructions_from_move_pair(&mut state, &s1, &s2, true);
        assert_eq!(branches.len(), 19);
        assert_eq!(state.serialize(), original);
        let mut state_mass = BTreeMap::<String, f64>::new();
        let mut call_mass = BTreeMap::<String, f64>::new();
        for branch in branches {
            let markers: Vec<_> = branch
                .instruction_list
                .iter()
                .filter_map(|i| match i {
                    Instruction::SleepTalkCalled(m) => Some(m),
                    _ => None,
                })
                .collect();
            assert_eq!(markers.len(), 1);
            let marker = markers[0];
            assert_eq!(
                marker.side_ref,
                if mirrored {
                    SideReference::SideOne
                } else {
                    SideReference::SideTwo
                }
            );
            state.apply_one_instruction(&Instruction::SleepTalkCalled(marker.clone()));
            assert_eq!(state.serialize(), original, "marker must be a no-op");
            state.reverse_one_instruction(&Instruction::SleepTalkCalled(marker.clone()));
            assert_eq!(
                state.serialize(),
                original,
                "marker inverse must be a no-op"
            );
            let rendered =
                render_branch_events(&mut state, &s1, &s2, &branch.instruction_list, true, &ctx);
            assert!(rendered.lossy.is_empty(), "{:?}", rendered.lossy);
            assert!(
                !rendered.is_attribution_unsafe(),
                "{:?}",
                rendered.attribution_unsafe
            );
            let callee = format!("{:?}", marker.move_id).to_ascii_lowercase();
            let calls: Vec<_> = rendered
                .lines
                .iter()
                .filter(|line| line.contains("[from] Sleep Talk"))
                .collect();
            assert_eq!(calls.len(), 1, "{:?}", rendered.lines);
            assert!(
                calls[0].contains(&format!("|{callee}|")),
                "{:?}",
                rendered.lines
            );
            *call_mass
                .entry(format!("{:?}", marker.move_id))
                .or_default() += branch.percentage as f64;
            assert_eq!(state.serialize(), original, "renderer must restore state");
            state.apply_instructions(&branch.instruction_list);
            let digest = format!("{:x}", Sha256::digest(state.serialize().as_bytes()));
            *state_mass.entry(digest).or_default() += branch.percentage as f64;
            state.reverse_instructions(&branch.instruction_list);
            assert_eq!(state.serialize(), original);
        }
        for id in ["EARTHQUAKE", "REST", "DOUBLEEDGE"] {
            assert!(
                (call_mass[id] - 100.0 / 3.0).abs() < 0.0001,
                "{call_mass:?}"
            );
        }
        if !mirrored {
            let expected = fixture["original_state_mass_percent"].as_object().unwrap();
            assert_eq!(state_mass.len(), expected.len());
            for (digest, mass) in state_mass {
                assert!((mass - expected[&digest].as_f64().unwrap()).abs() < 0.0001);
            }
        }
    }
}

#[test]
fn sleep_talk_lineage_is_required_and_malformed_lineage_still_refuses() {
    let (mut state, ctx, _) = recovered_sleep_talk_root();
    let s1 = move_choice_from_str("murkrow", &state, SideReference::SideOne).unwrap();
    let s2 = move_choice_from_str("sleeptalk", &state, SideReference::SideTwo).unwrap();
    let branches = generate_instructions_from_move_pair(&mut state, &s1, &s2, true);
    let base = &branches[0].instruction_list;
    let index = base
        .iter()
        .position(|i| matches!(i, Instruction::SleepTalkCalled(_)))
        .unwrap();
    for corruption in 0..4 {
        let mut broken = base.clone();
        match corruption {
            0 => {
                broken.remove(index);
            }
            1 => {
                let duplicate = broken[index].clone();
                broken.insert(index, duplicate);
            }
            2 => {
                if let Instruction::SleepTalkCalled(m) = &mut broken[index] {
                    m.side_ref = SideReference::SideOne;
                }
            }
            _ => {
                if let Instruction::SleepTalkCalled(m) = &mut broken[index] {
                    m.move_id = Choices::SPLASH;
                }
            }
        }
        let rendered = render_branch_events(&mut state, &s1, &s2, &broken, true, &ctx);
        assert!(
            rendered.is_attribution_unsafe(),
            "corruption {corruption}: {:?}",
            rendered.lines
        );
    }
    state.side_two.get_active().status = PokemonStatus::NONE;
    state.side_two.get_active().rest_turns = 0;
    state.side_two.get_active().rest_sleep_pending_refund = 0;
    let awake = generate_instructions_from_move_pair(&mut state, &s1, &s2, true);
    assert!(!awake.is_empty());
    assert!(awake.iter().all(|b| !b
        .instruction_list
        .iter()
        .any(|i| matches!(i, Instruction::SleepTalkCalled(_)))));
}
