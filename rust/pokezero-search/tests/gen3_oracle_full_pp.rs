//! High-PP abstraction is intentional in legacy search, not in the full-state oracle.
use poke_engine::engine::abilities::Abilities;
use poke_engine::choices::Choices;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::{MoveChoice, PokemonVolatileStatus};
use poke_engine::instruction::Instruction;
use poke_engine::state::{PokemonMoveIndex, PokemonStatus, State};

fn exercise(pp: i8, pressure: bool) {
    let mut state = State::default();
    for side in [&mut state.side_one, &mut state.side_two] {
        let mon = side.get_active();
        mon.hp = 4000;
        mon.maxhp = 5000;
        mon.replace_move(PokemonMoveIndex::M0, Choices::RECOVER);
        mon.moves[&PokemonMoveIndex::M0].pp = pp;
    }
    if pressure {
        state.side_one.get_active().replace_move(PokemonMoveIndex::M0, Choices::EARTHQUAKE);
        state.side_two.get_active().ability = Abilities::PRESSURE;
    }
    let before = state.serialize();
    let action = MoveChoice::Move(PokemonMoveIndex::M0);
    let branches = generate_instructions_from_move_pair(&mut state, &action, &action, true);
    assert_eq!(state.serialize(), before);
    assert!(!branches.is_empty());
    let mut mass = 0.0;
    for branch in branches {
        assert!(branch.percentage.is_finite() && branch.percentage >= 0.0);
        mass += branch.percentage;
        if branch.percentage == 0.0 { continue; }
        state.apply_instructions(&branch.instruction_list);
        for (index, side) in [&state.side_one, &state.side_two].iter().enumerate() {
            let spends = cfg!(feature = "oracle-full-pp") || pp < 10;
            let raw_cost = if pressure && index == 0 { 2 } else { 1 };
            let cost = if cfg!(feature = "oracle-full-pp") { raw_cost.min(pp.max(0)) } else { raw_cost };
            let expected = if spends { pp - cost } else { pp };
            assert_eq!(side.get_active_immutable().moves[&PokemonMoveIndex::M0].pp, expected,
                "profile={}, pp={pp}, pressure={pressure}, seat={index}",
                cfg!(feature = "oracle-full-pp"));
        }
        state.reverse_instructions(&branch.instruction_list);
        assert_eq!(state.serialize(), before);
    }
    assert!((mass - 100.0).abs() < 0.0001);
}

#[test]
fn high_pp_healing_respects_the_explicit_profile() { exercise(32, false); }

#[test]
fn high_pp_pressure_respects_the_explicit_profile() { exercise(32, true); }

#[test]
fn boundary_ten_pp_respects_the_explicit_profile() { exercise(10, false); }

#[test]
fn low_pp_and_pressure_remain_unchanged() {
    exercise(8, false);
    exercise(8, true);
}

#[test]
fn full_profile_clamps_pressure_at_the_last_pp() { exercise(1, true); }

#[test]
fn full_profile_charges_sleep_talk_caller_not_called_slot() {
    for pp in [1, 8, 32] { for pressure in [false, true] {
        let mut state = State::default();
        for side in [&mut state.side_one, &mut state.side_two] {
            side.get_active().hp = 4000;
            side.get_active().maxhp = 5000;
        }
        let mon = state.side_one.get_active();
        mon.status = PokemonStatus::SLEEP;
        mon.rest_turns = 2;
        mon.replace_move(PokemonMoveIndex::M0, Choices::SLEEPTALK);
        mon.replace_move(PokemonMoveIndex::M1, Choices::EARTHQUAKE);
        mon.moves[&PokemonMoveIndex::M0].pp = pp;
        mon.moves[&PokemonMoveIndex::M1].pp = 32;
        state.side_two.get_active().replace_move(PokemonMoveIndex::M0, Choices::SPLASH);
        if pressure { state.side_two.get_active().ability = Abilities::PRESSURE; }
        let before = state.serialize();
        let action = MoveChoice::Move(PokemonMoveIndex::M0);
        let branches = generate_instructions_from_move_pair(&mut state, &action, &action, true);
        assert_eq!(state.serialize(), before);
        let mut mass = 0.0;
        for branch in branches {
            mass += branch.percentage;
            if branch.percentage <= 0.0 { continue; }
            state.apply_instructions(&branch.instruction_list);
            let mon = state.side_one.get_active_immutable();
            assert_eq!(mon.moves[&PokemonMoveIndex::M0].pp,
                if cfg!(feature = "oracle-full-pp") { pp - 1 } else { pp });
            assert_eq!(mon.moves[&PokemonMoveIndex::M1].pp, 32);
            state.reverse_instructions(&branch.instruction_list);
            assert_eq!(state.serialize(), before);
        }
        assert!((mass - 100.0).abs() < 0.0001);
    }}
}

#[test]
fn fractional_sleep_talk_wakes_on_a_truant_loaf_never_pay_pp() {
    for pp in [8, 32] {
        let mut state = State::default();
        let mon = state.side_one.get_active();
        mon.ability = Abilities::TRUANT;
        mon.status = PokemonStatus::SLEEP;
        mon.rest_turns = 0;
        mon.sleep_turns = 2;
        mon.replace_move(PokemonMoveIndex::M0, Choices::SLEEPTALK);
        mon.replace_move(PokemonMoveIndex::M1, Choices::EARTHQUAKE);
        mon.moves[&PokemonMoveIndex::M0].pp = pp;
        state.side_one.volatile_statuses.insert(PokemonVolatileStatus::TRUANT);
        state.side_two.get_active().replace_move(PokemonMoveIndex::M0, Choices::SPLASH);
        let before = state.serialize();
        let action = MoveChoice::Move(PokemonMoveIndex::M0);
        let branches = generate_instructions_from_move_pair(&mut state, &action, &action, true);
        assert_eq!(state.serialize(), before);
        let (mut woke, mut asleep, mut mass) = (false, false, 0.0);
        for branch in branches {
            mass += branch.percentage;
            if branch.percentage <= 0.0 { continue; }
            state.apply_instructions(&branch.instruction_list);
            let mon = state.side_one.get_active_immutable();
            assert_eq!(mon.moves[&PokemonMoveIndex::M0].pp, pp);
            woke |= mon.status == PokemonStatus::NONE;
            asleep |= mon.status == PokemonStatus::SLEEP;
            state.reverse_instructions(&branch.instruction_list);
            assert_eq!(state.serialize(), before);
        }
        assert!(woke && asleep);
        assert!((mass - 100.0).abs() < 0.001);
    }
}

#[test]
fn fractional_sleep_talk_wakes_pay_only_after_the_remaining_gates() {
    for pp in [8, 32] { for confused in [false, true] {
        let mut state = State::default();
        for side in [&mut state.side_one, &mut state.side_two] {
            side.get_active().hp = 4000;
            side.get_active().maxhp = 5000;
        }
        state.side_one.get_active().speed = 301;
        state.side_two.get_active().speed = 199;
        let mon = state.side_one.get_active();
        mon.status = PokemonStatus::SLEEP;
        mon.rest_turns = 0;
        mon.sleep_turns = 2; // Nonzero wake and stay-asleep mass.
        mon.replace_move(PokemonMoveIndex::M0, Choices::SLEEPTALK);
        mon.replace_move(PokemonMoveIndex::M1, Choices::EARTHQUAKE);
        mon.moves[&PokemonMoveIndex::M0].pp = pp;
        mon.moves[&PokemonMoveIndex::M1].pp = 32;
        if confused {
            state.side_one.volatile_statuses.insert(PokemonVolatileStatus::CONFUSION);
            state.side_one.volatile_status_durations.confusion = 1;
        }
        state.side_two.get_active().replace_move(PokemonMoveIndex::M0, Choices::SPLASH);
        let before = state.serialize();
        let action = MoveChoice::Move(PokemonMoveIndex::M0);
        let branches = generate_instructions_from_move_pair(&mut state, &action, &action, true);
        assert_eq!(state.serialize(), before);
        let mut mass = 0.0;
        let (mut woke, mut asleep, mut blocked_wake) = (false, false, false);
        for branch in branches {
            mass += branch.percentage;
            if branch.percentage <= 0.0 { continue; }
            let blocked = branch.instruction_list.iter().any(|instruction| matches!(instruction,
                Instruction::Damage(d) if d.side_ref == poke_engine::state::SideReference::SideOne));
            state.apply_instructions(&branch.instruction_list);
            let mon = state.side_one.get_active_immutable();
            woke |= mon.status == PokemonStatus::NONE;
            asleep |= mon.status == PokemonStatus::SLEEP;
            blocked_wake |= blocked && mon.status == PokemonStatus::NONE;
            if cfg!(feature = "oracle-full-pp") {
                assert_eq!(mon.moves[&PokemonMoveIndex::M0].pp, pp - i8::from(!blocked),
                    "confused={confused}, branch={branch:?}");
            }
            assert_eq!(mon.moves[&PokemonMoveIndex::M1].pp, 32);
            state.reverse_instructions(&branch.instruction_list);
            assert_eq!(state.serialize(), before);
        }
        assert!(woke && asleep);
        if cfg!(feature = "oracle-full-pp") && confused { assert!(blocked_wake); }
        assert!((mass - 100.0).abs() < 0.001);
    }}
}
