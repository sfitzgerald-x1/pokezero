//! Pinned Gen3 residual27 clock boundaries, not full Truant/Trace fidelity.
//! Expectations correspond to tests/gpu_truant_clock_oracle.cjs. Root ACT/LOAF
//! is explicit input; every post-start switch uses the real switch hook.
use poke_engine::choices::Choices;
use poke_engine::engine::abilities::Abilities;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::{MoveChoice, PokemonVolatileStatus};
use poke_engine::instruction::StateInstructions;
use poke_engine::state::{PokemonIndex, PokemonMoveIndex, PokemonStatus, State};

fn base() -> State {
    let mut s = State::default();
    s.side_one.get_active().ability = Abilities::TRUANT;
    s.side_one.get_active().hp = 100;
    s.side_one.get_active().maxhp = 1000;
    s.side_one.get_active().speed = 301;
    s.side_two.get_active().speed = 199;
    s.side_one.get_active().replace_move(PokemonMoveIndex::M0, Choices::SLACKOFF);
    s.side_two.get_active().replace_move(PokemonMoveIndex::M0, Choices::SPLASH);
    s.side_one.get_active().moves[&PokemonMoveIndex::M0].pp = 8;
    s.side_two.get_active().moves[&PokemonMoveIndex::M0].pp = 8;
    let incoming = &mut s.side_one.pokemon[PokemonIndex::P1];
    incoming.ability = Abilities::TRUANT;
    incoming.hp = 100;
    incoming.maxhp = 1000;
    incoming.replace_move(PokemonMoveIndex::M0, Choices::SLACKOFF);
    incoming.moves[&PokemonMoveIndex::M0].pp = 8;
    s
}

fn advance(s: &mut State, a: MoveChoice, b: MoveChoice) -> Vec<StateInstructions> {
    let before = format!("{:?}", s);
    let branches = generate_instructions_from_move_pair(s, &a, &b, false);
    assert_eq!(before, format!("{:?}", s), "generation mutated input");
    assert!((branches.iter().map(|x| x.percentage).sum::<f32>() - 100.0).abs() < 1e-4);
    for branch in &branches {
        s.apply_instructions(&branch.instruction_list);
        s.reverse_instructions(&branch.instruction_list);
        assert_eq!(before, format!("{:?}", s), "branch failed reversibility");
    }
    branches
}

fn one(s: &mut State, a: MoveChoice, b: MoveChoice) {
    let branches = advance(s, a, b);
    let positive: Vec<_> = branches.iter().filter(|x| x.percentage > 0.0).collect();
    assert_eq!(positive.len(), 1, "unexpected full-state branching: {:?}", branches);
    assert!((positive[0].percentage - 100.0).abs() < 1e-4);
    s.apply_instructions(&positive[0].instruction_list);
}

fn action() -> MoveChoice { MoveChoice::Move(PokemonMoveIndex::M0) }
fn loaf(s: &State) -> bool {
    s.side_one.volatile_statuses.contains(&PokemonVolatileStatus::TRUANT)
}

#[test]
fn slackoff_alternates_without_any_damaging_hit() {
    let mut s = base();
    for (hp, pp, phase) in [(600, 7, true), (600, 7, false), (1000, 6, true), (1000, 6, false)] {
        one(&mut s, action(), action());
        assert_eq!(s.side_one.get_active_immutable().hp, hp);
        assert_eq!(s.side_one.get_active_immutable().moves[&PokemonMoveIndex::M0].pp, pp);
        assert_eq!(loaf(&s), phase);
    }
}

#[test]
fn sleep_and_confusion_on_loaf_keep_beforemove_priority() {
    let mut s = base();
    s.side_one.get_active().status = PokemonStatus::SLEEP;
    s.side_one.get_active().rest_turns = 3;
    s.side_one.volatile_statuses.insert(PokemonVolatileStatus::TRUANT);
    s.side_one.volatile_statuses.insert(PokemonVolatileStatus::CONFUSION);
    s.side_one.volatile_status_durations.confusion = 2;
    one(&mut s, action(), action());
    assert_eq!(s.side_one.get_active_immutable().rest_turns, 2);
    assert_eq!(s.side_one.get_active_immutable().moves[&PokemonMoveIndex::M0].pp, 8);
    assert_eq!(s.side_one.volatile_status_durations.confusion, 2);
    assert!(!loaf(&s));
}

#[test]
fn every_paralysis_outcome_flips_the_same_residual_clock() {
    let mut s = base();
    s.side_one.get_active().status = PokemonStatus::PARALYZE;
    for branch in advance(&mut s, action(), action()) {
        if branch.percentage <= 0.0 { continue; }
        s.apply_instructions(&branch.instruction_list);
        assert!(loaf(&s));
        s.reverse_instructions(&branch.instruction_list);
    }
    s.side_one.volatile_statuses.insert(PokemonVolatileStatus::TRUANT);
    // Truant precedes paralysis: the loaf boundary itself has no par branch.
    one(&mut s, action(), action());
    assert!(!loaf(&s));
    assert_eq!(s.side_one.get_active_immutable().moves[&PokemonMoveIndex::M0].pp, 8);
}

#[test]
fn injected_recharge_is_before_truant_but_does_not_pause_its_clock() {
    let mut s = base();
    s.side_one.volatile_statuses.insert(PokemonVolatileStatus::MUSTRECHARGE);
    one(&mut s, MoveChoice::None, action());
    assert!(!s.side_one.volatile_statuses.contains(&PokemonVolatileStatus::MUSTRECHARGE));
    assert!(loaf(&s));
    one(&mut s, action(), action());
    assert_eq!(s.side_one.get_active_immutable().hp, 100);
    assert!(!loaf(&s));
    one(&mut s, action(), action());
    assert_eq!(s.side_one.get_active_immutable().hp, 600);
}

#[test]
fn switch_timing_decides_whether_incoming_clock_gets_the_residual() {
    for boundary in ["ordinary", "pending", "postresidual"] {
        let mut s = base();
        s.side_one.get_active().ability = Abilities::NONE;
        if boundary != "ordinary" { s.side_one.get_active().hp = 0; }
        s.side_one.force_switch = boundary == "pending";
        one(&mut s, MoveChoice::Switch(PokemonIndex::P1),
            if boundary == "ordinary" { action() } else { MoveChoice::None });
        assert_eq!(loaf(&s), boundary == "postresidual", "{}", boundary);
        one(&mut s, action(), action());
        assert_eq!(s.side_one.get_active_immutable().hp,
            if boundary == "postresidual" { 100 } else { 600 }, "{}", boundary);
    }
}
