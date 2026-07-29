//! Gen 3 Sleep Clause / Rest-provenance pins, asserted directly against the
//! vendored gen3-patched poke-engine (`third_party/poke-engine-src/`).
//!
//! Ground truth for every expectation here was read off the **real** Node
//! Showdown simulator through `scripts/gen3_switch_differential.py`
//! (`hypnosisrestclause` and its control); that script is the gate and this file
//! is the engine-contract pin.
//!
//! WHAT THESE GUARD. gen3's Sleep Clause Mod exempts a REST sleep: `rulesets.ts`
//! walks the target's party and blocks a new sleep only for a sleeper whose
//! `statusState.source` is NOT its own ally, so a Pokemon asleep from its own
//! Rest leaves its side a legal sleep target. The engine spells that exemption
//! as a single condition — `has_alive_non_rested_sleeping_pkmn` counts a sleeper
//! only while `rest_turns == 0` (`gen3/state.rs`). Provenance therefore lives
//! entirely in `rest_turns`, and a world that builds a Rest-sleeper with a zeroed
//! counter does not merely mis-time its wake-up: it re-arms a clause the real
//! battle does not have. Both halves of the Python fix exist to keep that from
//! happening, and both are meaningless if these engine semantics move.
//!
//! The bench is the whole point. An ACTIVE Rest-sleeper reveals itself on the
//! next turn either way, but a benched one is invisible to everything except
//! this clause — which is exactly why the public tracker follows it off the field
//! and why these pins put the sleeper on the bench.
//!
//! No patch of ours is involved: this is upstream gen3 behaviour that the fix
//! DEPENDS on, pinned so a version bump or a re-vendor that changes it fails here
//! rather than silently unbuilding the world constructor's assumption.

use poke_engine::choices::Choices;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::MoveChoice;
use poke_engine::instruction::{Instruction, StateInstructions};
use poke_engine::state::{
    PokemonIndex, PokemonMoveIndex, PokemonStatus, SideReference, State,
};

/// Rest sets the counter to 3; each move ATTEMPT decrements it and the Pokemon
/// wakes at 1. The Python build half rebuilds this as `3 - k` from the public
/// attempt count, so k = 0, 1, 2 maps to 3, 2, 1 here.
const REST_TURNS_ON_REST: i8 = 3;

/// `generate_instructions_from_move_pair` must leave `state` untouched.
fn generate(
    state: &mut State,
    side_one: &MoveChoice,
    side_two: &MoveChoice,
) -> Vec<StateInstructions> {
    let before = format!("{:?}", state);
    let instructions = generate_instructions_from_move_pair(state, side_one, side_two, false);
    assert_eq!(before, format!("{:?}", state), "state was mutated");
    instructions
}

fn only_branch(instructions: Vec<StateInstructions>) -> Vec<Instruction> {
    assert_eq!(
        instructions.len(),
        1,
        "expected a single deterministic branch, got {:?}",
        instructions
    );
    instructions.into_iter().next().unwrap().instruction_list
}

fn assert_reverts_cleanly(state: &mut State, list: &Vec<Instruction>) {
    let before = format!("{:?}", state);
    state.apply_instructions(list);
    state.reverse_instructions(list);
    assert_eq!(before, format!("{:?}", state), "instructions did not revert");
}

/// Does any branch put side two's benched-or-active target to sleep?
fn applies_sleep(branches: &[StateInstructions], side_ref: SideReference) -> bool {
    branches.iter().any(|branch| {
        branch.instruction_list.iter().any(|instruction| match instruction {
            Instruction::ChangeStatus(change) => {
                change.side_ref == side_ref && change.new_status == PokemonStatus::SLEEP
            }
            _ => false,
        })
    })
}

fn wakes_up(list: &[Instruction], side_ref: SideReference) -> bool {
    list.iter().any(|instruction| match instruction {
        Instruction::ChangeStatus(change) => {
            change.side_ref == side_ref
                && change.old_status == PokemonStatus::SLEEP
                && change.new_status == PokemonStatus::NONE
        }
        _ => false,
    })
}

fn decrements_rest(list: &[Instruction], side_ref: SideReference) -> usize {
    list.iter()
        .filter(|instruction| match instruction {
            Instruction::DecrementRestTurns(decrement) => decrement.side_ref == side_ref,
            _ => false,
        })
        .count()
}

/// Side one carries Hypnosis. Side two leads with a healthy Pokemon and keeps a
/// SLEEPING one on the BENCH — `rest_turns` is the only thing that distinguishes
/// the two arms, which is precisely the claim under test.
fn hypnosis_into_benched_sleeper(bench_rest_turns: i8) -> State {
    let mut state = State::default();

    let attacker = state.side_one.get_active();
    attacker.replace_move(PokemonMoveIndex::M0, Choices::HYPNOSIS);

    // The lead: awake, and the target of the Hypnosis.
    let target = state.side_two.get_active();
    target.status = PokemonStatus::NONE;
    target.replace_move(PokemonMoveIndex::M0, Choices::SPLASH);

    // The bench: asleep and alive, so it is counted by the clause query at all.
    let benched = &mut state.side_two.pokemon[PokemonIndex::P1];
    benched.status = PokemonStatus::SLEEP;
    benched.rest_turns = bench_rest_turns;

    state
}

fn hypnosis_branches(bench_rest_turns: i8) -> Vec<StateInstructions> {
    let mut state = hypnosis_into_benched_sleeper(bench_rest_turns);
    generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    )
}

// ---------------------------------------------------------------------------
// The clause: exempt on the bench vs armed on the bench
// ---------------------------------------------------------------------------

/// A benched REST-sleeper does NOT engage the clause: Hypnosis resolves as an
/// ordinary 60%-accurate sleep move and fans out into a hit branch and a miss
/// branch, exactly as it would against a side with nobody asleep at all.
#[test]
fn a_benched_rest_sleeper_leaves_its_side_a_legal_sleep_target() {
    let branches = hypnosis_branches(REST_TURNS_ON_REST);

    assert!(
        applies_sleep(&branches, SideReference::SideTwo),
        "a Rest sleep is exempt from gen3's Sleep Clause, so Hypnosis must be able \
         to land; got {:?}",
        branches
    );

    let mut percentages: Vec<f32> = branches.iter().map(|branch| branch.percentage).collect();
    percentages.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert_eq!(
        percentages.len(),
        2,
        "expected Hypnosis's accuracy fan-out (hit + miss), got {:?}",
        branches
    );
    assert!(
        (percentages[0] - 60.0).abs() < 1e-3 && (percentages[1] - 40.0).abs() < 1e-3,
        "expected a 60/40 accuracy fan-out, got {:?}",
        percentages
    );
}

/// The control, and the reason the exemption has to be spelled out rather than
/// assumed: the SAME benched Pokemon, asleep with `rest_turns == 0` (an
/// opponent-induced sleep), does arm the clause and Hypnosis can no longer land.
#[test]
fn a_benched_induced_sleeper_arms_the_clause() {
    let branches = hypnosis_branches(0);

    assert!(
        !applies_sleep(&branches, SideReference::SideTwo),
        "an induced sleep on the bench must block a second sleep; got {:?}",
        branches
    );
}

/// The pin that makes the two arms above a measurement of `rest_turns` and not of
/// two different fixtures: the exemption tracks the counter across its whole
/// reachable range. k = 0, 1, 2 (rest_turns 3, 2, 1) all exempt; only 0 arms it.
#[test]
fn every_reachable_rest_counter_value_exempts_the_clause() {
    for rest_turns in 1..=REST_TURNS_ON_REST {
        assert!(
            applies_sleep(&hypnosis_branches(rest_turns), SideReference::SideTwo),
            "rest_turns={} must still read as a Rest sleep and exempt the clause",
            rest_turns
        );
    }
}

// ---------------------------------------------------------------------------
// The arithmetic's endpoints: what `3 - k` actually buys
// ---------------------------------------------------------------------------

/// k = 2 builds `rest_turns = 1`, and 1 is the LAST turn of the Rest: the mon
/// wakes on its next attempt and acts. Build it at 0 instead (today's behaviour
/// for an unannotated sleeper) and the engine reads an induced sleep with an
/// unknown clock; build it at 3 and search waits two turns too long.
#[test]
fn a_rest_counter_of_one_wakes_the_pokemon_on_its_next_attempt() {
    let mut state = State::default();
    let active = state.side_one.get_active();
    active.status = PokemonStatus::SLEEP;
    active.rest_turns = 1;
    active.replace_move(PokemonMoveIndex::M0, Choices::SPLASH);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::SPLASH);

    let list = only_branch(generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    ));

    assert!(
        wakes_up(&list, SideReference::SideOne),
        "rest_turns=1 is the last turn of the Rest; got {:?}",
        list
    );
    assert_eq!(
        decrements_rest(&list, SideReference::SideOne),
        1,
        "the waking attempt must also spend the counter; got {:?}",
        list
    );
    assert_reverts_cleanly(&mut state, &list);
}

/// The counterpart: a mid-Rest counter does NOT wake, it spends one attempt. This
/// is the tick the public `|cant|SLOT|slp` line counts, so it is the tick `k`
/// tracks — one engine attempt per public line, which is what makes `3 - k` an
/// identity rather than an estimate.
#[test]
fn a_mid_rest_counter_spends_one_attempt_without_waking() {
    for rest_turns in 2..=REST_TURNS_ON_REST {
        let mut state = State::default();
        let active = state.side_one.get_active();
        active.status = PokemonStatus::SLEEP;
        active.rest_turns = rest_turns;
        active.replace_move(PokemonMoveIndex::M0, Choices::SPLASH);
        state
            .side_two
            .get_active()
            .replace_move(PokemonMoveIndex::M0, Choices::SPLASH);

        let list = only_branch(generate(
            &mut state,
            &MoveChoice::Move(PokemonMoveIndex::M0),
            &MoveChoice::Move(PokemonMoveIndex::M0),
        ));

        assert!(
            !wakes_up(&list, SideReference::SideOne),
            "rest_turns={} must stay asleep; got {:?}",
            rest_turns,
            list
        );
        assert_eq!(
            decrements_rest(&list, SideReference::SideOne),
            1,
            "rest_turns={} must spend exactly one attempt; got {:?}",
            rest_turns,
            list
        );
        assert_reverts_cleanly(&mut state, &list);
    }
}

// ---------------------------------------------------------------------------
// Provenance survives the bench
// ---------------------------------------------------------------------------

/// Switching does not touch `rest_turns` — neither the departing Pokemon's nor
/// the arriving one's. This is the engine-side half of why the public tracker
/// counts ATTEMPTS and not elapsed turns: the counter is frozen while the mon is
/// off the field, so wall-clock turns would over-count its progress and hand
/// search a Rest that is closer to waking than it is (and, at the boundary, one
/// that has already lapsed into a clause-arming induced sleep).
#[test]
fn a_switch_preserves_the_rest_counter_on_both_sides_of_the_door() {
    let mut state = State::default();

    // Outgoing: mid-Rest and about to leave the field.
    let leaving = state.side_one.get_active();
    leaving.status = PokemonStatus::SLEEP;
    leaving.rest_turns = 2;

    // Incoming: also mid-Rest, at a different point in its own counter.
    let arriving = &mut state.side_one.pokemon[PokemonIndex::P1];
    arriving.status = PokemonStatus::SLEEP;
    arriving.rest_turns = REST_TURNS_ON_REST;

    let list = only_branch(generate(
        &mut state,
        &MoveChoice::Switch(PokemonIndex::P1),
        &MoveChoice::None,
    ));

    assert_eq!(
        decrements_rest(&list, SideReference::SideOne),
        0,
        "a switch is not a move attempt and must not tick any Rest; got {:?}",
        list
    );

    state.apply_instructions(&list);
    assert_eq!(
        state.side_one.pokemon[PokemonIndex::P0].rest_turns, 2,
        "the departing Pokemon's Rest must ride the bench untouched"
    );
    assert_eq!(
        state.side_one.pokemon[PokemonIndex::P0].status,
        PokemonStatus::SLEEP,
        "and so must its sleep"
    );
    assert_eq!(
        state.side_one.pokemon[PokemonIndex::P1].rest_turns, REST_TURNS_ON_REST,
        "the arriving Pokemon keeps its own counter"
    );
    state.reverse_instructions(&list);
}

/// And the consequence that matters for the clause: a Rest-sleeper that switches
/// OUT is still exempt from the bench. Without this, the fix would be correct
/// only for the case it was never needed in.
#[test]
fn a_rest_sleeper_that_switches_out_is_still_exempt_from_the_bench() {
    let mut state = State::default();
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::HYPNOSIS);

    // Side two's ACTIVE is the Rest-sleeper; it will leave and the awake bench
    // Pokemon takes its place, which is the position the differential scripts.
    let resting = state.side_two.get_active();
    resting.status = PokemonStatus::SLEEP;
    resting.rest_turns = REST_TURNS_ON_REST;

    let replacement = &mut state.side_two.pokemon[PokemonIndex::P1];
    replacement.status = PokemonStatus::NONE;

    let switch = only_branch(generate(
        &mut state,
        &MoveChoice::None,
        &MoveChoice::Switch(PokemonIndex::P1),
    ));
    state.apply_instructions(&switch);

    let branches = generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::None,
    );
    assert!(
        applies_sleep(&branches, SideReference::SideTwo),
        "the Rest-sleeper is now benched and must still exempt the clause; got {:?}",
        branches
    );
}
