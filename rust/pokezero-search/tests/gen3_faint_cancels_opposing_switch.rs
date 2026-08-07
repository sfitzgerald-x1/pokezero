//! Gen 3: a faint cancels the queued action of every still-active Pokemon, switches
//! included.
//!
//! Covers `third_party/poke-engine-gen3-faint-cancels-opposing-switch.patch`, which
//! closes holdout row `19100180/24`.
//!
//! Showdown's `faintMessages` (`sim/battle.ts:2609-2618`) runs
//! `queue.cancelAction(pokemon)` over `getAllActive()` in gen <= 3 singles. Ground
//! truth confirmed against real gen3 Showdown on four seeds by
//! `scripts/gen3_switch_differential.py`, scenarios `faintcancelsopposingswitch` and
//! `faintcancelsopposingswitchcontrol`: p1 lays Spikes, both sides then switch, p2's
//! incoming Pokemon dies on entry, and p1's switch does NOT happen.
//!
//! READ WITH `gen3_pursuit_switch_continues.rs`. That file pins the case where the
//! Pokemon that died IS the switcher, and its switch DOES proceed. The two are one
//! rule, not a rule and an exception: `getAllActive()` filters on `!fainted`, so a
//! Pursuit victim is outside the cancellation set entirely. The engine guard therefore
//! contains no Pursuit branch — it tests "someone newly fainted AND this side is still
//! standing", and the Pursuit case fails the second clause on its own.
//!
//! Both directions had to be pinned before the guard was written. Two earlier attempts
//! implemented one direction each and were withdrawn on their falsifiers: one cancelled
//! forced replacements (78 rows opened to close 1), the other cancelled Pursuit.

use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::MoveChoice;
use poke_engine::instruction::StateInstructions;
use poke_engine::state::{PokemonIndex, State};

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

fn only_branch(branches: Vec<StateInstructions>) -> Vec<String> {
    assert_eq!(
        branches.len(),
        1,
        "expected a single deterministic branch, got {:?}",
        branches
    );
    branches[0]
        .instruction_list
        .iter()
        .map(|instruction| format!("{:?}", instruction))
        .collect()
}

/// Both sides switch. Side two is faster, so its switch resolves first and its incoming
/// Pokemon walks into side two's own Spikes. `incoming_hp` decides whether that kills
/// it, and is the ONLY difference between the two arms.
fn double_switch_into_spikes(incoming_hp: i16) -> Vec<String> {
    let mut state = State::default();

    state.side_two.pokemon[PokemonIndex::P0].speed = 200;
    state.side_one.pokemon[PokemonIndex::P0].speed = 100;
    state.side_two.side_conditions.spikes = 1;

    let incoming = &mut state.side_two.pokemon[PokemonIndex::P1];
    incoming.maxhp = 80; // one Spikes layer deals 80 / 8 = 10
    incoming.hp = incoming_hp;

    only_branch(generate(
        &mut state,
        &MoveChoice::Switch(PokemonIndex::P1),
        &MoveChoice::Switch(PokemonIndex::P1),
    ))
}

/// The incoming Pokemon is at exactly the Spikes damage, so it faints on entry. Side
/// one is still standing, so its queued switch is cancelled — and because the switch
/// never happens, side two is left owing a replacement instead.
#[test]
fn a_faint_on_entry_cancels_the_opposing_queued_switch() {
    assert_eq!(
        double_switch_into_spikes(10),
        vec![
            "Switch SideTwo: P0 -> P1".to_string(),
            "Damage SideTwo: 10".to_string(),
            "ToggleSideTwoForceSwitch".to_string(),
        ],
        "a faint cancels every still-active Pokemon's queued action \
         (sim/battle.ts:2609-2618). Performing `Switch SideOne: P0 -> P1` here is the \
         defect behind holdout 19100180/24: the engine brought in a Pokemon Showdown \
         leaves benched, and charged it hazards it never took."
    );
}

/// Control: the identical ply with the incoming Pokemon able to survive the same
/// Spikes. Nothing faints, so nothing is cancelled and both switches happen. The only
/// difference from the test above is the faint, which is the variable under test.
#[test]
fn without_a_faint_both_switches_happen() {
    assert_eq!(
        double_switch_into_spikes(80),
        vec![
            "Switch SideTwo: P0 -> P1".to_string(),
            "Damage SideTwo: 10".to_string(),
            "Switch SideOne: P0 -> P1".to_string(),
        ],
        "with nothing fainting there is nothing to cancel"
    );
}

/// THE v1 REGRESSION PIN, and the direction that cost the most.
///
/// The first attempt at this mechanism tested `either active is at hp <= 0`. That is
/// also true of a side entering the ply already owing a replacement, whose active is a
/// 0-HP Pokemon awaiting a switch — so it cancelled forced replacements, leaving the
/// wrong Pokemon in and desynchronising every later residual: dev 2 -> 40, holdout
/// 3 -> 42, 78 rows opened to close 1. Only the sweep caught it, because the synthetic
/// double switch that shipped with it had nobody fainting and so never reproduced it.
///
/// Here BOTH sides owe a replacement and side two's replacement dies to Spikes on
/// entry. Side one's replacement must still be performed: it was never a *queued*
/// action in Showdown's sense, it is a fresh request phase, so `cancelAction` cannot
/// reach it. The guard survives this because a 0-HP active that was already down is not
/// *newly* fainted.
///
/// Without this the failure mode is unpinned and only a 200-game sweep would catch its
/// return, which is exactly the situation the fixture layer exists to end.
#[test]
fn a_forced_replacement_is_never_cancelled_even_when_the_other_one_dies_on_entry() {
    let mut state = State::default();

    state.side_one.force_switch = true;
    state.side_two.force_switch = true;
    state.side_one.pokemon[PokemonIndex::P0].hp = 0;
    state.side_two.pokemon[PokemonIndex::P0].hp = 0;
    state.side_two.side_conditions.spikes = 1;

    let incoming = &mut state.side_two.pokemon[PokemonIndex::P1];
    incoming.maxhp = 80;
    incoming.hp = 10; // dies to the 80 / 8 = 10 Spikes hit

    assert_eq!(
        only_branch(generate(
            &mut state,
            &MoveChoice::Switch(PokemonIndex::P1),
            &MoveChoice::Switch(PokemonIndex::P1),
        )),
        vec![
            "ToggleSideTwoForceSwitch".to_string(),
            "Switch SideTwo: P0 -> P1".to_string(),
            "Damage SideTwo: 10".to_string(),
            "ToggleSideOneForceSwitch".to_string(),
            "Switch SideOne: P0 -> P1".to_string(),
            "ToggleSideTwoForceSwitch".to_string(),
        ],
        "side one's forced replacement must still happen. Dropping \
         `Switch SideOne: P0 -> P1` here is the v1 defect that opened 78 rows to close 1."
    );
}
