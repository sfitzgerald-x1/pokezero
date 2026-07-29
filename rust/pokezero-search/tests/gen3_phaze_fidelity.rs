//! Gen 3 phazing (Whirlwind / Roar) fidelity pins, asserted directly against the
//! vendored gen3-patched poke-engine (`third_party/poke-engine-src/`).
//!
//! Every expectation here was read off the **real** Node Showdown simulator
//! driven through `scripts/gen3_switch_differential.py`, which is the
//! ground-truth gate; this file is the engine-contract pin.
//!
//! What it guards (`poke-engine-gen3-phaze-protect.patch`) and what it merely
//! records as already-correct:
//!
//! * Protect BLOCKS a phaze in gen3 — the fix. Gen 3 inherits gen4's override
//!   (`flags: { protect: 1, mirror: 1, bypasssub: 1, metronome: 1 }`), and
//!   upstream carried no protect flag at all, so Whirlwind dragged straight
//!   through a Protect.
//! * A Substitute does NOT block it (`bypasssub`), the fan-out is uniform over
//!   the alive reserve, entry hazards hit the dragged-in Pokemon, and the
//!   outgoing Pokemon's boosts are cleared. All four were already correct;
//!   they are pinned because the fix edits the same flag set they depend on.

use poke_engine::choices::Choices;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::{MoveChoice, PokemonVolatileStatus};
use poke_engine::instruction::{Instruction, StateInstructions};
use poke_engine::state::{PokemonIndex, PokemonMoveIndex, SideReference, State};

const PHAZE_MOVES: [Choices; 2] = [Choices::WHIRLWIND, Choices::ROAR];

fn generate(
    state: &mut State,
    side_one: &MoveChoice,
    side_two: &MoveChoice,
) -> Vec<StateInstructions> {
    let before = format!("{:?}", state);
    let instructions = generate_instructions_from_move_pair(state, side_one, side_two, false);
    assert_eq!(before, format!("{:?}", state), "state was mutated");
    for branch in &instructions {
        let mut probe = state.clone();
        let snapshot = format!("{:?}", probe);
        probe.apply_instructions(&branch.instruction_list);
        probe.reverse_instructions(&branch.instruction_list);
        assert_eq!(snapshot, format!("{:?}", probe), "branch did not revert");
    }
    instructions
}

fn switch_targets(instructions: &[StateInstructions]) -> Vec<PokemonIndex> {
    instructions
        .iter()
        .filter_map(|branch| {
            branch
                .instruction_list
                .iter()
                .find_map(|instruction| match instruction {
                    Instruction::Switch(switch) if switch.side_ref == SideReference::SideOne => {
                        Some(switch.next_index)
                    }
                    _ => None,
                })
        })
        .collect()
}

/// Side two phazes with `phaze_move` on M0; side one answers with `defender_move`.
fn phaze_state(phaze_move: Choices, defender_move: Choices) -> State {
    let mut state = State::default();
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, phaze_move);
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, defender_move);
    state
}

fn phaze(state: &mut State) -> Vec<StateInstructions> {
    generate(
        state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    )
}

// ---------------------------------------------------------------------------
// The fix: Protect blocks a phaze
// ---------------------------------------------------------------------------

/// Showdown gen3 ground truth (`scripts/gen3_switch_differential.py::whirlwindprotect`):
/// `|move|p1a: Skarmory|Whirlwind|p2a: Snorlax` followed by
/// `|-activate|p2a: Snorlax|Protect` and NO `|drag|`.
///
/// Upstream gave the phazing moves no protect flag, so `before_move` never
/// reached `remove_effects_for_protect()` — which is what clears `flags.drag` —
/// and the target was dragged out through its own Protect. Protect is on 43 gen3
/// randbats species and the pool's Whirlwind user (Skarmory) carries both.
#[test]
fn protect_blocks_a_phaze() {
    for phaze_move in PHAZE_MOVES {
        let mut state = phaze_state(phaze_move, Choices::PROTECT);
        // The Protect user is slower than the -6 priority phaze either way, but
        // make the ordering explicit.
        state.side_one.get_active().speed = 500;
        let instructions = phaze(&mut state);
        assert!(
            switch_targets(&instructions).is_empty(),
            "{:?} must not drag through Protect: {:?}",
            phaze_move,
            instructions
        );
    }
}

// ---------------------------------------------------------------------------
// Already-correct behaviour, pinned because the fix edits the same flag set
// ---------------------------------------------------------------------------

/// The no-regression direction, and the reason the fix is one gated flag rather
/// than a guard in the drag path: an ORDINARY phaze turn must still drag. The
/// same `flags.protect` that `before_move` consults to block the move is the
/// flag every other phaze turn leaves alone, so a fix that over-reached here
/// would silently delete phazing from the format.
#[test]
fn an_unprotected_phaze_still_drags() {
    for phaze_move in PHAZE_MOVES {
        let mut state = phaze_state(phaze_move, Choices::SPLASH);
        let instructions = phaze(&mut state);
        assert_eq!(
            switch_targets(&instructions).len(),
            5,
            "{:?} must still drag when the target did not Protect: {:?}",
            phaze_move,
            instructions
        );
    }
}

/// Protect is a SINGLE-TURN volatile, so the turn after it lapses the phaze
/// connects again. Guards against the flag being read as a permanent immunity.
#[test]
fn a_phaze_connects_the_turn_after_protect_lapses() {
    let mut state = phaze_state(Choices::WHIRLWIND, Choices::PROTECT);
    state.side_one.get_active().speed = 500;

    let blocked = phaze(&mut state);
    assert!(switch_targets(&blocked).is_empty(), "{:?}", blocked);
    // Take the branch forward, then let the Protect expire.
    state.apply_instructions(&blocked[0].instruction_list);
    state
        .side_one
        .volatile_statuses
        .remove(&PokemonVolatileStatus::PROTECT);

    let connects = phaze(&mut state);
    assert_eq!(
        switch_targets(&connects).len(),
        5,
        "the phaze connects once Protect is gone: {:?}",
        connects
    );
}

/// `bypasssub` is in gen3's flag set, so a Substitute does NOT stop a phaze.
/// Verified against the sim: the drag line fires straight through the sub.
#[test]
fn a_substitute_does_not_block_a_phaze() {
    for phaze_move in PHAZE_MOVES {
        let mut state = phaze_state(phaze_move, Choices::SPLASH);
        state
            .side_one
            .volatile_statuses
            .insert(PokemonVolatileStatus::SUBSTITUTE);
        let instructions = phaze(&mut state);
        assert_eq!(
            switch_targets(&instructions).len(),
            5,
            "{:?} must drag through a Substitute: {:?}",
            phaze_move,
            instructions
        );
    }
}

/// Showdown picks the replacement with `this.sample(possibleSwitches)` — uniform
/// over the target's alive, non-active party members. The engine fans out one
/// branch per candidate at equal probability, which is the search-tree spelling
/// of the same thing.
#[test]
fn a_phaze_fans_out_uniformly_over_the_alive_reserve() {
    let mut state = phaze_state(Choices::WHIRLWIND, Choices::SPLASH);
    // Faint two of the five reserves: the fan-out must shrink to match.
    state.side_one.pokemon[PokemonIndex::P3].hp = 0;
    state.side_one.pokemon[PokemonIndex::P4].hp = 0;

    let instructions = phaze(&mut state);
    let targets = switch_targets(&instructions);
    assert_eq!(
        targets,
        vec![PokemonIndex::P1, PokemonIndex::P2, PokemonIndex::P5],
        "only living, non-active reserves are draggable: {:?}",
        instructions
    );
    for branch in &instructions {
        assert!(
            (branch.percentage - 100.0 / 3.0).abs() < 1e-3,
            "uniform over 3 candidates, got {}: {:?}",
            branch.percentage,
            instructions
        );
    }
}

/// The dragged-in Pokemon eats entry hazards, exactly as a chosen switch does —
/// this is the component the strict matcher reads as `spikes_entry_damage`.
#[test]
fn the_dragged_in_pokemon_takes_entry_hazards() {
    let mut state = phaze_state(Choices::WHIRLWIND, Choices::SPLASH);
    state.side_one.side_conditions.spikes = 1;
    for index in [
        PokemonIndex::P1,
        PokemonIndex::P2,
        PokemonIndex::P3,
        PokemonIndex::P4,
        PokemonIndex::P5,
    ] {
        let mon = &mut state.side_one.pokemon[index];
        mon.maxhp = 266;
        mon.hp = 266;
    }

    let instructions = phaze(&mut state);
    assert_eq!(switch_targets(&instructions).len(), 5);
    for branch in &instructions {
        let hazard = branch.instruction_list.iter().any(|instruction| {
            matches!(instruction, Instruction::Damage(damage)
                if damage.side_ref == SideReference::SideOne && damage.damage_amount == 33)
        });
        assert!(
            hazard,
            "every dragged-in Pokemon takes floor(266/8) = 33: {:?}",
            branch.instruction_list
        );
    }
}

/// A phaze is an ordinary switch-out for the Pokemon leaving, so its boosts go
/// with it (`Pokemon.clearVolatile()` on the way out).
#[test]
fn a_phaze_clears_the_outgoing_boosts() {
    let mut state = phaze_state(Choices::WHIRLWIND, Choices::SPLASH);
    state.side_one.attack_boost = 2;
    state.side_one.speed_boost = -1;

    let instructions = phaze(&mut state);
    for branch in &instructions {
        let mut probe = state.clone();
        probe.apply_instructions(&branch.instruction_list);
        assert_eq!(
            probe.side_one.attack_boost, 0,
            "{:?}",
            branch.instruction_list
        );
        assert_eq!(
            probe.side_one.speed_boost, 0,
            "{:?}",
            branch.instruction_list
        );
    }
}

/// With nothing left to drag in, the phaze simply does nothing — and must not
/// emit a switch to the active itself or to a fainted slot.
#[test]
fn a_phaze_with_no_living_reserve_is_a_no_op() {
    let mut state = phaze_state(Choices::WHIRLWIND, Choices::SPLASH);
    for index in [
        PokemonIndex::P1,
        PokemonIndex::P2,
        PokemonIndex::P3,
        PokemonIndex::P4,
        PokemonIndex::P5,
    ] {
        state.side_one.pokemon[index].hp = 0;
    }

    let instructions = phaze(&mut state);
    assert!(
        switch_targets(&instructions).is_empty(),
        "nothing to drag: {:?}",
        instructions
    );
}
