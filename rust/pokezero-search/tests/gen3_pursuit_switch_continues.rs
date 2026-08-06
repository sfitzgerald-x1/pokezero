//! Gen 2-4: a Pursuit KO does NOT cancel the switch it interrupted.
//!
//! C134 §4, the queue-semantics fixture pack. Pursuit is the one case where a switch
//! is the SECOND action of a ply without being a double switch, and it is the case
//! that broke two consecutive faint-cancellation patches.
//!
//! Showdown states the rule in its own source, `sim/battle.ts:2790-2794`:
//!
//! ```text
//! if (this.actions.switchIn(action.target, action.pokemon.position, ...) === 'pursuitfaint') {
//!     // a pokemon fainted from Pursuit before it could switch
//!     if (this.gen <= 4) {
//!         // in gen 2-4, the switch still happens
//!         this.hint("Previously chosen switches continue in Gen 2-4 after a Pursuit target faints.");
//! ```
//!
//! Ground truth confirmed against real gen3 Showdown on four seeds by
//! `scripts/gen3_switch_differential.py`, scenarios `pursuitkoswitcher` (whose landmark
//! is that hint line, so it cannot pass without taking the gen<=4 branch) and
//! `pursuitnokocontrol`. Showdown performs the switch, runs the residual block, and the
//! opponent's Leftovers ticks — in BOTH the KO and the no-KO arm.
//!
//! WHY THIS PIN EXISTS. A faint-cancellation guard keyed on "an active newly reached
//! 0 HP" gets this exactly backwards: the Pokemon that fainted IS the switcher, and
//! gen 3 continues its switch anyway. Cancelling leaves the wrong Pokemon active and
//! the opponent's residual tick cannot be attributed, which is the observed signature
//! of the two rows that patch opened (`19000120` dev, `19100078` holdout, both
//! `component_missing_in_engine:itemleftovers`). This pin is green on `main`, which has
//! no cancellation at all, and red on the withdrawn v2 branch. It is a constraint on
//! any future v3, not a description of a bug in `main`.

use poke_engine::choices::Choices;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::MoveChoice;
use poke_engine::instruction::{Instruction, StateInstructions};
use poke_engine::state::{PokemonIndex, PokemonMoveIndex, SideReference, State};

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

fn switches(list: &[Instruction], side_ref: SideReference) -> Vec<(PokemonIndex, PokemonIndex)> {
    list.iter()
        .filter_map(|instruction| match instruction {
            Instruction::Switch(switch) if switch.side_ref == side_ref => {
                Some((switch.previous_index, switch.next_index))
            }
            _ => None,
        })
        .collect()
}

/// Side one switches out; side two Pursues it. `lethal` decides whether the Pursuit
/// hit kills the outgoing Pokemon, which is the only difference between the two arms.
fn pursuit_against_a_switch(lethal: bool) -> Vec<StateInstructions> {
    let mut state = State::default();

    // Side two is faster, so Pursuit resolves first and the switch is the second
    // action of the ply — the shape that makes `first_move` false for a switch.
    state.side_two.pokemon[PokemonIndex::P0].speed = 200;
    state.side_one.pokemon[PokemonIndex::P0].speed = 100;
    state.side_two.pokemon[PokemonIndex::P0]
        .replace_move(PokemonMoveIndex::M0, Choices::PURSUIT);

    let outgoing = &mut state.side_one.pokemon[PokemonIndex::P0];
    outgoing.maxhp = 300;
    outgoing.hp = if lethal { 1 } else { 300 };

    generate(
        &mut state,
        &MoveChoice::Switch(PokemonIndex::P1),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    )
}

/// Every branch in which the outgoing Pokemon actually fainted must still switch.
/// Asserting over all branches rather than a single one keeps the pin honest when
/// Pursuit's damage roll splits the ply into several.
#[test]
fn a_pursuit_ko_does_not_cancel_the_switch_it_interrupted() {
    let branches = pursuit_against_a_switch(true);
    assert!(!branches.is_empty(), "expected at least one branch");

    let mut checked = 0;
    for branch in &branches {
        let list = &branch.instruction_list;
        let killed = list.iter().any(|instruction| {
            matches!(instruction, Instruction::Damage(damage)
                if damage.side_ref == SideReference::SideOne && damage.damage_amount >= 1)
        });
        if !killed {
            continue;
        }
        checked += 1;
        assert_eq!(
            switches(list, SideReference::SideOne),
            vec![(PokemonIndex::P0, PokemonIndex::P1)],
            "gen 2-4 continues a switch whose Pursuit target fainted \
             (sim/battle.ts:2790-2794); cancelling it drops the opponent's residual \
             tick: {:?}",
            list
        );
    }
    assert!(
        checked > 0,
        "no branch dealt Pursuit damage, so the fixture proved nothing: {:?}",
        branches
    );
}

/// Control: the same ply with the outgoing Pokemon at full HP. Pursuit cannot KO, and
/// the switch obviously happens. If this ever diverges from the KO arm's expectation,
/// the difference is the faint, which is the variable under test.
#[test]
fn a_pursuit_that_does_not_ko_also_switches() {
    let branches = pursuit_against_a_switch(false);
    assert!(!branches.is_empty(), "expected at least one branch");

    for branch in &branches {
        assert_eq!(
            switches(&branch.instruction_list, SideReference::SideOne),
            vec![(PokemonIndex::P0, PokemonIndex::P1)],
            "a survived Pursuit must never suppress the switch: {:?}",
            branch.instruction_list
        );
    }
}
