//! Rest and end-of-turn residual composition pins.
//!
//! The retained rows 2901076/41, 3000156/47, and 3500842/79 initially looked
//! like missing Leftovers or Toxic residuals. Replay shows the residual tail is
//! present on every engine branch that survives the move; the apparent omission
//! is instead attached to a capped-lethal damage branch. These controls make
//! that boundary explicit so a future damage-lattice repair cannot be "fixed"
//! by incorrectly re-attaching residuals after a terminal faint.

use poke_engine::choices::Choices;
use poke_engine::engine::items::Items;
use poke_engine::engine::state::MoveChoice;
use poke_engine::instruction::{Instruction, StateInstructions};
use poke_engine::state::{PokemonIndex, PokemonMoveIndex, PokemonStatus, SideReference, State};

fn generate(
    state: &mut State,
    side_one: &MoveChoice,
    side_two: &MoveChoice,
) -> Vec<StateInstructions> {
    let before = format!("{:?}", state);
    let branches = poke_engine::engine::generate_instructions::generate_instructions_from_move_pair(
        state, side_one, side_two, false,
    );
    assert_eq!(before, format!("{:?}", state), "state was mutated");
    branches
}

fn has_heal(list: &[Instruction], side_ref: SideReference, amount: i16) -> bool {
    list.iter().any(|instruction| match instruction {
        Instruction::Heal(heal) => heal.side_ref == side_ref && heal.heal_amount == amount,
        _ => false,
    })
}

fn has_damage(list: &[Instruction], side_ref: SideReference, amount: i16) -> bool {
    list.iter().any(|instruction| match instruction {
        Instruction::Damage(damage) => {
            damage.side_ref == side_ref && damage.damage_amount == amount
        }
        _ => false,
    })
}

fn rests_side_two(list: &[Instruction]) -> bool {
    list.iter().any(|instruction| match instruction {
        Instruction::ChangeStatus(change) => {
            change.side_ref == SideReference::SideTwo && change.new_status == PokemonStatus::SLEEP
        }
        _ => false,
    })
}

/// This is the surviving half of the retained Rest shapes: Rest resolves, then
/// the opposing Toxic user gets its full order-10 residual tail. For 288 max HP,
/// Leftovers is 18 and Toxic stage two is 36.
#[test]
fn surviving_rest_turn_keeps_leftovers_and_toxic_tail() {
    let mut state = State::default();

    let one = state.side_one.get_active();
    one.maxhp = 288;
    one.hp = 157;
    one.speed = 200;
    one.item = Items::LEFTOVERS;
    one.status = PokemonStatus::TOXIC;
    one.replace_move(PokemonMoveIndex::M0, Choices::SPLASH);
    state.side_one.side_conditions.toxic_count = 1;

    let two = state.side_two.get_active();
    two.maxhp = 285;
    two.hp = 17;
    two.speed = 1;
    two.replace_move(PokemonMoveIndex::M0, Choices::REST);

    let branches = generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    );
    assert_eq!(branches.len(), 1, "expected one deterministic Rest branch");
    let list = &branches[0].instruction_list;

    assert!(rests_side_two(list), "Rest must execute: {list:?}");
    assert!(
        has_heal(list, SideReference::SideOne, 18),
        "the surviving Toxic user keeps its Leftovers tick: {list:?}"
    );
    assert!(
        has_damage(list, SideReference::SideOne, 36),
        "the surviving Toxic user keeps its stage-two poison tick: {list:?}"
    );
}

/// The terminal half of the same shape is intentionally the opposite: when
/// damage faints the last opposing Pokemon before it can Rest, no residual tail
/// may follow. This is why the capped-lethal retained branches cannot be repaired
/// by changing residual scheduling.
#[test]
fn terminal_rest_turn_never_readds_residual_tail() {
    let mut state = State::default();

    let one = state.side_one.get_active();
    one.maxhp = 288;
    one.hp = 157;
    one.speed = 200;
    one.item = Items::LEFTOVERS;
    one.replace_move(PokemonMoveIndex::M0, Choices::TACKLE);

    let two = state.side_two.get_active();
    two.maxhp = 285;
    two.hp = 1;
    two.speed = 1;
    two.replace_move(PokemonMoveIndex::M0, Choices::REST);
    for index in [
        PokemonIndex::P1,
        PokemonIndex::P2,
        PokemonIndex::P3,
        PokemonIndex::P4,
        PokemonIndex::P5,
    ] {
        state.side_two.pokemon[index].hp = 0;
    }

    let branches = generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    );
    assert!(!branches.is_empty(), "expected damaging branches");
    for branch in branches {
        let list = &branch.instruction_list;
        assert!(
            !rests_side_two(list),
            "a fainted Rest user cannot execute: {list:?}"
        );
        assert!(
            !list.iter().any(|instruction| matches!(
                instruction,
                Instruction::Heal(heal) if heal.side_ref == SideReference::SideOne
            )),
            "a terminal branch must not re-add Leftovers: {list:?}"
        );
    }
}
