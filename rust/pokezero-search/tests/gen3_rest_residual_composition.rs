//! Rest and end-of-turn residual composition pins.
//!
//! The historical claim for retained rows 2901076/41, 3000156/47, and 3500842/79
//! cannot be replayed because their raw reports are not retained. These controls
//! therefore establish only the scheduler boundary: a survivor gets its eligible
//! tail, while a terminal faint gets none. They must not be used to attribute any
//! of the retained rows to the scheduler or to a damage/composition mechanism.

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

/// Apply then reverse: search relies on every emitted instruction being an exact
/// inverse, so a Rest/residual reshape cannot corrupt the tree.
fn assert_reverts_cleanly(state: &mut State, list: &Vec<Instruction>) {
    let before = format!("{:?}", state);
    state.apply_instructions(list);
    state.reverse_instructions(list);
    assert_eq!(
        before,
        format!("{:?}", state),
        "instructions did not revert"
    );
}

fn index_of<F>(list: &[Instruction], predicate: F) -> usize
where
    F: Fn(&Instruction) -> bool,
{
    list.iter()
        .position(predicate)
        .unwrap_or_else(|| panic!("expected instruction not emitted: {:?}", list))
}

fn damage_at(list: &[Instruction], side_ref: SideReference, amount: i16) -> usize {
    index_of(list, |instruction| {
        matches!(instruction,
            Instruction::Damage(damage)
                if damage.side_ref == side_ref && damage.damage_amount == amount
        )
    })
}

fn heal_at(list: &[Instruction], side_ref: SideReference, amount: i16) -> usize {
    index_of(list, |instruction| {
        matches!(instruction,
            Instruction::Heal(heal)
                if heal.side_ref == side_ref && heal.heal_amount == amount
        )
    })
}

fn rest_at(list: &[Instruction]) -> usize {
    index_of(list, |instruction| {
        matches!(instruction,
            Instruction::ChangeStatus(change)
                if change.side_ref == SideReference::SideTwo
                    && change.new_status == PokemonStatus::SLEEP
        )
    })
}

fn side_two_rests(list: &[Instruction]) -> bool {
    list.iter().any(|instruction| {
        matches!(instruction,
            Instruction::ChangeStatus(change)
                if change.side_ref == SideReference::SideTwo
                    && change.new_status == PokemonStatus::SLEEP
        )
    })
}

/// Fixed damage keeps this a single branch: 100 damage leaves Rest's 285 HP user
/// alive, and Rest's 100 HP heal can therefore be ordered against the tail.
fn surviving_rest_state(toxic: bool) -> State {
    let mut state = State::default();

    let one = state.side_one.get_active();
    one.maxhp = 288;
    one.hp = 157;
    one.speed = 200;
    one.item = Items::LEFTOVERS;
    one.replace_move(PokemonMoveIndex::M0, Choices::SEISMICTOSS);
    if toxic {
        one.status = PokemonStatus::TOXIC;
        state.side_one.side_conditions.toxic_count = 1;
    }

    let two = state.side_two.get_active();
    two.maxhp = 285;
    two.hp = 285;
    two.speed = 1;
    two.replace_move(PokemonMoveIndex::M0, Choices::REST);

    state
}

fn only_rest_choice_branch(state: &mut State) -> Vec<Instruction> {
    let branches = generate(
        state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    );
    assert_eq!(branches.len(), 1, "expected one deterministic Rest branch");
    branches.into_iter().next().unwrap().instruction_list
}

/// This is the Leftovers-only survivor control. A fixed nonterminal hit must be
/// followed by Rest's full heal and then the opposing survivor's item tail.
#[test]
fn surviving_rest_turn_keeps_leftovers_tail() {
    let mut state = surviving_rest_state(false);
    let list = only_rest_choice_branch(&mut state);

    let hit = damage_at(&list, SideReference::SideTwo, 100);
    let rest = rest_at(&list);
    let full_heal = heal_at(&list, SideReference::SideTwo, 100);
    let leftovers = heal_at(&list, SideReference::SideOne, 18);
    assert!(
        hit < rest && rest < full_heal && full_heal < leftovers,
        "damage, Rest, full heal, then the survivor's Leftovers tail must stay ordered: {list:?}"
    );

    assert_reverts_cleanly(&mut state, &list);
    state.apply_instructions(&list);
    assert_eq!(
        state.side_two.get_active_immutable().hp,
        285,
        "Rest healed to full"
    );
    assert!(
        state.side_one.get_active_immutable().hp > 0,
        "attacker survived"
    );
    assert!(
        state.side_two.get_active_immutable().hp > 0,
        "Rest user survived"
    );
    assert_eq!(state.battle_is_over(), 0.0, "both sides remain in battle");
}

/// This is the same surviving shape with the Toxic tail enabled. For 288 max HP,
/// Leftovers is 18 and Toxic stage two is 36; the two instructions must be in
/// their same-Pokemon Gen 3 suborder, not merely present somewhere in the list.
#[test]
fn surviving_rest_turn_orders_leftovers_before_toxic_tail() {
    let mut state = surviving_rest_state(true);
    let list = only_rest_choice_branch(&mut state);

    let hit = damage_at(&list, SideReference::SideTwo, 100);
    let rest = rest_at(&list);
    let full_heal = heal_at(&list, SideReference::SideTwo, 100);
    let leftovers = heal_at(&list, SideReference::SideOne, 18);
    let toxic = damage_at(&list, SideReference::SideOne, 36);
    assert!(
        hit < rest && rest < full_heal && full_heal < leftovers && leftovers < toxic,
        "damage, Rest/full heal, then Leftovers before Toxic must stay ordered: {list:?}"
    );

    assert_reverts_cleanly(&mut state, &list);
    state.apply_instructions(&list);
    assert_eq!(
        state.side_two.get_active_immutable().hp,
        285,
        "Rest healed to full"
    );
    assert_eq!(state.side_one.get_active_immutable().hp, 139);
    assert_eq!(state.battle_is_over(), 0.0, "both sides remain in battle");
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
    one.status = PokemonStatus::TOXIC;
    one.replace_move(PokemonMoveIndex::M0, Choices::SEISMICTOSS);
    state.side_one.side_conditions.toxic_count = 1;

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

    assert_eq!(
        state.side_two.get_active_immutable().hp,
        1,
        "KO precondition"
    );
    assert!(
        [
            PokemonIndex::P1,
            PokemonIndex::P2,
            PokemonIndex::P3,
            PokemonIndex::P4,
            PokemonIndex::P5,
        ]
        .iter()
        .all(|index| state.side_two.pokemon[*index].hp == 0),
        "the Rest user is side two's last living Pokemon"
    );
    assert_eq!(state.battle_is_over(), 0.0, "battle is live before the KO");

    let list = only_rest_choice_branch(&mut state);
    assert_eq!(
        damage_at(&list, SideReference::SideTwo, 1),
        0,
        "the fixed hit KOs"
    );
    assert!(
        !side_two_rests(&list),
        "a fainted Rest user cannot execute: {list:?}"
    );
    assert!(
        !list.iter().any(|instruction| matches!(
            instruction,
            Instruction::Heal(heal) if heal.side_ref == SideReference::SideOne
        )),
        "a terminal branch must not re-add Leftovers: {list:?}"
    );
    assert!(
        !list.iter().any(|instruction| matches!(
            instruction,
            Instruction::Damage(damage) if damage.side_ref == SideReference::SideOne
        )),
        "a terminal branch must not re-add Toxic: {list:?}"
    );
    assert!(
        !list
            .iter()
            .any(|instruction| matches!(instruction, Instruction::ChangeSideCondition(_))),
        "a terminal branch must not advance any residual side condition: {list:?}"
    );

    assert_reverts_cleanly(&mut state, &list);
    state.apply_instructions(&list);
    assert_eq!(
        state.side_two.get_active_immutable().hp,
        0,
        "the final Pokemon fainted"
    );
    assert_eq!(state.battle_is_over(), 1.0, "the KO ended the battle");
}
