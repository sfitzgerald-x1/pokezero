//! Regression for a Gen 3 confusion-before-move boundary.
//!
//! In Gen 3, confusion's `onBeforeMove` handler runs before `useMove`.  A
//! self-hit therefore cancels a selected Substitute: it pays no quarter-HP
//! cost, creates no Substitute, and still reaches end-of-turn Leftovers.
//! The companion branch where confusion lets the move through still creates
//! the Substitute and pays its normal cost.

use poke_engine::choices::Choices;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::items::Items;
use poke_engine::engine::state::{MoveChoice, PokemonVolatileStatus};
use poke_engine::instruction::{Instruction, StateInstructions};
use poke_engine::state::{PokemonMoveIndex, SideReference, State};
use pokezero_search::events::{render_branch_events, EventContext};

fn substitute_boundary_state() -> State {
    let mut state = State::default();
    state.side_one.get_active().speed = 500;
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::SPLASH);

    let target = state.side_two.get_active();
    target.speed = 1;
    target.maxhp = 256;
    target.hp = 200;
    // At level 100, 108 Attack against the default 100 Defense yields the
    // 38 HP confusion self-hit.
    target.attack = 108;
    target.item = Items::LEFTOVERS;
    target.replace_move(PokemonMoveIndex::M0, Choices::SUBSTITUTE);
    state
        .side_two
        .volatile_statuses
        .insert(PokemonVolatileStatus::CONFUSION);
    state
}

fn generate(state: &mut State) -> Vec<StateInstructions> {
    let before = format!("{state:?}");
    let branches = generate_instructions_from_move_pair(
        state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
        false,
    );
    assert_eq!(
        before,
        format!("{state:?}"),
        "generation must not mutate state"
    );
    branches
}

fn has_side_two_damage(branch: &StateInstructions, amount: i16) -> bool {
    branch.instruction_list.iter().any(|instruction| {
        matches!(instruction, Instruction::Damage(damage)
            if damage.side_ref == SideReference::SideTwo && damage.damage_amount == amount)
    })
}

fn has_side_two_heal(branch: &StateInstructions, amount: i16) -> bool {
    branch.instruction_list.iter().any(|instruction| {
        matches!(instruction, Instruction::Heal(heal)
            if heal.side_ref == SideReference::SideTwo && heal.heal_amount == amount)
    })
}

fn creates_side_two_substitute(branch: &StateInstructions) -> bool {
    branch.instruction_list.iter().any(|instruction| {
        matches!(instruction, Instruction::ApplyVolatileStatus(apply)
            if apply.side_ref == SideReference::SideTwo
                && apply.volatile_status == PokemonVolatileStatus::SUBSTITUTE)
    })
}

fn render_branch(state: &mut State, branch: &StateInstructions) -> String {
    let rendered = render_branch_events(
        state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &branch.instruction_list,
        false,
        &EventContext {
            species: [vec!["Lead".into()], vec!["Opponent".into()]],
            turn: 1,
            hp_percent: [false, false],
        },
    );
    rendered.lines.join("\n")
}

#[test]
fn confusion_self_hit_cancels_substitute_but_keeps_leftovers() {
    let mut state = substitute_boundary_state();
    let branches = generate(&mut state);

    let self_hit = branches
        .iter()
        .find(|branch| has_side_two_damage(branch, 38))
        .unwrap_or_else(|| {
            panic!(
                "the -38 confusion self-hit branch must remain present: {branches:?}"
            )
        });
    assert!(
        !creates_side_two_substitute(self_hit),
        "a confusion self-hit must cancel Substitute: {:?}",
        self_hit.instruction_list
    );
    assert!(
        !has_side_two_damage(self_hit, 64),
        "the cancelled Substitute must not charge quarter HP: {:?}",
        self_hit.instruction_list
    );
    assert!(
        has_side_two_heal(self_hit, 16),
        "the self-hit survivor must still receive Leftovers: {:?}",
        self_hit.instruction_list
    );
}

#[test]
fn confusion_survival_branch_still_creates_substitute() {
    let mut state = substitute_boundary_state();
    let branches = generate(&mut state);
    let substitute = branches
        .iter()
        .find(|branch| creates_side_two_substitute(branch))
        .unwrap_or_else(|| {
            panic!("the non-self-hit branch must still execute Substitute: {branches:?}")
        });
    let events = render_branch(&mut state, substitute);
    assert!(
        events.contains("|move|p2a: Opponent|substitute|p2a: Opponent"),
        "the surviving confusion branch must still render its selected move: {events}"
    );
}

#[test]
fn rendered_self_hit_is_not_misread_as_a_substitute_action() {
    let mut state = substitute_boundary_state();
    let branches = generate(&mut state);
    let self_hit = branches
        .iter()
        .find(|branch| has_side_two_damage(branch, 38))
        .expect("expected -38 self-hit branch");
    let events = render_branch(&mut state, self_hit);
    assert!(
        events.contains("|-activate|p2a: Opponent|confusion")
            && events.contains("|-damage|p2a: Opponent|162/256"),
        "the self-hit must preserve the public confusion activation and bare damage: {events}"
    );
    assert!(
        events.contains("|-heal|p2a: Opponent|178/256|[from] item: Leftovers"),
        "the self-hit survivor must retain end-of-turn recovery: {events}"
    );
    assert!(
        !events.contains("|move|p2a: Opponent|substitute"),
        "a self-hit must cancel, not render, Substitute: {events}"
    );
}

#[test]
fn unconfused_substitute_still_renders_as_a_move() {
    let mut state = substitute_boundary_state();
    state
        .side_two
        .volatile_statuses
        .remove(&PokemonVolatileStatus::CONFUSION);
    let branches = generate(&mut state);
    let substitute = branches
        .iter()
        .find(|branch| creates_side_two_substitute(branch))
        .expect("unconfused Substitute must execute");
    let events = render_branch(&mut state, substitute);
    assert!(
        events.contains("|move|p2a: Opponent|substitute|p2a: Opponent"),
        "{events}"
    );
    assert!(
        !events.contains("[from] confusion"),
        "unconfused move must not acquire confusion attribution: {events}"
    );
}

#[test]
fn flinched_confused_substitute_stays_incapacitated() {
    let mut state = substitute_boundary_state();
    state
        .side_two
        .volatile_statuses
        .insert(PokemonVolatileStatus::FLINCH);
    let branches = generate(&mut state);
    assert_eq!(branches.len(), 1, "flinch must not fan into a move branch");
    let events = render_branch(&mut state, &branches[0]);
    assert!(events.contains("|cant|p2a: Opponent|flinch"), "{events}");
    assert!(
        !events.contains("|move|p2a: Opponent|substitute"),
        "{events}"
    );
    assert!(!events.contains("[from] confusion"), "{events}");
}
