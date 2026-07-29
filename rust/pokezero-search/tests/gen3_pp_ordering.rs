//! Gen 3 PP is charged only for a move that actually got past the BeforeMove
//! gate. Asserted against the vendored gen3-patched poke-engine.
//!
//! Showdown deducts in `BattleActions.runMove` (sim/battle-actions.ts:282),
//! inside the same `if (!externalMove)` block that ends in `moveUsed`, and only
//! after the gate has passed:
//!
//! ```text
//! const willTryMove = this.battle.runEvent('BeforeMove', pokemon, target, move);
//! if (!willTryMove) { ...; return; }                     // no PP charged
//! ...
//! if (!pokemon.deductPP(baseMove, null, target) && move.id !== 'struggle') {
//!     this.battle.add('cant', pokemon, 'nopp', move); ...; return;
//! }
//! pokemon.moveUsed(move, targetLoc);
//! ```
//!
//! So every immobilizer that returns `false` from its `onBeforeMove` — full
//! paralysis, staying asleep, staying frozen, flinch, a confusion self-hit,
//! infatuation — costs no PP at all. The deduction precedes `useMove`, so a move
//! that misses, fails outright or is blocked by Protect still pays.
//!
//! Pressure needs no separate handling: Showdown charges its extra point inside
//! `useMove` (sim/battle-actions.ts:482), downstream of the same gate, so the
//! engine's combined 1-or-2 decrement moves as one unit.
//!
//! The engine only emits a decrement when the slot is under 10 PP (an
//! instruction-count optimization), so every fixture here starts the move well
//! inside that window.

use poke_engine::choices::Choices;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::{MoveChoice, PokemonVolatileStatus};
use poke_engine::instruction::{Instruction, StateInstructions};
use poke_engine::state::{PokemonMoveIndex, PokemonStatus, PokemonType, SideReference, State};

const START_PP: i8 = 5;

fn generate(state: &mut State) -> Vec<StateInstructions> {
    let before = format!("{:?}", state);
    let instructions = generate_instructions_from_move_pair(
        state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
        false,
    );
    assert_eq!(before, format!("{:?}", state), "state was mutated");
    instructions
}

/// Side one is the seat under test, moving first, with `START_PP` on its move so
/// the engine's under-10 decrement window is active.
fn pp_state(attacker_move: Choices) -> State {
    let mut state = State::default();
    state.side_one.get_active().speed = 500;
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, attacker_move);
    state.side_one.get_active().moves[&PokemonMoveIndex::M0].pp = START_PP;
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::SPLASH);
    state
}

fn charges_pp(list: &[Instruction]) -> bool {
    list.iter().any(|instruction| match instruction {
        Instruction::DecrementPP(decrement) => decrement.side_ref == SideReference::SideOne,
        _ => false,
    })
}

/// Mass, in percent, of branches that charge side one a PP.
fn charged_mass(branches: &[StateInstructions]) -> f32 {
    branches
        .iter()
        .filter(|branch| charges_pp(&branch.instruction_list))
        .map(|branch| branch.percentage)
        .sum()
}

fn assert_close(actual: f32, expected: f32, what: &str) {
    assert!(
        (actual - expected).abs() < 1e-3,
        "{}: got {}%, expected {}%",
        what,
        actual,
        expected
    );
}

// ---------------------------------------------------------------------------
// Charged: the move executed
// ---------------------------------------------------------------------------

#[test]
fn an_executed_move_charges_pp() {
    let mut state = pp_state(Choices::SPLASH);
    assert_close(charged_mass(&generate(&mut state)), 100.0, "ordinary move");
}

/// The deduction precedes `useMove`, so accuracy has not been rolled yet: a move
/// that MISSES still pays. Thunder is 70% accurate.
#[test]
fn a_move_that_misses_still_charges_pp() {
    let mut state = pp_state(Choices::THUNDER);
    let branches = generate(&mut state);
    assert!(
        branches.len() > 1,
        "expected a hit/miss split: {:?}",
        branches
    );
    assert_close(charged_mass(&branches), 100.0, "hit and miss both pay");
}

/// Same reason: a move with no effect at all still pays, because
/// `move_has_no_effect` is checked after the charge point just as `useMove` runs
/// after the deduction. Thunder Wave into a Ground type does nothing.
#[test]
fn a_move_that_fails_outright_still_charges_pp() {
    let mut state = pp_state(Choices::THUNDERWAVE);
    state.side_two.get_active().types.0 = PokemonType::GROUND;
    assert_close(
        charged_mass(&generate(&mut state)),
        100.0,
        "a move that does nothing still pays",
    );
}

// ---------------------------------------------------------------------------
// Free: the move never started
// ---------------------------------------------------------------------------

/// Full paralysis is `par.onBeforeMove -> false` (gen4 mod). The 25% branch is
/// free; the 75% branch pays. The engine used to charge on all 100%.
#[test]
fn a_fully_paralyzed_turn_is_free() {
    let mut state = pp_state(Choices::SPLASH);
    state.side_one.get_active().status = PokemonStatus::PARALYZE;
    assert_close(
        charged_mass(&generate(&mut state)),
        75.0,
        "only the non-paralyzed branch pays",
    );
}

#[test]
fn a_flinched_turn_is_free() {
    let mut state = pp_state(Choices::SPLASH);
    state
        .side_one
        .volatile_statuses
        .insert(PokemonVolatileStatus::FLINCH);
    assert_close(charged_mass(&generate(&mut state)), 0.0, "flinched turn");
}

#[test]
fn an_infatuation_immobilized_turn_is_free() {
    let mut state = pp_state(Choices::SPLASH);
    state
        .side_one
        .volatile_statuses
        .insert(PokemonVolatileStatus::ATTRACT);
    assert_close(
        charged_mass(&generate(&mut state)),
        50.0,
        "only the branch that gets to move pays",
    );
}

/// A confusion self-hit is `confusion.onBeforeMove -> false` (gen4 mod), so it
/// is free; the other half of the same 50/50 pays.
#[test]
fn a_confusion_self_hit_is_free_but_the_other_half_pays() {
    let mut state = pp_state(Choices::SPLASH);
    state
        .side_one
        .volatile_statuses
        .insert(PokemonVolatileStatus::CONFUSION);
    assert_close(
        charged_mass(&generate(&mut state)),
        50.0,
        "self-hit is free, move-goes-through pays",
    );
}

/// Stacked immobilizers compose exactly as the BeforeMove chain does: confusion
/// (priority 3) then paralysis (1). Only the branch clearing BOTH pays.
#[test]
fn confusion_and_paralysis_compose_on_the_charge_point() {
    let mut state = pp_state(Choices::SPLASH);
    state
        .side_one
        .volatile_statuses
        .insert(PokemonVolatileStatus::CONFUSION);
    state.side_one.get_active().status = PokemonStatus::PARALYZE;
    assert_close(
        charged_mass(&generate(&mut state)),
        37.5,
        "only the branch clearing confusion AND paralysis pays",
    );
}

/// Sleep splits: staying asleep is free, waking and moving pays. At
/// `sleep_turns == 3` the wake chance is 1/2.
#[test]
fn sleeping_through_a_turn_is_free_but_waking_and_moving_pays() {
    let mut state = pp_state(Choices::SPLASH);
    state.side_one.get_active().status = PokemonStatus::SLEEP;
    state.side_one.get_active().sleep_turns = 3;
    assert_close(
        charged_mass(&generate(&mut state)),
        50.0,
        "only the waking branch pays",
    );
}

#[test]
fn a_pokemon_that_cannot_wake_this_turn_pays_nothing() {
    let mut state = pp_state(Choices::SPLASH);
    state.side_one.get_active().status = PokemonStatus::SLEEP;
    state.side_one.get_active().sleep_turns = 0;
    assert_close(charged_mass(&generate(&mut state)), 0.0, "still asleep");
}

/// Freeze: 20% thaw-and-move pays, 80% stays frozen and is free.
#[test]
fn a_frozen_turn_is_free_but_thawing_and_moving_pays() {
    let mut state = pp_state(Choices::SPLASH);
    state.side_one.get_active().status = PokemonStatus::FREEZE;
    assert_close(
        charged_mass(&generate(&mut state)),
        20.0,
        "only the thawing branch pays",
    );
}

// ---------------------------------------------------------------------------
// Applied state, and the interaction with Encore's PP reads
// ---------------------------------------------------------------------------

/// The instruction is not just absent — the resulting PP really is unchanged, so
/// a Pokemon that spends turns immobilized does not drain toward Struggle. This
/// is the property the Encore PP paths read: `encore.onResidual` ends Encore at
/// 0 PP and `encore.onStart` refuses a target already at 0, so phantom drain
/// would free a locked seat Showdown keeps locked.
#[test]
fn an_immobilized_turn_leaves_the_pp_value_untouched() {
    let mut state = pp_state(Choices::SPLASH);
    state.side_one.get_active().status = PokemonStatus::SLEEP;
    state.side_one.get_active().sleep_turns = 0;

    let branches = generate(&mut state);
    assert_eq!(branches.len(), 1, "a mon that cannot wake is deterministic");
    state.apply_instructions(&branches[0].instruction_list);
    assert_eq!(
        state.side_one.get_active().moves[&PokemonMoveIndex::M0].pp,
        START_PP,
        "sleeping through a turn must not spend PP: {:?}",
        branches[0].instruction_list
    );
}

/// Control for the pin above, on the same fixture shape: an executed turn does
/// move the value, so the assertion above is about the gate and not about the
/// decrement having been lost altogether.
#[test]
fn an_executed_turn_really_does_spend_the_pp() {
    let mut state = pp_state(Choices::SPLASH);
    let branches = generate(&mut state);
    state.apply_instructions(&branches[0].instruction_list);
    assert_eq!(
        state.side_one.get_active().moves[&PokemonMoveIndex::M0].pp,
        START_PP - 1,
        "an executed move spends exactly one PP"
    );
}

/// Pressure charges two, and moves as one unit with the base point — Showdown
/// deducts the extra inside `useMove`, downstream of the same gate, so an
/// immobilized turn owes neither.
#[test]
fn pressure_charges_two_when_the_move_executes_and_none_when_it_does_not() {
    let mut state = pp_state(Choices::TACKLE);
    state.side_two.get_active().ability = poke_engine::engine::abilities::Abilities::PRESSURE;
    let branches = generate(&mut state);
    state.apply_instructions(&branches[0].instruction_list);
    assert_eq!(
        state.side_one.get_active().moves[&PokemonMoveIndex::M0].pp,
        START_PP - 2,
        "Pressure costs two on an executed move"
    );

    let mut asleep = pp_state(Choices::TACKLE);
    asleep.side_two.get_active().ability = poke_engine::engine::abilities::Abilities::PRESSURE;
    asleep.side_one.get_active().status = PokemonStatus::SLEEP;
    asleep.side_one.get_active().sleep_turns = 0;
    assert_close(
        charged_mass(&generate(&mut asleep)),
        0.0,
        "Pressure owes nothing on a turn the move never started",
    );
}
