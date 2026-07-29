//! Gen 3 `last_used_move` semantics: a move counts as "used" only if it actually
//! got past the BeforeMove gate. Asserted against the vendored gen3-patched
//! poke-engine (`third_party/poke-engine-src/`).
//!
//! Showdown sets `lastMove` in `Pokemon.moveUsed()`, called from
//! `BattleActions.runMove` (sim/battle-actions.ts:291) at exactly one point:
//!
//! ```text
//! const willTryMove = this.battle.runEvent('BeforeMove', pokemon, target, move);
//! if (!willTryMove) { runEvent('MoveAborted', ...); ...; return; }   // no moveUsed
//! if (move.beforeMoveCallback) { ... return; }                        // no moveUsed
//! if (!pokemon.deductPP(baseMove, null, target) && move.id !== 'struggle') {
//!     this.battle.add('cant', pokemon, 'nopp', move); ... return;     // no moveUsed
//! }
//! pokemon.moveUsed(move, targetLoc);                                  // lastMove SET
//! ```
//!
//! Every immobilizer sits behind that gate by returning `false` from its
//! `onBeforeMove`, so none of them records a last move. `moveUsed` runs BEFORE
//! `useMove`, so a move that misses, fails or is blocked by Protect still counts.
//!
//! Truth table, with the gen3-chain handler that decides each row:
//!
//! | turn outcome                        | lastMove | source |
//! |-------------------------------------|----------|--------|
//! | move executes (hit / miss / fail)   | SET      | `moveUsed` precedes `useMove` |
//! | fully paralyzed                     | not set  | gen4 `par.onBeforeMove` -> false |
//! | asleep, stays asleep                | not set  | gen3 `slp.onBeforeMove` -> false |
//! | asleep, wakes and moves             | SET      | gen3 `slp` cures, returns undefined |
//! | asleep using Sleep Talk             | SET      | `sleepUsable` -> returns undefined |
//! | frozen, stays frozen                | not set  | gen4 `frz.onBeforeMove` -> false |
//! | frozen, thaws and moves             | SET      | gen4 `frz` cures, returns undefined |
//! | flinched                            | not set  | `flinch.onBeforeMove` -> false |
//! | confusion self-hit                  | not set  | gen4 `confusion` -> false |
//! | confusion snap-out / no self-hit    | SET      | returns undefined, move proceeds |
//! | infatuation immobilized             | not set  | `attract` -> false |
//!
//! Why it matters: Encore's `onStart` reads `lastMove`, so crediting a move the
//! target never made lets the engine Encore something Showdown would refuse.

use poke_engine::choices::Choices;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::{MoveChoice, PokemonVolatileStatus};
use poke_engine::instruction::{Instruction, StateInstructions};
use poke_engine::state::{
    LastUsedMove, PokemonMoveIndex, PokemonStatus, SideReference, State,
};

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

/// Side one is the seat under test. `use_last_used_move` is the engine's
/// conditional-mechanics flag (normally switched on by an Encore or Fake Out
/// being present); without it the engine tracks nothing at all.
fn tracked_state(attacker_move: Choices) -> State {
    let mut state = State::default();
    state.use_last_used_move = true;
    state.side_one.get_active().speed = 500;
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, attacker_move);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::SPLASH);
    state
}

fn records_a_last_move(list: &[Instruction]) -> bool {
    list.iter().any(|instruction| match instruction {
        Instruction::SetLastUsedMove(set) => {
            set.side_ref == SideReference::SideOne
                && matches!(set.last_used_move, LastUsedMove::Move(_))
        }
        _ => false,
    })
}

/// Mass, in percent, of the branches that record a last move for side one.
fn recorded_mass(branches: &[StateInstructions]) -> f32 {
    branches
        .iter()
        .filter(|branch| records_a_last_move(&branch.instruction_list))
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
// Rows that must RECORD
// ---------------------------------------------------------------------------

/// The baseline row: an unobstructed move is used.
#[test]
fn an_executed_move_is_recorded() {
    let mut state = tracked_state(Choices::SPLASH);
    assert_close(
        recorded_mass(&generate(&mut state)),
        100.0,
        "unobstructed move",
    );
}

/// `moveUsed` runs before `useMove`, so accuracy has not been rolled yet: a move
/// that MISSES still counts as used. Thunder is 70% accurate, and both outcomes
/// must record.
#[test]
fn a_move_that_misses_is_still_recorded() {
    let mut state = tracked_state(Choices::THUNDER);
    let branches = generate(&mut state);
    assert!(
        branches.len() > 1,
        "expected a hit/miss split to exercise the miss branch: {:?}",
        branches
    );
    assert_close(
        recorded_mass(&branches),
        100.0,
        "every hit-or-miss branch must record",
    );
}

/// Same reason, the other way: a move with no effect at all still counts as
/// used, because `move_has_no_effect` is checked after the record point just as
/// `useMove` runs after `moveUsed`. Thunder Wave into a Ground type does nothing.
#[test]
fn a_move_that_fails_outright_is_still_recorded() {
    let mut state = tracked_state(Choices::THUNDERWAVE);
    state.side_two.get_active().types.0 = poke_engine::state::PokemonType::GROUND;
    assert_close(
        recorded_mass(&generate(&mut state)),
        100.0,
        "a move that does nothing is still used",
    );
}

// ---------------------------------------------------------------------------
// Rows that must NOT record
// ---------------------------------------------------------------------------

/// Full paralysis is `par.onBeforeMove -> false` (gen4 mod), so the 25% branch
/// records nothing while the 75% branch does. This is the row that motivated the
/// whole fix: the engine used to record on all 100%.
#[test]
fn a_fully_paralyzed_turn_is_not_recorded() {
    let mut state = tracked_state(Choices::SPLASH);
    state.side_one.get_active().status = PokemonStatus::PARALYZE;
    assert_close(
        recorded_mass(&generate(&mut state)),
        75.0,
        "only the non-paralyzed branch may record",
    );
}

/// `flinch.onBeforeMove -> false`. The engine already short-circuited this one in
/// `cannot_use_move`; pinned so the record point cannot drift back above it.
#[test]
fn a_flinched_turn_is_not_recorded() {
    let mut state = tracked_state(Choices::SPLASH);
    state
        .side_one
        .volatile_statuses
        .insert(PokemonVolatileStatus::FLINCH);
    assert_close(recorded_mass(&generate(&mut state)), 0.0, "flinched turn");
}

/// Infatuation is `attract.onBeforeMove -> false` at priority 2, so the
/// immobilized half records nothing.
#[test]
fn an_infatuation_immobilized_turn_is_not_recorded() {
    let mut state = tracked_state(Choices::SPLASH);
    state
        .side_one
        .volatile_statuses
        .insert(PokemonVolatileStatus::ATTRACT);
    assert_close(
        recorded_mass(&generate(&mut state)),
        50.0,
        "only the branch that gets to move may record",
    );
}

/// A confusion self-hit is `confusion.onBeforeMove -> false` (gen4 mod), so it
/// records nothing; the other half of the same 50/50 does. Interacts directly
/// with the confusion ladder and its paralysis reordering — both stay green.
#[test]
fn a_confusion_self_hit_is_not_recorded_but_the_other_half_is() {
    let mut state = tracked_state(Choices::SPLASH);
    state
        .side_one
        .volatile_statuses
        .insert(PokemonVolatileStatus::CONFUSION);
    assert_close(
        recorded_mass(&generate(&mut state)),
        50.0,
        "self-hit records nothing, move-goes-through records",
    );
}

/// Stacked immobilizers compose multiplicatively, exactly as the BeforeMove chain
/// does: confusion (priority 3) then paralysis (1). Only the branch that clears
/// BOTH may record — 1/2 * 3/4.
#[test]
fn confusion_and_paralysis_compose_on_the_record_point() {
    let mut state = tracked_state(Choices::SPLASH);
    state
        .side_one
        .volatile_statuses
        .insert(PokemonVolatileStatus::CONFUSION);
    state.side_one.get_active().status = PokemonStatus::PARALYZE;
    assert_close(
        recorded_mass(&generate(&mut state)),
        37.5,
        "only the branch clearing confusion AND paralysis records",
    );
}

/// Sleep splits: the still-asleep branch records nothing, the waking branch does
/// (gen3 `slp` cures and returns undefined, so the move proceeds). At
/// `sleep_turns == 3` the wake chance is 1/2.
#[test]
fn a_slept_through_turn_is_not_recorded_but_waking_and_moving_is() {
    let mut state = tracked_state(Choices::SPLASH);
    state.side_one.get_active().status = PokemonStatus::SLEEP;
    state.side_one.get_active().sleep_turns = 3;
    assert_close(
        recorded_mass(&generate(&mut state)),
        50.0,
        "only the waking branch may record",
    );
}

/// A Pokemon that cannot wake yet records nothing at all.
#[test]
fn a_pokemon_that_cannot_wake_this_turn_records_nothing() {
    let mut state = tracked_state(Choices::SPLASH);
    state.side_one.get_active().status = PokemonStatus::SLEEP;
    state.side_one.get_active().sleep_turns = 0;
    assert_close(recorded_mass(&generate(&mut state)), 0.0, "still asleep");
}

/// Freeze splits the same way: 20% thaw-and-move records, 80% stays frozen and
/// does not (gen4 `frz.onBeforeMove -> false`).
#[test]
fn a_frozen_turn_is_not_recorded_but_thawing_and_moving_is() {
    let mut state = tracked_state(Choices::SPLASH);
    state.side_one.get_active().status = PokemonStatus::FREEZE;
    assert_close(
        recorded_mass(&generate(&mut state)),
        20.0,
        "only the thawing branch may record",
    );
}

// ---------------------------------------------------------------------------
// Sleep Talk: the caller is recorded, the called move is not
// ---------------------------------------------------------------------------

/// Sleep Talk is `sleepUsable`, so gen3's `slp.onBeforeMove` returns undefined
/// and Sleep Talk itself reaches `moveUsed`. The move it calls is run through
/// `useMove`, which never touches `lastMove` — so the CALLER is recorded and the
/// called move must not overwrite it. Moving the record point naively past the
/// Sleep Talk branch would have inverted exactly this.
#[test]
fn sleep_talk_records_itself_and_not_the_move_it_calls() {
    let mut state = tracked_state(Choices::SLEEPTALK);
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::TACKLE);
    state.side_one.get_active().status = PokemonStatus::SLEEP;
    state.side_one.get_active().sleep_turns = 0;

    let branches = generate(&mut state);
    assert_close(
        recorded_mass(&branches),
        100.0,
        "an asleep Sleep Talk user always records",
    );
    for branch in &branches {
        for instruction in &branch.instruction_list {
            if let Instruction::SetLastUsedMove(set) = instruction {
                if set.side_ref == SideReference::SideOne {
                    assert_eq!(
                        set.last_used_move,
                        LastUsedMove::Move(PokemonMoveIndex::M0),
                        "Sleep Talk must record ITSELF (M0), never the called move: {:?}",
                        branch.instruction_list
                    );
                }
            }
        }
    }
}
