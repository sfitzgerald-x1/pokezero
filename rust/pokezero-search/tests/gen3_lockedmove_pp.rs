//! Gen 3: a LOCKED continuation turn costs no PP. Asserted against the vendored
//! gen3-patched poke-engine.
//!
//! Showdown guards the deduction on `getLockedMove()`
//! (sim/battle-actions.ts:280-283):
//!
//! ```text
//! const lockedMove = pokemon.getLockedMove();
//! if (!lockedMove) {
//!     if (!pokemon.deductPP(baseMove, null, target) && move.id !== 'struggle') { ... }
//! } else {
//!     sourceEffect = this.dex.conditions.get('lockedmove');
//! }
//! ```
//!
//! `getLockedMove` fires the `LockMove` priority event. Its providers in gen3's
//! chain are the `lockedmove` condition (Outrage / Thrash / Petal Dance,
//! `data/conditions.ts:282`), `twoturnmove` (Solar Beam, Sky Attack, Dig, Fly,
//! Razor Wind, Skull Bash, `data/conditions.ts:317`), `mustrecharge`
//! (`data/conditions.ts:377`), and `rollout` / `bide` in `data/moves.ts`. So the
//! whole lock costs ONE PP, charged on the turn that starts it — which is why
//! Showdown tags the second half of a two-turn move `[from] lockedmove`.
//!
//! Reachability note, and the reason this is worth landing: the `lockedmove`
//! TRIO is absent from the gen3 randbats pool (0 carriers each for Outrage,
//! Thrash and Petal Dance). The reachable half of the same bug is `twoturnmove`
//! — Solar Beam is on 4 gen3 randbats species — where the engine charged two PP
//! per use instead of one. Both are fixed by the same guard, and both are pinned
//! here: the trio because the engine models it and a future format could reach
//! it, Solar Beam because today's format does.

use poke_engine::choices::Choices;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::{MoveChoice, PokemonVolatileStatus};
use poke_engine::instruction::{Instruction, StateInstructions};
use poke_engine::state::{PokemonMoveIndex, SideReference, State};

const START_PP: i8 = 5;

fn generate(state: &mut State) -> Vec<StateInstructions> {
    generate_instructions_from_move_pair(
        state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
        false,
    )
}

/// Side one uses `attacker_move`; side two is an inert Splash user that survives.
fn locked_state(attacker_move: Choices) -> State {
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
    // Bulky enough that nothing faints mid-lock and derails the sequence.
    state.side_two.get_active().hp = 10000;
    state.side_two.get_active().maxhp = 10000;
    state
}

fn charges_pp(list: &[Instruction]) -> bool {
    list.iter().any(|instruction| match instruction {
        Instruction::DecrementPP(decrement) => decrement.side_ref == SideReference::SideOne,
        _ => false,
    })
}

/// Walk the branch that keeps the lock going, returning whether each turn
/// charged PP. Picks the branch that did NOT self-hit or otherwise diverge, so
/// the sequence follows one coherent line.
fn walk(state: &mut State, turns: usize) -> Vec<bool> {
    let mut charged = Vec::new();
    for _ in 0..turns {
        let branches = generate(state);
        let branch = branches
            .iter()
            .max_by(|a, b| a.percentage.partial_cmp(&b.percentage).unwrap())
            .expect("at least one branch")
            .clone();
        charged.push(charges_pp(&branch.instruction_list));
        state.apply_instructions(&branch.instruction_list);
    }
    charged
}

// ---------------------------------------------------------------------------
// Two-turn moves: the reachable half
// ---------------------------------------------------------------------------

/// Solar Beam is on 4 gen3 randbats species, so this is the row that actually
/// bites. Charge turn pays, execute turn is free: one PP for the pair, matching
/// Showdown's 8-PP Sky Attack yielding 8 complete uses.
#[test]
fn a_two_turn_move_costs_one_pp_for_the_pair() {
    let mut state = locked_state(Choices::SOLARBEAM);
    assert_eq!(
        walk(&mut state, 2),
        vec![true, false],
        "charge turn charges, execute turn is a locked continuation"
    );
    assert_eq!(
        state.side_one.get_active().moves[&PokemonMoveIndex::M0].pp,
        START_PP - 1,
        "one complete Solar Beam costs exactly one PP"
    );
}

/// And the next use starts a fresh lock, so it pays again — the guard is about
/// being mid-lock, not about the move having ever been locked.
#[test]
fn the_next_two_turn_use_pays_again() {
    let mut state = locked_state(Choices::SOLARBEAM);
    assert_eq!(walk(&mut state, 4), vec![true, false, true, false]);
    assert_eq!(
        state.side_one.get_active().moves[&PokemonMoveIndex::M0].pp,
        START_PP - 2,
        "two complete Solar Beams cost exactly two PP"
    );
}

/// Sky Attack is the move the differential drives, pinned natively on the same
/// shape so the two halves of the evidence line up.
#[test]
fn sky_attack_also_costs_one_pp_for_the_pair() {
    let mut state = locked_state(Choices::SKYATTACK);
    assert_eq!(walk(&mut state, 2), vec![true, false]);
    assert_eq!(
        state.side_one.get_active().moves[&PokemonMoveIndex::M0].pp,
        START_PP - 1
    );
}

// ---------------------------------------------------------------------------
// The lockedmove trio: modelled by the engine, unreachable in randbats
// ---------------------------------------------------------------------------

/// Outrage charges once for the whole lock. The engine used to charge on the
/// continuation turns too. Not reachable in the current randbats pool (0
/// carriers), but the engine models the lock, so the row is pinned rather than
/// left to rot.
#[test]
fn a_locked_move_charges_only_on_the_turn_that_starts_it() {
    let mut state = locked_state(Choices::OUTRAGE);
    let charged = walk(&mut state, 3);
    assert!(charged[0], "the turn that starts the lock pays");
    assert!(
        !charged[1] && !charged[2],
        "continuation turns are free: {:?}",
        charged
    );
    assert_eq!(
        state.side_one.get_active().moves[&PokemonMoveIndex::M0].pp,
        START_PP - 1,
        "the whole Outrage lock costs exactly one PP"
    );
}

/// The guard keys on the LOCKEDMOVE volatile, so a turn on which it is present
/// is free regardless of how the state got there — this is the mid-lock case the
/// original empirical check exercised.
#[test]
fn a_mid_lock_turn_is_free_even_when_the_state_is_constructed_directly() {
    let mut state = locked_state(Choices::OUTRAGE);
    state
        .side_one
        .volatile_statuses
        .insert(PokemonVolatileStatus::LOCKEDMOVE);
    state.side_one.volatile_status_durations.lockedmove = 1;

    let branches = generate(&mut state);
    for branch in &branches {
        assert!(
            !charges_pp(&branch.instruction_list),
            "a mid-lock turn must not charge PP: {:?}",
            branch.instruction_list
        );
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

/// An ordinary move is unaffected: it pays every single turn, so the guard has
/// not simply switched the deduction off.
#[test]
fn an_ordinary_move_still_pays_every_turn() {
    let mut state = locked_state(Choices::TACKLE);
    assert_eq!(walk(&mut state, 3), vec![true, true, true]);
    assert_eq!(
        state.side_one.get_active().moves[&PokemonMoveIndex::M0].pp,
        START_PP - 3
    );
}

/// A Pokemon can START a two-turn move on its last PP and still complete the
/// lock, because the execute turn never consults PP again. Verified against
/// Showdown rather than assumed: `getLockedMove()` short-circuits the whole
/// deduction-and-abort block, so there is no second chance to fail on `pp <= 0`.
#[test]
fn a_two_turn_move_started_on_the_last_pp_still_completes() {
    let mut state = locked_state(Choices::SOLARBEAM);
    state.side_one.get_active().moves[&PokemonMoveIndex::M0].pp = 1;

    let charged = walk(&mut state, 2);
    assert_eq!(charged, vec![true, false], "charge pays, execute is free");
    assert_eq!(
        state.side_one.get_active().moves[&PokemonMoveIndex::M0].pp,
        0,
        "the slot is empty but the lock completed"
    );
}
