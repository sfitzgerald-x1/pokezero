//! Gen 3 self-destruct fires only if the move actually EXECUTES.
//!
//! Showdown reaches `selfdestruct` inside `useMoveInner`
//! (`sim/battle-actions.ts:501`):
//!
//! ```text
//! if (this.battle.gen !== 4 && move.selfdestruct === 'always') {
//!     this.battle.faint(pokemon, pokemon, move);
//! }
//! ```
//!
//! and `runMove` calls `useMove` only after the `BeforeMove` gate returns true. Every
//! move-time immobilizer (full paralysis, sleep, freeze, flinch, confusion self-hit,
//! infatuation) returns `false` from its `onBeforeMove`, so **an immobilized Pokemon
//! never explodes**.
//!
//! The engine applied the faint in `choice_before_move`, which runs BEFORE
//! `generate_instructions_from_existing_status_conditions` rolls those immobilizers. So
//! the blocked branch killed its own user, and the branch's end-of-turn residuals then
//! correctly did not run — for a mon the engine had just wrongly killed.
//!
//! **The symptom was a missing Leftovers tick, and that is what the family was first
//! named for.** `an_ordinary_blocked_move_keeps_its_residuals` is the control that
//! refutes the "blocked branches drop the residual block" reading: they do not, and
//! never did. Only self-destruct branches looked that way, because only they had a
//! corpse.

use poke_engine::choices::Choices;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::instruction::{Instruction, StateInstructions};
use poke_engine::engine::items::Items;
use poke_engine::engine::state::MoveChoice;
use poke_engine::engine::abilities::Abilities;
use poke_engine::state::{PokemonMoveIndex, PokemonStatus, SideReference, State};

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


/// Side TWO is the seat under test: paralyzed, holding Leftovers, below max HP so a
/// residual tick is observable. Side one does something inert and moves first.
fn paralyzed_user(move_id: Choices) -> State {
    let mut state = State::default();
    state.side_one.get_active().speed = 500;
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::SPLASH);

    let two = state.side_two.get_active();
    two.speed = 1;
    two.status = PokemonStatus::PARALYZE;
    two.item = Items::LEFTOVERS;
    two.maxhp = 320;
    two.hp = 90;
    two.replace_move(PokemonMoveIndex::M0, move_id);
    state
}

fn heals_side_two(list: &[Instruction]) -> bool {
    list.iter().any(|i| match i {
        Instruction::Heal(heal) => heal.side_ref == SideReference::SideTwo,
        _ => false,
    })
}

fn faints_side_two(list: &[Instruction]) -> bool {
    // The self-destruct faint is a Damage instruction for the user's whole HP.
    list.iter().any(|i| match i {
        Instruction::Damage(dmg) => dmg.side_ref == SideReference::SideTwo && dmg.damage_amount == 90,
        _ => false,
    })
}

/// The blocked branch is the 25% full-paralysis one: it is the branch with no damage
/// dealt to side one.
fn blocked_branch(branches: &[StateInstructions]) -> &StateInstructions {
    branches
        .iter()
        .find(|b| (b.percentage - 25.0).abs() < 0.01)
        .expect("expected a 25% fully-paralyzed branch")
}

#[test]
fn the_blocked_branch_no_longer_kills_its_own_user() {
    // THE FIX. Unpatched this branch carries `Damage SideTwo: 90` — the user exploding
    // despite being fully paralyzed — and therefore no residual.
    let mut state = paralyzed_user(Choices::EXPLOSION);
    let branches = generate(&mut state);
    let blocked = blocked_branch(&branches);
    assert!(
        !faints_side_two(&blocked.instruction_list),
        "a fully-paralyzed Pokemon must not self-destruct: {:?}",
        blocked.instruction_list
    );
}

#[test]
fn the_blocked_branch_keeps_its_end_of_turn_residuals() {
    // The other half of the same fix, stated in the terms the divergence was reported
    // in: the missing Leftovers tick comes back once the user is no longer a corpse.
    let mut state = paralyzed_user(Choices::EXPLOSION);
    let branches = generate(&mut state);
    let blocked = blocked_branch(&branches);
    assert!(
        heals_side_two(&blocked.instruction_list),
        "blocked self-destruct branch must still run residuals: {:?}",
        blocked.instruction_list
    );
}

#[test]
fn an_ordinary_blocked_move_keeps_its_residuals() {
    // CONTROL — passes BOTH patched and unpatched, and it is the load-bearing one.
    // It establishes that blocked branches never dropped residuals as a class, so the
    // fix cannot be "re-add residuals to blocked branches". Anyone who reads the family
    // name and reaches for that fix should fail this test's premise, not pass it.
    let mut state = paralyzed_user(Choices::TACKLE);
    let branches = generate(&mut state);
    let blocked = blocked_branch(&branches);
    assert!(
        heals_side_two(&blocked.instruction_list),
        "an ordinary blocked move has always kept its residuals: {:?}",
        blocked.instruction_list
    );
}

#[test]
fn the_firing_branch_still_faints_the_user_and_still_skips_residuals() {
    // The relocation must not change the branch where the move DOES execute: the user
    // still faints, still before damage, and a dead mon still takes no residual.
    let mut state = paralyzed_user(Choices::EXPLOSION);
    let branches = generate(&mut state);
    let firing = branches
        .iter()
        .find(|b| (b.percentage - 75.0).abs() < 0.01)
        .expect("expected a 75% move-executes branch");
    assert!(
        faints_side_two(&firing.instruction_list),
        "the executing branch must still self-destruct: {:?}",
        firing.instruction_list
    );
    assert!(
        !heals_side_two(&firing.instruction_list),
        "a fainted user takes no residual: {:?}",
        firing.instruction_list
    );
}

#[test]
fn damp_still_prevents_the_faint() {
    // The DAMP guard travelled with the relocation. Showdown's Damp prevents the move
    // outright via `onAnyTryMove`, so the user does not faint.
    let mut state = paralyzed_user(Choices::EXPLOSION);
    state.side_one.get_active().ability = Abilities::DAMP;
    let branches = generate(&mut state);
    for branch in &branches {
        assert!(
            !faints_side_two(&branch.instruction_list),
            "Damp must prevent the self-destruct faint on every branch: {:?}",
            branch.instruction_list
        );
    }
}

#[test]
fn a_healthy_user_still_explodes() {
    // Guard against over-gating: with no status at all there is one branch and it
    // self-destructs exactly as before.
    let mut state = paralyzed_user(Choices::EXPLOSION);
    state.side_two.get_active().status = PokemonStatus::NONE;
    let branches = generate(&mut state);
    assert!(
        branches
            .iter()
            .any(|b| faints_side_two(&b.instruction_list)),
        "an unimpeded self-destruct must still faint its user"
    );
}
