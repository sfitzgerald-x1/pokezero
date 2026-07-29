//! Gen 3 Rest failure-condition pins, asserted directly against the vendored
//! gen3-patched poke-engine (`third_party/poke-engine-src/`).
//!
//! Ground truth is `rest.onTry` (data/moves.ts — no gen mod in the chain
//! overrides Rest, so gen3 inherits the base), which fails in this order:
//!
//! ```text
//! 1. status === 'slp' || hasAbility('comatose')  -> return false
//! 2. hp === maxhp   -> `-fail <user> heal`, return null
//! 3. hasAbility('insomnia')     -> `-fail ... ability: Insomnia`
//! 4. hasAbility('vitalspirit')  -> `-fail ... ability: Vital Spirit`
//! ```
//!
//! `poke-engine-gen3-rest-fullhp.patch` adds (2). (1) was already guarded; (3)
//! and (4) are unreachable in the gen3 randbats pool (Comatose is a gen7 ability
//! and no set pairs Rest with Insomnia or Vital Spirit) and are left unmodelled
//! per the reachability convention.
//!
//! The divergence this closes is status-level, not heal-level: a full-HP Rest
//! slept the user, and that sleep suppressed an incoming status Showdown lets
//! land — changing the legal action set for the rest of the battle.

use poke_engine::choices::Choices;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::MoveChoice;
use poke_engine::instruction::{Instruction, StateInstructions};
use poke_engine::state::{PokemonMoveIndex, PokemonStatus, SideReference, State};

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

fn only_branch(instructions: Vec<StateInstructions>) -> Vec<Instruction> {
    assert_eq!(
        instructions.len(),
        1,
        "expected a single deterministic branch, got {:?}",
        instructions
    );
    instructions.into_iter().next().unwrap().instruction_list
}

fn sets_status(list: &[Instruction], side_ref: SideReference, status: PokemonStatus) -> bool {
    list.iter().any(|instruction| match instruction {
        Instruction::ChangeStatus(change) => {
            change.side_ref == side_ref && change.new_status == status
        }
        _ => false,
    })
}

fn heals(list: &[Instruction], side_ref: SideReference) -> i16 {
    list.iter()
        .filter_map(|instruction| match instruction {
            Instruction::Heal(heal) if heal.side_ref == side_ref => Some(heal.heal_amount),
            _ => None,
        })
        .sum()
}

fn decrements_pp(list: &[Instruction], side_ref: SideReference) -> bool {
    list.iter().any(|instruction| match instruction {
        Instruction::DecrementPP(pp) => pp.side_ref == side_ref,
        _ => false,
    })
}

/// Side one rests at `hp`/`maxhp`; side two is inert unless given a move.
fn rest_state(hp: i16, maxhp: i16) -> State {
    let mut state = State::default();
    let active = state.side_one.get_active();
    active.maxhp = maxhp;
    active.hp = hp;
    active.replace_move(PokemonMoveIndex::M0, Choices::REST);
    state
}

/// Showdown gen3 ground truth: a full-HP Rest emits `-fail <user> heal` and
/// nothing else — no sleep, no heal. Upstream slept the user.
#[test]
fn rest_fails_at_full_hp() {
    let mut state = rest_state(246, 246);
    let list = only_branch(generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::None,
    ));

    assert!(
        !sets_status(&list, SideReference::SideOne, PokemonStatus::SLEEP),
        "a full-HP Rest must not sleep the user: {:?}",
        list
    );
    assert_eq!(
        heals(&list, SideReference::SideOne),
        0,
        "and must not heal: {:?}",
        list
    );
}

/// The failure still costs the turn and the PP. Showdown deducts PP in
/// `runMove` (battle-actions.ts:282) BEFORE `useMove` runs the `Try` event
/// (:585), so `-fail` is a spent turn — which is what makes this a real tempo
/// loss rather than a free action. The engine's PP decrement likewise sits well
/// before this dispatcher, so skipping the effect cannot skip the cost.
///
/// The engine only EMITS a DecrementPP instruction once the move is under 10 PP
/// (a deliberate instruction-count optimisation above that), so the pin sets the
/// slot low to exercise the path that is actually observable.
#[test]
fn a_failed_rest_still_costs_its_pp() {
    let mut state = rest_state(246, 246);
    state.side_one.get_active().moves[&PokemonMoveIndex::M0].pp = 5;
    let list = only_branch(generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::None,
    ));

    assert!(
        decrements_pp(&list, SideReference::SideOne),
        "a failed Rest is still a spent turn: {:?}",
        list
    );
    assert!(
        !sets_status(&list, SideReference::SideOne, PokemonStatus::SLEEP),
        "...and still fails: {:?}",
        list
    );
}

/// The control: one HP below full and Rest works exactly as before — sleeps for
/// 3 turns and heals to full.
#[test]
fn a_damaged_rest_still_works() {
    let mut state = rest_state(245, 246);
    let list = only_branch(generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::None,
    ));

    assert!(
        sets_status(&list, SideReference::SideOne, PokemonStatus::SLEEP),
        "Rest below full HP still sleeps: {:?}",
        list
    );
    assert_eq!(
        heals(&list, SideReference::SideOne),
        1,
        "and heals the missing HP: {:?}",
        list
    );
}

/// The boundary, both sides of it, across max-HP values.
#[test]
fn the_full_hp_boundary_is_exact() {
    for maxhp in [246i16, 261, 393, 461] {
        let mut full = rest_state(maxhp, maxhp);
        let full_list = only_branch(generate(
            &mut full,
            &MoveChoice::Move(PokemonMoveIndex::M0),
            &MoveChoice::None,
        ));
        assert!(
            !sets_status(&full_list, SideReference::SideOne, PokemonStatus::SLEEP),
            "{}/{} must fail",
            maxhp,
            maxhp
        );

        let mut one_off = rest_state(maxhp - 1, maxhp);
        let one_off_list = only_branch(generate(
            &mut one_off,
            &MoveChoice::Move(PokemonMoveIndex::M0),
            &MoveChoice::None,
        ));
        assert!(
            sets_status(&one_off_list, SideReference::SideOne, PokemonStatus::SLEEP),
            "{}/{} must work",
            maxhp - 1,
            maxhp
        );
    }
}

/// The pre-existing guard is untouched: an already-asleep user still fails
/// (`onTry` condition 1).
#[test]
fn rest_still_fails_when_already_asleep() {
    let mut state = rest_state(100, 246);
    state.side_one.get_active().status = PokemonStatus::SLEEP;
    let list = only_branch(generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::None,
    ));

    assert_eq!(
        heals(&list, SideReference::SideOne),
        0,
        "an asleep user's Rest does nothing: {:?}",
        list
    );
}

/// The repro sequence, which is the point of the fix: a full-HP Rest fails, so
/// the incoming Toxic LANDS. Upstream slept the user and the sleep suppressed
/// the status entirely — a divergence that changes the legal action set for the
/// rest of the battle, not just one turn's HP.
#[test]
fn a_failed_rest_lets_the_incoming_status_land() {
    let mut state = rest_state(246, 246);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::TOXIC);
    // Toxic branches on accuracy; take the branch that connects.
    let branches = generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    );
    let poisoned = branches.iter().any(|branch| {
        sets_status(
            &branch.instruction_list,
            SideReference::SideOne,
            PokemonStatus::TOXIC,
        )
    });
    assert!(
        poisoned,
        "the Toxic must land on the seat whose Rest failed: {:?}",
        branches
    );
    for branch in &branches {
        assert!(
            !sets_status(
                &branch.instruction_list,
                SideReference::SideOne,
                PokemonStatus::SLEEP
            ),
            "and no branch may sleep it: {:?}",
            branch.instruction_list
        );
    }
}
