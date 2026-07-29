//! Gen 3 ENCORE application-failure pins, asserted directly against the vendored
//! gen3-patched poke-engine (`third_party/poke-engine-src/`).
//!
//! Companion to `gen3_encore_fidelity.rs`: that file pins how long an Encore
//! lasts, this one pins when it may start at all. Upstream asked only whether the
//! target had ever used a move, so the engine would happily Encore a Struggle.
//!
//! Showdown's `encore.condition.onStart` (base `data/moves.ts`; no gen3-chain mod
//! overrides `onStart` itself):
//!
//! ```text
//! let move = target.lastMove;
//! if (!move || target.volatiles['dynamax']) return false;
//! const moveSlot = target.getMoveData(move.id);
//! if (move.isZ || move.isMax || move.flags['failencore'] ||
//!     !moveSlot || moveSlot.pp <= 0) return false;
//! ```
//!
//! `dynamax` / `isZ` / `isMax` are unreachable in gen3. `!moveSlot` is
//! structurally unreachable in the engine, which stores `last_used_move` as a
//! SLOT INDEX rather than a move id, so it always denotes a real slot — the case
//! it catches in Showdown is Struggle, which is not in `moveSlots`, and which the
//! `failencore` list rejects anyway.
//!
//! The `failencore` set is resolved against gen3's ACTUAL chain
//! (gen3 -> gen4 -> gen5 -> gen6 -> gen7 -> gen8 -> base), where a mod's `flags`
//! object REPLACES its parent's wholesale:
//!
//! * `mimic`, `mirrormove`, `sketch`, `struggle` — flags from `data/mods/gen3`
//! * `encore`, `transform` — flags from `data/mods/gen4`
//!
//! and the inverse cases matter just as much: `assist` (gen3), `metronome` and
//! `naturepower` (gen4) and `sleeptalk` (gen6) all carry `failencore` in BASE but
//! lose it to a nearer override, so none of them fails Encore in gen3.

use poke_engine::choices::Choices;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::{MoveChoice, PokemonVolatileStatus};
use poke_engine::instruction::{Instruction, StateInstructions};
use poke_engine::state::{
    LastUsedMove, PokemonIndex, PokemonMoveIndex, SideReference, State,
};

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

/// Side one Encores; side two is the target, holding `target_last_move` in slot
/// M0 and having last used it. Side one moves first so the application turn is
/// the clean case (no `onStart` duration bump).
fn encore_attempt_state(target_last_move: Choices) -> State {
    let mut state = State::default();
    state.side_one.get_active().speed = 500;
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::ENCORE);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, target_last_move);
    state.side_two.last_used_move = LastUsedMove::Move(PokemonMoveIndex::M0);
    state
}

fn applies_encore(list: &[Instruction]) -> bool {
    list.iter().any(|instruction| match instruction {
        Instruction::ApplyVolatileStatus(apply) => {
            apply.side_ref == SideReference::SideTwo
                && apply.volatile_status == PokemonVolatileStatus::ENCORE
        }
        _ => false,
    })
}

fn encore_lands(state: &mut State) -> bool {
    let branches = generate(
        state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    );
    branches
        .iter()
        .any(|branch| applies_encore(&branch.instruction_list))
}

// ---------------------------------------------------------------------------
// The failencore set
// ---------------------------------------------------------------------------

/// The six moves that carry `flags.failencore` as gen3 resolves them. Encoring
/// any of them fails outright — most importantly Struggle, which upstream
/// happily encored.
#[test]
fn encore_fails_against_every_gen3_failencore_move() {
    for move_id in [
        Choices::STRUGGLE,
        Choices::MIMIC,
        Choices::MIRRORMOVE,
        Choices::SKETCH,
        Choices::TRANSFORM,
        Choices::ENCORE,
    ] {
        let mut state = encore_attempt_state(move_id);
        assert!(
            !encore_lands(&mut state),
            "Encore must fail against {:?}, which carries failencore in gen3",
            move_id
        );
    }
}

/// The inverse trap, and the reason the list was resolved per-level instead of
/// read off base `data/moves.ts`. All four of these DO carry `failencore` in
/// base, but a nearer override in gen3's chain re-declares `flags` without it —
/// assist at gen3, metronome and naturepower at gen4, sleeptalk at gen6 — so in
/// gen3 they are ordinary encorable moves. A list copied from base (or from
/// gen8, which re-adds the flag to sleeptalk) would wrongly reject all four.
#[test]
fn encore_succeeds_against_moves_that_lose_failencore_in_gen3() {
    for move_id in [
        Choices::ASSIST,
        Choices::METRONOME,
        Choices::NATUREPOWER,
        Choices::SLEEPTALK,
    ] {
        let mut state = encore_attempt_state(move_id);
        assert!(
            encore_lands(&mut state),
            "Encore must SUCCEED against {:?}: it carries failencore in base \
             data/moves.ts but loses it to a nearer override in gen3's chain",
            move_id
        );
    }
}

/// Control for both lists above: an ordinary move is encorable, so the failures
/// are about the flag and not about Encore being broken outright.
#[test]
fn encore_succeeds_against_an_ordinary_move() {
    let mut state = encore_attempt_state(Choices::TACKLE);
    assert!(encore_lands(&mut state), "Encore must land on a plain move");
}

// ---------------------------------------------------------------------------
// The PP arm
// ---------------------------------------------------------------------------

/// `moveSlot.pp <= 0` — Encore cannot lock a target into a move it can no longer
/// use. Upstream never checked, so search could commit a seat to an unusable
/// move. `<= 0` rather than `== 0` matches Showdown and survives the engine's own
/// decrement taking an already-empty slot negative.
#[test]
fn encore_fails_when_the_targets_last_move_has_no_pp_left() {
    for pp in [0, -1] {
        let mut state = encore_attempt_state(Choices::TACKLE);
        state.side_two.get_active().moves[&PokemonMoveIndex::M0].pp = pp;
        assert!(
            !encore_lands(&mut state),
            "Encore must fail against a move at {} PP",
            pp
        );
    }
}

/// Boundary control: one PP left is still usable, so Encore lands.
#[test]
fn encore_succeeds_when_the_targets_last_move_has_one_pp_left() {
    let mut state = encore_attempt_state(Choices::TACKLE);
    state.side_two.get_active().moves[&PokemonMoveIndex::M0].pp = 1;
    assert!(
        encore_lands(&mut state),
        "Encore must land while the move still has PP"
    );
}

// ---------------------------------------------------------------------------
// The "no last move" arms (upstream behaviour, pinned alongside)
// ---------------------------------------------------------------------------

/// `if (!move) return false`. These two arms were already right upstream and are
/// pinned here because they live in the same match the patch rewrites.
#[test]
fn encore_fails_when_the_target_has_never_moved() {
    let mut state = encore_attempt_state(Choices::TACKLE);
    state.side_two.last_used_move = LastUsedMove::None;
    assert!(
        !encore_lands(&mut state),
        "Encore must fail against a target with no last move"
    );
}

/// A fresh switch-in genuinely has no last move: `Pokemon.clearVolatile()` nulls
/// `lastMove` on switch-out, so Showdown's `!move` arm fires.
#[test]
fn encore_fails_against_a_pokemon_that_just_switched_in() {
    let mut state = encore_attempt_state(Choices::TACKLE);
    state.side_two.last_used_move = LastUsedMove::Switch(PokemonIndex::P1);
    assert!(
        !encore_lands(&mut state),
        "Encore must fail against a Pokemon that just switched in"
    );
}

/// A failed Encore is a genuine no-op: it must not half-apply, and the target
/// must keep every one of its move options rather than being silently locked.
#[test]
fn a_failed_encore_leaves_the_target_completely_unrestricted() {
    let mut state = encore_attempt_state(Choices::STRUGGLE);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::TACKLE);

    let branches = generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    );
    for branch in &branches {
        assert!(
            !applies_encore(&branch.instruction_list),
            "no branch may apply Encore: {:?}",
            branch.instruction_list
        );
        state.apply_instructions(&branch.instruction_list);
        assert!(
            !state
                .side_two
                .volatile_statuses
                .contains(&PokemonVolatileStatus::ENCORE),
            "a failed Encore must leave no volatile behind"
        );
        assert_eq!(
            state.side_two.volatile_status_durations.encore, 0,
            "a failed Encore must not seed the duration ladder"
        );
        state.reverse_instructions(&branch.instruction_list);
    }

    let (_, side_two_options) = state.get_all_options();
    assert!(
        side_two_options.contains(&MoveChoice::Move(PokemonMoveIndex::M1)),
        "the target keeps its other moves after a failed Encore: {:?}",
        side_two_options
    );
}
