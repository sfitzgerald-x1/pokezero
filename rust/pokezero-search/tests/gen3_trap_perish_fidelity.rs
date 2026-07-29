//! Gen 3 move-trapping and Perish Song fidelity pins, asserted directly against
//! the vendored gen3-patched poke-engine (`third_party/poke-engine-src/`).
//!
//! Companion to `gen3_switch_fidelity.rs` and `gen3_hazard_residual_fidelity.rs`:
//! every expectation here was read off the **real** Node Showdown simulator
//! driven through `scripts/gen3_switch_differential.py`, which is the
//! ground-truth gate; this file is the engine-contract pin, so a version bump
//! that silently drops one of the `third_party/poke-engine-gen3-*.patch` files
//! fails `cargo test` instead of quietly regressing search fidelity.
//!
//! Coverage:
//!
//! * Mean Look / Spider Web / Block trap the target, the victim loses every
//!   switch option, Protect and Substitute block the trap, the trapper leaving
//!   frees the victim, and Baton Pass carries the trap in gen3 —
//!   `poke-engine-gen3-move-trapping.patch`.
//! * The Perish Song ladder faints on the correct ply. This is a VERDICT pin,
//!   not a fix: the engine already agrees with the sim once the residual block
//!   is deferred across a forced replacement, and this file exists so a future
//!   change to either half fails loudly.

use poke_engine::choices::Choices;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::{MoveChoice, PokemonVolatileStatus};
use poke_engine::instruction::{Instruction, StateInstructions};
use poke_engine::state::{PokemonIndex, PokemonMoveIndex, SideReference, State};

/// `generate_instructions_from_move_pair` must leave `state` untouched — the
/// engine's own test suite asserts this and a patch that mutates without
/// emitting a reversible instruction would silently corrupt search.
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

fn only_branch(instructions: Vec<StateInstructions>) -> Vec<Instruction> {
    assert_eq!(
        instructions.len(),
        1,
        "expected a single deterministic branch, got {:?}",
        instructions
    );
    instructions.into_iter().next().unwrap().instruction_list
}

/// Apply then reverse: search relies on every emitted instruction being an exact
/// inverse, so a new volatile that is not undone corrupts the tree.
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

fn applies_volatile(
    list: &[Instruction],
    side_ref: SideReference,
    volatile_status: PokemonVolatileStatus,
) -> bool {
    list.iter().any(|instruction| match instruction {
        Instruction::ApplyVolatileStatus(apply) => {
            apply.side_ref == side_ref && apply.volatile_status == volatile_status
        }
        _ => false,
    })
}

fn removes_volatile(
    list: &[Instruction],
    side_ref: SideReference,
    volatile_status: PokemonVolatileStatus,
) -> bool {
    list.iter().any(|instruction| match instruction {
        Instruction::RemoveVolatileStatus(remove) => {
            remove.side_ref == side_ref && remove.volatile_status == volatile_status
        }
        _ => false,
    })
}

fn has_switch_option(options: &[MoveChoice]) -> bool {
    options
        .iter()
        .any(|option| matches!(option, MoveChoice::Switch(_)))
}

// ---------------------------------------------------------------------------
// Move-trapping (poke-engine-gen3-move-trapping.patch)
// ---------------------------------------------------------------------------

/// Side one holds the trapping move on M0; side two answers with `defender_move`.
fn trap_state(trapping_move: Choices, defender_move: Choices) -> State {
    let mut state = State::default();
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, trapping_move);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, defender_move);
    state
}

fn trap_the_victim(trapping_move: Choices) -> (State, Vec<Instruction>) {
    let mut state = trap_state(trapping_move, Choices::SPLASH);
    let list = only_branch(generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    ));
    assert_reverts_cleanly(&mut state, &list);
    (state, list)
}

/// Showdown gen3 ground truth: `|move|p1a: Umbreon|Mean Look|p2a: Snorlax`
/// followed by `|-activate|p2a: Snorlax|trapped`, and the victim's very next
/// request carries `trapped: true` with no switch options.
///
/// Upstream defined all three trapping moves as pure no-ops — no volatile, no
/// effect — so the engine believed Mean Look did nothing at all.
#[test]
fn the_trapping_moves_apply_the_trap_to_the_target() {
    for trapping_move in [Choices::MEANLOOK, Choices::SPIDERWEB, Choices::BLOCK] {
        let (_state, list) = trap_the_victim(trapping_move);
        assert!(
            applies_volatile(
                &list,
                SideReference::SideTwo,
                PokemonVolatileStatus::TRAPPED
            ),
            "{:?} must trap the target: {:?}",
            trapping_move,
            list
        );
    }
}

/// The trap is only worth anything if it actually removes the victim's switches:
/// `trapped.onTrapPokemon` calls `pokemon.tryTrap()`, and Showdown then offers
/// the seat no switch slots at all. The trapper keeps its own switches.
#[test]
fn a_trapped_pokemon_is_offered_no_switch_options() {
    let (mut state, list) = trap_the_victim(Choices::MEANLOOK);
    state.apply_instructions(&list);

    let (side_one_options, side_two_options) = state.get_all_options();
    assert!(
        has_switch_option(&side_one_options),
        "the trapper keeps its own switches: {:?}",
        side_one_options
    );
    assert!(
        !has_switch_option(&side_two_options),
        "the trapped side must be offered no switches: {:?}",
        side_two_options
    );
}

/// Gen 3 adds the protect flag to Mean Look and Block (`data/mods/gen5/moves.ts`,
/// inherited down to gen3); Spider Web carries it in the base data. The sim
/// answers `|-activate|p2a: Snorlax|Protect` and the target is NOT trapped.
#[test]
fn protect_blocks_the_trap() {
    for trapping_move in [Choices::MEANLOOK, Choices::SPIDERWEB, Choices::BLOCK] {
        let mut state = trap_state(trapping_move, Choices::PROTECT);
        // The defender must move first for its Protect to be up.
        state.side_two.get_active().speed = 500;
        let list = only_branch(generate(
            &mut state,
            &MoveChoice::Move(PokemonMoveIndex::M0),
            &MoveChoice::Move(PokemonMoveIndex::M0),
        ));
        assert!(
            !applies_volatile(
                &list,
                SideReference::SideTwo,
                PokemonVolatileStatus::TRAPPED
            ),
            "Protect must block {:?}: {:?}",
            trapping_move,
            list
        );
    }
}

/// None of the three carries `bypasssub`, so a Substitute blocks the trap
/// outright — the sim answers `|move|p1a: Umbreon|Mean Look||[still]` +
/// `|-fail|p1a: Umbreon`.
#[test]
fn a_substitute_blocks_the_trap() {
    let mut state = trap_state(Choices::MEANLOOK, Choices::SPLASH);
    state
        .side_two
        .volatile_statuses
        .insert(PokemonVolatileStatus::SUBSTITUTE);

    let list = only_branch(generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    ));

    assert!(
        !applies_volatile(
            &list,
            SideReference::SideTwo,
            PokemonVolatileStatus::TRAPPED
        ),
        "a Substitute must block the trap: {:?}",
        list
    );
}

/// `addVolatile('trapped', source, move, 'trapper')` LINKS the two volatiles, so
/// the trapper leaving the field runs `clearVolatile()` ->
/// `removeLinkedVolatiles()` and frees the victim. Verified against the sim: the
/// victim's request comes back untrapped on the very next boundary, both when
/// the trapper switches out and when it faints.
#[test]
fn the_trap_ends_when_the_trapper_leaves() {
    let (mut state, trap) = trap_the_victim(Choices::MEANLOOK);
    state.apply_instructions(&trap);

    let list = only_branch(generate(
        &mut state,
        &MoveChoice::Switch(PokemonIndex::P1),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    ));

    assert!(
        removes_volatile(
            &list,
            SideReference::SideTwo,
            PokemonVolatileStatus::TRAPPED
        ),
        "the trapper leaving must free the victim: {:?}",
        list
    );
    assert_reverts_cleanly(&mut state, &list);

    state.apply_instructions(&list);
    let (_, side_two_options) = state.get_all_options();
    assert!(
        has_switch_option(&side_two_options),
        "the freed victim gets its switches back: {:?}",
        side_two_options
    );
}

/// Gen 3 (and only gen 3/4) copies the trap through a Baton Pass:
/// `data/mods/gen4/conditions.ts` re-declares `trapped` with `noCopy: false`,
/// which gen5+ flips back. Verified against the sim: a trapped Misdreavus that
/// Baton Passes into Blissey hands the RECEIVER a `trapped: true` request.
#[test]
fn baton_pass_carries_the_trap() {
    let mut state = trap_state(Choices::MEANLOOK, Choices::SPLASH);
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::SPLASH);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::BATONPASS);

    let trap = only_branch(generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    ));
    state.apply_instructions(&trap);

    // A trapped Pokemon may still USE Baton Pass — the trap removes the switch
    // OPTION, not the move.
    let pass = only_branch(generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M1),
        &MoveChoice::Move(PokemonMoveIndex::M1),
    ));
    state.apply_instructions(&pass);
    assert!(
        state.side_two.baton_passing,
        "Baton Pass must arm the pass before the switch resolves: {:?}",
        pass
    );

    let list = only_branch(generate(
        &mut state,
        &MoveChoice::None,
        &MoveChoice::Switch(PokemonIndex::P1),
    ));
    state.apply_instructions(&list);
    assert!(
        state
            .side_two
            .volatile_statuses
            .contains(&PokemonVolatileStatus::TRAPPED),
        "the receiver must arrive still trapped: {:?}",
        list
    );

    let (_, side_two_options) = state.get_all_options();
    assert!(
        !has_switch_option(&side_two_options),
        "and must itself be offered no switches: {:?}",
        side_two_options
    );
}

/// The mirror case, and the one the randbats set is actually built around: the
/// TRAPPER Baton Passes. This does NOT free the victim.
///
/// `copyVolatileFrom` copies `trapper` to the receiver (gen3 inherits gen4's
/// `noCopy: false` for BOTH halves of the link), then DELETES the old trapper's
/// `linkedPokemon`/`linkedStatus`, and only then runs `pokemon.clearVolatile()`
/// — which therefore finds no link left to release. The victim's own link is
/// re-pointed to the receiver, so the trap changes owner. Verified against real
/// gen3 Showdown: after an Ariados webs and passes, the victim's request is
/// still `trapped: true`, and it only comes back untrapped once the RECEIVER
/// leaves. 2 of the 3 gen3 randbats Ariados sets carry Spider Web + Baton Pass,
/// so this is the designed line, not a corner case.
#[test]
fn a_trapper_that_baton_passes_keeps_the_victim_trapped() {
    let mut state = trap_state(Choices::MEANLOOK, Choices::SPLASH);
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::BATONPASS);

    let trap = only_branch(generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    ));
    state.apply_instructions(&trap);

    let pass = only_branch(generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M1),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    ));
    state.apply_instructions(&pass);
    assert!(
        state.side_one.baton_passing,
        "Baton Pass must arm the pass before the switch resolves: {:?}",
        pass
    );

    let switch = only_branch(generate(
        &mut state,
        &MoveChoice::Switch(PokemonIndex::P1),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    ));
    assert!(
        !removes_volatile(
            &switch,
            SideReference::SideTwo,
            PokemonVolatileStatus::TRAPPED
        ),
        "the trapper's Baton Pass must NOT release the victim: {:?}",
        switch
    );
    assert_reverts_cleanly(&mut state, &switch);
    state.apply_instructions(&switch);

    let (_, side_two_options) = state.get_all_options();
    assert!(
        !has_switch_option(&side_two_options),
        "the victim is still stuck on the receiver: {:?}",
        side_two_options
    );

    // ...and the RECEIVER's own departure is what frees it, because the link was
    // re-pointed rather than dropped.
    let receiver_leaves = only_branch(generate(
        &mut state,
        &MoveChoice::Switch(PokemonIndex::P2),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    ));
    assert!(
        removes_volatile(
            &receiver_leaves,
            SideReference::SideTwo,
            PokemonVolatileStatus::TRAPPED
        ),
        "the receiver leaving must free the victim: {:?}",
        receiver_leaves
    );
    state.apply_instructions(&receiver_leaves);
    let (_, freed_options) = state.get_all_options();
    assert!(
        has_switch_option(&freed_options),
        "the freed victim gets its switches back: {:?}",
        freed_options
    );
}

/// The gate is on the TRAPPED link specifically. Wrap/Fire Spin is not a linked
/// volatile — `partiallytrapped.onResidual` releases on `!source.isActive`,
/// which is true however the source left — so a Baton Pass still frees a
/// partial-trap victim, and that pre-existing release must stay unconditional.
#[test]
fn a_baton_pass_still_ends_a_partial_trap() {
    let mut state = State::default();
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::BATONPASS);
    state
        .side_two
        .volatile_statuses
        .insert(PokemonVolatileStatus::PARTIALLYTRAPPED);

    let pass = only_branch(generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::None,
    ));
    state.apply_instructions(&pass);

    let switch = only_branch(generate(
        &mut state,
        &MoveChoice::Switch(PokemonIndex::P1),
        &MoveChoice::None,
    ));
    assert!(
        removes_volatile(
            &switch,
            SideReference::SideTwo,
            PokemonVolatileStatus::PARTIALLYTRAPPED
        ),
        "a partial trap ends however the trapper left: {:?}",
        switch
    );
}

/// Control for the pin above: gen3's `noCopy: false` widens what a pass carries,
/// it does not stop an ordinary switch-out from clearing the trap. The reachable
/// route for a trapped Pokemon to leave without passing is being phazed out
/// (Roar / Whirlwind), which is `Pokemon.clearVolatile()` like any other switch.
#[test]
fn a_plain_switch_out_drops_the_trap() {
    let mut state = State::default();
    state
        .side_two
        .volatile_statuses
        .insert(PokemonVolatileStatus::TRAPPED);

    let list = only_branch(generate(
        &mut state,
        &MoveChoice::None,
        &MoveChoice::Switch(PokemonIndex::P1),
    ));

    assert!(
        removes_volatile(
            &list,
            SideReference::SideTwo,
            PokemonVolatileStatus::TRAPPED
        ),
        "a plain switch-out must clear the trap: {:?}",
        list
    );
    assert_reverts_cleanly(&mut state, &list);
}

// ---------------------------------------------------------------------------
// Perish Song ladder — VERDICT pin (no engine change)
// ---------------------------------------------------------------------------

/// Showdown gen3 ground truth, read off the real sim
/// (`scripts/gen3_switch_differential.py::perishladder`): Perish Song is used on
/// turn 1 and the residual block of that SAME turn announces `perish3`, then
/// `perish2`, then `perish1`, then `perish0` + `|faint|` — the faint lands on
/// the FOURTH end-of-turn block counting the move's own turn, and both seats
/// replace at one shared boundary.
///
/// The engine ladder is offset by one name (PERISH4 is applied by the move and
/// ticks to PERISH3 in that same block), so engine PERISH<n> is exactly
/// Showdown's `perish<n>` at every boundary and PERISH1 is the lethal tick.
/// There is no missing final tick: the `perish0` protocol line has no engine
/// counterpart because the engine faints from PERISH1 directly, which is a
/// protocol-emission difference, not a mechanical one.
///
/// This pin exists because the ladder only lines up while the end-of-turn block
/// runs on every ply — a turn that skips or doubles it slips the faint by a full
/// turn. Both halves it depends on (the Baton Pass carry and the residual
/// deferral across a forced replacement) landed recently.
#[test]
fn the_perish_song_ladder_faints_on_the_fourth_end_of_turn() {
    let mut state = State::default();
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::PERISHSONG);

    let sing = only_branch(generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::None,
    ));
    assert_reverts_cleanly(&mut state, &sing);
    state.apply_instructions(&sing);
    for side in [&state.side_one, &state.side_two] {
        assert!(
            side.volatile_statuses
                .contains(&PokemonVolatileStatus::PERISH3),
            "the singing turn's own residual leaves both seats on Showdown's \
             perish3: {:?}",
            sing
        );
    }

    for expected in [
        PokemonVolatileStatus::PERISH2,
        PokemonVolatileStatus::PERISH1,
    ] {
        let idle = only_branch(generate(&mut state, &MoveChoice::None, &MoveChoice::None));
        assert_reverts_cleanly(&mut state, &idle);
        state.apply_instructions(&idle);
        for side in [&state.side_one, &state.side_two] {
            assert!(
                side.volatile_statuses.contains(&expected),
                "the counter must tick to {:?}: {:?}",
                expected,
                idle
            );
        }
    }

    // Fourth block: the lethal tick.
    let lethal = only_branch(generate(&mut state, &MoveChoice::None, &MoveChoice::None));
    assert_reverts_cleanly(&mut state, &lethal);
    state.apply_instructions(&lethal);
    assert_eq!(
        state.side_one.get_active_immutable().hp,
        0,
        "the singer faints on the fourth end-of-turn block: {:?}",
        lethal
    );
    assert_eq!(
        state.side_two.get_active_immutable().hp,
        0,
        "so does the target: {:?}",
        lethal
    );

    // Showdown offers both replacements at one shared boundary.
    let (side_one_options, side_two_options) = state.get_all_options();
    assert!(
        has_switch_option(&side_one_options) && has_switch_option(&side_two_options),
        "the perish double faint replaces both seats at one boundary: {:?} / {:?}",
        side_one_options,
        side_two_options
    );
}

/// The ladder must not slip when the residual block is deferred across a forced
/// replacement: the deferred block still carries the perish tick, so a counter
/// sitting on a seat whose OPPONENT faints mid-turn still advances exactly once
/// that turn.
#[test]
fn a_deferred_residual_block_still_advances_the_perish_counter() {
    let mut state = State::default();
    state
        .side_one
        .volatile_statuses
        .insert(PokemonVolatileStatus::PERISH3);
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::SWIFT);
    state.side_two.get_active().hp = 1;

    // The KO defers the block entirely — the counter must NOT tick here.
    let faint = only_branch(generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::None,
    ));
    state.apply_instructions(&faint);
    assert!(
        state
            .side_one
            .volatile_statuses
            .contains(&PokemonVolatileStatus::PERISH3),
        "the deferred ply must not tick the counter: {:?}",
        faint
    );

    // ...and exactly once on the ply that resolves the replacement.
    let replacement = only_branch(generate(
        &mut state,
        &MoveChoice::None,
        &MoveChoice::Switch(PokemonIndex::P1),
    ));
    state.apply_instructions(&replacement);
    assert!(
        state
            .side_one
            .volatile_statuses
            .contains(&PokemonVolatileStatus::PERISH2),
        "the deferred block carries the perish tick: {:?}",
        replacement
    );
}
