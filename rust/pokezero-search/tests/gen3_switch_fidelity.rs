//! Gen 3 switch-out / Protect fidelity pins, asserted directly against the
//! vendored gen3-patched poke-engine (`third_party/poke-engine-src/`).
//!
//! Every expectation here was read off the **real** Node Showdown simulator
//! driven through `scripts/gen3_switch_differential.py`; that script is the
//! ground-truth gate and this file is the engine-contract pin, so a wheel
//! rebuild or a version bump that silently drops one of the
//! `third_party/poke-engine-gen3-*.patch` files fails `cargo test` instead of
//! quietly regressing search fidelity.
//!
//! Coverage, and which patch each pin guards:
//!
//! * Rapid Spin blocked by Protect leaves the spinner's hazards alone, and a
//!   connecting Rapid Spin clears hazards + Leech Seed + partial-trapping —
//!   `poke-engine-gen3-rapidspin-fidelity.patch`.
//! * Leech Seed ends when the seeded Pokemon leaves the field, and a partial
//!   trap ends both when the victim is dragged out and when the TRAPPER leaves
//!   — upstream behaviour, pinned because the two live in the same switch
//!   routine as the patched code and have no other regression cover.
//! * Baton Pass carries the Perish Song counter to the receiver —
//!   `poke-engine-gen3-batonpass-perish.patch`.

use poke_engine::choices::Choices;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::{MoveChoice, PokemonVolatileStatus};
use poke_engine::instruction::{Instruction, StateInstructions};
use poke_engine::state::{
    PokemonIndex, PokemonMoveIndex, PokemonSideCondition, SideReference, State,
};

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

fn damages_side(list: &[Instruction], side_ref: SideReference) -> bool {
    list.iter().any(|instruction| match instruction {
        Instruction::Damage(damage) => damage.side_ref == side_ref && damage.damage_amount > 0,
        _ => false,
    })
}

/// New lifecycle instructions must preserve tree reversibility just like move
/// damage does. A one-way volatile removal would corrupt sibling branches.
fn assert_reverts_cleanly(state: &mut State, list: &Vec<Instruction>) {
    let before = format!("{:?}", state);
    state.apply_instructions(list);
    state.reverse_instructions(list);
    assert_eq!(
        before,
        format!("{:?}", state),
        "instructions did not reverse cleanly"
    );
}

fn changes_side_condition(
    list: &[Instruction],
    side_ref: SideReference,
    side_condition: PokemonSideCondition,
) -> bool {
    list.iter().any(|instruction| match instruction {
        Instruction::ChangeSideCondition(change) => {
            change.side_ref == side_ref && change.side_condition == side_condition
        }
        _ => false,
    })
}

/// Side one spins, side two acts. `spikes` layers sit on the SPINNER's side —
/// Rapid Spin clears the user's own hazards.
fn spin_state(defender_move: Choices, defender_moves_first: bool) -> State {
    let mut state = State::default();
    state.side_one.side_conditions.spikes = 2;
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::RAPIDSPIN);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, defender_move);
    if defender_moves_first {
        state.side_two.get_active().speed = 500;
    }
    state
}

// ---------------------------------------------------------------------------
// Rapid Spin / Protect (poke-engine-gen3-rapidspin-fidelity.patch)
// ---------------------------------------------------------------------------

/// Showdown gen3: `|move|p1a: Forretress|Rapid Spin|p2a: Blissey` followed by
/// `|-activate|p2a: Blissey|Protect` and **no** `|-sideend| ... Spikes`.
///
/// The bug this guards: `Choice::remove_effects_for_protect()` zeroes
/// base_power/category and the declarative effect fields but leaves `move_id`
/// intact, and `choice_hazard_clear` dispatches on `move_id` — so a blocked
/// Rapid Spin used to strip the spinner's own Spikes.
#[test]
fn rapid_spin_blocked_by_protect_leaves_hazards_alone() {
    let mut state = spin_state(Choices::PROTECT, true);
    let list = only_branch(generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    ));

    assert!(
        !changes_side_condition(&list, SideReference::SideOne, PokemonSideCondition::Spikes),
        "Protect-blocked Rapid Spin must not clear the spinner's Spikes: {:?}",
        list
    );
}

/// The Protect guard is on Protect specifically, NOT on damage/hit_sub, so a
/// spin that connects still clears. Showdown gen3 additionally ends the
/// spinner's Leech Seed and partial-trapping on a connecting spin.
#[test]
fn connecting_rapid_spin_clears_hazards_leech_seed_and_partial_trap() {
    let mut state = spin_state(Choices::SPLASH, false);
    state
        .side_one
        .volatile_statuses
        .insert(PokemonVolatileStatus::LEECHSEED);
    state
        .side_one
        .volatile_statuses
        .insert(PokemonVolatileStatus::PARTIALLYTRAPPED);

    let list = only_branch(generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    ));

    assert!(
        changes_side_condition(&list, SideReference::SideOne, PokemonSideCondition::Spikes),
        "connecting Rapid Spin must clear the spinner's Spikes: {:?}",
        list
    );
    assert!(
        removes_volatile(
            &list,
            SideReference::SideOne,
            PokemonVolatileStatus::LEECHSEED
        ),
        "connecting Rapid Spin must end the spinner's Leech Seed: {:?}",
        list
    );
    assert!(
        removes_volatile(
            &list,
            SideReference::SideOne,
            PokemonVolatileStatus::PARTIALLYTRAPPED
        ),
        "connecting Rapid Spin must free the spinner from partial-trapping: {:?}",
        list
    );
}

// ---------------------------------------------------------------------------
// Leech Seed on switch-out
// ---------------------------------------------------------------------------

/// Leech Seed is an ordinary volatile in Showdown, so `Pokemon.clearVolatile()`
/// drops it when the seeded Pokemon leaves the field.
#[test]
fn leech_seed_ends_when_the_seeded_pokemon_switches_out() {
    let mut state = State::default();
    state
        .side_one
        .volatile_statuses
        .insert(PokemonVolatileStatus::LEECHSEED);

    let list = only_branch(generate(
        &mut state,
        &MoveChoice::Switch(PokemonIndex::P1),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    ));

    assert!(
        removes_volatile(
            &list,
            SideReference::SideOne,
            PokemonVolatileStatus::LEECHSEED
        ),
        "switching out must end Leech Seed: {:?}",
        list
    );
}

// ---------------------------------------------------------------------------
// Partial trapping (Wrap / Bind / Fire Spin / Clamp / Whirlpool)
// ---------------------------------------------------------------------------

/// A partially trapped Pokemon may not switch: Showdown's `partiallytrapped`
/// condition calls `pokemon.tryTrap()` from `onTrapPokemon` while the trapper
/// is still active.
#[test]
fn partially_trapped_pokemon_has_no_switch_options() {
    let mut state = State::default();
    state
        .side_one
        .volatile_statuses
        .insert(PokemonVolatileStatus::PARTIALLYTRAPPED);

    let (side_one_options, side_two_options) = state.get_all_options();

    assert!(
        !side_one_options
            .iter()
            .any(|option| matches!(option, MoveChoice::Switch(_))),
        "the trapped side must not be offered switches: {:?}",
        side_one_options
    );
    assert!(
        side_two_options
            .iter()
            .any(|option| matches!(option, MoveChoice::Switch(_))),
        "the untrapped side keeps its switches: {:?}",
        side_two_options
    );
}

/// Showdown gen3 (which inherits gen4 -> gen5) frees the victim once the
/// trapper is gone: `partiallytrapped.onResidual` deletes the volatile when
/// `!trapper.isActive || trapper.hp <= 0`, and `onTrapPokemon` stops trapping
/// the moment the source leaves. The engine models that at switch time.
#[test]
fn partial_trap_ends_when_the_trapper_switches_out() {
    let mut state = State::default();
    // Side two is the victim, so side one holds the trap; side one switching
    // out is the trapper leaving the field.
    state
        .side_two
        .volatile_statuses
        .insert(PokemonVolatileStatus::PARTIALLYTRAPPED);

    let list = only_branch(generate(
        &mut state,
        &MoveChoice::Switch(PokemonIndex::P1),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    ));

    assert!(
        removes_volatile(
            &list,
            SideReference::SideTwo,
            PokemonVolatileStatus::PARTIALLYTRAPPED
        ),
        "the trapper leaving must free the victim: {:?}",
        list
    );
}

/// Showdown Gen 3: a target can use Substitute while it is held by Wrap / Bind /
/// Fire Spin / Clamp / Whirlpool. On a successful Substitute, Showdown ends the
/// existing `partiallytrapped` volatile before residuals; the target pays its
/// Substitute cost, but does not take another partial-trap chip that turn.
///
/// This is deliberately a target-side lifecycle rule, distinct from the existing
/// source-leaves release above and the Rapid Spin release pin. The patch must not
/// clear the volatile until Substitute has actually succeeded.
#[test]
fn successful_substitute_ends_the_target_partial_trap_before_residuals() {
    let mut state = State::default();
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::SPLASH);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::SUBSTITUTE);
    state
        .side_two
        .volatile_statuses
        .insert(PokemonVolatileStatus::PARTIALLYTRAPPED);

    let list = only_branch(generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    ));

    assert!(
        applies_volatile(
            &list,
            SideReference::SideTwo,
            PokemonVolatileStatus::SUBSTITUTE
        ),
        "fixture must exercise a successful Substitute: {:?}",
        list
    );
    assert!(
        removes_volatile(
            &list,
            SideReference::SideTwo,
            PokemonVolatileStatus::PARTIALLYTRAPPED
        ),
        "a successful Substitute must end the target's partial trap: {:?}",
        list
    );
    assert_reverts_cleanly(&mut state, &list);
}

/// The lifecycle release is conditional on creating the Substitute. A user at
/// exactly one-quarter HP cannot pay the strict `hp > maxhp / 4` cost, so the
/// failed attempt leaves the existing partial trap in place and it still ticks.
#[test]
fn failed_substitute_keeps_the_existing_partial_trap() {
    let mut state = State::default();
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::SPLASH);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::SUBSTITUTE);
    let p2 = state.side_two.get_active();
    p2.hp = p2.maxhp / 4;
    state
        .side_two
        .volatile_statuses
        .insert(PokemonVolatileStatus::PARTIALLYTRAPPED);

    let list = only_branch(generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    ));

    assert!(
        !applies_volatile(
            &list,
            SideReference::SideTwo,
            PokemonVolatileStatus::SUBSTITUTE
        ),
        "fixture must exercise a failed Substitute: {:?}",
        list
    );
    assert!(
        !removes_volatile(
            &list,
            SideReference::SideTwo,
            PokemonVolatileStatus::PARTIALLYTRAPPED
        ),
        "a failed Substitute must not release the partial trap: {:?}",
        list
    );
    assert!(
        damages_side(&list, SideReference::SideTwo),
        "the retained partial trap must still produce its residual chip: {:?}",
        list
    );
    assert_reverts_cleanly(&mut state, &list);
}

/// Ordinary turns leave a live partial trap alone. This prevents the successful
/// Substitute release from becoming a blanket volatile clear.
#[test]
fn ordinary_trapped_turn_keeps_the_partial_trap_and_its_residual() {
    let mut state = State::default();
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::SPLASH);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::SPLASH);
    state
        .side_two
        .volatile_statuses
        .insert(PokemonVolatileStatus::PARTIALLYTRAPPED);

    let list = only_branch(generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    ));

    assert!(
        !removes_volatile(
            &list,
            SideReference::SideTwo,
            PokemonVolatileStatus::PARTIALLYTRAPPED
        ),
        "an ordinary trapped turn must not release the partial trap: {:?}",
        list
    );
    assert!(
        damages_side(&list, SideReference::SideTwo),
        "an ordinary trapped turn must retain the partial-trap residual: {:?}",
        list
    );
    assert_reverts_cleanly(&mut state, &list);
}

// ---------------------------------------------------------------------------
// Baton Pass carry-over (poke-engine-gen3-batonpass-perish.patch)
// ---------------------------------------------------------------------------

/// Drive a Baton Pass to completion: the move sets `force_switch` +
/// `baton_passing` and the switch itself resolves at the next decision
/// boundary, so the carry-over is only observable after both plies. Returns the
/// second ply's instructions together with the state they were generated from.
///
/// That second ply also carries the turn's end-of-turn residual block: gen3
/// sends the replacement out BEFORE the residuals run, so the block is deferred
/// across the switch (`poke-engine-gen3-residual-defer-on-faint.patch`).
/// Showdown's protocol for this exact line is `|move|p1a: Smeargle|Baton Pass`
/// and nothing else, then `|switch|p1a: Snorlax|461/461|[from] Baton Pass`
/// immediately followed by `|-start|p1a: Snorlax|perish2` and `|upkeep`.
fn baton_pass_then_switch(volatile_status: PokemonVolatileStatus) -> (State, Vec<Instruction>) {
    let mut state = State::default();
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::BATONPASS);
    state.side_one.volatile_statuses.insert(volatile_status);

    let pass = only_branch(generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    ));
    state.apply_instructions(&pass);
    assert!(
        state.side_one.baton_passing,
        "Baton Pass must arm the pass before the switch resolves"
    );

    let list = only_branch(generate(
        &mut state,
        &MoveChoice::Switch(PokemonIndex::P1),
        &MoveChoice::None,
    ));
    (state, list)
}

/// Showdown gen3 ground truth: a Perish-Songed Smeargle that Baton Passes into
/// Snorlax emits `|-start|p1a: Snorlax|perish0` and then `|faint|p1a: Snorlax`
/// — the counter rides the pass and kills the RECEIVER. `copyVolatileFrom`
/// copies every volatile without a `noCopy` flag, and the perish volatiles
/// carry none in gen3 (neither the gen4 nor the gen5 mod overrides them).
///
/// Upstream retained only Substitute and Leech Seed across a pass, so the
/// engine believed Baton Pass escapes Perish Song.
///
/// The receiver arrives before the residual block, so it also takes that turn's
/// tick: the assertion is on the counter the receiver ENDS the ply with, which
/// is what separates "the pass dropped the volatile" from "the volatile counted
/// down normally on the new Pokemon".
#[test]
fn baton_pass_carries_the_perish_counter() {
    for (passed, after_the_tick) in [
        (
            PokemonVolatileStatus::PERISH2,
            PokemonVolatileStatus::PERISH1,
        ),
        (
            PokemonVolatileStatus::PERISH3,
            PokemonVolatileStatus::PERISH2,
        ),
        (
            PokemonVolatileStatus::PERISH4,
            PokemonVolatileStatus::PERISH3,
        ),
    ] {
        let (mut state, list) = baton_pass_then_switch(passed);
        state.apply_instructions(&list);
        assert!(
            state.side_one.volatile_statuses.contains(&after_the_tick),
            "Baton Pass must carry {:?} to the receiver (expected {:?} after the \
             turn's tick): {:?}",
            passed,
            after_the_tick,
            list
        );
    }
}

/// The last step of the same countdown: a receiver that arrives on `PERISH1`
/// takes the lethal tick the moment the deferred residual block runs, which is
/// Showdown's `|-start| ... |perish0|` + `|faint|` on the ply the pass resolves.
#[test]
fn a_receiver_passed_the_last_perish_tick_faints_on_arrival() {
    let (mut state, list) = baton_pass_then_switch(PokemonVolatileStatus::PERISH1);
    state.apply_instructions(&list);
    assert_eq!(
        state.side_one.get_active_immutable().hp,
        0,
        "the receiver must faint to the perish counter it was passed: {:?}",
        list
    );
}

/// Control for the pin above: the two volatiles upstream already passed still
/// pass, so the fix widened the retention set rather than replacing it.
#[test]
fn baton_pass_still_carries_substitute_and_leech_seed() {
    for volatile_status in [
        PokemonVolatileStatus::SUBSTITUTE,
        PokemonVolatileStatus::LEECHSEED,
    ] {
        let (_state, list) = baton_pass_then_switch(volatile_status);
        assert!(
            !removes_volatile(&list, SideReference::SideOne, volatile_status),
            "Baton Pass must carry {:?}: {:?}",
            volatile_status,
            list
        );
    }
}

/// Negative control: an ordinary switch (no pass armed) still drops the perish
/// counter, matching `Pokemon.clearVolatile()` on a normal switch-out.
#[test]
fn an_ordinary_switch_still_drops_the_perish_counter() {
    let mut state = State::default();
    state
        .side_one
        .volatile_statuses
        .insert(PokemonVolatileStatus::PERISH3);

    let list = only_branch(generate(
        &mut state,
        &MoveChoice::Switch(PokemonIndex::P1),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    ));

    assert!(
        removes_volatile(
            &list,
            SideReference::SideOne,
            PokemonVolatileStatus::PERISH3
        ),
        "a plain switch-out must clear the perish counter: {:?}",
        list
    );
}
