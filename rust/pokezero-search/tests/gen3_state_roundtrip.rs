//! State serialization round-trip pins for the vendored gen3-patched
//! poke-engine (`third_party/poke-engine-src/`).
//!
//! `State::serialize` / `State::deserialize` is not an incidental debug helper:
//! it is the wire format the search crate receives its root world on
//! (`pokezero_search`'s entrypoint is `State::deserialize(state_str)`), so a
//! state that does not survive the round trip is a state the engine searches
//! differently from the one the caller built.
//!
//! Guards `third_party/poke-engine-gen3-state-roundtrip.patch`: `Side::serialize`
//! writes the volatile set with a TRAILING ":" and `Side::deserialize` split on
//! ":" without discarding the empty tail. `PokemonVolatileStatus::from_str("")`
//! does not fail — `define_enum_with_from_str!` declares `default = NONE` — so
//! every state carrying at least one volatile silently gained a
//! `PokemonVolatileStatus::NONE`, and re-serializing emitted it back.

use poke_engine::choices::Choices;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::{MoveChoice, PokemonVolatileStatus};
use poke_engine::state::{PokemonIndex, PokemonMoveIndex, SideReference, State};

fn round_trip(state: &State) -> State {
    State::deserialize(&state.serialize())
}

/// The bug, in its smallest form: one volatile, one round trip.
#[test]
fn a_single_volatile_survives_the_round_trip_unchanged() {
    let mut state = State::default();
    state
        .side_one
        .volatile_statuses
        .insert(PokemonVolatileStatus::SUBSTITUTE);

    let once = state.serialize();
    assert_eq!(
        once,
        round_trip(&state).serialize(),
        "serialize -> deserialize -> serialize must be a fixed point"
    );
    assert!(
        !round_trip(&state)
            .side_one
            .volatile_statuses
            .contains(&PokemonVolatileStatus::NONE),
        "the trailing separator must not be read back as a NONE volatile"
    );
}

/// Idempotent for many volatiles, on both sides, and for none at all.
#[test]
fn volatile_sets_round_trip_idempotently() {
    let mut state = State::default();
    for volatile_status in [
        PokemonVolatileStatus::SUBSTITUTE,
        PokemonVolatileStatus::LEECHSEED,
        PokemonVolatileStatus::CONFUSION,
        PokemonVolatileStatus::TRANSFORMED,
    ] {
        state.side_one.volatile_statuses.insert(volatile_status);
    }
    state
        .side_two
        .volatile_statuses
        .insert(PokemonVolatileStatus::PERISH3);

    let once = round_trip(&state);
    let twice = round_trip(&once);
    assert_eq!(state.serialize(), once.serialize());
    assert_eq!(once.serialize(), twice.serialize());
    for volatile_status in [
        PokemonVolatileStatus::SUBSTITUTE,
        PokemonVolatileStatus::LEECHSEED,
        PokemonVolatileStatus::CONFUSION,
        PokemonVolatileStatus::TRANSFORMED,
    ] {
        assert!(
            once.side_one.volatile_statuses.contains(&volatile_status),
            "{:?} must survive",
            volatile_status
        );
    }
    assert!(!once
        .side_one
        .volatile_statuses
        .contains(&PokemonVolatileStatus::NONE));
    assert!(!once
        .side_two
        .volatile_statuses
        .contains(&PokemonVolatileStatus::NONE));

    // The empty case was already correct; keep it that way.
    let clean = State::default();
    assert_eq!(clean.serialize(), round_trip(&clean).serialize());
}

/// The consequence that made this worth fixing rather than noting: the phantom
/// volatile reached the instruction stream. `remove_volatile_statuses_on_switch`
/// iterates the bitset, so a round-tripped state used to emit a
/// `RemoveVolatileStatus(NONE)` for a volatile nothing ever applied — an
/// instruction whose inverse re-applies it.
#[test]
fn a_round_tripped_state_does_not_switch_out_a_phantom_volatile() {
    let mut state = State::default();
    state
        .side_one
        .volatile_statuses
        .insert(PokemonVolatileStatus::LEECHSEED);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::SPLASH);

    let mut state = round_trip(&state);
    let before = format!("{:?}", state);
    let branches = generate_instructions_from_move_pair(
        &mut state,
        &MoveChoice::Switch(PokemonIndex::P1),
        &MoveChoice::Move(PokemonMoveIndex::M0),
        false,
    );
    assert_eq!(before, format!("{:?}", state), "state was mutated");
    assert_eq!(branches.len(), 1, "expected one branch: {:?}", branches);
    let list = branches.into_iter().next().unwrap().instruction_list;

    let removed: Vec<_> = list
        .iter()
        .filter_map(|instruction| match instruction {
            poke_engine::instruction::Instruction::RemoveVolatileStatus(remove)
                if remove.side_ref == SideReference::SideOne =>
            {
                Some(remove.volatile_status)
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        removed,
        vec![PokemonVolatileStatus::LEECHSEED],
        "only the volatile that was really set may be removed: {:?}",
        list
    );
}
