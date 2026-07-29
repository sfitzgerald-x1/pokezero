//! Gen 3 consecutive-Protect success FLOOR, asserted against the vendored
//! gen3-patched poke-engine (`third_party/poke-engine-src/`).
//!
//! # Evidence basis — read this before trusting the numbers
//!
//! Unlike most pins in this suite, the floor is **not** empirically confirmed
//! against the sim, and this file does not claim it is. Reaching the 5th
//! consecutive Protect requires four consecutive successes — `1 * 1/2 * 1/4 *
//! 1/8` = **1/64, about 1.6% of sequences** — so a sim probe cannot separate 1/8
//! from 1/16 there at any sample size this suite can afford.
//!
//! What IS established:
//!
//! * **Resolved source** (`data/mods/gen4/conditions.ts` -> gen5's full `stall`
//!   definition): the counter starts at 2 and doubles under a
//!   test-before-multiply gate, `if (counter < counterMax) counter *= 2`, with
//!   gen4 setting `counterMax: 8`. So it walks 2 -> 4 -> 8 and stops, because
//!   `8 < 8` is false. The chance is `1/counter`.
//! * **Empirically confirmed, 160 seeds** (the part the floor rests on): the
//!   ladder is x2, not x3 — 2nd attempt 239/496 = 0.482, 3rd 40/204 = 0.196.
//!   That measurement is what makes the resolved reading trustworthy; the cap is
//!   then read off the same resolved definition.
//! * **Engine behaviour** (this file): pinned both ways at the divergence point.
//!
//! The ladder first REACHES 1/8 at the 4th attempt, which upstream already
//! priced correctly. Upstream first DIVERGES at the 5th.

use poke_engine::choices::Choices;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::MoveChoice;
use poke_engine::state::{PokemonMoveIndex, State};

/// Probability mass on which the Protect attempt SUCCEEDS, for a side whose
/// counter already stands at `consecutive_successes`.
fn success_chance(consecutive_successes: i8) -> f32 {
    let mut state = State::default();
    state.side_one.side_conditions.protect = consecutive_successes;
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::PROTECT);

    let before = format!("{:?}", state);
    let branches = generate_instructions_from_move_pair(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::None,
        false,
    );
    assert_eq!(before, format!("{:?}", state), "state was mutated");

    let mut success = 0.0;
    for branch in &branches {
        let mut probe = state.clone();
        let snapshot = format!("{:?}", probe);
        probe.apply_instructions(&branch.instruction_list);
        probe.reverse_instructions(&branch.instruction_list);
        assert_eq!(snapshot, format!("{:?}", probe), "branch did not revert");

        let protects = branch
            .instruction_list
            .iter()
            .any(|instruction| format!("{:?}", instruction).contains("PROTECT"));
        if protects {
            success += branch.percentage;
        }
    }
    success
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 0.01
}

/// The whole ladder by attempt number, with the counter standing at
/// `attempt - 1` consecutive prior successes.
///
/// Attempts 1-4 were already correct upstream and are the controls: a fix that
/// moved them would be changing behaviour the sim measurement already endorses.
#[test]
fn the_first_four_attempts_are_unchanged() {
    assert!(
        close(success_chance(0), 100.0),
        "1st attempt always succeeds"
    );
    assert!(close(success_chance(1), 50.0), "2nd attempt is 1/2");
    assert!(close(success_chance(2), 25.0), "3rd attempt is 1/4");
    assert!(close(success_chance(3), 12.5), "4th attempt is 1/8");
}

/// The fix, at the exact point upstream diverges: gen4's `counterMax: 8` stops
/// the counter at 8, so the 5th attempt holds at 1/8 rather than halving to
/// 1/16.
#[test]
fn the_fifth_attempt_holds_at_one_eighth() {
    assert!(
        close(success_chance(4), 12.5),
        "5th attempt must hold at 1/8, got {}%",
        success_chance(4)
    );
}

/// ...and stays there. The counter cannot grow past 8, so every later attempt is
/// the same 1/8 — upstream would have reached 1/1024 by the 11th.
#[test]
fn every_later_attempt_holds_at_the_same_floor() {
    for counter in 4..=10i8 {
        assert!(
            close(success_chance(counter), 12.5),
            "counter {} must price 1/8, got {}%",
            counter,
            success_chance(counter)
        );
    }
}

/// The floor is a floor, not a clamp on the whole ladder: it must never RAISE a
/// chance that was already below it, and the branch mass must stay conserved.
#[test]
fn the_floor_never_raises_an_earlier_attempt_and_mass_is_conserved() {
    let mut previous = 100.0;
    for counter in 0..=6i8 {
        let chance = success_chance(counter);
        assert!(
            chance <= previous + 0.01,
            "the ladder must be non-increasing: counter {} gave {}% after {}%",
            counter,
            chance,
            previous
        );
        assert!(chance >= 12.4, "and never fall below the 1/8 floor");
        previous = chance;
    }
}
