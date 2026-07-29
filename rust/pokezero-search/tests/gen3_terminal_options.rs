//! The option-vector invariant MCTS depends on, and how terminal states resolve.
//!
//! `Node::maximize_ucb_for_side` (`src/mcts.rs`) initialises `let mut choice = 0`
//! and returns it unchanged when the option slice is EMPTY; `Node::expand` then
//! does `s2_options[s2_move_index]` on a zero-length vector and panics with
//! `index out of bounds: the len is 0 but the index is 0`, taking the whole
//! search down. Nothing downstream checks, so the invariant has to hold at the
//! source: **`get_all_options` must never hand back an empty vector for either
//! side.**
//!
//! Guards `third_party/poke-engine-gen3-terminal-options.patch`. Before it, two
//! exits violated the invariant — the `force_switch` branches, which call
//! `add_move_from_choice` (a no-op when the saved move is absent from the current
//! active's slots) and then return early, skipping the `len() == 0` guard every
//! other exit has. `root_get_all_options` guards its own exits, which is why the
//! panic only ever fired on INTERIOR tree nodes and never at the root.

use poke_engine::choices::Choices;
use poke_engine::engine::state::MoveChoice;
use poke_engine::state::{PokemonIndex, PokemonMoveIndex, State};

/// Side one owes a forced replacement; side two carries a saved-move commitment
/// its CURRENT active does not know. This is the exact shape the engine-search
/// census panicked on, reachable because `engine_world` samples a pending
/// Baton Pass commitment and search then explores lines where that side's active
/// changes.
fn unsatisfiable_commitment_state() -> State {
    let mut state = State::default();
    state.side_one.force_switch = true;
    state.side_two.switch_out_move_second_saved_move = Choices::BATONPASS;
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::SPLASH);
    state
}

#[test]
fn a_side_that_cannot_honour_its_commitment_still_gets_an_option() {
    let state = unsatisfiable_commitment_state();
    let (side_one_options, side_two_options) = state.get_all_options();

    assert!(
        !side_one_options.is_empty(),
        "the replacing side must be offered its switches"
    );
    assert_eq!(
        side_two_options,
        vec![MoveChoice::None],
        "an unsatisfiable commitment means NO action this boundary, not NO option"
    );
}

/// The same state at the root already behaved — `root_get_all_options` guards its
/// exits. Pinned so the two entry points cannot drift apart again.
#[test]
fn root_and_interior_option_sets_agree() {
    let state = unsatisfiable_commitment_state();
    assert_eq!(state.get_all_options(), state.root_get_all_options());
}

/// A commitment the active CAN honour is still honoured — the guard must not
/// swallow the real option.
#[test]
fn a_satisfiable_commitment_is_offered() {
    let mut state = unsatisfiable_commitment_state();
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::BATONPASS);

    let (_, side_two_options) = state.get_all_options();
    assert_eq!(side_two_options, vec![MoveChoice::Move(PokemonMoveIndex::M1)]);
}

/// The invariant itself, swept over the state shapes that reach these exits:
/// neither side may ever be handed an empty option vector.
#[test]
fn no_state_shape_yields_an_empty_option_vector() {
    for (s1_force, s2_force) in [(true, false), (false, true), (true, true)] {
        for saved in [Choices::NONE, Choices::BATONPASS, Choices::UTURN] {
            for reserves_alive in [true, false] {
                let mut state = State::default();
                state.side_one.force_switch = s1_force;
                state.side_two.force_switch = s2_force;
                state.side_one.switch_out_move_second_saved_move = saved;
                state.side_two.switch_out_move_second_saved_move = saved;
                if !reserves_alive {
                    for index in [
                        PokemonIndex::P1,
                        PokemonIndex::P2,
                        PokemonIndex::P3,
                        PokemonIndex::P4,
                        PokemonIndex::P5,
                    ] {
                        state.side_one.pokemon[index].hp = 0;
                        state.side_two.pokemon[index].hp = 0;
                    }
                }

                let (side_one_options, side_two_options) = state.get_all_options();
                let context = format!(
                    "force=({}, {}) saved={:?} reserves_alive={}",
                    s1_force, s2_force, saved, reserves_alive
                );
                assert!(!side_one_options.is_empty(), "side one empty: {}", context);
                assert!(!side_two_options.is_empty(), "side two empty: {}", context);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Terminal verdicts
// ---------------------------------------------------------------------------

/// A side with no living Pokemon has lost, and `battle_is_over` says so with the
/// sign MCTS's rollout maps to a 0/1 score (`src/mcts.rs::rollout`). Note this is
/// NOT the condition that caused the panic — `add_switches` already pushes
/// `MoveChoice::None` for a side with no living reserve, so a wiped side never
/// produced an empty vector.
#[test]
fn a_wiped_side_is_a_terminal_loss_for_that_side() {
    let mut state = State::default();
    for index in [
        PokemonIndex::P0,
        PokemonIndex::P1,
        PokemonIndex::P2,
        PokemonIndex::P3,
        PokemonIndex::P4,
        PokemonIndex::P5,
    ] {
        state.side_one.pokemon[index].hp = 0;
    }
    assert_eq!(
        state.battle_is_over(),
        -1.0,
        "side one is wiped, so side two has won"
    );

    let mut state = State::default();
    for index in [
        PokemonIndex::P0,
        PokemonIndex::P1,
        PokemonIndex::P2,
        PokemonIndex::P3,
        PokemonIndex::P4,
        PokemonIndex::P5,
    ] {
        state.side_two.pokemon[index].hp = 0;
    }
    assert_eq!(state.battle_is_over(), 1.0, "side two is wiped, side one won");

    // ... and it is still terminal-with-options, not a panic: the fainted active
    // is offered `MoveChoice::None` rather than an empty vector.
    let (side_one_options, _) = state.get_all_options();
    assert!(!side_one_options.is_empty());
}

/// Simultaneous last-mon faints (Explosion, Perish Song, recoil, a residual
/// double-KO — all reachable in gen3 randbats).
///
/// SHOWDOWN, gen 3: a TIE. `Battle.checkWin` (sim/battle.ts) runs
/// `if (this.sides.every(side => !side.pokemonLeft)) this.win(faintData && this.gen > 4 ? faintData.target.side : null)`
/// — the `gen > 4` guard means gen 3 passes `null`, and `win(null)` is a tie.
/// Only gen 5+ awards it to the side whose Pokemon fainted last.
///
/// THIS ENGINE: `battle_is_over` tests side one first and returns -1.0, i.e. it
/// calls the double wipe a side-two WIN. That is a real divergence from gen3, and
/// it is deliberately NOT fixed here: `battle_is_over` lives in the shared
/// `src/state.rs`, its {0.0, 1.0, -1.0} contract has no room for a fourth verdict
/// (0.0 already means "not over"), `src/search.rs` uses the value as a sign
/// MULTIPLIER, and the correct answer is generation-dependent. Encoding a tie is a
/// sentinel refactor across three consumers plus gen gating — a separate change
/// from this panic fix, filed as its own task.
///
/// What this test pins is what this patch is responsible for: the state is
/// TERMINAL and both sides still get an option, so the search reaches a verdict
/// instead of panicking.
#[test]
fn a_double_wipe_is_terminal_and_not_a_panic() {
    let mut state = State::default();
    for index in [
        PokemonIndex::P0,
        PokemonIndex::P1,
        PokemonIndex::P2,
        PokemonIndex::P3,
        PokemonIndex::P4,
        PokemonIndex::P5,
    ] {
        state.side_one.pokemon[index].hp = 0;
        state.side_two.pokemon[index].hp = 0;
    }

    assert_ne!(state.battle_is_over(), 0.0, "a double wipe must be terminal");
    assert_eq!(
        state.battle_is_over(),
        -1.0,
        "DIVERGENCE, pinned deliberately: gen3 Showdown ties this; the engine \
         checks side one first and awards it to side two. See the doc comment."
    );

    let (side_one_options, side_two_options) = state.get_all_options();
    assert!(!side_one_options.is_empty(), "no empty vector for MCTS to index");
    assert!(!side_two_options.is_empty(), "no empty vector for MCTS to index");
}
