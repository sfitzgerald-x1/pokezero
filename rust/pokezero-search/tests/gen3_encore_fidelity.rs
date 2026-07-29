//! Gen 3 ENCORE duration fidelity pins, asserted directly against the vendored
//! gen3-patched poke-engine (`third_party/poke-engine-src/`).
//!
//! Sibling of `gen3_confusion_fidelity.rs`, same hazard-ladder shape. The
//! divergence being pinned: upstream applies the ENCORE volatile and enforces
//! the lock, but NEVER expires it in gen3 — `volatile_status_durations.encore`
//! exists and is serialized, yet the only code that advances it is `#[cfg]`-gated
//! to gen5+, so a gen3 Encore locks its victim for the rest of the battle.
//!
//! Ground truth, read off the vendored Showdown:
//!
//! * `data/mods/gen3/moves.ts` overrides `encore.condition.durationCallback()`
//!   to `this.random(3, 7)` — uniform on {3,4,5,6}. gen3 inherits gen4, NOT
//!   gen5, and gen4's own override says `this.random(4, 9)`; gen3's re-override
//!   is what actually applies, so the window is 3-6 and not 4-8.
//! * `sim/battle.ts` decrements `handler.state.duration` once per turn in the
//!   RESIDUAL phase and calls `end` at zero. Encore qualifies because its
//!   condition carries `onResidual`/`onResidualOrder`. So the counter burns one
//!   tick per TURN, whether or not the encored Pokemon got to move.
//! * `encore.onStart` ends with
//!   `if (!this.queue.willMove(target)) this.effectState.duration!++`, which
//!   hands back the free tick when the target has already moved this turn. Net:
//!   the victim is locked for exactly `duration` turns either way.
//! * `onResidual` ends Encore early the moment the encored move hits 0 PP.
//! * The condition carries `noCopy: true`, so Baton Pass does NOT carry it —
//!   the opposite of confusion, and the reason there is no BP pin here.

use poke_engine::choices::Choices;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::{MoveChoice, PokemonVolatileStatus};
use poke_engine::instruction::{Instruction, StateInstructions};
use poke_engine::state::{
    LastUsedMove, PokemonIndex, PokemonMoveIndex, SideReference, State,
};

/// `this.random(3, 7)` — the encored Pokemon is locked for 3 to 6 turns.
const MIN_ENCORE_TURNS: i8 = 3;
const MAX_ENCORE_TURNS: i8 = 6;

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

/// Side two is the encored seat throughout; side one is strictly faster so no
/// speed tie doubles the branch set.
fn encored_state(ticks_already_burned: i8) -> State {
    let mut state = State::default();
    state.side_one.get_active().speed = 500;
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
        .insert(PokemonVolatileStatus::ENCORE);
    state.side_two.volatile_status_durations.encore = ticks_already_burned;
    // The engine panics if ENCORE is live without a move to be locked into.
    state.side_two.last_used_move = LastUsedMove::Move(PokemonMoveIndex::M0);
    state
}

fn both_splash(state: &mut State) -> Vec<StateInstructions> {
    generate(
        state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    )
}

fn ends_encore(list: &[Instruction]) -> bool {
    list.iter().any(|instruction| match instruction {
        Instruction::RemoveVolatileStatus(remove) => {
            remove.side_ref == SideReference::SideTwo
                && remove.volatile_status == PokemonVolatileStatus::ENCORE
        }
        _ => false,
    })
}

fn burns_a_tick(list: &[Instruction]) -> bool {
    list.iter().any(|instruction| match instruction {
        Instruction::ChangeVolatileStatusDuration(change) => {
            change.side_ref == SideReference::SideTwo
                && change.volatile_status == PokemonVolatileStatus::ENCORE
                && change.amount == 1
        }
        _ => false,
    })
}

fn mass(branches: &[StateInstructions], predicate: fn(&[Instruction]) -> bool) -> f32 {
    branches
        .iter()
        .filter(|branch| predicate(&branch.instruction_list))
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
// The duration ladder
// ---------------------------------------------------------------------------

/// Indexing: `encored_state(n)` sets the counter as it stands at the START of
/// the turn, the residual tick then increments it, and the ladder is read at the
/// END of the turn — so the rung reached is `chance_encore_ends(n + 1)`, and the
/// engine-side function is always indexed by ticks burned INCLUDING this one.
///
/// `P(duration == k | duration >= k)` for `duration ~ Uniform{3,4,5,6}`: zero
/// while `k < 3` (a gen3 Encore can never end before its third tick), then 1/4,
/// 1/3, 1/2, forced.
#[test]
fn encore_end_chance_matches_the_showdown_duration_roll() {
    for (burned, expected) in [
        (0, 0.0),
        (1, 0.0),
        (2, 25.0),
        (3, 100.0 / 3.0),
        (4, 50.0),
        (5, 100.0),
        (6, 100.0),
    ] {
        let mut state = encored_state(burned);
        let branches = both_splash(&mut state);
        assert_close(
            mass(&branches, ends_encore),
            expected,
            &format!("Encore end mass with {} tick(s) already burned", burned),
        );
    }
}

/// The analytical marginal: multiplying the ladder out has to give a UNIFORM
/// 1/4 for each of 3, 4, 5 and 6 ticks, and nothing may survive a seventh.
/// This is the property the hazard rungs exist to reproduce.
#[test]
fn encore_lasts_a_uniform_three_to_six_turns() {
    let mut survival = 1.0f32;
    for burned in 0..MAX_ENCORE_TURNS {
        let mut state = encored_state(burned);
        let branches = both_splash(&mut state);
        let ends_now = mass(&branches, ends_encore) / 100.0;

        let expected = if burned + 1 < MIN_ENCORE_TURNS { 0.0 } else { 25.0 };
        assert_close(
            survival * ends_now * 100.0,
            expected,
            &format!("marginal P(Encore lasts exactly {} ticks)", burned + 1),
        );
        survival *= 1.0 - ends_now;
    }
    assert_close(survival * 100.0, 0.0, "mass surviving a seventh tick");
}

/// Drive the never-ends-early path for real, applying instructions turn by turn:
/// the counter advances exactly once per turn and by the sixth tick the volatile
/// is gone on EVERY branch, so a permanent Encore cannot survive here.
#[test]
fn encore_never_persists_past_six_turns() {
    let mut state = encored_state(0);
    for tick in 1..=MAX_ENCORE_TURNS {
        let branches = both_splash(&mut state);
        if tick < MAX_ENCORE_TURNS {
            let survivor = branches
                .iter()
                .find(|branch| !ends_encore(&branch.instruction_list))
                .unwrap_or_else(|| panic!("no still-encored branch on tick {}", tick))
                .clone();
            state.apply_instructions(&survivor.instruction_list);
            assert_eq!(
                state.side_two.volatile_status_durations.encore, tick,
                "the counter must advance exactly once per turn"
            );
        } else {
            for branch in &branches {
                assert!(
                    ends_encore(&branch.instruction_list),
                    "Encore must be gone after {} ticks: {:?}",
                    MAX_ENCORE_TURNS,
                    branch.instruction_list
                );
            }
            state.apply_instructions(&branches[0].instruction_list);
            assert!(
                !state
                    .side_two
                    .volatile_statuses
                    .contains(&PokemonVolatileStatus::ENCORE),
                "Encore must not outlive the ladder"
            );
            assert_eq!(
                state.side_two.volatile_status_durations.encore, 0,
                "the counter must be zeroed with the volatile"
            );
        }
    }
}

/// While Encore is live the victim really is locked — the pin would be hollow if
/// the lock were not there to expire.
#[test]
fn an_encored_side_is_locked_to_its_last_used_move_until_the_ladder_fires() {
    let mut state = encored_state(0);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::TACKLE);

    let (_, side_two_options) = state.get_all_options();
    assert_eq!(
        side_two_options
            .iter()
            .filter(|option| matches!(option, MoveChoice::Move(_)))
            .count(),
        1,
        "an encored side keeps exactly one move option: {:?}",
        side_two_options
    );
    assert!(
        side_two_options.contains(&MoveChoice::Move(PokemonMoveIndex::M0)),
        "and it is the encored move: {:?}",
        side_two_options
    );
}

// ---------------------------------------------------------------------------
// Showdown's onStart compensation and onResidual early termination
// ---------------------------------------------------------------------------

/// `encore.onStart` bumps the duration when the target has already moved, so the
/// free residual tick at the end of the application turn does not eat a turn the
/// victim was never locked for. Counting up, that is a counter seeded one BELOW
/// zero when the encorer moves second, and at zero when it moves first — either
/// way the victim ends up locked for the same 3-6 turns.
#[test]
fn the_application_turn_seeds_the_counter_by_move_order() {
    for (encorer_speed, expected_seed) in [(500, 0), (1, -1)] {
        let mut state = State::default();
        state.side_one.get_active().speed = encorer_speed;
        state
            .side_one
            .get_active()
            .replace_move(PokemonMoveIndex::M0, Choices::ENCORE);
        state
            .side_two
            .get_active()
            .replace_move(PokemonMoveIndex::M0, Choices::SPLASH);
        // Encore fails outright unless the target has a move to be locked into.
        state.side_two.last_used_move = LastUsedMove::Move(PokemonMoveIndex::M0);

        let branches = both_splash(&mut state);
        assert_eq!(branches.len(), 1, "Encore application is deterministic here");
        state.apply_instructions(&branches[0].instruction_list);

        assert!(
            state
                .side_two
                .volatile_statuses
                .contains(&PokemonVolatileStatus::ENCORE),
            "Encore must land"
        );
        // The application turn's own residual tick has already been added, so the
        // seed is observed one higher than it was written.
        assert_eq!(
            state.side_two.volatile_status_durations.encore,
            expected_seed + 1,
            "encorer speed {} must seed the ladder at {}",
            encorer_speed,
            expected_seed
        );
    }
}

/// The other half of `onResidual`: Encore ends EARLY and deterministically when
/// the encored move runs out of PP. Without this the engine's own option filter
/// (which drops 0-PP moves) would leave the victim with no legal move at all.
///
/// Modelled as the realistic case: the victim spends its LAST PP on the locked
/// move this turn, so the slot hits 0 during the move phase and Encore ends in
/// that same turn's residual.
#[test]
fn encore_ends_early_when_the_encored_move_runs_out_of_pp() {
    let mut state = encored_state(0);
    state.side_two.get_active().moves[&PokemonMoveIndex::M0].pp = 1;

    let branches = both_splash(&mut state);
    assert_eq!(branches.len(), 1, "PP termination is deterministic");
    assert!(
        ends_encore(&branches[0].instruction_list),
        "a 0-PP encored move must end Encore: {:?}",
        branches[0].instruction_list
    );
    assert!(
        !burns_a_tick(&branches[0].instruction_list),
        "PP termination replaces the tick, it does not stack with it: {:?}",
        branches[0].instruction_list
    );
}

// ---------------------------------------------------------------------------
// Counter hygiene and composition
// ---------------------------------------------------------------------------

/// Showdown's `encore` condition carries `noCopy: true`, so unlike confusion the
/// volatile does NOT ride a Baton Pass — and an ordinary switch drops it too.
/// Either way the counter must be zeroed with it, or the next Encore on that
/// side would start part-way up the ladder.
#[test]
fn switching_out_drops_encore_and_zeroes_the_counter() {
    let mut state = encored_state(4);
    let switch = generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Switch(PokemonIndex::P1),
    );
    let list = switch
        .into_iter()
        .next()
        .expect("a switch branch")
        .instruction_list;

    assert!(
        ends_encore(&list),
        "switching out must clear Encore: {:?}",
        list
    );
    state.apply_instructions(&list);
    assert_eq!(
        state.side_two.volatile_status_durations.encore, 0,
        "switching out must zero the Encore counter: {:?}",
        list
    );
}

/// Composition with the residual deferral: `add_end_of_turn_instructions`
/// returns early on a ply that owes a forced replacement, so no tick is emitted
/// and the ladder cannot fire. That matches Showdown, which defers the entire
/// residual phase — duration decrements included — until the replacement is in.
#[test]
fn a_deferred_residual_ply_burns_no_encore_tick() {
    let mut state = encored_state(4);
    // Side one KOs side two's active, so the residual block defers.
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::SWIFT);
    state.side_two.get_active().hp = 1;

    let branches = both_splash(&mut state);
    for branch in &branches {
        assert!(
            !burns_a_tick(&branch.instruction_list),
            "a deferred residual ply must not tick Encore: {:?}",
            branch.instruction_list
        );
        assert!(
            !ends_encore(&branch.instruction_list)
                || branch
                    .instruction_list
                    .contains(&Instruction::ToggleSideTwoForceSwitch),
            "the only Encore removal here is the faint's own switch cleanup: {:?}",
            branch.instruction_list
        );
    }
}

/// Two encored seats tick independently in the same residual phase, so the
/// ladder must fork per side and multiply out rather than sharing one roll.
#[test]
fn both_seats_encored_fork_independently() {
    let mut state = encored_state(4);
    state
        .side_one
        .volatile_statuses
        .insert(PokemonVolatileStatus::ENCORE);
    state.side_one.volatile_status_durations.encore = 4;
    state.side_one.last_used_move = LastUsedMove::Move(PokemonMoveIndex::M0);

    let branches = both_splash(&mut state);
    let total: f32 = branches.iter().map(|branch| branch.percentage).sum();
    assert_close(total, 100.0, "probability mass must be conserved");
    assert_eq!(
        branches.len(),
        4,
        "two independent 1/2 rolls give four branches: {:?}",
        branches
    );
    assert_close(
        mass(&branches, ends_encore),
        50.0,
        "side two's own end chance is unaffected by side one's",
    );
}
