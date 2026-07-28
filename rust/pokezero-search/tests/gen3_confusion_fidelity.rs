//! Gen 3 CONFUSION duration + Baton Pass carry fidelity pins, asserted directly
//! against the vendored gen3-patched poke-engine (`third_party/poke-engine-src/`).
//!
//! Companion to `gen3_switch_fidelity.rs`. Every expectation here was read off
//! the **real** gen3 Showdown mod chain (`data/conditions.ts` `confusion` +
//! `data/mods/gen4/conditions.ts`'s `onBeforeMove` override, which is the one
//! gen3 resolves to) and gated end-to-end by the Node sim through
//! `scripts/gen3_switch_differential.py --only confusionduration
//! confusionbatonpass`.
//!
//! The divergence being pinned: upstream models CONFUSION as PERMANENT until
//! switch-out — there is no expiry path anywhere in `src/gen3/`. Real gen3 rolls
//! `time = this.random(2, 6)` once at `addVolatile` (uniform on {2,3,4,5}),
//! decrements it at the top of every `onBeforeMove` that actually runs, snaps out
//! and lets the move through when it hits zero, and only otherwise rolls the 50%
//! self-hit. So a confused Pokemon takes a self-hit roll on `time - 1` attacking
//! turns — uniform on {1,2,3,4} — and never a fifth.
//!
//! Coverage, and which behaviour each pin guards:
//!
//! * The hazard ladder (`chance_confusion_ends`) reproduces the roll-at-start
//!   duration exactly: per-turn snap-out 0, 1/4, 1/3, 1/2, 1, and a uniform
//!   marginal over 1-4 self-hit-risk turns.
//! * The 50% self-hit is untouched — the ladder must not eat into it.
//! * The counter only advances on turns the check actually runs (Showdown's
//!   `onBeforeMove` priority 3 is pre-empted by sleep/freeze at 10, flinch at 8,
//!   Taunt at 5, recharge at 11).
//! * Paralysis (priority 1) is rolled AFTER confusion (priority 3), so a
//!   fully-paralyzed turn still burns a confusion turn and the self-hit keeps its
//!   full 50% mass.
//! * Baton Pass carries CONFUSION **and** its remaining duration
//!   (`copyVolatileFrom` shallow-clones the volatile object and `confusion` has
//!   no `noCopy` in gen3's chain); an ordinary switch drops both.

use poke_engine::choices::Choices;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::{MoveChoice, PokemonVolatileStatus};
use poke_engine::instruction::{Instruction, StateInstructions};
use poke_engine::state::{
    PokemonIndex, PokemonMoveIndex, PokemonStatus, SideReference, State,
};

/// Showdown's `time` is uniform on {2,3,4,5}, so the number of attacking turns
/// that carry a self-hit roll is uniform on {1,2,3,4}.
const MAX_CONFUSED_TURNS: i8 = 4;

/// `generate_instructions_from_move_pair` must leave `state` untouched.
fn generate(state: &mut State, side_one: &MoveChoice, side_two: &MoveChoice) -> Vec<StateInstructions> {
    let before = format!("{:?}", state);
    let instructions = generate_instructions_from_move_pair(state, side_one, side_two, false);
    assert_eq!(before, format!("{:?}", state), "state was mutated");
    instructions
}

/// Both sides Splash, side one strictly faster so no speed tie doubles the
/// branch set. Side one is the confused seat throughout.
fn splash_state() -> State {
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
}

fn confused_state(turns_already_burned: i8) -> State {
    let mut state = splash_state();
    state
        .side_one
        .volatile_statuses
        .insert(PokemonVolatileStatus::CONFUSION);
    state.side_one.volatile_status_durations.confusion = turns_already_burned;
    state
}

fn both_splash(state: &mut State) -> Vec<StateInstructions> {
    generate(
        state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    )
}

fn drops_confusion(list: &[Instruction]) -> bool {
    list.iter().any(|instruction| match instruction {
        Instruction::RemoveVolatileStatus(remove) => {
            remove.side_ref == SideReference::SideOne
                && remove.volatile_status == PokemonVolatileStatus::CONFUSION
        }
        _ => false,
    })
}

fn burns_a_confusion_turn(list: &[Instruction]) -> bool {
    list.iter().any(|instruction| match instruction {
        Instruction::ChangeVolatileStatusDuration(change) => {
            change.side_ref == SideReference::SideOne
                && change.volatile_status == PokemonVolatileStatus::CONFUSION
                && change.amount == 1
        }
        _ => false,
    })
}

/// Splash deals no damage and nothing else in `splash_state` can, so any damage
/// to side one is the confusion self-hit.
fn hits_itself(list: &[Instruction]) -> bool {
    list.iter().any(|instruction| match instruction {
        Instruction::Damage(damage) => damage.side_ref == SideReference::SideOne,
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

/// Indexing, since the counter is read at two different points in a turn:
/// `confused_state(burned)` sets the counter as it stands at the START of the
/// turn, the confusion check then increments it, and the ladder is evaluated at
/// the END of the turn — so the rung reached here is
/// `chance_confusion_ends(burned + 1)`, and the engine-side function is always
/// indexed by turns burned INCLUDING the current one.
///
/// In those terms: given `burned` attacking turns already behind it, the chance
/// that the turn just taken was the LAST one carrying a self-hit roll is
/// `P(time == burned + 2 | time > burned + 1)` for `time ~ Uniform{2,3,4,5}` —
/// 1/4, 1/3, 1/2, forced. (The engine resolves the ladder at end of turn rather
/// than at the start of the next attacking turn; see `chance_confusion_ends`.
/// Showdown's snap-out turn carries no self-hit roll and lets the move through,
/// exactly like a turn on which the volatile is already gone.)
#[test]
fn snap_out_chance_matches_the_showdown_duration_roll() {
    for (burned, expected) in [
        (0, 25.0),
        (1, 100.0 / 3.0),
        (2, 50.0),
        (3, 100.0),
        (4, 100.0),
    ] {
        let mut state = confused_state(burned);
        let branches = both_splash(&mut state);
        assert_close(
            mass(&branches, drops_confusion),
            expected,
            &format!("snap-out mass with {} turn(s) already burned", burned),
        );
    }
}

/// The whole point of the ladder: the marginal number of attacking turns that
/// carry a self-hit roll is uniform on {1,2,3,4}, matching `time - 1` for
/// `time = this.random(2, 6)`. Walk the survive-and-keep-attacking path and
/// multiply out the per-turn survival mass.
#[test]
fn confusion_lasts_a_uniform_one_to_four_attacking_turns() {
    let mut survival = 1.0f32;
    for burned in 0..MAX_CONFUSED_TURNS {
        let mut state = confused_state(burned);
        let branches = both_splash(&mut state);
        let ends_now = mass(&branches, drops_confusion) / 100.0;
        assert_close(
            survival * ends_now * 100.0,
            25.0,
            &format!("marginal P(confusion lasts exactly {} turns)", burned + 1),
        );
        survival *= 1.0 - ends_now;
    }
    assert_close(survival * 100.0, 0.0, "mass surviving a fifth attacking turn");
}

/// Drive the never-snaps-out path for real, applying instructions turn by turn:
/// after four attacking turns the volatile is gone on EVERY branch and the
/// counter is back to zero, so a permanent confusion cannot survive here.
#[test]
fn confusion_never_persists_past_four_attacking_turns() {
    let mut state = confused_state(0);
    for turn in 1..=MAX_CONFUSED_TURNS {
        let branches = both_splash(&mut state);
        if turn < MAX_CONFUSED_TURNS {
            let survivor = branches
                .iter()
                .find(|branch| {
                    !drops_confusion(&branch.instruction_list)
                        && !hits_itself(&branch.instruction_list)
                })
                .unwrap_or_else(|| panic!("no still-confused branch on attacking turn {}", turn))
                .clone();
            state.apply_instructions(&survivor.instruction_list);
            assert_eq!(
                state.side_one.volatile_status_durations.confusion,
                turn,
                "the counter must advance once per attacking turn"
            );
        } else {
            for branch in &branches {
                assert!(
                    drops_confusion(&branch.instruction_list),
                    "confusion must be gone after {} attacking turns: {:?}",
                    MAX_CONFUSED_TURNS,
                    branch.instruction_list
                );
            }
            let survivor = branches[0].clone();
            state.apply_instructions(&survivor.instruction_list);
            assert!(
                !state
                    .side_one
                    .volatile_statuses
                    .contains(&PokemonVolatileStatus::CONFUSION),
                "confusion must not outlive the ladder"
            );
            assert_eq!(
                state.side_one.volatile_status_durations.confusion, 0,
                "the counter must be zeroed with the volatile"
            );
        }
    }
}

/// The ladder must not eat into the self-hit: gen3's `randomChance(1, 2)` is
/// rolled only after the snap-out check returns, so the self-hit keeps its full
/// 50% on every turn the Pokemon is still confused.
#[test]
fn the_fifty_percent_self_hit_is_unchanged_at_every_rung() {
    for burned in 0..=MAX_CONFUSED_TURNS {
        let mut state = confused_state(burned);
        let branches = both_splash(&mut state);
        assert_close(
            mass(&branches, hits_itself),
            50.0,
            &format!("self-hit mass with {} turn(s) burned", burned),
        );
    }
}

/// Own Tempo is immune, so no turn is ever burned and no branch is ever created.
#[test]
fn own_tempo_never_starts_the_ladder() {
    let mut state = confused_state(0);
    state.side_one.get_active().ability = poke_engine::engine::abilities::Abilities::OWNTEMPO;
    let branches = both_splash(&mut state);
    assert_eq!(branches.len(), 1, "Own Tempo must not branch: {:?}", branches);
    assert!(!burns_a_confusion_turn(&branches[0].instruction_list));
    assert!(!hits_itself(&branches[0].instruction_list));
}

// ---------------------------------------------------------------------------
// Which turns burn a confusion turn (Showdown onBeforeMove priority order)
// ---------------------------------------------------------------------------

/// Sleep is `onBeforeMovePriority: 10`, confusion is 3, and a handler returning
/// false aborts the event — so a Pokemon that spends the turn asleep never
/// reaches `confusion.onBeforeMove` and `time` is untouched. Gating the ladder
/// on the counter's VALUE rather than on the check having run would burn a turn
/// here for free.
#[test]
fn sleeping_through_a_turn_does_not_burn_a_confusion_turn() {
    let mut state = confused_state(2);
    state.side_one.get_active().status = PokemonStatus::SLEEP;
    let branches = both_splash(&mut state);

    for branch in &branches {
        assert!(
            !burns_a_confusion_turn(&branch.instruction_list),
            "a slept turn must not advance the confusion counter: {:?}",
            branch.instruction_list
        );
        assert!(
            !drops_confusion(&branch.instruction_list),
            "a slept turn must not roll the snap-out ladder: {:?}",
            branch.instruction_list
        );
    }
}

/// Paralysis is `onBeforeMovePriority: 1` — BELOW confusion's 3 — so Showdown
/// resolves the confusion check first and a confusion self-hit aborts the move
/// before paralysis is ever rolled. Two consequences pinned here: the self-hit
/// keeps its full 50% (not 3/8, which is what rolling the 25% full-paralysis
/// branch first would leave), and a fully-paralyzed turn still burns a confusion
/// turn on every branch.
#[test]
fn paralysis_is_rolled_after_the_confusion_check() {
    let mut state = confused_state(1);
    state.side_one.get_active().status = PokemonStatus::PARALYZE;
    let branches = both_splash(&mut state);

    assert_close(
        mass(&branches, hits_itself),
        50.0,
        "self-hit mass for a paralyzed + confused Pokemon",
    );
    assert_close(
        mass(&branches, drops_confusion),
        100.0 / 3.0,
        "snap-out mass for a paralyzed + confused Pokemon",
    );
    for branch in &branches {
        assert!(
            burns_a_confusion_turn(&branch.instruction_list),
            "every paralysis branch must still burn a confusion turn: {:?}",
            branch.instruction_list
        );
    }
}

// ---------------------------------------------------------------------------
// Composition with the residual deferral (residual-defer-on-faint patch)
// ---------------------------------------------------------------------------

fn removals_of_confusion(list: &[Instruction]) -> usize {
    list.iter()
        .filter(|instruction| match instruction {
            Instruction::RemoveVolatileStatus(remove) => {
                remove.side_ref == SideReference::SideOne
                    && remove.volatile_status == PokemonVolatileStatus::CONFUSION
            }
            _ => false,
        })
        .count()
}

/// The two end-of-turn patches share the `add_end_of_turn_branches` fork site,
/// so their composition needs pinning, not just review.
///
/// A confused Pokemon KOs the opposing active. `end_of_turn_is_deferred` then
/// suppresses the residual block on this ply and re-attaches it to the
/// replacement ply. The confusion ladder must NOT follow the residual block: the
/// counter was burned by THIS ply's confusion check, so the ladder belongs here.
/// It is keyed on the `+1` marker in the instruction list rather than on the
/// residual block, so it fires exactly once, on the deferring ply, at the same
/// mass it would have without the deferral.
#[test]
fn confusion_ladder_fires_once_when_residuals_are_deferred() {
    let mut state = confused_state(1);
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::SWIFT);
    state.side_two.get_active().hp = 1;

    let branches = both_splash(&mut state);

    // Same rung as the undeferred case: burning the second turn ends confusion
    // with probability 1/3 whether or not the opponent fainted.
    assert_close(
        mass(&branches, drops_confusion),
        100.0 / 3.0,
        "snap-out mass on a ply whose residuals are deferred",
    );
    for branch in &branches {
        assert!(
            burns_a_confusion_turn(&branch.instruction_list),
            "the confusion check ran, so every branch carries the marker: {:?}",
            branch.instruction_list
        );
        assert!(
            removals_of_confusion(&branch.instruction_list) <= 1,
            "the ladder must not fire twice within one ply: {:?}",
            branch.instruction_list
        );
    }

    // Take the line where the KO landed and the confusion survived, so the
    // replacement ply starts with a live counter and a pending force switch.
    let ko_survivor = branches
        .iter()
        .find(|branch| {
            !drops_confusion(&branch.instruction_list)
                && branch
                    .instruction_list
                    .contains(&Instruction::ToggleSideTwoForceSwitch)
        })
        .expect("a KO branch that keeps the confusion")
        .clone();
    state.apply_instructions(&ko_survivor.instruction_list);
    assert_eq!(
        state.side_one.volatile_status_durations.confusion, 2,
        "the burned turn must survive onto the replacement ply"
    );

    // The replacement ply re-attaches the deferred residual block, and
    // `end_of_turn_triggered` lets it through on the force_switch flag — so
    // `add_end_of_turn_branches` runs a SECOND time this turn. Its instruction
    // list starts empty and no confusion check runs on a replacement ply, so
    // there is no marker and the ladder must stay silent.
    let replacement = generate(
        &mut state,
        &MoveChoice::None,
        &MoveChoice::Switch(PokemonIndex::P1),
    );
    for branch in &replacement {
        assert!(
            !burns_a_confusion_turn(&branch.instruction_list),
            "a replacement ply must not burn a confusion turn: {:?}",
            branch.instruction_list
        );
        assert_eq!(
            removals_of_confusion(&branch.instruction_list),
            0,
            "the ladder must not fire again on the replacement ply: {:?}",
            branch.instruction_list
        );
    }
}

// ---------------------------------------------------------------------------
// Baton Pass carry-over
// ---------------------------------------------------------------------------

/// Drive a Baton Pass to completion: the move arms `force_switch` +
/// `baton_passing` and the switch resolves at the next decision boundary, so the
/// carry-over is only observable after both plies. Returns the switch ply's
/// single branch.
fn baton_pass_then_switch(state: &mut State) -> Vec<Instruction> {
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::BATONPASS);

    let pass = both_splash(state);
    // Take the line where the passer neither hit itself nor snapped out, so the
    // confusion that reaches the switch still has a live counter.
    let armed = pass
        .iter()
        .find(|branch| {
            !drops_confusion(&branch.instruction_list) && !hits_itself(&branch.instruction_list)
        })
        .expect("a Baton Pass branch that keeps the confusion")
        .clone();
    state.apply_instructions(&armed.instruction_list);
    assert!(
        state.side_one.baton_passing,
        "Baton Pass must arm the pass before the switch resolves"
    );

    let switch = generate(
        state,
        &MoveChoice::Switch(PokemonIndex::P1),
        &MoveChoice::None,
    );
    assert_eq!(switch.len(), 1, "the switch ply must be deterministic");
    switch.into_iter().next().unwrap().instruction_list
}

/// Showdown gen3 ground truth: `copyVolatileFrom` copies every volatile without
/// a `noCopy` flag, and `confusion` carries none anywhere in gen3's chain (base
/// `data/conditions.ts`; neither the gen4 nor the gen6 override adds one). The
/// copy is a shallow spread of the volatile object, so the remaining `time`
/// rides across untouched and `onStart` never re-runs — the receiver does NOT
/// get a fresh duration roll.
///
/// Upstream dropped confusion entirely on a pass. Carrying it was deliberately
/// deferred until the duration existed, because carrying a PERMANENT confusion
/// would have been worse than dropping it.
#[test]
fn baton_pass_carries_confusion_with_its_remaining_duration() {
    let mut state = confused_state(1);
    let list = baton_pass_then_switch(&mut state);

    assert!(
        !drops_confusion(&list),
        "Baton Pass must carry CONFUSION to the receiver: {:?}",
        list
    );
    state.apply_instructions(&list);
    assert!(
        state
            .side_one
            .volatile_statuses
            .contains(&PokemonVolatileStatus::CONFUSION),
        "the receiver must arrive confused"
    );
    assert_eq!(
        state.side_one.volatile_status_durations.confusion, 2,
        "the receiver must inherit the passer's burned-turn count, not a fresh roll"
    );
}

/// The inherited counter is a real position on the ladder, not decoration: a
/// receiver that arrives two turns into the ladder rolls the rung-2 snap-out
/// (1/2), not the fresh-confusion rung (1/4). A receiver given a fresh roll — or
/// no counter at all — would sit at 1/4 here.
#[test]
fn a_passed_confusion_resumes_the_ladder_where_the_passer_left_it() {
    let mut state = confused_state(1);
    let list = baton_pass_then_switch(&mut state);
    state.apply_instructions(&list);
    assert_eq!(state.side_one.volatile_status_durations.confusion, 2);

    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::SPLASH);
    let branches = both_splash(&mut state);
    assert_close(
        mass(&branches, drops_confusion),
        50.0,
        "receiver's snap-out mass two turns into the ladder",
    );
    assert_close(
        mass(&branches, hits_itself),
        50.0,
        "receiver's self-hit mass two turns into the ladder",
    );
}

/// Negative control: an ordinary switch still drops confusion — Showdown's
/// `Pokemon.clearVolatile()` blanks the whole volatile table — and must zero the
/// counter with it, or the next confusion on that side would start mid-ladder
/// and `reverse_instructions` would desync.
#[test]
fn an_ordinary_switch_drops_confusion_and_zeroes_the_counter() {
    let mut state = confused_state(3);
    let switch = generate(
        &mut state,
        &MoveChoice::Switch(PokemonIndex::P1),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    );
    assert_eq!(switch.len(), 1, "an ordinary switch must be deterministic");
    let list = switch.into_iter().next().unwrap().instruction_list;

    assert!(
        drops_confusion(&list),
        "a plain switch-out must clear confusion: {:?}",
        list
    );
    state.apply_instructions(&list);
    assert_eq!(
        state.side_one.volatile_status_durations.confusion, 0,
        "a plain switch-out must zero the confusion counter: {:?}",
        list
    );
}

/// Freshly applied confusion starts at the bottom of the ladder even if the side
/// carries a stale counter — the case a state deserialized from outside the
/// engine can produce.
#[test]
fn a_new_confusion_starts_at_the_bottom_of_the_ladder() {
    let mut state = splash_state();
    state.side_one.volatile_status_durations.confusion = 3;
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::CONFUSERAY);

    let branches = generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    );
    assert_eq!(branches.len(), 1, "Confuse Ray is deterministic here");
    state.apply_instructions(&branches[0].instruction_list);

    assert!(state
        .side_one
        .volatile_statuses
        .contains(&PokemonVolatileStatus::CONFUSION));
    assert_eq!(
        state.side_one.volatile_status_durations.confusion, 0,
        "a fresh confusion must reset the counter: {:?}",
        branches[0].instruction_list
    );
}
