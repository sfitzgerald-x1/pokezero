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

use poke_engine::engine::abilities::Abilities;
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

/// Mirrors `CONFUSION_SNAP_OUT_PENDING` in the engine. The end-of-turn ladder no
/// longer removes the volatile; it parks the counter here so the `|-end|` can be
/// announced on the next move, which is where Showdown announces it.
const CONFUSION_SNAP_OUT_PENDING: i8 = -4;

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

/// The ladder's snap-out as it is now expressed at end of turn: the counter is
/// parked on the sentinel instead of the volatile being removed.
///
/// Keyed on `amount <= CONFUSION_SNAP_OUT_PENDING`. Parking from a live rung `d`
/// in `1..=4` writes `-4 - d`, so every real mark is at most `-5`, and the `+1`
/// confusion-check burn is nowhere near it.
///
/// It is NOT disjoint from the switch-out reset in general: a switch at rung 4
/// emits exactly `-4`, which satisfies this predicate. Rung 4 is unreachable
/// from internal play -- the ladder forces the snap-out there -- but a state
/// deserialized from outside the engine can carry it, so the overlap is real
/// rather than theoretical. It is safe here only because a switch never shares a
/// the ladder, and no test mixes the two. Anything that keys on this predicate
/// across a switch boundary needs a different discriminator.
fn defers_snap_out(list: &[Instruction]) -> bool {
    list.iter().any(|instruction| match instruction {
        Instruction::ChangeVolatileStatusDuration(change) => {
            change.side_ref == SideReference::SideOne
                && change.volatile_status == PokemonVolatileStatus::CONFUSION
                && change.amount <= CONFUSION_SNAP_OUT_PENDING
        }
        _ => false,
    })
}

/// Confusion stops carrying self-hit rolls after this branch, whether the
/// volatile is removed outright (switch, Own Tempo) or parked pending its
/// announcement (the end-of-turn ladder).
///
/// Every probability pin below is stated against this predicate and keeps the
/// number it had when the ladder removed the volatile in place. That equality
/// is the evidence the deferral moved WHEN the end is announced without moving
/// any mass.
fn snaps_out(list: &[Instruction]) -> bool {
    drops_confusion(list) || defers_snap_out(list)
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
            mass(&branches, snaps_out),
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
        let ends_now = mass(&branches, snaps_out) / 100.0;
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
/// after four attacking turns EVERY branch has snapped out, and one ply later
/// the volatile is gone and the counter is back to zero — so a permanent
/// confusion cannot survive here.
///
/// The last ply is the point. The ladder now parks the snap-out instead of
/// applying it, so "the volatile is gone" is only true after the next move
/// consumes the mark. Stopping at the mark would pin a deferral that is allowed
/// never to land, which is a confusion that never ends — the exact upstream bug
/// this file exists to keep fixed.
#[test]
fn confusion_never_persists_past_four_attacking_turns() {
    let mut state = confused_state(0);
    for turn in 1..=MAX_CONFUSED_TURNS {
        let branches = both_splash(&mut state);
        if turn < MAX_CONFUSED_TURNS {
            let survivor = branches
                .iter()
                .find(|branch| {
                    !snaps_out(&branch.instruction_list)
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
                    snaps_out(&branch.instruction_list),
                    "confusion must snap out after {} attacking turns: {:?}",
                    MAX_CONFUSED_TURNS,
                    branch.instruction_list
                );
            }
            let survivor = branches[0].clone();
            state.apply_instructions(&survivor.instruction_list);
            assert!(
                state
                    .side_one
                    .volatile_statuses
                    .contains(&PokemonVolatileStatus::CONFUSION),
                "the volatile is held one ply so the next move can announce its end"
            );
            assert_eq!(
                state.side_one.volatile_status_durations.confusion,
                CONFUSION_SNAP_OUT_PENDING,
                "the forced rung must park the counter on the sentinel"
            );
        }
    }

    // The ply the deferral exists for. It must be deterministic, remove the
    // volatile, carry no self-hit roll, and not burn a fifth confused turn.
    let snap = both_splash(&mut state);
    assert_eq!(
        snap.len(),
        1,
        "a parked snap-out leaves nothing left to roll: {:?}",
        snap
    );
    let list = snap.into_iter().next().unwrap().instruction_list;
    assert!(
        drops_confusion(&list),
        "the parked snap-out must actually remove the volatile: {:?}",
        list
    );
    assert!(
        !hits_itself(&list),
        "Showdown's snap-out turn lets the move through with no self-hit: {:?}",
        list
    );
    assert!(
        !burns_a_confusion_turn(&list),
        "the snap-out turn is not a confused attacking turn: {:?}",
        list
    );
    state.apply_instructions(&list);
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
        mass(&branches, snaps_out),
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

/// Counts the ladder firing, however it expresses itself.
///
/// This deliberately counts BOTH the outright removal and the parked mark. The
/// removal count alone is now zero on every end-of-turn branch, so a
/// `<= 1`/`== 0` pin written against it would still be green with the ladder
/// firing twice — the "fires once" guard would have quietly stopped guarding.
fn snap_outs_of_confusion(list: &[Instruction]) -> usize {
    list.iter()
        .filter(|instruction| match instruction {
            Instruction::RemoveVolatileStatus(remove) => {
                remove.side_ref == SideReference::SideOne
                    && remove.volatile_status == PokemonVolatileStatus::CONFUSION
            }
            Instruction::ChangeVolatileStatusDuration(change) => {
                change.side_ref == SideReference::SideOne
                    && change.volatile_status == PokemonVolatileStatus::CONFUSION
                    && change.amount <= CONFUSION_SNAP_OUT_PENDING
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
        mass(&branches, snaps_out),
        100.0 / 3.0,
        "snap-out mass on a ply whose residuals are deferred",
    );
    for branch in &branches {
        assert!(
            burns_a_confusion_turn(&branch.instruction_list),
            "the confusion check ran, so every branch carries the marker: {:?}",
            branch.instruction_list
        );
        // Pinned in BOTH directions, per branch. `<= 1` alone is satisfied by
        // zero, so a helper that had stopped recognising the ladder entirely
        // would still pass it -- the guard would read green while guarding
        // nothing. Mutation-checked: forcing the counter to 0 survives `<= 1`
        // and is killed by this.
        let expected = usize::from(snaps_out(&branch.instruction_list));
        assert_eq!(
            snap_outs_of_confusion(&branch.instruction_list),
            expected,
            "the ladder must fire exactly once on a branch that snaps out, and \
             not at all on one that does not: {:?}",
            branch.instruction_list
        );
    }

    // Take the line where the KO landed and the confusion survived, so the
    // replacement ply starts with a live counter and a pending force switch.
    let ko_survivor = branches
        .iter()
        .find(|branch| {
            !snaps_out(&branch.instruction_list)
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
            snap_outs_of_confusion(&branch.instruction_list),
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
        mass(&branches, snaps_out),
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

/// A turn the mon cannot act on must not consume the parked snap-out.
///
/// Found by independent review, and it was a REGRESSION rather than a missing
/// nicety: the ladder's parked snap-out was published on a Rest-asleep turn, so
/// the publicly visible CONFUSION volatile was deleted from the world with no
/// protocol line to explain it -- and because the renderer's `DecrementRestTurns`
/// arm emits `|cant|slp` and drains the rest of the list, the world was ACCEPTED.
/// Silently wrong beats refused only in the wrong direction; before the deferral
/// this position was always refused.
///
/// Showdown resolves sleep at `onBeforeMove` priority 10 and confusion at 3, and
/// the sleep handler returns false, short-circuiting the event. So neither the
/// snap-out nor the counter burn happens on a turn spent asleep.
#[test]
fn a_rest_asleep_turn_neither_consumes_the_park_nor_burns_a_turn() {
    // Early Bird is a SEPARATELY gated arm (`3 if EARLYBIRD`, which decrements
    // twice and lands at 1), so it needs its own case -- review pointed out that
    // deleting its gate line would otherwise go undetected.
    for (rest_turns, early_bird) in [(2i8, false), (3i8, false), (3i8, true)] {
        let mut state = confused_state(0);
        state.side_one.volatile_status_durations.confusion = CONFUSION_SNAP_OUT_PENDING;
        state.side_one.get_active().status = PokemonStatus::SLEEP;
        state.side_one.get_active().rest_turns = rest_turns;
        if early_bird {
            state.side_one.get_active().ability = Abilities::EARLYBIRD;
        }

        let branches = both_splash(&mut state);
        for branch in &branches {
            assert!(
                !drops_confusion(&branch.instruction_list),
                "rest_turns={} early_bird={}: the park must not be consumed while asleep: {:?}",
                rest_turns,
                early_bird,
                branch.instruction_list
            );
            assert!(
                !burns_a_confusion_turn(&branch.instruction_list),
                "rest_turns={} early_bird={}: a turn spent asleep burns no confusion turn: {:?}",
                rest_turns,
                early_bird,
                branch.instruction_list
            );
            assert!(
                !hits_itself(&branch.instruction_list),
                "rest_turns={} early_bird={}: no self-hit roll while asleep: {:?}",
                rest_turns,
                early_bird,
                branch.instruction_list
            );
        }

        let survivor = branches[0].clone();
        state.apply_instructions(&survivor.instruction_list);
        assert!(
            state
                .side_one
                .volatile_statuses
                .contains(&PokemonVolatileStatus::CONFUSION),
            "rest_turns={} early_bird={}: the volatile must survive the sleeping turn",
            rest_turns,
            early_bird
        );
        assert_eq!(
            state.side_one.volatile_status_durations.confusion,
            CONFUSION_SNAP_OUT_PENDING,
            "rest_turns={} early_bird={}: the park must still be pending",
            rest_turns,
            early_bird
        );
    }
}

/// Positive control for the gate above, so it cannot be widened into "asleep
/// never snaps out". Sleep Talk is gen3's only `sleepUsable` move: `slp`
/// announces `|cant|slp` and returns UNDEFINED, so the BeforeMove event keeps
/// running and confusion's handler DOES fire. A parked mon using Sleep Talk
/// therefore snaps out on schedule.
#[test]
fn a_sleep_talk_turn_still_consumes_the_parked_snap_out() {
    let mut state = confused_state(0);
    state.side_one.volatile_status_durations.confusion = CONFUSION_SNAP_OUT_PENDING;
    state.side_one.get_active().status = PokemonStatus::SLEEP;
    state.side_one.get_active().rest_turns = 2;
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::SLEEPTALK);
    // Sleep Talk needs something to CALL. Without a second move
    // `get_sleep_talk_choices()` is empty, the recursion emits nothing, and the
    // branch set comes back EMPTY -- which makes `.all()` trivially true and the
    // whole control vacuous. That is exactly how the first version of this test
    // shipped, and review caught it: mutating both Rest arms to
    // `reaches_confusion_handler = false` -- the precise over-broad gate this
    // test exists to prevent -- left it green.
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::TACKLE);

    let branches = both_splash(&mut state);
    assert!(
        !branches.is_empty(),
        "the fixture must actually produce Sleep Talk branches, or this control \
         asserts nothing"
    );
    assert!(
        branches
            .iter()
            .all(|branch| drops_confusion(&branch.instruction_list)),
        "every Sleep Talk branch must consume the park: {:?}",
        branches
    );
}
