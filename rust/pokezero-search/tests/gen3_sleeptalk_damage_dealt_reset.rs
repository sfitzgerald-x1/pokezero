//! Pins for the Sleep Talk DOUBLE `damage_dealt` reset guard
//! (`third_party/poke-engine-gen3-sleeptalk-damage-dealt-double-reset.patch`).
//!
//! THE DEFECT. `generate_instructions_from_move` opens by emitting the turn-start
//! `damage_dealt` carry-over reset (`reset_damage_dealt`). The Sleep Talk block
//! further down calls `state.reverse_instructions(&incoming_instructions
//! .instruction_list)` before recursing into the callee, which UNDOES that reset
//! and restores the pre-reset carry-over into `state`; the callee's own call then
//! re-enters the same opening, reads the restored carry-over, and pushes the SAME
//! reset onto a list that already contains it. `reset_damage_dealt` reads `state`
//! and only APPENDS -- it never mutates the side -- so it cannot see the queued
//! instruction. The guard is `!choice.sleep_talk_move`: the callee is not a second
//! action, it is the same action continuing one level down.
//!
//! WHY IT IS NOT COSMETIC. `ChangeDamageDealtDamage` is a DELTA (`0 - damage`), so
//! applying it twice lands on `-damage` rather than 0, and
//! `ToggleDamageDealtHitSubstitute` is a TOGGLE, so applying it twice RESTORES the
//! flag the reset meant to clear. Only `ChangeDamageDealtMoveCatagory` is an
//! absolute set and therefore idempotent. `damage_dealt.damage == -137` is a value
//! no legal gen3 state can hold, and the opponent's Counter / Mirror Coat reads it
//! within the same turn (`gen3_fixed_damage_amount`).
//!
//! RENDERER CONSEQUENCE, which is what made this the largest divergence shape.
//! `consume_move_prelude` (`src/events.rs`) eats EVERY leading damage-dealt
//! instruction, so the tail handed to `identify_sleep_talk_called` starts after
//! BOTH resets, while that function's single regeneration emits ONE reset at the
//! head. Divergence is at INDEX 0 -- which is why the class always measured zero
//! `shape_branch_is_prefix_of_tail` and zero `shape_tail_is_prefix_of_branch`.
//!
//! `state.use_damage_dealt` is set by `set_conditional_mechanics` whenever ANY of
//! the twelve Pokemon knows Counter / Mirror Coat / Focus Punch, so the population
//! is common in gen3 randbats and team-dependent.

use poke_engine::choices::{Choices, MoveCategory, MOVES};
use poke_engine::engine::generate_instructions::{
    generate_instructions_from_move, generate_instructions_from_move_pair,
};
use poke_engine::engine::items::Items;
use poke_engine::engine::state::MoveChoice;
use poke_engine::instruction::{Instruction, StateInstructions};
use poke_engine::state::{PokemonMoveIndex, PokemonStatus, SideReference, State};

/// The carry-over the previous turn left behind. Any non-zero value reproduces
/// the defect; 137 is the value the original reproduction used.
const CARRIED_DAMAGE: i16 = 137;

fn is_damage_dealt(instruction: &Instruction) -> bool {
    matches!(
        instruction,
        Instruction::ChangeDamageDealtDamage(_)
            | Instruction::ChangeDamageDealtMoveCatagory(_)
            | Instruction::ToggleDamageDealtHitSubstitute(_)
    )
}

/// Sleeper on side two holding Sleep Talk in M0 plus three callees, with a
/// `damage_dealt` carry-over and the flag that makes the engine track it.
///
/// `use_damage_dealt` is set DIRECTLY rather than by giving a mon Counter,
/// deliberately: `set_conditional_mechanics` runs inside `State::deserialize`, not
/// on field assignment, so a test that only added Counter to a moveset would leave
/// the flag false and pass vacuously against the unpatched engine too.
fn sleeper_with_carry_over(
    callees: [Choices; 3],
    defender_move: Choices,
    carried_damage: i16,
    carried_hit_substitute: bool,
) -> State {
    sleeper_with_carry_over_ordered(callees, defender_move, carried_damage, carried_hit_substitute, true)
}

/// As above, but with the sleeper's move ORDER as a parameter.
///
/// Move order is load-bearing and was missed once: `new_choice.first_move =
/// choice.first_move` propagates the outer Sleep Talk's order into the callee, so
/// a guard written as `!choice.sleep_talk_move || !choice.first_move` restores the
/// double reset for every SECOND-MOVING Sleep Talk user while leaving a
/// sleeper-first battery entirely green. The #1048 attribution oracle sweeps both
/// orders for the same reason.
fn sleeper_with_carry_over_ordered(
    callees: [Choices; 3],
    defender_move: Choices,
    carried_damage: i16,
    carried_hit_substitute: bool,
    sleeper_first: bool,
) -> State {
    let mut state = State::default();
    state.use_damage_dealt = true;
    state.side_two.damage_dealt.damage = carried_damage;
    state.side_two.damage_dealt.hit_substitute = carried_hit_substitute;
    state.side_two.get_active().attack = 318;
    state.side_two.get_active().special_attack = 318;
    state.side_one.get_active().attack = 200;
    state.side_one.get_active().defense = 96;
    state.side_one.get_active().special_defense = 96;
    state.side_one.get_active().maxhp = 404;
    state.side_one.get_active().hp = 404;
    state.side_two.get_active().maxhp = 404;
    state.side_two.get_active().hp = 404;
    state.side_two.get_active().item = Items::LEFTOVERS;
    state.side_two.get_active().speed = if sleeper_first { 500 } else { 1 };
    state.side_one.get_active().speed = if sleeper_first { 1 } else { 500 };
    state.side_two.get_active().status = PokemonStatus::SLEEP;
    state.side_two.get_active().rest_turns = 0;
    state.side_two.get_active().sleep_turns = 0;
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::SLEEPTALK);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M1, callees[0]);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M2, callees[1]);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M3, callees[2]);
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, defender_move);
    state
}

/// Generate only the sleeper's own action (not the pair), which is where the
/// double emission happens -- the pair function's end-of-turn stage and
/// `combine_duplicate_instructions` would otherwise obscure the head of the list.
fn sleeper_action_branches(state: &mut State, defender_move: Choices) -> Vec<StateInstructions> {
    let mut sleep_talk = MOVES
        .get(&Choices::SLEEPTALK)
        .expect("SLEEPTALK is in the move table")
        .clone();
    sleep_talk.first_move = true;
    sleep_talk.move_index = PokemonMoveIndex::M0;
    let defender_choice = MOVES
        .get(&defender_move)
        .expect("defender move is in the move table")
        .clone();
    let mut branches: Vec<StateInstructions> = Vec::new();
    generate_instructions_from_move(
        state,
        &mut sleep_talk,
        &defender_choice,
        SideReference::SideTwo,
        StateInstructions::default(),
        &mut branches,
        true,
    );
    branches
}

/// THE PIN. Every Sleep Talk branch carries the carry-over reset EXACTLY ONCE.
///
/// Counted over the WHOLE branch, on the reset's SIGNATURE (`damage_change ==
/// -CARRIED_DAMAGE`), not over the leading run. The two copies are NOT adjacent:
/// the outer call pushes its reset first, then the sleep gate writes
/// `SetSleepTurns`, and only then does the callee's re-entry push the duplicate --
/// so the unpatched head is `[reset, SetSleepTurns, reset, ...]`. An earlier
/// revision of this test used `take_while(is_damage_dealt)`, which stops at
/// `SetSleepTurns` and therefore counted 1 against the DEFECTIVE engine too. It
/// was vacuous, and the mutation battery caught it: mutant M1 (revert the guard)
/// survived it. It now kills M1.
///
/// `-CARRIED_DAMAGE` cannot collide with `set_damage_dealt`'s recording delta,
/// which is the (non-negative) damage the callee actually dealt.
#[test]
fn a_sleep_talk_turn_with_a_carry_over_emits_the_reset_exactly_once() {
    for callees in [
        [Choices::EARTHQUAKE, Choices::TOXIC, Choices::SHADOWBALL],
        [Choices::BODYSLAM, Choices::REST, Choices::SPLASH],
        [Choices::SURF, Choices::SWORDSDANCE, Choices::SEISMICTOSS],
    ] {
        for carried_hit_substitute in [false, true] {
            let mut state = sleeper_with_carry_over(
                callees,
                Choices::TACKLE,
                CARRIED_DAMAGE,
                carried_hit_substitute,
            );
            let branches = sleeper_action_branches(&mut state, Choices::TACKLE);
            assert!(
                !branches.is_empty(),
                "the population must produce branches: {callees:?}"
            );
            for branch in &branches {
                let resets = branch
                    .instruction_list
                    .iter()
                    .filter(|instruction| match instruction {
                        Instruction::ChangeDamageDealtDamage(change) => {
                            change.damage_change == -CARRIED_DAMAGE
                        }
                        _ => false,
                    })
                    .count();
                assert_eq!(
                    resets, 1,
                    "the carry-over reset must be emitted exactly once (callees \
                     {callees:?}, hit_sub {carried_hit_substitute}): {:?}",
                    branch.instruction_list
                );
                // The substitute toggle is the other non-idempotent sub-field.
                // A toggle emitted twice is a no-op on the end state, so it has
                // to be counted structurally rather than through the state.
                if carried_hit_substitute {
                    let toggles = branch
                        .instruction_list
                        .iter()
                        .filter(|instruction| {
                            matches!(
                                instruction,
                                Instruction::ToggleDamageDealtHitSubstitute(_)
                            )
                        })
                        .count();
                    assert_eq!(
                        toggles, 1,
                        "the hit_substitute reset toggle must be emitted exactly \
                         once: {:?}",
                        branch.instruction_list
                    );
                }
                // And the head really is the shape the renderer's prelude walks:
                // a damage-dealt instruction at index 0, which is where the
                // divergence sat.
                assert!(
                    is_damage_dealt(&branch.instruction_list[0]),
                    "the carry-over reset opens the branch: {:?}",
                    branch.instruction_list
                );
            }
        }
    }
}

/// THE SAME PIN WITH THE SLEEPER MOVING SECOND, which the sleeper-first battery
/// above cannot see.
///
/// Added after a mutation gap: `state.use_damage_dealt && (!choice.sleep_talk_move
/// || !choice.first_move)` restores the double reset for every second-moving Sleep
/// Talk user and SURVIVED all six original pins, because every fixture gave the
/// sleeper speed 500 and passed `first_move = true`. `new_choice.first_move =
/// choice.first_move` in the engine's Sleep Talk block is what makes the callee's
/// `first_move` reachable as a discriminator at all.
///
/// Driven through `generate_instructions_from_move_pair` rather than the single
/// action, because a second-moving Sleep Talk only exists inside a pair: the
/// defender's action has to be generated first for `first_move` to be false.
/// Counted on SideTwo only -- side one emits its own reset for its own action, and
/// conflating the two would make this pass for the wrong reason.
#[test]
fn a_second_moving_sleep_talk_user_also_emits_the_reset_exactly_once() {
    let s1 = MoveChoice::Move(PokemonMoveIndex::M0);
    let s2 = MoveChoice::Move(PokemonMoveIndex::M0);
    for callees in [
        [Choices::EARTHQUAKE, Choices::TOXIC, Choices::SHADOWBALL],
        [Choices::BODYSLAM, Choices::REST, Choices::SPLASH],
    ] {
        for carried_hit_substitute in [false, true] {
            let mut state = sleeper_with_carry_over_ordered(
                callees,
                Choices::SPLASH,
                CARRIED_DAMAGE,
                carried_hit_substitute,
                false,
            );
            // The fixture must actually put the sleeper second, or this pin
            // silently degenerates into a duplicate of the one above.
            assert!(
                state.side_two.get_active_immutable().speed
                    < state.side_one.get_active_immutable().speed,
                "the sleeper must be slower than the defender"
            );
            let branches = generate_instructions_from_move_pair(&mut state, &s1, &s2, true);
            assert!(!branches.is_empty());
            for branch in &branches {
                let resets = branch
                    .instruction_list
                    .iter()
                    .filter(|instruction| match instruction {
                        Instruction::ChangeDamageDealtDamage(change) => {
                            change.side_ref == SideReference::SideTwo
                                && change.damage_change == -CARRIED_DAMAGE
                        }
                        _ => false,
                    })
                    .count();
                assert_eq!(
                    resets, 1,
                    "a SECOND-moving Sleep Talk user must emit the carry-over \
                     reset exactly once (callees {callees:?}, hit_sub \
                     {carried_hit_substitute}): {:?}",
                    branch.instruction_list
                );
            }
        }
    }
}

/// NEUTRALITY, stated as the absolute end state rather than as a diff against a
/// variant this build cannot produce.
///
/// A NON-DAMAGING callee must leave `damage_dealt` in the CLEARED state: damage 0,
/// category Physical, `hit_substitute` false. The unpatched engine lands this same
/// branch on `damage == -CARRIED_DAMAGE` (the delta applied twice) and on
/// `hit_substitute == true` (the toggle applied twice), so this pin is the direct
/// negation of the defect and cannot pass against it.
///
/// Toxic against a full-HP defender is the non-damaging callee, and Toxic's
/// accuracy branch means the MISS arm exists too -- both must be cleared.
#[test]
fn a_non_damaging_callee_leaves_damage_dealt_cleared_not_negated() {
    let callees = [Choices::TOXIC, Choices::SWORDSDANCE, Choices::SPLASH];
    for carried_hit_substitute in [false, true] {
        let mut state = sleeper_with_carry_over(
            callees,
            Choices::SPLASH,
            CARRIED_DAMAGE,
            carried_hit_substitute,
        );
        let branches = sleeper_action_branches(&mut state, Choices::SPLASH);
        assert!(!branches.is_empty());
        for branch in &branches {
            let mut end = state.clone();
            end.apply_instructions(&branch.instruction_list);
            assert_eq!(
                end.side_two.damage_dealt.damage, 0,
                "a non-damaging Sleep Talk callee must CLEAR the carry-over, not \
                 negate it: {:?}",
                branch.instruction_list
            );
            assert_eq!(
                end.side_two.damage_dealt.move_category,
                MoveCategory::Physical,
                "{:?}",
                branch.instruction_list
            );
            assert!(
                !end.side_two.damage_dealt.hit_substitute,
                "the hit_substitute toggle must land cleared, not restored: {:?}",
                branch.instruction_list
            );
        }
    }
}

/// A DAMAGING callee must record its own damage, and the recorded value must be
/// the damage actually dealt -- not that damage offset by the carry-over.
///
/// This is the arm where the unpatched engine reaches the SAME end state by a
/// different route (`set_damage_dealt` emits `damage - (-137)` from a corrupted
/// live value), which is why the end state alone does not discriminate here and
/// the INSTRUCTION is pinned as well.
#[test]
fn a_damaging_callee_records_its_own_damage_with_no_carry_over_offset() {
    let callees = [Choices::EARTHQUAKE, Choices::EARTHQUAKE, Choices::EARTHQUAKE];
    let mut state =
        sleeper_with_carry_over(callees, Choices::SPLASH, CARRIED_DAMAGE, false);
    let branches = sleeper_action_branches(&mut state, Choices::SPLASH);
    assert!(!branches.is_empty());
    let mut checked = 0usize;
    for branch in &branches {
        let dealt: i16 = branch
            .instruction_list
            .iter()
            .filter_map(|instruction| match instruction {
                Instruction::Damage(damage) if damage.side_ref == SideReference::SideOne => {
                    Some(damage.damage_amount)
                }
                _ => None,
            })
            .sum();
        if dealt == 0 {
            continue;
        }
        checked += 1;
        let mut end = state.clone();
        end.apply_instructions(&branch.instruction_list);
        assert_eq!(
            end.side_two.damage_dealt.damage, dealt,
            "recorded damage_dealt must equal the damage dealt: {:?}",
            branch.instruction_list
        );
        // The recording instruction itself: the LAST ChangeDamageDealtDamage is
        // `set_damage_dealt`'s, and its delta is measured from the CLEARED value,
        // so it equals the damage. Against the unpatched engine the live value is
        // `-CARRIED_DAMAGE` and this delta is `dealt + CARRIED_DAMAGE`.
        let last_delta = branch
            .instruction_list
            .iter()
            .filter_map(|instruction| match instruction {
                Instruction::ChangeDamageDealtDamage(change) => Some(change.damage_change),
                _ => None,
            })
            .next_back()
            .expect("a damaging callee records damage_dealt");
        assert_eq!(
            last_delta, dealt,
            "the recording delta must be measured from a CLEARED carry-over, not \
             from a doubly-reset one: {:?}",
            branch.instruction_list
        );
    }
    assert!(checked > 0, "no damaging branch was exercised");
}

/// THE OBSERVABLE CONSEQUENCE, and the reason this is a fidelity fix rather than a
/// tidy-up: the corrupted `-CARRIED_DAMAGE` is read by the opponent's Counter in
/// the SAME turn.
///
/// `gen3_fixed_damage_amount` returns `damage_dealt.damage * 2` for Counter when
/// the recorded category is Physical. After a non-damaging Sleep Talk callee the
/// correct reading is 0; the unpatched engine hands Counter `-274`. Pinned on the
/// pair so the defender's action actually runs.
#[test]
fn counter_after_a_non_damaging_sleep_talk_callee_reads_a_cleared_record() {
    let callees = [Choices::TOXIC, Choices::SWORDSDANCE, Choices::SPLASH];
    let mut state = sleeper_with_carry_over(callees, Choices::COUNTER, CARRIED_DAMAGE, false);
    let s1 = MoveChoice::Move(PokemonMoveIndex::M0);
    let s2 = MoveChoice::Move(PokemonMoveIndex::M0);
    let branches = generate_instructions_from_move_pair(&mut state, &s1, &s2, true);
    assert!(!branches.is_empty());
    let mut seen_cleared = 0usize;
    for branch in &branches {
        let mut end = state.clone();
        end.apply_instructions(&branch.instruction_list);
        assert!(
            end.side_two.damage_dealt.damage >= 0,
            "damage_dealt.damage must never go negative -- Counter doubles it: {:?}",
            branch.instruction_list
        );
        if end.side_two.damage_dealt.damage == 0 {
            seen_cleared += 1;
            // Counter read zero, so it must have dealt nothing to the sleeper.
            let counter_damage: i16 = branch
                .instruction_list
                .iter()
                .filter_map(|instruction| match instruction {
                    Instruction::Damage(damage)
                        if damage.side_ref == SideReference::SideTwo =>
                    {
                        Some(damage.damage_amount)
                    }
                    _ => None,
                })
                .sum();
            assert_eq!(
                counter_damage, 0,
                "Counter must deal nothing after a non-damaging callee: {:?}",
                branch.instruction_list
            );
        }
    }
    assert!(
        seen_cleared > 0,
        "the population must reach at least one cleared-record branch"
    );
}

/// The guard is scoped to the CALLEE, not to Sleep Talk turns generally: an
/// ORDINARY move still emits its own carry-over reset.
///
/// Without this, "delete the whole `if state.use_damage_dealt` block" would pass
/// every pin above -- and that revert breaks Counter for every non-Sleep-Talk turn
/// in the game.
#[test]
fn an_ordinary_move_still_resets_its_own_carry_over() {
    let mut state = State::default();
    state.use_damage_dealt = true;
    state.side_two.damage_dealt.damage = CARRIED_DAMAGE;
    state.side_two.get_active().maxhp = 404;
    state.side_two.get_active().hp = 404;
    state.side_one.get_active().maxhp = 404;
    state.side_one.get_active().hp = 404;
    state.side_two.get_active().speed = 500;
    state.side_one.get_active().speed = 1;
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::SPLASH);
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::SPLASH);
    let mut splash = MOVES.get(&Choices::SPLASH).unwrap().clone();
    splash.first_move = true;
    splash.move_index = PokemonMoveIndex::M0;
    assert!(
        !splash.sleep_talk_move,
        "an ordinary move choice must not be flagged as a Sleep Talk callee"
    );
    let defender_choice = MOVES.get(&Choices::SPLASH).unwrap().clone();
    let mut branches: Vec<StateInstructions> = Vec::new();
    generate_instructions_from_move(
        &mut state,
        &mut splash,
        &defender_choice,
        SideReference::SideTwo,
        StateInstructions::default(),
        &mut branches,
        true,
    );
    assert!(!branches.is_empty());
    for branch in &branches {
        let resets = branch
            .instruction_list
            .iter()
            .filter(|instruction| {
                matches!(instruction, Instruction::ChangeDamageDealtDamage(_))
            })
            .count();
        assert_eq!(
            resets, 1,
            "an ordinary move must still clear its own carry-over exactly once: {:?}",
            branch.instruction_list
        );
        let mut end = state.clone();
        end.apply_instructions(&branch.instruction_list);
        assert_eq!(end.side_two.damage_dealt.damage, 0);
    }
}

/// And the flag still gates: with `use_damage_dealt` false, NO reset is emitted at
/// all, for the callee or for anyone else. Pins the `&&` rather than a replacement
/// of the whole condition by `!choice.sleep_talk_move`.
#[test]
fn with_damage_dealt_tracking_off_no_reset_is_emitted() {
    let callees = [Choices::EARTHQUAKE, Choices::TOXIC, Choices::SHADOWBALL];
    let mut state = sleeper_with_carry_over(callees, Choices::TACKLE, CARRIED_DAMAGE, false);
    state.use_damage_dealt = false;
    let branches = sleeper_action_branches(&mut state, Choices::TACKLE);
    assert!(!branches.is_empty());
    for branch in &branches {
        assert!(
            !branch
                .instruction_list
                .iter()
                .any(|instruction| is_damage_dealt(instruction)),
            "damage-dealt tracking is off; nothing may be emitted: {:?}",
            branch.instruction_list
        );
    }
}
