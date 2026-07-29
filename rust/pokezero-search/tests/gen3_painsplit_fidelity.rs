//! Gen 3 Pain Split assignment pins, asserted directly against the vendored
//! gen3-patched poke-engine (`third_party/poke-engine-src/`).
//!
//! Ground truth is `painsplit.onHit` (data/moves.ts — no gen mod in the chain
//! overrides Pain Split, so gen3 inherits the base) plus `Pokemon#sethp`
//! (sim/pokemon.ts:1656), which is where the clamping lives:
//!
//! ```text
//! const averagehp = Math.floor((targetHP + pokemon.hp) / 2) || 1;
//! target.sethp(...);  pokemon.sethp(averagehp);
//!
//! sethp(d):  if (!this.hp) return 0;         // fainted: no-op
//!            d = trunc(d); if (d < 1) d = 1; // floor at 1, not 0
//!            if (this.hp > this.maxhp) this.hp = this.maxhp;   // clamp
//! ```
//!
//! Upstream assigned the raw average to both actives, so the mon with the
//! smaller maxhp came out ABOVE its maximum — a corrupt hp > maxhp state read by
//! every later damage/heal/faint check.
//!
//! Scope note: the #915 `choice_special_effect` audit verdicted this arm
//! "correct as-is" and that verdict stands — it was about Substitute routing,
//! which this file also re-pins. The audit covered the guard set, not the
//! assignment arithmetic underneath it.

use poke_engine::choices::Choices;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::{MoveChoice, PokemonVolatileStatus};
use poke_engine::instruction::Instruction;
use poke_engine::state::{PokemonMoveIndex, State};

fn generate(state: &mut State) -> Vec<Instruction> {
    let before = format!("{:?}", state);
    let instructions = generate_instructions_from_move_pair(
        state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::None,
        false,
    );
    assert_eq!(before, format!("{:?}", state), "state was mutated");
    assert_eq!(
        instructions.len(),
        1,
        "expected a single deterministic branch, got {:?}",
        instructions
    );
    let list = instructions.into_iter().next().unwrap().instruction_list;

    let mut probe = state.clone();
    let snapshot = format!("{:?}", probe);
    probe.apply_instructions(&list);
    probe.reverse_instructions(&list);
    assert_eq!(
        snapshot,
        format!("{:?}", probe),
        "instructions did not revert"
    );

    list
}

/// Side one Pain Splits. Returns the two actives' HP after the move.
fn split(user: (i16, i16), target: (i16, i16)) -> (i16, i16) {
    let mut state = State::default();
    {
        let a = state.side_one.get_active();
        a.hp = user.0;
        a.maxhp = user.1;
        a.replace_move(PokemonMoveIndex::M0, Choices::PAINSPLIT);
    }
    {
        let d = state.side_two.get_active();
        d.hp = target.0;
        d.maxhp = target.1;
    }
    let list = generate(&mut state);
    state.apply_instructions(&list);
    (
        state.side_one.get_active_immutable().hp,
        state.side_two.get_active_immutable().hp,
    )
}

/// Showdown's own arithmetic, so the expectations are derived rather than
/// restated: floor the average once, floor it at 1, then clamp per mon.
fn showdown_split(user: (i16, i16), target: (i16, i16)) -> (i16, i16) {
    let average = std::cmp::max((user.0 as i32 + target.0 as i32) / 2, 1) as i16;
    (
        std::cmp::min(average, user.1),
        std::cmp::min(average, target.1),
    )
}

/// The repro shape, with the sim's exact numbers: a 271-max Weezing splitting
/// with a full 341-max Groudon averages 306. Groudon lands on 306/341 UNCLAMPED
/// (306 < 341); Weezing lands on 271/271, CLAMPED. Upstream gave Weezing 306/271.
#[test]
fn the_lower_max_mon_clamps_and_the_higher_max_mon_does_not() {
    let (weezing, groudon) = split((271, 271), (341, 341));
    assert_eq!(weezing, 271, "the 271-max user clamps to its own maximum");
    assert_eq!(groudon, 306, "the 341-max target keeps the raw average");
}

/// The corrupt state the bug produced, stated as its own invariant: neither mon
/// may ever end above its maximum.
#[test]
fn neither_side_ever_ends_above_its_own_maxhp() {
    for user in [(271, 271), (100, 400), (50, 60), (1, 300), (341, 341)] {
        for target in [(341, 341), (400, 400), (60, 60), (206, 271), (1, 1)] {
            let (u, t) = split(user, target);
            assert!(
                u <= user.1,
                "user {}/{} vs {}/{} ended at {}",
                user.0,
                user.1,
                target.0,
                target.1,
                u
            );
            assert!(
                t <= target.1,
                "target {}/{} vs {}/{} ended at {}",
                target.0,
                target.1,
                user.0,
                user.1,
                t
            );
        }
    }
}

/// The control: when the average is below BOTH maxima nothing clamps, and both
/// mons land on it exactly. Sim-verified — a 71/271 Weezing splitting with a
/// full 341 Groudon averages 206 and both sides read 206.
#[test]
fn with_both_under_the_average_neither_clamps() {
    let (weezing, groudon) = split((71, 271), (341, 341));
    assert_eq!(weezing, 206);
    assert_eq!(groudon, 206);
}

/// The clamp-to-full case from the other direction: the user is the one already
/// at full and the target is the big one.
#[test]
fn the_clamp_applies_to_whichever_side_is_smaller() {
    let (user, target) = split((60, 60), (400, 400));
    assert_eq!(user, 60, "the 60-max user cannot exceed 60");
    assert_eq!(target, 230, "the 400-max target takes the raw average");
}

/// The arithmetic itself: floor ONCE on the sum, and floor the result at 1.
/// An odd total must round down, and a 1+1 split must not produce 0.
#[test]
fn the_average_floors_once_and_never_reaches_zero() {
    // 101 + 200 = 301 -> floor(150.5) = 150
    let (u, t) = split((101, 400), (200, 400));
    assert_eq!((u, t), (150, 150), "odd totals floor down");
    // Both on their last point: the average is 1, not 0.
    let (u, t) = split((1, 300), (1, 300));
    assert_eq!((u, t), (1, 1), "the average is floored at 1");
}

/// Swept against a transcription of Showdown's arithmetic rather than a
/// hand-written table, across max-HP pairs on both sides of every boundary.
#[test]
fn the_split_matches_showdown_across_max_hp_pairs() {
    for &user in &[(271i16, 271i16), (71, 271), (150, 300), (1, 50), (400, 400)] {
        for &target in &[(341i16, 341i16), (206, 271), (60, 60), (1, 1), (255, 300)] {
            assert_eq!(
                split(user, target),
                showdown_split(user, target),
                "user {:?} target {:?}",
                user,
                target
            );
        }
    }
}

/// The #915 audit verdict, re-pinned here because this file edits the same arm:
/// Pain Split still fails behind a Substitute and touches neither mon.
#[test]
fn pain_split_still_fails_behind_a_substitute() {
    let mut state = State::default();
    {
        let a = state.side_one.get_active();
        a.hp = 50;
        a.maxhp = 271;
        a.replace_move(PokemonMoveIndex::M0, Choices::PAINSPLIT);
    }
    {
        let d = state.side_two.get_active();
        d.hp = 341;
        d.maxhp = 341;
    }
    state
        .side_two
        .volatile_statuses
        .insert(PokemonVolatileStatus::SUBSTITUTE);
    state.side_two.substitute_health = 85;

    let list = generate(&mut state);
    state.apply_instructions(&list);
    assert_eq!(state.side_one.get_active_immutable().hp, 50, "{:?}", list);
    assert_eq!(state.side_two.get_active_immutable().hp, 341, "{:?}", list);
}

/// Guards the instruction stream itself, not just the resulting HP: the emitted
/// deltas must be exactly what moves each mon from its old HP to its new one, so
/// a consumer replaying instructions agrees with a consumer reading state.
#[test]
fn the_emitted_deltas_match_the_hp_change_on_both_sides() {
    let mut state = State::default();
    {
        let a = state.side_one.get_active();
        a.hp = 271;
        a.maxhp = 271;
        a.replace_move(PokemonMoveIndex::M0, Choices::PAINSPLIT);
    }
    {
        let d = state.side_two.get_active();
        d.hp = 341;
        d.maxhp = 341;
    }
    let list = generate(&mut state);

    let mut total: i16 = 0;
    for instruction in &list {
        if let Instruction::Damage(damage) = instruction {
            total += damage.damage_amount;
        }
    }
    // The user is at full and clamps, so it moves 0; the target drops 341 -> 306.
    assert_eq!(total, 35, "net emitted delta: {:?}", list);
}

/// Sanity guard on the fixture: a plain unclamped split really does emit a
/// negative delta for the healing side, which is the engine's existing spelling
/// and must survive the round trip.
#[test]
fn the_healing_side_emits_a_negative_delta() {
    let mut state = State::default();
    {
        let a = state.side_one.get_active();
        a.hp = 71;
        a.maxhp = 271;
        a.replace_move(PokemonMoveIndex::M0, Choices::PAINSPLIT);
    }
    {
        let d = state.side_two.get_active();
        d.hp = 341;
        d.maxhp = 341;
    }
    let list: Vec<Instruction> = generate(&mut state);
    let negative = list.iter().any(|instruction| match instruction {
        Instruction::Damage(damage) => damage.damage_amount < 0,
        _ => false,
    });
    assert!(
        negative,
        "the healed side carries a negative delta: {:?}",
        list
    );
}
