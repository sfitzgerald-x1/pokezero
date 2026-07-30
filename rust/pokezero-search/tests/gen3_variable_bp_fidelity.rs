//! Gen 3 variable-base-power fidelity pins, asserted directly against the
//! vendored gen3-patched poke-engine (`third_party/poke-engine-src/`).
//!
//! Ground truth is `data/mods/gen3/moves.ts`, which declares its OWN Flail and
//! Reversal callbacks — gen3 does **not** inherit gen4's here, and the two
//! ladders differ (48ths with cut points 2/5/10/17/33 vs 64ths with
//! 2/6/13/22/43):
//!
//! ```js
//! const ratio = Math.max(Math.floor(pokemon.hp * 48 / pokemon.maxhp), 1);
//! if (ratio < 2) bp = 200; else if (ratio < 5) bp = 150;
//! else if (ratio < 10) bp = 100; else if (ratio < 17) bp = 80;
//! else if (ratio < 33) bp = 40; else bp = 20;
//! ```
//!
//! What this guards (`poke-engine-gen3-variable-bp.patch`):
//!
//! * Flail had no arm and no base power at all — inert at every HP fraction.
//! * Reversal had the ladder as ROUNDED FLOAT ratios, which misclassifies a thin
//!   band on the wrong side of four of the five breakpoints. Every pin below
//!   sits on an EXACT boundary HP, one on each side, because that is the only
//!   place the two spellings disagree.

use poke_engine::choices::MOVES;
use poke_engine::choices::{Choices, MoveCategory};
use poke_engine::engine::abilities::{
    ability_modify_attack_against, ability_modify_attack_being_used, Abilities,
};
use poke_engine::engine::choice_effects::modify_choice;
use poke_engine::engine::state::MoveChoice;
use poke_engine::state::{PokemonMoveIndex, PokemonType, SideReference, State};

/// Base power the engine assigns to `move_id` with the attacker at `hp`/`maxhp`.
fn base_power_at(move_id: Choices, hp: i16, maxhp: i16) -> f32 {
    let mut state = State::default();
    {
        let attacker = state.side_one.get_active();
        attacker.maxhp = maxhp;
        attacker.hp = hp;
        attacker.replace_move(PokemonMoveIndex::M0, move_id);
    }
    let mut choice = state.side_one.get_active_immutable().moves[&PokemonMoveIndex::M0]
        .choice
        .clone();
    let defender_choice = state.side_two.get_active_immutable().moves[&PokemonMoveIndex::M0]
        .choice
        .clone();
    modify_choice(
        &state,
        &mut choice,
        &defender_choice,
        &SideReference::SideOne,
    );
    choice.base_power
}

/// Showdown's own arithmetic, used to derive the expectations rather than
/// restating a table by hand.
fn showdown_base_power(hp: i32, maxhp: i32) -> f32 {
    let ratio = std::cmp::max(hp * 48 / maxhp, 1);
    if ratio < 2 {
        200.0
    } else if ratio < 5 {
        150.0
    } else if ratio < 10 {
        100.0
    } else if ratio < 17 {
        80.0
    } else if ratio < 33 {
        40.0
    } else {
        20.0
    }
}

const LADDER_MOVES: [Choices; 2] = [Choices::FLAIL, Choices::REVERSAL];

/// The plain ladder at a max HP where every band is comfortably wide.
///
/// Flail was inert — `base_power` 0.0 at every one of these — which is the whole
/// reason the move contributed a missing-damage-component row to the strict
/// matcher rather than a wrong-value one.
#[test]
fn the_ladder_climbs_as_hp_falls() {
    // 480 max HP makes one 48th exactly 10 HP, so the bands land on round numbers.
    for move_id in LADDER_MOVES {
        // One 48th of 480 is exactly 10 HP, so ratio == floor(hp / 10) and the
        // bands are [330, 480] = 20, [170, 329] = 40, [100, 169] = 80,
        // [50, 99] = 100, [20, 49] = 150, [1, 19] = 200. Each pair below is the
        // first and last HP of a band.
        for (hp, expected) in [
            (480, 20.0),
            (330, 20.0),
            (329, 40.0),
            (170, 40.0),
            (169, 80.0),
            (100, 80.0),
            (99, 100.0),
            (50, 100.0),
            (49, 150.0),
            (20, 150.0),
            (19, 200.0),
            (1, 200.0),
        ] {
            assert_eq!(
                base_power_at(move_id, hp, 480),
                expected,
                "{:?} at {}/480",
                move_id,
                hp
            );
        }
    }
}

/// Every breakpoint, pinned at the exact HP on each side of it.
///
/// This is the assertion the rounded-float spelling fails: with 480 max HP the
/// `ratio >= 33` boundary is exactly 330 HP (0.6875), and the old constant
/// 0.688 put 330 in the 40 BP band instead of the 20 BP band.
#[test]
fn every_breakpoint_is_exact_on_both_sides() {
    let maxhp: i16 = 480;
    for move_id in LADDER_MOVES {
        // ratio boundaries 2, 5, 10, 17, 33 -> the first HP that reaches each.
        for ratio in [2, 5, 10, 17, 33] {
            let boundary = (ratio * maxhp as i32 / 48) as i16;
            assert_eq!(
                (boundary as i32 * 48) / maxhp as i32,
                ratio,
                "fixture error: {} HP is not the ratio-{} boundary",
                boundary,
                ratio
            );
            for hp in [boundary - 1, boundary] {
                assert_eq!(
                    base_power_at(move_id, hp, maxhp),
                    showdown_base_power(hp as i32, maxhp as i32),
                    "{:?} at {}/{} (ratio {})",
                    move_id,
                    hp,
                    maxhp,
                    (hp as i32 * 48) / maxhp as i32
                );
            }
        }
    }
}

/// A max HP that is NOT a multiple of 48, where the integer floor and any float
/// threshold part company most often. Sweeping every HP value leaves the
/// rounded-float spelling nowhere to hide.
#[test]
fn the_ladder_matches_showdown_at_every_hp_for_an_awkward_maxhp() {
    for maxhp in [261i16, 301, 353, 461] {
        for move_id in LADDER_MOVES {
            for hp in 1..=maxhp {
                assert_eq!(
                    base_power_at(move_id, hp, maxhp),
                    showdown_base_power(hp as i32, maxhp as i32),
                    "{:?} at {}/{}",
                    move_id,
                    hp,
                    maxhp
                );
            }
        }
    }
}

/// The four HP percentages where gen3's own ladder and gen4's disagree.
///
/// This is the pin that catches an implementer who followed the usual
/// gen3-inherits-gen4 rule here. At 100 max HP the two ladders differ at exactly
/// 4%, 10%, 35% and 68% — and gen3 is the HIGHER one at all four, so borrowing
/// gen4's table systematically under-powers both moves in precisely the
/// desperation region they exist for.
#[test]
fn the_four_percentages_where_gen3_and_gen4_disagree_follow_gen3() {
    fn gen4_base_power(hp: i32, maxhp: i32) -> f32 {
        let ratio = std::cmp::max(hp * 64 / maxhp, 1);
        if ratio < 2 {
            200.0
        } else if ratio < 6 {
            150.0
        } else if ratio < 13 {
            100.0
        } else if ratio < 22 {
            80.0
        } else if ratio < 43 {
            40.0
        } else {
            20.0
        }
    }

    for move_id in LADDER_MOVES {
        for (percent, gen3_bp, gen4_bp) in [
            (4, 200.0, 150.0),
            (10, 150.0, 100.0),
            (35, 80.0, 40.0),
            (68, 40.0, 20.0),
        ] {
            let observed = base_power_at(move_id, percent, 100);
            assert_eq!(
                observed, gen3_bp,
                "{:?} at {}% must follow gen3's 48-scale ladder",
                move_id, percent
            );
            assert_eq!(
                gen4_base_power(percent as i32, 100),
                gen4_bp,
                "fixture error: gen4 reference ladder drifted at {}%",
                percent
            );
            assert_ne!(
                observed, gen4_bp,
                "{:?} at {}% must NOT follow gen4's 64-scale ladder",
                move_id, percent
            );
        }
    }
}

/// Showdown uses fixed-point `chainModify`, which rounds a 1.5x or 0.5x
/// modifier down. The retained c15 rows include every member of these two
/// families; 95 is the odd base power that distinguishes the old float behavior
/// (142.5 / 47.5) from the integer value damage calculation must receive.
#[test]
fn gen3_base_power_modifiers_round_odd_products_down() {
    let defender_choice = MOVES[&Choices::TACKLE].clone();
    for (ability, move_type) in [
        (Abilities::TORRENT, PokemonType::WATER),
        (Abilities::BLAZE, PokemonType::FIRE),
        (Abilities::OVERGROW, PokemonType::GRASS),
        (Abilities::SWARM, PokemonType::BUG),
    ] {
        let mut state = State::default();
        let attacker = state.side_one.get_active();
        attacker.ability = ability;
        attacker.hp = 100;
        attacker.maxhp = 300;
        let mut choice = MOVES[&Choices::TACKLE].clone();
        choice.base_power = 95.0;
        choice.move_type = move_type;
        ability_modify_attack_being_used(
            &state,
            &mut choice,
            &defender_choice,
            &SideReference::SideOne,
        );
        assert_eq!(choice.base_power, 142.0, "{:?}", ability);
    }

    let mut state = State::default();
    state.side_two.get_active().ability = Abilities::THICKFAT;
    for move_type in [PokemonType::FIRE, PokemonType::ICE] {
        let mut choice = MOVES[&Choices::TACKLE].clone();
        choice.base_power = 95.0;
        choice.move_type = move_type;
        ability_modify_attack_against(
            &state,
            &mut choice,
            &defender_choice,
            &SideReference::SideOne,
        );
        assert_eq!(choice.base_power, 47.0, "{:?}", move_type);
    }
}

/// `Math.max(..., 1)` floors the ratio at 1, so a Pokemon too big for one 48th
/// of its max HP to reach 1 still lands in the 200 BP band rather than dividing
/// to zero.
#[test]
fn the_ratio_floor_of_one_holds_at_the_bottom() {
    for move_id in LADDER_MOVES {
        for maxhp in [100i16, 261, 480, 714] {
            assert_eq!(
                base_power_at(move_id, 1, maxhp),
                200.0,
                "{:?} at 1/{}",
                move_id,
                maxhp
            );
        }
    }
}

/// Both moves keep their own type and category — the shared ladder must not
/// have flattened Flail into Reversal.
#[test]
fn the_two_moves_keep_their_own_identity() {
    let mut state = State::default();
    for (move_id, expected_type) in [
        (Choices::FLAIL, poke_engine::state::PokemonType::NORMAL),
        (Choices::REVERSAL, poke_engine::state::PokemonType::FIGHTING),
    ] {
        state
            .side_one
            .get_active()
            .replace_move(PokemonMoveIndex::M0, move_id);
        let choice = state.side_one.get_active_immutable().moves[&PokemonMoveIndex::M0]
            .choice
            .clone();
        assert_eq!(choice.move_type, expected_type, "{:?} type", move_id);
        assert_eq!(
            choice.category,
            MoveCategory::Physical,
            "{:?} category",
            move_id
        );
    }
}

/// End to end: the ladder has to reach real damage, not just the Choice. A
/// 1 HP attacker must out-damage a full-HP one by roughly the 200/20 BP ratio.
#[test]
fn the_ladder_reaches_generated_damage() {
    fn damage_at(move_id: Choices, hp: i16) -> i16 {
        let mut state = State::default();
        {
            let attacker = state.side_one.get_active();
            attacker.maxhp = 480;
            attacker.hp = hp;
            attacker.replace_move(PokemonMoveIndex::M0, move_id);
        }
        {
            // The default defender has 100 max HP, which a 200 BP hit overkills
            // — the damage would be capped at its HP and the ratio below would
            // measure the cap instead of the ladder.
            let defender = state.side_two.get_active();
            defender.maxhp = 3000;
            defender.hp = 3000;
        }
        let instructions =
            poke_engine::engine::generate_instructions::generate_instructions_from_move_pair(
                &mut state,
                &MoveChoice::Move(PokemonMoveIndex::M0),
                &MoveChoice::None,
                false,
            );
        instructions
            .iter()
            .flat_map(|branch| branch.instruction_list.iter())
            .filter_map(|instruction| match instruction {
                poke_engine::instruction::Instruction::Damage(damage)
                    if damage.side_ref == SideReference::SideTwo =>
                {
                    Some(damage.damage_amount)
                }
                _ => None,
            })
            .max()
            .unwrap_or(0)
    }

    for move_id in LADDER_MOVES {
        let weak = damage_at(move_id, 480);
        let desperate = damage_at(move_id, 1);
        assert!(weak > 0, "{:?} must deal damage at full HP", move_id);
        assert!(
            desperate >= weak * 5,
            "{:?}: 200 BP at 1 HP ({}) must dwarf 20 BP at full HP ({})",
            move_id,
            desperate,
            weak
        );
    }
}
