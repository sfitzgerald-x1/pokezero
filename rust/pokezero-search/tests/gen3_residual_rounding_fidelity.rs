//! Gen 3 residual-damage ROUNDING pins, asserted directly against the vendored
//! gen3-patched poke-engine (`third_party/poke-engine-src/`).
//!
//! Every expectation here was read off the **real** Node Showdown simulator
//! driven through `scripts/gen3_switch_differential.py`, which is the
//! ground-truth gate; this file is the engine-contract pin.
//!
//! What it guards (`poke-engine-gen3-residual-rounding.patch`):
//!
//! * The toxic ladder floors ONCE, before the stage multiply:
//!   `clampIntRange(baseMaxhp / 16, 1) * stage`. Upstream floored after
//!   multiplying, which agrees only when max HP is a multiple of 16.
//! * Hail and sandstorm carry Showdown's minimum-1 damage clamp.
//! * Burn and poison were verified correct in the same sweep and are pinned so
//!   they stay that way — gen3 inherits gen6's `baseMaxhp / 8` burn, NOT the
//!   modern `/16`.

use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::{MoveChoice, PokemonVolatileStatus, Weather};
use poke_engine::instruction::Instruction;
use poke_engine::state::{PokemonSideCondition, PokemonStatus, PokemonType, SideReference, State};

fn generate(state: &mut State) -> Vec<Instruction> {
    let before = format!("{:?}", state);
    let instructions =
        generate_instructions_from_move_pair(state, &MoveChoice::None, &MoveChoice::None, false);
    assert_eq!(before, format!("{:?}", state), "state was mutated");
    assert_eq!(
        instructions.len(),
        1,
        "expected a single deterministic branch, got {:?}",
        instructions
    );
    let list = instructions.into_iter().next().unwrap().instruction_list;

    // Every emitted instruction must invert exactly.
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

fn damage_to(list: &[Instruction], side_ref: SideReference) -> i16 {
    let hits: Vec<i16> = list
        .iter()
        .filter_map(|instruction| match instruction {
            Instruction::Damage(damage) if damage.side_ref == side_ref => {
                Some(damage.damage_amount)
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        hits.len(),
        1,
        "expected exactly one residual hit: {:?}",
        list
    );
    hits[0]
}

fn toxic_counter_changes(list: &[Instruction], side_ref: SideReference) -> Vec<i8> {
    list.iter()
        .filter_map(|instruction| match instruction {
            Instruction::ChangeSideCondition(change)
                if change.side_ref == side_ref
                    && change.side_condition == PokemonSideCondition::ToxicCount =>
            {
                Some(change.amount)
            }
            _ => None,
        })
        .collect()
}

/// Side one's active carries `status`, at toxic stage `toxic_count + 1`. Side
/// two is inert, so the only damage in the branch is the residual under test.
fn residual_state(status: PokemonStatus, maxhp: i16, toxic_count: i8) -> State {
    let mut state = State::default();
    state.side_one.side_conditions.toxic_count = toxic_count;
    let active = state.side_one.get_active();
    active.maxhp = maxhp;
    active.hp = maxhp;
    active.status = status;
    state
}

fn residual_damage(status: PokemonStatus, maxhp: i16, toxic_count: i8) -> i16 {
    let mut state = residual_state(status, maxhp, toxic_count);
    let list = generate(&mut state);
    damage_to(&list, SideReference::SideOne)
}

// ---------------------------------------------------------------------------
// Toxic ladder rounding
// ---------------------------------------------------------------------------

/// Showdown gen3 ground truth (`scripts/gen3_switch_differential.py::toxicladder`):
/// a 651 max HP Blissey — 651 % 16 == 11, so the worst-case residue — ticks
/// 611 / 531 / 411 / 251 / 51, i.e. exactly 40, 80, 120, 160, 200. That is
/// `floor(651/16) * stage`, not `floor(651 * stage / 16)`, which would have
/// dealt 40, 81, 122, 162, 203.
#[test]
fn the_toxic_ladder_floors_before_multiplying_by_the_stage() {
    for (stage, expected) in [(1, 40), (2, 80), (3, 120), (4, 160), (5, 200)] {
        assert_eq!(
            residual_damage(PokemonStatus::TOXIC, 651, stage - 1),
            expected,
            "stage {} of a 651 max HP ladder",
            stage
        );
    }
}

/// The identity has to hold for every residue class of max HP mod 16, not just
/// the one the sim fixture happens to use: `floor(maxhp/16) * stage` and
/// `floor(maxhp * stage / 16)` coincide exactly when 16 divides maxhp, and the
/// gap is `floor((maxhp % 16) * stage / 16)` otherwise.
#[test]
fn the_toxic_ladder_is_exact_across_every_residue_class_mod_16() {
    for maxhp in 240..=255 {
        let per_stage = maxhp / 16;
        for stage in 1..=6i8 {
            let expected = per_stage * stage as i16;
            assert_eq!(
                residual_damage(PokemonStatus::TOXIC, maxhp, stage - 1),
                expected,
                "maxhp {} (residue {}) at stage {}",
                maxhp,
                maxhp % 16,
                stage
            );
        }
    }
}

/// `clampIntRange(baseMaxhp / 16, 1)` puts the minimum INSIDE the multiply, so a
/// Pokemon whose max HP is under 16 takes `1 * stage`, not a flat 1. Upstream's
/// `max(floor(maxhp * stage / 16), 1)` collapsed the whole ladder to 1.
#[test]
fn the_toxic_minimum_is_one_per_stage_not_one_total() {
    for stage in 1..=5i8 {
        assert_eq!(
            residual_damage(PokemonStatus::TOXIC, 15, stage - 1),
            stage as i16,
            "a 15 max HP ladder ticks 1 per stage at stage {}",
            stage
        );
    }
    // Shedinja, the pool's only sub-16 max HP Pokemon, dies at stage 1 either
    // way — the clamp is what makes the arithmetic right rather than lucky.
    assert_eq!(residual_damage(PokemonStatus::TOXIC, 1, 0), 1);
}

/// The constructed-world bridge seeds the engine with the pre-tick counter
/// (14 for Showdown's saturated stage 15). Advancing a real residual must keep
/// that counter at 14 so a later search ply cannot invent stage-16 damage.
#[test]
fn toxic_stage_fifteen_stays_capped_across_residual_advances() {
    let mut state = residual_state(PokemonStatus::TOXIC, 640, 14);
    let first = generate(&mut state);
    assert_eq!(damage_to(&first, SideReference::SideOne), 600);
    state.apply_instructions(&first);
    assert_eq!(state.side_one.side_conditions.toxic_count, 14);

    // Restore HP only to observe the second generated multiplier; the counter
    // itself came from the first real end-of-turn instruction list above.
    state.side_one.get_active().hp = 640;
    let second = generate(&mut state);
    assert_eq!(damage_to(&second, SideReference::SideOne), 600);
    assert_eq!(state.side_one.side_conditions.toxic_count, 14);
}

/// Toxic's stored counter is an i8 but only 0 through 14 are valid pre-tick
/// values. Arithmetic is performed after normalizing that stored value: valid
/// counters preserve the exact Showdown ladder, while malformed values fail
/// safe and are repaired by reversible side-condition instructions.
#[test]
fn toxic_counter_arithmetic_is_safe_and_normalizes_every_stored_i8_value() {
    for toxic_count in 0..=14i8 {
        let mut state = residual_state(PokemonStatus::TOXIC, 640, toxic_count);
        let list = generate(&mut state);
        let stage = toxic_count as i16 + 1;
        assert_eq!(damage_to(&list, SideReference::SideOne), 40 * stage);
        assert_eq!(
            toxic_counter_changes(&list, SideReference::SideOne),
            if toxic_count == 14 { vec![] } else { vec![1] },
            "valid counter {} must retain normal increment behavior",
            toxic_count
        );
        state.apply_instructions(&list);
        assert_eq!(
            state.side_one.side_conditions.toxic_count,
            toxic_count.min(13) + 1,
            "valid counter {} must end normalized",
            toxic_count
        );
    }

    for (toxic_count, stage, corrections, final_count) in [
        (-1, 1, vec![2], 1),
        (15, 15, vec![-1], 14),
        (i8::MAX, 15, vec![-113], 14),
        // i8::MIN needs two bounded deltas because ChangeSideCondition.amount
        // is i8 and each instruction must still be reversible on its own.
        (i8::MIN, 1, vec![i8::MAX, 2], 1),
    ] {
        let mut state = residual_state(PokemonStatus::TOXIC, 640, toxic_count);
        let list = generate(&mut state);
        assert_eq!(
            damage_to(&list, SideReference::SideOne),
            40 * stage,
            "stored counter {} must use fail-safe stage {}",
            toxic_count,
            stage
        );
        assert_eq!(
            toxic_counter_changes(&list, SideReference::SideOne),
            corrections,
            "stored counter {} must emit its exact reversible correction",
            toxic_count
        );
        state.apply_instructions(&list);
        assert_eq!(
            state.side_one.side_conditions.toxic_count, final_count,
            "stored counter {} must finish normalized",
            toxic_count
        );
    }
}

/// The tick is still capped by remaining HP, exactly as `Pokemon.damage` caps it.
#[test]
fn the_toxic_tick_cannot_overdraw_remaining_hp() {
    let mut state = residual_state(PokemonStatus::TOXIC, 651, 5);
    state.side_one.get_active().hp = 51;
    let list = generate(&mut state);
    assert_eq!(
        damage_to(&list, SideReference::SideOne),
        51,
        "stage 6 wants 240 but only 51 HP is left: {:?}",
        list
    );
}

// ---------------------------------------------------------------------------
// Burn / poison controls — verified correct, pinned so they stay correct
// ---------------------------------------------------------------------------

/// Gen 3 burn is `baseMaxhp / 8`, NOT the modern `/16`: gen3 inherits gen6's
/// `brn` override (data/mods/gen6/conditions.ts). There is no stage multiplier,
/// so burn cannot express the rounding-order bug — this pins both facts.
#[test]
fn burn_is_an_eighth_of_max_hp_floored() {
    for maxhp in [651, 648, 300, 101] {
        assert_eq!(
            residual_damage(PokemonStatus::BURN, maxhp, 0),
            maxhp / 8,
            "burn on {} max HP",
            maxhp
        );
    }
    assert_eq!(
        residual_damage(PokemonStatus::BURN, 1, 0),
        1,
        "minimum 1 HP"
    );
}

/// Ordinary poison is the base `psn` residual, `baseMaxhp / 8`, unchanged by any
/// gen3-chain override.
#[test]
fn poison_is_an_eighth_of_max_hp_floored() {
    for maxhp in [651, 648, 300, 101] {
        assert_eq!(
            residual_damage(PokemonStatus::POISON, maxhp, 0),
            maxhp / 8,
            "poison on {} max HP",
            maxhp
        );
    }
    assert_eq!(
        residual_damage(PokemonStatus::POISON, 1, 0),
        1,
        "minimum 1 HP"
    );
}

// ---------------------------------------------------------------------------
// Weather chip: the minimum-1 clamp
// ---------------------------------------------------------------------------

fn weather_damage(weather: Weather, maxhp: i16) -> i16 {
    let mut state = State::default();
    state.weather.weather_type = weather;
    state.weather.turns_remaining = 5;
    {
        let active = state.side_one.get_active();
        active.maxhp = maxhp;
        active.hp = maxhp;
    }
    // Side two is immune so the branch carries exactly one residual hit.
    state.side_two.get_active().types = (PokemonType::ROCK, PokemonType::ICE);
    let list = generate(&mut state);
    damage_to(&list, SideReference::SideOne)
}

/// Showdown's `sandstorm.onWeather` / `hail.onWeather` are both
/// `this.damage(target.baseMaxhp / 16)`, and every `damage()` runs through
/// `clampIntRange(damage, 1)` in `spreadDamage`. Upstream truncated
/// `maxhp * 0.0625` with no clamp, so a 1 max HP Shedinja — which IS in the gen3
/// randbats pool — sat in a sandstorm forever taking zero. The real sim kills it.
#[test]
fn weather_chip_never_deals_less_than_one_hp() {
    for weather in [Weather::SAND, Weather::HAIL] {
        assert_eq!(
            weather_damage(weather, 1),
            1,
            "{:?} must still bite a 1 max HP Pokemon",
            weather
        );
        assert_eq!(weather_damage(weather, 15), 1, "{:?} at 15 max HP", weather);
        // ...and is otherwise the ordinary floored sixteenth.
        assert_eq!(
            weather_damage(weather, 651),
            40,
            "{:?} at 651 max HP",
            weather
        );
        assert_eq!(
            weather_damage(weather, 640),
            40,
            "{:?} at 640 max HP",
            weather
        );
    }
}

/// Sanity guard on the fixture itself: the residual block really is running (a
/// silent no-op block would make every assertion above vacuous).
#[test]
fn the_residual_block_runs_at_all() {
    let mut state = residual_state(PokemonStatus::TOXIC, 651, 0);
    let list = generate(&mut state);
    assert!(
        list.iter()
            .any(|instruction| matches!(instruction, Instruction::ChangeSideCondition(_))),
        "the toxic counter must advance alongside the tick: {:?}",
        list
    );
    assert!(
        !state
            .side_one
            .volatile_statuses
            .contains(&PokemonVolatileStatus::NONE),
        "sanity: default state carries no phantom volatile"
    );
}
