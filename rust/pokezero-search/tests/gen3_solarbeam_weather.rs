//! Gen 3 Solar Beam is weakened by rain, sand and hail — and by nothing else.
//! Asserted against the vendored gen3-patched poke-engine.
//!
//! Showdown, `data/moves.ts` `solarbeam.onBasePower` (inherited unchanged by
//! gen3 — no mod in the chain overrides it):
//!
//! ```text
//! onBasePower(basePower, pokemon, target) {
//!     const weakWeathers = ['raindance', 'primordialsea', 'sandstorm', 'hail', 'snowscape'];
//!     if (weakWeathers.includes(pokemon.effectiveWeather())) {
//!         this.debug('weakened by weather');
//!         return this.chainModify(0.5);
//!     }
//! }
//! ```
//!
//! Of those, gen3 has rain, sand and hail. Clear weather is NOT in the list, and
//! sun is handled separately by `solarbeam.onTryMove`, which skips the charge
//! turn rather than touching power.
//!
//! The bug: the engine asked `weather_is_active(&state.weather.weather_type)`.
//! That helper is `weather_type == argument && not suppressed by Air Lock /
//! Cloud Nine`, so feeding it the CURRENT weather reduces to "weather is not
//! suppressed" — which is true in clear weather too, because `NONE == NONE`.
//! Every Solar Beam therefore did half damage. The idiom itself is fine and is
//! used correctly at three other sites (Morning Sun / Moonlight / Synthesis, the
//! Chlorophyll / Swift Swim speed boost, and `update_forecast`), all of which
//! pair it with a `match` on the specific weather so `NONE` falls through.
//!
//! Showdown ground truth for the numbers below (Exeggutor into Blissey, via
//! `scripts/gen3_switch_differential.py`): clear −136, sand −71, sun −131 with
//! no charge turn at all.

use poke_engine::choices::{Choices, MOVES};
use poke_engine::engine::abilities::Abilities;
use poke_engine::engine::choice_effects::modify_choice;
use poke_engine::engine::damage_calc::{calculate_damage, DamageRolls};
use poke_engine::engine::state::{PokemonVolatileStatus, Weather};
use poke_engine::state::{PokemonMoveIndex, SideReference, State};

/// Build the RELEASE turn of a Solar Beam under `weather` and run the same
/// `modify_choice` the move pipeline runs, returning (base power, non-crit,
/// crit, charge-flag-still-set).
fn release(weather: Weather, light_screen: i8, defender_ability: Abilities) -> (f32, i16, i16, bool) {
    let mut state = State::default();
    state.weather.weather_type = weather;
    state.weather.turns_remaining = 5;
    state.side_two.side_conditions.light_screen = light_screen;
    state.side_two.get_active().ability = defender_ability;

    let mut choice = MOVES.get(&Choices::SOLARBEAM).unwrap().clone();
    choice.move_index = PokemonMoveIndex::M0;
    // The charge volatile is already up, i.e. this is the second turn.
    state
        .side_one
        .volatile_statuses
        .insert(PokemonVolatileStatus::SOLARBEAM);

    modify_choice(
        &mut state,
        &mut choice,
        MOVES.get(&Choices::SPLASH).unwrap(),
        &SideReference::SideOne,
    );
    let (damage, crit) = calculate_damage(
        &state,
        &SideReference::SideOne,
        &choice,
        DamageRolls::Max,
    )
    .expect("Solar Beam deals damage");
    (choice.base_power, damage, crit, choice.flags.charge)
}

const FULL_BP: f32 = 120.0;
const WEAKENED_BP: f32 = 60.0;

// ---------------------------------------------------------------------------
// The fix
// ---------------------------------------------------------------------------

/// The regression itself: in CLEAR weather Solar Beam is full power. The engine
/// halved it, so every release dealt half damage — which is what the two-turn
/// release audit measured as "executes now, wrong size".
#[test]
fn clear_weather_does_not_weaken_solar_beam() {
    let (bp, damage, crit, _) = release(Weather::NONE, 0, Abilities::NONE);
    assert_eq!(bp, FULL_BP, "clear weather must not halve Solar Beam");
    assert_eq!(
        (damage, crit),
        (102, 204),
        "full-power release on the default fixture"
    );
}

/// The three weathers Showdown actually lists.
#[test]
fn rain_sand_and_hail_weaken_solar_beam() {
    for weather in [Weather::RAIN, Weather::SAND, Weather::HAIL] {
        let (bp, damage, crit, _) = release(weather, 0, Abilities::NONE);
        assert_eq!(
            bp, WEAKENED_BP,
            "{:?} must halve Solar Beam's base power",
            weather
        );
        assert_eq!(
            (damage, crit),
            (52, 104),
            "weakened release under {:?}",
            weather
        );
    }
}

/// Sun is a different mechanism entirely: `onTryMove` skips the charge turn, and
/// power is untouched. Pinned so a future edit cannot "fix" sun into the
/// weakening list.
#[test]
fn sun_skips_the_charge_turn_at_full_power() {
    let (bp, damage, _, charge) = release(Weather::SUN, 0, Abilities::NONE);
    assert_eq!(bp, FULL_BP, "sun must not change Solar Beam's power");
    assert!(!charge, "sun clears the charge flag so the move fires at once");
    assert_eq!(damage, 102);
}

/// Air Lock and Cloud Nine suppress the weather, so a Solar Beam under a
/// suppressed sandstorm is NOT weakened — `weather_is_active` carries that check,
/// which is the half of the old idiom that was worth keeping.
#[test]
fn suppressed_weather_does_not_weaken_solar_beam() {
    for ability in [Abilities::AIRLOCK, Abilities::CLOUDNINE] {
        let (bp, _, _, _) = release(Weather::SAND, 0, ability);
        assert_eq!(
            bp, FULL_BP,
            "{:?} suppresses the sandstorm, so Solar Beam is full power",
            ability
        );
    }
}

// ---------------------------------------------------------------------------
// The crit ratio: 4x was Light Screen, not a second bug
// ---------------------------------------------------------------------------

/// The audit saw the engine's own branches disagree by ~4x (−41 vs −163) and
/// read it as the non-crit branch halving base power while the crit branch did
/// not. It is not: `calculate_damage` derives both from the SAME `choice`, so a
/// base-power error cannot separate them. The 4x is Light Screen, which gen3
/// crits correctly ignore — the screen halves the non-crit branch only, and 2x
/// crit on top of that is exactly 4x.
///
/// Pinned in both directions so the diagnosis cannot be re-litigated: 2x with no
/// screen, 4x with one, at full power after the fix.
#[test]
fn the_crit_ratio_is_two_without_a_screen_and_four_with_one() {
    let (_, damage, crit, _) = release(Weather::NONE, 0, Abilities::NONE);
    assert_eq!(crit, damage * 2, "no screen: crit is exactly 2x");

    let (_, screened, screened_crit, _) = release(Weather::NONE, 5, Abilities::NONE);
    assert_eq!(
        screened,
        damage / 2,
        "Light Screen halves the non-crit branch"
    );
    assert_eq!(
        screened_crit, crit,
        "and crits ignore the screen entirely, so the crit branch is unchanged"
    );
    assert_eq!(
        screened_crit,
        screened * 4,
        "which is the 4x the release audit observed"
    );
}

/// The shape of the original repro, reduced to its invariant: the fix roughly
/// DOUBLES the release damage. The audit's engine numbers were −41 non-crit /
/// −163 crit against Showdown's −74; doubling the non-crit lands at ~82, and
/// −74 is that at a 0.90 roll.
///
/// "Roughly" is exact, not hand-waving: gen3's formula ends
/// `.../50 + 2`, so halving the base power halves everything EXCEPT that
/// trailing constant. The relationship is therefore `full - 2 == 2 * (weak - 2)`
/// rather than `full == 2 * weak`, and asserting the naive form would fail by
/// exactly the 2.
#[test]
fn the_fix_doubles_release_damage_in_clear_weather() {
    let (full_bp, full, full_crit, _) = release(Weather::NONE, 0, Abilities::NONE);
    let (weak_bp, weak, weak_crit, _) = release(Weather::RAIN, 0, Abilities::NONE);

    assert_eq!((full_bp, weak_bp), (FULL_BP, WEAKENED_BP));
    assert_eq!((full, weak), (102, 52), "the exact before/after numbers");
    assert_eq!(
        full - 2,
        2 * (weak - 2),
        "clear-weather release is twice the buggy halving, net of the +2 term"
    );
    assert_eq!(
        full_crit - 4,
        2 * (weak_crit - 4),
        "and the crit branch doubles too, net of its doubled +2"
    );
}
