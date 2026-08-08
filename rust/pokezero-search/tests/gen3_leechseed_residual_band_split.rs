//! Ledger G8 — the residual-KILL arm is split per roll when the killing residual
//! is Leech Seed. Asserted directly against the vendored gen3-patched poke-engine
//! (`third_party/poke-engine-src/`).
//!
//! Guards `third_party/poke-engine-gen3-leechseed-residual-band-split.patch`.
//!
//! # The defect
//!
//! A residual that KILLS is capped by the HP that happened to be left, so its
//! magnitude inherits the damage roll. Leech Seed then TRANSFERS that capped
//! amount to the other side, where Showdown renders it as a bare silent heal and
//! the fidelity comparator checks it EXACTLY. Over the lethal band the drain
//! `min(maxhp/8, hp_after_move + leftovers)` is INJECTIVE in the roll, so the one
//! arm the partition priced at the band's threshold matched exactly one of the
//! band's rolls and every other roll had no arm at all. Dev row `19000191/63`;
//! `reports/c140_last_dev_row_diagnosis.md` §6a and `reports/c149_*`.
//!
//! # Scope, and why the controls are POISONED rather than clean
//!
//! Only the two `residual_disjoint_bands` call sites whose `ceiling` argument is
//! `i16::MAX` are touched — the non-crit and crit sites reached when the fan
//! CANNOT kill on the hit.
//!
//! Each split fixture is paired with a control that is identical **except that the
//! killing residual is poison instead of Leech Seed**. That pairing is the whole
//! point, and a clean unstatused control would be worthless: with no residual at
//! all there is no lethality threshold, `residual_disjoint_bands` returns `None`,
//! and no band arm is emitted on either build — so an unstatused control passes
//! with the gate deleted. Gen 3 poison ticks `maxhp / 8` and Leech Seed drains
//! `maxhp / 8`, so the two controls below share the fixture's entire arithmetic —
//! same threshold, same band, same roll count — and differ only in which residual
//! does the killing. Deleting `if defender_leech_seeded {` reddens them.
//!
//! # No self-referential pins
//!
//! The expected damage values are the gen 3 roll fan `floor(max * r / 100)`,
//! computed in this file by `integer_fan` rather than read off the engine, and the
//! masses are derived from the crit rate and the band's roll count rather than
//! transcribed from a run.

use poke_engine::choices::Choices;
use poke_engine::engine::items::Items;
use poke_engine::engine::state::{MoveChoice, PokemonVolatileStatus};
use poke_engine::instruction::{Instruction, StateInstructions};
use poke_engine::state::{
    PokemonIndex, PokemonMoveIndex, PokemonStatus, PokemonType, SideReference, State,
};

/// Which residual is going to do the killing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Killer {
    /// Leech Seed: drains `maxhp / 8` and TRANSFERS it. In scope.
    LeechSeed,
    /// Poison: ticks `maxhp / 8` and transfers nothing. The control.
    Poison,
}

/// `generate_instructions_from_move_pair` must leave `state` untouched.
fn generate(state: &mut State) -> Vec<StateInstructions> {
    let before = format!("{:?}", state);
    let branches =
        poke_engine::engine::generate_instructions::generate_instructions_from_move_pair(
            state,
            &MoveChoice::Move(PokemonMoveIndex::M0),
            &MoveChoice::None,
            true,
        );
    assert_eq!(before, format!("{:?}", state), "state was mutated");
    branches
}

/// The damage the attacker's move deals in a branch: the FIRST `Damage` landing on
/// side two. Residual ticks come later in the list.
fn move_damage(list: &[Instruction]) -> i16 {
    list.iter()
        .find_map(|instruction| match instruction {
            Instruction::Damage(damage) if damage.side_ref == SideReference::SideTwo => {
                Some(damage.damage_amount)
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("branch deals no damage to side two: {:?}", list))
}

fn side_two_fainted(state: &State, list: &Vec<Instruction>) -> bool {
    let mut probe = state.clone();
    probe.apply_instructions(list);
    let fainted = probe.side_two.get_active().hp == 0;
    probe.reverse_instructions(list);
    fainted
}

/// `(move damage, mass)` for every branch where side two survives the HIT and then
/// dies — i.e. exactly the residual-kill arms. A hit-KO arm (the crit one, in
/// fixture A) deals at least the defender's HP and is excluded by the `< pre_hp`
/// test rather than by naming it.
fn residual_kill_arms(state: &mut State) -> Vec<(i16, f32)> {
    let pre_hp = state.side_two.get_active().hp;
    let branches = generate(state);
    let mut arms: Vec<(i16, f32)> = branches
        .iter()
        .filter(|branch| branch.percentage > 0.0)
        .filter(|branch| {
            move_damage(&branch.instruction_list) < pre_hp
                && side_two_fainted(state, &branch.instruction_list)
        })
        .map(|branch| (move_damage(&branch.instruction_list), branch.percentage))
        .collect();
    arms.sort_by_key(|(damage, _)| *damage);
    arms
}

fn total_mass(state: &mut State) -> f32 {
    generate(state).iter().map(|branch| branch.percentage).sum()
}

/// The 16 gen 3 damage rolls, `floor(max * r / 100)` for `r` in `85..=100`.
///
/// Written out here rather than imported from the engine: the assertion is that
/// the split arms land on rolls SHOWDOWN can throw, and reusing the engine's own
/// helper would make that circular.
fn integer_fan(max_damage: i32) -> Vec<i16> {
    (85..=100).map(|r| (max_damage * r / 100) as i16).collect()
}

/// The fan members at or above `threshold`, duplicates kept: those are the rolls
/// of the band, and two rolls landing on the same integer are two rolls.
fn band_rolls(max_damage: i32, threshold: i16) -> Vec<i16> {
    integer_fan(max_damage)
        .into_iter()
        .filter(|roll| *roll >= threshold)
        .collect()
}

fn bare_state() -> State {
    let mut state = State::default();
    for side_ref in [SideReference::SideOne, SideReference::SideTwo] {
        let side = state.get_side(&side_ref);
        let active = side.get_active();
        active.level = 100;
        active.types = (PokemonType::NORMAL, PokemonType::TYPELESS);
        active.item = Items::NONE;
        active.replace_move(PokemonMoveIndex::M0, Choices::SPLASH);
    }
    // Reserves alive on both sides, so no faint can end the battle and truncate
    // the residual block these fixtures are reading.
    for index in [
        PokemonIndex::P1,
        PokemonIndex::P2,
        PokemonIndex::P3,
        PokemonIndex::P4,
        PokemonIndex::P5,
    ] {
        state.side_one.pokemon[index].hp = 100;
        state.side_two.pokemon[index].hp = 100;
    }
    state
}

fn fixture(attack: i16, defense: i16, maxhp: i16, hp: i16, killer: Killer) -> State {
    let mut state = bare_state();
    {
        let attacker = state.side_one.get_active();
        attacker.attack = attack;
        attacker.speed = 300;
        // Return: 102 BP, physical, NO secondary and no recoil, so the branch set
        // is the roll partition and nothing else.
        attacker.replace_move(PokemonMoveIndex::M0, Choices::RETURN);
    }
    {
        let defender = state.side_two.get_active();
        defender.defense = defense;
        defender.speed = 100;
        defender.maxhp = maxhp;
        defender.hp = hp;
    }
    match killer {
        Killer::LeechSeed => {
            state
                .side_two
                .volatile_statuses
                .insert(PokemonVolatileStatus::LEECHSEED);
        }
        Killer::Poison => {
            state.side_two.get_active().status = PokemonStatus::POISON;
        }
    }
    state
}

// ---------------------------------------------------------------------------
// Fixture A — the NON-CRIT `i16::MAX`-ceiling site.
//
// Return at attack 200 into defense 150 has a 174-max fan, and 174 < 200 HP, so
// no non-crit roll can kill on the hit. `maxhp / 8` is 40, so the residual
// lethality threshold is `200 - 40 = 160`, which sits strictly inside the fan
// (floor 147, max 174). That placement is what makes the band non-degenerate:
// below the floor every roll would be lethal and the threshold is skipped;
// above the max none would be and the partition never fires.
//
// The crit fan (348 max, 295 floor) clears 200 HP on every roll, so the crit
// straddle block is not entered and the crit arm is a plain hit-KO. That keeps
// this fixture pointed at ONE call site.
// ---------------------------------------------------------------------------

const A_ATTACK: i16 = 200;
const A_DEFENSE: i16 = 150;
const A_MAXHP: i16 = 320;
const A_HP: i16 = 200;
const A_MAX_DAMAGE: i32 = 174;
const A_THRESHOLD: i16 = 160;

// ---------------------------------------------------------------------------
// Fixture B — the CRIT `i16::MAX`-ceiling site.
//
// Return at attack 60 into defense 250 has a 33-max non-crit fan and a 66-max
// crit fan, and 66 < 76 HP, so not even a max crit kills on the hit: the crit
// straddle block is skipped and the sibling `else` — the second `i16::MAX` site —
// runs. `maxhp / 8` is 15, so the threshold is `76 - 15 = 61`, inside the crit fan
// (floor 56, max 66) and ABOVE the whole non-crit fan, so the non-crit site cannot
// fire and this fixture points at the other call site alone.
//
// This fan also has COLLIDING rolls — `66*92/100` and `66*95/100` both floor to 62
// — which exercises the merge half: the split emits sixteen slots and
// `combine_duplicate_instructions` folds the collisions back together, so a
// collided damage value must come back at DOUBLE mass rather than as two arms.
// ---------------------------------------------------------------------------

const B_ATTACK: i16 = 60;
const B_DEFENSE: i16 = 250;
const B_MAXHP: i16 = 120;
const B_HP: i16 = 76;
const B_MAX_CRIT_DAMAGE: i32 = 66;
const B_THRESHOLD: i16 = 61;

/// 1/16 crit rate, so a non-crit roll carries `(15/16) * (1/16)` of the boundary.
const NON_CRIT_ROLL_MASS: f32 = 100.0 * (15.0 / 16.0) / 16.0;
/// ...and a crit roll carries `(1/16) * (1/16)`.
const CRIT_ROLL_MASS: f32 = 100.0 * (1.0 / 16.0) / 16.0;

fn assert_close(actual: f32, expected: f32, what: &str) {
    assert!(
        (actual - expected).abs() < 1e-3,
        "{}: expected {:.6}, got {:.6}",
        what,
        expected,
        actual
    );
}

// ------------------------------------------------------------------ the split

/// THE headline. One arm per roll of the band, at the roll's own damage.
#[test]
fn a_seeded_residual_kill_band_emits_one_arm_per_roll() {
    let mut state = fixture(A_ATTACK, A_DEFENSE, A_MAXHP, A_HP, Killer::LeechSeed);
    let arms = residual_kill_arms(&mut state);

    let expected = band_rolls(A_MAX_DAMAGE, A_THRESHOLD);
    assert_eq!(expected.len(), 9, "fixture sanity: the band is nine rolls");
    assert_eq!(
        arms.iter().map(|(damage, _)| *damage).collect::<Vec<_>>(),
        expected,
        "the residual-kill arms must be exactly the band's rolls, not one arm at \
         the threshold: {:?}",
        arms
    );
}

/// Every split arm is a damage amount Showdown can actually deal. This is the
/// property that makes the split safe rather than a re-pricing, and it is why the
/// integer fan is used and not `compare_health_with_damage_multiples`'s f32
/// accumulator, whose drift would put arms at unreachable values.
///
/// NOT claimed as the sole killer of anything, and the reason is worth recording:
/// fixture A's two fans agree, so swapping the integer fan for the f32 accumulator
/// leaves this green. What kills that mutant is the source-text pin on the fan
/// expression in `tests/test_poke_engine_patch_stack.py`. This test is the
/// behavioural statement of the property, and it does redden on a full revert
/// through the split-sanity assertion below.
#[test]
fn every_split_arm_lands_on_a_roll_showdown_can_throw() {
    let mut state = fixture(A_ATTACK, A_DEFENSE, A_MAXHP, A_HP, Killer::LeechSeed);
    let fan = integer_fan(A_MAX_DAMAGE);
    let arms = residual_kill_arms(&mut state);
    assert!(arms.len() > 1, "fixture sanity: the band split");
    for (damage, _) in arms {
        assert!(
            fan.contains(&damage),
            "arm at {} is not a member of the fan {:?}",
            damage,
            fan
        );
    }
}

/// THE GATE. Same HP, same maxhp, same threshold, same band — poison instead of
/// Leech Seed — and the collapsed single arm at the threshold must survive
/// untouched. Deleting `if defender_leech_seeded {` reddens this.
#[test]
fn a_poisoned_defender_with_the_same_band_keeps_the_single_collapsed_arm() {
    let mut state = fixture(A_ATTACK, A_DEFENSE, A_MAXHP, A_HP, Killer::Poison);
    let arms = residual_kill_arms(&mut state);
    assert_eq!(
        arms.iter().map(|(damage, _)| *damage).collect::<Vec<_>>(),
        vec![A_THRESHOLD],
        "an unsplit site must emit ONE arm, priced at the threshold: {:?}",
        arms
    );
}

/// The split moves no mass: the band's total is what the single arm carried, and
/// each new arm is exactly one roll of the fan.
#[test]
fn the_split_conserves_the_bands_mass_and_prices_each_arm_at_one_sixteenth() {
    let mut seeded = fixture(A_ATTACK, A_DEFENSE, A_MAXHP, A_HP, Killer::LeechSeed);
    let mut poisoned = fixture(A_ATTACK, A_DEFENSE, A_MAXHP, A_HP, Killer::Poison);

    let split = residual_kill_arms(&mut seeded);
    let collapsed = residual_kill_arms(&mut poisoned);
    assert!(split.len() > collapsed.len(), "fixture sanity: it did split");

    for (damage, mass) in &split {
        assert_close(*mass, NON_CRIT_ROLL_MASS, &format!("arm at {}", damage));
    }
    assert_close(
        split.iter().map(|(_, mass)| *mass).sum::<f32>(),
        collapsed.iter().map(|(_, mass)| *mass).sum::<f32>(),
        "split band total vs collapsed arm",
    );
}

// ------------------------------------------------------------- the crit site

/// The second `i16::MAX`-ceiling site, and the collision-merge behaviour with it.
#[test]
fn the_crit_fan_site_splits_too_with_colliding_rolls_merged() {
    let mut state = fixture(B_ATTACK, B_DEFENSE, B_MAXHP, B_HP, Killer::LeechSeed);
    let arms = residual_kill_arms(&mut state);

    let rolls = band_rolls(B_MAX_CRIT_DAMAGE, B_THRESHOLD);
    assert_eq!(rolls.len(), 8, "fixture sanity: the crit band is eight rolls");

    let mut distinct: Vec<i16> = rolls.clone();
    distinct.dedup();
    assert_eq!(
        distinct.len(),
        6,
        "fixture sanity: two pairs of rolls collide on the same integer"
    );
    assert_eq!(
        arms.iter().map(|(damage, _)| *damage).collect::<Vec<_>>(),
        distinct,
        "the crit residual-kill arms must be the band's DISTINCT rolls: {:?}",
        arms
    );

    // A collided value carries two rolls' mass, not one, and not as two arms.
    for (damage, mass) in &arms {
        let multiplicity = rolls.iter().filter(|roll| *roll == damage).count() as f32;
        assert_close(
            *mass,
            CRIT_ROLL_MASS * multiplicity,
            &format!("crit arm at {} ({} rolls)", damage, multiplicity),
        );
    }
}

/// The gate again, on the crit site.
#[test]
fn a_poisoned_defender_on_the_crit_fan_keeps_its_single_arm() {
    let mut state = fixture(B_ATTACK, B_DEFENSE, B_MAXHP, B_HP, Killer::Poison);
    let arms = residual_kill_arms(&mut state);
    assert_eq!(
        arms.iter().map(|(damage, _)| *damage).collect::<Vec<_>>(),
        vec![B_THRESHOLD],
        "the crit site must emit ONE arm at the threshold when the killer does not \
         transfer: {:?}",
        arms
    );
}

// ----------------------------------------------------------- whole-fan health

/// Branch masses still sum to 100 % on every fixture. A partition that emits `n`
/// arms of `1/16` where it used to emit one of `n/16` is exactly the shape that
/// silently loses or duplicates mass, and `update_percentage` has no conservation
/// check of its own.
#[test]
fn every_fixture_still_sums_to_one_hundred_percent() {
    for (label, killer, attack, defense, maxhp, hp) in [
        ("A seeded", Killer::LeechSeed, A_ATTACK, A_DEFENSE, A_MAXHP, A_HP),
        ("A poisoned", Killer::Poison, A_ATTACK, A_DEFENSE, A_MAXHP, A_HP),
        ("B seeded", Killer::LeechSeed, B_ATTACK, B_DEFENSE, B_MAXHP, B_HP),
        ("B poisoned", Killer::Poison, B_ATTACK, B_DEFENSE, B_MAXHP, B_HP),
    ] {
        let mut state = fixture(attack, defense, maxhp, hp, killer);
        assert_close(total_mass(&mut state), 100.0, label);
    }
}

/// Every emitted instruction must invert exactly, on the split arms too: search
/// folds these branches onto a live `State`, and a non-inverting arm corrupts the
/// tree rather than failing.
#[test]
fn every_split_arm_reverts_cleanly() {
    let mut state = fixture(A_ATTACK, A_DEFENSE, A_MAXHP, A_HP, Killer::LeechSeed);
    let branches = generate(&mut state);
    assert!(branches.len() > 3, "fixture sanity: the fixture did split");
    for branch in &branches {
        let before = format!("{:?}", state);
        state.apply_instructions(&branch.instruction_list);
        state.reverse_instructions(&branch.instruction_list);
        assert_eq!(before, format!("{:?}", state), "arm did not revert");
    }
}

// ---------------------------------------------------------------------------
// Fixture C — THE DECLINE PATH. A band where the two fan bases DISAGREE.
//
// Everything above exercises the split. Nothing above reaches the count guard's
// negative arm, because fixtures A and B are both on fans where the f32
// accumulator in `compare_health_with_damage_multiples` and the exact integer fan
// `floor(max * r / 100)` agree about how many rolls sit in the band — so all eight
// of the tests above stay GREEN with the guard deleted, and so does the whole
// crate suite. This fixture closes that hole.
//
// Return at attack 66 into defense 300 into a 47 HP / 160 maxhp Leech-Seeded
// defender: `maxhp / 8` is 20, so the threshold is `47 - 20 = 27`, and on this fan
// the comparator counts a different number of rolls at or above 27 than the
// integer fan contains. The guard therefore DECLINES, and the site keeps the
// single collapsed arm at the threshold that it emits today.
//
// The mass assertion is the load-bearing half. With the guard deleted the site
// emits one arm per integer-fan member of the window while the survive arm is
// still discounted by the comparator's count, so the branch masses sum to
// 105.859375 % — mass conjured out of a basis mismatch, which is exactly the
// failure `update_percentage` cannot see and which no other test in the repo
// would catch.
// ---------------------------------------------------------------------------

const C_ATTACK: i16 = 66;
const C_DEFENSE: i16 = 300;
const C_MAXHP: i16 = 160;
const C_HP: i16 = 47;
const C_THRESHOLD: i16 = 27;

/// THE COUNT GUARD. When the comparator's band count and the integer fan disagree,
/// the split declines and today's single arm survives — so a band that would lose
/// its matching arm keeps it, which is the whole "strict improvement or no-op,
/// never a trade" claim.
#[test]
fn a_band_whose_fan_bases_disagree_declines_the_split_and_keeps_one_arm() {
    let mut state = fixture(C_ATTACK, C_DEFENSE, C_MAXHP, C_HP, Killer::LeechSeed);
    let arms = residual_kill_arms(&mut state);
    assert_eq!(
        arms.iter().map(|(damage, _)| *damage).collect::<Vec<_>>(),
        vec![C_THRESHOLD],
        "the guard must decline and leave ONE arm at the threshold: {:?}",
        arms
    );
}

/// ...and declining is what keeps the masses summing to 100 %. Splitting a band on
/// a basis the mass was not priced from conjures mass: the guard-deleted engine
/// reads 105.859375 % here.
#[test]
fn declining_the_split_keeps_the_branch_mass_at_one_hundred_percent() {
    let mut state = fixture(C_ATTACK, C_DEFENSE, C_MAXHP, C_HP, Killer::LeechSeed);
    assert_close(total_mass(&mut state), 100.0, "C seeded (decline path)");
}
