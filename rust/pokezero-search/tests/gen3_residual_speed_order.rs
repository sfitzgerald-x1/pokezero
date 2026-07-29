//! Gen 3 end-of-turn residual ORDERING, pinned against the vendored gen3-patched
//! poke-engine (`third_party/poke-engine-src/`).
//!
//! Guards `third_party/poke-engine-gen3-residual-speed-order.patch`.
//!
//! # What the sim actually does
//!
//! `Battle.fieldEvent('Residual')` builds ONE handler list and calls
//! `this.speedSort(handlers)` (`sim/battle.ts:507`). The comparator is
//! `Battle.comparePriority` (`sim/battle.ts:404-411`):
//!
//! ```text
//! order ASC -> priority DESC -> SPEED DESC -> subOrder ASC -> effectOrder ASC
//! ```
//!
//! so within an order class the FASTER Pokemon resolves its ENTIRE residual set
//! before the slower one resolves any. gen3 keeps the pre-gen5 numbering, in which
//! abilities (10.3), items (10.4), Leech Seed (10.5), status damage (10.6),
//! partial trap (10.9) and the Encore/Taunt/Yawn ticks are ALL one class — so the
//! whole mid-block is speed-major and subOrder only orders within a single mon.
//! Resolved through `Dex.mod('gen3')` (`scripts/gen3_dex_resolve.py`'s rule),
//! which reaches `data/mods/gen3/items.ts` and `data/mods/gen4/abilities.ts`; the
//! 5.x values in `data/{items,abilities}.ts` are gen5+ and do not apply.
//!
//! # NO SELF-REFERENTIAL PINS
//!
//! Every expectation below is TRANSCRIBED from a run of the real Node simulator,
//! never read off this engine. Each test cites the protocol it came from, and each
//! has a matching Showdown-side scenario in `scripts/gen3_switch_differential.py`
//! (`residualspeedmajor`, `residualspeedmajorfast`, `residualspeedpara`,
//! `residualspeedparacontrol`, `residualspeedtie`, `residualspeedsand`,
//! `residualspeedleech`, `residualsuborder`) which re-derives it from the sim on
//! every run. The pin this file replaced asserted the engine's own wrong ordering
//! and stayed green for three cycles.
//!
//! Every fixture here is DUAL-SIDE. A single-mon fixture cannot observe this bug:
//! the interleaving between the two sides IS the bug.

use poke_engine::engine::abilities::Abilities;
use poke_engine::engine::items::Items;
use poke_engine::engine::state::{MoveChoice, PokemonVolatileStatus, Weather};
use poke_engine::instruction::{Instruction, StateInstructions};
use poke_engine::state::{
    PokemonIndex, PokemonMoveIndex, PokemonStatus, PokemonType, SideReference, State,
};

/// `generate_instructions_from_move_pair` must leave `state` untouched.
fn generate(state: &mut State) -> Vec<StateInstructions> {
    let before = format!("{:?}", state);
    let instructions = poke_engine::engine::generate_instructions::generate_instructions_from_move_pair(
        state,
        &MoveChoice::None,
        &MoveChoice::None,
        false,
    );
    assert_eq!(before, format!("{:?}", state), "state was mutated");
    instructions
}

fn only_branch(instructions: Vec<StateInstructions>) -> Vec<Instruction> {
    assert_eq!(
        instructions.len(),
        1,
        "expected a single deterministic branch, got {:?}",
        instructions
    );
    instructions.into_iter().next().unwrap().instruction_list
}

/// Apply then reverse: search relies on every emitted instruction being an exact
/// inverse, so a restructured block that forgets one corrupts the tree.
fn assert_reverts_cleanly(state: &mut State, list: &Vec<Instruction>) {
    let before = format!("{:?}", state);
    state.apply_instructions(list);
    state.reverse_instructions(list);
    assert_eq!(before, format!("{:?}", state), "instructions did not revert");
}

fn index_of<F>(list: &[Instruction], predicate: F) -> usize
where
    F: Fn(&Instruction) -> bool,
{
    list.iter()
        .position(predicate)
        .unwrap_or_else(|| panic!("expected instruction not emitted: {:?}", list))
}

fn damage_at(list: &[Instruction], side: SideReference) -> usize {
    index_of(list, |i| {
        matches!(i, Instruction::Damage(d) if d.side_ref == side)
    })
}

fn heal_at(list: &[Instruction], side: SideReference) -> usize {
    index_of(list, |i| {
        matches!(i, Instruction::Heal(h) if h.side_ref == side)
    })
}

fn damages(list: &[Instruction], side: SideReference) -> Vec<i16> {
    list.iter()
        .filter_map(|i| match i {
            Instruction::Damage(d) if d.side_ref == side => Some(d.damage_amount),
            _ => None,
        })
        .collect()
}

fn heals(list: &[Instruction], side: SideReference) -> Vec<i16> {
    list.iter()
        .filter_map(|i| match i {
            Instruction::Heal(h) if h.side_ref == side => Some(h.heal_amount),
            _ => None,
        })
        .collect()
}

/// The distinct heal shapes a branch set produces for one side, deduplicated —
/// the thing a residual speed-tie fork is supposed to (or not to) multiply.
fn distinct_heal_shapes(branches: &[StateInstructions], side: SideReference) -> Vec<Vec<i16>> {
    let mut out: Vec<Vec<i16>> = Vec::new();
    for branch in branches {
        let shape = heals(&branch.instruction_list, side);
        if !out.contains(&shape) {
            out.push(shape);
        }
    }
    out
}

/// Whatever the branching, the probability mass must still add to one.
fn assert_mass_is_whole(branches: &[StateInstructions]) {
    let total: f32 = branches.iter().map(|branch| branch.percentage).sum();
    assert!(
        (total - 100.0).abs() < 1e-2,
        "branch percentages must sum to 100, got {} from {:?}",
        total,
        branches
    );
}

/// Two Pokemon with nothing but the residuals under test, and a move phase that
/// does nothing (`MoveChoice::None` both sides) so the whole instruction list IS
/// the residual block. Types are forced NORMAL/TYPELESS so no weather immunity
/// can quietly remove an entry.
fn bare_state(side_one_speed: i16, side_two_speed: i16) -> State {
    let mut state = State::default();
    for side in [SideReference::SideOne, SideReference::SideTwo] {
        let speed = match side {
            SideReference::SideOne => side_one_speed,
            SideReference::SideTwo => side_two_speed,
        };
        let s = state.get_side(&side);
        let active = s.get_active();
        active.maxhp = 320;
        active.hp = 200;
        active.speed = speed;
        active.types = (PokemonType::NORMAL, PokemonType::TYPELESS);
        active.replace_move(
            PokemonMoveIndex::M0,
            poke_engine::choices::Choices::SPLASH,
        );
    }
    // Reserves alive on both sides, so no residual faint can end the battle and
    // truncate a sequence these tests are trying to read.
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

// --------------------------------------------------------------- the headline

/// THE demonstrated divergence.
///
/// Showdown, transcribed (`residualspeedmajor`; slow Leftovers+burn Snorlax vs
/// faster toxic'd Aipom):
///
/// ```text
/// |-damage|p2a: Aipom|206/251 tox|[from] psn            <- spe 206, 10.6
/// |-heal|p1a: Snorlax|432/461 brn|[from] item: Leftovers <- spe 96,  10.4
/// |-damage|p1a: Snorlax|375/461 brn|[from] brn           <- spe 96,  10.6
/// ```
///
/// The faster mon's status tick precedes the slower mon's Leftovers heal even
/// though Leftovers has the LOWER subOrder. A section-major engine emits the heal
/// first in every case; that is the bug.
#[test]
fn the_faster_mon_resolves_its_whole_set_before_the_slower_one() {
    let mut state = bare_state(96, 206);
    state.side_one.get_active().item = Items::LEFTOVERS;
    state.side_one.get_active().status = PokemonStatus::BURN;
    state.side_two.get_active().status = PokemonStatus::POISON;

    let list = only_branch(generate(&mut state));

    let fast_psn = damage_at(&list, SideReference::SideTwo);
    let slow_heal = heal_at(&list, SideReference::SideOne);
    let slow_brn = damage_at(&list, SideReference::SideOne);
    assert!(
        fast_psn < slow_heal,
        "the faster mon's 10.6 poison tick precedes the slower mon's 10.4 heal: {:?}",
        list
    );
    assert!(
        slow_heal < slow_brn,
        "within the slower mon, subOrder still orders 10.4 before 10.6: {:?}",
        list
    );
    assert_reverts_cleanly(&mut state, &list);
}

/// The mirror (`residualspeedmajorfast`). Same three entries, opposite grouping:
///
/// ```text
/// |-heal|p2a: Aipom|235/251 brn|[from] item: Leftovers   <- spe 206, 10.4
/// |-damage|p2a: Aipom|204/251 brn|[from] brn             <- spe 206, 10.6
/// |-damage|p1a: Snorlax|277/461 tox|[from] psn           <- spe 96,  10.6
/// ```
///
/// Together with the test above this pins that the order tracks SPEED — not the
/// seat, not the section, not the subOrder.
#[test]
fn swapping_which_mon_is_faster_swaps_the_grouping() {
    let mut state = bare_state(96, 206);
    state.side_one.get_active().status = PokemonStatus::POISON;
    state.side_two.get_active().item = Items::LEFTOVERS;
    state.side_two.get_active().status = PokemonStatus::BURN;

    let list = only_branch(generate(&mut state));

    let fast_heal = heal_at(&list, SideReference::SideTwo);
    let fast_brn = damage_at(&list, SideReference::SideTwo);
    let slow_psn = damage_at(&list, SideReference::SideOne);
    assert!(
        fast_heal < fast_brn && fast_brn < slow_psn,
        "the faster mon runs its whole set (10.4 then 10.6) before the slower mon \
         runs any: {:?}",
        list
    );
    assert_reverts_cleanly(&mut state, &list);
}

// ------------------------------------------------------------- which speed (Q1)

/// Paralysis is in the stamped speed.
///
/// `resolvePriority` stamps `handler.speed = pokemon.speed` (`sim/battle.ts:1003`),
/// refreshed by `updateSpeed()` immediately before the residual phase
/// (`sim/battle.ts:2838`) — `getStat('spe', false, false)`, so every `ModifySpe`
/// applies, and gen3 paralysis is `chainModify(0.25)` (inherited from
/// `data/mods/gen4/conditions.ts`).
///
/// Showdown, transcribed (`residualspeedparacontrol` then `residualspeedpara`;
/// 126-speed Ampharos badly poisoned vs 206-speed Aipom holding Leftovers):
///
/// ```text
/// control  |-heal|p2a: Aipom|181/251|[from] item: Leftovers
///          |-damage|p1a: Ampharos|201/321 tox|[from] psn
///
/// paralysed|-damage|p1a: Ampharos|201/321 tox|[from] psn
///          |-heal|p2a: Aipom|181/251 par|[from] item: Leftovers
/// ```
///
/// Decisive because subOrder points the other way (10.4 before 10.6): only the
/// paralysed arm flips, so the flip can only be the speed term.
#[test]
fn paralysis_quarters_the_speed_the_residual_sort_reads() {
    for paralysed in [false, true] {
        let mut state = bare_state(126, 206);
        state.side_one.get_active().status = PokemonStatus::POISON;
        state.side_two.get_active().item = Items::LEFTOVERS;
        if paralysed {
            state.side_two.get_active().status = PokemonStatus::PARALYZE;
        }

        let list = only_branch(generate(&mut state));
        let psn = damage_at(&list, SideReference::SideOne);
        let heal = heal_at(&list, SideReference::SideTwo);
        if paralysed {
            assert!(
                psn < heal,
                "206 * 0.25 = 51 < 126, so the poisoned mon goes first: {:?}",
                list
            );
        } else {
            assert!(
                heal < psn,
                "206 > 126, so the Leftovers holder goes first: {:?}",
                list
            );
        }
        assert_reverts_cleanly(&mut state, &list);
    }
}

/// Boosts are in it too — the sort reads the speed as of the START of the residual
/// phase, after the move phase has resolved. Showdown ground truth: an Ampharos
/// (126) that uses Agility on the measured turn overtakes a 206-speed Aipom in the
/// SAME turn's residual order, and stays ahead afterwards, while the no-Agility
/// control keeps Aipom first throughout.
#[test]
fn speed_boosts_are_in_the_speed_the_residual_sort_reads() {
    for boosted in [false, true] {
        let mut state = bare_state(126, 206);
        state.side_one.get_active().status = PokemonStatus::POISON;
        state.side_two.get_active().item = Items::LEFTOVERS;
        if boosted {
            // +2 doubles: 126 -> 252 > 206.
            state.side_one.speed_boost = 2;
        }

        let list = only_branch(generate(&mut state));
        let psn = damage_at(&list, SideReference::SideOne);
        let heal = heal_at(&list, SideReference::SideTwo);
        assert_eq!(
            psn < heal,
            boosted,
            "boosted={} should decide who resolves first: {:?}",
            boosted,
            list
        );
        assert_reverts_cleanly(&mut state, &list);
    }
}

// --------------------------------------------------------------- ties (Q2)

/// An exact speed tie is RANDOM in Showdown — `speedSort` finishes it with
/// `this.prng.shuffle` (`sim/battle.ts:455-457`). Ground truth
/// (`residualspeedtie`): two identical 146-speed Blisseys, both badly poisoned,
/// swap residual order between seeds AND between turns of a single battle, so no
/// deterministic pick is correct.
///
/// The engine therefore BRANCHES — but only when the two orders actually reach
/// different states. Here they do. Side one is 30 HP from full and seeds side two:
///
///   * seeder first — Leftovers pays 20 (hp 310), then the sap's drain is capped
///     to the remaining 10;
///   * victim first — the sap's drain pays 30 outright (hp 320), and the Leftovers
///     heal then has nothing to do and is not emitted at all.
///
/// Note that an exact residual tie is also an exact MOVE-phase tie, and the engine
/// already forks that; asserting on the number of DISTINCT instruction lists rather
/// than on the branch count keeps this test about the residual fork alone.
#[test]
fn an_exact_speed_tie_forks_when_the_two_orders_differ() {
    let mut state = bare_state(146, 146);
    state.side_one.get_active().hp = 290;
    state.side_one.get_active().item = Items::LEFTOVERS;
    state
        .side_two
        .volatile_statuses
        .insert(PokemonVolatileStatus::LEECHSEED);

    let branches = generate(&mut state);
    let outcomes = distinct_heal_shapes(&branches, SideReference::SideOne);
    assert_eq!(
        outcomes.len(),
        2,
        "a tie whose orders differ must fork, not pick a side: {:?}",
        branches
    );
    assert!(
        outcomes.contains(&vec![20, 10]) && outcomes.contains(&vec![30]),
        "both orders must be offered, with their own amounts: {:?}",
        branches
    );
    assert_mass_is_whole(&branches);
    for branch in branches {
        assert_reverts_cleanly(&mut state, &branch.instruction_list);
    }
}

/// The other half of the tie contract: when the two orders reach the SAME state,
/// the fork is collapsed. Two independent Leftovers heals commute, so forking them
/// would hand search two identical children carrying half the mass each — a
/// spurious stochastic node in every mirror matchup.
#[test]
fn an_exact_speed_tie_collapses_when_the_two_orders_agree() {
    let mut state = bare_state(146, 146);
    state.side_one.get_active().item = Items::LEFTOVERS;
    state.side_two.get_active().item = Items::LEFTOVERS;

    let branches = generate(&mut state);
    let mut distinct: Vec<Vec<Instruction>> = Vec::new();
    for branch in &branches {
        if !distinct.contains(&branch.instruction_list) {
            distinct.push(branch.instruction_list.clone());
        }
    }
    assert_eq!(
        distinct.len(),
        1,
        "commuting residuals must not fork: {:?}",
        branches
    );
    assert_mass_is_whole(&branches);
}

// ----------------------------------------------------- subOrder within one mon

/// The full within-mon ladder, opposite a slower Leftovers holder.
///
/// Showdown, transcribed (`residualsuborder`; a burned Ludicolo with Rain Dish and
/// Leftovers in its own rain, spe 176, vs a 96-speed Snorlax with Leftovers):
///
/// ```text
/// |-weather|RainDance|[upkeep]
/// |-heal|p2a: Ludicolo|181/301 brn|[from] ability: Rain Dish   <- 10.3
/// |-heal|p2a: Ludicolo|199/301 brn|[from] item: Leftovers      <- 10.4
/// |-damage|p2a: Ludicolo|162/301 brn|[from] brn                <- 10.6
/// |-heal|p1a: Snorlax|417/461|[from] item: Leftovers           <- slow mon, 10.4
/// ```
///
/// This is the dual-side coverage the original `poke-engine-gen3-residual-order`
/// scenarios never had. That patch believed Leftovers was "order 5" and left Rain
/// Dish and Speed Boost behind the status tick; gen3 has no order-5 bucket at all.
#[test]
fn abilities_then_items_then_status_within_a_mon_then_the_slower_mon() {
    let mut state = bare_state(96, 176);
    state.weather.weather_type = Weather::RAIN;
    state.weather.turns_remaining = 5;
    state.side_one.get_active().item = Items::LEFTOVERS;
    {
        let fast = state.side_two.get_active();
        fast.item = Items::LEFTOVERS;
        fast.ability = Abilities::RAINDISH;
        fast.status = PokemonStatus::BURN;
    }

    let list = only_branch(generate(&mut state));

    let fast_heals: Vec<usize> = list
        .iter()
        .enumerate()
        .filter(|(_, i)| matches!(i, Instruction::Heal(h) if h.side_ref == SideReference::SideTwo))
        .map(|(index, _)| index)
        .collect();
    assert_eq!(
        fast_heals.len(),
        2,
        "Rain Dish (10.3) and Leftovers (10.4) both fire: {:?}",
        list
    );
    let fast_brn = damage_at(&list, SideReference::SideTwo);
    let slow_heal = heal_at(&list, SideReference::SideOne);
    assert!(
        fast_heals[0] < fast_heals[1] && fast_heals[1] < fast_brn && fast_brn < slow_heal,
        "10.3 -> 10.4 -> 10.6 within the fast mon, then the slow mon's 10.4: {:?}",
        list
    );
    assert_reverts_cleanly(&mut state, &list);
}

// ------------------------------------------------- class boundaries above and below

/// Weather is a FIELD handler at order 8, so BOTH sides' chips precede EVERY
/// order-10 entry on either side — the class boundary the speed-major rule does not
/// cross. Within the class the chip is still speed-sorted, because the weather's
/// `onFieldResidual` runs `eachEvent('Weather')`, which speed-sorts the actives.
///
/// Showdown, transcribed (`residualspeedsand`):
///
/// ```text
/// |-weather|Sandstorm|[upkeep]
/// |-damage|p2a: Aipom|236/251|[from] Sandstorm        <- fast, order 8
/// |-damage|p1a: Snorlax|333/461|[from] Sandstorm      <- slow, order 8
/// |-heal|p2a: Aipom|251/251|[from] item: Leftovers    <- fast, 10.4
/// |-heal|p1a: Snorlax|361/461|[from] item: Leftovers  <- slow, 10.4
/// ```
#[test]
fn both_sides_take_weather_before_either_runs_an_order_ten_entry() {
    let mut state = bare_state(96, 206);
    state.weather.weather_type = Weather::SAND;
    state.weather.turns_remaining = 5;
    state.side_one.get_active().item = Items::LEFTOVERS;
    state.side_two.get_active().item = Items::LEFTOVERS;

    let list = only_branch(generate(&mut state));

    let fast_sand = damage_at(&list, SideReference::SideTwo);
    let slow_sand = damage_at(&list, SideReference::SideOne);
    let fast_heal = heal_at(&list, SideReference::SideTwo);
    let slow_heal = heal_at(&list, SideReference::SideOne);
    assert!(
        fast_sand < slow_sand && slow_sand < fast_heal && fast_heal < slow_heal,
        "order 8 for both sides, THEN order 10 for both sides: {:?}",
        list
    );
    assert_reverts_cleanly(&mut state, &list);
}

/// Wish is order 7 — ahead of the weather chip at 8 no matter who is faster.
///
/// Showdown, transcribed (slow wisher, sandstorm up):
///
/// ```text
/// |-heal|p2a: Blissey|651/651|[from] move: Wish|[wisher] Blissey
/// |-weather|Sandstorm|[upkeep]
/// |-damage|p2a: Blissey|611/651|[from] Sandstorm
/// ```
#[test]
fn wish_resolves_before_the_weather_chip() {
    let mut state = bare_state(96, 206);
    state.weather.weather_type = Weather::SAND;
    state.weather.turns_remaining = 5;
    // The SLOWER side is the wisher, so a speed-only rule would put it last.
    state.side_one.wish = (1, 0);

    let list = only_branch(generate(&mut state));

    let wish = heal_at(&list, SideReference::SideOne);
    let weather = index_of(&list, |i| {
        matches!(i, Instruction::DecrementWeatherTurnsRemaining)
    });
    assert!(
        wish < weather,
        "order 7 precedes order 8 regardless of speed: {:?}",
        list
    );
    assert_reverts_cleanly(&mut state, &list);
}

/// Future Sight is order 11 — after every order-10 handler on BOTH sides, and it is
/// held by the TARGET's slot, so it carries the target's speed rather than the
/// user's.
///
/// Showdown, transcribed (a 96-speed Slowbro's Future Sight landing on a 206-speed
/// Aipom, Slowbro holding Leftovers):
///
/// ```text
/// |-heal|p1a: Slowbro|291/331|[from] item: Leftovers   <- 10.4, spe 96
/// |-end|p2a: Aipom|move: Future Sight                  <- 11,  spe 206
/// ```
///
/// The engine used to resolve Future Sight near the TOP of the block, before every
/// order-10 entry on either side.
#[test]
fn future_sight_resolves_after_both_sides_order_ten() {
    let mut state = bare_state(96, 206);
    state.side_one.get_active().item = Items::LEFTOVERS;
    state.side_two.get_active().status = PokemonStatus::POISON;
    state.side_one.future_sight = (1, PokemonIndex::P0);

    let list = only_branch(generate(&mut state));

    let leftovers = heal_at(&list, SideReference::SideOne);
    let future = index_of(&list, |i| {
        matches!(i, Instruction::DecrementFutureSight(_))
    });
    let psn = damage_at(&list, SideReference::SideTwo);
    assert!(
        psn < leftovers && leftovers < future,
        "order 11 lands after the slower side's 10.4 heal, not before it: {:?}",
        list
    );
    // Two damages on side two: the poison tick, then the Future Sight hit.
    assert_eq!(
        damages(&list, SideReference::SideTwo).len(),
        2,
        "the Future Sight damage lands on the TARGET's side: {:?}",
        list
    );
    assert_reverts_cleanly(&mut state, &list);
}

/// The one cross-side emission inside the class: the Leech Seed sap damages the
/// seeded side and heals the seeder in the same breath, at the SEEDED side's 10.5
/// slot. With the victim faster, the seeder's (silent) drain heal is emitted BEFORE
/// its own 10.4 Leftovers heal — an ordering no per-side rule can produce, and the
/// reason `ResidualPlan` reads the segment instead of predicting from speed.
///
/// Showdown, transcribed (`residualspeedleech`):
///
/// ```text
/// |-heal|p2a: Aipom|135/251|[from] item: Leftovers            <- victim, 10.4
/// |-damage|p2a: Aipom|104/251|[from] Leech Seed|[of] p1a: Cacturne  <- victim, 10.5
/// |-heal|p1a: Cacturne|260/281|[silent]                       <- seeder's drain
/// |-heal|p1a: Cacturne|277/281|[from] item: Leftovers         <- seeder, 10.4
/// ```
#[test]
fn the_leech_seed_drain_heal_is_emitted_at_the_victims_slot() {
    let mut state = bare_state(96, 206);
    state.side_one.get_active().item = Items::LEFTOVERS;
    state.side_two.get_active().item = Items::LEFTOVERS;
    state
        .side_two
        .volatile_statuses
        .insert(PokemonVolatileStatus::LEECHSEED);

    let list = only_branch(generate(&mut state));

    let victim_heal = heal_at(&list, SideReference::SideTwo);
    let seeder_heals: Vec<usize> = list
        .iter()
        .enumerate()
        .filter(|(_, i)| matches!(i, Instruction::Heal(h) if h.side_ref == SideReference::SideOne))
        .map(|(index, _)| index)
        .collect();
    let victim_sap = damage_at(&list, SideReference::SideTwo);
    assert_eq!(
        seeder_heals.len(),
        2,
        "the drain heal and the seeder's own Leftovers both fire: {:?}",
        list
    );
    assert!(
        victim_heal < victim_sap && victim_sap < seeder_heals[0],
        "the faster victim runs 10.4 then 10.5, and the drain rides the sap: {:?}",
        list
    );
    assert!(
        seeder_heals[1] > seeder_heals[0],
        "the seeder's own Leftovers is emitted AFTER the drain, because the victim \
         resolved first: {:?}",
        list
    );
    assert_reverts_cleanly(&mut state, &list);
}

/// The same fixture with the seeder faster: now its Leftovers is emitted at its own
/// slot, BEFORE the sap. The pair is what makes the cross-side ordering a real
/// property rather than an artefact of the seat numbering.
#[test]
fn a_faster_seeder_heals_before_the_sap_it_drains() {
    let mut state = bare_state(206, 96);
    state.side_one.get_active().item = Items::LEFTOVERS;
    state.side_two.get_active().item = Items::LEFTOVERS;
    state
        .side_two
        .volatile_statuses
        .insert(PokemonVolatileStatus::LEECHSEED);

    let list = only_branch(generate(&mut state));

    let seeder_heals: Vec<usize> = list
        .iter()
        .enumerate()
        .filter(|(_, i)| matches!(i, Instruction::Heal(h) if h.side_ref == SideReference::SideOne))
        .map(|(index, _)| index)
        .collect();
    let victim_sap = damage_at(&list, SideReference::SideTwo);
    assert_eq!(seeder_heals.len(), 2, "{:?}", list);
    assert!(
        seeder_heals[0] < victim_sap && victim_sap < seeder_heals[1],
        "the faster seeder's own Leftovers precedes the sap; the drain follows it: \
         {:?}",
        list
    );
    assert_reverts_cleanly(&mut state, &list);
}
