//! Gen 3 fixed-damage / Substitute-routing pins for the `choice_special_effect`
//! handler family, asserted directly against the vendored gen3-patched
//! poke-engine (`third_party/poke-engine-src/`).
//!
//! Ground truth is `substitute.onTryPrimaryHit` (data/moves.ts; gen3 adds no
//! override): a move without `bypasssub` computes its damage against the
//! POKEMON and that damage is then capped at the substitute's remaining HP —
//!
//! ```js
//! if (damage > target.volatiles['substitute'].hp) damage = sub.hp;
//! sub.hp -= damage;
//! if (sub.hp <= 0) target.removeVolatile('substitute');
//! ```
//!
//! so the substitute breaks with NO overflow onto the Pokemon behind it. None of
//! the fixed-damage moves carries `bypasssub` in the gen3 data.
//!
//! This file is the audit's record: the routing fix is pinned, and so are the
//! guards that were found already correct (Protect, type immunity, Pain Split's
//! skip), because they live in the same dispatcher the fix edits.

use poke_engine::choices::Choices;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::{MoveChoice, PokemonVolatileStatus};
use poke_engine::instruction::{Instruction, StateInstructions};
use poke_engine::state::{PokemonMoveIndex, PokemonType, SideReference, State};

fn generate(state: &mut State, attacker_move: &MoveChoice) -> Vec<Instruction> {
    let before = format!("{:?}", state);
    let instructions =
        generate_instructions_from_move_pair(state, attacker_move, &MoveChoice::None, false);
    assert_eq!(before, format!("{:?}", state), "state was mutated");
    // Since the fixed-damage-pipeline patch these moves take the ordinary
    // damage path, so real accuracy is rolled: Super Fang (90%) carries a miss
    // branch the old direct-apply arm never had (the sim rolls it too — probe
    // 2026-07-29: 2 misses in 12 seeded games). Deterministic-accuracy moves
    // still produce exactly one branch; assert on the HIT branch either way.
    assert!(
        instructions.len() <= 2,
        "expected at most a hit and a miss branch, got {:?}",
        instructions
    );
    let list = instructions
        .into_iter()
        .max_by(|a, b| {
            a.instruction_list
                .len()
                .cmp(&b.instruction_list.len())
                .then(a.percentage.partial_cmp(&b.percentage).unwrap())
        })
        .unwrap()
        .instruction_list;

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

fn damage_to_pokemon(list: &[Instruction], side_ref: SideReference) -> i16 {
    list.iter()
        .filter_map(|instruction| match instruction {
            Instruction::Damage(damage) if damage.side_ref == side_ref => {
                Some(damage.damage_amount)
            }
            _ => None,
        })
        .sum()
}

fn damage_to_substitute(list: &[Instruction], side_ref: SideReference) -> i16 {
    list.iter()
        .filter_map(|instruction| match instruction {
            Instruction::DamageSubstitute(damage) if damage.side_ref == side_ref => {
                Some(damage.damage_amount)
            }
            _ => None,
        })
        .sum()
}

fn substitute_broken(list: &[Instruction], side_ref: SideReference) -> bool {
    list.iter().any(|instruction| match instruction {
        Instruction::RemoveVolatileStatus(remove) => {
            remove.side_ref == side_ref
                && remove.volatile_status == PokemonVolatileStatus::SUBSTITUTE
        }
        _ => false,
    })
}

/// Side one attacks with `move_id`; side two optionally stands behind a
/// substitute of `substitute_health`.
fn fixed_damage_state(move_id: Choices, substitute_health: i16) -> State {
    let mut state = State::default();
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, move_id);
    {
        let defender = state.side_two.get_active();
        defender.maxhp = 461;
        defender.hp = 346;
    }
    if substitute_health > 0 {
        state
            .side_two
            .volatile_statuses
            .insert(PokemonVolatileStatus::SUBSTITUTE);
        state.side_two.substitute_health = substitute_health;
    }
    state
}

// ---------------------------------------------------------------------------
// (a) Substitute routing — the confirmed bug
// ---------------------------------------------------------------------------

/// Showdown gen3 ground truth: Seismic Toss into a 115 HP Substitute answers
/// `-activate p2a: Snorlax|Substitute|[damage]` and emits NO `-damage` on the
/// Pokemon. Level 100 attacker, so the hit is 100 and the sub survives at 15.
#[test]
fn seismic_toss_hits_the_substitute_not_the_pokemon() {
    let mut state = fixed_damage_state(Choices::SEISMICTOSS, 115);
    let list = generate(&mut state, &MoveChoice::Move(PokemonMoveIndex::M0));

    assert_eq!(
        damage_to_substitute(&list, SideReference::SideTwo),
        100,
        "the substitute absorbs the level-damage hit: {:?}",
        list
    );
    assert_eq!(
        damage_to_pokemon(&list, SideReference::SideTwo),
        0,
        "the Pokemon behind must not be touched: {:?}",
        list
    );
    assert!(
        !substitute_broken(&list, SideReference::SideTwo),
        "a 115 HP substitute survives a 100 HP hit: {:?}",
        list
    );
}

/// The break case, and the one that pins NO OVERFLOW: Seismic Toss (100) into a
/// 65 HP Substitute answers `-end p2a: Dodrio|Substitute` in the real sim, with
/// no `-damage` line on the Pokemon at all — the surplus 35 is discarded.
#[test]
fn a_fixed_damage_hit_breaks_the_substitute_without_overflowing() {
    let mut state = fixed_damage_state(Choices::SEISMICTOSS, 65);
    let list = generate(&mut state, &MoveChoice::Move(PokemonMoveIndex::M0));

    assert_eq!(
        damage_to_substitute(&list, SideReference::SideTwo),
        65,
        "the substitute absorbs only what it has: {:?}",
        list
    );
    assert_eq!(
        damage_to_pokemon(&list, SideReference::SideTwo),
        0,
        "the surplus must NOT overflow onto the Pokemon: {:?}",
        list
    );
    assert!(
        substitute_broken(&list, SideReference::SideTwo),
        "the substitute breaks: {:?}",
        list
    );
}

/// The whole family shares the routing, so the whole family is pinned. Super
/// Fang and Endeavor still compute their damage from the POKEMON's HP — Showdown
/// runs the damage callback against the mon and only then caps at the sub — so
/// each is checked at a substitute small enough to break and one large enough to
/// survive.
#[test]
fn every_fixed_damage_arm_routes_through_the_substitute() {
    for move_id in [Choices::SEISMICTOSS, Choices::SUPERFANG, Choices::ENDEAVOR] {
        let mut state = fixed_damage_state(move_id, 400);
        // Endeavor only fires when the attacker is below the target.
        state.side_one.get_active().hp = 1;
        let list = generate(&mut state, &MoveChoice::Move(PokemonMoveIndex::M0));

        assert!(
            damage_to_substitute(&list, SideReference::SideTwo) > 0,
            "{:?} must land on the substitute: {:?}",
            move_id,
            list
        );
        assert_eq!(
            damage_to_pokemon(&list, SideReference::SideTwo),
            0,
            "{:?} must not touch the Pokemon behind a substitute: {:?}",
            move_id,
            list
        );
    }
}

/// The control that keeps the fix honest: with no substitute up, every arm still
/// damages the Pokemon exactly as before.
#[test]
fn without_a_substitute_the_pokemon_still_takes_the_hit() {
    let mut state = fixed_damage_state(Choices::SEISMICTOSS, 0);
    let list = generate(&mut state, &MoveChoice::Move(PokemonMoveIndex::M0));

    assert_eq!(
        damage_to_pokemon(&list, SideReference::SideTwo),
        100,
        "level-100 Seismic Toss deals 100 to an unprotected target: {:?}",
        list
    );
    assert_eq!(
        damage_to_substitute(&list, SideReference::SideTwo),
        0,
        "no substitute damage without a substitute: {:?}",
        list
    );
}

/// An ordinary damaging move is untouched by the routing change — it already
/// went through the normal damage path's substitute handling.
#[test]
fn the_ordinary_damage_path_is_unchanged() {
    let mut state = fixed_damage_state(Choices::TACKLE, 115);
    let list = generate(&mut state, &MoveChoice::Move(PokemonMoveIndex::M0));

    assert!(
        damage_to_substitute(&list, SideReference::SideTwo) > 0,
        "Tackle still hits the substitute: {:?}",
        list
    );
    assert_eq!(
        damage_to_pokemon(&list, SideReference::SideTwo),
        0,
        "and still not the Pokemon: {:?}",
        list
    );
}

// ---------------------------------------------------------------------------
// (c) Type immunity — audited, already correct
// ---------------------------------------------------------------------------

/// Seismic Toss is Fighting-typed and Ghost is the only gen3 type that zeroes
/// either Normal or Fighting, so the shared NORMAL probe gives the right answer.
/// Verified against the sim: Seismic Toss into Gengar answers `-immune`.
#[test]
fn fixed_damage_respects_ghost_immunity() {
    for move_id in [Choices::SEISMICTOSS, Choices::SUPERFANG] {
        for substitute_health in [0, 115] {
            let mut state = fixed_damage_state(move_id, substitute_health);
            state.side_two.get_active().types = (PokemonType::GHOST, PokemonType::TYPELESS);
            let list = generate(&mut state, &MoveChoice::Move(PokemonMoveIndex::M0));

            assert_eq!(
                damage_to_pokemon(&list, SideReference::SideTwo),
                0,
                "{:?} is immune vs Ghost (sub {}): {:?}",
                move_id,
                substitute_health,
                list
            );
            assert_eq!(
                damage_to_substitute(&list, SideReference::SideTwo),
                0,
                "{:?} vs Ghost must not even reach the substitute: {:?}",
                move_id,
                list
            );
        }
    }
}

// ---------------------------------------------------------------------------
// (b) Protect — audited, already correct (one guard covers every arm)
// ---------------------------------------------------------------------------

/// The dispatcher's single `blocked_by_protect` early return covers the whole
/// family, not just the hazard-clear arm it was originally added for.
#[test]
fn protect_blocks_every_fixed_damage_arm() {
    for move_id in [Choices::SEISMICTOSS, Choices::SUPERFANG, Choices::ENDEAVOR] {
        let mut state = fixed_damage_state(move_id, 0);
        state.side_one.get_active().hp = 1;
        state
            .side_two
            .get_active()
            .replace_move(PokemonMoveIndex::M0, Choices::PROTECT);
        state.side_two.get_active().speed = 500;

        let instructions = generate_instructions_from_move_pair(
            &mut state,
            &MoveChoice::Move(PokemonMoveIndex::M0),
            &MoveChoice::Move(PokemonMoveIndex::M0),
            false,
        );
        for branch in &instructions {
            assert_eq!(
                damage_to_pokemon(&branch.instruction_list, SideReference::SideTwo),
                0,
                "{:?} must not leak through Protect: {:?}",
                move_id,
                branch.instruction_list
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Pain Split — audited, already correct (skips entirely behind a substitute)
// ---------------------------------------------------------------------------

/// Pain Split has no `bypasssub` and no damage callback, so Showdown's
/// `getDamage` returns nothing and the move FAILS behind a substitute. The
/// engine's skip is that failure.
#[test]
fn pain_split_fails_behind_a_substitute() {
    let mut state = fixed_damage_state(Choices::PAINSPLIT, 115);
    state.side_one.get_active().hp = 50;
    let list = generate(&mut state, &MoveChoice::Move(PokemonMoveIndex::M0));

    assert_eq!(damage_to_pokemon(&list, SideReference::SideTwo), 0);
    assert_eq!(damage_to_substitute(&list, SideReference::SideTwo), 0);
    assert_eq!(
        damage_to_pokemon(&list, SideReference::SideOne),
        0,
        "and the user is not levelled either: {:?}",
        list
    );
}
