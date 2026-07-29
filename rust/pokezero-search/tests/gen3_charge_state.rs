//! Mid-charge (two-turn move) state, asserted against the vendored gen3-patched
//! poke-engine.
//!
//! The engine already models this completely — `charge_choice_to_volatile`,
//! `active_is_charging_move` locking `get_all_options`, and the release in
//! `generate_instructions`. **The volatile IS the commitment.** What was missing
//! was upstream of the engine: the public mid-charge state never reached world
//! construction, so a sampled world was built with the charging Pokemon FREE and
//! search started a fresh charge instead of releasing (repro seed 1350004 step 66).
//!
//! These pin the engine contract the Python halves now depend on, so a wheel or
//! patch-set change that breaks it fails here rather than silently re-charging in
//! a searched world.
//!
//! Reachability: Solar Beam is the ONLY charge move in the gen3 randbats pool — of
//! the 17 moves carrying the dex `charge: 1` flag it is the one that appears, in 4
//! sets (Exeggutor, Sunflora, Tangela, Victreebel).

use poke_engine::choices::Choices;
use poke_engine::engine::state::{MoveChoice, PokemonVolatileStatus};
use poke_engine::instruction::{Instruction, StateInstructions};
use poke_engine::state::{PokemonIndex, PokemonMoveIndex, SideReference, State};

/// Side one is a Solar Beam carrier with two other options, so "only Solar Beam is
/// offered" is a real restriction rather than the only thing available.
fn solar_beam_state(charging: bool) -> State {
    let mut state = State::default();

    let attacker = state.side_one.get_active();
    attacker.hp = 300;
    attacker.maxhp = 300;
    attacker.special_attack = 250;
    attacker.speed = 140;
    attacker.replace_move(PokemonMoveIndex::M0, Choices::SOLARBEAM);
    attacker.replace_move(PokemonMoveIndex::M1, Choices::PSYCHIC);
    attacker.replace_move(PokemonMoveIndex::M2, Choices::SPLASH);

    let defender = state.side_two.get_active();
    defender.hp = 460;
    defender.maxhp = 460;
    defender.special_defense = 220;
    defender.speed = 60;
    defender.replace_move(PokemonMoveIndex::M0, Choices::SPLASH);

    if charging {
        state
            .side_one
            .volatile_statuses
            .insert(PokemonVolatileStatus::SOLARBEAM);
    }
    state
}

fn generate(state: &mut State, side_one: &MoveChoice, side_two: &MoveChoice) -> Vec<Instruction> {
    let before = format!("{:?}", state);
    let branches: Vec<StateInstructions> =
        poke_engine::engine::generate_instructions::generate_instructions_from_move_pair(
            state, side_one, side_two, false,
        );
    assert_eq!(before, format!("{:?}", state), "state was mutated");
    let best = branches
        .into_iter()
        .max_by(|a, b| a.percentage.partial_cmp(&b.percentage).unwrap())
        .expect("expected at least one branch");
    best.instruction_list
}

fn damage_to_side_two(list: &[Instruction]) -> i16 {
    list.iter()
        .filter_map(|instruction| match instruction {
            Instruction::Damage(damage) if damage.side_ref == SideReference::SideTwo => {
                Some(damage.damage_amount)
            }
            _ => None,
        })
        .sum()
}

fn applies_charge(list: &[Instruction]) -> bool {
    list.iter().any(|instruction| match instruction {
        Instruction::ApplyVolatileStatus(apply) => {
            apply.volatile_status == PokemonVolatileStatus::SOLARBEAM
        }
        _ => false,
    })
}

fn removes_charge(list: &[Instruction]) -> bool {
    list.iter().any(|instruction| match instruction {
        Instruction::RemoveVolatileStatus(remove) => {
            remove.volatile_status == PokemonVolatileStatus::SOLARBEAM
        }
        _ => false,
    })
}

/// The lock: while charging, the side has exactly one legal action — the move it
/// committed to. Not its other moves, and not a switch.
#[test]
fn a_charging_side_is_offered_only_the_charged_move() {
    let state = solar_beam_state(true);
    let (side_one_options, _) = state.get_all_options();

    assert_eq!(
        side_one_options,
        vec![MoveChoice::Move(PokemonMoveIndex::M0)],
        "a charging side must be locked to Solar Beam alone"
    );
}

/// The control: with no charge state the same side has its full option set, so the
/// pin above is measuring the volatile and not the fixture.
#[test]
fn an_uncharged_side_has_its_full_option_set() {
    let state = solar_beam_state(false);
    let (side_one_options, _) = state.get_all_options();

    assert!(
        side_one_options.len() > 1,
        "expected moves and switches, got {:?}",
        side_one_options
    );
    assert!(
        side_one_options.contains(&MoveChoice::Switch(PokemonIndex::P1)),
        "an uncharged side can switch: {:?}",
        side_one_options
    );
}

/// The payoff. A world built WITH the charge state releases: real damage, and the
/// commitment is consumed rather than re-armed.
#[test]
fn a_charged_world_releases_instead_of_re_charging() {
    let mut state = solar_beam_state(true);
    let list = generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    );

    assert!(
        damage_to_side_two(&list) > 0,
        "the release must deal damage: {:?}",
        list
    );
    assert!(
        removes_charge(&list),
        "the release must consume the commitment: {:?}",
        list
    );
    assert!(
        !applies_charge(&list),
        "the release must NOT re-arm the charge: {:?}",
        list
    );
}

/// The bug this whole change exists to close, stated as a contrast. Built WITHOUT
/// the charge state — which is what world construction did before the parser
/// surfaced it — the identical click CHARGES: no damage, and a fresh commitment.
/// A search running this world prices a turn that already happened.
#[test]
fn an_uncharged_world_starts_a_fresh_charge_and_deals_nothing() {
    let mut state = solar_beam_state(false);
    let list = generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    );

    assert_eq!(
        damage_to_side_two(&list),
        0,
        "the charge turn deals nothing: {:?}",
        list
    );
    assert!(
        applies_charge(&list),
        "the charge turn arms the commitment: {:?}",
        list
    );
}

/// The two worlds must not merely differ — the charged one has to be the one that
/// lands the hit, which is the whole reason the state is worth carrying.
#[test]
fn carrying_the_charge_is_worth_a_full_solar_beam() {
    let mut charged = solar_beam_state(true);
    let mut uncharged = solar_beam_state(false);
    let charged_damage = damage_to_side_two(&generate(
        &mut charged,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    ));
    let uncharged_damage = damage_to_side_two(&generate(
        &mut uncharged,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    ));

    assert!(
        charged_damage > uncharged_damage,
        "charged {} vs uncharged {}",
        charged_damage,
        uncharged_damage
    );
    assert_eq!(uncharged_damage, 0);
}

/// The volatile round-trips through the wire format the search crate receives its
/// root world on — the property `require_charge_state_support` probes from Python.
#[test]
fn the_charge_volatile_survives_serialization() {
    let state = solar_beam_state(true);
    let serialized = state.serialize();
    assert!(
        serialized.to_uppercase().contains("SOLARBEAM"),
        "the charge volatile must reach the wire format"
    );

    let round_tripped = State::deserialize(&serialized);
    assert!(round_tripped
        .side_one
        .volatile_statuses
        .contains(&PokemonVolatileStatus::SOLARBEAM));
    assert_eq!(serialized, round_tripped.serialize());
}
