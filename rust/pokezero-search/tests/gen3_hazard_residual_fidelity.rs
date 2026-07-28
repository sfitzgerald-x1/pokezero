//! Gen 3 Spikes-layer and end-of-turn-residual fidelity pins, asserted directly
//! against the vendored gen3-patched poke-engine (`third_party/poke-engine-src/`).
//!
//! Companion to `gen3_switch_fidelity.rs`: every expectation here was read off
//! the **real** Node Showdown simulator driven through
//! `scripts/gen3_switch_differential.py`, which is the ground-truth gate; this
//! file is the engine-contract pin, so a version bump that silently drops one of
//! the `third_party/poke-engine-gen3-*.patch` files fails `cargo test` instead
//! of quietly regressing search fidelity.
//!
//! Coverage, and which patch each pin guards:
//!
//! * Spikes deal 1/8, 1/6, 1/4 of max HP at one, two and three layers, floored,
//!   with a 1 HP minimum — `poke-engine-gen3-spikes-layers.patch`.
//! * The end-of-turn residual block is deferred past a mid-turn faint's forced
//!   replacement and applies once, to the replacement too —
//!   `poke-engine-gen3-residual-defer-on-faint.patch`.

use poke_engine::choices::Choices;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::{MoveChoice, Weather};
use poke_engine::instruction::{Instruction, StateInstructions};
use poke_engine::state::{PokemonIndex, PokemonMoveIndex, SideReference, State};

/// `generate_instructions_from_move_pair` must leave `state` untouched — the
/// engine's own test suite asserts this and a patch that mutates without
/// emitting a reversible instruction would silently corrupt search.
fn generate(
    state: &mut State,
    side_one: &MoveChoice,
    side_two: &MoveChoice,
) -> Vec<StateInstructions> {
    let before = format!("{:?}", state);
    let instructions = generate_instructions_from_move_pair(state, side_one, side_two, false);
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

/// Apply then reverse: search relies on every emitted instruction being an
/// exact inverse, so a restructured block that forgets one corrupts the tree.
fn assert_reverts_cleanly(state: &mut State, list: &Vec<Instruction>) {
    let before = format!("{:?}", state);
    state.apply_instructions(list);
    state.reverse_instructions(list);
    assert_eq!(
        before,
        format!("{:?}", state),
        "instructions did not revert"
    );
}

fn damages(list: &[Instruction], side_ref: SideReference) -> Vec<i16> {
    list.iter()
        .filter_map(|instruction| match instruction {
            Instruction::Damage(damage) if damage.side_ref == side_ref => {
                Some(damage.damage_amount)
            }
            _ => None,
        })
        .collect()
}

fn position<F>(list: &[Instruction], predicate: F) -> Option<usize>
where
    F: Fn(&Instruction) -> bool,
{
    list.iter().position(predicate)
}

fn weather_ticks(list: &[Instruction]) -> usize {
    list.iter()
        .filter(|instruction| matches!(instruction, Instruction::DecrementWeatherTurnsRemaining))
        .count()
}

// ---------------------------------------------------------------------------
// Spikes layer fractions (poke-engine-gen3-spikes-layers.patch)
// ---------------------------------------------------------------------------

/// Side one switches P0 -> P1 onto `layers` of its own Spikes. Side two does
/// nothing, so the only damage in the branch is the hazard, and `(Switch, None)`
/// keeps the end-of-turn block out of the way.
fn switch_into_spikes(layers: i8, maxhp: i16, hp: i16) -> Vec<Instruction> {
    let mut state = State::default();
    state.side_one.side_conditions.spikes = layers;
    let incoming = &mut state.side_one.pokemon[PokemonIndex::P1];
    incoming.maxhp = maxhp;
    incoming.hp = hp;

    let list = only_branch(generate(
        &mut state,
        &MoveChoice::Switch(PokemonIndex::P1),
        &MoveChoice::None,
    ));
    assert_reverts_cleanly(&mut state, &list);
    list
}

fn spikes_damage(layers: i8, maxhp: i16) -> i16 {
    let taken = damages(
        &switch_into_spikes(layers, maxhp, maxhp),
        SideReference::SideOne,
    );
    assert_eq!(
        taken.len(),
        1,
        "expected exactly one hazard hit at {} layer(s)",
        layers
    );
    taken[0]
}

/// Showdown gen3 ground truth (`scripts/gen3_switch_differential.py`, scenarios
/// `spikes2layers` / `spikes3layers`): a 461 HP Snorlax switching into one, two
/// and three layers lands on `404/461`, `385/461` and `346/461` — 57, 76 and
/// 115 HP, i.e. 1/8, 1/6 and 1/4 of max HP.
///
/// The bug this guards: upstream dealt `maxhp * layers / 8`, exactly 1.5x too
/// much at two layers (1/4 instead of 1/6) and three layers (3/8 instead of
/// 1/4). The engine only agreed with the sim at a single layer.
#[test]
fn spikes_deal_an_eighth_a_sixth_and_a_quarter_by_layer() {
    assert_eq!(spikes_damage(1, 461), 57, "one layer is maxhp/8");
    assert_eq!(spikes_damage(2, 461), 76, "two layers are maxhp/6");
    assert_eq!(spikes_damage(3, 461), 115, "three layers are maxhp/4");
}

/// Showdown computes `[0, 3, 4, 6][layers] * maxhp / 24` and rounds it through
/// `clampIntRange(damage, 1)` — a FLOOR, not a round. 100 max HP is the sharp
/// case: 1/6 of it is 16.67 and 1/8 is 12.5, both of which round UP but floor
/// DOWN.
#[test]
fn spikes_damage_floors_rather_than_rounds() {
    assert_eq!(spikes_damage(1, 100), 12, "floor(100 * 3 / 24)");
    assert_eq!(spikes_damage(2, 100), 16, "floor(100 * 4 / 24)");
    assert_eq!(spikes_damage(3, 100), 25, "floor(100 * 6 / 24)");
    // 24 max HP is the smallest value where every layer divides exactly.
    assert_eq!(spikes_damage(1, 24), 3);
    assert_eq!(spikes_damage(2, 24), 4);
    assert_eq!(spikes_damage(3, 24), 6);
    // 23 * 3 / 24 = 2.875: floored, and comfortably above the 1 HP clamp.
    assert_eq!(spikes_damage(1, 23), 2);
}

/// `clampIntRange(damage, 1)` also floors the result at 1 HP, so a 1 HP
/// Shedinja FAINTS to a single layer — Showdown emits
/// `|-damage|p2a: Shedinja|0 fnt|[from] Spikes` followed by `|faint|`.
/// Upstream's `maxhp * layers / 8` truncated to zero and let it walk in free.
///
/// Shedinja's 1 max HP is the case the sim can actually be asked about (no gen3
/// Pokemon has a max HP between 2 and 11); 4 and 7 exercise the same clamp on
/// either side of it.
#[test]
fn spikes_always_deal_at_least_one_hp() {
    for maxhp in [1, 4, 7] {
        assert_eq!(
            spikes_damage(1, maxhp),
            1,
            "one layer must still bite a {} HP Pokemon",
            maxhp
        );
    }
    // Two and three layers of 1 HP are the same clamp.
    assert_eq!(spikes_damage(2, 1), 1);
    assert_eq!(spikes_damage(3, 1), 1);
}

/// `Pokemon.damage` caps the hit at the remaining HP, so a Pokemon that comes
/// back in below the hazard fraction takes exactly what it has left.
#[test]
fn spikes_damage_is_capped_at_remaining_hp() {
    let taken = damages(&switch_into_spikes(3, 461, 10), SideReference::SideOne);
    assert_eq!(taken, vec![10], "hazard cannot overdraw the incoming HP");
}

// ---------------------------------------------------------------------------
// Residual deferral across a forced replacement
// (poke-engine-gen3-residual-defer-on-faint.patch)
// ---------------------------------------------------------------------------

/// Sandstorm up, side one holding a 100%-accuracy attack, side two's active on
/// its last HP so the attack is a guaranteed, branch-free KO.
fn sandstorm_kill_state() -> State {
    let mut state = State::default();
    state.weather.weather_type = Weather::SAND;
    state.weather.turns_remaining = 5;
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::SWIFT);
    state.side_two.get_active().hp = 1;
    state
}

/// Showdown gen3 ground truth (`scripts/gen3_switch_differential.py`, scenario
/// `faintresiduals`): the faint ply's protocol block ends at
/// `|faint|p2a: Misdreavus` — no `|-weather|Sandstorm|[upkeep]`, no `|upkeep`,
/// no `|turn|`. `runAction` sees the pending switch flag, issues a `switch`
/// request and returns with the queued `residual` action untouched.
///
/// Upstream ran the whole residual block in the same instruction set as the
/// faint, which ticked the weather and the screens on the wrong side of the
/// replacement switch.
#[test]
fn a_mid_turn_faint_defers_the_whole_residual_block() {
    let mut state = sandstorm_kill_state();
    let list = only_branch(generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::None,
    ));

    assert_eq!(
        damages(&list, SideReference::SideTwo),
        vec![1],
        "the KO is the only damage on the fainting side: {:?}",
        list
    );
    assert!(
        damages(&list, SideReference::SideOne).is_empty(),
        "the survivor must not take its sandstorm tick on the faint ply: {:?}",
        list
    );
    assert_eq!(
        weather_ticks(&list),
        0,
        "the weather counter must not tick on the faint ply: {:?}",
        list
    );
    assert!(
        list.contains(&Instruction::ToggleSideTwoForceSwitch),
        "the faint must be flagged as a replacement owed: {:?}",
        list
    );
    assert_reverts_cleanly(&mut state, &list);

    // The flagged side is the only one that acts at the next boundary.
    state.apply_instructions(&list);
    let (side_one_options, side_two_options) = state.get_all_options();
    assert_eq!(side_one_options, vec![MoveChoice::None]);
    assert!(
        side_two_options
            .iter()
            .all(|option| matches!(option, MoveChoice::Switch(_))),
        "the fainted side is offered replacements only: {:?}",
        side_two_options
    );
}

/// The other half of the deferral: the block re-attaches to the ply that
/// resolves the replacement, AFTER the switch, and hits the incoming Pokemon
/// too. Showdown's step-2 protocol is `|switch|p2a: Blissey|651/651`, then
/// `|-weather|Sandstorm|[upkeep]`, then
/// `|-damage|p2a: Blissey|611/651|[from] Sandstorm`.
#[test]
fn the_deferred_block_lands_after_the_replacement_and_hits_it() {
    let mut state = sandstorm_kill_state();
    let faint = only_branch(generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::None,
    ));
    state.apply_instructions(&faint);

    let list = only_branch(generate(
        &mut state,
        &MoveChoice::None,
        &MoveChoice::Switch(PokemonIndex::P1),
    ));

    assert_eq!(
        weather_ticks(&list),
        1,
        "the deferred weather tick lands exactly once: {:?}",
        list
    );
    assert_eq!(
        damages(&list, SideReference::SideOne),
        vec![6],
        "the survivor takes its deferred sandstorm tick: {:?}",
        list
    );
    assert_eq!(
        damages(&list, SideReference::SideTwo),
        vec![6],
        "the REPLACEMENT takes sandstorm on the turn it comes in: {:?}",
        list
    );

    let switch = position(&list, |instruction| {
        matches!(instruction, Instruction::Switch(switch) if switch.side_ref == SideReference::SideTwo)
    })
    .expect("the replacement switch is in the list");
    let residual = position(&list, |instruction| {
        matches!(instruction, Instruction::DecrementWeatherTurnsRemaining)
    })
    .expect("the deferred weather tick is in the list");
    assert!(
        switch < residual,
        "gen3 sends the replacement out BEFORE the residual block: {:?}",
        list
    );

    assert_reverts_cleanly(&mut state, &list);
}

/// The double-apply guard. A faint caused BY the residual block happens after
/// the block has already run, so the replacement ply that follows must NOT run
/// it a second time. Keying the deferral on the `force_switch` flag (set only
/// for a replacement still owed) rather than on "the active is fainted" is what
/// separates the two cases.
#[test]
fn a_residual_induced_faint_does_not_re_run_the_block() {
    let mut state = State::default();
    state.weather.weather_type = Weather::SAND;
    state.weather.turns_remaining = 5;
    // Side two's active dies to its own sandstorm tick, inside the block.
    state.side_two.get_active().hp = 1;

    let faint = only_branch(generate(&mut state, &MoveChoice::None, &MoveChoice::None));
    assert_eq!(
        weather_ticks(&faint),
        1,
        "the block runs normally when nothing is fainted going in: {:?}",
        faint
    );
    assert_eq!(damages(&faint, SideReference::SideTwo), vec![1]);
    assert!(
        !faint.contains(&Instruction::ToggleSideTwoForceSwitch),
        "a faint from inside the block is not a pending deferral: {:?}",
        faint
    );
    state.apply_instructions(&faint);

    let replacement = only_branch(generate(
        &mut state,
        &MoveChoice::None,
        &MoveChoice::Switch(PokemonIndex::P1),
    ));
    assert_eq!(
        weather_ticks(&replacement),
        0,
        "the block must not run twice in one turn: {:?}",
        replacement
    );
    assert!(
        damages(&replacement, SideReference::SideOne).is_empty(),
        "no second sandstorm tick for the survivor: {:?}",
        replacement
    );
    assert!(
        damages(&replacement, SideReference::SideTwo).is_empty(),
        "no sandstorm tick for a replacement that entered after the block: {:?}",
        replacement
    );
}

/// Both seats owing a replacement is one simultaneous boundary in Showdown, and
/// the `hp <= 0` path already modelled it that way. Now that a faint sets
/// `force_switch`, the flag path has to agree — otherwise the replacements
/// serialize and the second seat is handed a stale
/// `switch_out_move_second_saved_move` to "use" while fainted.
#[test]
fn both_seats_owing_a_replacement_switch_at_the_same_boundary() {
    let mut state = State::default();
    state.side_one.force_switch = true;
    state.side_two.force_switch = true;

    let (side_one_options, side_two_options) = state.get_all_options();

    for (side, options) in [("one", &side_one_options), ("two", &side_two_options)] {
        assert!(
            !options.is_empty()
                && options
                    .iter()
                    .all(|option| matches!(option, MoveChoice::Switch(_))),
            "side {} must be offered replacements: {:?}",
            side,
            options
        );
    }
}
