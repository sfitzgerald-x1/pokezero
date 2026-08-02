//! Battle-end truncation of the end-of-turn residual block.
//!
//! Showdown STOPS the residual block the moment a faint ends the battle; the
//! engine ran it to completion. 32 of 298 residue rows (11%), census-decidable by
//! the `|win|` line.
//!
//! Ground truth is `Battle.fieldEvent` (sim/battle.ts), the residual dispatch
//! loop:
//!
//! ```js
//! while (handlers.length) {
//!     const handler = handlers[0]; handlers.shift();
//!     ...
//!     if (handler.callback) { this.singleEvent(handlerEventid, effect, ...); }
//!     this.faintMessages();
//!     if (this.ended) return;      // <-- between entries
//! }
//! ```
//!
//! The boundary is BETWEEN entries, not mid-entry: the faint-causing entry runs to
//! completion, `faintMessages()` resolves it and `checkWin` sets `ended`, and every
//! remaining handler is skipped — the WINNER's own Leftovers included.
//!
//! Guards `third_party/poke-engine-gen3-battle-end-residuals.patch`.

use poke_engine::choices::Choices;
use poke_engine::engine::state::MoveChoice;
use poke_engine::instruction::{Instruction, StateInstructions};
use poke_engine::state::{PokemonIndex, PokemonMoveIndex, SideReference, State};

fn generate(state: &mut State, side_one: &MoveChoice, side_two: &MoveChoice) -> Vec<Instruction> {
    let before = format!("{:?}", state);
    let branches: Vec<StateInstructions> =
        poke_engine::engine::generate_instructions::generate_instructions_from_move_pair(
            state, side_one, side_two, false,
        );
    assert_eq!(before, format!("{:?}", state), "state was mutated");
    branches
        .into_iter()
        .max_by(|a, b| a.percentage.partial_cmp(&b.percentage).unwrap())
        .expect("expected a branch")
        .instruction_list
}

fn heals(list: &[Instruction], side_ref: SideReference) -> Vec<i16> {
    list.iter()
        .filter_map(|instruction| match instruction {
            Instruction::Heal(heal) if heal.side_ref == side_ref => Some(heal.heal_amount),
            _ => None,
        })
        .collect()
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

fn terminal_residual_roll_limit_state() -> State {
    let mut state = State::default();

    let attacker = state.side_one.get_active();
    attacker.level = 87;
    attacker.types = (
        poke_engine::state::PokemonType::ICE,
        poke_engine::state::PokemonType::GROUND,
    );
    attacker.hp = 154;
    attacker.maxhp = 316;
    attacker.attack = 224;
    attacker.speed = 137;
    attacker.item = poke_engine::engine::items::Items::LEFTOVERS;
    attacker.replace_move(PokemonMoveIndex::M0, Choices::EARTHQUAKE);

    let defender = state.side_two.get_active();
    defender.types = (
        poke_engine::state::PokemonType::NORMAL,
        poke_engine::state::PokemonType::TYPELESS,
    );
    defender.hp = 182;
    defender.maxhp = 362;
    defender.defense = 201;
    defender.speed = 201;
    defender.status = poke_engine::state::PokemonStatus::TOXIC;
    defender.replace_move(PokemonMoveIndex::M0, Choices::SPLASH);
    state.side_two.side_conditions.toxic_count = 2;

    for index in [
        PokemonIndex::P1,
        PokemonIndex::P2,
        PokemonIndex::P3,
        PokemonIndex::P4,
        PokemonIndex::P5,
    ] {
        state.side_one.pokemon[index].hp = 0;
        state.side_two.pokemon[index].hp = 0;
    }
    state
}

/// This is an ablation pin, not a fidelity assertion. The compact result below
/// demonstrates why the withdrawn Toxic-only splitter cannot be restored: a
/// correct comparison must evaluate the complete move pair and residual queue.
#[test]
fn terminal_residual_roll_split_remains_an_explicit_comparison_limit() {
    let mut state = terminal_residual_roll_limit_state();
    let before = format!("{:?}", state);
    let branches = poke_engine::engine::generate_instructions::generate_instructions_from_move_pair(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
        true,
    );

    assert_eq!(before, format!("{:?}", state), "branch generation mutated state");
    assert!(
        (branches.iter().map(|branch| branch.percentage).sum::<f32>() - 100.0).abs() < 1e-6,
        "the compact ablation must preserve probability mass"
    );
    assert_eq!(branches.len(), 2, "no production residual roll splitter is installed");
    assert!(branches.iter().any(|branch| {
        damages(&branch.instruction_list, SideReference::SideTwo).first() == Some(&113)
            && heals(&branch.instruction_list, SideReference::SideOne) == vec![19]
    }));
}

/// Side two's active is poisoned on 1 HP, so the residual block kills it — and it is
/// the FASTER mon (200 vs 50), so under gen3's speed-major ordering its ENTIRE
/// order-10 set resolves before side one runs any of its own.
///
/// Side one is burned and holds Leftovers. Both of those are order 10 (10.4 and
/// 10.6), i.e. the same class as the fatal poison tick, so BOTH sit behind it in
/// the sorted handler list and both are truncated. This fixture used to assert the
/// opposite for the heal, on the belief that Leftovers was "order 5" and therefore
/// a class ahead of status damage — that is the gen5+ table, which gen3 does not
/// use, and the expectation was only ever green because the engine had the same
/// wrong model. See `gen3_residual_speed_order.rs` for the order table and its
/// resolution.
///
/// `reserves_alive` decides whether that faint ends the battle (no living reserve)
/// or is an ordinary mid-block faint.
fn poison_kill_state(reserves_alive: bool) -> State {
    let mut state = State::default();

    let winner = state.side_one.get_active();
    winner.hp = 200;
    winner.maxhp = 300;
    winner.item = poke_engine::engine::items::Items::LEFTOVERS;
    winner.status = poke_engine::state::PokemonStatus::BURN;
    winner.speed = 50;
    winner.replace_move(PokemonMoveIndex::M0, Choices::SPLASH);

    let loser = state.side_two.get_active();
    loser.hp = 1;
    loser.maxhp = 300;
    loser.status = poke_engine::state::PokemonStatus::POISON;
    loser.speed = 200;
    loser.replace_move(PokemonMoveIndex::M0, Choices::SPLASH);

    for index in [
        PokemonIndex::P1,
        PokemonIndex::P2,
        PokemonIndex::P3,
        PokemonIndex::P4,
        PokemonIndex::P5,
    ] {
        state.side_one.pokemon[index].hp = 0;
        state.side_two.pokemon[index].hp = if reserves_alive { 100 } else { 0 };
    }
    state
}

/// The bug. Side two's last Pokemon dies to poison in the residual block, which
/// ends the battle — so EVERY side-one residual still queued behind it, Leftovers
/// heal included, never executes.
///
/// Transcribed from the real sim, not from this engine
/// (`scripts/gen3_switch_differential.py::residualwin`; probe protocol, slow
/// Leftovers+burn Snorlax vs faster toxic'd Aipom, final step):
///
/// ```text
/// |-damage|p2a: Aipom|0 fnt|[from] psn
/// |faint|p2a: Aipom
/// |win|PokeZero p1
/// ```
///
/// Nothing between `|faint|` and `|win|`, and no `|upkeep`: the winner's Leftovers
/// and its own burn tick are both gone. The two steps before it show the same pair
/// firing normally — `psn`, then `item: Leftovers`, then `brn` — so their absence
/// here is truncation and not a missing source.
#[test]
fn a_residual_faint_that_ends_the_battle_truncates_the_block() {
    let mut state = poison_kill_state(false);
    let list = generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    );

    assert_eq!(
        damages(&list, SideReference::SideTwo),
        vec![1],
        "the poison tick that ends it still resolves in full: {:?}",
        list
    );
    assert!(
        heals(&list, SideReference::SideOne).is_empty(),
        "the winner's Leftovers is order 10.4 — the same class as the fatal 10.6 \
         poison tick and behind it on speed — so it is truncated too: {:?}",
        list
    );
    assert!(
        damages(&list, SideReference::SideOne).is_empty(),
        "the winner's burn is queued AFTER the fatal entry and must never fire: {:?}",
        list
    );

    state.apply_instructions(&list);
    assert_eq!(
        state.battle_is_over(),
        1.0,
        "side two is wiped, so side one has won"
    );
}

/// The control, and the #876 non-regression: the identical residual faint with a
/// living reserve does NOT end the battle, so the block behaves exactly as before —
/// the deferral fires and a replacement is owed.
#[test]
fn a_non_final_residual_faint_still_defers() {
    let mut state = poison_kill_state(true);
    let list = generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    );

    assert_eq!(
        damages(&list, SideReference::SideTwo),
        vec![1],
        "the poison tick still lands: {:?}",
        list
    );
    assert_eq!(
        damages(&list, SideReference::SideOne),
        vec![37],
        "the block runs to completion when the battle continues — the burn queued \
         after the faint still fires: {:?}",
        list
    );
    assert_eq!(heals(&list, SideReference::SideOne), vec![18]);

    state.apply_instructions(&list);
    assert_eq!(
        state.battle_is_over(),
        0.0,
        "a side with a living reserve has not lost"
    );
    assert_eq!(
        state.side_two.get_active_immutable().hp,
        0,
        "and it does owe a replacement, which get_all_options offers off hp <= 0"
    );
}

/// The reasoning the entry guard rests on, pinned directly: the deferral and the
/// battle-over stop are MUTUALLY EXCLUSIVE. `end_of_turn_is_deferred` fires only
/// when a side owes a replacement, and a side can only owe one while it still has a
/// living Pokemon to send — so a faint that ends the battle can never also defer.
#[test]
fn deferral_and_battle_end_are_mutually_exclusive() {
    let mut deferring = poison_kill_state(true);
    let deferred = generate(
        &mut deferring,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    );
    deferring.apply_instructions(&deferred);

    let mut ending = poison_kill_state(false);
    let ended = generate(
        &mut ending,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    );
    ending.apply_instructions(&ended);

    // Continuing battle: a replacement is owed and the block was NOT truncated.
    assert_eq!(deferring.battle_is_over(), 0.0);
    assert_eq!(deferring.side_two.get_active_immutable().hp, 0);
    assert_eq!(damages(&deferred, SideReference::SideOne), vec![37]);

    // Ended battle: nobody owes a replacement, and the block WAS truncated. The two
    // conditions cannot both hold, which is why the entry guard and the deferral
    // cannot interact.
    assert_ne!(ending.battle_is_over(), 0.0);
    assert!(damages(&ended, SideReference::SideOne).is_empty());
    assert!(!ended.contains(&Instruction::ToggleSideTwoForceSwitch));
}

/// Neither side may take a residual after the battle ends — not just the winner.
/// Sand chips BOTH actives, so without truncation the winning side would tick too.
#[test]
fn no_side_takes_a_residual_after_the_battle_ends() {
    let mut state = poison_kill_state(false);
    state.weather = poke_engine::state::StateWeather {
        weather_type: poke_engine::engine::state::Weather::SAND,
        turns_remaining: 5,
    };
    // Neither active is Rock/Ground/Steel, so both would normally take sand.
    let list = generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    );

    state.apply_instructions(&list);
    assert_eq!(state.battle_is_over(), 1.0);
    assert!(
        damages(&list, SideReference::SideOne).is_empty(),
        "the winner takes no residual once the battle is over — not sand, not its \
         own burn: {:?}",
        list
    );
}

/// The #888 tie divergence is untouched by this change. A double wipe is still
/// terminal and still resolves the same (divergent) way — gen3 Showdown ties it,
/// the engine awards it to side two. Deliberately deferred there, and pinned here
/// so this fix cannot quietly move it.
#[test]
fn the_double_wipe_verdict_is_unchanged() {
    let mut state = State::default();
    for index in [
        PokemonIndex::P0,
        PokemonIndex::P1,
        PokemonIndex::P2,
        PokemonIndex::P3,
        PokemonIndex::P4,
        PokemonIndex::P5,
    ] {
        state.side_one.pokemon[index].hp = 0;
        state.side_two.pokemon[index].hp = 0;
    }
    assert_eq!(state.battle_is_over(), -1.0);
    let (side_one_options, side_two_options) = state.get_all_options();
    assert!(!side_one_options.is_empty());
    assert!(!side_two_options.is_empty());
}
