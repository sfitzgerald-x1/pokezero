//! Gen 2-4: a Pursuit KO does NOT cancel the switch it interrupted.
//!
//! C134 §4, the queue-semantics fixture pack. Pursuit is the one case where a switch
//! is the SECOND action of a ply without being a double switch, and it is the case
//! that broke two consecutive faint-cancellation patches.
//!
//! Showdown states the rule in its own source, `sim/battle.ts:2790-2794`:
//!
//! ```text
//! if (this.actions.switchIn(action.target, action.pokemon.position, ...) === 'pursuitfaint') {
//!     // a pokemon fainted from Pursuit before it could switch
//!     if (this.gen <= 4) {
//!         // in gen 2-4, the switch still happens
//!         this.hint("Previously chosen switches continue in Gen 2-4 after a Pursuit target faints.");
//!         action.priority = -101;
//!         this.queue.unshift(action);
//! ```
//!
//! The switch is **re-queued**, not run in place, so ORDER is the mechanism and these
//! tests assert the whole instruction list rather than mere membership. Ground truth
//! confirmed against real gen3 Showdown on four seeds by
//! `scripts/gen3_switch_differential.py`, scenarios `pursuitkoswitcher` (landmarked on
//! that hint line, so it cannot pass without taking the gen<=4 branch) and
//! `pursuitnokocontrol`.
//!
//! WHY THIS PIN EXISTS. A faint-cancellation guard keyed on "an active newly reached
//! 0 HP" gets this exactly backwards: the Pokemon that fainted IS the switcher, and
//! gen 3 continues its switch anyway. Cancelling leaves the wrong Pokemon active and
//! the opponent's residual tick cannot be attributed — the observed signature of the
//! two rows the withdrawn v2 guard opened (`19000120` dev, `19100078` holdout, both
//! `component_missing_in_engine:itemleftovers`). This pin is GREEN on `main`, which has
//! no cancellation at all, and RED against that v2 guard. It is a constraint on any
//! future v3, not a description of a bug in `main`.
//!
//! The hunter holds Leftovers deliberately. Without an item there is no `Heal` in the
//! branch, and the pin would assert the switch while saying nothing about the residual
//! tick it is named for — so a v3 that performed the switch but dropped the tick, or
//! that additionally emitted a spurious `ToggleSideOneForceSwitch`, would pass. The
//! exact-list assertions below close both.

use poke_engine::choices::Choices;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::items::Items;
use poke_engine::engine::state::MoveChoice;
use poke_engine::instruction::StateInstructions;
use poke_engine::state::{PokemonIndex, PokemonMoveIndex, State};

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

/// The one deterministic branch, rendered as debug strings so a mismatch prints the
/// whole list rather than a bare `false`.
fn only_branch(branches: Vec<StateInstructions>) -> Vec<String> {
    assert_eq!(
        branches.len(),
        1,
        "expected a single deterministic branch, got {:?}",
        branches
    );
    branches[0]
        .instruction_list
        .iter()
        .map(|instruction| format!("{:?}", instruction))
        .collect()
}

/// Side one switches out while side two attacks it with `move_id`.
///
/// Note there is deliberately NO speed manipulation. Pursuit intercepts a switching
/// target regardless of speed order, which `an_ordinary_move_does_not_intercept_the_switch`
/// below demonstrates using the identical state and differing only in the move. An
/// earlier revision set speeds and claimed they were what made the switch the second
/// action; they were inert, and the claim was wrong.
fn switch_into(move_id: Choices, outgoing_hp: i16) -> Vec<String> {
    let mut state = State::default();

    let hunter = &mut state.side_two.pokemon[PokemonIndex::P0];
    hunter.replace_move(PokemonMoveIndex::M0, move_id);
    hunter.item = Items::LEFTOVERS;
    hunter.maxhp = 300;
    hunter.hp = 100; // damaged, so Leftovers actually emits a Heal

    let outgoing = &mut state.side_one.pokemon[PokemonIndex::P0];
    outgoing.maxhp = 300;
    outgoing.hp = outgoing_hp;

    only_branch(generate(
        &mut state,
        &MoveChoice::Switch(PokemonIndex::P1),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    ))
}

/// The outgoing Pokemon is at 1 HP and Pursuit deals exactly 1, so the hit is provably
/// lethal — the damage equals the target's entire HP. The switch must still happen,
/// and the hunter's Leftovers must still tick after it.
#[test]
fn a_pursuit_ko_does_not_cancel_the_switch_it_interrupted() {
    assert_eq!(
        switch_into(Choices::PURSUIT, 1),
        vec![
            "Damage SideOne: 1".to_string(),
            "Switch SideOne: P0 -> P1".to_string(),
            "Heal SideTwo: 18".to_string(),
        ],
        "gen 2-4 continues a switch whose Pursuit target fainted \
         (sim/battle.ts:2790-2794). Cancelling it yields \
         [Damage SideOne: 1, ToggleSideOneForceSwitch]: the switch suppressed, the \
         opponent's residual tick dropped, and a replacement wrongly owed."
    );
}

/// Control: the identical ply with the target at full HP. Pursuit connects but cannot
/// KO, and the list is the same except for the damage. This is what makes the KO test
/// meaningful — the ONLY difference between the two is the faint.
#[test]
fn a_pursuit_that_does_not_ko_also_switches() {
    assert_eq!(
        switch_into(Choices::PURSUIT, 300),
        vec![
            "Damage SideOne: 63".to_string(),
            "Switch SideOne: P0 -> P1".to_string(),
            "Heal SideTwo: 18".to_string(),
        ],
        "a survived Pursuit must never suppress the switch"
    );
}

/// Control that the fixture shape is genuinely Pursuit's. An ordinary move does NOT
/// intercept the switch: the switch resolves first and the damage lands on the
/// REPLACEMENT, so `Switch` precedes `Damage`. If this ever matched the Pursuit
/// ordering, the two tests above would be pinning the generic switch path rather than
/// Pursuit's interception, and would prove nothing about the mechanism.
#[test]
fn an_ordinary_move_does_not_intercept_the_switch() {
    let list = switch_into(Choices::TACKLE, 300);
    let switch_at = list.iter().position(|i| i.starts_with("Switch SideOne"));
    let damage_at = list.iter().position(|i| i.starts_with("Damage SideOne"));
    assert!(
        switch_at.is_some() && damage_at.is_some(),
        "expected both a switch and damage: {:?}",
        list
    );
    assert!(
        switch_at < damage_at,
        "an ordinary move hits the replacement AFTER the switch; Pursuit is the \
         exception that hits before it: {:?}",
        list
    );
}
