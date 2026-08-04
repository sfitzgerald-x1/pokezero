//! Gen 3 phazing (Whirlwind / Roar) fidelity pins, asserted directly against the
//! vendored gen3-patched poke-engine (`third_party/poke-engine-src/`).
//!
//! Every expectation here was read off the **real** Node Showdown simulator
//! driven through `scripts/gen3_switch_differential.py`, which is the
//! ground-truth gate; this file is the engine-contract pin.
//!
//! What it guards (`poke-engine-gen3-phaze-protect.patch`) and what it merely
//! records as already-correct:
//!
//! * Protect BLOCKS a phaze in gen3 — the fix. Gen 3 inherits gen4's override
//!   (`flags: { protect: 1, mirror: 1, bypasssub: 1, metronome: 1 }`), and
//!   upstream carried no protect flag at all, so Whirlwind dragged straight
//!   through a Protect.
//! * A Substitute does NOT block it (`bypasssub`), the fan-out is uniform over
//!   the alive reserve, entry hazards hit the dragged-in Pokemon, and the
//!   outgoing Pokemon's boosts are cleared. All four were already correct;
//!   they are pinned because the fix edits the same flag set they depend on.

use poke_engine::choices::Choices;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::{MoveChoice, PokemonVolatileStatus};
use poke_engine::instruction::{Instruction, StateInstructions};
use poke_engine::state::{PokemonIndex, PokemonMoveIndex, SideReference, State};
use pokezero_search::events::{render_branch_events, EventContext};

const PHAZE_MOVES: [Choices; 2] = [Choices::WHIRLWIND, Choices::ROAR];

fn generate(
    state: &mut State,
    side_one: &MoveChoice,
    side_two: &MoveChoice,
) -> Vec<StateInstructions> {
    let before = format!("{:?}", state);
    let instructions = generate_instructions_from_move_pair(state, side_one, side_two, false);
    assert_eq!(before, format!("{:?}", state), "state was mutated");
    for branch in &instructions {
        let mut probe = state.clone();
        let snapshot = format!("{:?}", probe);
        probe.apply_instructions(&branch.instruction_list);
        probe.reverse_instructions(&branch.instruction_list);
        assert_eq!(snapshot, format!("{:?}", probe), "branch did not revert");
    }
    instructions
}

fn switch_targets(instructions: &[StateInstructions]) -> Vec<PokemonIndex> {
    instructions
        .iter()
        .filter_map(|branch| {
            branch
                .instruction_list
                .iter()
                .find_map(|instruction| match instruction {
                    Instruction::Switch(switch) if switch.side_ref == SideReference::SideOne => {
                        Some(switch.next_index)
                    }
                    _ => None,
                })
        })
        .collect()
}

/// Side two phazes with `phaze_move` on M0; side one answers with `defender_move`.
fn phaze_state(phaze_move: Choices, defender_move: Choices) -> State {
    let mut state = State::default();
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, phaze_move);
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, defender_move);
    state
}

fn phaze(state: &mut State) -> Vec<StateInstructions> {
    generate(
        state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
    )
}

// ---------------------------------------------------------------------------
// The fix: Protect blocks a phaze
// ---------------------------------------------------------------------------

/// Showdown gen3 ground truth (`scripts/gen3_switch_differential.py::whirlwindprotect`):
/// `|move|p1a: Skarmory|Whirlwind|p2a: Snorlax` followed by
/// `|-activate|p2a: Snorlax|Protect` and NO `|drag|`.
///
/// Upstream gave the phazing moves no protect flag, so `before_move` never
/// reached `remove_effects_for_protect()` — which is what clears `flags.drag` —
/// and the target was dragged out through its own Protect. Protect is on 43 gen3
/// randbats species and the pool's Whirlwind user (Skarmory) carries both.
#[test]
fn protect_blocks_a_phaze() {
    for phaze_move in PHAZE_MOVES {
        let mut state = phaze_state(phaze_move, Choices::PROTECT);
        // The Protect user is slower than the -6 priority phaze either way, but
        // make the ordering explicit.
        state.side_one.get_active().speed = 500;
        let instructions = phaze(&mut state);
        assert!(
            switch_targets(&instructions).is_empty(),
            "{:?} must not drag through Protect: {:?}",
            phaze_move,
            instructions
        );
    }
}

// ---------------------------------------------------------------------------
// Already-correct behaviour, pinned because the fix edits the same flag set
// ---------------------------------------------------------------------------

/// The no-regression direction, and the reason the fix is one gated flag rather
/// than a guard in the drag path: an ORDINARY phaze turn must still drag. The
/// same `flags.protect` that `before_move` consults to block the move is the
/// flag every other phaze turn leaves alone, so a fix that over-reached here
/// would silently delete phazing from the format.
#[test]
fn an_unprotected_phaze_still_drags() {
    for phaze_move in PHAZE_MOVES {
        let mut state = phaze_state(phaze_move, Choices::SPLASH);
        let instructions = phaze(&mut state);
        assert_eq!(
            switch_targets(&instructions).len(),
            5,
            "{:?} must still drag when the target did not Protect: {:?}",
            phaze_move,
            instructions
        );
    }
}

/// Protect is a SINGLE-TURN volatile, so the turn after it lapses the phaze
/// connects again. Guards against the flag being read as a permanent immunity.
#[test]
fn a_phaze_connects_the_turn_after_protect_lapses() {
    let mut state = phaze_state(Choices::WHIRLWIND, Choices::PROTECT);
    state.side_one.get_active().speed = 500;

    let blocked = phaze(&mut state);
    assert!(switch_targets(&blocked).is_empty(), "{:?}", blocked);
    // Take the branch forward, then let the Protect expire.
    state.apply_instructions(&blocked[0].instruction_list);
    state
        .side_one
        .volatile_statuses
        .remove(&PokemonVolatileStatus::PROTECT);

    let connects = phaze(&mut state);
    assert_eq!(
        switch_targets(&connects).len(),
        5,
        "the phaze connects once Protect is gone: {:?}",
        connects
    );
}

/// `bypasssub` is in gen3's flag set, so a Substitute does NOT stop a phaze.
/// Verified against the sim: the drag line fires straight through the sub.
#[test]
fn a_substitute_does_not_block_a_phaze() {
    for phaze_move in PHAZE_MOVES {
        let mut state = phaze_state(phaze_move, Choices::SPLASH);
        state
            .side_one
            .volatile_statuses
            .insert(PokemonVolatileStatus::SUBSTITUTE);
        let instructions = phaze(&mut state);
        assert_eq!(
            switch_targets(&instructions).len(),
            5,
            "{:?} must drag through a Substitute: {:?}",
            phaze_move,
            instructions
        );
    }
}

/// Showdown picks the replacement with `this.sample(possibleSwitches)` — uniform
/// over the target's alive, non-active party members. The engine fans out one
/// branch per candidate at equal probability, which is the search-tree spelling
/// of the same thing.
#[test]
fn a_phaze_fans_out_uniformly_over_the_alive_reserve() {
    let mut state = phaze_state(Choices::WHIRLWIND, Choices::SPLASH);
    // Faint two of the five reserves: the fan-out must shrink to match.
    state.side_one.pokemon[PokemonIndex::P3].hp = 0;
    state.side_one.pokemon[PokemonIndex::P4].hp = 0;

    let instructions = phaze(&mut state);
    let targets = switch_targets(&instructions);
    assert_eq!(
        targets,
        vec![PokemonIndex::P1, PokemonIndex::P2, PokemonIndex::P5],
        "only living, non-active reserves are draggable: {:?}",
        instructions
    );
    for branch in &instructions {
        assert!(
            (branch.percentage - 100.0 / 3.0).abs() < 1e-3,
            "uniform over 3 candidates, got {}: {:?}",
            branch.percentage,
            instructions
        );
    }
}

/// The dragged-in Pokemon eats entry hazards, exactly as a chosen switch does —
/// this is the component the strict matcher reads as `spikes_entry_damage`.
#[test]
fn the_dragged_in_pokemon_takes_entry_hazards() {
    let mut state = phaze_state(Choices::WHIRLWIND, Choices::SPLASH);
    state.side_one.side_conditions.spikes = 1;
    for index in [
        PokemonIndex::P1,
        PokemonIndex::P2,
        PokemonIndex::P3,
        PokemonIndex::P4,
        PokemonIndex::P5,
    ] {
        let mon = &mut state.side_one.pokemon[index];
        mon.maxhp = 266;
        mon.hp = 266;
    }

    let instructions = phaze(&mut state);
    assert_eq!(switch_targets(&instructions).len(), 5);
    for branch in &instructions {
        let hazard = branch.instruction_list.iter().any(|instruction| {
            matches!(instruction, Instruction::Damage(damage)
                if damage.side_ref == SideReference::SideOne && damage.damage_amount == 33)
        });
        assert!(
            hazard,
            "every dragged-in Pokemon takes floor(266/8) = 33: {:?}",
            branch.instruction_list
        );
    }
}

/// A phaze is an ordinary switch-out for the Pokemon leaving, so its boosts go
/// with it (`Pokemon.clearVolatile()` on the way out).
#[test]
fn a_phaze_clears_the_outgoing_boosts() {
    let mut state = phaze_state(Choices::WHIRLWIND, Choices::SPLASH);
    state.side_one.attack_boost = 2;
    state.side_one.speed_boost = -1;

    let instructions = phaze(&mut state);
    for branch in &instructions {
        let mut probe = state.clone();
        probe.apply_instructions(&branch.instruction_list);
        assert_eq!(
            probe.side_one.attack_boost, 0,
            "{:?}",
            branch.instruction_list
        );
        assert_eq!(
            probe.side_one.speed_boost, 0,
            "{:?}",
            branch.instruction_list
        );
    }
}

/// With nothing left to drag in, the phaze simply does nothing — and must not
/// emit a switch to the active itself or to a fainted slot.
#[test]
fn a_phaze_with_no_living_reserve_is_a_no_op() {
    let mut state = phaze_state(Choices::WHIRLWIND, Choices::SPLASH);
    for index in [
        PokemonIndex::P1,
        PokemonIndex::P2,
        PokemonIndex::P3,
        PokemonIndex::P4,
        PokemonIndex::P5,
    ] {
        state.side_one.pokemon[index].hp = 0;
    }

    let instructions = phaze(&mut state);
    assert!(
        switch_targets(&instructions).is_empty(),
        "nothing to drag: {:?}",
        instructions
    );
}

// ---------------------------------------------------------------------------
// The RENDERED hazard chip must carry `[from] Spikes` (reports/c117 cause B1)
//
// `the_dragged_in_pokemon_takes_entry_hazards` above asserts the INSTRUCTION and
// never the rendered line, so it passed identically before and after the fix --
// a textbook M3 fixture sitting exactly where the defect was. Independent review
// of #1081 measured the whole crate suite at 362 passed BOTH WAYS with the 36-line
// renderer fix reverted: not one test distinguished the trees.
//
// This program has already diagnosed this exact mutation as invisible once, in
// tests/test_sleep_talk_phaze_drag.py, whose docstring records that deleting
// `dragged[side] = true` left all 25 tests green "because no fixture had Spikes".
// That file then wrote a positive and a negative pin. The same idiom now applies
// to render_move_phase, which had none.
// ---------------------------------------------------------------------------

fn render_phaze(state: &mut State, branch: &StateInstructions) -> Vec<String> {
    let before = state.serialize();
    let rendered = render_branch_events(
        state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &branch.instruction_list,
        false,
        &EventContext {
            species: [vec!["Victim".into()], vec!["Phazer".into()]],
            turn: 1,
            hp_percent: [false, false],
        },
    );
    assert_eq!(
        before,
        state.serialize(),
        "rendering mutated the source state"
    );
    rendered.lines
}

/// Sized so the chip is unambiguous: maxhp 266 at one layer is 266/8 = 33.
fn phaze_into_spikes(layers: i8) -> State {
    let mut state = phaze_state(Choices::WHIRLWIND, Choices::SPLASH);
    state.side_one.side_conditions.spikes = layers;
    for index in [
        PokemonIndex::P1,
        PokemonIndex::P2,
        PokemonIndex::P3,
        PokemonIndex::P4,
        PokemonIndex::P5,
    ] {
        let mon = &mut state.side_one.pokemon[index];
        mon.maxhp = 266;
        mon.hp = 266;
    }
    state
}

/// POSITIVE PIN. Full line equality, not `contains("|-damage|")`: the bug WAS a
/// missing `[from]` suffix, so a substring assertion would have passed through it.
#[test]
fn the_rendered_hazard_chip_is_tagged_from_spikes() {
    let mut state = phaze_into_spikes(1);
    let branches = phaze(&mut state);
    let mut checked = 0;
    for branch in &branches {
        let lines = render_phaze(&mut state, branch);
        let drag = lines.iter().position(|line| line.starts_with("|drag|"));
        let Some(drag_at) = drag else { continue };
        let incoming = lines[drag_at]
            .split('|')
            .nth(2)
            .expect("drag line carries an ident")
            .to_string();
        let chip = lines
            .iter()
            .skip(drag_at + 1)
            .find(|line| line.starts_with("|-damage|"))
            .unwrap_or_else(|| panic!("no chip after the drag in {lines:?}"));
        assert_eq!(
            *chip,
            format!("|-damage|{incoming}|233/266|[from] Spikes"),
            "the chip on the dragged-in Pokemon must be tagged `[from] Spikes`, \
             byte for byte. An untagged `|-damage|` is filed roll-scaled by the \
             differential while Showdown files it as an exact (spikes, -n) \
             component, so the two never compare -- 11 holdout rows and 1 dev row.",
        );
        checked += 1;
    }
    assert_eq!(checked, 5, "expected five drag targets to check");
}

/// CONTROL. Without it the positive pin cannot fail for the right reason: it
/// proves the tag tracks the hazard rather than being emitted unconditionally.
#[test]
fn with_no_spikes_the_drag_emits_no_chip_and_no_spikes_tag() {
    let mut state = phaze_into_spikes(0);
    let branches = phaze(&mut state);
    for branch in &branches {
        let lines = render_phaze(&mut state, branch);
        assert!(
            !lines.iter().any(|line| line.contains("[from] Spikes")),
            "no Spikes are set, so nothing may claim to be Spikes: {lines:?}",
        );
        if let Some(drag_at) = lines.iter().position(|line| line.starts_with("|drag|")) {
            assert!(
                !lines
                    .iter()
                    .skip(drag_at + 1)
                    .any(|line| line.starts_with("|-damage|")),
                "an unhazarded drag must chip nothing: {lines:?}",
            );
        }
    }
}

/// TWO LAYERS. This is the case the holdout actually hit (Victreebel, 45 at two
/// layers), and it pins magnitude and tag together: 266 * 4 / 24 = 44.
#[test]
fn two_spikes_layers_chip_the_deeper_amount_and_stay_tagged() {
    let mut state = phaze_into_spikes(2);
    let branches = phaze(&mut state);
    let mut checked = 0;
    for branch in &branches {
        let lines = render_phaze(&mut state, branch);
        let Some(drag_at) = lines.iter().position(|line| line.starts_with("|drag|")) else {
            continue;
        };
        let chip = lines
            .iter()
            .skip(drag_at + 1)
            .find(|line| line.starts_with("|-damage|"))
            .unwrap_or_else(|| panic!("no chip after the drag in {lines:?}"));
        assert!(
            chip.ends_with("|222/266|[from] Spikes"),
            "two layers must chip 266*4/24 = 44 AND stay tagged, got {chip}",
        );
        checked += 1;
    }
    assert_eq!(checked, 5);
}

/// ARM ORDER, pinned in source text because it cannot be pinned behaviourally in
/// gen3: `dragged[attacker]` is unreachable (`get_instructions_from_drag` always
/// switches the defender and returns immediately), and no gen3 phazing move deals
/// damage. But the fix depends on the new arm PRECEDING the generic
/// `damage.side_ref == defender` arm -- move it after and the fix silently
/// reverts with every test green. Same idiom as test_crit_kill_split_patch's
/// static pins.
#[test]
fn the_dragged_arm_precedes_the_generic_defender_damage_arm() {
    let source = include_str!("../src/events.rs");
    let dragged_arm = source
        .find("if dragged_this_phase[side_usize(damage.side_ref)]")
        .expect("the dragged-chip arm must exist");
    // Searched from the START of the file, not from `dragged_arm`. Searching the
    // suffix made the assertion below DEAD -- adding `dragged_arm` back guarantees
    // `generic_arm >= dragged_arm`, so only the `.expect` could ever fire. An
    // assertion that cannot fail inside an M3 pin is the wrong shape. Both search
    // strings are unique in events.rs, so two independent finds make the compare
    // load-bearing. Found by review of #1081.
    let generic_arm = source
        .find("Instruction::Damage(damage) if damage.side_ref == defender")
        .expect("the generic defender-damage arm must exist");
    assert!(
        dragged_arm < generic_arm,
        "the dragged-chip arm must come FIRST or the generic arm shadows it and \
         the chip renders untagged again",
    );
}
