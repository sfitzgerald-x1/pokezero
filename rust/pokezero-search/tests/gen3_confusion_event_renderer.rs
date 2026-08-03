//! Event-renderer controls for Gen 3's pre-move confusion check.
//!
//! The engine collapses equal instruction deltas. A `duration +1, Damage(user)`
//! tail is therefore only a confusion self-hit when its fixed 40-power damage
//! does not collide with a real move's crash or self-faint path. The renderer
//! must prove that identity without relying on optional PP/last-move deltas.

use poke_engine::choices::Choices;
use poke_engine::engine::abilities::Abilities;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::items::Items;
use poke_engine::engine::state::{MoveChoice, PokemonVolatileStatus};
use poke_engine::instruction::{Instruction, StateInstructions};
use poke_engine::state::{
    PokemonIndex, PokemonMoveIndex, PokemonStatus, PokemonType, SideReference, State,
};
use pokezero_search::events::{attribution_unsafe_label, render_branch_events, EventContext};

fn confused_state(move_id: Choices) -> State {
    let mut state = State::default();
    state.side_one.get_active().speed = 500;
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::SPLASH);
    state.side_two.get_active().speed = 1;
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, move_id);
    state
        .side_two
        .volatile_statuses
        .insert(PokemonVolatileStatus::CONFUSION);
    state
}

fn generate(state: &mut State) -> Vec<StateInstructions> {
    let before = format!("{state:?}");
    let branches = generate_instructions_from_move_pair(
        state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
        false,
    );
    assert_eq!(
        before,
        format!("{state:?}"),
        "generation mutated the source state"
    );
    branches
}

fn render(state: &mut State, branch: &StateInstructions) -> String {
    let before = state.serialize();
    let rendered = render_branch_events(
        state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &branch.instruction_list,
        false,
        &EventContext {
            species: [vec!["Lead".into()], vec!["Opponent".into()]],
            turn: 1,
            hp_percent: [false, false],
        },
    );
    assert_eq!(
        before,
        state.serialize(),
        "rendering mutated the source state"
    );
    rendered.lines.join("\n")
}

fn rendered(
    state: &mut State,
    branch: &StateInstructions,
) -> pokezero_search::events::RenderedEvents {
    let before = state.serialize();
    let rendered = render_branch_events(
        state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &branch.instruction_list,
        false,
        &EventContext {
            species: [vec!["Lead".into()], vec!["Opponent".into()]],
            turn: 1,
            hp_percent: [false, false],
        },
    );
    assert_eq!(
        before,
        state.serialize(),
        "rendering mutated the source state"
    );
    rendered
}

fn damage_to(branch: &StateInstructions, side: SideReference, amount: i16) -> bool {
    branch.instruction_list.iter().any(|instruction| {
        matches!(instruction, Instruction::Damage(damage)
            if damage.side_ref == side && damage.damage_amount == amount)
    })
}

/// Mirrors `CONFUSION_SNAP_OUT_PENDING` in the engine.
const CONFUSION_SNAP_OUT_PENDING: i8 = -4;

fn expires_confusion(branch: &StateInstructions) -> bool {
    branch.instruction_list.iter().any(|instruction| {
        matches!(instruction, Instruction::RemoveVolatileStatus(remove)
            if remove.side_ref == SideReference::SideTwo
                && remove.volatile_status == PokemonVolatileStatus::CONFUSION)
    })
}

/// The end-of-turn ladder's snap-out, which now parks the counter on the
/// sentinel rather than removing the volatile. `expires_confusion` is kept
/// separate and still means an outright removal, so the tests below can say
/// which of the two they mean.
fn parks_snap_out(branch: &StateInstructions) -> bool {
    branch.instruction_list.iter().any(|instruction| {
        matches!(instruction, Instruction::ChangeVolatileStatusDuration(change)
            if change.side_ref == SideReference::SideTwo
                && change.volatile_status == PokemonVolatileStatus::CONFUSION
                && change.amount <= CONFUSION_SNAP_OUT_PENDING)
    })
}

fn self_hit_branch(branches: &[StateInstructions], amount: i16) -> &StateInstructions {
    branches
        .iter()
        .find(|branch| damage_to(branch, SideReference::SideTwo, amount))
        .expect("expected confusion self-hit branch")
}

fn assert_in_order(events: &str, lines: &[&str]) {
    let mut cursor = 0;
    for line in lines {
        let offset = events[cursor..]
            .find(line)
            .unwrap_or_else(|| panic!("missing {line:?} in {events:?}"));
        cursor += offset + line.len();
    }
}

#[test]
fn exact_self_hit_renders_activation_and_cancels_substitute() {
    let mut state = confused_state(Choices::SUBSTITUTE);
    state.side_two.get_active().maxhp = 256;
    state.side_two.get_active().hp = 200;
    state.side_two.get_active().attack = 108; // exact 38 damage
    let branches = generate(&mut state);
    let events = render(&mut state, self_hit_branch(&branches, 38));
    assert!(
        events.contains("|-activate|p2a: Opponent|confusion"),
        "{events}"
    );
    assert!(
        events.contains("|-damage|p2a: Opponent|162/256"),
        "{events}"
    );
    assert!(
        !events.contains("|move|p2a: Opponent|substitute"),
        "{events}"
    );
}

#[test]
fn switch_prefixed_exact_self_hit_is_untagged_and_safe() {
    let mut state = confused_state(Choices::SUBSTITUTE);
    state.side_one.pokemon[PokemonIndex::P1] = state.side_one.get_active_immutable().clone();
    state.side_two.get_active().maxhp = 256;
    state.side_two.get_active().hp = 200;
    state.side_two.get_active().attack = 108; // exact 38 confusion damage

    let side_one = MoveChoice::Switch(PokemonIndex::P1);
    let side_two = MoveChoice::Move(PokemonMoveIndex::M0);
    let before = state.serialize();
    let branches = generate_instructions_from_move_pair(&mut state, &side_one, &side_two, false);
    assert_eq!(
        before,
        state.serialize(),
        "generation mutated the source state"
    );
    let branch = branches
        .iter()
        .find(|branch| {
            damage_to(branch, SideReference::SideTwo, 38)
                && branch.instruction_list.iter().any(|instruction| {
                    matches!(instruction, Instruction::Switch(switch)
                        if switch.side_ref == SideReference::SideOne
                            && switch.next_index == PokemonIndex::P1)
                })
        })
        .expect("expected switch-prefixed exact confusion self-hit branch");

    let rendered = render_branch_events(
        &mut state,
        &side_one,
        &side_two,
        &branch.instruction_list,
        false,
        &EventContext {
            species: [vec!["Lead".into(), "Bench".into()], vec!["Opponent".into()]],
            turn: 1,
            hp_percent: [false, false],
        },
    );
    assert_eq!(
        before,
        state.serialize(),
        "rendering mutated the source state"
    );
    let events = rendered.lines.join("\n");
    let activation = "|-activate|p2a: Opponent|confusion";
    let exact_damage = "|-damage|p2a: Opponent|162/256";
    assert_in_order(&events, &["|switch|p1a: Bench", activation, exact_damage]);
    assert!(
        rendered
            .lines
            .windows(2)
            .any(|pair| pair[0] == activation && pair[1] == exact_damage),
        "activation and exact damage must share one rendered confusion arm: {events}"
    );
    assert!(
        !events.contains(&format!("{exact_damage}|[from]")),
        "{events}"
    );
    assert!(rendered.attribution_unsafe.is_empty(), "{rendered:?}");
    assert!(rendered.lossy.is_empty(), "{rendered:?}");
}

#[test]
fn destiny_bond_bookkeeping_precedes_confusion_without_phantom_moves() {
    for move_id in [Choices::SUBSTITUTE, Choices::BELLYDRUM, Choices::CURSE] {
        let mut state = confused_state(move_id);
        state
            .side_two
            .volatile_statuses
            .insert(PokemonVolatileStatus::DESTINYBOND);
        let branches = generate(&mut state);
        let self_hit = self_hit_branch(&branches, 35);
        let destiny_index = self_hit
            .instruction_list
            .iter()
            .position(|instruction| {
                matches!(instruction, Instruction::RemoveVolatileStatus(remove)
                    if remove.side_ref == SideReference::SideTwo
                        && remove.volatile_status == PokemonVolatileStatus::DESTINYBOND)
            })
            .expect("expected Destiny Bond cleanup");
        let confusion_index = self_hit
            .instruction_list
            .iter()
            .position(|instruction| {
                matches!(instruction, Instruction::ChangeVolatileStatusDuration(change)
                    if change.side_ref == SideReference::SideTwo
                        && change.volatile_status == PokemonVolatileStatus::CONFUSION
                        && change.amount == 1)
            })
            .expect("expected confusion check");
        assert!(destiny_index < confusion_index, "{self_hit:?}");

        let events = render(&mut state, self_hit);
        assert!(
            events.contains("|-activate|p2a: Opponent|confusion"),
            "{events}"
        );
        assert!(
            !events.contains(&format!(
                "|move|p2a: Opponent|{}",
                format!("{move_id:?}").to_lowercase()
            )),
            "{events}"
        );
    }
}

#[test]
fn choice_band_locked_move_and_future_sight_prefixes_reach_confusion() {
    let mut choice_band = confused_state(Choices::TACKLE);
    choice_band.side_two.get_active().item = Items::CHOICEBAND;
    choice_band
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::SPLASH);
    let choice_band_branches = generate(&mut choice_band);
    let choice_band_hit = self_hit_branch(&choice_band_branches, 35);
    assert!(
        choice_band_hit
            .instruction_list
            .iter()
            .position(|instruction| matches!(instruction, Instruction::DisableMove(_)))
            < choice_band_hit
                .instruction_list
                .iter()
                .position(|instruction| {
                    matches!(instruction, Instruction::ChangeVolatileStatusDuration(change)
                    if change.volatile_status == PokemonVolatileStatus::CONFUSION
                        && change.amount == 1)
                }),
        "{choice_band_hit:?}"
    );
    let choice_band_events = render(&mut choice_band, choice_band_hit);
    assert!(
        !choice_band_events.contains("|move|p2a: Opponent|tackle"),
        "{choice_band_events}"
    );

    let mut locked = confused_state(Choices::OUTRAGE);
    locked
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::SPLASH);
    let locked_branches = generate(&mut locked);
    let locked_hit = self_hit_branch(&locked_branches, 35);
    assert!(
        locked_hit
            .instruction_list
            .iter()
            .any(|instruction| matches!(instruction, Instruction::DisableMove(_))),
        "{locked_hit:?}"
    );
    let locked_events = render(&mut locked, locked_hit);
    assert!(
        !locked_events.contains("|move|p2a: Opponent|outrage"),
        "{locked_events}"
    );

    let mut future_sight = confused_state(Choices::FUTURESIGHT);
    let future_branches = generate(&mut future_sight);
    let future_hit = self_hit_branch(&future_branches, 35);
    let future_index = future_hit
        .instruction_list
        .iter()
        .position(|instruction| matches!(instruction, Instruction::SetFutureSight(_)))
        .expect("expected Future Sight bookkeeping");
    let confusion_index = future_hit
        .instruction_list
        .iter()
        .position(|instruction| {
            matches!(instruction, Instruction::ChangeVolatileStatusDuration(change)
                if change.volatile_status == PokemonVolatileStatus::CONFUSION
                    && change.amount == 1)
        })
        .expect("expected confusion marker");
    assert!(future_index < confusion_index, "{future_hit:?}");
    let future_events = render(&mut future_sight, future_hit);
    assert!(
        !future_events.contains("|move|p2a: Opponent|futuresight"),
        "{future_events}"
    );
}

#[test]
fn crash_miss_remains_a_move_not_a_confusion_self_hit() {
    let mut state = confused_state(Choices::HIGHJUMPKICK);
    let branches = generate(&mut state);
    let crash = branches
        .iter()
        .find(|branch| damage_to(branch, SideReference::SideTwo, 50))
        .expect("expected High Jump Kick crash branch");
    let events = render(&mut state, crash);
    assert!(
        events.contains("|move|p2a: Opponent|highjumpkick|p1a: Lead|[miss]"),
        "{events}"
    );
    assert!(
        events.contains("|-damage|p2a: Opponent|50/100|[from] highjumpkick"),
        "{events}"
    );
    assert!(
        events.contains("|-activate|p2a: Opponent|confusion"),
        "{events}"
    );
}

#[test]
fn recoil_after_an_executed_move_is_not_confusion_damage() {
    let mut state = confused_state(Choices::DOUBLEEDGE);
    let branches = generate(&mut state);
    let recoil = branches
        .iter()
        .find(|branch| {
            damage_to(branch, SideReference::SideOne, 100)
                && branch.instruction_list.iter().any(|instruction| {
                    matches!(instruction, Instruction::Damage(damage)
                        if damage.side_ref == SideReference::SideTwo)
                })
        })
        .expect("expected executed Double-Edge branch");
    let events = render(&mut state, recoil);
    assert!(
        events.contains("|move|p2a: Opponent|doubleedge|p1a: Lead"),
        "{events}"
    );
    assert!(events.contains("[from] Recoil"), "{events}");
    assert!(
        events.contains("|-activate|p2a: Opponent|confusion"),
        "{events}"
    );
}

#[test]
fn explosion_behind_protect_is_not_misrendered_as_confusion() {
    let mut state = confused_state(Choices::EXPLOSION);
    state.side_two.get_active().speed = 500;
    state.side_one.get_active().speed = 1;
    state
        .side_one
        .volatile_statuses
        .insert(PokemonVolatileStatus::PROTECT);
    let branches = generate(&mut state);
    let explosion = branches
        .iter()
        .find(|branch| damage_to(branch, SideReference::SideTwo, 100))
        .expect("expected executing Explosion branch");
    let events = render(&mut state, explosion);
    assert!(
        events.contains("|move|p2a: Opponent|explosion|p1a: Lead"),
        "{events}"
    );
    assert!(events.contains("|-activate|p1a: Lead|Protect"), "{events}");
    assert!(events.contains("|faint|p2a: Opponent"), "{events}");
    assert!(
        events.contains("|-activate|p2a: Opponent|confusion"),
        "{events}"
    );
}

#[test]
fn explosion_into_an_immune_target_is_not_misrendered_as_confusion() {
    let mut state = confused_state(Choices::EXPLOSION);
    state.side_two.get_active().speed = 500;
    state.side_one.get_active().speed = 1;
    state.side_one.get_active().types.0 = PokemonType::GHOST;
    let branches = generate(&mut state);
    let explosion = branches
        .iter()
        .find(|branch| damage_to(branch, SideReference::SideTwo, 100))
        .expect("expected executing Explosion branch");
    let events = render(&mut state, explosion);
    assert!(
        events.contains("|move|p2a: Opponent|explosion|p1a: Lead"),
        "{events}"
    );
    assert!(events.contains("|-immune|p1a: Lead"), "{events}");
    assert!(events.contains("|faint|p2a: Opponent"), "{events}");
    assert!(
        events.contains("|-activate|p2a: Opponent|confusion"),
        "{events}"
    );
}

#[test]
fn lethal_self_hit_still_emits_confusion_then_faint() {
    let mut state = confused_state(Choices::SUBSTITUTE);
    state.side_two.get_active().hp = 20;
    let branches = generate(&mut state);
    let events = render(&mut state, self_hit_branch(&branches, 20));
    assert!(
        events.contains("|-activate|p2a: Opponent|confusion"),
        "{events}"
    );
    assert!(events.contains("|-damage|p2a: Opponent|0 fnt"), "{events}");
    assert!(events.contains("|faint|p2a: Opponent"), "{events}");
    assert!(
        !events.contains("|move|p2a: Opponent|substitute"),
        "{events}"
    );
}

#[test]
fn confusion_expiry_is_never_emitted_before_showdown_can_observe_it() {
    for previous_turns in 0..=3 {
        let mut state = confused_state(Choices::SPLASH);
        state.side_two.volatile_status_durations.confusion = previous_turns;
        let branches = generate(&mut state);
        let expires = branches
            .iter()
            .find(|branch| {
                parks_snap_out(branch)
                    && branch.instruction_list.iter().any(|instruction| {
                        matches!(instruction, Instruction::ChangeVolatileStatusDuration(change)
                            if change.side_ref == SideReference::SideTwo
                                && change.volatile_status == PokemonVolatileStatus::CONFUSION
                                && change.amount
                                    == CONFUSION_SNAP_OUT_PENDING - (previous_turns + 1))
                    })
            })
            .expect("expected residual snap-out branch");
        // Still the original invariant: nothing announces the end here.
        let expire_rendered = rendered(&mut state, expires);
        let expire_events = expire_rendered.lines.join("\n");
        assert!(
            !expire_events.contains("|-end|p2a: Opponent|confusion"),
            "{expire_events}"
        );
        assert!(
            !expires_confusion(expires),
            "the ladder must park the snap-out, not apply it: {:?}",
            expires.instruction_list
        );
        // ...but it is no longer bought with a rejected branch. The engine now
        // holds the volatile until the next move, so the renderer has nothing
        // early to suppress and this line is fully attributable. This is the
        // whole point of the deferral: `confusion_expiry_timing_unobservable`
        // was 35% of the clean-band world-construction refusals, and a refused
        // world is a decision that falls back to raw play.
        assert!(
            !expire_rendered
                .attribution_unsafe
                .iter()
                .any(|reason| reason == "confusion_expiry_timing_unobservable"),
            "the deferral must remove the refusal, not relocate it: {expire_rendered:?}"
        );
    }

    let mut state = confused_state(Choices::SPLASH);
    let branches = generate(&mut state);
    let survives = branches
        .iter()
        .find(|branch| damage_to(branch, SideReference::SideTwo, 35) && !expires_confusion(branch))
        .expect("expected surviving self-hit");
    let survive_events = rendered(&mut state, survives).lines.join("\n");
    assert!(
        !survive_events.contains("|-end|p2a: Opponent|confusion"),
        "{survive_events}"
    );
}

#[test]
fn segmentation_fallback_cannot_leak_confusion_expiry() {
    let mut state = confused_state(Choices::SPLASH);
    state.side_two.volatile_status_durations.confusion = 3;
    state.side_one.pokemon[PokemonIndex::P1] = state.side_one.get_active_immutable().clone();
    let branches = generate(&mut state);
    let expires = branches
        .iter()
        .find(|branch| parks_snap_out(branch))
        .expect("expected residual snap-out branch");
    let before = state.serialize();
    // Deliberately supply a switch-first public shape instead of the branch so
    // segmentation fails. Its diagnostic path must not surface the engine's
    // end-of-turn expiry before Showdown can announce it on a later move.
    let rendered = render_branch_events(
        &mut state,
        &MoveChoice::Switch(PokemonIndex::P1),
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &expires.instruction_list,
        false,
        &EventContext {
            species: [vec!["Lead".into()], vec!["Opponent".into()]],
            turn: 1,
            hp_percent: [false, false],
        },
    );
    assert_eq!(
        before,
        state.serialize(),
        "rendering mutated the source state"
    );
    assert!(
        rendered
            .attribution_unsafe
            .iter()
            .any(|reason| reason == "segmentation_failed"),
        "{rendered:?}"
    );
    assert!(
        !rendered
            .lines
            .iter()
            .any(|line| line == "|-end|p2a: Opponent|confusion"),
        "{rendered:?}"
    );
}

#[test]
fn survival_branch_announces_confusion_before_the_move() {
    let mut state = confused_state(Choices::SUBSTITUTE);
    let branches = generate(&mut state);
    let substitute = branches
        .iter()
        .find(|branch| {
            branch.instruction_list.iter().any(|instruction| {
                matches!(instruction, Instruction::ApplyVolatileStatus(apply)
                    if apply.side_ref == SideReference::SideTwo
                        && apply.volatile_status == PokemonVolatileStatus::SUBSTITUTE)
            })
        })
        .expect("expected move-through branch");
    let events = render(&mut state, substitute);
    let activation = events
        .find("|-activate|p2a: Opponent|confusion")
        .expect("activation");
    let selected_move = events.find("|move|p2a: Opponent|substitute").expect("move");
    assert!(activation < selected_move, "{events}");
}

#[test]
fn confusion_survives_then_attract_blocks_without_a_move_window() {
    let mut state = confused_state(Choices::SUBSTITUTE);
    state
        .side_two
        .volatile_statuses
        .insert(PokemonVolatileStatus::ATTRACT);
    let branches = generate(&mut state);
    let attract_blocked = branches
        .iter()
        .find(|branch| {
            branch.instruction_list.iter().any(|instruction| {
                matches!(instruction, Instruction::ChangeVolatileStatusDuration(change)
                    if change.side_ref == SideReference::SideTwo
                        && change.volatile_status == PokemonVolatileStatus::CONFUSION
                        && change.amount == 1)
            }) && !branch.instruction_list.iter().any(|instruction| {
                matches!(instruction, Instruction::Damage(damage)
                    if damage.side_ref == SideReference::SideTwo)
                    || matches!(instruction, Instruction::ApplyVolatileStatus(apply)
                        if apply.side_ref == SideReference::SideTwo
                            && apply.volatile_status == PokemonVolatileStatus::SUBSTITUTE)
            })
        })
        .expect("expected confusion-survives Attract immobilization branch");
    let rendered = rendered(&mut state, attract_blocked);
    let events = rendered.lines.join("\n");
    assert!(
        events.contains("|-activate|p2a: Opponent|confusion"),
        "{events}"
    );
    assert!(events.contains("|cant|p2a: Opponent|Attract"), "{events}");
    assert!(
        !events.contains("|-activate|p2a: Opponent|move: Attract"),
        "{events}"
    );
    assert!(
        !events.contains("|move|p2a: Opponent|substitute"),
        "{events}"
    );
    assert!(
        rendered
            .lossy
            .iter()
            .any(|reason| reason == "attract_immobilization_source_unknown"),
        "{rendered:?}"
    );
}

#[test]
fn still_asleep_sleep_talk_runs_confusion_before_the_called_move() {
    // Real Showdown corpus evidence, including Rest-origin sleep: the public
    // sleep gate appears before Sleep Talk and the called move.
    let rest_corpus = include_str!("../../../tests/fixtures/showdown/tier2-cb-pidgeot-game.log");
    assert_in_order(
        rest_corpus,
        &[
            "|cant|p1a: Regice|slp",
            "|move|p1a: Regice|Sleep Talk|p1a: Regice",
            "|move|p1a: Regice|Rest|p1a: Regice|[from] Sleep Talk",
        ],
    );
    let corpus = include_str!(
        "../../../tests/fixtures/showdown/capture/lines-battle-gen3randombattle-controlled-20260710001.log"
    );
    assert_in_order(
        corpus,
        &[
            "|cant|p2a: Articuno|slp",
            "|move|p2a: Articuno|Sleep Talk|p2a: Articuno",
            "|move|p2a: Articuno|Hidden Power|p1a: Ledian|[from] Sleep Talk",
        ],
    );

    let mut state = confused_state(Choices::SLEEPTALK);
    state.side_two.get_active().status = PokemonStatus::SLEEP;
    state.side_two.get_active().sleep_turns = 0;
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::SUBSTITUTE);
    let branches = generate(&mut state);

    let self_hit = self_hit_branch(&branches, 35);
    let self_hit_events = render(&mut state, self_hit);
    assert!(
        self_hit_events.contains("|-activate|p2a: Opponent|confusion"),
        "{self_hit_events}"
    );
    assert_in_order(
        &self_hit_events,
        &[
            "|cant|p2a: Opponent|slp",
            "|-activate|p2a: Opponent|confusion",
        ],
    );
    assert!(
        !self_hit_events.contains("|move|p2a: Opponent|sleeptalk"),
        "{self_hit_events}"
    );

    let called = branches
        .iter()
        .find(|branch| {
            branch.instruction_list.iter().any(|instruction| {
                matches!(instruction, Instruction::ApplyVolatileStatus(apply)
                    if apply.side_ref == SideReference::SideTwo
                        && apply.volatile_status == PokemonVolatileStatus::SUBSTITUTE)
            })
        })
        .expect("expected Sleep Talk call branch");
    let called_render = rendered(&mut state, called);
    let called_events = called_render.lines.join("\n");
    assert_eq!(
        called_render.lossy,
        Vec::<String>::new(),
        "{called_render:?}"
    );
    assert!(
        called_render.attribution_unsafe.is_empty(),
        "{called_render:?}"
    );
    assert_in_order(
        &called_events,
        &[
            "|cant|p2a: Opponent|slp",
            "|-activate|p2a: Opponent|confusion",
            "|move|p2a: Opponent|sleeptalk|p2a: Opponent",
            "|move|p2a: Opponent|substitute|p2a: Opponent|[from] Sleep Talk",
        ],
    );
}

#[test]
fn rest_sleep_talk_keeps_the_same_sleep_and_confusion_gate_order() {
    let mut state = confused_state(Choices::SLEEPTALK);
    state.side_two.get_active().status = PokemonStatus::SLEEP;
    state.side_two.get_active().rest_turns = 2;
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::SUBSTITUTE);
    let branches = generate(&mut state);
    let called = branches
        .iter()
        .find(|branch| {
            branch.instruction_list.iter().any(|instruction| {
                matches!(instruction, Instruction::ApplyVolatileStatus(apply)
                    if apply.side_ref == SideReference::SideTwo
                        && apply.volatile_status == PokemonVolatileStatus::SUBSTITUTE)
            })
        })
        .expect("expected Rest Sleep Talk call branch");
    let events = render(&mut state, called);
    assert_in_order(
        &events,
        &[
            "|cant|p2a: Opponent|slp",
            "|-activate|p2a: Opponent|confusion",
            "|move|p2a: Opponent|sleeptalk|p2a: Opponent",
            "|move|p2a: Opponent|substitute|p2a: Opponent|[from] Sleep Talk",
        ],
    );
}

#[test]
fn protected_memento_emits_one_protect_activation() {
    let mut state = confused_state(Choices::MEMENTO);
    state
        .side_two
        .volatile_statuses
        .remove(&PokemonVolatileStatus::CONFUSION);
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::PROTECT);
    let branches = generate(&mut state);
    assert_eq!(branches.len(), 1, "{branches:?}");
    let rendered = rendered(&mut state, &branches[0]);
    assert!(rendered.attribution_unsafe.is_empty(), "{rendered:?}");
    assert_eq!(
        rendered.lines,
        vec![
            "|",
            "|move|p1a: Lead|protect|p1a: Lead",
            "|move|p2a: Opponent|memento|p1a: Lead",
            "|-activate|p1a: Lead|Protect",
            "|",
            "|upkeep",
            "|turn|2",
        ],
        "{rendered:?}"
    );
}

#[test]
fn successful_memento_is_not_a_confusion_self_faint_collision() {
    let mut state = confused_state(Choices::MEMENTO);
    let branches = generate(&mut state);
    let memento = branches
        .iter()
        .find(|branch| {
            branch.instruction_list.iter().any(|instruction| {
                matches!(instruction, Instruction::Boost(boost)
                    if boost.side_ref == SideReference::SideOne)
            })
        })
        .expect("successful Memento must carry its target stat drops");
    let rendered = rendered(&mut state, memento);
    let events = rendered.lines.join("\n");
    assert!(
        events.contains("|move|p2a: Opponent|memento|p1a: Lead"),
        "{events}"
    );
    assert!(events.contains("|faint|p2a: Opponent"), "{events}");
    assert_eq!(
        rendered.lines,
        vec![
            "|",
            "|move|p1a: Lead|splash||[still]",
            "|-activate|p2a: Opponent|confusion",
            "|move|p2a: Opponent|memento|p1a: Lead",
            "|-unboost|p1a: Lead|atk|2",
            "|-unboost|p1a: Lead|spa|2",
            "|faint|p2a: Opponent",
            "|",
        ],
        "Memento drops the target's stats before silently fainting its user: {rendered:?}"
    );
    assert!(
        !events.contains("Liquid Ooze"),
        "Memento's reversible negative heal is not drain reversal: {events}"
    );
    assert!(
        !rendered
            .attribution_unsafe
            .iter()
            .any(|reason| reason == "confusion_selfhit_ambiguous_executed_self_damage"),
        "{rendered:?}"
    );
}

fn attracted_paralyzed_state(confused: bool) -> State {
    let mut state = confused_state(Choices::TACKLE);
    if !confused {
        state
            .side_two
            .volatile_statuses
            .remove(&PokemonVolatileStatus::CONFUSION);
    }
    state
        .side_two
        .volatile_statuses
        .insert(PokemonVolatileStatus::ATTRACT);
    state.side_two.get_active().status = PokemonStatus::PARALYZE;
    state
}

#[test]
fn attracted_and_paralyzed_empty_tails_fail_closed_with_or_without_confusion() {
    for confused in [false, true] {
        let mut state = attracted_paralyzed_state(confused);
        let branches = generate(&mut state);
        let rendered = branches
            .iter()
            .map(|branch| rendered(&mut state, branch))
            .find(|events| {
                events
                    .attribution_unsafe
                    .iter()
                    .any(|reason| reason.starts_with("attract_empty_tail_ambiguous"))
            })
            .unwrap_or_else(|| {
                panic!(
                    "expected the combined Attract/paralysis empty tail to fail closed; \\
                     confused={confused}, branches={branches:?}"
                )
            });
        let text = rendered.lines.join("\n");
        assert!(
            !text.contains("|cant|p2a: Opponent|Attract"),
            "the collapsed branch cannot be attributed wholly to Attract: {text}"
        );
        assert!(
            !text.contains("|cant|p2a: Opponent|par"),
            "the collapsed branch cannot be attributed wholly to paralysis: {text}"
        );
        if confused {
            assert!(
                text.contains("|-activate|p2a: Opponent|confusion"),
                "confusion still precedes the fail-closed mixed tail: {text}"
            );
        } else {
            assert!(
                !text.contains("|-activate|p2a: Opponent|confusion"),
                "unexpected confusion activation without the volatile: {text}"
            );
        }
    }
}

#[test]
fn switching_away_clears_confusion_silently_without_an_early_end_line() {
    let mut state = confused_state(Choices::SPLASH);
    state.side_two.pokemon[PokemonIndex::P1] = state.side_two.get_active_immutable().clone();
    let s1 = MoveChoice::Move(PokemonMoveIndex::M0);
    let s2 = MoveChoice::Switch(PokemonIndex::P1);
    let branches = generate_instructions_from_move_pair(&mut state, &s1, &s2, false);
    let branch = branches
        .iter()
        .find(|branch| {
            branch.instruction_list.iter().any(|instruction| {
                matches!(instruction, Instruction::Switch(switch)
                    if switch.side_ref == SideReference::SideTwo
                        && switch.next_index == PokemonIndex::P1)
            })
        })
        .expect("expected side-two switch branch");
    let before = state.serialize();
    let rendered = render_branch_events(
        &mut state,
        &s1,
        &s2,
        &branch.instruction_list,
        false,
        &EventContext {
            species: [vec!["Lead".into()], vec!["Opponent".into(), "Bench".into()]],
            turn: 1,
            hp_percent: [false, false],
        },
    );
    assert_eq!(
        before,
        state.serialize(),
        "rendering mutated the source state"
    );
    let events = rendered.lines.join("\n");
    assert!(events.contains("|switch|p2a: Bench"), "{events}");
    assert!(
        !events.contains("|-end|p2a: Opponent|confusion"),
        "{events}"
    );
    assert!(rendered.attribution_unsafe.is_empty(), "{rendered:?}");
}

#[test]
fn waking_then_hitting_self_remains_a_confusion_activation() {
    let mut state = confused_state(Choices::SUBSTITUTE);
    state.side_two.get_active().status = PokemonStatus::SLEEP;
    state.side_two.get_active().sleep_turns = 3;
    let branches = generate(&mut state);
    let woke_and_hit = branches
        .iter()
        .find(|branch| {
            damage_to(branch, SideReference::SideTwo, 35)
                && branch.instruction_list.iter().any(|instruction| {
                    matches!(instruction, Instruction::ChangeStatus(change)
                        if change.side_ref == SideReference::SideTwo
                            && change.old_status == PokemonStatus::SLEEP
                            && change.new_status == PokemonStatus::NONE)
                })
        })
        .expect("expected waking confusion self-hit branch");
    let events = render(&mut state, woke_and_hit);
    assert!(
        events.contains("|-activate|p2a: Opponent|confusion"),
        "{events}"
    );
    assert!(!events.contains("|cant|p2a: Opponent|slp"), "{events}");
}

#[test]
fn own_tempo_never_creates_a_confusion_activation() {
    let mut state = confused_state(Choices::SUBSTITUTE);
    state.side_two.get_active().ability = Abilities::OWNTEMPO;
    let branches = generate(&mut state);
    assert_eq!(branches.len(), 1, "Own Tempo must not branch: {branches:?}");
    let events = render(&mut state, &branches[0]);
    assert!(
        events.contains("|move|p2a: Opponent|substitute|p2a: Opponent"),
        "{events}"
    );
    assert!(
        !events.contains("|-activate|p2a: Opponent|confusion"),
        "{events}"
    );
}

#[test]
fn collapsed_crash_collision_is_rejected_without_causeless_damage() {
    let mut state = confused_state(Choices::HIGHJUMPKICK);
    // Confusion's fixed 40-power hit is exactly 50 here, colliding with the
    // 50%-max-HP crash. The engine combines those outcomes into one delta.
    state.side_two.get_active().attack = 143;
    let branches = generate(&mut state);
    let collision = self_hit_branch(&branches, 50);
    let rendered = rendered(&mut state, collision);
    let events = rendered.lines.join("\n");
    assert!(
        rendered
            .attribution_unsafe
            .iter()
            .any(|reason| reason == "confusion_selfhit_ambiguous_executed_self_damage"),
        "{rendered:?}"
    );
    assert!(
        !events.contains("|-activate|p2a: Opponent|confusion"),
        "{events}"
    );
    assert!(
        !events.contains("|move|p2a: Opponent|highjumpkick"),
        "{events}"
    );
    assert!(
        !events.lines().any(|line| line.starts_with("|-damage|")),
        "{events}"
    );
}

/// The deferred snap-out must render `-end` where Showdown renders it, with NO
/// fabricated `-activate`.
///
/// This is the assertion the sentinel collision failed. At the first attempt the
/// pending marker's restore amount was `+1` -- bit-for-bit the ladder's "the
/// confusion check ran" marker -- so the renderer took the self-hit arm, emitted
/// `|-activate|...|confusion`, emitted no `-end` at all, and ACCEPTED the world.
/// Fail-open, and strictly worse than the refusal it replaced: the observation
/// layer only clears `volatile:confusion` on `-end`, so the volatile would stick
/// and `confusion_elapsed` would ramp without bound.
#[test]
fn the_deferred_snap_out_emits_end_and_no_activation() {
    // Drive the ladder until a branch parks a pending snap-out, then take the
    // victim's next move attempt -- the ply Showdown snaps out on.
    let mut state = confused_state(Choices::SPLASH);
    let mut pending = None;
    for _ in 0..6 {
        let branches = generate(&mut state);
        if let Some(branch) = branches.iter().find(|b| {
            b.instruction_list.iter().any(|i| matches!(
                i,
                Instruction::ChangeVolatileStatusDuration(c)
                    if c.volatile_status == PokemonVolatileStatus::CONFUSION && c.amount < 0
            ))
        }) {
            state.apply_instructions(&branch.instruction_list);
            pending = Some(());
            break;
        }
        let first = branches.into_iter().next().expect("a branch");
        state.apply_instructions(&first.instruction_list);
    }
    pending.expect("the ladder must park a pending snap-out within six plies");

    let branches = generate(&mut state);
    let snap = branches
        .iter()
        .find(|b| {
            b.instruction_list.iter().any(|i| matches!(
                i,
                Instruction::RemoveVolatileStatus(r)
                    if r.volatile_status == PokemonVolatileStatus::CONFUSION
            ))
        })
        .expect("the next move attempt must consume the pending snap-out")
        .clone();
    let events = render(&mut state, &snap);
    assert!(
        events.contains("|-end|") && events.contains("confusion"),
        "Showdown emits -end on the snap-out turn: {events}"
    );
    assert!(
        !events.contains("|-activate|p2a: Opponent|confusion"),
        "no activation on the snap-out turn -- onBeforeMove returns before it: {events}"
    );
}


/// The mirror of the collision, and the assertion that actually catches it.
///
/// An ORDINARY confused turn emits the ladder's `+1` "check ran" marker. If the
/// snap-out restore amount equals that marker, the renderer misreads a normal
/// check as a snap-out: spurious `-end`, missing `-activate`, and the world
/// layer clears a volatile that is still attached. Testing only the snap-out
/// direction misses this entirely -- my first attempt did, and the mutant that
/// restores the collision passed it.
#[test]
fn an_ordinary_confusion_check_is_not_misread_as_a_snap_out() {
    let mut state = confused_state(Choices::SPLASH);
    let branches = generate(&mut state);
    // The surviving (move-goes-through) arm carries the +1 marker and no removal.
    let ordinary = branches
        .iter()
        .find(|b| {
            b.instruction_list.iter().any(|i| matches!(
                i,
                Instruction::ChangeVolatileStatusDuration(c)
                    if c.volatile_status == PokemonVolatileStatus::CONFUSION && c.amount == 1
            )) && !b.instruction_list.iter().any(|i| matches!(
                i,
                Instruction::RemoveVolatileStatus(r)
                    if r.volatile_status == PokemonVolatileStatus::CONFUSION
            ))
        })
        .expect("an ordinary confused turn must carry the +1 check marker")
        .clone();
    let events = render(&mut state, &ordinary);
    assert!(
        !events.contains("|-end|p2a: Opponent|confusion"),
        "a confusion CHECK must not render as a snap-out: {events}"
    );
    assert!(
        events.contains("|-activate|p2a: Opponent|confusion"),
        "a surviving confusion check announces itself: {events}"
    );
}

/// The refusal must name WHICH ambiguity refused, not just that one did.
///
/// The five predicates behind `attract_empty_tail_ambiguous` were function-local
/// and discarded at the refusal, so nothing recorded the split -- which meant the
/// only available plan for this refusal class was "patch the engine and hope".
/// The split decides the fix: the `paralyzed` arm is downgradeable to lossy (both
/// outcomes are "no move used, no reveal, no PP", and Attract dominates 4:1),
/// while the noop/miss arms are not downgradeable at any price because they erase
/// a `|move|` reveal.
///
/// Pinned so the sub-case cannot silently collapse back to a single bare slug,
/// which would quietly destroy the measurement again.
#[test]
fn the_attract_refusal_names_its_subcase() {
    let mut state = attracted_paralyzed_state(false);
    let branches = generate(&mut state);
    let reasons: Vec<String> = branches
        .iter()
        .flat_map(|branch| rendered(&mut state.clone(), branch).attribution_unsafe)
        .filter(|reason| reason.starts_with("attract_empty_tail_ambiguous"))
        .collect();
    assert!(!reasons.is_empty(), "expected an attract refusal to measure");
    for reason in &reasons {
        assert_ne!(
            reason, "attract_empty_tail_ambiguous",
            "the bare slug carries no sub-case and cannot be measured: {reasons:?}"
        );
        assert!(
            reason.starts_with("attract_empty_tail_ambiguous:"),
            "malformed sub-case slug: {reason}"
        );
    }
    // Tackle at 100% accuracy into a normal target: paralysis is the only live
    // predicate, so this is the CLEAN paralyzed case -- the one where the cheap
    // lossy downgrade really would be safe.
    assert!(
        reasons
            .iter()
            .any(|reason| reason == "attract_empty_tail_ambiguous:paralyzed"),
        "{reasons:?}"
    );
}

/// The slug must report EVERY live predicate, not just the first one.
///
/// Found by independent review, and it is the difference between a probe that
/// answers its question and one that answers it backwards. `attacker_paralyzed`
/// is a property of the ATTACKER; `miss` and `noop` are properties of the MOVE.
/// They co-occur freely, so a first-match bucket files the non-downgradeable
/// arms under the one label that looks safe to downgrade -- and the contamination
/// is unrecoverable from the emitted data, because the other predicates are
/// discarded at the refusal.
///
/// Measured masses at the refusal: a paralyzed attacker using a 70%-accuracy move
/// carries 15.3% miss, and one whose target is immune carries 37.5% noop. Reading
/// either as "paralyzed" would say "ship the lossy downgrade" over mass that
/// erases a `|move|` reveal.
#[test]
fn the_attract_subcase_reports_every_live_predicate_not_just_the_first() {
    fn slugs(state: &mut State) -> Vec<String> {
        let branches = generate(state);
        branches
            .iter()
            .flat_map(|branch| rendered(&mut state.clone(), branch).attribution_unsafe)
            .filter(|reason| reason.starts_with("attract_empty_tail_ambiguous"))
            .collect()
    }

    // Paralyzed + a move that can miss. THUNDER is 70% accuracy in gen3.
    let mut miss_state = attracted_paralyzed_state(false);
    miss_state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::THUNDER);
    let miss = slugs(&mut miss_state);
    assert!(!miss.is_empty(), "expected an attract refusal to measure");
    assert!(
        // Exact joint string, not two `contains` calls. Order is source-order so
        // the bucket key is stable across builds; asserting the parts separately
        // let a swap of the pushes pass green, which would split one bucket into
        // permutations and silently halve both counts.
        miss.iter()
            .any(|reason| reason == "attract_empty_tail_ambiguous:paralyzed+miss"),
        "a paralyzed attacker using a 70%-accuracy move must report BOTH          predicates, or the miss mass hides inside the paralyzed bucket: {miss:?}"
    );
    assert!(
        !miss.iter().any(|r| r == "attract_empty_tail_ambiguous:paralyzed"),
        "the clean-paralyzed slug must not be emitted when miss is also live: {miss:?}"
    );
}

/// Every sub-case literal must be pinned, not just the two that were convenient.
///
/// Found by independent review: renaming `noop`, `volatile` or `cannot_act` left
/// the entire suite green. `noop` is the worst of those to leave unpinned -- it
/// carries the largest non-downgradeable mass (37.5% when the target is immune),
/// so a refactor that renamed or dropped it would make the probe read the
/// non-downgradeable share as ZERO. That is the same wrong-direction error as the
/// first-match bucketing this telemetry was written to fix.
///
/// Also pins the joint slug's FIXED ordering. Order is source-order rather than
/// evaluation-order so the key is stable, but nothing asserted it, and a swap
/// would silently split one bucket into permutations across builds.
#[test]
fn every_attract_subcase_literal_is_pinned() {
    fn slug_for(state: &mut State) -> Vec<String> {
        let branches = generate(state);
        branches
            .iter()
            .flat_map(|branch| rendered(&mut state.clone(), branch).attribution_unsafe)
            .filter(|reason| reason.starts_with("attract_empty_tail_ambiguous"))
            .collect()
    }

    // noop: the move cannot change anything -- a Normal move into a Ghost.
    let mut noop_state = attracted_paralyzed_state(false);
    noop_state.side_one.get_active().types =
        (poke_engine::state::PokemonType::GHOST, poke_engine::state::PokemonType::TYPELESS);
    let noop = slug_for(&mut noop_state);
    assert!(
        noop.iter().any(|reason| reason.contains("noop")),
        "the immune-target arm must report `noop`: {noop:?}"
    );
    // ...and the joint order is source-order, not evaluation-order.
    assert!(
        noop.iter().any(|reason| reason.contains("paralyzed+noop")),
        "joint slug order must be fixed as `paralyzed+noop`: {noop:?}"
    );

    // cannot_act: Splash can never act. Note the ATTACKER is side TWO -- that is
    // where `attracted_paralyzed_state` puts ATTRACT and PARALYZE -- so the move
    // has to be replaced there. Replacing side one's move instead leaves the
    // attacker on Tackle and the slug comes back a bare `:paralyzed`, which is
    // how this fixture was wrong the first time.
    let mut cant_state = attracted_paralyzed_state(false);
    cant_state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::SPLASH);
    let cant = slug_for(&mut cant_state);
    assert!(
        cant.iter().any(|reason| reason.contains("cannot_act")),
        "Splash must report `cannot_act`: {cant:?}"
    );
}

/// Mirror of `_REASON_DETAIL_LIMIT` in `src/pokezero/engine_search.py`.
///
/// The refusal message crosses into Python and becomes a `world_failure_reasons`
/// key; that seam is the only place a length limit exists. Pinning it from this
/// side means a future slug that outgrows the budget fails HERE, in the crate
/// suite that owns the slug, rather than silently at the seam in a campaign run.
const PY_REASON_DETAIL_LIMIT: usize = 512;

/// Both sides refusing with DIFFERENT sub-case sets must key canonically and fit.
///
/// Found by independent review. The first fix deduped IDENTICAL reasons, which
/// missed the likelier case: `miss`/`noop`/`cannot_act` are properties of the
/// MOVE, and the two sides have different moves, so two-sided refusals usually
/// carry two DIFFERENT slugs. Two bugs lived in that gap:
///
/// 1. **Truncation.** At the old 160-char seam the joined pair overflowed and the
///    tail label was cut. `{paralyzed+cannot_act, paralyzed+miss}` and
///    `{paralyzed+cannot_act, paralyzed+miss+volatile}` both landed in the same
///    `...paralyzed+mis` bucket — the `+volatile` arm, which is exactly the
///    non-downgradeable mass this whole split exists to measure, vanished into a
///    bucket that looked like a different question.
/// 2. **Order.** The join preserved render order, which is SPEED order, so the
///    same pair of sub-cases keyed two ways depending only on who moved first.
///
/// Asserting the label rather than the `PyErr` keeps this test interpreter-free.
#[test]
fn a_two_sided_refusal_keys_canonically_and_fits_the_python_seam() {
    // Both sides attracted AND paralyzed, with moves that add DIFFERENT second
    // predicates: Splash can never act, Thunder is 70% accurate in gen3.
    fn two_sided_state(lead_is_faster: bool) -> State {
        let mut state = attracted_paralyzed_state(false);
        state
            .side_one
            .volatile_statuses
            .insert(PokemonVolatileStatus::ATTRACT);
        state.side_one.get_active().status = PokemonStatus::PARALYZE;
        state
            .side_one
            .get_active()
            .replace_move(PokemonMoveIndex::M0, Choices::SPLASH);
        state
            .side_two
            .get_active()
            .replace_move(PokemonMoveIndex::M0, Choices::THUNDER);
        // Speed decides RENDER order, which is what the sort has to neutralise.
        state.side_one.get_active().speed = if lead_is_faster { 500 } else { 1 };
        state.side_two.get_active().speed = if lead_is_faster { 1 } else { 500 };
        state
    }

    let mut labels = Vec::new();
    for lead_is_faster in [true, false] {
        let mut state = two_sided_state(lead_is_faster);
        let branches = generate(&mut state);
        let two_sided = branches
            .iter()
            .map(|branch| rendered(&mut state.clone(), branch))
            .find(|events| {
                events
                    .attribution_unsafe
                    .iter()
                    .filter(|reason| reason.starts_with("attract_empty_tail_ambiguous"))
                    .count()
                    >= 2
            })
            .unwrap_or_else(|| {
                panic!(
                    "expected a branch where BOTH sides refuse; \
                     lead_is_faster={lead_is_faster}"
                )
            });

        let label = attribution_unsafe_label(&two_sided);

        // Distinct slugs, so dedupe alone could not have saved this.
        assert!(
            label.contains("cannot_act") && label.contains("miss"),
            "fixture must produce two DIFFERENT sub-case sets, got: {label}"
        );

        // Canonical order: sorted, never render/speed order.
        let mut sorted = label.split(',').collect::<Vec<_>>();
        sorted.sort_unstable();
        assert_eq!(
            label,
            sorted.join(","),
            "reasons must be sorted so one measurement lands in one bucket: {label}"
        );

        // Fits the seam WITH the prefix the Python side prepends. The lane string
        // varies; `tree/model fold` is the longest in use.
        let full = format!(
            "attribution-unsafe renderer branch rejected before tree/model fold: {label}"
        );
        assert!(
            full.len() <= PY_REASON_DETAIL_LIMIT,
            "refusal message is {} chars, over the {PY_REASON_DETAIL_LIMIT}-char seam \
             budget -- it would be truncated into a `world_failure_reasons` key: {full}",
            full.len()
        );

        labels.push(label);
    }

    // The whole point of the sort: speed order must not change the key.
    assert_eq!(
        labels[0], labels[1],
        "the same pair of sub-cases keyed two ways depending on who moved first, \
         splitting one measurement across two buckets"
    );
}
/// The sleep-talk refusal must name its cause in `attribution_unsafe` while
/// leaving the `lossy` tag exactly as it was.
///
/// `sleeptalk_called_unidentified` is 48.9% of world failures on the era-55
/// probe -- bigger than every other class combined -- and the single slug hid
/// which of two OPPOSITE problems was happening. `ambiguous` (two candidates
/// regenerate byte-identical tails) can only be fixed by the engine recording
/// which move it called. `none_matched` (no candidate reproduces the tail) means
/// the replay diverges from what the engine did, which is a different defect.
///
/// The `lossy` tag must NOT split: `engine_transition_differential.py` matches it
/// with `set(lossy) == {_SLEEPTALK_LOSSY_MARKER}` to decide branch usability, and
/// that file's bytes are pinned by the certification lifecycle. Splitting it
/// would silently change which branches the differential accepts.
#[test]
fn the_sleeptalk_refusal_subcases_without_moving_the_lossy_contract() {
    let mut state = confused_state(Choices::SLEEPTALK);
    state
        .side_two
        .volatile_statuses
        .remove(&PokemonVolatileStatus::CONFUSION);
    state.side_two.get_active().status = PokemonStatus::SLEEP;
    state.side_two.get_active().rest_turns = 0;
    // Two callees whose instruction lists are BYTE-IDENTICAL: Harden and
    // Withdraw are both +1 Defense on the user. Splash and Harden do NOT work --
    // Harden emits a boost and Splash emits nothing, so the tails differ and the
    // engine identifies each correctly. Ambiguity needs genuinely equal effects.
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::HARDEN);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M2, Choices::WITHDRAW);

    let branches = generate(&mut state);
    let mut saw_subcase = false;
    for branch in &branches {
        let r = rendered(&mut state.clone(), branch);
        for reason in &r.attribution_unsafe {
            if reason.starts_with("sleeptalk_called_unidentified") {
                saw_subcase = true;
                assert_ne!(
                    reason, "sleeptalk_called_unidentified",
                    "the bare slug carries no cause and cannot be measured: {:?}",
                    r.attribution_unsafe
                );
                // Pin WHICH sub-case, not merely that one of the two appeared.
                // Accepting either is a tautology over the implementation's only
                // two outputs: independent review swapped the literals and the
                // whole 336-test suite stayed green. Harden and Withdraw
                // regenerate byte-identical tails, so the cause is `ambiguous`
                // by construction -- and `ambiguous` is the arm the PR's own
                // table says is fixable ONLY by an engine change, so reading it
                // as `none_matched` would send the next phase at the renderer
                // and waste the cycle.
                assert_eq!(
                    reason, "sleeptalk_called_unidentified:ambiguous",
                    "two byte-identical callees must report `ambiguous`: {:?}",
                    r.attribution_unsafe
                );
            }
        }
        // The lossy CONTRACT tag stays bare, whatever the sub-case is.
        if r.attribution_unsafe
            .iter()
            .any(|x| x.starts_with("sleeptalk_called_unidentified"))
        {
            assert!(
                r.lossy.iter().any(|x| x == "sleeptalk_called_unidentified"),
                "the differential matches the lossy tag EXACTLY; it must stay \
                 unsplit: {:?}",
                r.lossy
            );
            assert!(
                !r.lossy
                    .iter()
                    .any(|x| x.starts_with("sleeptalk_called_unidentified:")),
                "a sub-cased lossy tag would change which branches the \
                 differential accepts: {:?}",
                r.lossy
            );
        }
    }
    assert!(saw_subcase, "fixture produced no sleep-talk refusal to measure");
}

/// `none_matched` IS reachable, and the cause is the OPPONENT's chosen move.
///
/// Found by independent review after two wrong answers from me. The identifier
/// regenerates every Sleep Talk candidate against a hardcoded
/// `&Choice::default()` for the defender (`events.rs`,
/// `identify_sleep_talk_called`). But the engine gates C31's 32-roll
/// enumeration on the REAL defender choice: `branch_on_damage && choice
/// .first_move && pending_hp_reading_move(defender_choice)`, where
/// `pending_hp_reading_move` is {SUBSTITUTE, FLAIL, REVERSAL}. So when the
/// defender picks one of those and the sleeper moves first, the engine emits one
/// of 32 specific rolls while the renderer regenerates the ordinary 2-branch
/// max/crit collapse against a `NONE` move. Nothing matches.
///
/// This is why the earlier "0 of 7,560 combinations" sweep found nothing: it
/// varied the SLEEPER's state exhaustively and held the opponent on Splash,
/// which is exactly the zero case. Coverage, not weighting.
///
/// It matters because it is renderer-side. Reach: 294 of 1682 rolled gen3
/// randbats VARIANTS (17.5%) carry Substitute, Flail or Reversal -- the
/// variant level is the right one, since that is how a world is sampled; the
/// movepool-membership figure is 88 of 393 sets. Neither is the probability the
/// opponent PICKS one of those moves on a given turn, which is what the actual
/// mass depends on and which nothing here measures. The fix is to thread the
/// real defender choice (and `first_move`) into the identifier rather than to
/// patch the engine. `ambiguous` is the arm that needs an engine change; this one does
/// not. Tracked separately so this PR stays telemetry-only.
#[test]
fn the_none_matched_subcase_is_reachable_via_the_defenders_move() {
    // Sleep Talk + a realistic randbats callee set, sleeper FASTER, defender on
    // Substitute, damage branching on (production: tree.rs plies 1-2).
    let mut state = confused_state(Choices::SLEEPTALK);
    state
        .side_two
        .volatile_statuses
        .remove(&PokemonVolatileStatus::CONFUSION);
    state.side_two.get_active().status = PokemonStatus::SLEEP;
    state.side_two.get_active().rest_turns = 0;
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::BODYSLAM);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M2, Choices::EARTHQUAKE);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M3, Choices::REST);
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::SUBSTITUTE);
    state.side_one.get_active().speed = 1;
    state.side_two.get_active().speed = 500;

    let before = format!("{state:?}");
    let branches = generate_instructions_from_move_pair(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
        true,
    );
    assert_eq!(before, format!("{state:?}"), "generation mutated the source state");

    let mut none_matched = 0usize;
    for branch in &branches {
        let r = render_branch_events(
            &mut state.clone(),
            &MoveChoice::Move(PokemonMoveIndex::M0),
            &MoveChoice::Move(PokemonMoveIndex::M0),
            &branch.instruction_list,
            true,
            &EventContext {
                species: [vec!["Lead".into()], vec!["Opponent".into()]],
                turn: 1,
                hp_percent: [false, false],
            },
        );
        for reason in &r.attribution_unsafe {
            if !reason.starts_with("sleeptalk_called_unidentified") {
                continue;
            }
            if reason == "sleeptalk_called_unidentified:none_matched" {
                none_matched += 1;
                // Contract tag stays bare on this arm too.
                assert!(
                    r.lossy.iter().any(|x| x == "sleeptalk_called_unidentified"),
                    "the differential matches the lossy tag EXACTLY: {:?}",
                    r.lossy
                );
                assert!(
                    !r.lossy
                        .iter()
                        .any(|x| x.starts_with("sleeptalk_called_unidentified:")),
                    "a sub-cased lossy tag would change which branches the \
                     differential accepts: {:?}",
                    r.lossy
                );
            }
        }
    }

    assert!(
        none_matched > 0,
        "the defender-move axis must still reach `none_matched`; if this fires, \
         either the renderer now threads the real defender choice (good -- delete \
         this test and keep the fix) or the arm regressed to reporting `ambiguous` \
         (bad -- the largest failure class is being mis-diagnosed)"
    );
}

/// A Sleep Talk callee must sometimes be IDENTIFIED, not only refused.
///
/// Every other sleeptalk fixture in this file asserts a refusal, which left the
/// happy path completely unpinned. Independent review showed the consequence:
/// hardcoding the regeneration's `branch_on_damage` to either `false` or `true`
/// left all 343 tests green. The `false` variant is the serious one -- it
/// re-creates the exact pre-C87 divergence (renderer collapses damage while the
/// engine branches it), so under production's `branch_on_damage: true` every
/// damaging callee would fail to identify and be refused as `none_matched`. The
/// measurement would swing wholesale with a green suite, because a fixture that
/// only asserts `none_matched > 0` is satisfied by MORE refusals.
///
/// Same shape as the `none_matched` fixture but with the defender on a move that
/// does NOT trip `pending_hp_reading_move`, so identification should succeed.
/// This is also the only test that pins C87's engine/renderer alignment from the
/// correct side.
#[test]
fn a_sleeptalk_callee_is_identified_when_the_defender_does_not_read_hp() {
    // Run at BOTH damage-branching settings. Production uses `true` for plies
    // 1-2 (`tree.rs:545`) and `false` deeper, and the renderer must track the
    // engine on each: hardcoding the regeneration to `false` breaks the shallow
    // plies, hardcoding it to `true` breaks the deep ones. A fixture at one
    // setting only catches one of those.
    //
    // ...and run over TWO callee sets, because parametrising the setting is not
    // enough on its own. Review demonstrated that the bod -> `true` mutant
    // survives a set of nine ordinary damaging callees: the engine's unbranched
    // collapsed damage (94) happens to coincide with one of the renderer's
    // branched values for all of them, so the mutant is invisible. It is NOT
    // invisible for Petal Dance, whose crit arm restructures the tail rather
    // than just changing an integer -- KO -> ToggleSideOneForceSwitch, and no
    // LOCKEDMOVE duration instruction -- so 94 is in neither branch. The lesson
    // generalises: a damage-integer-only fixture cannot catch a
    // branching-shape divergence.
    for (label, fourth_move) in [
        // The realistic RestTalk shape, kept because it is what randbats rolls.
        ("resttalk", Choices::REST),
        // The shape-changing callee that actually exercises the deep-ply arm.
        ("shape-changing", Choices::PETALDANCE),
    ] {
    for branch_on_damage in [true, false] {
        let mut state = confused_state(Choices::SLEEPTALK);
        state
            .side_two
            .volatile_statuses
            .remove(&PokemonVolatileStatus::CONFUSION);
        state.side_two.get_active().status = PokemonStatus::SLEEP;
        state.side_two.get_active().rest_turns = 0;
        state
            .side_two
            .get_active()
            .replace_move(PokemonMoveIndex::M1, Choices::BODYSLAM);
        state
            .side_two
            .get_active()
            .replace_move(PokemonMoveIndex::M2, Choices::EARTHQUAKE);
        state
            .side_two
            .get_active()
            .replace_move(PokemonMoveIndex::M3, fourth_move);
        // Splash reads no HP, so C31's 32-roll enumeration does not fire and the
        // renderer's regeneration is on the same footing as the engine.
        state
            .side_one
            .get_active()
            .replace_move(PokemonMoveIndex::M0, Choices::SPLASH);
        state.side_one.get_active().speed = 1;
        state.side_two.get_active().speed = 500;

        let branches = generate_instructions_from_move_pair(
            &mut state,
            &MoveChoice::Move(PokemonMoveIndex::M0),
            &MoveChoice::Move(PokemonMoveIndex::M0),
            branch_on_damage,
        );

        let mut identified_a_damaging_callee = false;
        let mut none_matched = 0usize;
        for branch in &branches {
            let r = render_branch_events(
                &mut state.clone(),
                &MoveChoice::Move(PokemonMoveIndex::M0),
                &MoveChoice::Move(PokemonMoveIndex::M0),
                &branch.instruction_list,
                branch_on_damage,
                &EventContext {
                    species: [vec!["Lead".into()], vec!["Opponent".into()]],
                    turn: 1,
                    hp_percent: [false, false],
                },
            );
            if r.attribution_unsafe
                .iter()
                .any(|x| x == "sleeptalk_called_unidentified:none_matched")
            {
                none_matched += 1;
                continue;
            }
            let text = r.lines.join("\n");
            if text.contains("|[from] Sleep Talk")
                && (text.contains("|bodyslam|") || text.contains("|earthquake|"))
            {
                identified_a_damaging_callee = true;
            }
        }

        assert!(
            identified_a_damaging_callee,
            "no branch identified a damaging Sleep Talk callee at \
             branch_on_damage={branch_on_damage}, callee set {label}. If the \
             regeneration stops \
             tracking the engine's setting, damaging callees become \
             unidentifiable and the largest failure class inflates -- with a \
             suite that only checks for refusals staying green."
        );
        assert_eq!(
            none_matched, 0,
            "a defender that reads no HP must not produce `none_matched` \
             (branch_on_damage={branch_on_damage}, callee set {label}); that arm \
             belongs to the pending_hp_reading_move gate"
        );
    }
    }
}
