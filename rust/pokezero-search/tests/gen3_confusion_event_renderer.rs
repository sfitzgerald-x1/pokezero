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
    // AND NOT LOSSY AT ALL. This is a NAMED-callee Protect render on a fully clean branch,
    // and `set(lossy)` here must stay EMPTY.
    //
    // It guards the safety argument for the Protect render counter in the unnamed-callee
    // walk. That counter is safe only because the walk has ALREADY pushed
    // `SLEEPTALK_LOSSY_TAG`, so `set(lossy)` does not move. There are three OTHER
    // `|-activate|...|Protect` sites on the named path, and the obvious next edit for
    // anyone reading "COUNT IT" is to add the same call to them. Doing that here would take
    // `set(lossy)` from `{}` to `{sleeptalk_called_unidentified}` on a branch where Sleep
    // Talk never ran -- which makes `fidelity_gate_events.py` and `leaf_vs_reality.py` drop
    // the row as `skip:lossy_render`, flips the transition differential's `sleeptalk_union`,
    // and counts a `lossy_render` that is not lossy.
    //
    // Found by review as mutant N9, which passed all 423 tests: this test asserted the full
    // `lines` vector and `attribution_unsafe`, but never `lossy`. Production-only defects
    // are the reason the counter is pinned by assertion rather than by comment, and the pin
    // covered only the call site that exists.
    assert!(rendered.lossy.is_empty(), "{rendered:?}");
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
    // Two callees whose instruction lists are BYTE-IDENTICAL and whose tail the walk still
    // CANNOT render: Mean Look and Spider Web both apply TRAPPED and nothing else. So the
    // branch is genuinely Ambiguous -- no candidate can be named -- and it reaches the
    // unnamed-callee walk, which has no line for a trapping volatile.
    //
    // THIS FIXTURE HAS MIGRATED TWICE, for the same reason each time: the family it used
    // stopped refusing, and a fail-closed guard has to live on a family that still blocks.
    //
    //   1. Harden/Withdraw (+1 Defense each) went first, when the walk learned `|-boost|`.
    //      That pair is now pinned positively by
    //      `byte_identical_callees_with_a_boost_tail_are_now_usable_and_render_the_line`.
    //   2. Recover/Soft-Boiled went second, when the walk learned the direct self-heal. That
    //      pair is now pinned positively by
    //      `a_direct_self_heal_renders_the_exact_bare_line_on_the_healed_side` -- which is what
    //      this fixture should have BECOME rather than being retargeted away from, and review
    //      said so.
    //
    // `volatile` is the right family for the third home because it is still blocked, and
    // deliberately so: admitting `RemoveVolatileStatus` wholesale is the defect #1133's
    // substitute-break guard exists to prevent, so the walk renders the SUBSTITUTE volatile
    // and nothing else.
    //
    // The half-HP line below is VESTIGIAL. It existed so Recover was not a no-op; a trapping
    // move does not care about HP. Kept only because it is harmless, and labelled so nobody
    // reads it as load-bearing.
    state.side_two.get_active().hp = state.side_two.get_active().maxhp / 2;
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::MEANLOOK);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M2, Choices::SPIDERWEB);

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
                // Pin WHICH sub-case, not merely that one of the two appeared. Accepting
                // either is a tautology over the implementation's only two outputs:
                // independent review swapped the literals and the whole suite stayed green.
                //
                // Mean Look and Spider Web regenerate byte-identical tails, so the
                // identification is `ambiguous` BY CONSTRUCTION rather than by luck of the
                // fixture. And an ambiguous tail of `[ApplyVolatileStatus(TRAPPED)]` is
                // unrenderable, because the walk emits HP decreases, drags, faints, boosts,
                // the substitute hit and break, and the direct self-heal -- not a trapping
                // volatile. So this fixture is the UNRENDERABLE arm: it must still refuse, and
                // it must say WHICH family blocked it.
                //
                // The literal is `:volatile`, and it is derived from the tail through the real
                // production render path rather than asserted in prose beside a coarser one.
                // It stays pinned to ONE exact string: accepting a prefix would restore the
                // tautology review caught. Before the three-way split this assertion read
                // `:ambiguous`, which is now the arm that does NOT refuse, so leaving it would
                // have pinned the opposite of the intended behaviour.
                assert_eq!(
                    reason, "sleeptalk_called_unidentified:ambiguous_unrenderable:volatile",
                    "byte-identical callees whose tail carries an effect the walk \
                     cannot render must refuse, and name that arm: {:?}",
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

/// A BOOST tail is now rendered, so the branch is usable AND the line is emitted.
///
/// This is the case the refusal fixture above used to cover. `ambiguous_unrenderable` was
/// 8,149 world failures in era 59 -- 51.6% of the abort channel and the largest single
/// world-level refusal in the era -- and #1124's family split existed to scope exactly this.
/// The oracle corpus put 10 of its 16 refused tails on a bare `[Boost]`, from
/// identical-boost pairs like Harden/Withdraw where the callee cannot be named but the
/// transition is proven.
///
/// TWO assertions, because either alone is satisfiable by a broken change: the branch must
/// stop refusing, AND the walk must actually emit the `|-boost|` line. Admitting the family
/// without rendering it would silently drop a boost into the fold -- precisely the
/// C52-mirror defect `ambiguous_tail_is_fully_renderable` exists to prevent -- and rendering
/// without admitting would leave the world refused for nothing.
#[test]
fn byte_identical_callees_with_a_boost_tail_are_now_usable_and_render_the_line() {
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
        .replace_move(PokemonMoveIndex::M1, Choices::HARDEN);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M2, Choices::WITHDRAW);

    let branches = generate(&mut state);
    let mut saw_boost_line = false;
    let mut saw_lossy_subcase = false;
    for branch in &branches {
        let r = rendered(&mut state.clone(), branch);

        assert!(
            !r.attribution_unsafe
                .iter()
                .any(|x| x.starts_with("sleeptalk_called_unidentified")),
            "a boost tail is fully renderable now and must not refuse: {:?}",
            r.attribution_unsafe
        );

        if r.lossy
            .iter()
            .any(|x| x == "sleeptalk_called_unidentified")
        {
            saw_lossy_subcase = true;
            // The CONTRACT tag stays bare so the differential still accepts the branch.
            assert!(
                !r.lossy
                    .iter()
                    .any(|x| x.starts_with("sleeptalk_called_unidentified:")),
                "the lossy contract tag must stay unsplit: {:?}",
                r.lossy
            );
        }
        // EXACT LINE, not a prefix. `starts_with("|-boost|")` is a tautology over this
        // arm's only output shape, and review showed it letting three mutations through the
        // whole suite: the wrong ident (boost credited to the other Pokemon), the wrong stat
        // code, and a spurious `[from] item: Leftovers` tag -- which still starts with
        // `|-boost|` while making the FOLD ignore the boost entirely, since its `-boost` arm
        // is gated on `from_payload is None`. The sibling refusal test's own comment warns
        // that "accepting a prefix would restore the tautology that review caught here"; this
        // one had reintroduced it.
        if r.lines.iter().any(|line| line == "|-boost|p2a: Opponent|def|1") {
            saw_boost_line = true;
        }
        assert!(
            !r.lines.iter().any(|line| line.starts_with("|-boost|")
                && line != "|-boost|p2a: Opponent|def|1"),
            "no boost line other than the +1 Defense the tail carries: {:?}",
            r.lines
        );
    }

    assert!(
        saw_lossy_subcase,
        "the ambiguity must still be COUNTED as lossy -- a class that stops refusing must \
         not stop being measured"
    );
    assert!(
        saw_boost_line,
        "the walk must EMIT the boost line, not merely stop refusing: admitting the family \
         without rendering it drops the boost into the fold silently"
    );
}

/// A NEGATIVE boost, on the OPPONENT, with an exact line.
///
/// Charm and Feather Dance are both -2 Attack on the target, so their tails are
/// byte-identical and the callee cannot be named. This covers two gaps the Harden/Withdraw
/// fixture structurally cannot, both of which review found live and green:
///
///   * `|-unboost|` at all. An `amount != 0` -> `amount > 0` mutation silently dropped every
///     negative boost line -- the C52-mirror defect verbatim, family admitted and effect
///     dropped. (The guard is now gone entirely, which removes the mutation target, but the
///     coverage belongs here regardless.)
///   * A boost whose target is NOT the active side. Harden/Withdraw boosts the user, so
///     `boost.side_ref` and the walk's `side` coincide and using either produces the same
///     ident. An opponent-target boost separates them.
#[test]
fn a_negative_opponent_boost_renders_the_exact_unboost_line() {
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
        .replace_move(PokemonMoveIndex::M1, Choices::CHARM);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M2, Choices::FEATHERDANCE);

    let branches = generate(&mut state);
    let mut saw_unboost = false;
    for branch in &branches {
        let r = rendered(&mut state.clone(), branch);
        assert!(
            !r.attribution_unsafe
                .iter()
                .any(|x| x.starts_with("sleeptalk_called_unidentified")),
            "a negative boost is as renderable as a positive one: {:?}",
            r.attribution_unsafe
        );
        // The TARGET is side one, not the sleeper on side two -- that is the whole point.
        if r.lines.iter().any(|line| line == "|-unboost|p1a: Lead|atk|2") {
            saw_unboost = true;
        }
    }

    assert!(
        saw_unboost,
        "the exact `|-unboost|p1a: Lead|atk|2` line must be emitted -- the magnitude is \
         unsigned and the head flips on sign, and the ident must be the TARGET's"
    );
}

/// The walk's contract is IN ORDER, and a `[Damage, Boost..]` tail is where that bites.
///
/// Ancient Power and Silver Wind are both 60 BP with a +1 all-stats secondary, so their
/// tails are byte-identical and shaped `[..., Damage, Boost x5]`. Before the residual flush
/// was added to the Boost arm the stream put all five boosts BEFORE the damage they follow.
/// Both branches refused on main, so the misordering would have been a fidelity regression
/// introduced by the very change that admits them -- and the full suite passed with and
/// without the fix, which is why this exists.
#[test]
fn renders_a_damage_then_boost_tail_in_order() {
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
        .replace_move(PokemonMoveIndex::M1, Choices::ANCIENTPOWER);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M2, Choices::SILVERWIND);

    let branches = generate(&mut state);
    let mut checked = false;
    for branch in &branches {
        let r = rendered(&mut state.clone(), branch);
        let damage_at = r.lines.iter().position(|l| l.starts_with("|-damage|"));
        let first_boost_at = r.lines.iter().position(|l| l.starts_with("|-boost|"));
        if let (Some(damage), Some(boost)) = (damage_at, first_boost_at) {
            checked = true;
            assert!(
                damage < boost,
                "the engine's tail is damage-then-boosts, so the stream must be too: {:?}",
                r.lines
            );
        }
    }

    assert!(
        checked,
        "fixture produced no branch carrying both a damage and a boost line -- without one \
         this test asserts nothing"
    );
}

/// The OTHER arm: byte-identical callees whose tail the walk CAN render are USABLE.
///
/// Tackle and Scratch are both 40-power physical Normal, so their tails are
/// byte-identical AND contain nothing but damage -- exactly the shape
/// `ambiguous_tail_is_fully_renderable` admits. The transition is proven and the walk
/// describes it completely, so refusing would discard a world for a missing label.
///
/// Without this test the fail-closed predicate could be `|_| false` and the suite would
/// stay green on the sibling above, which is the whole reason the split needs two
/// fixtures rather than one.
#[test]
fn byte_identical_callees_with_a_renderable_tail_are_usable_not_refused() {
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
        .replace_move(PokemonMoveIndex::M1, Choices::TACKLE);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M2, Choices::SCRATCH);

    let branches = generate(&mut state);
    let mut saw_usable = false;
    for branch in &branches {
        let r = rendered(&mut state.clone(), branch);
        if !r
            .lossy_subcases
            .iter()
            .any(|x| x == "sleeptalk_called_unidentified:ambiguous")
        {
            continue;
        }
        saw_usable = true;
        // NOT refused: nothing sleep-talk-shaped may reach `attribution_unsafe`.
        assert!(
            !r.attribution_unsafe
                .iter()
                .any(|x| x.starts_with("sleeptalk_called_unidentified")),
            "a renderable ambiguous tail must not refuse: {:?}",
            r.attribution_unsafe
        );
        // The differential's contract tag stays bare and present.
        assert!(
            r.lossy.iter().any(|x| x == "sleeptalk_called_unidentified"),
            "the usable arm must still carry the bare lossy tag: {:?}",
            r.lossy
        );
        // ...and NOTHING sub-cased may join it. This is the arm the differential actually
        // consumes, and it gates usability on exact set equality
        // (`set(lossy) == {_SLEEPTALK_LOSSY_MARKER}`). An extra entry makes every
        // newly-usable branch UNUSABLE to the matcher -- silently deleting this change's
        // entire benefit. The refusing arm already asserts this, but refused branches are
        // discarded anyway, so the assertion was on the arm that does not matter.
        assert!(
            !r.lossy
                .iter()
                .any(|x| x.starts_with("sleeptalk_called_unidentified:")),
            "a sub-cased lossy tag on the USABLE arm changes which branches the \
             differential accepts: {:?}",
            r.lossy
        );
        // ...and neither candidate may be NAMED, or the render invents evidence.
        for callee in ["tackle", "scratch"] {
            assert!(
                !r.lines
                    .iter()
                    .any(|line| line.starts_with("|move|p2a:") && line.contains(callee)),
                "named {callee:?} despite ambiguity: {:?}",
                r.lines
            );
        }
    }
    assert!(
        saw_usable,
        "fixture produced no USABLE ambiguous branch, so the renderable arm is untested"
    );
}

/// REGRESSION: the defender's move must no longer break Sleep Talk attribution.
///
/// This fixture used to assert the OPPOSITE. `identify_sleep_talk_called`
/// regenerated every candidate against a hardcoded `&Choice::default()`, while
/// the engine gates its 32-roll damage enumeration on the real defender choice --
/// `branch_on_damage && choice.first_move && pending_hp_reading_move(defender)`,
/// where `pending_hp_reading_move` is {SUBSTITUTE, FLAIL, REVERSAL}. Defender
/// picks one, sleeper moves first: engine emits one of 32 rolls, renderer
/// regenerates the ordinary 2-branch max/crit collapse against a `NONE` move,
/// nothing matches, and the ENTIRE WORLD is refused as
/// `sleeptalk_called_unidentified:none_matched`.
///
/// Measured before the fix: 25 / 37 / 31 refused branches for Substitute / Flail
/// / Reversal with the sleeper first, 0 for every other defender move tried.
/// Reach: 294 of 1682 rolled gen3 randbats variants (17.5%) carry one of those
/// three. (That is not the probability the opponent PICKS it on a given turn,
/// which is what the mass actually depends on and which nothing here measures.)
///
/// Both halves of the fix are load-bearing and both are pinned below:
///   * the real defender choice reaches the regeneration, and
///   * the candidate inherits the outer Sleep Talk choice's `first_move`, since
///     the move table's default is `true` and a second-moving Sleep Talk
///     regenerated its callee as if it had moved first.
///
/// Petal Dance is in the callee set deliberately: its crit arm restructures the
/// tail (KO -> force-switch, no LOCKEDMOVE duration) rather than only changing a
/// damage integer. Every bug found in this area was invisible to
/// damage-integer-only fixtures.
#[test]
fn the_defenders_move_no_longer_breaks_sleeptalk_attribution() {
    for defender in [Choices::SUBSTITUTE, Choices::FLAIL, Choices::REVERSAL] {
        for sleeper_first in [true, false] {
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
            // Shape-changing callee, not just a different damage integer.
            state
                .side_two
                .get_active()
                .replace_move(PokemonMoveIndex::M3, Choices::PETALDANCE);
            state
                .side_one
                .get_active()
                .replace_move(PokemonMoveIndex::M0, defender);
            // Both move orders: `first_move` is the second half of the fix.
            state.side_one.get_active().speed = if sleeper_first { 1 } else { 500 };
            state.side_two.get_active().speed = if sleeper_first { 500 } else { 1 };

            let before = format!("{state:?}");
            let branches = generate_instructions_from_move_pair(
                &mut state,
                &MoveChoice::Move(PokemonMoveIndex::M0),
                &MoveChoice::Move(PokemonMoveIndex::M0),
                true,
            );
            assert_eq!(
                before,
                format!("{state:?}"),
                "generation mutated the source state"
            );

            let mut none_matched = 0usize;
            let mut identified = 0usize;
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
                if r.attribution_unsafe
                    .iter()
                    .any(|x| x.starts_with("sleeptalk_called_unidentified:none_matched"))
                    // NOT self-testing, and that is inherent: reverting this to equality
                    // survives the suite, because on a CORRECT build this fixture identifies
                    // its callee and the guard counts 0 either way. The difference only shows
                    // with the C31 bug present, which is how review found it -- pre-PR the test
                    // FAILED under that bug, post-PR it passed. Testing that a regression guard
                    // catches a regression requires introducing the regression.
                    //
                    // Review's 2x2, so a future reader can reproduce in one command: with the
                    // C31 bug present this test FAILS under `starts_with` and PASSES under
                    // equality; on a clean build both pass. The property that actually matters
                    // -- that the bare two-segment slug is never emitted -- is pinned at unit
                    // level by `every_shape_token_is_in_the_subcase_vocabulary`.
                {
                    none_matched += 1;
                    continue;
                }
                let text = r.lines.join("\n");
                if text.contains("|[from] Sleep Talk") {
                    identified += 1;
                }
            }

            assert_eq!(
                none_matched, 0,
                "defender {defender:?} (sleeper_first={sleeper_first}) still refuses \
                 the world: the regeneration is not seeing the real defender choice, \
                 or not inheriting first_move. Every refused world here is a whole \
                 world's search discarded."
            );
            assert!(
                identified > 0,
                "defender {defender:?} (sleeper_first={sleeper_first}) produced no \
                 identified callee at all -- the refusal is gone but so is the \
                 attribution, which is not a fix"
            );
        }
    }
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
                .any(|x| x.starts_with("sleeptalk_called_unidentified:none_matched"))
                    // NOT self-testing, and that is inherent: reverting this to equality
                    // survives the suite, because on a CORRECT build this fixture identifies
                    // its callee and the guard counts 0 either way. The difference only shows
                    // with the C31 bug present, which is how review found it -- pre-PR the test
                    // FAILED under that bug, post-PR it passed. Testing that a regression guard
                    // catches a regression requires introducing the regression.
                    //
                    // Review's 2x2, so a future reader can reproduce in one command: with the
                    // C31 bug present this test FAILS under `starts_with` and PASSES under
                    // equality; on a clean build both pass. The property that actually matters
                    // -- that the bare two-segment slug is never emitted -- is pinned at unit
                    // level by `every_shape_token_is_in_the_subcase_vocabulary`.
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

/// The SUBSTITUTE BREAK, the other half of `ambiguous_unrenderable` and the last family the
/// unnamed-callee walk could not express.
///
/// #1131 rendered `[Boost]` and took the attribution oracle from 16 unrenderable to 6. All six
/// survivors were the same shape -- `[DamageSubstitute, RemoveVolatileStatus]`, classified
/// `substitute+volatile` -- so the walk now emits `|-activate|...|Substitute|[damage]` and, for
/// the SUBSTITUTE volatile only, `|-end|...|Substitute`. Oracle: usable 231 -> 237,
/// unrenderable 6 -> 0, with `branches`, `agree` and WRONG unmoved.
///
/// The lines are asserted EXACTLY, not by prefix. The boost sibling shipped
/// `starts_with("|-boost|")` and review showed it admitting a wrong ident, a wrong stat, and a
/// spurious `[from]` tag -- the last of which makes the fold ignore the event entirely. The
/// break has the same exposure: `|-end|p1a: X|Substitute` differs from `|-end|p1a: X|move: Taunt`
/// only past the prefix, and crediting the break to the WRONG SIDE is exactly the C52-shaped
/// defect this walk exists to avoid.
#[test]
fn byte_identical_callees_that_break_a_substitute_render_both_exact_lines() {
    let mut state = confused_state(Choices::SLEEPTALK);
    state
        .side_two
        .volatile_statuses
        .remove(&PokemonVolatileStatus::CONFUSION);
    state.side_two.get_active().status = PokemonStatus::SLEEP;
    state.side_two.get_active().rest_turns = 0;
    // Byte-identical callees, so identification is genuinely Ambiguous: same type, power and
    // category means the two branches differ in no observable byte.
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::TACKLE);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M2, Choices::SCRATCH);
    // A substitute thin enough that either callee BREAKS it, which is what pairs
    // `DamageSubstitute` with `RemoveVolatileStatus(SUBSTITUTE)` in one tail. A fat
    // substitute yields a HIT only, which was already renderable and would make this test
    // pass without exercising the break arm at all.
    state
        .side_one
        .volatile_statuses
        .insert(PokemonVolatileStatus::SUBSTITUTE);
    state.side_one.substitute_health = 1;

    let branches = generate(&mut state);
    let mut saw_break = false;
    let mut saw_ambiguous = false;
    for branch in &branches {
        let r = rendered(&mut state.clone(), branch);
        if !r
            .lossy_subcases
            .iter()
            .any(|x| x == "sleeptalk_called_unidentified:ambiguous")
        {
            continue;
        }
        saw_ambiguous = true;
        // NOT refused. This is the whole point: before this change these branches landed in
        // `attribution_unsafe` as `:ambiguous_unrenderable` and every world using them was
        // thrown away.
        assert!(
            !r.attribution_unsafe
                .iter()
                .any(|x| x.starts_with("sleeptalk_called_unidentified")),
            "a substitute-break tail is renderable now and must not refuse: {:?}",
            r.attribution_unsafe
        );
        // The differential's contract tag stays bare AND alone -- it gates usability on exact
        // set equality, so one extra entry silently deletes this change's entire benefit.
        assert!(
            r.lossy.iter().any(|x| x == "sleeptalk_called_unidentified"),
            "the usable arm must still carry the bare lossy tag: {:?}",
            r.lossy
        );
        assert!(
            !r.lossy
                .iter()
                .any(|x| x.starts_with("sleeptalk_called_unidentified:")),
            "a sub-cased lossy tag on the USABLE arm changes which branches the \
             differential accepts: {:?}",
            r.lossy
        );
        // Neither candidate may be NAMED, or the render invents evidence it does not have.
        for callee in ["tackle", "scratch"] {
            assert!(
                !r.lines
                    .iter()
                    .any(|line| line.starts_with("|move|p2a:") && line.contains(callee)),
                "an ambiguous callee must stay unnamed, saw {callee}: {:?}",
                r.lines
            );
        }
        if r.lines
            .iter()
            .any(|line| line == "|-end|p1a: Lead|Substitute")
        {
            saw_break = true;
            // ORDER, which is why both arms call `emit_residuals!()` first and why the hit
            // arm sits ahead of the break arm in the chain. Showdown reports the damage
            // before the substitute falls; emitting `-end` first is a protocol log no real
            // battle produces, and #1131 shipped exactly this defect for `[Damage, Boost..]`
            // with the whole suite green.
            let hit = r
                .lines
                .iter()
                .position(|line| line == "|-activate|p1a: Lead|Substitute|[damage]")
                .expect("a break must be preceded by the hit that caused it");
            let end = r
                .lines
                .iter()
                .position(|line| line == "|-end|p1a: Lead|Substitute")
                .unwrap();
            assert!(
                hit < end,
                "the substitute hit must precede the break: {:?}",
                r.lines
            );
            // The break belongs to the DEFENDER. Crediting it to the sleeping attacker is the
            // single most likely wiring error here -- `side_ref` versus `defender` -- and it
            // survives any prefix assertion.
            assert!(
                !r.lines
                    .iter()
                    .any(|line| line == "|-end|p2a: Opponent|Substitute"),
                "the break was credited to the wrong side: {:?}",
                r.lines
            );
        }
    }
    assert!(
        saw_ambiguous,
        "VACUOUS: no branch was ambiguous, so nothing here exercised the walk"
    );
    assert!(
        saw_break,
        "VACUOUS: no branch rendered the substitute break, so the new arm never ran \
         and this test would pass with it deleted"
    );
}

/// A PHAZE must keep refusing: `RemoveVolatileStatus(SUBSTITUTE)` has two producers and only
/// one of them is a break Showdown narrates.
///
/// This is review's reproduction of a defect in the first version of the substitute-break
/// change, kept as a test because the defect is invisible from the instruction alone:
///
///   * `generate_instructions.rs` emits the removal right after a same-side
///     `DamageSubstitute`. That is a real break and Showdown runs `onEnd`, emitting
///     `|-end|<ident>|Substitute`.
///   * `state.rs`'s `remove_volatile_statuses_on_switch` emits the SAME variant on every
///     non-Baton-Pass switch-out, phazing drags included. Showdown clears volatiles there with
///     `this.volatiles = {}` and never runs `onEnd`, so it emits NOTHING. (`gen3_phaze_fidelity`
///     separately pins that a Substitute does not block a phaze at all, via `bypasssub`.)
///
/// Keying admission on the volatile identity alone therefore rendered a PHANTOM `|-end|` on a
/// `[RemoveVolatileStatus(SUBSTITUTE), Switch]` tail and SEARCHED a world that used to refuse.
/// An extra line is the same defect class as a missing one -- a wrong world, not a refused one
/// -- which is the exact harm the change was written to avoid.
///
/// Not reachable in today's gen3 randbats: review checked all three cached universes (6,364
/// variants) and ZERO sets pair Sleep Talk with Roar or Whirlwind. The test exists anyway,
/// because the file's own rule is that "the predicate blocks those anyway" is a reachability
/// argument and not an invariant, and because the set list is data that can change under us.
#[test]
fn a_phaze_that_clears_a_substitute_keeps_refusing_and_renders_no_end_line() {
    let mut state = confused_state(Choices::SLEEPTALK);
    state
        .side_two
        .volatile_statuses
        .remove(&PokemonVolatileStatus::CONFUSION);
    state.side_two.get_active().status = PokemonStatus::SLEEP;
    state.side_two.get_active().rest_turns = 0;
    // Two phazing callees, so identification is Ambiguous and the unnamed walk is reached.
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::ROAR);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M2, Choices::WHIRLWIND);
    // The phazed side holds a Substitute, so its switch-out pushes the removal with NO
    // `DamageSubstitute` anywhere in the tail.
    state
        .side_one
        .volatile_statuses
        .insert(PokemonVolatileStatus::SUBSTITUTE);
    state.side_one.substitute_health = 40;

    let branches = generate(&mut state);
    let mut saw_refusal = false;
    for branch in &branches {
        let r = rendered(&mut state.clone(), branch);
        // NO `|-end|Substitute` on ANY branch, refused or not. Showdown emits nothing for a
        // switch-out volatile clear, so this line is a phantom wherever it appears.
        assert!(
            !r.lines.iter().any(|line| line.contains("|Substitute")),
            "a switch-out substitute clear must render NO substitute line: {:?}",
            r.lines
        );
        if r.attribution_unsafe
            .iter()
            .any(|x| x == "sleeptalk_called_unidentified:ambiguous_unrenderable:volatile")
        {
            saw_refusal = true;
        }
    }
    assert!(
        saw_refusal,
        "VACUOUS: no branch refused under `volatile`, so this fixture no longer exercises \
         the phaze case and the phantom-line guard above is unexercised"
    );
}

/// A phaze must not render a phantom `|-unboost|` either: `Boost` has the same two-producer
/// problem the substitute break has, and #1131 admitted it unconditionally.
///
///   * A move's own stat change. Showdown narrates `|-boost|` / `|-unboost|`.
///   * The switch path's `reset_boosts(&switching_side_ref, ..)`, called when
///     `!baton_passing` in the pre-switch block. Showdown drops boosts inside
///     `clearVolatile()` and narrates NOTHING.
///
/// This crate's own `render_switch_phase` already discriminates correctly — it renders only
/// the Intimidate case and drops the rest through an arm commented "Pre-switch bookkeeping
/// (volatile clears, boost resets, ...): no lines." The unnamed-callee walk contradicted it.
///
/// The fixture carries a live `attack_boost`, so the drag emits `Boost(SideOne, Attack, -2)`.
/// The tail must REFUSE (family `boost`) rather than render silence — see
/// `boost_may_be_a_switch_out_reset` for why fail-closed is the right disposition when the
/// alternative rests on a reachability argument.
#[test]
fn a_phaze_that_resets_boosts_renders_no_unboost_line() {
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
        .replace_move(PokemonMoveIndex::M1, Choices::ROAR);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M2, Choices::WHIRLWIND);
    // A live boost on the side about to be dragged, so the switch-out reset is nonzero.
    state.side_one.attack_boost = 2;

    let branches = generate(&mut state);
    let mut saw_refusal = false;
    for branch in &branches {
        let r = rendered(&mut state.clone(), branch);
        // No boost line of EITHER polarity, on any branch, refused or not — the walk runs
        // either way and Showdown emits nothing for a switch-out reset.
        assert!(
            !r.lines
                .iter()
                .any(|line| line.starts_with("|-boost|") || line.starts_with("|-unboost|")),
            "a switch-out boost reset must render NO boost line: {:?}",
            r.lines
        );
        if r.attribution_unsafe
            .iter()
            .any(|x| x.starts_with("sleeptalk_called_unidentified:ambiguous_unrenderable:"))
        {
            saw_refusal = true;
        }
    }
    // BOTH guards do independent work, verified by removing each half of the fix:
    //   * classifier AND walk arm removed (the pre-fix state) -> the phantom appears, and the
    //     assertion above fires on `|-unboost|p1a: Lead|atk|2` sitting before the drag.
    //   * classifier removed, walk arm kept -> no phantom, but the tail is ADMITTED and the
    //     walk renders nothing, i.e. a searched world with a MISSING line. That is a distinct
    //     defect of the same class, and only this vacuity guard catches it.
    // So neither assertion is redundant, which is not obvious from reading them.
    assert!(
        saw_refusal,
        "VACUOUS: no branch refused, so either the fixture no longer reaches the reset case \
         or the tail is being admitted and silently under-rendered"
    );
}

/// The direct self-heal renders EXACTLY `|-heal|{ident}|{condition}` — no `[from]`, right side.
///
/// This is the pin the heal change shipped without, and review demonstrated the cost: with the
/// classifier and predicate well tested but the RENDER ARM untested, four mutations produced
/// silently SEARCHED wrong worlds against 32 green suites —
///
///   * appending `|[from] ability: Volt Absorb|[of] ...` to the line;
///   * deleting the `out.lines.push` outright;
///   * crediting the heal to `other_side(side)`;
///   * passing index `0` instead of `index` to the predicate.
///
/// The first is the exact failure the whole change is argued on — the fold reads `[from]`, so a
/// fabricated tag FABRICATES a belief. The third is the defect `emit_residuals!()`'s own comment
/// records as having shipped once before: "Rendering the heal direction was shipped once and
/// emitted lines for the wrong Pokemon."
///
/// Recover and Soft-Boiled are byte-identical (both heal 50% of max HP), so the branch is
/// genuinely Ambiguous and reaches the unnamed-callee walk. The sleeper is put below max HP or
/// the heal is a no-op and the fixture proves nothing.
#[test]
fn a_direct_self_heal_renders_the_exact_bare_line_on_the_healed_side() {
    let mut state = confused_state(Choices::SLEEPTALK);
    state
        .side_two
        .volatile_statuses
        .remove(&PokemonVolatileStatus::CONFUSION);
    state.side_two.get_active().status = PokemonStatus::SLEEP;
    state.side_two.get_active().rest_turns = 0;
    state.side_two.get_active().hp = state.side_two.get_active().maxhp / 2;
    // The OPPONENT sits at a DIFFERENT HP from the healer's post-heal total, deliberately.
    // SAME FAILURE MODE as the `cross_side` control in
    // `the_renderable_allowlist_is_exactly_what_it_was`, one screen away in events.rs: there,
    // every fixture paired SideOne with SideOne, so replacing the predicate's
    // `switch.side_ref == boost.side_ref` with `true` survived the whole suite. An assertion
    // whose two sides COINCIDE in the fixture cannot see a mutant that swaps them.
    // With both sides at 100/100 the condition string is identical either way, so a mutant
    // reading `hp_condition(other_side(side))` renders a correct-looking line and survives --
    // review found exactly that. 70 against 100 makes the wrong side observable.
    state.side_one.get_active().hp = 70;
    state.side_one.get_active().maxhp = 100;
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::RECOVER);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M2, Choices::SOFTBOILED);

    let branches = generate(&mut state);
    let mut saw_heal = false;
    for branch in &branches {
        let r = rendered(&mut state.clone(), branch);
        if !r
            .lossy_subcases
            .iter()
            .any(|x| x == "sleeptalk_called_unidentified:ambiguous")
        {
            continue;
        }
        // WHOLE-LINE EQUALITY, matching the three sibling pins in this file, and NOT a
        // structural check over every `-heal` line in the branch. Review found both halves of
        // that mistake:
        //
        //   * Asserting the tag, the side and the FIELD COUNT left the condition VALUE
        //     unasserted, and two reachable mutants exploited it. Reading
        //     `sim.hp_condition(side)` BEFORE `sim.apply(instruction)` renders the PRE-heal HP
        //     (`50/100`) into a SEARCHED world -- the C52 defect class this walk documents,
        //     where a stale consumer baseline surfaces later as an impossible component. And
        //     `hp_condition(other_side(side))` puts the opponent's HP inside the attacker's
        //     heal line: right ident, no tag, four fields, wrong number.
        //   * Looping over every `-heal|` line in the BRANCH blamed this arm for lines it did
        //     not emit. With Leftovers on the opponent -- a large share of real gen3 sets --
        //     an ordinary end-of-turn tick `|-heal|p1a: Lead|56/100|[from] item: Leftovers`
        //     failed the pin with a message accusing the direct-self-heal arm of fabricating
        //     a belief.
        //
        // The fixture is fully deterministic: maxhp 100, hp set to 50, Recover heals 50.
        if r.lines.iter().any(|l| l == "|-heal|p2a: Opponent|100/100") {
            saw_heal = true;
        }
    }
    assert!(
        saw_heal,
        "VACUOUS: no branch rendered the exact line `|-heal|p2a: Opponent|100/100`, so either \
         the render arm never ran or it emitted something else -- both of which this test \
         exists to catch"
    );
}


/// The `none_matched` shape must be pinned END TO END, through the real
/// `identify_sleep_talk_called` aggregation — not just on the pure classifier.
///
/// Review's mutation battery found every kill landing on the pure helper or the enum, and
/// every SURVIVOR in the aggregation that actually computes what an era reads:
///
///   * swapping `(branch, tail)` at the `divergence_shape` call — SURVIVED;
///   * `min` → `max`, which inverts the whole measurement to report the FARTHEST miss — SURVIVED;
///   * changing the seed — SURVIVED.
///
/// `min` → `max` is the serious one: the reported shape becomes near-universally the least
/// informative bucket, with a fully green suite. The PR claimed the ordering was "pinned"; the
/// ORDERING was, the USE of `min` was not, and no test crossed the boundary.
///
/// This drives the C31 fixture — the one `none_matched` population this repo can reproduce —
/// and asserts the shape the aggregation actually emits.
#[test]
fn the_emitted_none_matched_shape_comes_from_the_real_aggregation() {
    let mut state = confused_state(Choices::SLEEPTALK);
    state
        .side_two
        .volatile_statuses
        .remove(&PokemonVolatileStatus::CONFUSION);
    state.side_two.get_active().status = PokemonStatus::SLEEP;
    state.side_two.get_active().rest_turns = 0;
    // Two byte-identical damaging callees against a defender whose own move gates the engine's
    // 32-roll enumeration. This is the C31 shape.
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::BODYSLAM);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M2, Choices::EARTHQUAKE);

    let branches = generate(&mut state);
    let mut shapes: Vec<String> = Vec::new();
    for branch in &branches {
        let r = rendered(&mut state.clone(), branch);
        for reason in &r.attribution_unsafe {
            if reason.starts_with("sleeptalk_called_unidentified:none_matched") {
                shapes.push(reason.clone());
            }
        }
    }

    // On a CORRECT build this fixture identifies its callee, so there is nothing to classify.
    // That is the honest state of affairs and it is asserted rather than left implicit: the
    // shape aggregation has no naturally-occurring input in this repo, which is exactly why
    // review could invert it undetected and why the ceiling on this test is what it is.
    if shapes.is_empty() {
        return;
    }
    // If any DID refuse, every emitted slug must be a registered three-segment shape token --
    // never the bare two-segment form, which is what the old equality-based guards matched and
    // which would mean the sub-casing silently regressed.
    for slug in &shapes {
        assert_ne!(
            slug, "sleeptalk_called_unidentified:none_matched",
            "the bare slug carries no shape and cannot be ranked"
        );
        assert!(
            slug.contains(":shape_"),
            "every none_matched slug must name a registered shape: {slug}"
        );
    }
}

/// A Protect-blocked Sleep Talk callee renders `|-activate|<target>|Protect`.
///
/// END-TO-END through the real render path, which is the point. The unit tests for
/// `protect_blocked_marker_side` exercise a pure function; review demonstrated that
/// DELETING THE ENTIRE WALK ARM left all 418 tests green, and that lowercasing the
/// keyword or hardcoding the PROTECT read also survived. Every one of those mutants
/// has to die here or the behaviour change is unverified.
///
/// The shape: the sleeper calls a protect-flagged move via Sleep Talk, the defender is
/// behind Protect, so gen3 pushes a zero-amount `Heal` on the DEFENDER as a branch
/// marker (generate_instructions.rs, gated on `blocked_by_protect`). Before this fix
/// that marker refused the whole world as `ambiguous_unrenderable:heal_zero_marker`.
#[test]
fn a_protect_blocked_sleep_talk_callee_renders_the_exact_protect_line() {
    let mut state = confused_state(Choices::SLEEPTALK);
    state.side_two.get_active().status = PokemonStatus::SLEEP;
    state.side_two.get_active().sleep_turns = 0;
    // TWO protect-flagged damaging moves, which is what makes the callee AMBIGUOUS and so
    // routes this through the UNNAMED-CALLEE WALK rather than the named path. Protect
    // STRIPS the move, so both candidates regenerate the identical tail -- just the
    // zero-amount `Heal` marker -- and `identify_sleep_talk_called` cannot tell them apart.
    //
    // That is not a contrivance, it is the mechanism behind the whole 3,365-world class:
    // Protect makes every protect-flagged callee look the same, which is exactly why these
    // worlds were ambiguous and refused. A FIRST VERSION of this fixture gave the sleeper
    // ONE callee, which `identify_sleep_talk_called` named -- so the NAMED path rendered
    // Protect, the fixture passed, and deleting the entire new walk arm still passed.
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::TACKLE);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M2, Choices::SCRATCH);
    // THE DISCRIMINATOR, set explicitly. This is the state fact the tail cannot supply
    // and the whole safety argument rests on: without it the marker is indistinguishable
    // from a full-HP absorb activation, whose line is an ABILITY activation.
    state
        .side_one
        .volatile_statuses
        .insert(PokemonVolatileStatus::PROTECT);

    let branches = generate(&mut state);
    let mut saw_protect = false;
    for branch in &branches {
        let events = render(&mut state, branch);
        if events.contains("Protect") {
            saw_protect = true;
            // EXACT string, not a substring match on "Protect". A lowercased keyword or a
            // dropped `-activate` both survived review's mutation battery precisely
            // because nothing asserted the literal.
            assert!(
                events.contains("|-activate|p1a: Lead|Protect"),
                "expected the exact Protect activation line, got:\n{events}"
            );
        }
    }
    assert!(
        saw_protect,
        "no branch rendered a Protect line -- the walk arm did not fire. Branches: {}",
        branches.len()
    );
    // AND it must be the UNNAMED path. If a callee gets NAMED, the named renderer emits the
    // Protect line and this fixture would pass with the new arm deleted -- which is exactly
    // what the first version did.
    //
    // AND the world must actually be RECLAIMED. This is the PR's headline benefit and it was
    // unverified: `sleeptalk_refusal_is_unsafe` only chooses WHICH tag is emitted, and the
    // walk renders either way -- so reverting it to the fail-closed form left every world
    // still refused with the whole suite green (review's M19). The tag is the only thing
    // that distinguishes "rendered and searched" from "rendered and thrown away".
    let mut checked = 0;
    for branch in &branches {
        let r = rendered(&mut state, branch);
        let events = r.lines.join("\n");
        if events.contains("Protect") {
            assert!(
                !events.contains("[from] Sleep Talk"),
                "the callee was NAMED, so the named path rendered Protect and this fixture \
                 proves nothing about the walk arm:\n{events}"
            );
            assert!(
                r.attribution_unsafe.is_empty(),
                "the Protect marker is rendered but the world is STILL REFUSED as \
                 attribution-unsafe, so the fix reclaims nothing: {r:?}"
            );
            // AND it must be counted EXACTLY ONCE PER RENDERED LINE. Closing this family
            // deleted the only number that tracked it: era 62 measured the shape at 3,365
            // worlds solely because it aborted into `world_failure_reasons`.
            //
            // COUNT, not presence. `.any()` was the first version of this assertion and
            // review killed it: duplicating the `mark_lossy_subcase` call passed all 423
            // tests. Over-firing is the failure direction the whole change is most exposed
            // to, because the number it inflates is the one the next era reads as evidence
            // the fix worked -- and M3 only covers over-firing on the WRONG branch, not
            // twice on the right one.
            // KNOWN COVERAGE LIMIT, stated rather than implied. `markers` is 1 in every
            // branch this fixture generates, so this is operationally `counted == 1` and it
            // does NOT distinguish once-per-line from once-per-branch: review's N3, which
            // hoists the push to a single one-per-branch call, survives all 423 tests. It
            // does kill the duplicate-push mutant, which is the failure direction named
            // above.
            //
            // The denominator is exact despite being a substring match. The renderer has
            // exactly four `|Protect` producers and all four are `|-activate|{ident}|Protect`
            // -- `|-singleturn|...|Protect` is never emitted, and move ids render lowercase
            // so `|move|...|protect` cannot collide with a case-sensitive match. THREE of
            // the four are NAMED-path sites this counter must never count; the clean-branch
            // pin in `protected_memento_emits_one_protect_activation` is what holds that.
            let markers = events.matches("|Protect").count();
            let counted = r
                .lossy_subcases
                .iter()
                .filter(|s| *s == "sleeptalk_called_unidentified:protect_marker_rendered")
                .count();
            assert_eq!(
                counted, markers,
                "the Protect counter must fire exactly once per rendered Protect line, \
                 got {counted} for {markers} line(s): {r:?}"
            );
            assert!(
                counted > 0,
                "the Protect marker rendered but emitted NO telemetry, so a production era \
                 cannot tell a firing marker from a silent one: {r:?}"
            );
            // AND `set(lossy)` must be UNCHANGED. This is the contract
            // `engine_transition_differential.py` matches exactly
            // (`set(lossy) == {_SLEEPTALK_LOSSY_MARKER}`) to decide branch usability, and
            // that file's bytes are pinned by the certification lifecycle, so it cannot be
            // edited to follow a renderer change. Counting via a NEW lossy tag would
            // silently narrow which branches the differential accepts -- green here, wrong
            // in production. Pinned as a SET so a repeated push of the same tag passes and
            // an added distinct tag fails.
            let distinct: std::collections::BTreeSet<&str> =
                r.lossy.iter().map(String::as_str).collect();
            assert_eq!(
                distinct,
                ["sleeptalk_called_unidentified"].into_iter().collect(),
                "the Protect counter changed set(lossy); the transition differential's \
                 acceptance set moves with it: {r:?}"
            );
            checked += 1;
        }
    }
    assert!(checked > 0, "no branch reached the tag assertion");
}

/// WITHOUT the PROTECT volatile, no Protect line is rendered.
///
/// The negative half, and the one that pins the SAFETY ARGUMENT. The positive fixture
/// always sets the volatile, so hardcoding `defender_protected = true` survives it --
/// review's M14. This is the case that mutant breaks: a zero-amount `Heal` with no
/// Protect in state is the OTHER producer, a full-HP absorb activation, whose line is an
/// ABILITY activation. Rendering `|-activate|...|Protect` there fabricates a line in a
/// searched world, which is the defect that hit the substitute-break and boost arms.
#[test]
fn without_the_protect_volatile_no_protect_line_is_invented() {
    let mut state = confused_state(Choices::SLEEPTALK);
    state.side_two.get_active().status = PokemonStatus::SLEEP;
    state.side_two.get_active().sleep_turns = 0;
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::TACKLE);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M2, Choices::SCRATCH);
    // NO PROTECT volatile inserted -- that is the whole point.

    let branches = generate(&mut state);
    for branch in &branches {
        let r = rendered(&mut state, branch);
        let events = r.lines.join("\n");
        assert!(
            !events.contains("Protect"),
            "a Protect line was invented with no PROTECT volatile in state:\n{events}"
        );
        // The COUNTER must be silent too. A counter that fires without a rendered line
        // inflates the one number era 63 reads to decide whether the marker works, and
        // it inflates it in the direction that manufactures a win.
        assert!(
            !r.lossy_subcases
                .iter()
                .any(|s| s == "sleeptalk_called_unidentified:protect_marker_rendered"),
            "the Protect counter fired with no Protect line rendered: {r:?}"
        );
    }
}

/// PRODUCER 2, the one that would be FABRICATED over: a full-HP absorb activation.
///
/// This is the case the whole absorb guard exists for, and review showed it was untested:
/// hardcoding BOTH guards true made this exact branch emit `|-activate|...|Protect` with the
/// entire suite green. A Water Absorb defender at full HP takes a Water move and gen3 pushes
/// a zero-amount `Heal` on it -- byte-identical to the Protect marker, on the same side --
/// but the line it owes is an ABILITY activation, not Protect.
///
/// No PROTECT volatile here, and the defender holds WATERABSORB, so BOTH axes of the guard
/// must hold for nothing to be rendered.
#[test]
fn a_full_hp_absorb_activation_is_never_dressed_as_protect() {
    let mut state = confused_state(Choices::SLEEPTALK);
    state.side_two.get_active().status = PokemonStatus::SLEEP;
    state.side_two.get_active().sleep_turns = 0;
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::WATERGUN);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M2, Choices::BUBBLE);
    // The OTHER producer: a heal-carrying absorb ability on a full-HP defender.
    state.side_one.get_active().ability = Abilities::WATERABSORB;
    let maxhp = state.side_one.get_active().maxhp;
    state.side_one.get_active().hp = maxhp;

    let branches = generate(&mut state);
    for branch in &branches {
        let r = rendered(&mut state, branch);
        let events = r.lines.join("\n");
        assert!(
            !events.contains("Protect"),
            "a full-HP absorb activation was dressed as Protect -- this is the fabricated \
             line the absorb guard exists to prevent:\n{events}"
        );
        // The counter is silent here for the same reason the LINE is: this zero-amount
        // Heal is byte-identical to a Protect marker and is NOT one. A counter that
        // cannot tell the two producers apart measures the wrong population.
        assert!(
            !r.lossy_subcases
                .iter()
                .any(|s| s == "sleeptalk_called_unidentified:protect_marker_rendered"),
            "the Protect counter fired on a full-HP absorb activation: {r:?}"
        );
    }
}

/// PROTECT *and* an absorb ability: refuse. The over-refusal, pinned as deliberate.
///
/// A Water Absorb mon that uses Protect hits this routinely -- it is not a corner case, and
/// review measured it as a real ongoing cost: those worlds stay refused even though the
/// marker is almost certainly a genuine Protect block. The guard is kept anyway because the
/// absorb abilities RESTORE `flags.protect` (gen3/abilities.rs), so a protect-bypassing Water
/// or Electric move would leave `blocked_by_protect == false` with PROTECT still set -- a
/// zero `Heal` that is NOT a Protect marker while the volatile says otherwise.
///
/// Pinning it as a TEST rather than a comment because hardcoding the absorb read to `false`
/// at the call site survived every other fixture (review's M14b). Without this, the axis
/// could be silently deleted.
#[test]
fn protect_plus_an_absorb_ability_refuses_rather_than_guessing() {
    let mut state = confused_state(Choices::SLEEPTALK);
    state.side_two.get_active().status = PokemonStatus::SLEEP;
    state.side_two.get_active().sleep_turns = 0;
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::TACKLE);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M2, Choices::SCRATCH);
    state
        .side_one
        .volatile_statuses
        .insert(PokemonVolatileStatus::PROTECT);
    // BOTH conditions true at once. The positive fixture has PROTECT and no absorb; this
    // adds the absorb, and the expected outcome flips from rendered to refused.
    state.side_one.get_active().ability = Abilities::WATERABSORB;

    let branches = generate(&mut state);
    for branch in &branches {
        let events = render(&mut state, branch);
        assert!(
            !events.contains("Protect"),
            "with a heal-carrying absorb ability present the marker is ambiguous and must \
             be refused, not guessed:\n{events}"
        );
    }
}
