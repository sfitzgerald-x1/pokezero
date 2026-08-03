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
use pokezero_search::events::{render_branch_events, EventContext};

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
    // This fixture is attracted AND paralyzed, which is the arm that decides
    // whether a cheap lossy downgrade is available.
    assert!(
        reasons
            .iter()
            .any(|reason| reason == "attract_empty_tail_ambiguous:paralyzed"),
        "{reasons:?}"
    );
}
