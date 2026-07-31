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
use poke_engine::state::{PokemonMoveIndex, PokemonStatus, PokemonType, SideReference, State};
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
    assert_eq!(before, format!("{state:?}"), "generation mutated the source state");
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
    assert_eq!(before, state.serialize(), "rendering mutated the source state");
    rendered.lines.join("\n")
}

fn rendered(state: &mut State, branch: &StateInstructions) -> pokezero_search::events::RenderedEvents {
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
    assert_eq!(before, state.serialize(), "rendering mutated the source state");
    rendered
}

fn damage_to(branch: &StateInstructions, side: SideReference, amount: i16) -> bool {
    branch.instruction_list.iter().any(|instruction| {
        matches!(instruction, Instruction::Damage(damage)
            if damage.side_ref == side && damage.damage_amount == amount)
    })
}

fn expires_confusion(branch: &StateInstructions) -> bool {
    branch.instruction_list.iter().any(|instruction| {
        matches!(instruction, Instruction::RemoveVolatileStatus(remove)
            if remove.side_ref == SideReference::SideTwo
                && remove.volatile_status == PokemonVolatileStatus::CONFUSION)
    })
}

fn self_hit_branch(branches: &[StateInstructions], amount: i16) -> &StateInstructions {
    branches
        .iter()
        .find(|branch| damage_to(branch, SideReference::SideTwo, amount))
        .expect("expected confusion self-hit branch")
}

#[test]
fn exact_self_hit_renders_activation_and_cancels_substitute() {
    let mut state = confused_state(Choices::SUBSTITUTE);
    state.side_two.get_active().maxhp = 256;
    state.side_two.get_active().hp = 200;
    state.side_two.get_active().attack = 108; // exact 38 damage
    let branches = generate(&mut state);
    let events = render(&mut state, self_hit_branch(&branches, 38));
    assert!(events.contains("|-activate|p2a: Opponent|confusion"), "{events}");
    assert!(events.contains("|-damage|p2a: Opponent|162/256"), "{events}");
    assert!(!events.contains("|move|p2a: Opponent|substitute"), "{events}");
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
        assert!(events.contains("|-activate|p2a: Opponent|confusion"), "{events}");
        assert!(
            !events.contains(&format!("|move|p2a: Opponent|{}", format!("{move_id:?}").to_lowercase())),
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
            < choice_band_hit.instruction_list.iter().position(|instruction| {
                matches!(instruction, Instruction::ChangeVolatileStatusDuration(change)
                    if change.volatile_status == PokemonVolatileStatus::CONFUSION
                        && change.amount == 1)
            }),
        "{choice_band_hit:?}"
    );
    let choice_band_events = render(&mut choice_band, choice_band_hit);
    assert!(!choice_band_events.contains("|move|p2a: Opponent|tackle"), "{choice_band_events}");

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
    assert!(!locked_events.contains("|move|p2a: Opponent|outrage"), "{locked_events}");

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
    assert!(!future_events.contains("|move|p2a: Opponent|futuresight"), "{future_events}");
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
    assert!(events.contains("|move|p2a: Opponent|highjumpkick|p1a: Lead|[miss]"), "{events}");
    assert!(events.contains("|-damage|p2a: Opponent|50/100|[from] highjumpkick"), "{events}");
    assert!(events.contains("|-activate|p2a: Opponent|confusion"), "{events}");
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
    assert!(events.contains("|move|p2a: Opponent|doubleedge|p1a: Lead"), "{events}");
    assert!(events.contains("[from] Recoil"), "{events}");
    assert!(events.contains("|-activate|p2a: Opponent|confusion"), "{events}");
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
    assert!(events.contains("|move|p2a: Opponent|explosion|p1a: Lead"), "{events}");
    assert!(events.contains("|-activate|p1a: Lead|Protect"), "{events}");
    assert!(events.contains("|faint|p2a: Opponent"), "{events}");
    assert!(events.contains("|-activate|p2a: Opponent|confusion"), "{events}");
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
    assert!(events.contains("|move|p2a: Opponent|explosion|p1a: Lead"), "{events}");
    assert!(events.contains("|-immune|p1a: Lead"), "{events}");
    assert!(events.contains("|faint|p2a: Opponent"), "{events}");
    assert!(events.contains("|-activate|p2a: Opponent|confusion"), "{events}");
}

#[test]
fn lethal_self_hit_still_emits_confusion_then_faint() {
    let mut state = confused_state(Choices::SUBSTITUTE);
    state.side_two.get_active().hp = 20;
    let branches = generate(&mut state);
    let events = render(&mut state, self_hit_branch(&branches, 20));
    assert!(events.contains("|-activate|p2a: Opponent|confusion"), "{events}");
    assert!(events.contains("|-damage|p2a: Opponent|0 fnt"), "{events}");
    assert!(events.contains("|faint|p2a: Opponent"), "{events}");
    assert!(!events.contains("|move|p2a: Opponent|substitute"), "{events}");
}

#[test]
fn confusion_expiry_and_survival_keep_their_distinct_protocol_markers() {
    for previous_turns in 0..=3 {
        let mut state = confused_state(Choices::SPLASH);
        state.side_two.volatile_status_durations.confusion = previous_turns;
        let branches = generate(&mut state);
        let expires = branches
            .iter()
            .find(|branch| {
                expires_confusion(branch)
                    && branch.instruction_list.iter().any(|instruction| {
                        matches!(instruction, Instruction::ChangeVolatileStatusDuration(change)
                            if change.side_ref == SideReference::SideTwo
                                && change.volatile_status == PokemonVolatileStatus::CONFUSION
                                && change.amount == -(previous_turns + 1))
                    })
            })
            .expect("expected residual expiry branch");
        let expire_events = render(&mut state, expires);
        assert_eq!(
            expire_events.matches("|-end|p2a: Opponent|confusion").count(),
            1,
            "{expire_events}"
        );
    }

    let mut state = confused_state(Choices::SPLASH);
    let branches = generate(&mut state);
    let survives = branches
        .iter()
        .find(|branch| damage_to(branch, SideReference::SideTwo, 35) && !expires_confusion(branch))
        .expect("expected surviving self-hit");
    let survive_events = render(&mut state, survives);
    assert!(!survive_events.contains("|-end|p2a: Opponent|confusion"), "{survive_events}");
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
    let activation = events.find("|-activate|p2a: Opponent|confusion").expect("activation");
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
    assert!(events.contains("|-activate|p2a: Opponent|confusion"), "{events}");
    assert!(events.contains("|cant|p2a: Opponent|Attract"), "{events}");
    assert!(!events.contains("|-activate|p2a: Opponent|move: Attract"), "{events}");
    assert!(!events.contains("|move|p2a: Opponent|substitute"), "{events}");
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
    assert!(self_hit_events.contains("|-activate|p2a: Opponent|confusion"), "{self_hit_events}");
    assert!(!self_hit_events.contains("|cant|p2a: Opponent|slp"), "{self_hit_events}");
    assert!(!self_hit_events.contains("|move|p2a: Opponent|sleeptalk"), "{self_hit_events}");

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
    assert_eq!(called_render.lossy, Vec::<String>::new(), "{called_render:?}");
    assert!(!called_events.contains("|cant|p2a: Opponent|slp"), "{called_events}");
    let activation = called_events
        .find("|-activate|p2a: Opponent|confusion")
        .expect("activation");
    let sleep_talk = called_events
        .find("|move|p2a: Opponent|sleeptalk|p2a: Opponent")
        .expect("Sleep Talk");
    let substitute = called_events
        .find("|move|p2a: Opponent|substitute|p2a: Opponent|[from] Sleep Talk")
        .expect("called Substitute");
    assert!(activation < sleep_talk && sleep_talk < substitute, "{called_events}");
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
    assert!(events.contains("|-activate|p2a: Opponent|confusion"), "{events}");
    assert!(!events.contains("|cant|p2a: Opponent|slp"), "{events}");
}

#[test]
fn own_tempo_never_creates_a_confusion_activation() {
    let mut state = confused_state(Choices::SUBSTITUTE);
    state.side_two.get_active().ability = Abilities::OWNTEMPO;
    let branches = generate(&mut state);
    assert_eq!(branches.len(), 1, "Own Tempo must not branch: {branches:?}");
    let events = render(&mut state, &branches[0]);
    assert!(events.contains("|move|p2a: Opponent|substitute|p2a: Opponent"), "{events}");
    assert!(!events.contains("|-activate|p2a: Opponent|confusion"), "{events}");
}

#[test]
fn collapsed_crash_collision_is_fold_isolated_and_lossy() {
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
            .lossy
            .iter()
            .any(|reason| reason == "confusion_selfhit_ambiguous_executed_self_damage"),
        "{rendered:?}"
    );
    assert!(!events.contains("|-activate|p2a: Opponent|confusion"), "{events}");
    assert!(!events.contains("|move|p2a: Opponent|highjumpkick"), "{events}");
    assert!(events.lines().any(|line| line == "|"), "{events}");
}
