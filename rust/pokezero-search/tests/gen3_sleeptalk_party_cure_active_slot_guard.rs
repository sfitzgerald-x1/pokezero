//! The pre-move phase boundary when a Sleep Talk callee is a PARTY-WIDE CURE.
//!
//! # The defect these fixtures pin
//!
//! `consume_move_prelude`'s wake and thaw arms matched
//! `ChangeStatus { SLEEP|FREEZE -> NONE }` on **`side_ref` alone**, never on
//! `pokemon_index`. gen3's party-wide cures walk `pokemon_index_iter()` in SLOT ORDER
//! (`gen3/choice_effects.rs`, `Choices::HEALBELL` / `Choices::AROMATHERAPY`) and emit one clear
//! per statused party member, so when a BENCHED member sits at a lower slot than the active,
//! the FIRST instruction of the callee's own transition is a status clear on a NON-ACTIVE
//! slot — and the prelude consumed it as the acting mon's own wake.
//!
//! Two faults followed, each alone sufficient to defeat the byte-exact match in
//! `identify_sleep_talk_called`:
//!
//!   1. **The tail was cut one instruction too late.** `tail` is `&segment[cursor..]`, and the
//!      cursor had stepped past the callee's own first instruction.
//!   2. **The regeneration state was corrupted.** The prelude had already `sim.apply`-ed that
//!      clear, so the benched member was no longer statused when the candidate scan re-ran the
//!      cure, and the cure regenerated a SHORTER list.
//!
//! They do not cancel: the regenerated list came out equal to `tail[1..]`, a proper SUFFIX.
//! A suffix is a prefix in neither direction, so the containment split could not see it and a
//! transition that was exactly right was refused as
//! `sleeptalk_called_unidentified:none_matched:shape_length`.
//!
//! # Which world turns each test below red
//!
//! Every test here names its null world in its own doc comment, because they do not share
//! one. Reverting the guard turns the SLEEP and FREEZE tests red. It does NOT turn the
//! genuine-wake tests red — those are the OPPOSITE direction, and their null world is a guard
//! that has been widened into excluding the acting mon's own wake. Both worlds are exercised;
//! see the PR body's mutation matrix.
//!
//! # Callee provenance closes the formerly pinned active-first gap
//!
//! The SleepTalkCalled marker precedes the callee's instructions, so the
//! prelude stops before either a benched or active cure. Both formerly pinned
//! active-first layouts now require the exact public callee line and no refusal.
//!
//! ⚠ **That condition is WIDER than "no benched statused member", which is how an earlier
//! revision of this note stated it.** Independent review supplied the second layout: a benched
//! statused member at a HIGHER slot index than the active is walked AFTER the active, so the
//! active's own clear still leads. Both layouts are pinned below —
//! `the_active_first_walk_is_a_known_remaining_gap_with_no_bench` and
//! `..._with_a_higher_slot_bench` — because a gap fixture that pins one of two layouts invites
//! the reader to believe the other is fixed.
//!
//! This is a DIFFERENT defect with a different symptom (a dropped line, never a refusal) and a
//! far wider blast radius, so it wants its own change and its own census. Closing it needs a
//! second, independent predicate (a wake cannot arrive after the sleep gate has already fired),
//! which is measured on the PR and closes BOTH layouts.

use poke_engine::choices::Choices;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::MoveChoice;
use poke_engine::instruction::{Instruction, StateInstructions};
use poke_engine::state::{PokemonIndex, PokemonMoveIndex, PokemonStatus, SideReference, State};
use pokezero_search::events::{render_branch_events, EventContext, RenderedEvents};

/// The refusal class this change closes. Matched as a PREFIX so that the shape suffix
/// (`:shape_length`, `:shape_empty`, ...) can be relabelled — as PR #1241 relabels it —
/// without these fixtures going quiet.
const NONE_MATCHED: &str = "sleeptalk_called_unidentified:none_matched";

/// The line the render must produce once the callee is identified. Sleep Talk's own line is
/// separate; THIS is the called move, and it is what the mis-cut tail lost.
const CALLEE_LINE: &str = "|move|p1a: Sleeper|healbell|p1a: Sleeper|[from] Sleep Talk";

/// A Rest-asleep Sleep Talk user whose callees include a party-wide cure, plus one benched
/// party member whose status is the single fact under test.
///
/// The bench sits at `P0`, BELOW the active at `P1`, because the cure walks slots in order: it
/// is the ORDERING that puts a non-active clear at the HEAD of the callee's own instructions.
/// Reverse the slots and the mechanism does not fire, which is why the ordering has its own
/// pin below.
fn party_cure_sleeper(bench_status: PokemonStatus) -> State {
    party_cure_sleeper_at(PokemonIndex::P1, PokemonIndex::P0, bench_status)
}

/// The same fixture with the two slots named, so the WALK ORDER can be varied. `active` before
/// `bench` puts a non-active clear at the head of the callee's instructions (the defect this
/// change fixes); `bench` after `active` puts the ACTIVE's own clear there instead (the gap it
/// does not).
fn party_cure_sleeper_at(
    active_index: PokemonIndex,
    bench_index: PokemonIndex,
    bench_status: PokemonStatus,
) -> State {
    let mut state = State::default();

    state.side_one.active_index = active_index;
    for index in [active_index, bench_index] {
        let pkmn = &mut state.side_one.pokemon[index];
        pkmn.maxhp = 300;
        pkmn.hp = 300;
        pkmn.speed = 500;
    }
    let bench = &mut state.side_one.pokemon[bench_index];
    bench.status = bench_status;
    if bench_status == PokemonStatus::SLEEP {
        // Rest sleep, not natural sleep. This is the LATCH: `DecrementRestTurns` applies to
        // the ACTIVE, so a benched Rest sleep never decays on its own and the precondition
        // persists for the rest of the battle rather than firing once.
        bench.rest_turns = 3;
    }

    // The ACTIVE: Rest-asleep with two turns left, so it stays asleep this turn and the sleep
    // gate fires BEFORE the callee's instructions — which is what makes the benched clear,
    // not the sleep gate, the thing the wake arm reaches first.
    let active = state.side_one.get_active();
    active.status = PokemonStatus::SLEEP;
    active.rest_turns = 2;
    active.replace_move(PokemonMoveIndex::M0, Choices::SLEEPTALK);
    active.replace_move(PokemonMoveIndex::M1, Choices::HEALBELL);
    active.replace_move(PokemonMoveIndex::M2, Choices::REST);
    active.replace_move(PokemonMoveIndex::M3, Choices::NONE);

    // Slower, and its move touches status on neither side.
    let defender = state.side_two.get_active();
    defender.maxhp = 300;
    defender.hp = 300;
    defender.speed = 1;
    defender.replace_move(PokemonMoveIndex::M0, Choices::SPLASH);

    state
}

/// An active that wakes from Rest THIS turn (`rest_turns == 1`), with nothing benched.
///
/// The engine emits `ChangeStatus { SLEEP -> NONE }` and THEN `DecrementRestTurns`
/// (`gen3/generate_instructions.rs`, the `rest_turns == 1` arm), and the `DecrementRestTurns`
/// arm emits `|cant|..|slp` unless `prelude.woke_up` is already set. So this state is the
/// must-fire direction for the flag the guard rides on: if the guard were widened into
/// excluding the acting mon's own wake, `woke_up` would stay false and a spurious
/// `|cant|..|slp` would appear on a turn the mon actually moves.
fn active_wakes_from_rest(move_id: Choices) -> State {
    let mut state = State::default();
    state.side_one.active_index = PokemonIndex::P1;
    let active = state.side_one.get_active();
    active.maxhp = 300;
    active.hp = 300;
    active.speed = 500;
    active.status = PokemonStatus::SLEEP;
    active.rest_turns = 1;
    active.replace_move(PokemonMoveIndex::M0, move_id);
    active.replace_move(PokemonMoveIndex::M1, Choices::HEALBELL);
    active.replace_move(PokemonMoveIndex::M2, Choices::REST);
    active.replace_move(PokemonMoveIndex::M3, Choices::NONE);

    let defender = state.side_two.get_active();
    defender.maxhp = 300;
    defender.hp = 300;
    defender.speed = 1;
    defender.replace_move(PokemonMoveIndex::M0, Choices::SPLASH);
    state
}

/// An active that THAWS this turn, with nothing benched: the same must-fire direction for the
/// thaw arm. The engine's freeze arm splits 80/20 and the 20% branch carries
/// `ChangeStatus { FREEZE -> NONE }` on `current_active_index`.
fn active_thaws() -> State {
    let mut state = State::default();
    state.side_one.active_index = PokemonIndex::P1;
    let active = state.side_one.get_active();
    active.maxhp = 300;
    active.hp = 300;
    active.speed = 500;
    active.status = PokemonStatus::FREEZE;
    active.replace_move(PokemonMoveIndex::M0, Choices::TACKLE);
    active.replace_move(PokemonMoveIndex::M1, Choices::NONE);
    active.replace_move(PokemonMoveIndex::M2, Choices::NONE);
    active.replace_move(PokemonMoveIndex::M3, Choices::NONE);

    let defender = state.side_two.get_active();
    defender.maxhp = 300;
    defender.hp = 300;
    defender.speed = 1;
    defender.replace_move(PokemonMoveIndex::M0, Choices::SPLASH);
    state
}

fn branches(state: &mut State) -> Vec<StateInstructions> {
    generate_instructions_from_move_pair(
        state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
        false,
    )
}

fn render(state: &mut State, branch: &StateInstructions) -> RenderedEvents {
    render_branch_events(
        state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &branch.instruction_list,
        false,
        &EventContext {
            species: [
                vec!["Bench".to_string(), "Sleeper".to_string()],
                vec!["Opponent".to_string()],
            ],
            turn: 1,
            hp_percent: [false, false],
        },
    )
}

/// The branch carrying the callee: the one that clears the ACTIVE's own sleep.
fn branch_curing_the_active(generated: &[StateInstructions]) -> &StateInstructions {
    branch_curing_slot(generated, PokemonIndex::P1)
}

/// Parameterised on the slot, because the two gap layouts put the active at different indices and
/// a finder hardcoded to `P1` silently found nothing for one of them.
fn branch_curing_slot(
    generated: &[StateInstructions],
    active_index: PokemonIndex,
) -> &StateInstructions {
    generated
        .iter()
        .find(|branch| {
            branch.instruction_list.iter().any(|ins| {
                matches!(ins, Instruction::ChangeStatus(change)
                    if change.side_ref == SideReference::SideOne
                        && change.pokemon_index == active_index
                        && change.old_status == PokemonStatus::SLEEP
                        && change.new_status == PokemonStatus::NONE)
            })
        })
        .unwrap_or_else(|| panic!("no branch clears {active_index:?}'s own sleep: {generated:?}"))
}

fn branch_with<'a>(
    generated: &'a [StateInstructions],
    want: impl Fn(&Instruction) -> bool,
) -> &'a StateInstructions {
    generated
        .iter()
        .find(|branch| branch.instruction_list.iter().any(&want))
        .unwrap_or_else(|| panic!("no branch carries the wanted instruction: {generated:?}"))
}

/// Every `none_matched` slug the render emitted, from BOTH channels.
///
/// `mark_attribution_unsafe_subcase` pushes onto `attribution_unsafe`; `mark_lossy_subcase`
/// pushes onto `lossy_subcases`. Scanning both is the fail-loud direction: a slug that moves
/// channels must not make these fixtures go quiet.
fn none_matched(rendered: &RenderedEvents) -> Vec<String> {
    let mut found: Vec<String> = rendered
        .attribution_unsafe
        .iter()
        .chain(rendered.lossy_subcases.iter())
        .filter(|s| s.starts_with(NONE_MATCHED))
        .cloned()
        .collect();
    found.sort();
    found.dedup();
    found
}

// ---------------------------------------------------------------------------
// Direction 1 — the truth constructs. Null world: the guard reverted.
// ---------------------------------------------------------------------------

/// **The refusal class, closed.**
///
/// One benched Rest-asleep party member below the active is the whole precondition. No
/// opposing Sleep Talk carrier and no sampled world is needed: the census's pool-set
/// co-occurrence requirement is a condition on REACHING this position over a long game, not a
/// condition on the mechanism.
///
/// NULL WORLD: revert the guard and this goes red on BOTH assertions — the refusal reappears
/// as `none_matched:shape_length` and the callee line vanishes.
#[test]
fn a_benched_asleep_party_member_no_longer_derails_the_callee_identification() {
    let mut state = party_cure_sleeper(PokemonStatus::SLEEP);
    let generated = branches(&mut state);
    let branch = branch_curing_the_active(&generated).clone();
    let rendered = render(&mut state, &branch);

    assert_eq!(
        none_matched(&rendered),
        Vec::<String>::new(),
        "the class must be GONE, not relabelled: attribution_unsafe {:?}, lossy_subcases {:?}, \
         lines {:?}",
        rendered.attribution_unsafe,
        rendered.lossy_subcases,
        rendered.lines
    );
    // STOPPING REFUSING IS NOT ENOUGH — the render has to start COUNTING the callee. A change
    // that silenced the refusal without producing this line would be a regression, not a fix.
    assert!(
        rendered.lines.iter().any(|l| l == CALLEE_LINE),
        "the identified callee must be rendered: lines {:?}",
        rendered.lines
    );
    assert!(
        rendered.attribution_unsafe.is_empty(),
        "nothing may remain attribution-unsafe: {:?}",
        rendered.attribution_unsafe
    );
}

/// **The same defect on the THAW arm, which presented as a silent dropped line.**
///
/// A benched FREEZE is cured by the same walk. Before the guard, the thaw arm ate the benched
/// clear, which left the ACTIVE's own clear next in line for the wake arm — so BOTH were
/// consumed, the callee regenerated an empty transition, and the render dropped the callee
/// line with NO refusal to mark it. A silent wrong render is worse than a refusal, so this
/// case is fixed here rather than left for the census to not notice.
///
/// NULL WORLD: revert the guard on the FREEZE arm alone and this goes red while the test above
/// stays green — which is why guarding only the SLEEP arm is not enough.
#[test]
fn a_benched_frozen_party_member_no_longer_swallows_the_callee_line() {
    let mut state = party_cure_sleeper(PokemonStatus::FREEZE);
    let generated = branches(&mut state);
    let branch = branch_curing_the_active(&generated).clone();
    let rendered = render(&mut state, &branch);

    assert!(
        rendered.lines.iter().any(|l| l == CALLEE_LINE),
        "the callee line was swallowed by the thaw arm: lines {:?}",
        rendered.lines
    );
    assert_eq!(
        none_matched(&rendered),
        Vec::<String>::new(),
        "and it must not have traded the silent drop for a refusal: {:?} / {:?}",
        rendered.attribution_unsafe,
        rendered.lossy_subcases
    );
}

// ---------------------------------------------------------------------------
// Direction 2 — the guard did not become too wide. Null world: a guard that
// also excludes the acting mon's own status change.
// ---------------------------------------------------------------------------

/// **A GENUINE Rest wake still sets `woke_up`, so the sleep gate stays silent.**
///
/// This is the regression the guard risked and the reason the fix was not landed as a label
/// change. The engine emits the wake BEFORE `DecrementRestTurns`, and that arm emits
/// `|cant|..|slp` unless `woke_up` is already set. Widen the guard into excluding the active
/// and this turns red with a `|cant|..|slp` on a turn the mon moves — a FABRICATED protocol
/// line, which is strictly worse than any refusal.
///
/// NULL WORLD: `changes_the_active_slot` negated, or the guard pointed at the other side.
#[test]
fn a_genuine_rest_wake_still_suppresses_the_sleep_gate_and_renders_the_move() {
    let mut state = active_wakes_from_rest(Choices::TACKLE);
    let generated = branches(&mut state);
    let branch = branch_with(&generated, |ins| {
        matches!(ins, Instruction::ChangeStatus(change)
            if change.side_ref == SideReference::SideOne
                && change.old_status == PokemonStatus::SLEEP
                && change.new_status == PokemonStatus::NONE)
    })
    .clone();
    let rendered = render(&mut state, &branch);

    assert!(
        !rendered
            .lines
            .iter()
            .any(|l| l.contains("|cant|") && l.contains("slp")),
        "a mon that woke up must not also be blocked by sleep: lines {:?}",
        rendered.lines
    );
    assert!(
        rendered
            .lines
            .iter()
            .any(|l| l.starts_with("|move|p1a: Sleeper|tackle")),
        "the woken mon must render its move: lines {:?}",
        rendered.lines
    );
}

/// The same must-fire direction with SLEEP TALK as the click, because the sleep gate's Sleep
/// Talk continuation rule is the branch the guard sits closest to.
///
/// NULL WORLD: as above.
#[test]
fn a_genuine_rest_wake_under_sleep_talk_still_suppresses_the_sleep_gate() {
    let mut state = active_wakes_from_rest(Choices::SLEEPTALK);
    let generated = branches(&mut state);
    let branch = branch_with(&generated, |ins| {
        matches!(ins, Instruction::ChangeStatus(change)
            if change.side_ref == SideReference::SideOne
                && change.old_status == PokemonStatus::SLEEP
                && change.new_status == PokemonStatus::NONE)
    })
    .clone();
    let rendered = render(&mut state, &branch);

    assert!(
        !rendered
            .lines
            .iter()
            .any(|l| l.contains("|cant|") && l.contains("slp")),
        "the woken Sleep Talk user must not be blocked by sleep: lines {:?}",
        rendered.lines
    );
}

/// **A GENUINE thaw is still consumed by the thaw arm.**
///
/// The thaw's own null world. If the guard excluded the acting mon, the thaw would fall into
/// the move phase; the move phase renders a cure silently, so the visible consequence is on
/// the `|move|` line, which must still be there.
///
/// NULL WORLD: `changes_the_active_slot` negated on the FREEZE arm.
#[test]
fn a_genuine_thaw_still_lets_the_move_through() {
    let mut state = active_thaws();
    let generated = branches(&mut state);
    let branch = branch_with(&generated, |ins| {
        matches!(ins, Instruction::ChangeStatus(change)
            if change.side_ref == SideReference::SideOne
                && change.old_status == PokemonStatus::FREEZE
                && change.new_status == PokemonStatus::NONE)
    })
    .clone();
    let rendered = render(&mut state, &branch);

    assert!(
        rendered
            .lines
            .iter()
            .any(|l| l.starts_with("|move|p1a: Sleeper|tackle")),
        "the thawed mon must render its move: lines {:?}",
        rendered.lines
    );
    assert!(
        !rendered.lines.iter().any(|l| l.contains("|cant|")),
        "a thawed mon is not blocked: lines {:?}",
        rendered.lines
    );
}

// ---------------------------------------------------------------------------
// Invariance and canaries. These are GREEN IN BOTH WORLDS BY DESIGN and are
// labelled as such: they are not evidence for the fix, they are tripwires.
// ---------------------------------------------------------------------------

/// CANARY, green in both worlds. A benched BURN was never eaten — no arm matches it — so the
/// callee was already identified. Recorded so that "the callee line appears" cannot be read as
/// something the guard invented: it appears here without the guard too.
#[test]
fn a_benched_burn_was_already_identified_and_still_is() {
    let mut state = party_cure_sleeper(PokemonStatus::BURN);
    let generated = branches(&mut state);
    let branch = branch_curing_the_active(&generated).clone();
    let rendered = render(&mut state, &branch);

    assert!(
        rendered.lines.iter().any(|l| l == CALLEE_LINE),
        "lines {:?}",
        rendered.lines
    );
    assert_eq!(none_matched(&rendered), Vec::<String>::new());
}

/// CANARY, green in both worlds. The ENGINE ordering the whole mechanism rests on: the
/// party-wide cure clears the BENCHED slot before the active's own. If a future engine change
/// reorders `pokemon_index_iter()`, or moves the cure ahead of the sleep gate, THIS is the
/// assertion that should move first — the render tests above would otherwise start agreeing
/// with each other for the wrong reason.
#[test]
fn the_party_cure_clears_the_benched_slot_before_the_actives_own() {
    let mut state = party_cure_sleeper(PokemonStatus::SLEEP);
    let generated = branches(&mut state);
    let branch = branch_curing_the_active(&generated).clone();

    let clears: Vec<PokemonIndex> = branch
        .instruction_list
        .iter()
        .filter_map(|ins| match ins {
            Instruction::ChangeStatus(change)
                if change.side_ref == SideReference::SideOne
                    && change.old_status == PokemonStatus::SLEEP
                    && change.new_status == PokemonStatus::NONE =>
            {
                Some(change.pokemon_index)
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        clears,
        vec![PokemonIndex::P0, PokemonIndex::P1],
        "the cure must clear the BENCHED slot first, or this class has a different cause: {:?}",
        branch.instruction_list
    );
}

/// Callee provenance delimits a cure even when the active clears first.
#[test]
fn callee_marker_preserves_active_first_cure_with_no_bench() {
    assert_active_first_call(party_cure_sleeper(PokemonStatus::NONE), PokemonIndex::P1);
}

#[test]
fn callee_marker_preserves_active_first_cure_with_a_higher_slot_bench() {
    assert_active_first_call(
        party_cure_sleeper_at(PokemonIndex::P0, PokemonIndex::P2, PokemonStatus::SLEEP),
        PokemonIndex::P0,
    );
}

fn assert_active_first_call(mut state: State, active_index: PokemonIndex) {
    let generated = branches(&mut state);
    let branch = branch_curing_slot(&generated, active_index).clone();
    let before = state.serialize();
    let rendered = render(&mut state, &branch);
    let name = if active_index == PokemonIndex::P0 {
        "Bench"
    } else {
        "Sleeper"
    };
    let expected = format!("|move|p1a: {name}|healbell|p1a: {name}|[from] Sleep Talk");
    assert_eq!(
        rendered.lines.iter().filter(|l| **l == expected).count(),
        1,
        "{rendered:?}"
    );
    assert!(rendered.attribution_unsafe.is_empty(), "{rendered:?}");
    assert!(rendered.lossy.is_empty(), "{rendered:?}");
    assert_eq!(state.serialize(), before);
}
