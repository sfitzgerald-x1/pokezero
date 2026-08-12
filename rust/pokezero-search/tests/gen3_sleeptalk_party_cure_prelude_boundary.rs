//! The `sleeptalk_called_unidentified:none_matched` class, reduced to its
//! preconditions and made reachable from a CONSTRUCTED state.
//!
//! # What this file is for
//!
//! This class was measured **unreachable from every turn-start state the repo could
//! construct** -- 0 `none_matched` in 1,341,623 rendered branches over three sweeps -- while
//! firing in the production tree. It was captured live at production budget (the
//! `none-matched-dump` capture directory, off-repo; seed 9900068, 36 of 36 decisions
//! identical in shape) and the capture names the mechanism. These fixtures are that
//! mechanism, written as the two-sided measurement the sweeps could not make: one state that
//! produces the class and one that differs in a SINGLE fact and does not.
//!
//! # The mechanism, read from the capture
//!
//! `consume_move_prelude`'s Rest/natural-wake arm matches
//! `ChangeStatus { old_status: SLEEP, new_status: NONE }` on **`side_ref` alone**. It never
//! checks `pokemon_index` against the acting side's `active_index`. gen3's Heal Bell
//! (`gen3/choice_effects.rs`, `Choices::HEALBELL`) walks `pokemon_index_iter()` in SLOT
//! ORDER and emits a status clear for every statused party member, so when a BENCHED party
//! member at a lower slot than the active is asleep, the callee's own first instruction is a
//! `ChangeStatus { SLEEP -> NONE }` on a NON-ACTIVE slot. The prelude eats it as "the
//! sleeper woke up".
//!
//! Two consequences, each alone sufficient to defeat the byte-exact match in
//! `identify_sleep_talk_called`:
//!
//!   1. **The tail is truncated at its head.** `tail` is `&segment[cursor..]` and the cursor
//!      has advanced one instruction too far, so the tail begins mid-callee.
//!   2. **The regeneration state is corrupted.** The prelude `sim.apply`-ed that clear, so
//!      the benched member is no longer asleep when the candidate scan re-runs Heal Bell --
//!      and Heal Bell then has nothing to cure there and emits a SHORTER list.
//!
//! The regenerated branch therefore equals `tail[1..]`: a proper **SUFFIX** of the tail.
//! A suffix is not a prefix in either direction, so the containment split cannot see it and
//! the class reports `shape_length` -- "a genuinely different transition" -- for a transition
//! that is exactly right and a phase boundary that is off by one instruction.
//!
//! # THE GUARD ALREADY EXISTS, ONE FUNCTION AWAY
//!
//! The missing conjunct is not new logic. `active_status_transition` (`src/events.rs`, ~1,400
//! lines above the arm) already compares
//! `state.get_side_immutable(&change.side_ref).active_index == change.pokemon_index` -- and the
//! wake arm CALLS it, on this very instruction, to decide whether to record a transition. It
//! correctly answers "not the active" and records nothing; the arm then consumes and applies
//! the instruction anyway. So the discriminator is computed, consulted for one purpose, and
//! discarded for the other. Applying the existing check at the second site is the whole of the
//! eventual fix.
//!
//! That is the campaign's recurring shape: the value was public, and the consumer refused to
//! read it.
//!
//! **ACTED ON.** The paragraph above used to end "recorded here rather than acted on ... guarding
//! it is a render-behaviour change that needs its own PR, review and census arm". That PR is the
//! one that edited this file: the conjunct is applied at the second site, as
//! `changes_the_active_slot`, on the wake arm AND on the thaw arm beside it. The thaw arm needed
//! it too, and needed it more -- eating a benched THAW left the active's own clear next in line
//! for the wake arm, so both were consumed and the render dropped the callee's `|move|` line
//! with no refusal at all, which is the silent half of the same defect.
//!
//! Consequence for this file, stated because a fixture whose subject is fixed is the easiest
//! place in a repo to leave a lie: the positive test below **no longer observes the class**,
//! because the class is gone. It has been TURNED rather than deleted. This file's third test
//! FORESAW the two converging -- it warns that the two tests above it "would then simply agree
//! with each other" -- and that foresight is why turning is the right action; it is NOT an
//! instruction to turn this particular test, and an earlier revision of this note claimed it
//! was. See `tests/gen3_sleeptalk_party_cure_active_slot_guard.rs` for the guard's own
//! two-sided pins.
//!
//! ⚠ **The `divergence_shape` no-op label fix DID lose its end-to-end pin here, and an earlier
//! revision of this note wrongly said it did not.** Its classifier-level pin
//! (`divergence_shape(&[], &[dmg(30)]) == None`) does NOT hold the CALL SITE, which is where
//! the fix's observable effect lives: independent review mutated
//! `.filter_map(..)` to `.map(.. .unwrap_or(NoneMatchedShape::Empty))` and killed exactly the
//! assertion retired below on `main`, while surviving the whole suite here. The replacement is
//! `mod none_matched_shape_call_site` in `src/events.rs`, which pins the emitted set at the
//! production call site.
//!
//! # NOT branch merging
//!
//! The standing hypothesis was that `combine_duplicate_instructions` merges branches so a
//! candidate's regenerated list is not shape-comparable to the engine's. That is refuted
//! here and in the capture: the Heal Bell branch carries `percentage: 100`, nothing is
//! merged, and the callee IS identifiable -- the renderer regenerates its transition
//! correctly and compares it against a mis-cut tail.
//!
//! # Why the refusal WAS correct, and why removing it is not a relaxation
//!
//! `NoneMatched` proved neither the transition nor the slicing, so refusing was right for as
//! long as the slicing was wrong. The guard does not relax the refusal -- it removes the
//! refusal's CAUSE. The callee is now identified by the same byte-exact match that always
//! governed it, on a tail that is finally cut in the right place, and the render emits the
//! callee's own `|move|..|[from] Sleep Talk` line. `attribution_unsafe` stays empty because
//! nothing was guessed.

use poke_engine::choices::Choices;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::MoveChoice;
use poke_engine::instruction::{Instruction, StateInstructions};
use poke_engine::state::{PokemonIndex, PokemonMoveIndex, PokemonStatus, SideReference, State};
use pokezero_search::events::{render_branch_events, EventContext};

/// The refusal predicate this class emits. Matched as a PREFIX, so the shape suffixes
/// (`:shape_length` etc.) can change without this fixture going quiet.
const NONE_MATCHED: &str = "sleeptalk_called_unidentified:none_matched";

/// The sleeper's side, its party, and the one fact under test.
///
/// Side one's active is a Rest-asleep Sleep Talk user whose only callees are a PARTY-WIDE
/// CURE (Heal Bell) and Rest. Side two does nothing that touches status.
///
/// `bench_asleep` is the single fact that separates the positive from the control. The
/// benched member sits at `P0`, BELOW the active at `P1`, because Heal Bell walks slots in
/// order: it is the ORDERING that puts a non-active clear at the head of the callee's own
/// instructions, which is what the prelude eats.
fn party_cure_sleeper(bench_asleep: bool) -> State {
    let mut state = State::default();

    state.side_one.active_index = PokemonIndex::P1;
    for index in [PokemonIndex::P0, PokemonIndex::P1] {
        let pkmn = &mut state.side_one.pokemon[index];
        pkmn.maxhp = 300;
        pkmn.hp = 300;
        pkmn.speed = 500;
    }
    // The BENCHED party member. Asleep from its own Rest in the positive case, and
    // otherwise identical.
    let bench = &mut state.side_one.pokemon[PokemonIndex::P0];
    if bench_asleep {
        bench.status = PokemonStatus::SLEEP;
        bench.rest_turns = 3;
    }

    // The ACTIVE: Rest-asleep, Sleep Talk in M0, a party-wide cure and Rest as its callees.
    let active = state.side_one.get_active();
    active.status = PokemonStatus::SLEEP;
    active.rest_turns = 2;
    active.replace_move(PokemonMoveIndex::M0, Choices::SLEEPTALK);
    active.replace_move(PokemonMoveIndex::M1, Choices::HEALBELL);
    active.replace_move(PokemonMoveIndex::M2, Choices::REST);
    active.replace_move(PokemonMoveIndex::M3, Choices::NONE);

    // The defender: slower, and its move touches no status on either side.
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

fn render(state: &mut State, branch: &StateInstructions) -> pokezero_search::events::RenderedEvents {
    render_branch_events(
        state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &branch.instruction_list,
        false,
        &EventContext {
            species: [
                vec!["Bench".into(), vec!["Sleeper".to_string()].concat()],
                vec!["Opponent".into()],
            ],
            turn: 1,
            hp_percent: [false, false],
        },
    )
}

/// The branch that carries the callee: the one that cures the sleeper's own sleep.
fn callee_branch(branches: &[StateInstructions]) -> &StateInstructions {
    branches
        .iter()
        .find(|branch| {
            branch.instruction_list.iter().any(|ins| {
                matches!(ins, Instruction::ChangeStatus(change)
                    if change.side_ref == SideReference::SideOne
                        && change.pokemon_index == PokemonIndex::P1
                        && change.old_status == PokemonStatus::SLEEP
                        && change.new_status == PokemonStatus::NONE)
            })
        })
        .unwrap_or_else(|| panic!("no branch cures the sleeper's own sleep: {branches:?}"))
}

/// Every `none_matched` slug the render emitted, from BOTH channels.
///
/// `mark_attribution_unsafe_subcase` pushes onto `attribution_unsafe` (and `lossy`), not onto
/// `lossy_subcases` -- which is where the first version of this helper looked, and it read
/// the class as absent while the render was refusing with it. Scanning both is the
/// fail-loud direction: a slug moving channels must not make this fixture go quiet.
fn none_matched_reasons(rendered: &pokezero_search::events::RenderedEvents) -> Vec<String> {
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

/// **THE CLASS, CLOSED, FROM THE SAME CONSTRUCTED STATE.**
///
/// One benched asleep party member below the active is the whole precondition. No opposing
/// Sleep Talk carrier, no long game, no sampled world: the census's
/// `lapras-2-variant-5` + `machamp-2-variant-5` CO-OCCURRENCE requirement is a condition on
/// REACHING this position over 731 games, not a condition on the mechanism -- this state has
/// neither carrier and used to produce the class.
///
/// **THIS ASSERTION WAS TURNED, NOT DELETED.** It read `assert!(rendered.is_attribution_unsafe())`
/// plus the exact slug set `{none_matched:shape_length}`. The state, the branch selection and
/// the two-sided structure are unchanged; only the expected verdict moved, from "refuses, with
/// this exact label" to "does not refuse, and names the callee".
///
/// ⚠ **The authority for turning it is weaker than an earlier revision of this comment claimed,
/// and the misreading is recorded rather than quietly fixed.** The line *"THIS is the assertion
/// that should move -- not the two above"* sits on
/// `the_callee_emits_a_non_active_status_clear_ahead_of_the_actives_own`, refers to ITSELF, and
/// explicitly EXCLUDES the two tests above it -- of which this is the first. The designated
/// assertion is byte-unchanged. What that note really provides is FORESIGHT: it says that if the
/// prelude arm grew the guard, the two tests above would "simply agree with each other", which is
/// exactly what happened and is why turning beats deleting. It is not an instruction to turn
/// this test.
///
/// The retired half WAS the label fix's end-to-end witness, and that loss is real. It is
/// replaced at the call site by `mod none_matched_shape_call_site` in `src/events.rs`; see that
/// module for why the replacement is in kind rather than in degree.
#[test]
fn a_benched_asleep_party_member_no_longer_makes_the_tail_non_containable() {
    let mut state = party_cure_sleeper(true);
    let generated = branches(&mut state);
    let branch = callee_branch(&generated).clone();
    let rendered = render(&mut state, &branch);

    assert_eq!(
        none_matched_reasons(&rendered),
        Vec::<String>::new(),
        "the class must be GONE, not relabelled: subcases {:?}, reasons {:?}, lines {:?}",
        rendered.lossy_subcases,
        rendered.attribution_unsafe,
        rendered.lines
    );
    assert!(
        !rendered.is_attribution_unsafe(),
        "nothing may remain attribution-unsafe: {:?}",
        rendered.attribution_unsafe
    );
    // AND THE RENDER MUST START COUNTING WHAT IT STOPPED REFUSING. A change that silenced the
    // refusal without emitting the callee's own line would be a regression wearing a fix's
    // clothes, which is the failure mode #1157 shipped.
    assert!(
        rendered
            .lines
            .iter()
            .any(|l| l == "|move|p1a: Sleeper|healbell|p1a: Sleeper|[from] Sleep Talk"),
        "the identified callee must be rendered: lines {:?}",
        rendered.lines
    );
}

/// **NO LONGER A CONTROL, AND SAYING SO IS THE POINT.** Wake the bench: the class does not fire
/// here either.
///
/// ⚠ This comment used to read *"THE CONTROL, differing in ONE fact ... it is the null world"*,
/// and that was true only while the test above it asserted PRESENCE. Both tests now assert
/// ABSENCE, so this one is a strict subset of that one and contrasts with nothing -- which is
/// precisely the "a test that passes in both worlds is not a test" failure this file's module doc
/// claims to guard against. Left in place as an INVARIANCE check (the guard must not have
/// introduced a refusal on the no-benched-sleeper path) and relabelled so it cannot be read as
/// evidence for anything.
///
/// The real null world for the test above now lives in
/// `tests/gen3_sleeptalk_party_cure_active_slot_guard.rs`, whose fixtures each name the mutant
/// that turns them red, and in the mutation matrix on the PR.
#[test]
fn the_same_state_with_no_benched_sleeper_identifies_the_callee() {
    let mut state = party_cure_sleeper(false);
    let generated = branches(&mut state);
    let branch = callee_branch(&generated).clone();
    let rendered = render(&mut state, &branch);

    assert!(
        none_matched_reasons(&rendered).is_empty(),
        "with no benched sleeper the callee is identifiable and nothing in this class may \
         fire: subcases {:?}, reasons {:?}, lines {:?}",
        rendered.lossy_subcases,
        rendered.attribution_unsafe,
        rendered.lines
    );
}

/// The SIGNATURE, pinned on the instruction list rather than inferred from the label.
///
/// The callee's own first instruction is a status clear on a NON-ACTIVE slot, and it is the
/// instruction the prelude's wake arm matches. `add_remove_status_instructions` is what
/// emits it; `pokemon_index_iter()` in `Choices::HEALBELL` is what puts it first.
///
/// If a future engine change reorders Heal Bell's walk, or the prelude arm grows the
/// `pokemon_index` guard it lacks, THIS is the assertion that should move -- not the two
/// above, which would then simply agree with each other.
#[test]
fn the_callee_emits_a_non_active_status_clear_ahead_of_the_actives_own() {
    let mut state = party_cure_sleeper(true);
    let generated = branches(&mut state);
    let branch = callee_branch(&generated).clone();

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
        "the party-wide cure must clear the BENCHED slot before the active's own, or the \
         prelude has nothing wrong to eat and this whole class has a different cause: {:?}",
        branch.instruction_list
    );
}
