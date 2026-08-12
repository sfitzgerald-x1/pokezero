//! The `sleeptalk_called_unidentified:none_matched` class, reduced to its
//! preconditions and made reachable from a CONSTRUCTED state.
//!
//! # What this file is for
//!
//! This class was measured **unreachable from every turn-start state the repo could
//! construct** -- 0 `none_matched` in 1,341,623 rendered branches over three sweeps -- while
//! firing in the production tree. It was captured live at production budget
//! (`/Users/scott/workspace/agents/none-matched-dump/`, seed 9900068, 36 of 36 decisions
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
//! # NOT branch merging
//!
//! The standing hypothesis was that `combine_duplicate_instructions` merges branches so a
//! candidate's regenerated list is not shape-comparable to the engine's. That is refuted
//! here and in the capture: the Heal Bell branch carries `percentage: 100`, nothing is
//! merged, and the callee IS identifiable -- the renderer regenerates its transition
//! correctly and compares it against a mis-cut tail.
//!
//! # Why the refusal is still CORRECT
//!
//! `NoneMatched` proves neither the transition nor the slicing, and these fixtures do not
//! relax it. They pin the CAUSE so that the label stops pointing at the wrong owner.

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

/// **THE CLASS, REACHED FROM A CONSTRUCTED STATE.**
///
/// One benched asleep party member below the active is the whole precondition. No opposing
/// Sleep Talk carrier, no long game, no sampled world: the census's
/// `lapras-2-variant-5` + `machamp-2-variant-5` CO-OCCURRENCE requirement is a condition on
/// REACHING this position over 731 games, not a condition on the mechanism -- this state has
/// neither carrier and produces the class.
#[test]
fn a_benched_asleep_party_member_makes_the_sleep_talk_tail_non_containable() {
    let mut state = party_cure_sleeper(true);
    let generated = branches(&mut state);
    let branch = callee_branch(&generated).clone();
    let rendered = render(&mut state, &branch);

    assert!(
        rendered.is_attribution_unsafe(),
        "the branch must refuse: {:?} / lines {:?}",
        rendered.attribution_unsafe,
        rendered.lines
    );
    // THE EXACT SLUG SET, not merely non-empty -- because the set is the deliverable.
    //
    // `shape_empty` is ABSENT, and its absence is the whole of the `divergence_shape` label
    // fix landing in this same change. `REST` is a callee here and regenerates a single EMPTY
    // branch (Rest while already asleep), so before the fix this set was
    // `{shape_empty, shape_length}` -- and `shape_empty` was present on every `none_matched`
    // decision the gen3 randbats pool can produce (12 of 33 callees emit an empty branch
    // while the user is asleep), carrying zero bits. Reverting the fix turns this assertion
    // red, which is what makes it a test of the fix rather than a description of it.
    assert_eq!(
        none_matched_reasons(&rendered),
        vec![format!("{NONE_MATCHED}:shape_length")],
        "the refusal must be `none_matched` with `shape_length` ALONE: subcases {:?}, \
         reasons {:?}",
        rendered.lossy_subcases,
        rendered.attribution_unsafe
    );
}

/// **THE CONTROL, differing in ONE fact.** Wake the bench and the class disappears.
///
/// This is what makes the fixture above a measurement rather than an assertion: it is the
/// null world. Without it, "the class fires here" is consistent with the class firing on
/// every Sleep Talk render, which is exactly what the earlier sweeps could not rule out.
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
