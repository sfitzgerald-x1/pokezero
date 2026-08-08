//! Gen 3 Transform (Ditto) fidelity pins, asserted directly against the
//! vendored gen3-patched poke-engine (`third_party/poke-engine-src/`).
//!
//! Companion to `gen3_switch_fidelity.rs`. Every expectation here was read off
//! the **real** Node Showdown simulator driven through
//! `scripts/gen3_switch_differential.py` (the `transform*` scenarios) and
//! cross-read against `sim/pokemon.ts::transformInto` with `gen == 3`; this file
//! is the engine-contract pin, so a wheel rebuild or a version bump that
//! silently drops `third_party/poke-engine-gen3-transform.patch` fails
//! `cargo test` instead of quietly regressing search fidelity.
//!
//! Before the patch, `Choices::TRANSFORM` had no implementation at all: it fell
//! through every gen3 effect dispatcher to `_ => {}`, so clicking Transform
//! produced no state change whatsoever and the engine-world constructor could
//! not express a transformed Ditto. Ditto (movepool `["transform"]`) and Mew are
//! the two gen3 randbats Transform carriers.
//!
//! Coverage:
//!
//! * species / types / the five non-HP stats / ability / boosts / moves are
//!   copied; HP, max HP and status are not.
//! * copied move slots hold 5 PP each (`pp: Math.min(5, move.pp)`).
//! * Transform goes THROUGH a Substitute in gen3 (the `substitute` guard in
//!   `transformInto` is gated on `gen >= 5`, and gen3 inherits gen4's
//!   `bypasssub` flag).
//! * Transform FAILS against an already-transformed target
//!   (`pokemon.transformed && this.battle.gen >= 2`).
//! * switching out reverts everything (`Pokemon.clearVolatile()` ends with
//!   `setSpecies(this.baseSpecies)`).
//! * apply-then-reverse of the emitted instruction list restores the pre-move
//!   state bit-exactly — the property the whole search depends on.

use poke_engine::choices::Choices;
use poke_engine::engine::abilities::Abilities;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::{MoveChoice, PokemonVolatileStatus};
use poke_engine::instruction::{Instruction, StateInstructions};
use poke_engine::pokemon::PokemonName;
use poke_engine::state::{
    PokemonIndex, PokemonMoveIndex, PokemonStatus, PokemonType, State,
};

/// `generate_instructions_from_move_pair` must leave `state` untouched — the
/// engine's own test suite asserts this and a patch that mutates without
/// emitting a reversible instruction would silently corrupt search.
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

fn only_branch(instructions: Vec<StateInstructions>) -> Vec<Instruction> {
    assert_eq!(
        instructions.len(),
        1,
        "expected a single deterministic branch, got {:?}",
        instructions
    );
    instructions.into_iter().next().unwrap().instruction_list
}

/// Side one is a Ditto-shaped transformer (one move: Transform, everything else
/// deliberately distinct from the target so every copied field is observable).
/// Side two is the copy target.
fn transform_state() -> State {
    let mut state = State::default();

    let ditto = state.side_one.get_active();
    ditto.id = PokemonName::DITTO;
    ditto.hp = 200;
    ditto.maxhp = 200;
    ditto.types = (PokemonType::NORMAL, PokemonType::TYPELESS);
    ditto.base_types = (PokemonType::NORMAL, PokemonType::TYPELESS);
    ditto.ability = Abilities::LIMBER;
    ditto.base_ability = Abilities::LIMBER;
    ditto.attack = 132;
    ditto.defense = 132;
    ditto.special_attack = 132;
    ditto.special_defense = 132;
    ditto.speed = 132;
    ditto.replace_move(PokemonMoveIndex::M0, Choices::TRANSFORM);
    // 8 rather than a full 16 so the engine's `pp < 10` PP decrement actually
    // fires and the interaction with the copied moveset is pinned.
    ditto.moves[&PokemonMoveIndex::M0].pp = 8;
    for empty_slot in [
        PokemonMoveIndex::M1,
        PokemonMoveIndex::M2,
        PokemonMoveIndex::M3,
    ] {
        ditto.replace_move(empty_slot, Choices::NONE);
        ditto.moves[&empty_slot].pp = 0;
    }

    let target = state.side_two.get_active();
    target.id = PokemonName::MACHAMP;
    target.hp = 90;
    target.maxhp = 321;
    target.types = (PokemonType::FIGHTING, PokemonType::TYPELESS);
    target.base_types = (PokemonType::FIGHTING, PokemonType::TYPELESS);
    target.ability = Abilities::GUTS;
    target.base_ability = Abilities::GUTS;
    target.attack = 296;
    target.defense = 196;
    target.special_attack = 166;
    target.special_defense = 206;
    target.speed = 146;
    target.replace_move(PokemonMoveIndex::M0, Choices::BULKUP);
    target.moves[&PokemonMoveIndex::M0].pp = 31;
    target.replace_move(PokemonMoveIndex::M1, Choices::CROSSCHOP);
    target.moves[&PokemonMoveIndex::M1].pp = 8;
    target.replace_move(PokemonMoveIndex::M2, Choices::SPLASH);
    target.moves[&PokemonMoveIndex::M2].pp = 63;
    target.replace_move(PokemonMoveIndex::M3, Choices::NONE);
    target.moves[&PokemonMoveIndex::M3].pp = 0;

    state
}

/// Run "side one Transforms, side two Splashes" and leave the state with the
/// resulting instructions APPLIED, returning the list.
fn transform_and_apply(state: &mut State) -> Vec<Instruction> {
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M2, Choices::SPLASH);
    let branches = generate(
        state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M2),
    );
    // NOT `only_branch`, and the reason is the attract/paralysis marker patch rather than
    // anything about Transform. `transform_does_not_copy_hp_or_status` paralyses side two
    // so that "status is not copied" is observable, and side two's click is Splash -- an
    // empty delta. Before the immobilizer markers, the fully-paralyzed branch and the
    // Splash-executed branch had IDENTICAL instruction lists, so
    // `combine_duplicate_instructions` merged them and this really was one branch. That
    // merge is the defect the markers fix: it is exactly why the renderer could not tell
    // "no move happened" from "a move happened and changed nothing".
    //
    // Every assertion downstream is about SIDE ONE's transformed Pokemon, and side one's
    // instructions are identical across the split, so dropping the immobilized branch is
    // information-preserving here. `only_branch` stays as it is for the tests that really
    // are deterministic.
    let list = single_acting_branch(branches);
    state.apply_instructions(&list);
    list
}

/// The branch on which the second mover actually ACTED: the one carrying no
/// `MoveImmobilized` marker. Panics unless exactly one such branch exists, so it cannot
/// silently absorb a NEW split the way a bare `[0]` would.
fn single_acting_branch(instructions: Vec<StateInstructions>) -> Vec<Instruction> {
    let mut acting: Vec<Vec<Instruction>> = instructions
        .into_iter()
        .filter(|branch| {
            !branch
                .instruction_list
                .iter()
                .any(|i| matches!(i, Instruction::MoveImmobilized(_)))
        })
        .map(|branch| branch.instruction_list)
        .collect();
    assert_eq!(
        acting.len(),
        1,
        "expected exactly one non-immobilized branch, got {acting:?}"
    );
    acting.pop().unwrap()
}

// ---------------------------------------------------------------------------
// What Transform copies
// ---------------------------------------------------------------------------

/// Showdown gen3: `|move|p1a: Ditto|Transform|p2a: Machamp` /
/// `|-transform|p1a: Ditto|p2a: Machamp`, after which p1's request lists
/// Machamp's moves and the copied Bulk Up works from the Ditto slot.
#[test]
fn transform_copies_species_types_stats_and_ability() {
    let mut state = transform_state();
    transform_and_apply(&mut state);

    let ditto = state.side_one.get_active_immutable();
    assert_eq!(ditto.id, PokemonName::MACHAMP, "species must be copied");
    assert_eq!(
        ditto.types,
        (PokemonType::FIGHTING, PokemonType::TYPELESS),
        "current types must be copied"
    );
    assert_eq!(
        ditto.base_types,
        (PokemonType::NORMAL, PokemonType::TYPELESS),
        "base_types is the revert anchor and must NOT be overwritten"
    );
    assert_eq!((ditto.attack, ditto.defense), (296, 196));
    assert_eq!((ditto.special_attack, ditto.special_defense), (166, 206));
    assert_eq!(ditto.speed, 146);
    // `if (this.battle.gen > 2) this.setAbility(pokemon.ability, ...)` — verified
    // against the sim: a Ditto that copies Flygon is `-immune` to Earthquake.
    assert_eq!(ditto.ability, Abilities::GUTS, "ability must be copied");
    assert_eq!(
        ditto.base_ability,
        Abilities::LIMBER,
        "base_ability is what ability_on_switch_out restores and must not move"
    );
}

/// `transformInto` never touches hp/maxhp/status: the loop it runs is over
/// `storedStats`, which is `StatIDExceptHP`.
#[test]
fn transform_does_not_copy_hp_or_status() {
    let mut state = transform_state();
    state.side_two.get_active().status = PokemonStatus::PARALYZE;
    transform_and_apply(&mut state);

    let ditto = state.side_one.get_active_immutable();
    assert_eq!(
        (ditto.hp, ditto.maxhp),
        (200, 200),
        "HP must NOT be copied (the target is at 90/321)"
    );
    assert_eq!(
        ditto.status,
        PokemonStatus::NONE,
        "status must NOT be copied"
    );
}

/// Every copied slot gets `Math.min(5, move.pp)` PP, and no gen3 move has a base
/// PP below 5 — so exactly 5. A target slot the engine models as empty
/// (`Choices::NONE`) stays empty at 0 PP so it cannot be selected.
#[test]
fn transform_copies_moves_at_five_pp() {
    let mut state = transform_state();
    transform_and_apply(&mut state);

    let moves = &state.side_one.get_active_immutable().moves;
    assert_eq!(moves.m0.id, Choices::BULKUP);
    assert_eq!(moves.m1.id, Choices::CROSSCHOP);
    assert_eq!(moves.m2.id, Choices::SPLASH);
    assert_eq!(moves.m3.id, Choices::NONE);
    assert_eq!(
        (moves.m0.pp, moves.m1.pp, moves.m2.pp),
        (5, 5, 5),
        "copied slots hold 5 PP regardless of the target's remaining PP"
    );
    assert_eq!(moves.m3.pp, 0, "an empty slot must stay unselectable");
    assert_eq!(
        moves.m0.choice.move_id,
        Choices::BULKUP,
        "the cached Choice must be re-resolved, not left pointing at Transform"
    );

    // The point of the whole patch: SEARCH must see the copied moves as options.
    let (side_one_options, _) = state.root_get_all_options();
    let move_options: Vec<_> = side_one_options
        .iter()
        .filter_map(|option| match option {
            MoveChoice::Move(index) => Some(state.side_one.get_active_immutable().moves[index].id),
            _ => None,
        })
        .collect();
    assert_eq!(
        move_options,
        vec![Choices::BULKUP, Choices::CROSSCHOP, Choices::SPLASH],
        "the transformed Pokemon's options are the copied moveset, not [Transform]"
    );
}

/// `for (boostName in pokemon.boosts) this.boosts[boostName] = pokemon.boosts[boostName]`
/// — verified against the sim: after copying a +6/+6 Machamp, the transformed
/// Ditto's own Bulk Up reports `|-boost|p1a: Ditto|atk|0`.
#[test]
fn transform_copies_boosts() {
    let mut state = transform_state();
    state.side_one.attack_boost = -2;
    state.side_two.attack_boost = 6;
    state.side_two.defense_boost = 6;
    state.side_two.evasion_boost = 3;
    transform_and_apply(&mut state);

    assert_eq!(state.side_one.attack_boost, 6);
    assert_eq!(state.side_one.defense_boost, 6);
    assert_eq!(state.side_one.evasion_boost, 3);
    assert_eq!(
        state.side_two.attack_boost, 6,
        "the target's own boosts are untouched"
    );
}

// ---------------------------------------------------------------------------
// Failure cases
// ---------------------------------------------------------------------------

/// gen3 inherits gen4's `transform: { flags: { bypasssub, ... } }` and the
/// `volatiles['substitute']` bail-out in `transformInto` is gated on
/// `gen >= 5`. Verified against the sim: Machamp Substitutes, Ditto Transforms,
/// `|-transform|` still fires.
#[test]
fn transform_goes_through_a_substitute() {
    let mut state = transform_state();
    state
        .side_two
        .volatile_statuses
        .insert(PokemonVolatileStatus::SUBSTITUTE);
    state.side_two.substitute_health = 80;
    transform_and_apply(&mut state);

    assert_eq!(
        state.side_one.get_active_immutable().id,
        PokemonName::MACHAMP,
        "gen3 Transform is not blocked by a Substitute"
    );
}

/// `pokemon.transformed && this.battle.gen >= 2` -> return false. Verified
/// against the sim: in a Ditto mirror the second Transform emits
/// `|-fail|p1a: Ditto` and `|debug|move failed because it did nothing`.
#[test]
fn transform_fails_against_an_already_transformed_target() {
    let mut state = transform_state();
    state
        .side_two
        .volatile_statuses
        .insert(PokemonVolatileStatus::TRANSFORMED);

    let before = state.serialize();
    let list = transform_and_apply(&mut state);

    assert!(
        !list.iter().any(|instruction| matches!(
            instruction,
            Instruction::FormeChange(_) | Instruction::ChangeMoveId(_)
        )),
        "Transform into a transformed target must do nothing: {:?}",
        list
    );
    // The only surviving effect is the PP the move itself spends.
    state.reverse_instructions(&list);
    assert_eq!(before, state.serialize());
}

/// The USER already being transformed is only a failure from gen 5 on, so a
/// Ditto that copied a Mew (a gen3 randbats Transform carrier) can Transform
/// again — and doing so must NOT clobber the record of its true base form.
#[test]
fn a_second_transform_keeps_the_original_snapshot() {
    let mut state = transform_state();
    transform_and_apply(&mut state);

    // The copied moveset is Machamp's; give slot 0 Transform back so the
    // transformed Ditto can use it a second time, and swap the target.
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::TRANSFORM);
    state.side_one.get_active().moves[&PokemonMoveIndex::M0].pp = 5;
    let target = state.side_two.get_active();
    target.id = PokemonName::SNORLAX;
    target.attack = 256;
    target.speed = 96;
    target.types = (PokemonType::NORMAL, PokemonType::TYPELESS);
    transform_and_apply(&mut state);

    assert_eq!(state.side_one.get_active_immutable().id, PokemonName::SNORLAX);
    let snapshot = state
        .side_one
        .get_active_immutable()
        .pre_transform
        .as_ref()
        .expect("still transformed");
    assert_eq!(
        snapshot.id,
        PokemonName::DITTO,
        "the snapshot must still be the TRUE base form, not the first copy"
    );
    assert_eq!(snapshot.attack, 132);
}

// ---------------------------------------------------------------------------
// Revert on switch-out
// ---------------------------------------------------------------------------

/// `Pokemon.clearVolatile()` restores `baseMoveSlots`, clears `transformed` and
/// ends with `setSpecies(this.baseSpecies)`. Verified against the sim: after a
/// switch out and back, p1's request lists Ditto with only Transform again.
#[test]
fn transform_reverts_when_the_transformer_switches_out() {
    let mut state = transform_state();
    transform_and_apply(&mut state);
    let transformed = state.serialize();

    let list = only_branch(generate(
        &mut state,
        &MoveChoice::Switch(PokemonIndex::P1),
        &MoveChoice::Move(PokemonMoveIndex::M2),
    ));
    state.apply_instructions(&list);

    let ditto = &state.side_one.pokemon[PokemonIndex::P0];
    assert_eq!(ditto.id, PokemonName::DITTO, "species must revert");
    assert_eq!(
        ditto.types,
        (PokemonType::NORMAL, PokemonType::TYPELESS),
        "types must revert"
    );
    assert_eq!(ditto.ability, Abilities::LIMBER, "ability must revert");
    assert_eq!(
        (
            ditto.attack,
            ditto.defense,
            ditto.special_attack,
            ditto.special_defense,
            ditto.speed
        ),
        (132, 132, 132, 132, 132),
        "stats must revert"
    );
    assert_eq!(ditto.moves.m0.id, Choices::TRANSFORM, "moves must revert");
    assert_eq!(
        ditto.moves.m0.pp, 7,
        "the PP Transform itself spent stays spent across the revert — Showdown's \
         moveSlots and baseMoveSlots share slot objects until the transform, so the \
         deduction is already baked into the base form"
    );
    assert_eq!(ditto.moves.m1.id, Choices::NONE);
    assert!(
        ditto.pre_transform.is_none(),
        "the snapshot must be dropped once it has been consumed"
    );
    assert!(
        !state
            .side_one
            .volatile_statuses
            .contains(&PokemonVolatileStatus::TRANSFORMED),
        "the TRANSFORMED volatile must not survive the switch"
    );

    // ... and the whole switch is itself reversible.
    state.reverse_instructions(&list);
    assert_eq!(transformed, state.serialize(), "switch-out revert must undo");
}

/// Interaction pin with `poke-engine-gen3-residual-defer-on-faint.patch`: a
/// transformed Pokemon that faints mid-turn owes a replacement, and the whole
/// end-of-turn block — and the transform revert with it — lands on the ply that
/// resolves that replacement, not on the faint ply. Showdown reaches the same
/// end state (`faint()` clears the volatile table either way); what this pins is
/// that the deferral did not lose or double-apply the revert.
#[test]
fn a_transformed_pokemon_that_faints_reverts_on_its_replacement() {
    let mut state = transform_state();
    transform_and_apply(&mut state);
    // Put the transformed Ditto on the brink so the next hit is a certain KO, and
    // keep the ply deterministic: copying the target's Speed creates a speed TIE
    // (which forks 50/50), and Cross Chop's 80% accuracy forks again.
    state.side_one.get_active().hp = 1;
    state.side_two.get_active().speed = 400;
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::EARTHQUAKE);

    let faint = only_branch(generate(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M2), // the copied Splash
        &MoveChoice::Move(PokemonMoveIndex::M1), // Earthquake
    ));
    assert!(
        faint.contains(&Instruction::ToggleSideOneForceSwitch),
        "the faint must be flagged as a replacement owed: {:?}",
        faint
    );
    assert!(
        !faint
            .iter()
            .any(|instruction| matches!(instruction, Instruction::FormeChange(_))),
        "the revert belongs to the replacement ply, not the faint ply: {:?}",
        faint
    );
    state.apply_instructions(&faint);
    let fainted = state.serialize();

    let list = only_branch(generate(
        &mut state,
        &MoveChoice::Switch(PokemonIndex::P1),
        &MoveChoice::None,
    ));
    state.apply_instructions(&list);

    let ditto = &state.side_one.pokemon[PokemonIndex::P0];
    assert_eq!(ditto.id, PokemonName::DITTO, "species must revert");
    assert_eq!(ditto.moves.m0.id, Choices::TRANSFORM, "moves must revert");
    assert_eq!(ditto.attack, 132, "stats must revert");
    assert!(ditto.pre_transform.is_none(), "snapshot must be consumed");

    state.reverse_instructions(&list);
    assert_eq!(fainted, state.serialize(), "the replacement ply must invert");
}

// ---------------------------------------------------------------------------
// Instruction reversibility
// ---------------------------------------------------------------------------

/// The property the search depends on: applying the emitted list and then
/// inverting it restores the pre-move state bit-exactly. Asserted on both the
/// full `Debug` rendering and the serialized form so neither a missed field nor
/// a lossy serializer can hide a leak.
#[test]
fn transform_instructions_round_trip_exactly() {
    for boosts in [(0i8, 0i8), (-2, 6), (6, -6)] {
        let mut state = transform_state();
        state.side_one.attack_boost = boosts.0;
        state.side_two.attack_boost = boosts.1;
        state.side_two.speed_boost = boosts.1;
        state
            .side_two
            .get_active()
            .replace_move(PokemonMoveIndex::M2, Choices::SPLASH);

        let debug_before = format!("{:?}", state);
        let serialized_before = state.serialize();

        let list = only_branch(generate(
            &mut state,
            &MoveChoice::Move(PokemonMoveIndex::M0),
            &MoveChoice::Move(PokemonMoveIndex::M2),
        ));
        state.apply_instructions(&list);
        assert_ne!(
            serialized_before,
            state.serialize(),
            "the round trip is vacuous unless Transform actually changed something"
        );

        state.reverse_instructions(&list);
        assert_eq!(debug_before, format!("{:?}", state), "boosts {:?}", boosts);
        assert_eq!(
            serialized_before,
            state.serialize(),
            "boosts {:?}",
            boosts
        );
    }
}

/// The snapshot rides `Pokemon::serialize`/`deserialize` so a transformed state
/// handed to the crate from outside can still be reverted, and an UNtransformed
/// Pokemon serializes byte-for-byte as it did before the patch.
#[test]
fn transform_snapshot_survives_serialization() {
    let untransformed = transform_state();
    assert_eq!(
        untransformed.side_one.get_active_immutable().serialize(),
        State::deserialize(&untransformed.serialize())
            .side_one
            .get_active_immutable()
            .serialize(),
        "an untransformed Pokemon must round-trip unchanged"
    );

    let mut state = transform_state();
    transform_and_apply(&mut state);
    let round_tripped = State::deserialize(&state.serialize());

    let snapshot = round_tripped
        .side_one
        .get_active_immutable()
        .pre_transform
        .as_ref()
        .expect("the snapshot must survive serialization");
    assert_eq!(snapshot.id, PokemonName::DITTO);
    assert_eq!(snapshot.attack, 132);
    assert_eq!(snapshot.moves[0], (Choices::TRANSFORM, 7));
    assert_eq!(snapshot.moves[1], (Choices::NONE, 0));
    assert_eq!(
        state.side_one.get_active_immutable().serialize(),
        round_tripped.side_one.get_active_immutable().serialize()
    );
    // Whole-State fixed point. This only holds because
    // `poke-engine-gen3-state-roundtrip.patch` stopped `Side::deserialize` from
    // reading the volatile set's trailing separator back as a `NONE` volatile —
    // a transformed Pokemon always carries TRANSFORMED and TYPECHANGE, so this
    // assertion is one of the states that used to drift. See
    // tests/gen3_state_roundtrip.rs.
    assert_eq!(state.serialize(), round_tripped.serialize());
}

/// A state that carries the TRANSFORMED volatile with no snapshot (an
/// engine-world constructor that expresses an already-transformed Ditto without
/// filling the snapshot in) must degrade to "no revert", never to a panic or a
/// half-reverted Pokemon.
#[test]
fn switching_out_without_a_snapshot_is_a_no_op_revert() {
    let mut state = transform_state();
    state
        .side_one
        .volatile_statuses
        .insert(PokemonVolatileStatus::TRANSFORMED);

    let list = only_branch(generate(
        &mut state,
        &MoveChoice::Switch(PokemonIndex::P1),
        &MoveChoice::Move(PokemonMoveIndex::M2),
    ));
    let before = state.serialize();
    state.apply_instructions(&list);

    assert!(
        !list
            .iter()
            .any(|instruction| matches!(instruction, Instruction::FormeChange(_))),
        "nothing to revert to, so nothing may be reverted: {:?}",
        list
    );
    assert!(
        !state
            .side_one
            .volatile_statuses
            .contains(&PokemonVolatileStatus::TRANSFORMED),
        "the volatile is still dropped on switch-out"
    );

    state.reverse_instructions(&list);
    assert_eq!(before, state.serialize());
}
