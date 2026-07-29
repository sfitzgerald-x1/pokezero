//! Gen 3 status-move accuracy pins, asserted directly against the vendored
//! gen3-patched poke-engine (`third_party/poke-engine-src/`).
//!
//! The fix (`poke-engine-gen3-toxic-accuracy.patch`) is narrow: a Poison-type
//! using Toxic does NOT bypass the accuracy roll in gen3. Showdown gates that
//! rule on **generation 8** in `sim/battle-actions.ts:622` and `:726` —
//!
//! ```text
//! this.battle.gen >= 8 && move.id === 'toxic' && pokemon.hasType('Poison')
//! ```
//!
//! — and upstream applied it unconditionally, so a Poison-type Toxic was priced
//! as guaranteed poison.
//!
//! The rest of this file is the audit that established the fix is narrow: every
//! other imperfect-accuracy status move in the gen3 randbats pool already
//! branches correctly, so this was never a class failure of status-move
//! accuracy. Those pins are controls, not restatements of the fix.

use poke_engine::choices::Choices;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::MoveChoice;
use poke_engine::instruction::Instruction;
use poke_engine::state::{PokemonMoveIndex, PokemonStatus, PokemonType, SideReference, State};

/// Branch percentages for `move_id` used by an attacker of `attacker_type`,
/// sorted ascending so the miss branch reads first.
fn branch_percentages(move_id: Choices, attacker_type: PokemonType) -> Vec<f32> {
    let mut state = State::default();
    {
        let attacker = state.side_one.get_active();
        attacker.replace_move(PokemonMoveIndex::M0, move_id);
        attacker.types = (attacker_type, PokemonType::TYPELESS);
    }
    let before = format!("{:?}", state);
    let out = generate_instructions_from_move_pair(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::None,
        false,
    );
    assert_eq!(before, format!("{:?}", state), "state was mutated");
    for branch in &out {
        let mut probe = state.clone();
        let snapshot = format!("{:?}", probe);
        probe.apply_instructions(&branch.instruction_list);
        probe.reverse_instructions(&branch.instruction_list);
        assert_eq!(snapshot, format!("{:?}", probe), "branch did not revert");
    }
    let mut pcts: Vec<f32> = out.iter().map(|b| b.percentage).collect();
    pcts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    pcts
}

/// The branches that apply `status` to the defending side, and those that do not.
fn status_branches(
    move_id: Choices,
    attacker_type: PokemonType,
    status: PokemonStatus,
) -> (f32, f32) {
    let mut state = State::default();
    {
        let attacker = state.side_one.get_active();
        attacker.replace_move(PokemonMoveIndex::M0, move_id);
        attacker.types = (attacker_type, PokemonType::TYPELESS);
    }
    let out = generate_instructions_from_move_pair(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::None,
        false,
    );
    let mut applied = 0.0;
    let mut missed = 0.0;
    for branch in &out {
        let hits = branch
            .instruction_list
            .iter()
            .any(|instruction| match instruction {
                Instruction::ChangeStatus(change) => {
                    change.side_ref == SideReference::SideTwo && change.new_status == status
                }
                _ => false,
            });
        if hits {
            applied += branch.percentage;
        } else {
            missed += branch.percentage;
        }
    }
    (applied, missed)
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 0.01
}

// ---------------------------------------------------------------------------
// The fix
// ---------------------------------------------------------------------------

/// A POISON-type Toxic user still rolls 85/15 in gen3. Upstream returned a
/// single 100% branch carrying the poison, because it applied Showdown's
/// gen8-gated never-miss rule to every generation.
#[test]
fn a_poison_type_toxic_user_still_rolls_its_accuracy() {
    let pcts = branch_percentages(Choices::TOXIC, PokemonType::POISON);
    assert_eq!(pcts.len(), 2, "Toxic must branch hit/miss: {:?}", pcts);
    assert!(close(pcts[0], 15.0), "miss branch is 15%: {:?}", pcts);
    assert!(close(pcts[1], 85.0), "hit branch is 85%: {:?}", pcts);
}

/// ...and the miss branch must carry no poison at all.
#[test]
fn the_toxic_miss_branch_applies_no_status() {
    for attacker_type in [PokemonType::POISON, PokemonType::NORMAL] {
        let (applied, missed) =
            status_branches(Choices::TOXIC, attacker_type, PokemonStatus::TOXIC);
        assert!(
            close(applied, 85.0),
            "85% of the mass poisons ({:?} user): {}",
            attacker_type,
            applied
        );
        assert!(
            close(missed, 15.0),
            "15% of the mass does not ({:?} user): {}",
            attacker_type,
            missed
        );
    }
}

/// The control that localises the bug: a NON-Poison Toxic user was already
/// correct, so the fix must not have changed it.
#[test]
fn a_non_poison_toxic_user_was_already_correct() {
    let pcts = branch_percentages(Choices::TOXIC, PokemonType::NORMAL);
    assert_eq!(pcts.len(), 2, "{:?}", pcts);
    assert!(close(pcts[0], 15.0) && close(pcts[1], 85.0), "{:?}", pcts);
}

// ---------------------------------------------------------------------------
// The audit: every other imperfect-accuracy status move in the pool
// ---------------------------------------------------------------------------

/// Accuracy branching is NOT broken as a class — this was a Toxic-specific
/// bypass. Every imperfect-accuracy status move in the gen3 randbats pool
/// branches at its real gen3 accuracy, walked through the full inheritance
/// chain (gen3 -> gen4 -> ... -> current), not a truncated one.
#[test]
fn every_imperfect_accuracy_status_move_in_the_pool_branches() {
    for (move_id, accuracy) in [
        (Choices::HYPNOSIS, 60.0),
        (Choices::LOVELYKISS, 75.0),
        (Choices::STUNSPORE, 75.0),
        (Choices::WILLOWISP, 75.0),
        (Choices::SLEEPPOWDER, 75.0),
        (Choices::TOXIC, 85.0),
        (Choices::LEECHSEED, 90.0),
    ] {
        let pcts = branch_percentages(move_id, PokemonType::NORMAL);
        assert_eq!(pcts.len(), 2, "{:?} must branch: {:?}", move_id, pcts);
        assert!(
            close(pcts[0], 100.0 - accuracy) && close(pcts[1], accuracy),
            "{:?} must branch {}/{}: {:?}",
            move_id,
            accuracy,
            100.0 - accuracy,
            pcts
        );
    }
}

/// The other half of the audit: Thunder Wave really is 100 in gen3, so a single
/// branch is correct for it. Reading a truncated inheritance chain
/// (gen3 -> gen4 -> gen5 -> base, skipping gen6/gen7) reports 90 and would make
/// this look like a missing miss branch.
#[test]
fn thunder_wave_is_perfectly_accurate_in_gen3() {
    let pcts = branch_percentages(Choices::THUNDERWAVE, PokemonType::NORMAL);
    assert_eq!(
        pcts.len(),
        1,
        "gen3 Thunder Wave never misses, so there is no branch: {:?}",
        pcts
    );
    assert!(close(pcts[0], 100.0), "{:?}", pcts);
}
