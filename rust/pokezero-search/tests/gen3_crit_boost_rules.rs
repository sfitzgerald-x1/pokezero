//! Gen 3 critical hits ignore the attacker's NEGATIVE stat stages and the
//! defender's POSITIVE stat stages — and nothing else. Asserted against the
//! vendored gen3-patched poke-engine.
//!
//! Showdown, `sim/battle-actions.ts:1687-1704` (no gen3-chain mod overrides it):
//!
//! ```text
//! let ignoreNegativeOffensive = !!move.ignoreNegativeOffensive;
//! let ignorePositiveDefensive = !!move.ignorePositiveDefensive;
//! if (moveHit.crit) {
//!     ignoreNegativeOffensive = true;
//!     ignorePositiveDefensive = true;
//! }
//! const ignoreOffensive = !!(move.ignoreOffensive || (ignoreNegativeOffensive && atkBoosts < 0));
//! const ignoreDefensive = !!(move.ignoreDefensive || (ignorePositiveDefensive && defBoosts > 0));
//! if (ignoreOffensive) atkBoosts = 0;
//! if (ignoreDefensive) defBoosts = 0;
//! ```
//!
//! So the sign matters, in both directions:
//!
//! | stage | on a crit |
//! |---|---|
//! | attacker's boost > 0 | KEPT — the crit still benefits |
//! | attacker's boost < 0 | IGNORED — the crit escapes the drop |
//! | defender's boost > 0 | IGNORED — the crit punches through the setup |
//! | defender's boost < 0 | KEPT — the crit still benefits |
//!
//! Why this file exists: the two-turn release audit (ledger J.4) measured the
//! engine's own branches at −41 non-crit / −163 crit and reasoned from the ~4x
//! spread. The first explanation offered — Light Screen, which crits also ignore
//! — turned out to be **unreachable in gen3 randbats** (Light Screen and Reflect
//! are 0/220 species). Dumping the actual repro row settled it: the defender
//! (Mew, seed 1350004 step 66) carried **special_defense_boost = +2** with every
//! side condition at zero. Solar Beam is Special, so the +2 SpD halves the
//! non-crit branch while the crit ignores it — 2x2 = 4x, from a rule that is
//! implemented CORRECTLY.
//!
//! These pins lock all four sign cases so that inference is never load-bearing
//! again, and so a future edit cannot "simplify" the rule into ignoring every
//! boost on a crit.

use poke_engine::choices::{Choices, MOVES};
use poke_engine::engine::damage_calc::{calculate_damage, DamageRolls};
use poke_engine::state::{PokemonMoveIndex, SideReference, State};

/// (non-crit, crit) for `move_id` with the given attacker/defender stage on the
/// stat that move actually uses.
fn damage(move_id: Choices, attacker_stage: i8, defender_stage: i8) -> (i16, i16) {
    let mut state = State::default();
    let mut choice = MOVES.get(&move_id).unwrap().clone();
    choice.move_index = PokemonMoveIndex::M0;

    match move_id {
        Choices::TACKLE => {
            state.side_one.attack_boost = attacker_stage;
            state.side_two.defense_boost = defender_stage;
        }
        _ => {
            state.side_one.special_attack_boost = attacker_stage;
            state.side_two.special_defense_boost = defender_stage;
        }
    }

    calculate_damage(&state, &SideReference::SideOne, &choice, DamageRolls::Max)
        .expect("damaging move")
}

/// Special and physical both, so a fix applied to one arm cannot silently miss
/// the other — the engine implements them as two separate match arms.
const CASES: [Choices; 2] = [Choices::PSYCHIC, Choices::TACKLE];

// ---------------------------------------------------------------------------
// The four sign cases
// ---------------------------------------------------------------------------

/// Baseline: with no stages anywhere, a gen3 crit is 2x.
///
/// Within one point, not exactly: the engine applies `CRIT_MULTIPLIER` to the
/// unfloored damage and floors afterwards, so `floor(2x)` can exceed
/// `2*floor(x)` by one. Asserting exact equality passes for Psychic and fails
/// for Tackle purely on where the fraction lands, which would be a fixture
/// coincidence rather than a rule.
#[test]
fn an_unboosted_crit_is_double() {
    for move_id in CASES {
        let (normal, crit) = damage(move_id, 0, 0);
        assert!(
            (crit - normal * 2).abs() <= 1,
            "{:?}: unboosted crit is 2x within a rounding point ({} vs {})",
            move_id,
            crit,
            normal * 2
        );
    }
}

/// Attacker's POSITIVE stages are KEPT on a crit — the crit still benefits, so
/// it differs from the unboosted crit.
#[test]
fn a_crit_keeps_the_attackers_positive_stages() {
    for move_id in CASES {
        let (_, boosted_crit) = damage(move_id, 2, 0);
        let (_, plain_crit) = damage(move_id, 0, 0);
        assert!(
            boosted_crit > plain_crit,
            "{:?}: a crit must still benefit from the attacker's +2 ({} vs {})",
            move_id,
            boosted_crit,
            plain_crit
        );
    }
}

/// Attacker's NEGATIVE stages are IGNORED on a crit — the crit lands exactly as
/// if the drop were not there.
#[test]
fn a_crit_ignores_the_attackers_negative_stages() {
    for move_id in CASES {
        let (normal, crit) = damage(move_id, -2, 0);
        let (_, plain_crit) = damage(move_id, 0, 0);
        assert_eq!(
            crit, plain_crit,
            "{:?}: a crit must escape the attacker's -2 entirely",
            move_id
        );
        assert!(
            normal < crit / 2,
            "{:?}: the non-crit branch still suffers the drop",
            move_id
        );
    }
}

/// Defender's POSITIVE stages are IGNORED on a crit. **This is the rule that
/// produced the 4x in the release-damage repro**: the defender's +2 SpD halves
/// the non-crit branch and leaves the crit untouched.
#[test]
fn a_crit_ignores_the_defenders_positive_stages() {
    for move_id in CASES {
        let (normal, crit) = damage(move_id, 0, 2);
        let (plain_normal, plain_crit) = damage(move_id, 0, 0);
        assert_eq!(
            crit, plain_crit,
            "{:?}: a crit must punch straight through the defender's +2",
            move_id
        );
        assert!(
            normal < plain_normal,
            "{:?}: the non-crit branch is still reduced by it",
            move_id
        );
    }
}

/// Defender's NEGATIVE stages are KEPT on a crit — the crit still benefits from
/// the drop, so it is bigger than the unboosted crit.
#[test]
fn a_crit_keeps_the_defenders_negative_stages() {
    for move_id in CASES {
        let (_, crit) = damage(move_id, 0, -2);
        let (_, plain_crit) = damage(move_id, 0, 0);
        assert!(
            crit > plain_crit,
            "{:?}: a crit must still benefit from the defender's -2 ({} vs {})",
            move_id,
            crit,
            plain_crit
        );
    }
}

// ---------------------------------------------------------------------------
// The repro's actual mechanism
// ---------------------------------------------------------------------------

/// The release-damage row, reduced to its mechanism: a Special hit into a
/// defender holding +2 SpD, with NO side conditions. That is the state dumped
/// from seed 1350004 step 66 — `special_defense_boost = 2`, side_conditions all
/// zero — and it reproduces the ~4x spread between the engine's own branches
/// without a screen anywhere in sight.
///
/// The distinction matters because Light Screen and Reflect are **0/220 in the
/// gen3 randbats pool**, so no screen mechanism can explain a row that came out
/// of the randbats re-measurement.
#[test]
fn a_defender_positive_special_stage_produces_the_repro_ratio() {
    let (normal, crit) = damage(Choices::PSYCHIC, 0, 2);
    let ratio = crit as f32 / normal as f32;
    assert!(
        (ratio - 4.0).abs() < 0.1,
        "+2 SpD on the defender gives the ~4x the audit saw: got {:.2} ({} vs {})",
        ratio,
        normal,
        crit
    );
}

/// And the control that rules the other explanation out on the row's own terms:
/// at +0 the ratio is the plain 2x, so the spread really is the stage and not
/// something intrinsic to the crit branch.
#[test]
fn the_same_move_without_the_stage_is_only_double() {
    let (normal, crit) = damage(Choices::PSYCHIC, 0, 0);
    let ratio = crit as f32 / normal as f32;
    assert!(
        (ratio - 2.0).abs() < 0.1,
        "without the defender's stage the ratio is 2x: got {:.2}",
        ratio
    );
}
