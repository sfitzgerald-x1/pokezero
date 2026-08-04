//! Engine-as-environment stepping surface.
//!
//! The search stack drives the engine through [`crate::model`] and never needs
//! a *single realized line of play* — it enumerates every chance branch and
//! backs up an exact expectation. A self-play COLLECTOR wants the opposite:
//! one seeded realization per joint action, advanced in place, plus the
//! protocol lines that realization would have emitted so the caller can keep a
//! fold (and a public-information ledger) in step with it.
//!
//! That is all this module adds — three thin exports over primitives the crate
//! already uses:
//!
//! * [`env_options`] — the legal option surface for both seats plus the
//!   force-switch / terminal flags, i.e. "who must act and with what".
//! * [`env_step`] — generate the chance branches for a joint action, sample
//!   exactly one with a caller-supplied seed weighted by the engine's own
//!   branch percentages, render it to protocol lines, apply it, and hand back
//!   the post-state.
//! * [`env_battle_over`] — the terminal read on its own, for callers that
//!   already hold a state string.
//!
//! Nothing here is search: no tree, no evaluator, no backup. The seeded
//! sampler is the whole point — it is what makes an episode reproducible from
//! `(start state, seed)` alone.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::MoveChoice;
use poke_engine::state::{Side, SideReference, State};

use crate::events::{
    move_choice_from_str, reject_attribution_unsafe, render_branch_events, EventContext,
};
use crate::{move_display, parse_state, sample_branch_index};

/// Parse an option display back into a [`MoveChoice`].
///
/// SHARP EDGE: [`crate::move_display`] renders switches as `switch <species>`
/// to match the poke-engine Python binding's display convention, but the
/// engine's own `MoveChoice::from_string` matches a switch by BARE species and
/// does not strip that prefix — so the crate's displays do not round-trip
/// as-is. Both [`env_options`] and `LeafEncoder.self_action_map` emit the
/// prefixed form, and those are exactly the strings an env hands back here, so
/// the prefix is stripped on the way in. Plain move displays are unaffected.
fn env_move_choice(name: &str, state: &State, side: SideReference) -> PyResult<MoveChoice> {
    match move_choice_from_str(name, state, side) {
        Ok(choice) => Ok(choice),
        Err(err) => match name.strip_prefix("switch ") {
            Some(species) => move_choice_from_str(species, state, side),
            None => Err(err),
        },
    }
}

/// Render one seat's option surface, tagging the "this seat is not acting"
/// shape the engine encodes as a lone [`MoveChoice::None`].
fn options_payload(side: &Side, options: &[MoveChoice]) -> (Vec<String>, bool) {
    let requested =
        !(options.is_empty() || (options.len() == 1 && matches!(options[0], MoveChoice::None)));
    let displays = options.iter().map(|c| move_display(side, c)).collect();
    (displays, requested)
}

/// The legal option surface for both seats at `state_str`.
///
/// Returns JSON:
/// `{"p1": [display...], "p2": [display...], "p1_requested": bool,
///   "p2_requested": bool, "p1_force_switch": bool, "p2_force_switch": bool,
///   "battle_over": f32}`.
///
/// `root = true` uses `root_get_all_options` (force-trapped / slow-uturn
/// aware — the decision-point surface); `false` uses the interior
/// `get_all_options`. A seat that is not acting this ply reports a single
/// `none` option and `*_requested = false`, which is how a forced replacement
/// becomes a one-seat decision point without any Showdown request.
///
/// `battle_over` is the engine's own terminal read: `0.0` while the battle is
/// live, `> 0.0` when side one has won, `< 0.0` when side two has.
#[pyfunction]
#[pyo3(signature = (state_str, root = true))]
pub fn env_options(state_str: &str, root: bool) -> PyResult<String> {
    let state = parse_state(state_str)?;
    let (s1_options, s2_options) = if root {
        state.root_get_all_options()
    } else {
        state.get_all_options()
    };
    let (p1, p1_requested) = options_payload(&state.side_one, &s1_options);
    let (p2, p2_requested) = options_payload(&state.side_two, &s2_options);
    let report = serde_json::json!({
        "p1": p1,
        "p2": p2,
        "p1_requested": p1_requested,
        "p2_requested": p2_requested,
        "p1_force_switch": state.side_one.force_switch,
        "p2_force_switch": state.side_two.force_switch,
        "battle_over": state.battle_is_over(),
    });
    serde_json::to_string(&report)
        .map_err(|e| PyValueError::new_err(format!("serialize options: {e}")))
}

/// The engine's terminal read at `state_str` (`0.0` = live).
#[pyfunction]
pub fn env_battle_over(state_str: &str) -> PyResult<f32> {
    Ok(parse_state(state_str)?.battle_is_over())
}

/// Advance the battle one ply along a single seeded chance branch.
///
/// Enumerates `generate_instructions_from_move_pair(state, s1_move, s2_move,
/// branch_on_damage)`, samples ONE branch with `StdRng::seed_from_u64(seed)`
/// weighted by the engine's own `percentage` field, renders that branch to
/// protocol lines through the instruction→event mapper, applies it, and
/// returns the post-state.
///
/// Returns JSON:
/// `{"post_state", "events": [line...], "turn_completed": bool,
///   "lossy": [slug...], "attribution_unsafe": bool,
///   "attribution_unsafe_reasons": [slug...], "percentage": f32, "branch_index": i64,
///   "branch_count": usize, "battle_over": f32}`.
///
/// `ctx_json` is the [`EventContext`] shape:
/// `{"p1": [display species...], "p2": [...], "turn": N}` in ENGINE PARTY
/// ORDER. `events` are exactly the lines the caller must feed to a
/// `FoldState` (and to any public-information ledger) to stay in step with the
/// realized line of play; `turn_completed` says whether this ply emitted
/// `|turn|N+1`, so the caller can advance its own turn counter.
///
/// Determinism contract: the sampled branch is a pure function of `seed` and
/// the enumerated branch list, so a caller that derives per-ply seeds from an
/// episode seed replays the identical trajectory.
#[pyfunction]
#[pyo3(signature = (state_str, s1_move, s2_move, ctx_json, seed, branch_on_damage = true))]
pub fn env_step(
    state_str: &str,
    s1_move: &str,
    s2_move: &str,
    ctx_json: &str,
    seed: u64,
    branch_on_damage: bool,
) -> PyResult<String> {
    let mut state = parse_state(state_str)?;
    let ctx = EventContext::from_json(ctx_json).map_err(PyValueError::new_err)?;
    let s1 = env_move_choice(s1_move, &state, SideReference::SideOne)?;
    let s2 = env_move_choice(s2_move, &state, SideReference::SideTwo)?;

    let branches = generate_instructions_from_move_pair(&mut state, &s1, &s2, branch_on_damage);
    if branches.is_empty() {
        // The engine declined to produce instructions (already-terminal shapes
        // reach here). Report it as a no-op ply rather than panicking: the
        // caller's terminal check owns the decision to stop.
        let report = serde_json::json!({
            "post_state": state.serialize(),
            "events": ["|"],
            "turn_completed": false,
            "lossy": ["empty_instruction_list"],
            "attribution_unsafe": false,
            "attribution_unsafe_reasons": [],
            "lossy_subcases": [],
            "percentage": 100.0,
            "branch_index": -1,
            "branch_count": 0,
            "battle_over": state.battle_is_over(),
        });
        return serde_json::to_string(&report)
            .map_err(|e| PyValueError::new_err(format!("serialize step: {e}")));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let index = sample_branch_index(&mut rng, &branches);
    let branch = &branches[index];

    // The mapper wants the PRE-branch state; `render_branch_events` mutates
    // and restores in place, so render before applying.
    let rendered = render_branch_events(
        &mut state,
        &s1,
        &s2,
        &branch.instruction_list,
        branch_on_damage,
        &ctx,
    );
    // Do not apply an exact engine endpoint while handing the caller an event
    // stream that cannot prove its action owner. The caller can retain its
    // current fold/state and choose its established fallback/error handling.
    reject_attribution_unsafe(&rendered, "env_step")?;
    state.apply_instructions(&branch.instruction_list);
    let attribution_unsafe = rendered.is_attribution_unsafe();

    let report = serde_json::json!({
        "post_state": state.serialize(),
        "events": rendered.lines,
        "turn_completed": rendered.turn_completed,
        "lossy": rendered.lossy,
        "attribution_unsafe": attribution_unsafe,
        "attribution_unsafe_reasons": rendered.attribution_unsafe,
        "lossy_subcases": rendered.lossy_subcases,
        "percentage": branch.percentage,
        "branch_index": index as i64,
        "branch_count": branches.len(),
        "battle_over": state.battle_is_over(),
    });
    serde_json::to_string(&report)
        .map_err(|e| PyValueError::new_err(format!("serialize step: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = include_str!("test_fixtures/minimal.state");

    fn ctx_json() -> String {
        serde_json::json!({
            "p1": ["Charmander", "Charmeleon"],
            "p2": ["Squirtle", "Wartortle"],
            "turn": 1,
        })
        .to_string()
    }

    fn fixture() -> String {
        MINIMAL.trim().to_string()
    }

    fn attribution_unsafe_fixture() -> String {
        use poke_engine::choices::Choices;
        use poke_engine::engine::state::PokemonVolatileStatus;
        use poke_engine::state::PokemonMoveIndex;

        // Confusion's 40-power hit and High Jump Kick's crash are both 50 HP
        // here, so the engine merges an action-attribution-ambiguous branch.
        let mut state = parse_state(&fixture()).expect("fixture parses");
        state.side_one.get_active().speed = 500;
        state
            .side_one
            .get_active()
            .replace_move(PokemonMoveIndex::M0, Choices::SPLASH);
        state.side_two.get_active().speed = 1;
        state.side_two.get_active().attack = 143;
        state
            .side_two
            .get_active()
            .replace_move(PokemonMoveIndex::M0, Choices::HIGHJUMPKICK);
        state
            .side_two
            .volatile_statuses
            .insert(PokemonVolatileStatus::CONFUSION);
        state.serialize()
    }

    /// The minimal fixture is a 1-v-1, so it can never produce a switch
    /// option. Give each side a live second mon so the switch surface — the
    /// half of the display contract that does NOT round-trip naively — is
    /// actually exercised.
    fn fixture_with_bench() -> String {
        use poke_engine::state::PokemonIndex;
        fn wake_bench(side: &mut Side) {
            let bench = &mut side.pokemon[PokemonIndex::P1];
            bench.hp = 100;
            bench.maxhp = 100;
            bench.level = 100;
        }
        let mut state = parse_state(&fixture()).expect("fixture parses");
        wake_bench(&mut state.side_one);
        wake_bench(&mut state.side_two);
        state.serialize()
    }

    #[test]
    fn options_report_both_seats_and_live_battle() {
        let report: serde_json::Value =
            serde_json::from_str(&env_options(&fixture(), true).expect("options")).expect("json");
        assert!(!report["p1"].as_array().expect("p1 array").is_empty());
        assert!(!report["p2"].as_array().expect("p2 array").is_empty());
        assert!(report["p1_requested"].as_bool().expect("p1_requested"));
        assert!(report["p2_requested"].as_bool().expect("p2_requested"));
        assert_eq!(report["battle_over"].as_f64(), Some(0.0));
    }

    /// Every display the option surface reports must round-trip through
    /// `MoveChoice::from_string` — that is the contract the env relies on when
    /// it hands an action back as a string.
    #[test]
    fn option_displays_round_trip_into_step() {
        let state_str = fixture_with_bench();
        let report: serde_json::Value =
            serde_json::from_str(&env_options(&state_str, true).expect("options")).expect("json");
        let options = report["p1"].as_array().expect("p1 array");
        // Guard the guard: if the bench never produced a switch option, this
        // test would silently degrade back to move-only coverage.
        assert!(
            options
                .iter()
                .any(|o| o.as_str().unwrap_or("").starts_with("switch ")),
            "fixture produced no switch option: {options:?}"
        );
        let s2 = report["p2"][0].as_str().expect("p2 option").to_string();
        for option in options {
            let s1 = option.as_str().expect("display");
            env_step(&state_str, s1, &s2, &ctx_json(), 7, true)
                .unwrap_or_else(|e| panic!("step rejected round-tripped display {s1:?}: {e}"));
        }
    }

    #[test]
    fn step_is_deterministic_in_seed_and_varies_across_seeds() {
        let state_str = fixture();
        let step = |seed: u64| {
            serde_json::from_str::<serde_json::Value>(
                &env_step(&state_str, "tackle", "tackle", &ctx_json(), seed, true).expect("step"),
            )
            .expect("json")
        };
        // Same seed => byte-identical realization.
        assert_eq!(step(11), step(11));
        // The damage roll branches, so *some* seed pair must disagree;
        // otherwise the sampler is ignoring its seed.
        let first = step(0);
        let differs = (1..64u64).any(|seed| step(seed) != first);
        assert!(differs, "sampler never varied across 64 seeds");
    }

    #[test]
    fn step_emits_fold_lines_and_a_post_state_that_reparses() {
        let report: serde_json::Value = serde_json::from_str(
            &env_step(&fixture(), "tackle", "tackle", &ctx_json(), 3, true).expect("step"),
        )
        .expect("json");
        let events = report["events"].as_array().expect("events");
        assert!(!events.is_empty(), "a resolved ply must render lines");
        assert_eq!(events[0].as_str(), Some("|"));
        let post = report["post_state"].as_str().expect("post_state");
        // The post-state must be a state string the crate can consume again —
        // that is what makes stepping chainable.
        env_options(post, true).expect("post_state reparses");
        assert!(report["branch_count"].as_u64().expect("branch_count") >= 1);
    }

    /// Branch sampling must respect the engine's percentages, not just pick
    /// index 0. Over many seeds a damage-branched ply must visit more than one
    /// branch index.
    #[test]
    fn sampler_visits_multiple_branches_across_seeds() {
        let state_str = fixture();
        let mut seen = std::collections::BTreeSet::new();
        for seed in 0..128u64 {
            let report: serde_json::Value = serde_json::from_str(
                &env_step(&state_str, "tackle", "tackle", &ctx_json(), seed, true).expect("step"),
            )
            .expect("json");
            seen.insert(report["branch_index"].as_i64().expect("branch_index"));
        }
        assert!(
            seen.len() > 1,
            "damage-branched ply collapsed to a single branch: {seen:?}"
        );
    }

    /// `branch_on_damage = false` is the deep-tree regime: the damage roll
    /// stops splitting, so the enumerated branch count must strictly shrink.
    #[test]
    fn branch_on_damage_toggle_shrinks_the_branch_count() {
        let count = |branch_on_damage: bool| {
            serde_json::from_str::<serde_json::Value>(
                &env_step(
                    &fixture(),
                    "tackle",
                    "tackle",
                    &ctx_json(),
                    5,
                    branch_on_damage,
                )
                .expect("step"),
            )
            .expect("json")["branch_count"]
                .as_u64()
                .expect("branch_count")
        };
        let branched = count(true);
        let unbranched = count(false);
        assert!(
            unbranched < branched,
            "branch_on_damage=false did not shrink the surface: {unbranched} vs {branched}"
        );
        assert!(unbranched >= 1);
    }

    #[test]
    fn invalid_move_is_rejected_rather_than_panicking() {
        assert!(env_step(&fixture(), "not-a-move", "tackle", &ctx_json(), 1, true).is_err());
        assert!(env_step(&fixture(), "tackle", "not-a-move", &ctx_json(), 1, true).is_err());
    }

    #[test]
    fn malformed_ctx_json_is_rejected() {
        assert!(env_step(&fixture(), "tackle", "tackle", "{\"turn\": 1}", 1, true).is_err());
    }

    #[test]
    fn malformed_state_is_a_value_error_not_a_panic() {
        assert!(env_options("definitely not a state", true).is_err());
        assert!(env_battle_over("definitely not a state").is_err());
    }

    #[test]
    fn unsafe_renderer_branch_is_rejected_before_env_post_state() {
        Python::initialize();
        let state = attribution_unsafe_fixture();
        let error = (0..128u64)
            .find_map(|seed| {
                env_step(&state, "splash", "highjumpkick", &ctx_json(), seed, false)
                    .err()
                    .map(|error| error.to_string())
            })
            .expect("at least one seed must sample the known ambiguous branch");
        assert!(
            error.contains("attribution-unsafe renderer branch rejected before env_step"),
            "{error}"
        );
    }
}
