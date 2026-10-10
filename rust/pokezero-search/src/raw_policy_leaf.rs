//! Terminal-only raw masked-argmax valuation on the incumbent encoded tree.
//! Both seats use their own certified private-safe policy bridge. There is no
//! auxiliary-head, uniform-policy, HP, or checkpoint-value fallback. Caps and
//! invalid surfaces refuse the world; a deadline cancels the entire batch.

use std::collections::HashMap;
use std::time::Instant;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::state::State;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::{rngs::StdRng, SeedableRng};
use crate::events::EventContext;
use crate::leaf::{LeafMeta, LeafMetaCtx};
use crate::policy_bridge::PolicyOpponentBridge;
use crate::tree::DecisionNode;

pub(crate) struct RawPolicyLeaf {
    pub bridges: [PolicyOpponentBridge; 2],
    pub max_plies: u32,
    pub seed: u64,
    pub branch_on_damage: bool,
}

#[derive(Default)]
pub(crate) struct RawPolicyLeafStats {
    pub started: usize,
    pub terminal: usize,
    pub cap_refusals: usize,
    pub dead_end_refusals: usize,
    pub plies: usize,
    pub provider_calls: usize,
    pub policy_evals: usize,
    pub policy_nanos: u128,
    pub cancelled_traversals: usize,
    pub cancelled_rows: usize,
    pub discarded_terminal_rows: usize,
}

fn leaf_seed(seed: u64, ordinal: u64) -> u64 {
    let mut value = seed ^ 0x5241_5750_4f4c_4943 ^ ordinal.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

pub(crate) fn expired(deadline: Option<Instant>) -> bool {
    deadline.is_some_and(|deadline| Instant::now() >= deadline)
}

/// `None` is cancellation, NEVER a draw/leaf score. An owned hypothetical
/// state and forked prefix are discarded on every exit; the search state and
/// its persistent public ledgers cannot be changed by a continuation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn price(
    leaf: &State, key: (usize, usize), ordinal: u64, ctx: &EventContext,
    meta: &LeafMeta, meta_ctx: &LeafMetaCtx, cfg: &RawPolicyLeaf,
    deadline: Option<Instant>, stats: &mut RawPolicyLeafStats,
    lossy: &mut crate::abort_telemetry::LossySubcaseLedger,
) -> PyResult<Option<f32>> {
    if expired(deadline) { return Ok(None); }
    let bridges = [cfg.bridges[0].fork_at(key)?, cfg.bridges[1].fork_at(key)?];
    let mut state = leaf.clone();
    let mut ctx = ctx.clone();
    let mut meta = meta.clone();
    let mut parent = key;
    let mut rng = StdRng::seed_from_u64(leaf_seed(cfg.seed, ordinal));
    stats.started += 1;
    let result = (|| {
        for ply in 0..=cfg.max_plies {
            if expired(deadline) { return Ok(None); }
            // Native's battle_is_over gives seat two the simultaneous-faint
            // case. That is not a certified draw/win: refuse instead.
            let one_dead = state.side_one.pokemon.into_iter().all(|p| p.hp <= 0);
            let two_dead = state.side_two.pokemon.into_iter().all(|p| p.hp <= 0);
            if one_dead && two_dead {
                return Err(PyValueError::new_err("raw policy terminal: simultaneous team faint is unresolved"));
            }
            let over = state.battle_is_over();
            if over != 0. {
                stats.terminal += 1;
                return Ok(Some(if over > 0. { 1. } else { 0. }));
            }
            if ply == cfg.max_plies {
                stats.cap_refusals += 1;
                return Err(PyValueError::new_err("raw policy terminal: nonterminal ply cap; no fallback value"));
            }
            let (s1_options, s2_options) = state.get_all_options();
            if s1_options.is_empty() || s2_options.is_empty() {
                stats.dead_end_refusals += 1;
                return Err(PyValueError::new_err("raw policy terminal: empty legal surface"));
            }
            let node = DecisionNode { visits: 0, depth: 0,
                s1_stats: crate::make_stats(&state.side_one, &s1_options),
                s2_stats: crate::make_stats(&state.side_two, &s2_options),
                s1_options, s2_options, children: HashMap::new() };
            let i = bridges[0].provide(&state, &node, Some(parent), &ctx)?.certain_action()?;
            if expired(deadline) { return Ok(None); }
            let j = bridges[1].provide(&state, &node, Some(parent), &ctx)?.certain_action()?;
            if expired(deadline) { return Ok(None); }
            let s1 = node.s1_options[i];
            let s2 = node.s2_options[j];
            let branches = generate_instructions_from_move_pair(
                &mut state, &s1, &s2, cfg.branch_on_damage);
            let mass: f32 = branches.iter().map(|b| b.percentage).sum();
            if branches.is_empty() || !mass.is_finite() || (mass - 100.).abs() > 0.5
                || branches.iter().any(|b| !b.percentage.is_finite() || b.percentage < 0.) {
                return Err(PyValueError::new_err("raw policy terminal: invalid chance mass"));
            }
            let selected = crate::sample_branch_index(&mut rng, &branches);
            let instructions = &branches[selected].instruction_list;
            if instructions.is_empty() {
                stats.dead_end_refusals += 1;
                return Err(PyValueError::new_err("raw policy terminal: nonterminal dead end"));
            }
            let rendered = crate::events::render_branch_events(
                &mut state, &s1, &s2, instructions, cfg.branch_on_damage, &ctx);
            for subcase in &rendered.lossy_subcases { lossy.record(subcase); }
            crate::events::reject_attribution_unsafe(&rendered, "raw policy terminal")?;
            meta = crate::leaf::evolve_leaf_meta_with_status_transitions(
                &meta, &rendered.lines, meta_ctx, &rendered.active_status_transitions);
            // This namespace is local to a fork and cannot collide with any
            // actual tree branch or contaminate another leaf's continuation.
            let next = (usize::MAX, ply as usize);
            for bridge in &bridges { bridge.record_linear(parent, next, &rendered.lines, &ctx, &meta)?; }
            state.apply_instructions(instructions);
            if rendered.turn_completed { ctx.turn += 1; }
            parent = next;
            stats.plies += 1;
        }
        unreachable!("inclusive cap check returns on the last boundary")
    })();
    for bridge in &bridges {
        stats.provider_calls += bridge.provider_calls.get();
        stats.policy_evals += bridge.evaluations.get();
        stats.policy_nanos += bridge.policy_nanos.get();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use poke_engine::choices::Choices;
    use poke_engine::state::PokemonMoveIndex;
    use pyo3::ffi::c_str;

    fn fixture(side_one_wins: bool, cap: u32) -> (State, EventContext, LeafMeta, RawPolicyLeaf) {
        Python::initialize();
        let callback = Python::attach(|py| {
            PyModule::from_code(py, c_str!("import json\ndef choose(payload):\n    row = json.loads(payload)['native_request_bundle']['native_action_indices']\n    return [float(i == 0) for i in range(len(row))]\n"),
                c_str!("raw_leaf_fixture.py"), c_str!("raw_leaf_fixture"))
                .unwrap().getattr("choose").unwrap().unbind()
        });
        let mut state = crate::parse_state(include_str!("test_fixtures/minimal.state").trim()).unwrap();
        for (seat, side) in [&mut state.side_one, &mut state.side_two].into_iter().enumerate() {
            let winner = (seat == 0) == side_one_wins;
            let active = side.get_active();
            active.hp = 200;
            active.maxhp = 200;
            active.replace_move(PokemonMoveIndex::M0, if winner { Choices::SEISMICTOSS } else { Choices::SPLASH });
            active.replace_move(PokemonMoveIndex::M1, Choices::NONE);
            active.moves.m0.pp = 32;
        }
        let ctx = EventContext { species: [vec!["Charmander".into()], vec!["Squirtle".into()]],
            turn: 1, hp_percent: [false, false] };
        let meta = LeafMeta { active: ["charmander".into(), "squirtle".into()], ..Default::default() };
        let bridge = |side_one: bool| PolicyOpponentBridge::new(
            Python::attach(|py| callback.clone_ref(py)), side_one,
            ctx.species[usize::from(!side_one)].clone(),
            [("seismictoss".into(), 32), ("splash".into(), 64)].into(),
            [("seismictoss".into(), 20), ("splash".into(), 40)].into(),
            if side_one { state.side_one.clone() } else { state.side_two.clone() }, ctx.clone());
        let cfg = RawPolicyLeaf { bridges: [bridge(true), bridge(false)],
            max_plies: cap, seed: 17, branch_on_damage: true };
        for bridge in &cfg.bridges { bridge.record(None, (0, 0), &[], &ctx, &meta).unwrap(); }
        (state, ctx, meta, cfg)
    }

    #[test]
    fn terminal_only_both_seats_and_frontier_isolation() {
        for side_one_wins in [true, false] {
            let (state, ctx, meta, cfg) = fixture(side_one_wins, 2);
            let before = state.serialize();
            for _ in 0..2 {
                let mut stats = RawPolicyLeafStats::default();
                let value = price(&state, (0, 0), 0, &ctx, &meta, &LeafMetaCtx::default(), &cfg,
                    None, &mut stats, &mut Default::default()).unwrap();
                assert_eq!(value, Some(if side_one_wins { 1. } else { 0. }));
                assert_eq!(stats.started, 1);
                assert_eq!(stats.terminal, 1);
                assert_eq!(stats.plies, 2);
                assert_eq!(stats.provider_calls, 4);
                assert_eq!(state.serialize(), before);
                assert_eq!(cfg.bridges[0].provider_calls.get(), 0);
                assert_eq!(cfg.bridges[1].provider_calls.get(), 0);
            }
        }
    }

    #[test]
    fn a_cap_refuses_instead_of_returning_hp_model_or_draw() {
        let (state, ctx, meta, cfg) = fixture(true, 1);
        let mut stats = RawPolicyLeafStats::default();
        let error = price(&state, (0, 0), 0, &ctx, &meta, &LeafMetaCtx::default(), &cfg,
            None, &mut stats, &mut Default::default()).unwrap_err();
        assert!(error.to_string().contains("nonterminal ply cap"));
        assert_eq!(stats.terminal, 0);
        assert_eq!(stats.cap_refusals, 1);
        assert_eq!(stats.plies, 1);
    }

    #[test]
    fn expired_deadline_does_not_call_policy_or_produce_a_score() {
        let (state, ctx, meta, cfg) = fixture(true, 2);
        let mut stats = RawPolicyLeafStats::default();
        assert_eq!(price(&state, (0, 0), 0, &ctx, &meta, &LeafMetaCtx::default(), &cfg,
            Some(Instant::now()), &mut stats, &mut Default::default()).unwrap(), None);
        assert_eq!(stats.started, 0);
        assert_eq!(stats.provider_calls, 0);
    }

    #[test]
    fn uncertain_callbacks_are_never_native_order_argmax_or_sampled() {
        for weights in [&[0.5, 0.5][..], &[0., 1., 1.][..], &[1., 0.0001][..]] {
            let distribution = crate::policy_opponent::ActionDistribution::new(weights, weights.len()).unwrap();
            assert!(distribution.certain_action().is_err());
        }
        assert_eq!(crate::policy_opponent::ActionDistribution::new(&[0., 1., 0.], 3)
            .unwrap().certain_action().unwrap(), 1);
    }
    #[test]
    fn seed_is_separate_reproducible_and_leaf_owned() {
        assert_eq!(leaf_seed(17, 2), leaf_seed(17, 2));
        assert_ne!(leaf_seed(17, 2), 17);
        assert_ne!(leaf_seed(17, 2), leaf_seed(17, 3));
        assert_ne!(leaf_seed(17, 2), leaf_seed(18, 2));
    }
    #[test]
    fn deadline_is_explicit_not_an_outcome() {
        assert!(!expired(None));
        assert!(expired(Some(Instant::now())));
        assert!(!expired(Some(Instant::now() + std::time::Duration::from_secs(60))));
    }
}
