//! Terminal-only raw masked-argmax valuation on the incumbent encoded tree.
//! Both seats use their own certified private-safe policy bridge. There is no
//! auxiliary-head, uniform-policy, HP, or checkpoint-value fallback. Caps and
//! invalid surfaces refuse the world; a deadline cancels the entire batch.

use std::collections::HashMap;
use std::time::Instant;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::{MoveChoice, PokemonVolatileStatus};
use poke_engine::state::{Pokemon, Side, State};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::{rngs::StdRng, SeedableRng};
use crate::events::EventContext;
use crate::leaf::{LeafMeta, LeafMetaCtx};
use crate::policy_bridge::PolicyOpponentBridge;
use crate::policy_request::PrivateTrapObservation;
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
    pub choice_attempts: usize,
    pub trapped_switch_rejections: usize,
    pub private_redecisions: usize,
}

/// Ordinary choices are derived from OWN facts only. Native phase options are
/// retained for public replacement/WAIT boundaries, which never consult hidden
/// trapping. Do not clone an opponent and erase its ability to invent a world.
fn choice_options(side: &Side, replacement_phase: bool, native: &[MoveChoice],
    trap_observation: PrivateTrapObservation) -> Vec<MoveChoice> {
    if replacement_phase {
        return native.to_vec();
    }
    let mut options = Vec::new();
    if side.volatile_statuses.contains(&PokemonVolatileStatus::MUSTRECHARGE) {
        options.push(MoveChoice::None);
    } else if let Some(index) = side.active_is_charging_move() {
        options.push(MoveChoice::Move(index));
    } else {
        side.get_active_immutable().add_available_moves(&mut options, &side.last_used_move,
            side.volatile_statuses.contains(&PokemonVolatileStatus::ENCORE),
            side.volatile_statuses.contains(&PokemonVolatileStatus::TAUNT));
        if trap_observation == PrivateTrapObservation::None && !side.trapped(&Pokemon::default()) {
            side.add_switches(&mut options);
        }
    }
    if options.is_empty() { options.push(MoveChoice::None); }
    options
}

fn choice_node(state: &State, one: Vec<MoveChoice>, two: Vec<MoveChoice>) -> DecisionNode {
    DecisionNode { visits: 0, depth: 0,
        s1_stats: crate::make_stats(&state.side_one, &one),
        s2_stats: crate::make_stats(&state.side_two, &two),
        s1_options: one, s2_options: two, children: HashMap::new() }
}

/// Preserve exact move/phase semantics. The ONLY allowed difference is the
/// switch-attempt surface at a hidden trapping boundary. No missing move,
/// phantom forced choice, duplicated option or unrelated discrepancy is repaired.
fn certify_attempt_surface(native: &[MoveChoice], attempts: &[MoveChoice]) -> PyResult<()> {
    let is_switch = |option: &&MoveChoice| matches!(option, MoveChoice::Switch(_));
    let native_moves: Vec<_> = native.iter().filter(|o| !is_switch(o)).collect();
    let attempt_moves: Vec<_> = attempts.iter().filter(|o| !is_switch(o)).collect();
    let duplicated = |options: &[MoveChoice]| options.iter().enumerate()
        .any(|(index, option)| options[..index].contains(option));
    if duplicated(native) || duplicated(attempts)
        || native_moves != attempt_moves || native.iter().any(|option| !attempts.contains(option))
        || (native != attempts && native.iter().any(|o| matches!(o, MoveChoice::Switch(_)))) {
        return Err(PyValueError::new_err("raw policy terminal: unrelated private/environment surface mismatch"));
    }
    Ok(())
}

/// Both initial choices are submitted before resolving private redecisions.
/// Retain the other choice. A rejection changes no State, public prefix, PP,
/// turn, or chance RNG. Only an actual chosen switch can reveal trapping to its
/// own policy. Each seat has at most one rejection at this request boundary.
#[allow(clippy::too_many_arguments)]
fn select_choices(state: &State, parent: (usize, usize), ctx: &EventContext,
    bridges: &[PolicyOpponentBridge; 2], deadline: Option<Instant>,
    stats: &mut RawPolicyLeafStats) -> PyResult<Option<(MoveChoice, MoveChoice)>> {
    let (one_native, two_native) = state.get_all_options();
    if one_native.is_empty() || two_native.is_empty() {
        stats.dead_end_refusals += 1;
        return Err(PyValueError::new_err("raw policy terminal: empty legal surface"));
    }
    let replacement_phase = state.side_one.force_switch || state.side_two.force_switch
        || state.side_one.get_active_immutable().hp <= 0 || state.side_two.get_active_immutable().hp <= 0;
    // Pinned Showdown Gen 3 Shadow Tag sets trapped=true, not hidden tryTrap.
    // This is an initial OWN request observation, never the ability/name or
    // an arbitrary environment legal mask. Arena Trap/Magnet Pull remain
    // unknown until an actual switch is rejected. Replacement/WAIT ignores it.
    let observations = [&state.side_two, &state.side_one].map(|other| {
        if !replacement_phase && other.get_active_immutable().ability
            == poke_engine::engine::abilities::Abilities::SHADOWTAG {
            PrivateTrapObservation::InitialRequest
        } else { PrivateTrapObservation::None }
    });
    let node = choice_node(state,
        choice_options(&state.side_one, replacement_phase, &one_native, observations[0]),
        choice_options(&state.side_two, replacement_phase, &two_native, observations[1]));
    certify_attempt_surface(&one_native, &node.s1_options)?;
    certify_attempt_surface(&two_native, &node.s2_options)?;
    let mut choices = [MoveChoice::None; 2];
    for seat in 0..2 {
        if expired(deadline) { return Ok(None); }
        stats.choice_attempts += 1;
        let index = bridges[seat].provide_with_trap(state, &node, Some(parent), ctx,
            observations[seat])?.certain_action()?;
        if expired(deadline) { return Ok(None); }
        choices[seat] = if seat == 0 { node.s1_options[index] } else { node.s2_options[index] };
    }
    for seat in 0..2 {
        let (side, other, native) = if seat == 0 {
            (&state.side_one, &state.side_two, &one_native)
        } else { (&state.side_two, &state.side_one, &two_native) };
        if native.contains(&choices[seat]) { continue; }
        if !matches!(choices[seat], MoveChoice::Switch(_))
            || side.trapped(&Pokemon::default())
            || !side.trapped(other.get_active_immutable()) {
            return Err(PyValueError::new_err("raw policy terminal: unproved switch rejection"));
        }
        stats.trapped_switch_rejections += 1;
        if expired(deadline) { return Ok(None); }
        let corrected = if seat == 0 {
            choice_node(state, one_native.clone(), node.s2_options.clone())
        } else { choice_node(state, node.s1_options.clone(), two_native.clone()) };
        stats.private_redecisions += 1;
        stats.choice_attempts += 1;
        let index = bridges[seat].provide_with_trap(state, &corrected,
            Some(parent), ctx, PrivateTrapObservation::RejectedSwitch)?.certain_action()?;
        if expired(deadline) { return Ok(None); }
        choices[seat] = native[index];
        if matches!(choices[seat], MoveChoice::Switch(_)) {
            return Err(PyValueError::new_err("raw policy terminal: repeated rejected switch"));
        }
    }
    Ok(Some((choices[0], choices[1])))
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

/// Diagnostic censoring is typed, never inferred by matching an exception.
/// The search adapter below retains its historical cap-refusal semantics.
#[derive(Debug, PartialEq)]
pub(crate) enum TerminalLabel {
    Terminal(f32),
    Capped,
    Deadline,
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
    match label_price(leaf, key, ordinal, ctx, meta, meta_ctx, cfg, deadline, stats, lossy)? {
        TerminalLabel::Terminal(value) => Ok(Some(value)),
        TerminalLabel::Deadline => Ok(None),
        TerminalLabel::Capped => Err(PyValueError::new_err(
            "raw policy terminal: nonterminal ply cap; no fallback value")),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn label_price(
    leaf: &State, key: (usize, usize), ordinal: u64, ctx: &EventContext,
    meta: &LeafMeta, meta_ctx: &LeafMetaCtx, cfg: &RawPolicyLeaf,
    deadline: Option<Instant>, stats: &mut RawPolicyLeafStats,
    lossy: &mut crate::abort_telemetry::LossySubcaseLedger,
) -> PyResult<TerminalLabel> {
    if expired(deadline) { return Ok(TerminalLabel::Deadline); }
    let bridges = [cfg.bridges[0].fork_at(key)?, cfg.bridges[1].fork_at(key)?];
    let mut state = leaf.clone();
    let mut ctx = ctx.clone();
    let mut meta = meta.clone();
    let mut parent = key;
    let mut rng = StdRng::seed_from_u64(leaf_seed(cfg.seed, ordinal));
    stats.started += 1;
    let result = (|| {
        for ply in 0..=cfg.max_plies {
            if expired(deadline) { return Ok(TerminalLabel::Deadline); }
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
                return Ok(TerminalLabel::Terminal(if over > 0. { 1. } else { 0. }));
            }
            if ply == cfg.max_plies {
                stats.cap_refusals += 1;
                return Ok(TerminalLabel::Capped);
            }
            let Some((s1, s2)) = select_choices(&state, parent, &ctx, &bridges, deadline, stats)?
                else { return Ok(TerminalLabel::Deadline); };
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
pub(crate) mod tests {
    use super::*;
    use poke_engine::choices::Choices;
    use poke_engine::engine::abilities::Abilities;
    use poke_engine::state::PokemonMoveIndex;
    use pyo3::ffi::c_str;
    use std::ffi::CString;
    use std::str::FromStr;
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub(crate) fn fixture(side_one_wins: bool, cap: u32) -> (State, EventContext, LeafMeta, RawPolicyLeaf) {
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
    fn diagnostic_caps_are_typed_while_search_cap_still_refuses() {
        let (state, ctx, meta, cfg) = fixture(true, 1);
        let mut stats = RawPolicyLeafStats::default();
        assert_eq!(label_price(&state, (0, 0), 0, &ctx, &meta, &LeafMetaCtx::default(),
            &cfg, None, &mut stats, &mut Default::default()).unwrap(), TerminalLabel::Capped);
        assert_eq!(stats.terminal, 0);
        assert_eq!(stats.plies, 1);
        assert_eq!(stats.cap_refusals, 1);
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

    fn choice_fixture(side_one: bool, ability: Abilities, switch: bool)
        -> (State, EventContext, RawPolicyLeaf, Py<PyAny>) {
        let (mut state, mut ctx, meta, _) = fixture(true, 2);
        let own = if side_one { &mut state.side_one } else { &mut state.side_two };
        own.pokemon.pkmn[1] = own.get_active_immutable().clone();
        own.pokemon.pkmn[1].id = FromStr::from_str("BULBASAUR").unwrap();
        ctx.species[usize::from(!side_one)].push("Bulbasaur".into());
        let other = if side_one { &mut state.side_two } else { &mut state.side_one };
        other.get_active().ability = ability;
        let module = Python::attach(|py| {
            // from_code registers its name in sys.modules. Parallel tests
            // must not re-execute one shared module or overwrite its globals.
            static NEXT_FIXTURE: AtomicUsize = AtomicUsize::new(0);
            let name = CString::new(format!("raw_choice_fixture_{}",
                NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed))).unwrap();
            let filename = CString::new(format!("{}.py", name.to_str().unwrap())).unwrap();
            let module = PyModule::from_code(py, c_str!(
                "import json, time\npayloads = []\nswitches = True\nsleep_on_redecision = False\ndef choose(raw):\n    p = json.loads(raw)\n    payloads.append(p)\n    r = p['native_request_bundle']\n    if sleep_on_redecision and r['request'].get('active', [{}])[0].get('trapped'):\n        time.sleep(.2)\n    indices = r['native_action_indices']\n    target = 4 if switches and 4 in indices else indices[0]\n    return [float(i == target) for i in indices]\n"),
                &filename, &name).unwrap();
            module.setattr("switches", switch).unwrap();
            module.unbind().into_any()
        });
        let bridge = |seat: bool| PolicyOpponentBridge::new(
            Python::attach(|py| module.getattr(py, "choose").unwrap()), seat,
            ctx.species[usize::from(!seat)].clone(),
            [("seismictoss".into(), 32), ("splash".into(), 64)].into(),
            [("seismictoss".into(), 20), ("splash".into(), 40)].into(),
            if seat { state.side_one.clone() } else { state.side_two.clone() }, ctx.clone());
        let cfg = RawPolicyLeaf { bridges: [bridge(true), bridge(false)],
            max_plies: 2, seed: 17, branch_on_damage: true };
        for bridge in &cfg.bridges { bridge.record(None, (0, 0), &[], &ctx, &meta).unwrap(); }
        (state, ctx, cfg, module)
    }

    fn payloads(module: &Py<PyAny>) -> Vec<serde_json::Value> {
        Python::attach(|py| {
            let text: String = py.import("json").unwrap().getattr("dumps").unwrap()
                .call1((module.getattr(py, "payloads").unwrap(),)).unwrap().extract().unwrap();
            serde_json::from_str(&text).unwrap()
        })
    }

    #[test]
    fn identical_initial_requests_then_private_trap_redecision_at_both_seats() {
        for side_one in [true, false] {
            let mut initial = Vec::new();
            for ability in [Abilities::SANDVEIL, Abilities::ARENATRAP] {
                let (state, ctx, cfg, module) = choice_fixture(side_one, ability, true);
                let before = state.serialize();
                let mut stats = RawPolicyLeafStats::default();
                let choices = select_choices(&state, (0, 0), &ctx, &cfg.bridges, None, &mut stats)
                    .unwrap().unwrap();
                let rows = payloads(&module);
                let seat = usize::from(!side_one);
                initial.push(rows[seat].clone());
                if ability != Abilities::SANDVEIL {
                    assert_eq!(rows.len(), 3);
                    assert_eq!(rows[2]["opponent_slot"], if side_one { "p1" } else { "p2" });
                    assert_eq!(rows[2]["native_request_bundle"]["request"]["active"][0]["trapped"], true);
                    assert_eq!(rows[2]["public_branch_lines"], rows[seat]["public_branch_lines"]);
                    assert_eq!(rows[2]["native_request_bundle"]["self_move_states"],
                        rows[seat]["native_request_bundle"]["self_move_states"]);
                    assert!(matches!(if side_one { choices.0 } else { choices.1 }, MoveChoice::Move(_)));
                    assert_eq!(stats.choice_attempts, 3);
                    assert_eq!(stats.trapped_switch_rejections, 1);
                    assert_eq!(stats.private_redecisions, 1);
                } else {
                    assert_eq!(rows.len(), 2);
                    assert!(matches!(if side_one { choices.0 } else { choices.1 }, MoveChoice::Switch(_)));
                    assert_eq!(stats.trapped_switch_rejections, 0);
                }
                assert_eq!(state.serialize(), before);
                assert_eq!(stats.plies, 0);
                assert_eq!(stats.terminal, 0);
                assert_eq!(ctx.turn, 1);
                assert!(rows.iter().all(|row| row["public_branch_lines"] == serde_json::json!([])));
            }
            assert!(initial.windows(2).all(|pair| pair[0] == pair[1]),
                "initial policy input leaked hidden trapping");
        }
    }

    #[test]
    fn terminal_price_charges_private_redecisions_without_mutating_frontier() {
        let (state, ctx, cfg, _) = choice_fixture(true, Abilities::ARENATRAP, true);
        let meta = LeafMeta { active: ["charmander".into(), "squirtle".into()],
            ..Default::default() };
        let before = state.serialize();
        let mut stats = RawPolicyLeafStats::default();
        let value = price(&state, (0, 0), 0, &ctx, &meta, &LeafMetaCtx::default(), &cfg,
            None, &mut stats, &mut Default::default()).unwrap();
        assert_eq!(value, Some(1.));
        assert_eq!(stats.terminal, 1);
        assert_eq!(stats.plies, 2);
        assert_eq!(stats.choice_attempts, 6);
        assert_eq!(stats.provider_calls, stats.choice_attempts);
        assert_eq!(stats.trapped_switch_rejections, 2);
        assert_eq!(stats.private_redecisions, 2);
        assert_eq!(state.serialize(), before);
        assert!(cfg.bridges.iter().all(|bridge| bridge.provider_calls.get() == 0));
    }

    #[test]
    fn unattempted_switch_does_not_reveal_hidden_trap_or_redispatch_opponent() {
        for side_one in [true, false] {
            let (state, ctx, cfg, module) = choice_fixture(side_one, Abilities::ARENATRAP, false);
            let mut stats = RawPolicyLeafStats::default();
            select_choices(&state, (0, 0), &ctx, &cfg.bridges, None, &mut stats).unwrap().unwrap();
            assert_eq!(payloads(&module).len(), 2);
            assert_eq!(stats.choice_attempts, 2);
            assert_eq!(stats.trapped_switch_rejections, 0);
            assert_eq!(stats.private_redecisions, 0);
            assert_eq!(cfg.bridges[0].provider_calls.get(), 1);
            assert_eq!(cfg.bridges[1].provider_calls.get(), 1);
        }
    }

    #[test]
    fn gen3_shadow_tag_is_an_initial_private_request_not_a_rejected_attempt() {
        for side_one in [true, false] {
            let (state, ctx, cfg, module) = choice_fixture(side_one, Abilities::SHADOWTAG, true);
            let before = state.serialize();
            let mut stats = RawPolicyLeafStats::default();
            let choices = select_choices(&state, (0, 0), &ctx, &cfg.bridges, None, &mut stats)
                .unwrap().unwrap();
            let rows = payloads(&module);
            let seat = usize::from(!side_one);
            assert_eq!(rows.len(), 2);
            assert_eq!(rows[seat]["native_request_bundle"]["request"]["active"][0]["trapped"], true);
            assert!(!rows[seat].to_string().contains("shadowtag"));
            assert!(matches!(if side_one { choices.0 } else { choices.1 }, MoveChoice::Move(_)));
            assert_eq!(stats.choice_attempts, 2);
            assert_eq!(stats.trapped_switch_rejections, 0);
            assert_eq!(stats.private_redecisions, 0);
            assert_eq!(state.serialize(), before);
            assert!(rows.iter().all(|row| row["public_branch_lines"] == serde_json::json!([])));
        }
    }

    #[test]
    fn magnet_pull_redecides_only_steel_and_arena_trap_respects_own_levitate() {
        use poke_engine::state::PokemonType;
        for side_one in [true, false] {
            for (ability, steel, levitate, rejects) in [
                (Abilities::MAGNETPULL, true, false, 1),
                (Abilities::MAGNETPULL, false, false, 0),
                (Abilities::ARENATRAP, false, true, 0),
            ] {
                let (mut state, ctx, cfg, module) = choice_fixture(side_one, ability, true);
                let own = if side_one { &mut state.side_one } else { &mut state.side_two };
                if steel { own.get_active().types = (PokemonType::STEEL, PokemonType::TYPELESS); }
                if levitate { own.get_active().ability = Abilities::LEVITATE; }
                let before = state.serialize();
                let mut stats = RawPolicyLeafStats::default();
                let choices = select_choices(&state, (0, 0), &ctx, &cfg.bridges, None, &mut stats)
                    .unwrap().unwrap();
                assert_eq!(stats.trapped_switch_rejections, rejects);
                assert_eq!(stats.private_redecisions, rejects);
                assert_eq!(stats.choice_attempts, 2 + rejects);
                assert_eq!(payloads(&module).len(), 2 + rejects);
                assert_eq!(matches!(if side_one { choices.0 } else { choices.1 }, MoveChoice::Switch(_)),
                    rejects == 0);
                assert_eq!(state.serialize(), before);
            }
        }
    }

    #[test]
    fn known_own_trapping_excludes_switches_without_attempt_or_redecision() {
        let (mut state, ctx, cfg, module) = choice_fixture(true, Abilities::ARENATRAP, true);
        state.side_one.volatile_statuses.insert(PokemonVolatileStatus::TRAPPED);
        let mut stats = RawPolicyLeafStats::default();
        select_choices(&state, (0, 0), &ctx, &cfg.bridges, None, &mut stats).unwrap().unwrap();
        let rows = payloads(&module);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["native_request_bundle"]["request"]["active"][0]["trapped"], true);
        assert_eq!(stats.private_redecisions, 0);
    }

    #[test]
    fn replacement_wait_surface_is_not_repaired_as_hidden_trapping() {
        for replacing_one in [true, false] {
            let (mut state, ctx, cfg, module) = choice_fixture(replacing_one, Abilities::SHADOWTAG, true);
            if replacing_one { state.side_one.force_switch = true; }
            else { state.side_two.force_switch = true; }
            let before = state.serialize();
            let mut stats = RawPolicyLeafStats::default();
            let choices = select_choices(&state, (0, 0), &ctx, &cfg.bridges, None, &mut stats)
                .unwrap().unwrap();
            let rows = payloads(&module);
            let replacing_seat = usize::from(!replacing_one);
            assert!(matches!(if replacing_one { choices.0 } else { choices.1 }, MoveChoice::Switch(_)));
            assert!(matches!(if replacing_one { choices.1 } else { choices.0 }, MoveChoice::None));
            assert_eq!(rows[replacing_seat]["native_request_bundle"]["request"]["forceSwitch"],
                serde_json::json!([true]));
            assert_eq!(rows[1 - replacing_seat]["native_request_bundle"]["request"]["wait"], true);
            assert_eq!(stats.choice_attempts, 2);
            assert_eq!(stats.trapped_switch_rejections, 0);
            assert_eq!(stats.private_redecisions, 0);
            assert_eq!(state.serialize(), before);
        }
    }

    #[test]
    fn simultaneous_hidden_traps_redecide_only_each_own_request() {
        let (mut state, mut ctx, mut cfg, module) = choice_fixture(true, Abilities::ARENATRAP, true);
        state.side_one.get_active().ability = Abilities::ARENATRAP;
        state.side_two.pokemon.pkmn[1] = state.side_two.get_active_immutable().clone();
        state.side_two.pokemon.pkmn[1].id = FromStr::from_str("BULBASAUR").unwrap();
        ctx.species[1].push("Bulbasaur".into());
        cfg.bridges[1] = PolicyOpponentBridge::new(
            Python::attach(|py| module.getattr(py, "choose").unwrap()), false,
            ctx.species[1].clone(),
            [("seismictoss".into(), 32), ("splash".into(), 64)].into(),
            [("seismictoss".into(), 20), ("splash".into(), 40)].into(),
            state.side_two.clone(), ctx.clone());
        let meta = LeafMeta { active: ["charmander".into(), "squirtle".into()], ..Default::default() };
        cfg.bridges[1].record(None, (0, 0), &[], &ctx, &meta).unwrap();
        let before = state.serialize();
        let mut stats = RawPolicyLeafStats::default();
        let choices = select_choices(&state, (0, 0), &ctx, &cfg.bridges, None, &mut stats)
            .unwrap().unwrap();
        let rows = payloads(&module);
        assert!(matches!(choices.0, MoveChoice::Move(_)));
        assert!(matches!(choices.1, MoveChoice::Move(_)));
        assert_eq!(rows.len(), 4);
        assert_eq!(stats.choice_attempts, 4);
        assert_eq!(stats.trapped_switch_rejections, 2);
        assert_eq!(stats.private_redecisions, 2);
        for seat in 0..2 {
            assert_eq!(rows[seat]["opponent_slot"], rows[seat + 2]["opponent_slot"]);
            assert_eq!(rows[seat + 2]["native_request_bundle"]["request"]["active"][0]["trapped"], true);
            assert_eq!(rows[seat]["public_branch_lines"], rows[seat + 2]["public_branch_lines"]);
            assert_eq!(rows[seat]["native_request_bundle"]["self_move_states"],
                rows[seat + 2]["native_request_bundle"]["self_move_states"]);
            assert_eq!(cfg.bridges[seat].provider_calls.get(), 2);
        }
        assert_eq!(state.serialize(), before);
        assert_eq!(stats.plies, 0);
        assert_eq!(stats.terminal, 0);
    }

    #[test]
    fn unrelated_native_move_loss_is_a_refusal_not_a_mask_repair() {
        use poke_engine::state::PokemonIndex;
        let mv = MoveChoice::Move(PokemonMoveIndex::M0);
        let sw = MoveChoice::Switch(PokemonIndex::P1);
        assert!(certify_attempt_surface(&[mv], &[mv, sw]).is_ok());
        for (native, attempted) in [(vec![mv], vec![sw]), (vec![sw], vec![mv]),
            (vec![mv, sw], vec![mv]), (vec![sw], vec![sw, mv]),
            (vec![mv], vec![mv, sw, sw])] {
            assert!(certify_attempt_surface(&native, &attempted).is_err());
        }
    }

    #[test]
    fn cancelled_private_redecision_has_no_environment_or_terminal_outcome() {
        let (state, ctx, cfg, module) = choice_fixture(true, Abilities::ARENATRAP, true);
        Python::attach(|py| module.setattr(py, "sleep_on_redecision", true).unwrap());
        let before = state.serialize();
        let mut stats = RawPolicyLeafStats::default();
        let result = select_choices(&state, (0, 0), &ctx, &cfg.bridges,
            Some(Instant::now() + std::time::Duration::from_millis(100)), &mut stats).unwrap();
        assert!(result.is_none());
        assert_eq!(state.serialize(), before);
        assert_eq!(stats.trapped_switch_rejections, 1);
        assert_eq!(stats.private_redecisions, 1);
        assert_eq!(stats.terminal, 0);
        assert_eq!(stats.plies, 0);
    }
}
