import random
from dataclasses import replace
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch, Mock

from _showdown_root import requires_showdown, showdown_root
from pokezero.env import BattleStartOverride
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv, LocalShowdownError
from pokezero.observation import ObservationFeatureMasks
from pokezero.showdown import V2_1_REPLAY_OBSERVATION_SPEC
from pokezero.showdown_fixture import FixturePokemon, pack_team
from pokezero.mcts_eval.paper_reference_pending import public_history

from pokezero.mcts_eval.paper_reference_stage_inputs import build_stage_inputs
from pokezero.mcts_eval.paper_reference_stage_inputs import HistoricalStageInputs
from pokezero.mcts_eval.paper_reference_stage_memo import StageIdentityMemo


class FakeEnv:
    def __init__(self):
        self.restores = 0
        self.observations = []
        self.history = ("|turn|1",)
        self.requested = ("p1", "p2")
        self.own_mask = (True, False, True)

    def restore_search_snapshot(self, snapshot):
        self.restores += 1
        self.history = snapshot.history

    def requested_players(self):
        return self.requested

    def observe(self, player):
        self.observations.append(player)
        return S(legal_action_mask=self.own_mask, history=self.history, player=player)

    def public_materialization_state(self, player):
        return self.history


class HistoricalStageInputsTest(unittest.TestCase):
    def setUp(self):
        self.env = FakeEnv()
        self.particle = S(snapshot=S(history=("|turn|1",)))
        self.evaluations = 0
        self.memo = StageIdentityMemo({})

    def build(self, own=0, need_history=True):
        def evaluate(observation):
            self.evaluations += 1
            return [0, 2], S(priors=[.2, .8])
        with patch("pokezero.mcts_eval.paper_reference_stage_inputs.public_history", lambda value: value):
            return build_stage_inputs(env=self.env, particle=self.particle,
                subject="p1", opponent="p2", own=own,
                evaluator=evaluate, need_history=need_history)

    def test_same_particle_multiplicity_skips_restore_and_observe_not_rng(self):
        rng = random.Random(9)
        expected = random.Random(9)
        one = self.memo.get(self.particle, 0, self.build)
        self.env.history = ("hypothetical previous chance trial",)
        two = self.memo.get(self.particle, 0, self.build)
        self.assertIs(one, two)
        self.assertEqual(two.history, ("|turn|1",))
        self.assertEqual(self.env.restores, 1)
        self.assertEqual(self.env.observations, ["p1", "p2"])
        self.assertEqual(self.evaluations, 1)
        # Two multiplicities still receive two fresh actions and chance seeds.
        for inputs in (one, two):
            self.assertEqual(rng.choices(inputs.legal, weights=inputs.probabilities),
                             expected.choices([0, 2], weights=[.2, .8]))
            self.assertEqual(rng.getrandbits(64), expected.getrandbits(64))
        self.assertEqual(rng.getstate(), expected.getstate())

    def test_later_stage_does_not_reuse_previous_inputs(self):
        self.memo.get(self.particle, 0, self.build)
        self.memo.get(self.particle, 1, self.build)
        self.assertEqual(self.env.restores, 2)
        self.assertEqual(self.evaluations, 2)

    def test_invalid_own_action_stops_before_policy_or_history(self):
        self.assertIsNone(self.build(own=1))
        self.assertEqual(self.evaluations, 0)
        self.assertEqual(self.env.observations, ["p1"])

    def test_one_sided_request_has_no_opponent_inference(self):
        self.env.requested = ("p1",)
        inputs = self.build()
        self.assertEqual(inputs.requested, frozenset(["p1"]))
        self.assertEqual(inputs.legal, ())
        self.assertIsNone(inputs.opponent_observation)
        self.assertEqual(self.evaluations, 0)

    def test_opponent_only_request_has_no_own_observation(self):
        self.env.requested = ("p2",)
        inputs = self.build(own=None)
        self.assertEqual(self.env.observations, ["p2"])
        self.assertEqual(inputs.probabilities, (.2, .8))

    def test_wrong_request_role_is_zero_weight_without_inference(self):
        self.env.requested = ("p2",)
        self.assertIsNone(self.build())
        self.assertEqual(self.evaluations, 0)

    def test_projection_error_propagates_and_is_not_cached(self):
        self.env.restore_search_snapshot = lambda snapshot: (_ for _ in ()).throw(RuntimeError("native restore"))
        with self.assertRaisesRegex(RuntimeError, "native restore"):
            self.memo.get(self.particle, 0, self.build)
        self.assertEqual(self.memo.entries, {})

    def test_priors_are_detached_from_mutable_evaluator_list(self):
        inputs = self.build()
        self.assertIsInstance(inputs.legal, tuple)
        self.assertIsInstance(inputs.probabilities, tuple)


def original_inputs(env, particle, subject, opponent, own, evaluator, need_history):
    env.restore_search_snapshot(particle.snapshot)
    requested = frozenset(env.requested_players())
    if (subject in requested) != (own is not None): return None
    if own is not None:
        mask = env.observe(subject).legal_action_mask
        if not 0 <= own < len(mask) or not mask[own]: return None
    observation, legal, probabilities = None, (), ()
    if opponent in requested:
        observation = env.observe(opponent)
        legal, evaluation = evaluator(observation)
        legal, probabilities = tuple(legal), tuple(evaluation.priors)
    history = public_history(env.public_materialization_state(subject)) if need_history else ()
    return HistoricalStageInputs(requested, observation, legal, probabilities, history)


@requires_showdown()
class PairedHistoricalInputsTest(unittest.TestCase):
    def config(self, narrow=False):
        return LocalShowdownConfig(showdown_root=showdown_root(),
            observation_spec=V2_1_REPLAY_OBSERVATION_SPEC, set_belief_source=True,
            feature_masks=ObservationFeatureMasks(tier2_investment=True, investment_belief_narrowing=narrow))

    def override(self):
        return BattleStartOverride(observation_format_id='gen3randombattle', player_teams={
            'p1': pack_team((FixturePokemon('Scizor', ('Baton Pass', 'Double-Edge'), ability='Swarm', level=76),
                            FixturePokemon('Tauros', ('Double-Edge', 'Earthquake'), ability='Intimidate', level=76))),
            'p2': pack_team((FixturePokemon('Shuckle', ('Splash', 'Rock Slide'), ability='Sturdy', level=80),))})

    def evaluator(self, observation):
        legal = tuple(i for i, value in enumerate(observation.legal_action_mask) if value)
        return legal, S(priors=tuple(1/len(legal) for _ in legal))

    def test_owned_snapshots_match_full_path_for_both_roles_and_forced_baton(self):
        for narrow in (False, True):
            with LocalShowdownEnv(self.config(narrow)) as env:
                env.reset_with_start_override(seed=17, start_override=self.override())
                for force in (False, True):
                    if force:
                        env.step({'p1': 0, 'p2': 0})
                        self.assertEqual(env.requested_players(), ('p1',))
                    snapshot = env.snapshot_for_search()
                    for subject, opponent in (('p1','p2'), ('p2','p1')):
                        own = min(snapshot.search_choice_cache[subject]) if subject in env.requested_players() else None
                        for need_history in (False, True):
                            args = dict(particle=S(snapshot=snapshot), subject=subject,
                                        opponent=opponent, own=own, evaluator=self.evaluator, need_history=need_history)
                            expected = original_inputs(env, **args)
                            with patch.object(env, 'observe', side_effect=AssertionError('mutable shell observation')), \
                                 patch.object(env, 'restore_search_snapshot', side_effect=AssertionError('redundant native restore')), \
                                 patch.object(env, 'observe_search_snapshot', wraps=env.observe_search_snapshot) as observe, \
                                 patch.object(env, 'public_materialization_state', side_effect=AssertionError('unused full projection')):
                                actual = build_stage_inputs(env=env, **args)
                            self.assertEqual(actual, expected)
                            self.assertEqual([c.args[1] for c in observe.call_args_list],
                                             [opponent] if opponent in actual.requested else [])
                    env.restore_search_snapshot(snapshot)
                    env.release_search_snapshot(snapshot)

    def test_cached_legality_preserves_invalid_and_float_error_paths(self):
        with LocalShowdownEnv(self.config()) as env:
            env.reset_with_start_override(seed=17, start_override=self.override())
            snapshot = env.snapshot_for_search()
            args = dict(particle=S(snapshot=snapshot), subject='p1', opponent='p2',
                        evaluator=self.evaluator, need_history=True)
            for own in (-1, 8, 100):
                self.assertIsNone(build_stage_inputs(env=env, own=own, **args))
                self.assertIsNone(original_inputs(env, own=own, **args))
            with self.assertRaises(TypeError): build_stage_inputs(env=env, own=0., **args)
            with self.assertRaises(TypeError): original_inputs(env, own=0., **args)
            env.release_search_snapshot(snapshot)

    def test_missing_cache_retains_original_observation_path(self):
        with LocalShowdownEnv(self.config()) as env:
            env.reset_with_start_override(seed=17, start_override=self.override())
            resident = env.snapshot_for_search()
            snapshot = replace(resident, search_choice_cache={})
            with patch.object(env, 'observe', wraps=env.observe) as observe, \
                 patch.object(env, 'public_materialization_state', wraps=env.public_materialization_state) as public:
                result = build_stage_inputs(env=env, particle=S(snapshot=snapshot), subject='p1',
                    opponent='p2', own=0, evaluator=self.evaluator, need_history=True)
            self.assertEqual([c.args[0] for c in observe.call_args_list], ['p1','p2'])
            public.assert_called_once_with('p1')
            self.assertTrue(result.history)
            env.release_search_snapshot(resident)

    def test_isolated_projection_leaves_different_native_and_python_shell_unchanged(self):
        for narrow in (False, True):
            with LocalShowdownEnv(self.config(narrow)) as env:
                env.reset_with_start_override(seed=17, start_override=self.override())
                resident = env.snapshot_for_search()
                expected = {side: env.observe(side) for side in env.requested_players()}
                actions = {'p1': 1, 'p2': 0}
                first = env.step_from_search_snapshot_for_conditioning(resident, actions, chance_seed=57)
                current_native = env.snapshot().bridge_snapshot
                current_lines = tuple(env._lines)
                current_requests = env._latest_requests.copy()
                current_belief = env._belief_engine.snapshot()
                current_parser, current_engine = env._parser, env._belief_engine
                current_trackers = dict(env._tier2_trackers)
                captured = []
                original_observe = LocalShowdownEnv._observe
                def observe(projection, player, **kwargs):
                    captured.append(projection)
                    return original_observe(projection, player, **kwargs)
                with patch.object(LocalShowdownEnv, '_bridge_request_event',
                                  side_effect=AssertionError('projection made native RPC')), \
                     patch.object(LocalShowdownEnv, '_observe', observe):
                    actual = {side: env.observe_search_snapshot(resident, side) for side in expected}
                self.assertEqual(actual, expected)
                self.assertEqual(env.snapshot().bridge_snapshot, current_native)
                self.assertEqual(tuple(env._lines), current_lines)
                self.assertEqual(env._latest_requests, current_requests)
                self.assertEqual(env._belief_engine.snapshot(), current_belief)
                self.assertIs(env._parser, current_parser)
                self.assertIs(env._belief_engine, current_engine)
                for side, tracker in current_trackers.items():
                    self.assertIs(env._tier2_trackers[side], tracker)
                for projection in captured:
                    self.assertIsNot(projection, env)
                    self.assertIsNot(projection._parser, resident.replay)
                    self.assertIsNot(projection._belief_engine, resident.belief_engine)
                    self.assertIsNone(projection._process)
                    for side, tracker in resident.annotation_cache.tier2_trackers.items():
                        self.assertIsNot(projection._tier2_trackers[side], tracker)
                    projection.close()  # never owns the real bridge
                second = env.step_from_search_snapshot_for_conditioning(resident, actions, chance_seed=57)
                self.assertEqual(first, second)
                self.assertEqual(env.snapshot().bridge_snapshot, current_native)
                env.release_search_snapshot(resident)

    def test_projection_preserves_sampled_world_format_and_request_guards(self):
        with LocalShowdownEnv(self.config()) as env:
            env.reset(seed=17)
            generic = env.snapshot()
            with self.assertRaisesRegex(LocalShowdownError, 'sampled search world'):
                env.observe_search_snapshot(generic, 'p2')
            env.reset_with_start_override(seed=17, start_override=self.override())
            generic = env.snapshot()
            resident = env.snapshot_for_search()
            with self.assertRaisesRegex(ValueError, 'resident search handle'):
                env.observe_search_snapshot(generic, 'p1')
            with self.assertRaisesRegex(ValueError, 'format differs'):
                env.observe_search_snapshot(replace(resident, format_id='gen4randombattle'), 'p1')
            with self.assertRaisesRegex(ValueError, 'no retained actionable request'):
                env.observe_search_snapshot(resident, 'p3')
            env.step_from_search_snapshot_for_conditioning(resident, {'p1': 0, 'p2': 0}, chance_seed=3)
            forced = env.snapshot_for_search()
            with self.assertRaisesRegex(ValueError, 'no retained actionable request'):
                env.observe_search_snapshot(forced, 'p2')
            env.release_search_snapshot(forced)
            env.release_search_snapshot(resident)


if __name__ == "__main__":
    unittest.main()
