"""Pending policy conditioning never uses a live opponent commitment."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest import mock

from pokezero.local_showdown import PublicBattleMaterializationState
from pokezero.mcts_eval.paper_reference import DecisionState, ReferenceRefusal
from pokezero.mcts_eval.paper_reference_pending import (
    FaintReplacementTransition, PendingPolicyTransition, condition_pending_world,
    requires_faint_encore_replay, validate_transition,
)


class CertificateTests(unittest.TestCase):
    def setUp(self):
        def replay(lines, pending):
            return SimpleNamespace(requests={}, pending_baton_pass=pending,
                public_events=[SimpleNamespace(raw_line=line) for line in lines])
        self.before = PublicBattleMaterializationState(player_id='p1', format_id='gen3randombattle',
            observation_format_id='gen3randombattle', replay=replay(['|turn|1'], set()),
            belief_engine=SimpleNamespace(), self_request={'active': [{'moves': [
                {'id': 'thunderbolt'}, {'id': 'batonpass'}]}]})
        self.current = replace(self.before, self_request={'forceSwitch': [True]},
            replay=replay(['|turn|1', '|move|p1a: Jolteon|Baton Pass|p1a: Jolteon'], {'p1'}))
        self.observation = SimpleNamespace(metadata={}, legal_action_mask=(True, True, False, False))
        self.certificate = PendingPolicyTransition(self.before, self.observation, 1, 'sets')

    def validate(self, certificate=None):
        with mock.patch('pokezero.mcts_eval.paper_reference_pending.PublicObservation.from_observation') as public, \
             mock.patch('pokezero.mcts_eval.paper_reference_pending._public_belief_view', return_value={}), \
             mock.patch('pokezero.mcts_eval.paper_reference_pending.decision_state', return_value='exact-root'):
            public.return_value.to_observation.return_value = SimpleNamespace(metadata={})
            return validate_transition(certificate or self.certificate, self.current, self.observation, 'sets')

    def test_only_actor_known_baton_action_public_prefix_and_bound_source_certify(self):
        self.assertEqual(self.validate(), 'exact-root')
        for changed in (replace(self.certificate, own_action=0),
                        replace(self.certificate, own_action=True),
                        replace(self.certificate, set_source_hash='different'),
                        replace(self.certificate, before_state=replace(self.before, player_id='p2'))):
            with self.subTest(changed=changed), self.assertRaises(ReferenceRefusal):
                self.validate(changed)

    def test_private_request_and_nonpublic_metadata_are_rejected(self):
        replay = SimpleNamespace(**vars(self.before.replay))
        replay.requests = {'p2': {'private': 'canary'}}
        with self.assertRaisesRegex(ReferenceRefusal, 'source/player/request'):
            self.validate(replace(self.certificate, before_state=replace(self.before, replay=replay)))
        observation = SimpleNamespace(metadata={'private_canary': 'must-not-cross'}, legal_action_mask=(True, True))
        with self.assertRaisesRegex(ReferenceRefusal, 'nonpublic observation'):
            self.validate(replace(self.certificate, before_observation=observation))

    def test_missing_certificate_or_changed_public_turn_is_not_fabricated(self):
        with self.assertRaisesRegex(ReferenceRefusal, 'sampled-policy certificate'):
            validate_transition(None, self.current, self.observation, 'sets')
        self.current.replay.public_events.insert(1, SimpleNamespace(raw_line='|turn|2'))
        with self.assertRaisesRegex(ReferenceRefusal, 'exact public interrupted'):
            self.validate()


class ConditionalSamplingTests(unittest.TestCase):
    def fixture(self, matches):
        env = SimpleNamespace(requested=('p1', 'p2'), match=False, plays=[])
        env.requested_players = lambda: env.requested
        env.terminal = lambda: None
        env.observe = lambda player: SimpleNamespace(match=env.match)
        env.reseed_simulator_rng = lambda seed: None
        env.public_materialization_state = lambda player: SimpleNamespace(
            deferred_opponent_action_player='p2', match=env.match)
        def step(actions):
            env.plays.append(actions)
            env.requested = ('p1',)
        env.step = step
        root = DecisionState(b'current', ('action:4',), 0)
        certificate = SimpleNamespace(before_state='public-before', before_observation='public-observation', own_action=1)
        factory = SimpleNamespace(env=env, pending_transition=certificate, evaluator=lambda obs:
            ((0, 5), SimpleNamespace(priors=(.9, .1))), set_source='bound-source', known=('same-known-traits',),
            allow_earlier_compatible_template=True, max_known_set_draws=128,
            state=SimpleNamespace(player_id='p1', match=True), root=root,
            _release=lambda evidence: evidence.update(released=True))
        worlds, receipts = [], []
        class Prior:
            known = factory.known
            root = DecisionState(b'previous', ('action:1',), 0)
            def __init__(self):
                self.receipts = receipts
            def __call__(self, rng):
                env.requested = ('p1', 'p2')
                env.match = matches[len(worlds)]
                receipt = dict(ordinal=len(worlds), status='ROOT_VALIDATED', packed_team_sha256='draw')
                receipts.append(receipt)
                world = SimpleNamespace(closed=False, release=lambda: receipt.update(released=True))
                def close():
                    if not world.closed:
                        world.closed = True
                        world.release()
                world.close = close
                worlds.append(world)
                return world
        prior = Prior()
        class Rng:
            def choices(self, legal, *, weights, k):
                self.seen = (tuple(legal), tuple(weights), k)
                return [legal[1]]  # Lower-probability legal SWITCH, NEVER argmax.
            def getrandbits(self, count):
                return 42
        rng = Rng()
        def history(state):
            if state is factory.state:
                return ('exact-public-transition',)
            return ('exact-public-transition',) if state.match else ('different-transition',)
        def key(observation, *, player):
            return root if observation.match else DecisionState(b'different', ('action:4',), 0)
        patches = (mock.patch('pokezero.mcts_eval.paper_reference_factory.PublicRootWorldFactory', return_value=prior),
                   mock.patch('pokezero.mcts_eval.paper_reference_pending.public_history', side_effect=history),
                   mock.patch('pokezero.mcts_eval.paper_reference_pending.decision_state', side_effect=key))
        return factory, rng, prior, worlds, patches

    def sample(self, matches, cap=128):
        factory, rng, prior, worlds, patches = self.fixture(matches)
        evidence = {}
        with patches[0], patches[1], patches[2]:
            world = condition_pending_world(factory, rng, evidence, max_attempts=cap)
        return world, evidence, factory, rng, worlds

    def test_full_champion_distribution_is_sampled_not_argmax_or_live_action(self):
        world, evidence, factory, rng, worlds = self.sample([True])
        self.assertEqual(rng.seen, ((0, 5), (.9, .1), 1))
        self.assertEqual(factory.env.plays, [{'p1': 1, 'p2': 5}])
        self.assertEqual(evidence['pending_policy_conditioning']['sampled_action'], 5)
        self.assertFalse(evidence['pending_policy_conditioning']['live_opponent_action_used'])
        world.close()
        self.assertTrue(evidence['released'])

    def test_nonmatching_world_is_released_and_fresh_world_conditions_exact_root(self):
        world, evidence, factory, rng, worlds = self.sample([False, True])
        self.assertTrue(worlds[0].closed)
        self.assertFalse(worlds[1].closed)
        self.assertEqual(evidence['pending_policy_conditioning']['attempts'], 2)
        self.assertEqual(len(evidence['pending_policy_conditioning']['rejected']), 1)
        world.close()
        self.assertTrue(all(w.closed for w in worlds))

    def test_exhausted_conditioning_refuses_instead_of_fallback_or_redraw_of_position(self):
        factory, rng, prior, worlds, patches = self.fixture([False, False])
        with patches[0], patches[1], patches[2], self.assertRaisesRegex(ReferenceRefusal, 'explicit rejection cap'):
            condition_pending_world(factory, rng, {}, max_attempts=2)
        self.assertTrue(all(world.closed for world in worlds))

    def test_faint_replacement_retains_actual_queue_without_inventing_deferred_opponent(self):
        factory, rng, prior, worlds, patches = self.fixture([True])
        factory.pending_transition = FaintReplacementTransition('public-before', 'public-observation', 1, 'sets')
        factory.state.deferred_opponent_action_player = None
        factory.env.public_materialization_state = lambda player: SimpleNamespace(
            deferred_opponent_action_player=None, match=True)
        evidence = {}
        with patches[0], patches[1], patches[2]:
            world = condition_pending_world(factory, rng, evidence)
        self.assertEqual(evidence['pending_policy_conditioning']['transition_kind'],
                         'pre-upkeep-faint-replacement')
        self.assertFalse(evidence['pending_policy_conditioning']['live_opponent_action_used'])
        self.assertEqual(factory.env.plays, [{'p1': 1, 'p2': 5}])
        world.close()
        self.assertTrue(evidence['released'])


class FaintCertificateTests(unittest.TestCase):
    validate = CertificateTests.validate
    test_private_request_and_nonpublic_metadata_are_rejected = (
        CertificateTests.test_private_request_and_nonpublic_metadata_are_rejected)

    def setUp(self):
        CertificateTests.setUp(self)
        self.current = replace(self.current, replay=SimpleNamespace(requests={}, pending_baton_pass=set(),
            volatiles={'p2': ['encore']}, public_events=[SimpleNamespace(raw_line=line) for line in
                ['|turn|1', '|move|p1a: Aipom|Thunderbolt|p2a: Slowbro',
                 '|move|p2a: Slowbro|Surf|p1a: Aipom', '|faint|p1a: Aipom']]),
            self_request={'forceSwitch': [True], 'side': {'pokemon': [
                {'ident': 'p1: Aipom', 'condition': '0 fnt', 'active': True}]}})
        self.certificate = FaintReplacementTransition(self.before, self.observation, 0, 'sets')

    def test_public_faint_certifies_only_the_exact_pre_upkeep_replacement(self):
        self.assertTrue(requires_faint_encore_replay(self.current))
        self.assertEqual(self.validate(), 'exact-root')
        for changed in (replace(self.certificate, own_action=True),
                        replace(self.certificate, own_action=3),
                        replace(self.certificate, set_source_hash='different')):
            with self.subTest(changed=changed), self.assertRaises(ReferenceRefusal):
                self.validate(changed)

    def test_missing_faint_or_post_upkeep_cannot_fabricate_a_clock(self):
        for lines in (['|turn|1'], ['|turn|1', '|faint|p1a: Other'],
                      ['|turn|1', '|faint|p1a: Aipom', '|upkeep'],
                      ['|turn|2', '|faint|p1a: Aipom']):
            self.current.replay.public_events = [SimpleNamespace(raw_line=x) for x in lines]
            with self.subTest(lines=lines), self.assertRaisesRegex(ReferenceRefusal, 'pre-upkeep replacement'):
                self.validate()


if __name__ == '__main__':
    unittest.main()
