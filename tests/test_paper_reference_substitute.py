"""Hidden Substitute HP is sampled only by public-history conditioning."""
from dataclasses import replace
import json
from types import SimpleNamespace
import unittest
from unittest import mock

from pokezero.mcts_eval.paper_reference import ReferenceRefusal
from pokezero.mcts_eval.paper_reference_substitute import (
    MAX_SUBSTITUTE_REJECTION_ATTEMPTS,
    SubstituteHistoryTransition, condition_substitute_world,
    requires_substitute_replay, validate_substitute_transition,
)
from tests.test_paper_reference_pending import CertificateTests, ConditionalSamplingTests


class SubstituteCertificateTests(unittest.TestCase):
    def setUp(self):
        old = CertificateTests()
        old.setUp()
        self.before, self.observation = old.before, old.observation
        self.before.replay.substitute_health_state = {'p1': 'full'}
        self.before.replay.volatiles = {'p1': ['substitute']}
        self.current = replace(self.before, replay=SimpleNamespace(requests={},
            substitute_health_state={'p1': 'unknown'}, volatiles={'p1': ['substitute']},
            public_events=[SimpleNamespace(raw_line=x) for x in (
                '|turn|1', '|move|p2a: Shuckle|Wrap|p1a: Lugia',
                '|-activate|p1a: Lugia|Substitute|[damage]', '|upkeep', '|turn|2')]))
        self.certificate = SubstituteHistoryTransition(self.before, self.observation, 1, 'sets')

    def validate(self, certificate=None):
        with mock.patch('pokezero.mcts_eval.paper_reference_substitute.PublicObservation.from_observation') as public, \
             mock.patch('pokezero.mcts_eval.paper_reference_substitute._public_belief_view', return_value={}), \
             mock.patch('pokezero.mcts_eval.paper_reference_substitute.decision_state', return_value='exact-root'):
            public.return_value.to_observation.return_value = SimpleNamespace(metadata={})
            return validate_substitute_transition(certificate or self.certificate,
                self.current, self.observation, 'sets')

    def test_unknown_surviving_substitute_requires_exact_public_damage_history(self):
        self.assertTrue(requires_substitute_replay(self.current))
        self.assertEqual(self.validate(), 'exact-root')
        self.current.replay.public_events.pop(2)
        with self.assertRaisesRegex(ReferenceRefusal, 'nonbreaking damage'):
            self.validate()

    def test_private_source_player_unknown_anchor_and_invalid_own_actions_refuse(self):
        private = SimpleNamespace(**vars(self.before.replay))
        private.requests = {'p2': {'canary': 'private'}}
        unknown = SimpleNamespace(**vars(self.before.replay))
        unknown.substitute_health_state = {'p1': 'unknown'}
        for changed in (replace(self.certificate, own_action=True),
                        replace(self.certificate, own_action=3),
                        replace(self.certificate, continuation_actions=(False,)),
                        replace(self.certificate, continuation_actions=(-1,)),
                        replace(self.certificate, set_source_hash='drift'),
                        replace(self.certificate, before_state=replace(self.before, player_id='p2')),
                        replace(self.certificate, before_state=replace(self.before, replay=private)),
                        replace(self.certificate, before_state=replace(self.before, replay=unknown))):
            with self.subTest(changed=changed), self.assertRaises(ReferenceRefusal):
                self.validate(changed)

    def test_nonpublic_observation_is_rejected(self):
        observation = SimpleNamespace(metadata={'private_canary': True}, legal_action_mask=(True, True))
        with self.assertRaisesRegex(ReferenceRefusal, 'nonpublic'):
            self.validate(replace(self.certificate, before_observation=observation))

    def test_history_records_only_own_actions_including_opponent_only_boundaries(self):
        chained = self.certificate.append(0).append(None)
        self.assertEqual(chained.continuation_actions, (0, None))
        self.assertEqual(self.validate(chained), 'exact-root')
        with self.assertRaises(ReferenceRefusal):
            chained.append({'p2': 3})


class SubstituteSamplingTests(unittest.TestCase):
    def fixture(self, matches):
        factory, rng, prior, worlds, _ = ConditionalSamplingTests().fixture(matches)
        factory.pending_transition = SubstituteHistoryTransition('public-before', 'public-observation', 1, 'sets')
        factory.state.deferred_opponent_action_player = None
        factory.env.observe = lambda player: SimpleNamespace(match=factory.env.match, legal_action_mask=(True,)*6)
        factory.env.public_materialization_state = lambda player: SimpleNamespace(
            deferred_opponent_action_player=None, match=factory.env.match)
        factory.env.snapshot = lambda: SimpleNamespace(bridge_snapshot={'battle': {'sides': [
            {'pokemon': [{'volatiles': {'substitute': {'hp': 55}}}]},
            {'pokemon': [{'volatiles': {}}]}]}})
        prior.known = ('different-before-known-traits',)  # New revealed moves are conditioned by the transition.
        def history(state):
            return ('observed',) if state is factory.state or state.match else ('different',)
        def key(observation, *, player):
            return factory.root if observation.match else None
        patches = (mock.patch('pokezero.mcts_eval.paper_reference_factory.PublicRootWorldFactory', return_value=prior),
                   mock.patch('pokezero.mcts_eval.paper_reference_substitute.public_history', side_effect=history),
                   mock.patch('pokezero.mcts_eval.paper_reference_substitute.decision_state', side_effect=key))
        return factory, rng, worlds, patches

    def test_public_conditioning_samples_lower_probability_switch_and_owns_world(self):
        factory, rng, worlds, patches = self.fixture([False, True])
        evidence = {}
        with patches[0], patches[1], patches[2]:
            world = condition_substitute_world(factory, rng, evidence)
        witness = evidence['substitute_policy_conditioning']
        self.assertEqual(witness['attempts'], 2)
        self.assertEqual(witness['sampled_substitute_hp'], {'p1': 55})
        self.assertEqual(witness['steps'][0]['opponent_action'], 5)
        self.assertEqual(witness['steps'][0]['opponent_priors'], [.9, .1])
        self.assertFalse(witness['live_opponent_action_used'])
        self.assertFalse(witness['live_hidden_hp_used'])
        self.assertTrue(worlds[0].closed)
        world.close()
        self.assertTrue(evidence['released'])

    def test_public_mismatch_exhaustion_refuses_and_releases_every_draw(self):
        factory, rng, worlds, patches = self.fixture([False, False])
        with patches[0], patches[1], patches[2], self.assertRaisesRegex(ReferenceRefusal, 'explicit rejection cap'):
            condition_substitute_world(factory, rng, {}, max_attempts=2)
        self.assertTrue(all(w.closed for w in worlds))

    def test_default_continues_original_proposals_after_attempt_128(self):
        factory, rng, worlds, patches = self.fixture([False] * 128 + [True])
        evidence = {}
        with patches[0], patches[1], patches[2]:
            world = condition_substitute_world(factory, rng, evidence)
        witness = evidence['substitute_policy_conditioning']
        self.assertEqual(witness['attempts'], 129)
        self.assertEqual(witness['max_attempts'], 2048)
        self.assertEqual([r['attempt'] for r in witness['rejected']], list(range(128)))
        self.assertTrue(all(w.closed for w in worlds[:-1]))
        self.assertEqual(witness['sampled_substitute_hp'], {'p1': 55})
        self.assertFalse(witness['live_opponent_action_used'])
        self.assertFalse(witness['live_hidden_hp_used'])
        world.close()
        self.assertTrue(evidence['released'])

    def test_early_accepted_draw_is_identical_except_declared_limit(self):
        witnesses = []
        for limit in (128, MAX_SUBSTITUTE_REJECTION_ATTEMPTS):
            factory, rng, worlds, patches = self.fixture([False, True])
            evidence = {}
            with patches[0], patches[1], patches[2]:
                world = condition_substitute_world(factory, rng, evidence, max_attempts=limit)
            world.close()
            witness = dict(evidence['substitute_policy_conditioning'])
            self.assertEqual(witness.pop('max_attempts'), limit)
            witnesses.append(witness)
        self.assertEqual(witnesses[0], witnesses[1])

    def test_extended_limit_still_refuses_and_releases_every_rejection(self):
        factory, rng, worlds, patches = self.fixture([False] * MAX_SUBSTITUTE_REJECTION_ATTEMPTS)
        with patches[0], patches[1], patches[2], self.assertRaisesRegex(ReferenceRefusal, 'explicit rejection cap'):
            condition_substitute_world(factory, rng, {})
        self.assertTrue(all(w.closed for w in worlds))

    def test_exhaustion_retains_json_safe_public_diagnostic_without_accepting_work(self):
        factory, rng, worlds, patches = self.fixture([False, False])
        evidence = {}
        with patches[0], patches[1], patches[2], self.assertRaises(ReferenceRefusal) as refused:
            condition_substitute_world(factory, rng, evidence, max_attempts=2)
        diagnostic = json.loads(json.dumps(refused.exception.sampling_diagnostic))
        self.assertEqual(diagnostic['attempts'], 2)
        self.assertEqual(diagnostic['max_attempts'], 2)
        self.assertEqual(diagnostic['accepted_worlds'], 0)
        self.assertEqual(diagnostic['rejection_reasons'], {'different public transition/request': 2})
        self.assertEqual(diagnostic['public_prefix_checks'], [{'matched_lines': 0, 'count': 2}])
        self.assertEqual(diagnostic['first_public_mismatches'], [dict(
            index=0, expected='observed', hypothetical='different', count=2)])
        self.assertTrue(diagnostic['prefix_counts_are_intermediate_checks'])
        self.assertFalse(diagnostic['live_opponent_action_used'])
        self.assertFalse(diagnostic['live_hidden_hp_used'])
        self.assertNotIn('substitute_policy_conditioning', evidence)
        self.assertTrue(all(world.closed for world in worlds))

    def test_invalid_limit_refuses_before_sampling(self):
        for limit in (True, 0, -1, 2.0, MAX_SUBSTITUTE_REJECTION_ATTEMPTS + 1):
            factory, rng, worlds, patches = self.fixture([True])
            with self.subTest(limit=limit), patches[0], patches[1], patches[2], \
                    self.assertRaisesRegex(ReferenceRefusal, 'invalid Substitute rejection limit'):
                condition_substitute_world(factory, rng, {}, max_attempts=limit)
            self.assertFalse(any(w.closed for w in worlds))


if __name__ == '__main__':
    unittest.main()
