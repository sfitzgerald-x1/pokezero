"""Excluded engineering fixtures for team-only, nondeployable diagnostics."""
from dataclasses import fields, replace
import pickle
import random
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from _showdown_root import requires_showdown, showdown_root
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
from pokezero.mcts_eval.paper_reference import Evaluation, ReferenceRefusal
from pokezero.mcts_eval.paper_reference_factory import PublicRootWorldFactory
from pokezero.mcts_eval.paper_reference_particle_party import condition_necessary_party
from pokezero.mcts_eval.paper_reference_runtime import PublicRootRequest, ShowdownWorkerFactory, _ShowdownRuntime
from pokezero.mcts_eval.paper_reference_sampling import KnownSetTraits
from pokezero.mcts_eval.search_over_raw import SearchConfiguration
from pokezero.mcts_eval.search_over_raw_oracle import (DiagnosticTeamOracleSearchAdapter,
    OracleRootRequest, OracleTeamSampler, OracleWorkerFactory, TeamOracle, _OracleRuntime,
    opening_team, public_root_binding, team_sha256, validate_oracle_work)
from pokezero.policy import PolicyContext, PolicyDecision
from pokezero.randbat import load_gen3_randbat_source_cached
from pokezero.showdown_fixture import pack_team
from pokezero.trajectory import BattleTrajectory
from tests.test_search_over_raw_adapters import work


class CountingEvaluator:
    def __init__(self):
        self.forwards = 0

    def __call__(self, observation):
        self.forwards += 1
        legal = tuple(i for i, enabled in enumerate(observation.legal_action_mask) if enabled)
        return legal, Evaluation((1/len(legal),)*len(legal), 0.)


@requires_showdown()
class TeamOracleTests(unittest.TestCase):
    def setUp(self):
        config = LocalShowdownConfig(showdown_root=showdown_root(), set_belief_source=True)
        self.live, self.sampled = LocalShowdownEnv(config), LocalShowdownEnv(config)
        self.addCleanup(self.live.close)
        self.addCleanup(self.sampled.close)
        self.live.reset(seed=2026101009)
        self.snapshot = self.live.snapshot_actionable_boundary()
        self.source = load_gen3_randbat_source_cached(showdown_root())
        self.public = self.live.public_materialization_state('p1')
        self.observation = self.live.observe('p1')
        self.request = PublicRootRequest.capture(self.public, self.observation)
        self.oracle = TeamOracle.capture(self.snapshot, self.request, self.source)
        self.contract = SimpleNamespace(checkpoint_path='weights', checkpoint_sha256='a'*64,
            showdown_source_sha256=self.source.metadata.source_hash)
        self.factory_binding = ShowdownWorkerFactory('weights', 'a'*64, str(showdown_root()),
            self.source.metadata.source_hash)

    def factory(self):
        return PublicRootWorldFactory(env=self.sampled, state=self.public,
            observation=self.observation, evaluator=CountingEvaluator(), set_source=self.source)

    def context(self):
        return PolicyContext('p1', 0, 'excluded:oracle', 'gen3randombattle', 2026101009,
            self.observation, tuple(self.live.requested_players()),
            BattleTrajectory('excluded:oracle', 'gen3randombattle', 2026101009),
            requested_observations={p: self.live.observe(p) for p in self.live.requested_players()},
            requested_legal_action_masks={p: self.live.observe(p).legal_action_mask
                for p in self.live.requested_players()}, public_materialization_state=self.public)

    def reference_adapter(self, leaf='model'):
        adapter = DiagnosticTeamOracleSearchAdapter(SearchConfiguration('reference', 'oracle', leaf,
            workers=20), oracle=self.oracle, checkpoint_contract=self.contract,
            showdown_root=str(showdown_root()), reference_factory=self.factory_binding)
        self.addCleanup(adapter.close)
        return adapter

    def test_capture_is_original_team_only_and_pickle_contains_no_snapshot(self):
        self.oracle.validate(self.request, self.source)
        self.assertEqual(self.oracle.opponent_team, opening_team(self.snapshot.first_requests['p2'], 'p2', self.source))
        self.assertEqual({f.name for f in fields(self.oracle)}, {'subject', 'set_source_hash',
            'root_binding', 'own_team_sha256', 'opponent_team', 'opponent_team_sha256'})
        wire = pickle.dumps(OracleRootRequest(self.request, self.oracle))
        self.assertNotIn(b'LocalShowdownSnapshot', wire)
        self.assertNotIn(b'bridge_snapshot', wire)
        restored = pickle.loads(wire)
        restored.oracle.validate(restored.public, self.source)
        self.assertFalse(restored.public.state.replay.requests)
        self.assertIsNone(restored.public.state.belief_engine.set_source)
        self.assertFalse(self.oracle.receipt()['deployable'])

    def test_capture_binds_public_history_not_only_cached_request(self):
        bad = replace(self.snapshot, replay=replace(self.snapshot.replay, public_events=()))
        with self.assertRaisesRegex(ValueError, 'does not match'):
            TeamOracle.capture(bad, self.request, self.source)

    def test_oracle_root_source_and_team_drift_refuse(self):
        for changed in (replace(self.oracle, root_binding='wrong'),
                replace(self.oracle, own_team_sha256='wrong'),
                replace(self.oracle, opponent_team_sha256='wrong'),
                replace(self.oracle, set_source_hash='wrong')):
            with self.assertRaisesRegex(ValueError, 'binding changed'):
                changed.validate(self.request, self.source)
        self.assertEqual(public_root_binding(self.request), self.oracle.root_binding)

    def test_second_seat_capture_has_same_original_teams_reversed(self):
        request = PublicRootRequest.capture(self.live.public_materialization_state('p2'), self.live.observe('p2'))
        oracle = TeamOracle.capture(self.snapshot, request, self.source)
        self.assertEqual(oracle.opponent_team_sha256, self.oracle.own_team_sha256)
        self.assertEqual(oracle.own_team_sha256, self.oracle.opponent_team_sha256)
        self.assertNotEqual(oracle.root_binding, self.oracle.root_binding)

    def test_fixed_team_does_not_consume_hidden_rng_or_generate_unknown_party(self):
        factory = self.factory()
        sampler = OracleTeamSampler(self.oracle.opponent_team, self.source)
        rng = random.Random(7)
        before = rng.getstate()
        draw = sampler.draw(factory.known, rng)
        self.assertEqual(rng.getstate(), before)
        self.assertEqual(draw.team, self.oracle.opponent_team)
        self.assertEqual(draw.packed_team_sha256, self.oracle.opponent_team_sha256)
        self.assertFalse(draw.unknown_party_seeds)
        self.assertTrue(all(not row.seeds and not row.forced for row in draw.known))

    def test_original_public_trait_contradictions_and_exclusions_refuse(self):
        mon = self.oracle.opponent_team[0]
        sampler = OracleTeamSampler(self.oracle.opponent_team, self.source)
        incompatible = [KnownSetTraits(mon.species, level=max(1, mon.level-1)),
            KnownSetTraits(mon.species, moves=('invalidmove',))]
        if mon.ability:
            incompatible.append(KnownSetTraits(mon.species, ruled_out_abilities=(mon.ability,)))
        if mon.item:
            incompatible.append(KnownSetTraits(mon.species, ruled_out_items=(mon.item,)))
        for traits in incompatible:
            with self.assertRaisesRegex(ReferenceRefusal, 'contradicts public'):
                sampler.draw((traits,), random.Random(1))

    def test_mutated_or_incomplete_original_party_refuses(self):
        with self.assertRaisesRegex(ValueError, 'complete original party'):
            OracleTeamSampler(self.oracle.opponent_team[:5], self.source)
        sampler = OracleTeamSampler(self.oracle.opponent_team, self.source)
        sampler.team = (replace(sampler.team[0], level=1), *sampler.team[1:])
        with self.assertRaisesRegex(ValueError, 'mutated'):
            sampler.draw((), random.Random(1))

    def test_real_public_materialization_uses_truth_and_keeps_source_unchanged(self):
        factory = self.factory()
        factory.install_diagnostic_oracle(self.oracle.opponent_team)
        self.addCleanup(factory.close)
        world = factory(random.Random(1))
        world.close()
        receipt = factory.receipts[0]
        self.assertEqual(receipt['packed_team_sha256'], self.oracle.opponent_team_sha256)
        self.assertEqual(receipt['diagnostic_oracle_team_sha256'], self.oracle.opponent_team_sha256)
        self.assertEqual(receipt['status'], 'ROOT_VALIDATED')
        self.assertTrue(receipt['released'])
        self.assertEqual(public_root_binding(PublicRootRequest.capture(
            self.live.public_materialization_state('p1'), self.live.observe('p1'))), self.oracle.root_binding)
        with self.assertRaisesRegex(ReferenceRefusal, 'cannot replace'):
            factory.install_diagnostic_oracle(self.oracle.opponent_team)

    def test_membership_first_keeps_truth_and_refuses_impossible_public_species(self):
        factory = self.factory()
        factory.install_diagnostic_oracle(self.oracle.opponent_team)
        factory.pending_transition = None
        mon = self.oracle.opponent_team[0]
        history = [f'|switch|p2a: fixture|{mon.species}, L{mon.level}|100/100']
        receipt = condition_necessary_party(factory, history, 'p2', lambda: None,
            membership_first=True, native_membership_batch_size=8)
        rng = random.Random(4)
        before = rng.getstate()
        self.assertEqual(factory.sampler.draw((), rng).team, self.oracle.opponent_team)
        self.assertEqual(rng.getstate(), before)
        self.assertEqual(receipt['status'], 'DIAGNOSTIC_ORACLE_MEMBERSHIP_CHECK')
        self.assertFalse(receipt['unknown_party_sampled'])
        absent = next(p for p in self.source.universes if p not in {m.species.lower() for m in self.oracle.opponent_team})
        other = self.factory()
        other.install_diagnostic_oracle(self.oracle.opponent_team)
        condition_necessary_party(other, [f'|switch|p2a: fixture|{absent}|100/100'], 'p2', lambda: None,
            membership_first=True)
        with self.assertRaisesRegex(ReferenceRefusal, 'contradicts public membership'):
            other.sampler.draw((), rng)

    def test_actual_runtime_prepares_public_world_then_installs_team_only_payload(self):
        base = object.__new__(_ShowdownRuntime)
        base.source, base.env, base.evaluator = self.source, self.sampled, CountingEvaluator()
        base.allow_earlier_compatible_template, base.max_known_set_draws = False, 10
        runtime = _OracleRuntime(base)
        self.addCleanup(runtime.close)
        prepared = runtime.prepare(OracleRootRequest(self.request, self.oracle))
        world = prepared.sample_world(random.Random(1))
        world.close()
        evidence = prepared.evidence()
        self.assertEqual(evidence['team_oracle'], self.oracle.receipt())
        self.assertEqual(evidence['draws'][0]['packed_team_sha256'], self.oracle.opponent_team_sha256)
        with self.assertRaisesRegex(ValueError, 'explicit diagnostic transport'):
            runtime.prepare(self.request)

    def test_reference_oracle_composes_with_both_leaves_without_changing_tree(self):
        from pokezero.mcts_eval.search_over_raw_leaves import ReferenceLeafWorkerFactory
        identities = set()
        for leaf in ('model', 'hp_fraction', 'raw_rollout'):
            with patch('pokezero.mcts_eval.paper_reference_parallel.ParallelTrajectorySearch') as pool:
                adapter = self.reference_adapter(leaf)
                wrapped = pool.call_args.args[1]
                if leaf != 'model':
                    self.assertIsInstance(wrapped, ReferenceLeafWorkerFactory)
                    self.assertEqual(wrapped.leaf, leaf)
                    wrapped = wrapped.base
                self.assertIsInstance(wrapped, OracleWorkerFactory)
                self.assertIs(wrapped.base, self.factory_binding)
                self.assertFalse(adapter.configuration.deployable)
                identities.add(adapter.runtime_sha256)
        self.assertEqual(len(identities), 3)

    def test_oracle_worker_receipts_cannot_label_sampled_fallback_as_truth(self):
        measured = work()
        evidence = measured.worker_receipts[0]['evidence']
        evidence['team_oracle'] = self.oracle.receipt()
        with self.assertRaisesRegex(ValueError, 'lost truth'):
            validate_oracle_work(measured, self.oracle)
        evidence['draws'][0].update(diagnostic_oracle_team_sha256=self.oracle.opponent_team_sha256,
            information_scope='original_opponent_team_only')
        validate_oracle_work(measured, self.oracle)
        evidence['team_oracle'] = {**self.oracle.receipt(), 'deployable': True}
        with self.assertRaisesRegex(ValueError, 'binding drift'):
            validate_oracle_work(measured, self.oracle)

    def test_reference_missing_oracle_evidence_poisoned_without_retry(self):
        with patch('pokezero.mcts_eval.paper_reference_parallel.ParallelTrajectorySearch') as pool:
            adapter = self.reference_adapter()
            pool.return_value.search.return_value = work()
            with self.assertRaisesRegex(ValueError, 'binding drift'):
                adapter.select(self.context(), root_id='excluded:missing', selection_seed=1)
            sent = pool.return_value.search.call_args.args[0]
            self.assertIsInstance(sent, OracleRootRequest)
            self.assertFalse(sent.public.state.replay.requests)
            self.assertTrue(adapter._poisoned)
            with self.assertRaisesRegex(ValueError, 'poisoned'):
                adapter.select(self.context(), root_id='excluded:retry', selection_seed=1)

    def test_incumbent_override_is_team_only_and_public_context_stays_private_free(self):
        legal = next(i for i, enabled in enumerate(self.observation.legal_action_mask) if enabled)
        native = SimpleNamespace(_fixed_override=None, reset=Mock(),
            select_action_with_context=Mock(return_value=PolicyDecision(legal, 'excluded:oracle')))
        decider = Mock()
        decider._snapshot_stats.return_value = dict(depth_reached_histogram={}, fallback_decisions=0,
            prior_fallbacks=0, total_iterations=0, model_evals=0)
        decider._changed_depth.return_value = 0
        with patch('pokezero.mcts_eval.search_over_raw_adapters._incumbent_runtime',
                return_value=(decider, native, 'config')), \
                patch('pokezero.mcts_eval.policy_opponent_profile.validate_selection'):
            adapter = DiagnosticTeamOracleSearchAdapter(SearchConfiguration('incumbent', 'oracle'),
                oracle=self.oracle, checkpoint_contract=self.contract, showdown_root=str(showdown_root()))
            self.addCleanup(adapter.close)
            row = adapter.select(self.context(), root_id='excluded:inc', selection_seed=2)
            seen = native.select_action_with_context.call_args.args[0]
            self.assertEqual(set(seen.requested_observations), {'p1'})
            self.assertEqual(native._fixed_override.player_teams['p2'], pack_team(self.oracle.opponent_team))
            self.assertEqual(team_sha256(opening_team(self.request.state.self_initial_request, 'p1',
                self.source)), self.oracle.own_team_sha256)
            self.assertEqual(row['evidence']['team_oracle'], self.oracle.receipt())
            self.assertFalse(adapter.configuration.deployable)


if __name__ == '__main__':
    unittest.main()
