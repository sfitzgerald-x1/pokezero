"""Cooperative sampler expiry retains only completed work, never a fallback."""
import json
import time
import unittest
from unittest import mock

from pokezero.mcts_eval.paper_reference import (
    Evaluation, ReferenceConfig, ReferenceRefusal, SamplingDeadlineExceeded,
)
from pokezero.mcts_eval.paper_reference_factory import PublicRootWorldFactory
from pokezero.mcts_eval.paper_reference_parallel import ParallelTrajectorySearch, PreparedDecision
from pokezero.mcts_eval.paper_reference_substitute import condition_substitute_world
from test_paper_reference import ROOT, ToyWorld, searcher
from test_paper_reference_substitute import SubstituteSamplingTests


class BudgetRuntime:
    def __init__(self, index):
        self.index = index

    def prepare(self, both_cancel):
        deadline, draws = [None], []
        def sample(rng):
            cancelled = both_cancel or self.index == 0
            draws.append(dict(status='DEADLINE_CANCELLED' if cancelled else 'ROOT_VALIDATED'))
            if cancelled:
                while time.perf_counter() < deadline[0]:
                    time.sleep(.001)
                raise SamplingDeadlineExceeded('explicit test deadline')
            time.sleep(.003)
            return ToyWorld()
        def evidence():
            result = dict(draws=list(draws))
            draws.clear()
            return result
        return PreparedDecision(ROOT, lambda root: Evaluation((.5, .5), 0), sample,
                                evidence, lambda value: deadline.__setitem__(0, value))

    def close(self):
        pass


class SamplingDeadlineTests(unittest.TestCase):
    def test_completed_work_and_attempted_rng_ordinal_survive_clock_cancellation(self):
        search, now, worlds = searcher(), [0.], []
        def sample(rng):
            if worlds:
                now[0] = 1.
                raise SamplingDeadlineExceeded('expired')
            world = ToyWorld()
            worlds.append(world)
            return world
        result = search.search_batch(ROOT, battle_id='battle-1', seed=12,
            evaluate_root=lambda root: Evaluation((.5, .5), 0), sample_world=sample,
            trajectories=3, deadline_at=1., clock=lambda: now[0])
        self.assertEqual((result.trajectories, result.world_draws), (1, 2))
        self.assertEqual(search._ordinal, 2)
        self.assertTrue(result.deadline_exhausted)
        self.assertTrue(search._usable)
        self.assertTrue(worlds[0].closed)
        self.assertEqual(sum(search.nodes[ROOT.key].visits), 1)

    def test_premature_or_unbound_deadline_exception_is_still_an_integrity_refusal(self):
        for deadline in (None, 10.):
            search = searcher()
            def sample(rng):
                raise SamplingDeadlineExceeded('unproved expiry')
            with self.assertRaisesRegex(ReferenceRefusal, 'before the search deadline'):
                search.search_batch(ROOT, battle_id='battle-1', seed=12,
                    evaluate_root=lambda root: Evaluation((.5, .5), 0), sample_world=sample,
                    trajectories=1, deadline_at=deadline, clock=lambda: 0.)
            self.assertFalse(search._usable)

    def test_public_factory_cancel_is_json_safe_and_releases_ownership(self):
        factory = PublicRootWorldFactory.__new__(PublicRootWorldFactory)
        factory.active, factory.receipts, factory.sampling_deadline_at = False, [], None
        factory.bind_sampling_deadline(1.)
        with mock.patch('pokezero.mcts_eval.paper_reference_factory.time.perf_counter', return_value=1.), \
                self.assertRaises(SamplingDeadlineExceeded):
            factory(None)
        self.assertFalse(factory.active)
        row = json.loads(json.dumps(factory.receipts[-1]))
        self.assertEqual(row['status'], 'DEADLINE_CANCELLED')
        self.assertFalse(row['sampling_diagnostic']['accepted_world'])
        self.assertFalse(row['sampling_diagnostic']['backed_up'])

    def test_substitute_interruption_closes_its_unaccepted_world(self):
        factory, rng, worlds, patches = SubstituteSamplingTests().fixture([True])
        checks = [0]
        def check():
            checks[0] += 1
            if checks[0] == 2:
                raise SamplingDeadlineExceeded('fixture clock reached')
        factory.check_sampling_deadline = check
        evidence = {}
        with patches[0], patches[1], patches[2], self.assertRaises(SamplingDeadlineExceeded):
            condition_substitute_world(factory, rng, evidence)
        self.assertTrue(worlds[0].closed)
        self.assertNotIn('substitute_policy_conditioning', evidence)

    def test_parallel_can_select_only_when_another_worker_completed_new_search(self):
        with ParallelTrajectorySearch(ReferenceConfig(.5, 1), BudgetRuntime, workers=2) as pool:
            result = pool.search(False, ROOT, battle_id='budget', seed=1, deadline_seconds=.15)
            self.assertGreater(result.result.trajectories, 0)
            cancelled = [r for r in result.worker_receipts if r['worker'] == 0]
            self.assertEqual(sum(r['batch'].trajectories for r in cancelled), 0)
            self.assertEqual(sum(r['batch'].world_draws for r in cancelled), 1)
            self.assertEqual(cancelled[0]['evidence']['draws'][0]['status'], 'DEADLINE_CANCELLED')
        with ParallelTrajectorySearch(ReferenceConfig(.5, 1), BudgetRuntime, workers=2) as pool:
            with self.assertRaisesRegex(ReferenceRefusal, 'zero NEW complete trajectories'):
                pool.search(True, ROOT, battle_id='budget', seed=1, deadline_seconds=.15)


if __name__ == '__main__':
    unittest.main()
