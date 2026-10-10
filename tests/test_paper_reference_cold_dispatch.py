"""Real spawned toy workers; dispatch law/clock tests, not native strength."""
import unittest

from pokezero.mcts_eval.paper_reference import ReferenceConfig, ReferenceRefusal, _rng
from pokezero.mcts_eval.paper_reference_parallel import ParallelTrajectorySearch, ParallelRefusal, _worker_seed
from test_paper_reference import ROOT
from test_paper_reference_parallel import Request, RngWitnessRuntime, ToyRuntime


class QueuedFailureRuntime(ToyRuntime):
    def prepare(self, request):
        if self.index == 1:
            raise ReferenceRefusal('deliberate queued preparation failure')
        return super().prepare(request)


class ColdDispatchTests(unittest.TestCase):
    def test_invalid_width_refuses_before_spawning(self):
        for width in (0, -1, True, 1.5, 3):
            with self.subTest(width=width), self.assertRaisesRegex(ReferenceRefusal, 'dispatch width'):
                ParallelTrajectorySearch(ReferenceConfig(.5, 1), ToyRuntime, workers=2,
                    initial_dispatch_workers=width)

    def test_default_keeps_immediate_broadcast(self):
        with ParallelTrajectorySearch(ReferenceConfig(.5, 1), ToyRuntime, workers=3) as pool:
            result = pool.search(Request(), ROOT, battle_id='default', seed=11,
                trajectories_per_worker=1)
        self.assertEqual(result.result.trajectories, 3)
        self.assertEqual([r['worker'] for r in result.initial_dispatch_schedule], [0, 1, 2])
        self.assertTrue(all(r['released_by_first_update'] is None for r in result.initial_dispatch_schedule))

    def test_staggering_preserves_each_worker_rng_and_own_counts(self):
        def run(width):
            with ParallelTrajectorySearch(ReferenceConfig(.5, 1), RngWitnessRuntime, workers=4,
                    initial_dispatch_workers=width) as pool:
                result = pool.search(Request(), ROOT, battle_id='rng', seed=17,
                    trajectories_per_worker=12)
            draws = {i: [] for i in range(4)}
            for receipt in result.worker_receipts:
                draws[receipt['worker']].extend(receipt['evidence']['hidden_rng_witness'])
            return result, draws
        immediate, original = run(None)
        staggered, actual = run(1)
        self.assertEqual(actual, original)
        self.assertEqual(staggered.result.trajectories, 48)
        self.assertEqual(sum(staggered.result.root_visits), 48)
        self.assertEqual(staggered.exchange_count, 8)
        self.assertEqual(len(set(staggered.worker_pids)), 4)
        schedule = staggered.initial_dispatch_schedule
        self.assertEqual([r['worker'] for r in schedule], [0, 1, 2, 3])
        self.assertEqual([r['released_by_first_update'] for r in schedule], [None, 0, 1, 2])
        self.assertEqual([r['seconds_from_decision_start'] for r in schedule],
            sorted(r['seconds_from_decision_start'] for r in schedule))

    def test_expired_queued_workers_have_no_new_clock_draw_or_backup(self):
        with ParallelTrajectorySearch(ReferenceConfig(.5, 1), ToyRuntime, workers=4,
                initial_dispatch_workers=1) as pool:
            result = pool.search(Request(delay=.01), ROOT, battle_id='deadline', seed=2,
                deadline_seconds=.035)
        self.assertGreater(result.result.trajectories, 0)
        self.assertEqual(len(result.initial_dispatch_schedule), 4)
        self.assertEqual({r['worker'] for r in result.worker_receipts}, {0, 1, 2, 3})
        for receipt in result.worker_receipts:
            if receipt['worker'] > 0:
                self.assertEqual(receipt['batch'].world_draws, 0)
                self.assertEqual(receipt['batch'].trajectories, 0)
                self.assertEqual(receipt['batch'].transitions, 0)
                self.assertTrue(receipt['batch'].deadline_exhausted)
        self.assertTrue(all(r['deadline_expired'] for r in result.initial_dispatch_schedule[1:]))

    def test_width_resets_for_every_decision_with_persistent_workers(self):
        with ParallelTrajectorySearch(ReferenceConfig(.5, 1), RngWitnessRuntime, workers=3,
                initial_dispatch_workers=1) as pool:
            first = pool.search(Request(), ROOT, battle_id='retained', seed=31,
                trajectories_per_worker=2)
            second = pool.search(Request(), ROOT, battle_id='retained', seed=32,
                trajectories_per_worker=2)
        self.assertEqual(first.worker_pids, second.worker_pids)
        self.assertEqual([r['released_by_first_update'] for r in second.initial_dispatch_schedule], [None, 0, 1])
        self.assertEqual(second.result.trajectories, 6)
        self.assertEqual(sum(second.result.root_visits), 12)

    def test_staggered_recovery_preserves_accepted_stats_and_rng_ordinals(self):
        with ParallelTrajectorySearch(ReferenceConfig(.5, 1), RngWitnessRuntime, workers=2) as old:
            old.search(Request(), ROOT, battle_id='retained', seed=1, trajectories_per_worker=2)
            checkpoint = old._master.snapshot()
        with ParallelTrajectorySearch(ReferenceConfig(.5, 1), RngWitnessRuntime, workers=2,
                initial_dispatch_workers=1) as fresh:
            fresh.restore_statistics(checkpoint, worker_ordinals=(2, 2), decision_id=11)
            result = fresh.search(Request(), ROOT, battle_id='retained', seed=2, trajectories_per_worker=2)
        self.assertEqual(result.decision_id, 12)
        self.assertEqual(result.result.trajectories, 4)
        self.assertEqual(sum(result.result.root_visits), 8)
        for receipt in result.worker_receipts:
            self.assertEqual(receipt['evidence']['hidden_rng_witness'], tuple(
                _rng(_worker_seed(2, receipt['worker']), ordinal, 'hidden').getrandbits(64)
                for ordinal in (2, 3)))

    def test_first_zero_work_update_still_releases_all_queued_workers(self):
        with ParallelTrajectorySearch(ReferenceConfig(.5, 1), ToyRuntime, workers=3,
                initial_dispatch_workers=1) as pool:
            with self.assertRaisesRegex(ParallelRefusal, 'zero NEW') as caught:
                pool.search(Request(preparation_delay=.03), ROOT, battle_id='expired', seed=13,
                    deadline_seconds=.005)
        evidence = caught.exception.evidence
        self.assertEqual([r['worker'] for r in evidence['initial_dispatch_schedule']], [0, 1, 2])
        self.assertEqual([r['released_by_first_update'] for r in evidence['initial_dispatch_schedule']], [None, 0, 1])
        self.assertEqual({r['worker'] for r in evidence['receipts']}, {0, 1, 2})
        self.assertTrue(all(r['batch'].world_draws == 0 and r['batch'].trajectories == 0
            for r in evidence['receipts']))
        self.assertTrue(all(not p.is_alive() for p in pool._processes))

    def test_queued_preparation_failure_is_fatal_and_all_owned_workers_close(self):
        with ParallelTrajectorySearch(ReferenceConfig(.5, 1), QueuedFailureRuntime, workers=3,
                initial_dispatch_workers=1) as pool:
            with self.assertRaisesRegex(ParallelRefusal, 'nonbankable') as caught:
                pool.search(Request(), ROOT, battle_id='failure', seed=13, trajectories_per_worker=1)
            self.assertTrue(pool._poisoned)
        self.assertIn('deliberate queued preparation failure', str(caught.exception.evidence['errors']))
        self.assertEqual(len(pool._processes), 3)
        self.assertTrue(all(not p.is_alive() for p in pool._processes))


if __name__ == '__main__':
    unittest.main()
