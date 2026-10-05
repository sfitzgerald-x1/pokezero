"""Real spawned-process exchange; no Pokémon playing-strength qualification."""
from dataclasses import dataclass, replace
import os
import signal
import subprocess
import sys
import time
import unittest

from pokezero.mcts_eval.paper_reference import Evaluation, ReferenceConfig, ReferenceRefusal
from pokezero.mcts_eval.paper_reference_parallel import (
    ParallelRefusal, ParallelTrajectorySearch, PreparedDecision,
)
from test_paper_reference import ROOT, ToyWorld, searcher


@dataclass(frozen=True)
class Request:
    root: object = ROOT
    delay: float = 0.
    preparation_delay: float = 0.
    failure: bool = False


class ToyRuntime:
    def __init__(self, index):
        self.index = index

    def prepare(self, request):
        time.sleep(request.preparation_delay)
        if request.failure and self.index == 0:
            raise ReferenceRefusal("deliberate public-root refusal")
        def world(rng):
            time.sleep(request.delay)
            return ToyWorld(root=request.root, value=1 if self.index % 2 == 0 else -1)
        return PreparedDecision(request.root, lambda state: Evaluation((.5, .5), 0.), world)

    def close(self):
        pass


class StoppedChildRuntime(ToyRuntime):
    def __init__(self, index):
        super().__init__(index)
        self.child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        os.kill(self.child.pid, signal.SIGSTOP)

    def prepare(self, request):
        prepared = super().prepare(request)
        prepared.evidence = lambda: {"child_pid": self.child.pid, "group": os.getpgrp()}
        return prepared

    # Deliberately leave the stopped child for coordinator escalation.


class ParallelReferenceTests(unittest.TestCase):
    def test_actual_twenty_persistent_processes_exchange_every_ten_without_double_counts(self):
        with ParallelTrajectorySearch(ReferenceConfig(.5, 1), ToyRuntime) as pool:
            first = pool.search(Request(), ROOT, battle_id="b1", seed=1, trajectories_per_worker=12)
            self.assertEqual(len(set(first.worker_pids)), 20)
            self.assertEqual(first.result.trajectories, 240)
            self.assertEqual(sum(first.result.root_visits), 240)
            self.assertEqual(first.exchange_count, 40)
            self.assertEqual(sorted(r["batch"].trajectories for r in first.worker_receipts), [2] * 20 + [10] * 20)
            self.assertEqual(sum(next(row for row in pool._master.snapshot().rows
                if row.state == ROOT).totals), 0.)
            second = pool.search(Request(), ROOT, battle_id="b1", seed=2, trajectories_per_worker=10)
            self.assertEqual(second.worker_pids, first.worker_pids)
            self.assertEqual(sum(second.result.root_visits), 440)
            self.assertEqual(second.result.trajectories, 200)
            self.assertEqual({r["sequence"] for r in second.worker_receipts}, {3})
            new_root = replace(ROOT, key=b"after-real-faint", faint_count=1)
            third = pool.search(Request(new_root), new_root, battle_id="b1", seed=3,
                                trajectories_per_worker=1)
            self.assertEqual(sum(third.result.root_visits), 20)
            self.assertTrue(all(row.state.faint_count >= 1 for row in pool._master.snapshot().rows))
            reset = pool.search(Request(), ROOT, battle_id="b2", seed=4, trajectories_per_worker=1)
            self.assertEqual(reset.worker_pids, first.worker_pids)
            self.assertEqual(sum(reset.result.root_visits), 20)
            self.assertEqual({r["sequence"] for r in reset.worker_receipts}, {1})
        self.assertTrue(all(not process.is_alive() for process in pool._processes))

    def test_partial_deadline_batch_accepted_only_at_coordinator_with_new_work(self):
        with ParallelTrajectorySearch(ReferenceConfig(.5, 1), ToyRuntime, workers=2) as pool:
            result = pool.search(Request(delay=.03), ROOT, battle_id="b", seed=3,
                                 deadline_seconds=.18)
            self.assertGreater(result.result.trajectories, 0)
            self.assertLess(result.result.trajectories, 20)
            self.assertGreaterEqual(result.result.world_draws, result.result.trajectories)
            self.assertTrue(result.result.deadline_exhausted)
            self.assertGreaterEqual(result.result.elapsed_seconds, .18)
            self.assertGreaterEqual(result.result.deadline_overrun_seconds, 0.)

    def test_preparation_is_timed_and_zero_new_work_refuses_despite_old_tree(self):
        with ParallelTrajectorySearch(ReferenceConfig(.5, 1), ToyRuntime, workers=2) as pool:
            pool.search(Request(), ROOT, battle_id="b", seed=1, trajectories_per_worker=1)
            with self.assertRaisesRegex(ParallelRefusal, "zero NEW") as caught:
                pool.search(Request(preparation_delay=.025), ROOT, battle_id="b", seed=2,
                            deadline_seconds=.005)
            self.assertEqual(sum(r["batch"].trajectories for r in caught.exception.evidence["receipts"]), 0)
            self.assertTrue(all(r["preparation_seconds"] >= .025
                for r in caught.exception.evidence["receipts"]))
            with self.assertRaises(ReferenceRefusal):
                pool.search(Request(), ROOT, battle_id="b", seed=3, trajectories_per_worker=1)

    def test_worker_refusal_keeps_evidence_without_replacement_or_retry(self):
        with ParallelTrajectorySearch(ReferenceConfig(.5, 1), ToyRuntime, workers=2) as pool:
            with self.assertRaisesRegex(ParallelRefusal, "nonbankable") as caught:
                pool.search(Request(failure=True), ROOT, battle_id="b", seed=3,
                            trajectories_per_worker=10)
            self.assertIn("deliberate public-root refusal", str(caught.exception.evidence["errors"]))
            self.assertEqual(len(pool._processes), 2)
        self.assertTrue(all(not p.is_alive() for p in pool._processes))

    def test_empty_worker_batch_has_no_action_and_no_raw_or_stale_fallback(self):
        search = searcher()
        arguments = dict(battle_id="battle-1", evaluate_root=lambda state: Evaluation((.5, .5), 0.),
                         sample_world=lambda rng: ToyWorld(), seed=1, trajectories=10)
        first = search.search_batch(ROOT, deadline_at=1., clock=lambda: 2., **arguments)
        self.assertEqual(first.trajectories, 0)
        self.assertFalse(hasattr(first, "action"))
        self.assertEqual(search.nodes, {})
        next_batch = search.search_batch(ROOT, **arguments)
        self.assertEqual(next_batch.trajectories, 10)
        empty = search.search_batch(ROOT, deadline_at=1., clock=lambda: 2., **arguments)
        self.assertEqual(empty.trajectories, 0)
        self.assertEqual(sum(search.nodes[ROOT.key].visits), 10)

    def test_nonpaper_exchange_cadence_and_invalid_work_refuse(self):
        with self.assertRaises(ReferenceRefusal):
            ParallelTrajectorySearch(ReferenceConfig(.5, 1), ToyRuntime, batch_size=3)
        with ParallelTrajectorySearch(ReferenceConfig(.5, 1), ToyRuntime, workers=1) as pool:
            for args in ({}, {"trajectories_per_worker": True}, {"deadline_seconds": 0.}):
                with self.subTest(args=args), self.assertRaises(ReferenceRefusal):
                    pool.search(Request(), ROOT, battle_id="b", seed=1, **args)

    @unittest.skipUnless(os.name == "posix", "POSIX isolated child process group")
    def test_close_reaches_own_stopped_simulator_child_after_worker_exits(self):
        with ParallelTrajectorySearch(ReferenceConfig(.5, 1), StoppedChildRuntime, workers=1) as pool:
            result = pool.search(Request(), ROOT, battle_id="b", seed=1, trajectories_per_worker=1)
            child = result.worker_receipts[0]["evidence"]["child_pid"]
            self.assertEqual(result.worker_receipts[0]["evidence"]["group"], result.worker_pids[0])
        # An orphan can briefly remain a dead zombie pending OS reaping; never
        # mistake that for a live/stopped simulator still consuming resources.
        status = subprocess.run(["ps", "-o", "stat=", "-p", str(child)],
                                capture_output=True, text=True).stdout.strip()
        self.assertTrue(not status or status.startswith("Z"), status)


if __name__ == "__main__":
    unittest.main()
