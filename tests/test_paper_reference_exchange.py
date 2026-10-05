"""Master totals count real worker samples once; P remains worker-local."""
from dataclasses import asdict, replace
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.paper_reference import DecisionState, Evaluation, ReferenceRefusal
from pokezero.mcts_eval.paper_reference_exchange import (
    MasterSnapshot, Statistics, StatisticsMaster, WorkerUpdate,
)
from test_paper_reference import ROOT, ToyWorld, run, searcher


def update(worker, sequence, visits=(1, 0), totals=(1., 0.), state=ROOT):
    return WorkerUpdate("battle-1", worker, sequence,
        (Statistics(state, visits, totals, sum(visits)),))


class MasterExchangeTests(unittest.TestCase):
    def test_resend_and_stale_receipts_do_not_double_count_or_undo(self):
        master = StatisticsMaster("battle-1")
        first = master.accept(update("a", 1))
        self.assertEqual(master.accept(update("a", 1)), first)
        latest = master.accept(update("a", 2, (2, 0), (0., 0.)))
        self.assertEqual(master.accept(update("a", 1)), latest)
        result = master.accept(update("b", 1, (0, 2), (0., -2.)))
        self.assertEqual(result.rows[0].visits, (2, 2))
        self.assertEqual(result.rows[0].totals, (0., -2.))
        self.assertEqual(result.rows[0].count, 4)
        self.assertNotIn("priors", asdict(result.rows[0]))

    def test_revision_alias_regression_and_mutated_old_value_refuse_atomically(self):
        master = StatisticsMaster("battle-1")
        initial = master.accept(update("a", 1))
        for invalid in (update("a", 1, (1, 0), (-1., 0.)),
                update("a", 2, (0, 0), (0., 0.)), update("a", 2, (1, 0), (-1., 0.)),
                update("a", 2, (1, 0), (1. - 1e-10, 0.)),
                WorkerUpdate("battle-1", "a", 2, ()),
                update("b", 1, state=replace(ROOT, actions=("different", "switch:b"))),
                replace(update("b", 1), battle_id="another-battle")):
            with self.subTest(invalid=invalid), self.assertRaises(ReferenceRefusal):
                master.accept(invalid)
            self.assertEqual(master.snapshot(), initial)

    def test_pruning_uses_real_faints_and_late_old_messages_cannot_resurrect(self):
        master = StatisticsMaster("battle-1")
        past = update("a", 1)
        future_state = DecisionState(b"future", ("move:a", "switch:b"), 2)
        master.accept(past)
        master.accept(update("b", 1, state=future_state))
        master.advance_real_faints(1)
        self.assertEqual([r.state for r in master.accept(past).rows], [future_state])
        master.accept(update("a", 2))
        self.assertEqual([r.state for r in master.snapshot().rows], [future_state])
        with self.assertRaises(ReferenceRefusal):
            master.advance_real_faints(0)

    def test_invalid_counts_and_duplicate_keys_refuse(self):
        for visits, totals, count in (((1, 0), (2., 0.), 1),
                ((1, 0), (1.0000000008, 0.), 1),
                ((1, 0), (1., 0.), 2), ((True, 0), (1., 0.), 1),
                ((0, 0), (.1, 0.), 0), ((1, 0), (float("nan"), 0.), 1)):
            with self.subTest(visits=visits, totals=totals), self.assertRaises(ReferenceRefusal):
                Statistics(ROOT, visits, totals, count)
        row = Statistics(ROOT, (1, 0), (1., 0.), 1)
        with self.assertRaisesRegex(ReferenceRefusal, "duplicate"):
            WorkerUpdate("battle-1", "a", 1, (row, row))

    def test_aggregate_validation_failure_does_not_commit_an_update(self):
        master = StatisticsMaster("battle-1")
        before = master.accept(update("a", 1))
        with patch.object(master, "_snapshot", side_effect=ReferenceRefusal("aggregate invalid")):
            with self.assertRaisesRegex(ReferenceRefusal, "aggregate invalid"):
                master.accept(update("b", 1))
        self.assertEqual(master.snapshot(), before)


class WorkerExchangeTests(unittest.TestCase):
    def test_two_workers_exchange_every_ten_and_export_only_own_backups(self):
        workers = [searcher(), searcher()]
        master = StatisticsMaster("battle-1")
        for sequence in range(1, 4):
            for index, worker in enumerate(workers):
                run(worker, lambda rng: ToyWorld(value=1 if index == 0 else -1), trajectories=10)
                message = worker.export_statistics(worker_id=str(index), sequence=sequence)
                self.assertEqual(next(r for r in message.rows if r.state == ROOT).count, sequence * 10)
                worker.merge_statistics(master.accept(message))
            self.assertEqual(next(r for r in master.snapshot().rows if r.state == ROOT).count,
                sequence * 20)
        self.assertEqual(sum(next(r for r in master.snapshot().rows if r.state == ROOT).totals), 0.)

    def test_imported_priors_are_recomputed_locally_without_new_leaf_stop(self):
        a, b = searcher(), searcher()
        master = StatisticsMaster("battle-1")
        # The first worker expands a leaf; its zero-count expansion is shared.
        run(a, lambda rng: ToyWorld(depth=2), trajectories=1)
        b.merge_statistics(master.accept(a.export_statistics(worker_id="a", sequence=1)))
        self.assertEqual(b.nodes, {})  # P wasn't sent.
        world = ToyWorld(depth=2, value=-1)
        run(b, lambda rng: world, trajectories=1)
        self.assertEqual(world.index, 2)  # Not stopped at remotely-known leaf.
        self.assertEqual(len(world.evaluations), 1)  # Recomputes local leaf P.
        self.assertEqual(b.nodes[b"leaf:1"].priors, (1.,))
        own = b.export_statistics(worker_id="b", sequence=1)
        self.assertEqual(next(r for r in own.rows if r.state == ROOT).count, 1)
        result = master.accept(own)
        self.assertEqual(next(r for r in result.rows if r.state == ROOT).count, 2)

    def test_reset_and_refusal_protect_battle_scope_and_completed_evidence(self):
        worker = searcher()
        run(worker, lambda rng: ToyWorld(), trajectories=2)
        with self.assertRaisesRegex(ReferenceRefusal, "unexported"):
            worker.merge_statistics(MasterSnapshot("battle-1", 1, 0, ()))
        master = StatisticsMaster("battle-1")
        snapshot = master.accept(worker.export_statistics(worker_id="a", sequence=1))
        worker.merge_statistics(snapshot)
        with self.assertRaisesRegex(ReferenceRefusal, "same master version"):
            worker.merge_statistics(replace(snapshot, rows=(
                Statistics(ROOT, (2, 0), (-2., 0.), 2),)))
        with self.assertRaisesRegex(ReferenceRefusal, "same master version"):
            worker.merge_statistics(replace(snapshot, acknowledged=(("a", 99),)))
        with self.assertRaises(ReferenceRefusal):
            worker.merge_statistics(replace(snapshot, version=0))
        worker.reset_battle("battle-2")
        self.assertFalse(worker.export_statistics(worker_id="a", sequence=1).rows)
        with self.assertRaises(ReferenceRefusal):
            worker.merge_statistics(snapshot)

    def test_old_snapshot_cannot_erase_unsent_work_even_if_master_counts_dominate(self):
        a, b = searcher(), searcher()
        master = StatisticsMaster("battle-1")
        run(a, lambda rng: ToyWorld(value=0), trajectories=10)
        master.accept(a.export_statistics(worker_id="a", sequence=1))
        run(b, lambda rng: ToyWorld(value=0), trajectories=100)
        old = master.accept(b.export_statistics(worker_id="b", sequence=1))
        a.merge_statistics(old)
        run(a, lambda rng: ToyWorld(value=0), trajectories=3)
        self.assertEqual(sum(a.nodes[ROOT.key].visits), 113)
        with self.assertRaisesRegex(ReferenceRefusal, "unexported"):
            a.merge_statistics(old)
        self.assertEqual(sum(a.nodes[ROOT.key].visits), 113)
        new = a.export_statistics(worker_id="a", sequence=2)
        with self.assertRaisesRegex(ReferenceRefusal, "acknowledged"):
            a.merge_statistics(old)
        a.merge_statistics(master.accept(new))
        self.assertEqual(sum(a.nodes[ROOT.key].visits), 113)

    def test_own_export_identity_and_sequence_are_battle_scoped(self):
        worker = searcher()
        run(worker, lambda rng: ToyWorld(), trajectories=2)
        message = worker.export_statistics(worker_id="a", sequence=1)
        self.assertEqual(worker.export_statistics(worker_id="a", sequence=1), message)
        run(worker, lambda rng: ToyWorld(), trajectories=1)
        with self.assertRaisesRegex(ReferenceRefusal, "same export sequence"):
            worker.export_statistics(worker_id="a", sequence=1)
        with self.assertRaises(ReferenceRefusal):
            worker.export_statistics(worker_id="b", sequence=2)
        self.assertEqual(next(r for r in worker.export_statistics(worker_id="a", sequence=2).rows
            if r.state == ROOT).count, 3)


if __name__ == "__main__":
    unittest.main()
