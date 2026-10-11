"""Actual visited-world capture/label chain on synthetic owned worlds."""
from copy import deepcopy
from dataclasses import replace
import math
import random
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch, Mock

from pokezero.mcts_eval.paper_reference import Evaluation, ReferenceConfig, ReferenceRefusal
from pokezero.mcts_eval.paper_reference_parallel import PreparedDecision, ParallelTrajectorySearch, ParallelRefusal
from pokezero.mcts_eval.paper_reference_showdown import ShowdownTrajectoryWorld, decision_state
from pokezero.mcts_eval.search_over_raw_leaves import ReferenceLeafWorld
from pokezero.mcts_eval.search_over_raw_value_diagnostics import (
    ValueDiagnosticContract, VisitedValueBank, _ValueRuntime, calibration_bounds, reconcile_collected_values)
from pokezero.mcts_eval.search_over_raw import digest
from tests.test_search_over_raw_leaves import LeafEnv, evaluator


class DiagnosticEnv(LeafEnv):
    def legal_actions(self, seat):
        return (True, True)

    def step(self, actions):
        super().step(actions)
        return SimpleNamespace(terminal=self.terminal())

    def snapshot(self):
        return SimpleNamespace(index=self.index, rounds=self.rounds, capped=self.capped, winner=self.winner)

    def restore(self, snapshot):
        self.index, self.rounds, self.capped, self.winner = snapshot.index, snapshot.rounds, snapshot.capped, snapshot.winner


def actual_world(env=None):
    return ShowdownTrajectoryWorld(env or DiagnosticEnv(), subject="p1", evaluator=evaluator, release=lambda: None)


class SyntheticBaseRuntime:
    def __init__(self, index):
        self.index = index

    def prepare(self, request):
        def sample(rng):
            return actual_world()
        root = actual_world().frame().subject
        return PreparedDecision(root, lambda _: Evaluation((.4,.6), .3), sample,
            lambda: {"synthetic_only": True})

    def close(self):
        pass


class SyntheticValueRuntime(_ValueRuntime):
    def __init__(self, index):
        super().__init__(SyntheticBaseRuntime(index), ValueDiagnosticContract(71, time.perf_counter()+60), index)


class BrokenValueRuntime(SyntheticValueRuntime):
    def prepare(self, request):
        prepared = super().prepare(request)
        prepared.finish_diagnostics = lambda progress=None: (_ for _ in ()).throw(RuntimeError("label restore failed"))
        return prepared


class ValueDiagnosticTests(unittest.TestCase):
    def bank(self, env=None, **kwargs):
        world = actual_world(env)
        state = world.frame().subject
        bank = VisitedValueBank(ValueDiagnosticContract(7, 100., **kwargs), worker=0, root_key=state.key,
            clock=lambda: 0.)
        return world, state, bank

    def test_contract_rejects_cap_replicate_seed_or_memory_changes(self):
        for kwargs in ({"sample_seed": True}, {"original_deadline_at": math.inf}, {"replicates": 1},
                {"maximum_boundaries": 251}, {"leaves_per_worker": 0}, {"maximum_snapshot_bytes": 9*1024*1024}):
            with self.assertRaises(ValueError):
                ValueDiagnosticContract(**{"sample_seed": 7, "original_deadline_at": 100., **kwargs})

    def test_actual_prediction_labels_raw_actions_and_restoration(self):
        world, state, bank = self.bank()
        model = world.evaluate(state)
        bank.observe(world, state, model)
        world.env.index = 1  # Simulate search continuing after capture.
        world.close()
        result = bank.label()
        row = result["leaves"][0]
        self.assertEqual(row["model_signed_value"], .3)
        self.assertEqual([label["signed_outcome"] for label in row["labels"]], [1.]*8)
        self.assertEqual(world.env.actions, [{"p1": 1, "p2": 1}]*16)
        self.assertEqual(world.env.index, 1)
        self.assertFalse(result["labels_used_for_backup"])
        self.assertEqual(bank.rows, {})
        self.assertEqual(bank.bytes, 0)
        self.assertNotIn("snapshot", row)

    def test_eight_disjoint_deterministic_label_streams(self):
        streams = []
        for _ in range(2):
            world, state, bank = self.bank()
            bank.observe(world, state, world.evaluate(state))
            bank.label()
            streams.append(world.env.seeds)
        self.assertEqual(streams[0], streams[1])
        self.assertEqual(len(set(streams[0])), 16)

    def test_sampling_priorities_do_not_depend_on_predictions(self):
        rosters = []
        for value in (-1., 1.):
            world, state, bank = self.bank()
            for _ in range(20):
                bank.observe(world, state, Evaluation((.4,.6), value))
            self.assertEqual(len(bank.rows), 2)
            rosters.append(sorted(bank.rows))
        self.assertEqual(rosters[0], rosters[1])

    def test_memory_cap_refuses_instead_of_dropping_large_states(self):
        world, state, bank = self.bank(maximum_snapshot_bytes=1)
        with self.assertRaisesRegex(ValueError, "memory cap"):
            bank.observe(world, state, world.evaluate(state))
        self.assertFalse(bank.rows)

    def test_capture_rejects_live_source_and_wrong_visited_state(self):
        world, state, bank = self.bank()
        world.env._search_snapshot_permitted = False
        with self.assertRaisesRegex(ValueError, "owned hypothetical"):
            bank.observe(world, state, world.evaluate(state))
        world.env._search_snapshot_permitted = True
        world.env.index = 1
        with self.assertRaisesRegex(ValueError, "binding differ"):
            bank.observe(world, state, Evaluation((.4,.6), .3))

    def test_snapshot_mutation_is_not_a_valid_label(self):
        world, state, bank = self.bank()
        bank.observe(world, state, world.evaluate(state))
        next(iter(bank.rows.values()))["snapshot"].index = 1
        with self.assertRaisesRegex(ValueError, "mutated"):
            bank.label()
        self.assertFalse(bank.rows)
        with self.assertRaisesRegex(ValueError, "retried"):
            bank.label()

    def test_boundary_cap_keeps_all_labels_uncertain(self):
        world, state, bank = self.bank(DiagnosticEnv(rounds=251))
        bank.observe(world, state, world.evaluate(state))
        result = bank.label()
        labels = result["leaves"][0]["labels"]
        self.assertEqual([(row["status"], row["boundaries"], row["signed_outcome"]) for row in labels],
            [("CAPPED_UNCERTAIN", 250, None)]*8)
        self.assertEqual(result["calibration"][6]["win_score_interval"], [0., 1.])
        self.assertEqual(world.env.index, 0)

    def test_capped_simulator_terminal_is_unknown(self):
        world, state, bank = self.bank(DiagnosticEnv(capped=True))
        bank.observe(world, state, world.evaluate(state))
        labels = bank.label()["leaves"][0]["labels"]
        self.assertTrue(all(row["status"] == "CAPPED_UNCERTAIN" and row["signed_outcome"] is None for row in labels))

    def test_expired_original_deadline_does_not_touch_environment(self):
        world, state, bank = self.bank()
        bank.observe(world, state, world.evaluate(state))
        bank.clock = lambda: 101.
        world.env.snapshot = lambda: (_ for _ in ()).throw(AssertionError("late I/O"))
        result = bank.label()
        self.assertEqual([row["status"] for row in result["leaves"][0]["labels"]], ["DEADLINE_UNCERTAIN"]*8)
        self.assertFalse(world.env.actions)

    def test_invalid_terminal_halts_and_clears_private_state(self):
        world, state, bank = self.bank(DiagnosticEnv(winner="invalid"))
        bank.observe(world, state, world.evaluate(state))
        with self.assertRaisesRegex(ValueError, "terminal"):
            bank.label()
        self.assertFalse(bank.rows)
        self.assertEqual(world.env.index, 0)

    def test_calibration_includes_unknowns_and_draw_scores(self):
        labels = [dict(replicate=i, status="COMPLETE" if i < 4 else "DEADLINE_UNCERTAIN",
            signed_outcome=0. if i < 4 else None) for i in range(8)]
        row = calibration_bounds([dict(model_signed_value=0., labels=labels)])[5]
        self.assertEqual(row["labels"], 8)
        self.assertEqual(row["unknown_labels"], 4)
        self.assertEqual(row["win_score_interval"], [.25,.75])
        self.assertEqual(row["squared_error_interval"], [0.,.125])

    def collected(self):
        world, state, bank = self.bank()
        bank.observe(world, state, world.evaluate(state))
        selected = dict(root_id="synthetic:config", configuration_sha256="a"*64, runtime_sha256="b"*64)
        values = dict(schema="pokezero.selected-visited-value.v1", root_id=selected["root_id"],
            selected_receipt_sha256=digest(selected), configuration_sha256="a"*64, runtime_sha256="b"*64,
            workers=[dict(worker=0, evidence=bank.label())], labels_used_for_backup=False,
            scientific_strength_evidence=False)
        kwargs = dict(selected=selected, contract=bank.contract, workers=1, information_key=state.key.hex())
        return values, kwargs

    def test_collected_reconciliation_retains_all_labels_and_validates_empty_workers(self):
        values, kwargs = self.collected()
        summary = reconcile_collected_values(values, **kwargs)
        self.assertEqual((summary["sampled_leaves"], summary["labels"], summary["unknown_labels"]), (1, 8, 0))
        worker = values["workers"][0]["evidence"]
        worker.update(model_evaluations_seen=0, leaves=[], calibration=calibration_bounds([]))
        summary = reconcile_collected_values(values, **kwargs)
        self.assertEqual(summary["labels"], 0)
        self.assertTrue(all(row["win_score_interval"] is None for row in summary["calibration"]))

    def test_collected_reconciliation_rejects_incomplete_malformed_and_unbound_rows(self):
        values, kwargs = self.collected()
        worker = lambda row: row["workers"][0]["evidence"]
        leaf = lambda row: worker(row)["leaves"][0]
        label = lambda row: leaf(row)["labels"][0]
        mutations = [lambda row: row.update(selected_receipt_sha256="c"*64),
            lambda row: row.update(runtime_sha256="c"*64), lambda row: row.update(workers=[]),
            lambda row: row["workers"][0].update(worker=True),
            lambda row: worker(row)["contract"].update(original_deadline_at=101.),
            lambda row: worker(row).update(model_evaluations_seen=2),
            lambda row: worker(row).update(source_truth_transported=True),
            lambda row: leaf(row).update(snapshot={"private": "secret"}),
            lambda row: leaf(row).update(evaluation_ordinal=True),
            lambda row: leaf(row).update(model_signed_value=True),
            lambda row: leaf(row).update(priority=1),
            lambda row: leaf(row).update(labels=leaf(row)["labels"][:-1]),
            lambda row: label(row).update(replicate=True), lambda row: label(row).update(boundaries=251),
            lambda row: label(row).update(status="CAPPED_UNCERTAIN"),
            lambda row: worker(row)["calibration"][6].update(labels=1)]
        for mutate in mutations:
            with self.subTest(mutation=mutations.index(mutate)):
                changed = deepcopy(values)
                mutate(changed)
                with self.assertRaises(ValueError):
                    reconcile_collected_values(changed, **kwargs)

    def test_collected_reconciliation_rejects_correctly_hashed_but_wrong_reservoir(self):
        from pokezero.mcts_eval.search_over_raw import rng_seed
        values, kwargs = self.collected()
        worker = values["workers"][0]["evidence"]
        worker["model_evaluations_seen"] = 10
        candidates = sorted((rng_seed("visited-model-sample.v1", kwargs["contract"].sample_seed,
            0, kwargs["information_key"], ordinal), ordinal) for ordinal in range(10))
        wrong = []
        for priority, ordinal in candidates[-2:]:
            leaf = deepcopy(worker["leaves"][0])
            leaf.update(priority=priority, evaluation_ordinal=ordinal)
            wrong.append(leaf)
        worker.update(leaves=wrong, calibration=calibration_bounds(wrong))
        with self.assertRaisesRegex(ValueError, "priority drift"):
            reconcile_collected_values(values, **kwargs)

    def test_runtime_model_capture_preserves_prediction_and_hidden_rng(self):
        runtime = _ValueRuntime(SyntheticBaseRuntime(0), ValueDiagnosticContract(7, time.perf_counter()+60), 0)
        prepared = runtime.prepare(None)
        rng = random.Random(9)
        before = rng.getstate()
        world = prepared.sample_world(rng)
        state = world.frame().subject
        self.assertEqual(world.evaluate(state), Evaluation((.4,.6), .3))
        self.assertEqual(rng.getstate(), before)
        world.close()
        self.assertEqual(prepared.finish_diagnostics()["model_evaluations_seen"], 1)
        runtime.close()

    def test_alternative_leaf_captures_model_before_changed_value(self):
        world = actual_world()
        leaf = ReferenceLeafWorld(world, leaf="raw_rollout", deadline=lambda: None, receipts=[])
        root = world.frame().subject
        base = SimpleNamespace(prepare=lambda _: PreparedDecision(root, lambda _: Evaluation((.4,.6), .3),
            lambda rng: leaf, lambda: {}), close=lambda: None)
        runtime = _ValueRuntime(base, ValueDiagnosticContract(7, time.perf_counter()+60), 0)
        prepared = runtime.prepare(None)
        sampled = prepared.sample_world(random.Random(9))
        self.assertEqual(sampled.evaluate(sampled.frame().subject).value, 1.)
        sampled.close()
        result = prepared.finish_diagnostics()
        self.assertEqual(result["leaves"][0]["model_signed_value"], .3)
        runtime.close()

    def test_spawned_selection_then_labels_do_not_change_statistics(self):
        root = actual_world().frame().subject
        with ParallelTrajectorySearch(ReferenceConfig(.5, 1.), SyntheticValueRuntime, workers=2) as pool:
            selected = pool.search(None, root, battle_id="synthetic-value", seed=7, trajectories_per_worker=2)
            before = pool._master.snapshot()
            rows = pool.finish_diagnostics(selected)
            self.assertEqual(pool._master.snapshot(), before)
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["evidence"]["leaves"] for row in rows))
            self.assertEqual(set(pool.last_evidence["diagnostic_progress"]), {0, 1})
            with self.assertRaisesRegex(ReferenceRefusal, "unconsumed"):
                pool.finish_diagnostics(selected)

    def test_spawned_diagnostic_failure_poisoned_no_retry(self):
        root = actual_world().frame().subject
        with ParallelTrajectorySearch(ReferenceConfig(.5, 1.), BrokenValueRuntime, workers=1) as pool:
            selected = pool.search(None, root, battle_id="synthetic-failure", seed=7, trajectories_per_worker=1)
            with self.assertRaisesRegex(ParallelRefusal, "diagnostic failed"):
                pool.finish_diagnostics(selected)
            self.assertTrue(pool._poisoned)
            with self.assertRaises(ReferenceRefusal):
                pool.finish_diagnostics(selected)

    def test_fabricated_selection_cannot_label(self):
        root = actual_world().frame().subject
        with ParallelTrajectorySearch(ReferenceConfig(.5, 1.), SyntheticValueRuntime, workers=1) as pool:
            selected = pool.search(None, root, battle_id="synthetic-binding", seed=7, trajectories_per_worker=1)
            with self.assertRaisesRegex(ReferenceRefusal, "exact latest"):
                pool.finish_diagnostics(replace(selected))
            pool.finish_diagnostics(selected)

    def test_adapter_optin_wraps_all_reference_leaves_with_distinct_identity(self):
        from pokezero.mcts_eval.search_over_raw_adapters import PublicModelSearchAdapter
        from pokezero.mcts_eval.search_over_raw import SearchConfiguration
        from pokezero.mcts_eval.paper_reference_runtime import ShowdownWorkerFactory
        from pokezero.mcts_eval.search_over_raw_value_diagnostics import ReferenceValueDiagnosticFactory
        checkpoint = SimpleNamespace(checkpoint_path="weights", checkpoint_sha256="a"*64,
            showdown_source_sha256="source")
        factory = ShowdownWorkerFactory("weights", "a"*64, "showdown", "source")
        diagnostic = ValueDiagnosticContract(7, 100.)
        for leaf in ("model", "hp_fraction", "raw_rollout"):
            with patch("pokezero.mcts_eval.paper_reference_parallel.ParallelTrajectorySearch") as pool:
                cfg = SearchConfiguration("reference", leaf=leaf, workers=20)
                normal = PublicModelSearchAdapter(cfg, checkpoint_contract=checkpoint,
                    showdown_root="showdown", reference_factory=factory)
                instrumented = PublicModelSearchAdapter(cfg, checkpoint_contract=checkpoint,
                    showdown_root="showdown", reference_factory=factory, value_diagnostics=diagnostic)
                self.assertIsInstance(pool.call_args.args[1], ReferenceValueDiagnosticFactory)
                self.assertNotEqual(normal.runtime_sha256, instrumented.runtime_sha256)
                self.assertTrue(instrumented.runtime_configuration["visited_value_diagnostics"]["instrumentation_can_change_selection"])
                normal.close()
                instrumented.close()

    def test_adapter_labels_bind_exact_selected_receipt_and_consume_once(self):
        from pokezero.mcts_eval.search_over_raw_adapters import PublicModelSearchAdapter
        from pokezero.mcts_eval.search_over_raw import digest
        adapter = object.__new__(PublicModelSearchAdapter)
        adapter._closed = adapter._poisoned = False
        selected = dict(root_id="root", action=1)
        measured = object()
        adapter._pending_value_diagnostics = (selected, digest(selected), measured)
        adapter.configuration = SimpleNamespace(identity="configuration")
        adapter.runtime_sha256 = "runtime"
        adapter._pool = Mock()
        adapter._pool.finish_diagnostics.return_value = ()
        with self.assertRaisesRegex(ValueError, "receipt changed"):
            adapter.finish_value_diagnostics(dict(selected))
        result = adapter.finish_value_diagnostics(selected)
        self.assertEqual(result["selected_receipt_sha256"], digest(selected))
        adapter._pool.finish_diagnostics.assert_called_once_with(measured)
        with self.assertRaisesRegex(ValueError, "no live"):
            adapter.finish_value_diagnostics(selected)

    def test_diagnostic_runtime_close_discards_unlabeled_private_leaves(self):
        runtime = _ValueRuntime(SyntheticBaseRuntime(0), ValueDiagnosticContract(7, time.perf_counter()+60), 0)
        prepared = runtime.prepare(None)
        world = prepared.sample_world(random.Random(9))
        world.evaluate(world.frame().subject)
        world.close()
        self.assertTrue(runtime.bank.rows)
        runtime.close()
        self.assertFalse(runtime.bank.rows)
        self.assertTrue(runtime.bank.closed)

    def test_actual_public_collector_belief_wrapper_composes_every_leaf(self):
        from pokezero.mcts_eval.search_over_raw_belief_diagnostics import PublicBeliefDiagnosticSearchAdapter, BeliefDiagnosticWorkerFactory
        from pokezero.mcts_eval.paper_reference_runtime import ShowdownWorkerFactory
        from pokezero.mcts_eval.search_over_raw import SearchConfiguration
        from pokezero.mcts_eval.search_over_raw_value_diagnostics import ReferenceValueDiagnosticFactory
        checkpoint = SimpleNamespace(checkpoint_path="weights", checkpoint_sha256="a"*64,
            showdown_source_sha256="b"*64)
        factory = ShowdownWorkerFactory("weights", "a"*64, "server", "b"*64)
        for leaf in ("model", "hp_fraction", "raw_rollout"):
            with patch("pokezero.mcts_eval.paper_reference_parallel.ParallelTrajectorySearch") as pool:
                adapter = PublicBeliefDiagnosticSearchAdapter(SearchConfiguration("reference", leaf=leaf, workers=20),
                    checkpoint_contract=checkpoint, showdown_root="server", reference_factory=factory,
                    value_diagnostics=ValueDiagnosticContract(7, 100.))
                self.assertIsInstance(pool.call_args.args[1], ReferenceValueDiagnosticFactory)
                self.assertIsInstance(pool.call_args.args[1].base, BeliefDiagnosticWorkerFactory)
                self.assertFalse(adapter.runtime_configuration["belief_diagnostics"]["truth_sent_to_public_worker"])
                adapter.close()

    def test_actual_oracle_collector_belief_and_leaf_layers_preserved(self):
        from pokezero.mcts_eval.search_over_raw_belief_diagnostics import OracleBeliefDiagnosticSearchAdapter, BeliefDiagnosticWorkerFactory
        from pokezero.mcts_eval.search_over_raw_oracle import TeamOracle, OracleWorkerFactory, team_sha256
        from pokezero.mcts_eval.paper_reference_runtime import ShowdownWorkerFactory
        from pokezero.mcts_eval.search_over_raw_leaves import ReferenceLeafWorkerFactory
        from pokezero.mcts_eval.search_over_raw import SearchConfiguration
        from tests.test_search_over_raw_belief_diagnostics import team
        checkpoint = SimpleNamespace(checkpoint_path="weights", checkpoint_sha256="a"*64,
            showdown_source_sha256="b"*64)
        factory = ShowdownWorkerFactory("weights", "a"*64, "server", "b"*64)
        source = SimpleNamespace(metadata=SimpleNamespace(source_hash="b"*64))
        oracle = TeamOracle("p1", "b"*64, "c"*64, "d"*64, team(), team_sha256(team()))
        for leaf in ("model", "hp_fraction", "raw_rollout"):
            with patch("pokezero.randbat.load_gen3_randbat_source_cached", return_value=source), \
                    patch("pokezero.mcts_eval.paper_reference_parallel.ParallelTrajectorySearch") as pool:
                adapter = OracleBeliefDiagnosticSearchAdapter(SearchConfiguration("reference", "oracle", leaf, workers=20),
                    oracle=oracle, checkpoint_contract=checkpoint, showdown_root="server", reference_factory=factory,
                    value_diagnostics=ValueDiagnosticContract(7, 100.))
                wrapped = pool.call_args.args[1].base
                self.assertIsInstance(wrapped, BeliefDiagnosticWorkerFactory)
                base = wrapped.base
                if leaf != "model":
                    self.assertIsInstance(base, ReferenceLeafWorkerFactory)
                    base = base.base
                self.assertIsInstance(base, OracleWorkerFactory)
                self.assertEqual(base.base, factory)
                adapter.close()
