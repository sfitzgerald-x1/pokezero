"""Mocked selectors only: no source games, engine calls or scientific outcomes."""
from dataclasses import dataclass, replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pokezero.mcts_eval.paper_reference import BatchResult, SearchResult
from pokezero.mcts_eval.paper_reference_parallel import ParallelResult
from pokezero.mcts_eval.paper_reference_runtime import ShowdownWorkerFactory
from pokezero.mcts_eval.search_over_raw import SearchConfiguration
from pokezero.mcts_eval.search_over_raw_adapters import _incumbent_runtime
from pokezero.mcts_eval.search_over_raw_fixed_work import FixedWorkDiagnosticAdapter, FixedWorkDiagnosticContract
from pokezero.policy import PolicyContext, PolicyDecision
from pokezero.trajectory import BattleTrajectory

SHA = "a" * 64
ROOT = SimpleNamespace(key=b"fixed-work-fixture")


@dataclass(frozen=True)
class NativeConfigFixture:
    rollout_seed: int = 0


def measured_work():
    receipts = tuple(dict(worker=i, batch=BatchResult(4, 4, 4, .2, False, 0.),
        evidence=dict(draws=[dict(status="ROOT_VALIDATED", released=True) for _ in range(4)]))
        for i in range(20))
    return ParallelResult(SearchResult("action:0", 80, 80, 80, (40, 40), (0., 0.), 2., False, 0.),
        tuple(range(1, 21)), receipts, 20, .5, 1)


class FixedWorkDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.work = FixedWorkDiagnosticContract()
        self.contract = SimpleNamespace(checkpoint_path="fixture-weights", checkpoint_sha256=SHA,
            showdown_source_sha256="fixture-source")
        self.factory = ShowdownWorkerFactory("fixture-weights", SHA, "fixture-showdown", "fixture-source")
        obs = SimpleNamespace(legal_action_mask=(True, True))
        self.context = PolicyContext("p1", 0, "fixture", "gen3randombattle", 123,
            obs, ("p1",), BattleTrajectory("fixture", "gen3randombattle", 123),
            requested_observations={"p1": obs}, public_materialization_state="fixture-public")
        self.request = SimpleNamespace(observation=obs, set_source_hash="fixture-source")

    def select(self, adapter):
        with patch("pokezero.mcts_eval.head_to_head.public_only_context", side_effect=lambda c: c), \
             patch("pokezero.mcts_eval.paper_reference_runtime.PublicRootRequest.capture", return_value=self.request), \
             patch("pokezero.mcts_eval.paper_reference_showdown.decision_state", return_value=ROOT):
            return adapter.select(self.context, root_id="fixture-root", selection_seed=12)

    def test_contract_binds_actual_backend_boundary_and_no_equal_cpu_claim(self):
        self.assertEqual(self.work.accepted_boundary("incumbent"), 64)
        self.assertEqual(self.work.accepted_boundary("reference"), 80)
        row = self.work.receipt()
        self.assertIsNone(row["search_decision_ceiling"])
        self.assertFalse(row["deployment_eligible"])
        self.assertFalse(row["equal_cpu_claim"])
        self.assertFalse(row["runtime_qualified"])
        self.assertEqual(row["watchdog_result"], "DIAGNOSTIC_INCOMPLETE")
        for key, value in (("accepted_target", 32), ("incumbent_worlds", 1),
                           ("reference_workers", 1), ("reference_trajectories_per_worker", 3),
                           ("emergency_watchdog_seconds", 10), ("accepted_target", 64.)):
            with self.assertRaises(ValueError):
                replace(self.work, **{key: value})

    def test_rejects_deployment_ceiling_oracle_and_raw_before_runtime(self):
        for cfg in (SearchConfiguration("raw"), SearchConfiguration("reference", seconds=3., workers=20),
                    SearchConfiguration("incumbent", belief="oracle")):
            with patch("pokezero.mcts_eval.paper_reference_parallel.ParallelTrajectorySearch") as pool:
                with self.assertRaises(ValueError):
                    FixedWorkDiagnosticAdapter(cfg, fixed_work=self.work,
                        checkpoint_contract=self.contract, showdown_root="fixture-showdown")
                pool.assert_not_called()

    def test_native_fixed_work_setup_keeps_depth_batch_worlds_and_no_decision_clock(self):
        with patch("pokezero.mcts_eval.policy_opponent_profile.make_profile_decider") as make:
            decider, _, cfg = _incumbent_runtime(self.contract, "fixture-showdown", 1.,
                leaf="raw_rollout", fixed_work_iterations=16)
            self.assertEqual((cfg.depth, cfg.sims, cfg.batch, cfg.worlds), (6, 16, 16, 4))
            self.assertEqual(make.call_args.kwargs["mode"], "fixed_work")
            self.assertEqual(make.call_args.kwargs["native_batch_guard_ms"], 0)
            self.assertEqual(make.call_args.kwargs["model_leaf_override"], "raw_policy_terminal")
            decider.close()
        # Check the real profile constructor's mapping, without creating an engine.
        from pokezero.mcts_eval.policy_opponent_profile import make_profile_decider
        with patch("pokezero.mcts_eval.policy_opponent_profile._LiveEngineTimingDecider") as live:
            make_profile_decider(self.contract, "fixture-showdown", arm="incumbent_mcts",
                mode="fixed_work", opponent_seed=0, deadline_ms=1000, native_batch_guard_ms=0)
            self.assertIsNone(live.call_args.kwargs["model_decision_time_ms"])
            self.assertEqual(live.call_args.kwargs["model_native_batch_guard_ms"], 0)

    def test_legacy_native_setup_retains_deadline_and_original_work_capacity(self):
        with patch("pokezero.mcts_eval.policy_opponent_profile.make_profile_decider") as make:
            decider, _, cfg = _incumbent_runtime(self.contract, "fixture-showdown", 3.)
            self.assertEqual((cfg.depth, cfg.sims, cfg.batch, cfg.worlds), (6, 4096, 16, 4))
            self.assertEqual(make.call_args.kwargs["mode"], "matched_deadline")
            self.assertEqual(make.call_args.kwargs["deadline_ms"], 3000)
            self.assertEqual(make.call_args.kwargs["native_batch_guard_ms"], 64)
            decider.close()
        with patch("pokezero.mcts_eval.policy_opponent_profile.make_profile_decider") as make:
            with self.assertRaises(ValueError):
                _incumbent_runtime(self.contract, "fixture-showdown", 1., fixed_work_iterations=16.)
            make.assert_not_called()

    def reference(self, pool_type, leaf="model"):
        adapter = FixedWorkDiagnosticAdapter(SearchConfiguration("reference", leaf=leaf, seconds=1., workers=20),
            fixed_work=self.work, checkpoint_contract=self.contract, showdown_root="fixture-showdown",
            reference_factory=self.factory)
        self.addCleanup(adapter.close)
        pool_type.return_value.search.return_value = measured_work()
        return adapter

    def test_reference_uses_all_twenty_workers_fixed_work_and_reports_eighty(self):
        with patch("pokezero.mcts_eval.paper_reference_parallel.ParallelTrajectorySearch") as pool_type:
            adapter = self.reference(pool_type)
            row = self.select(adapter)
            call = pool_type.return_value.search.call_args.kwargs
            self.assertEqual(call["trajectories_per_worker"], 4)
            self.assertNotIn("deadline_seconds", call)
            self.assertEqual(pool_type.call_args.kwargs["workers"], 20)
            cost = row["fixed_work_diagnostic"]
            self.assertEqual((cost["accepted_target"], cost["accepted_actual"]), (64, 80))
            self.assertIsNone(row["nominal_search_seconds"])
            self.assertFalse(cost["resource_cost_complete"])
            self.assertIsNone(cost["cpu_seconds_including_children"])
            self.assertFalse(cost["deployment_eligible"])

    def test_reference_partial_or_asymmetric_allocation_cannot_complete(self):
        with patch("pokezero.mcts_eval.paper_reference_parallel.ParallelTrajectorySearch") as pool_type:
            adapter = self.reference(pool_type)
            for measured in (replace(measured_work(), result=replace(measured_work().result, trajectories=64)),
                             replace(measured_work(), worker_receipts=tuple(
                                 dict(r, worker=0) for r in measured_work().worker_receipts)),
                             replace(measured_work(), result=replace(measured_work().result, deadline_exhausted=True))):
                with self.assertRaises(ValueError):
                    adapter._validate_diagnostic_work(measured)

    def test_terminal_reference_still_requires_actual_terminal_leaf_evidence(self):
        with patch("pokezero.mcts_eval.paper_reference_parallel.ParallelTrajectorySearch") as pool_type:
            adapter = self.reference(pool_type, leaf="raw_rollout")
            with self.assertRaisesRegex(ValueError, "authoritative worker evidence"):
                self.select(adapter)
            self.assertTrue(adapter._poisoned)
            with self.assertRaisesRegex(ValueError, "poisoned"):
                self.select(adapter)

    def test_native_routes_fixed_work_validation_and_requires_sixty_four(self):
        native, decider = Mock(), Mock()
        native._fixed_override = None
        native._config = NativeConfigFixture()
        native.select_action_with_context.return_value = PolicyDecision(0, "fixture", metadata={"engine_mcts": {}})
        decider._snapshot_stats.side_effect = [dict(depth_reached_histogram={}, fallback_decisions=0,
            prior_fallbacks=0, total_iterations=n, model_evals=n) for n in (0, 64)]
        decider._changed_depth.return_value = 1
        with patch("pokezero.mcts_eval.search_over_raw_fixed_work._incumbent_runtime",
                return_value=(decider, native, "fixture-config")) as setup, \
             patch("pokezero.mcts_eval.policy_opponent_profile.validate_selection") as validate:
            adapter = FixedWorkDiagnosticAdapter(SearchConfiguration("incumbent", seconds=1.),
                fixed_work=self.work, checkpoint_contract=self.contract, showdown_root="fixture-showdown")
            self.addCleanup(adapter.close)
            row = self.select(adapter)
            self.assertEqual(setup.call_args.kwargs["fixed_work_iterations"], 16)
            self.assertEqual(validate.call_args.kwargs["mode"], "fixed_work")
            self.assertEqual(row["fixed_work_diagnostic"]["accepted_actual"], 64)
            self.assertEqual(adapter.runtime_configuration["incumbent"]["native_batch_guard_ms"], 0)
            with self.assertRaisesRegex(ValueError, "complete accepted allocation"):
                adapter._execution_receipt(1., dict(total_iterations=63))

    def test_fixed_work_runtime_identity_distinguishes_it_from_deployment(self):
        from pokezero.mcts_eval.search_over_raw_adapters import PublicModelSearchAdapter
        cfg = SearchConfiguration("reference", seconds=1., workers=20)
        with patch("pokezero.mcts_eval.paper_reference_parallel.ParallelTrajectorySearch") as pool_type:
            fixed = self.reference(pool_type)
            timed = PublicModelSearchAdapter(cfg, checkpoint_contract=self.contract,
                showdown_root="fixture-showdown", reference_factory=self.factory)
            self.addCleanup(timed.close)
            self.assertNotEqual(fixed.runtime_sha256, timed.runtime_sha256)


if __name__ == "__main__":
    unittest.main()
