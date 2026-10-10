"""Real public root boundaries plus instrumented selectors; not strength data."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from _showdown_root import requires_showdown, showdown_root
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
from pokezero.mcts_eval.paper_reference import BatchResult, Evaluation, SearchResult
from pokezero.mcts_eval.paper_reference_parallel import ParallelResult
from pokezero.mcts_eval.paper_reference_runtime import ShowdownWorkerFactory
from pokezero.mcts_eval.paper_reference_showdown import ChampionEvaluator
from pokezero.mcts_eval.search_over_raw import SearchConfiguration
from pokezero.mcts_eval.search_over_raw_adapters import PublicModelSearchAdapter, validate_reference_work
from pokezero.policy import PolicyContext, PolicyDecision
from pokezero.trajectory import BattleTrajectory

SHA = "a" * 64


def work():
    batch = BatchResult(1, 2, 1, .01, False, 0.)
    return ParallelResult(SearchResult("action:0", 1, 2, 1, (1,), (0.,), .01, False, 0.),
        tuple(range(1, 21)), ({"worker": 0, "batch": batch,
            "evidence": {"draws": [{"status": "ROOT_VALIDATED", "released": True}]}},), 1, .1, 1)


class FakeEvaluator(ChampionEvaluator):
    def __init__(self):
        self.policy = SimpleNamespace(weights_sha256=SHA)
        self.forwards = 0
        self.seen = None

    def __call__(self, obs):
        self.seen = obs
        self.forwards += 1
        legal = tuple(i for i, enabled in enumerate(obs.legal_action_mask) if enabled)
        return legal, Evaluation((1/len(legal),)*len(legal), .2)


class AdapterContractTests(unittest.TestCase):
    def test_unsupported_oracle_and_leaves_fail_before_runtime_construction(self):
        for arm, belief, leaf in [("reference", "oracle", "model"), ("incumbent", "oracle", "raw_rollout")]:
            with self.assertRaisesRegex(ValueError, "adapters remain required"):
                PublicModelSearchAdapter(SearchConfiguration(arm, belief, leaf, workers=20 if arm == "reference" else 1),
                    checkpoint_contract=None, showdown_root="")

    def test_reference_cannot_change_resource_allocation_or_omit_factory(self):
        with self.assertRaisesRegex(ValueError, "resource allocation"):
            PublicModelSearchAdapter(SearchConfiguration("reference", workers=1),
                checkpoint_contract=None, showdown_root="")
        with self.assertRaisesRegex(ValueError, "factory required"):
            PublicModelSearchAdapter(SearchConfiguration("reference", workers=20),
                checkpoint_contract=None, showdown_root="")

    def test_reference_checkpoint_binding_checked_before_pool(self):
        contract = SimpleNamespace(checkpoint_path="weights", checkpoint_sha256=SHA,
            showdown_source_sha256="source")
        factory = ShowdownWorkerFactory("weights", "b"*64, "showdown", "source")
        with patch("pokezero.mcts_eval.paper_reference_parallel.ParallelTrajectorySearch") as pool:
            with self.assertRaisesRegex(ValueError, "binding differs"):
                PublicModelSearchAdapter(SearchConfiguration("reference", workers=20),
                    checkpoint_contract=contract, showdown_root="showdown", reference_factory=factory)
            pool.assert_not_called()

    def test_reference_leaf_factory_and_runtime_identity_do_not_alias_model(self):
        from pokezero.mcts_eval.search_over_raw_leaves import ReferenceLeafWorkerFactory
        contract = SimpleNamespace(checkpoint_path="weights", checkpoint_sha256=SHA,
            showdown_source_sha256="source")
        factory = ShowdownWorkerFactory("weights", SHA, "showdown", "source")
        identities = set()
        for leaf in ("model", "hp_fraction", "raw_rollout"):
            with patch("pokezero.mcts_eval.paper_reference_parallel.ParallelTrajectorySearch") as pool:
                adapter = PublicModelSearchAdapter(SearchConfiguration("reference", leaf=leaf, workers=20),
                    checkpoint_contract=contract, showdown_root="showdown", reference_factory=factory)
                self.addCleanup(adapter.close)
                dispatched = pool.call_args.args[1]
                if leaf == "model":
                    self.assertIs(dispatched, factory)
                else:
                    self.assertIsInstance(dispatched, ReferenceLeafWorkerFactory)
                    self.assertIs(dispatched.base, factory)
                    self.assertEqual(dispatched.leaf, leaf)
                    self.assertEqual(adapter.runtime_configuration["reference_leaf"]["priors"], "unchanged_champion")
                identities.add(adapter.runtime_sha256)
        self.assertEqual(len(identities), 3)

    def test_incumbent_hp_uses_the_explicit_model_tree_hook_and_distinct_identity(self):
        contract = SimpleNamespace(checkpoint_path="weights", checkpoint_sha256=SHA,
            showdown_source_sha256="source")
        from unittest.mock import Mock
        identities = []
        for leaf in ("model", "hp_fraction", "raw_rollout"):
            decider = Mock()
            with patch("pokezero.mcts_eval.search_over_raw_adapters._incumbent_runtime",
                    return_value=(decider, Mock(), "config")) as runtime:
                adapter = PublicModelSearchAdapter(SearchConfiguration("incumbent", leaf=leaf),
                    checkpoint_contract=contract, showdown_root="showdown")
                self.addCleanup(adapter.close)
                self.assertEqual(runtime.call_args.kwargs, {} if leaf == "model" else {"leaf": leaf})
                identities.append(adapter.runtime_sha256)
                if leaf == "hp_fraction":
                    self.assertEqual(adapter.runtime_configuration["incumbent_leaf"]["tree"],
                        "unchanged_encoded_model_tree")
        self.assertEqual(len(set(identities)), 3)

    def test_reference_receipt_reconciles_new_work(self):
        validate_reference_work(work())
        for changed in [replace(work(), worker_pids=(1,)*20),
                replace(work(), result=replace(work().result, trajectories=0)),
                replace(work(), result=replace(work().result, transitions=3))]:
            with self.assertRaises(ValueError):
                validate_reference_work(changed)

    def test_refused_and_unreleased_draws_are_not_work(self):
        for fields in [{"status": "REFUSED"}, {"status": "ROOT_VALIDATED", "released": False}]:
            measured = work()
            measured.worker_receipts[0]["evidence"]["draws"][0] = fields
            with self.assertRaisesRegex(ValueError, "refused or unreleased"):
                validate_reference_work(measured)

    def test_cancelled_draws_require_zero_backups_and_actual_expired_clock(self):
        measured = work()
        cancelled = {"status": "DEADLINE_CANCELLED", "sampling_diagnostic": {
            "schema": "pokezero.world-sampling-deadline.v1", "checked_at": 2., "deadline_at": 1.,
            "accepted_world": False, "backed_up": False}}
        receipt = {"worker": 1, "batch": BatchResult(0, 0, 1, .1, True, 0.),
            "evidence": {"draws": [cancelled]}}
        measured = replace(measured, result=replace(measured.result, world_draws=2),
            worker_receipts=(*measured.worker_receipts, receipt))
        validate_reference_work(measured)
        for key,value in [("backed_up", True), ("accepted_world", True),
                ("checked_at", .5), ("deadline_at", float("nan"))]:
            changed = deepcopy(measured)
            changed.worker_receipts[-1]["evidence"]["draws"][0]["sampling_diagnostic"][key] = value
            with self.assertRaisesRegex(ValueError, "zero-backup"):
                validate_reference_work(changed)


@requires_showdown()
class PublicBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.env = LocalShowdownEnv(LocalShowdownConfig(showdown_root=showdown_root(), set_belief_source=True))
        self.addCleanup(self.env.close)
        # Excluded engineering fixture seed; never part of an outcome panel.
        self.env.reset(seed=2026101009)
        own, opponent = self.env.observe("p1"), self.env.observe("p2")
        self.public = self.env.public_materialization_state("p1")
        source_hash = self.public.belief_engine.set_source.metadata.source_hash
        self.contract = SimpleNamespace(checkpoint_path="weights", checkpoint_sha256=SHA,
            showdown_source_sha256=source_hash)
        own = replace(own, metadata={**own.metadata, "private_canary": "never-transport"})
        self.context = PolicyContext("p1", 0, "fixture", "gen3randombattle", 2026101009,
            own, tuple(self.env.requested_players()), BattleTrajectory("fixture", "gen3randombattle", 2026101009),
            requested_legal_action_masks={"p1": own.legal_action_mask, "p2": opponent.legal_action_mask},
            requested_observations={"p1": own, "p2": opponent}, public_materialization_state=self.public)

    def raw(self):
        evaluator = FakeEvaluator()
        adapter = PublicModelSearchAdapter(SearchConfiguration("raw"), checkpoint_contract=self.contract,
            showdown_root=str(showdown_root()), evaluator=evaluator)
        self.addCleanup(adapter.close)
        return adapter, evaluator

    def test_public_capture_preserves_engine_semantic_switch_vocabulary(self):
        from pokezero.engine_search import EngineMctsPolicy
        from pokezero.mcts_eval.paper_reference_runtime import PublicRootRequest
        policy = object.__new__(EngineMctsPolicy)
        request = PublicRootRequest.capture(self.public, self.context.observation)
        original = policy._choice_vocabulary(self.context)
        captured = policy._choice_vocabulary(replace(self.context, observation=request.observation))
        self.assertTrue(original.switch_index_by_species)
        self.assertEqual(captured, original)

    def test_raw_argmax_uses_one_forward_and_strips_extra_metadata(self):
        adapter, evaluator = self.raw()
        row = adapter.select(self.context, root_id="fixture:0", selection_seed=5)
        self.assertEqual(row["action"], next(i for i,x in enumerate(self.context.observation.legal_action_mask) if x))
        self.assertEqual(evaluator.forwards, 1)
        self.assertNotIn("private_canary", evaluator.seen.metadata)
        self.assertFalse(row["scientific_strength_evidence"])
        self.assertEqual(row["runtime_sha256"], adapter.runtime_sha256)
        with self.assertRaisesRegex(ValueError, "fresh attempt"):
            adapter.select(self.context, root_id="fixture:0", selection_seed=5)

    def test_failure_poisoned_and_no_raw_fallback_or_retry(self):
        adapter, evaluator = self.raw()
        with self.assertRaisesRegex(ValueError, "nonterminal"):
            adapter.select(replace(self.context, requested_players=("p2",)), root_id="bad", selection_seed=0)
        self.assertEqual(evaluator.forwards, 0)
        self.assertEqual(adapter.last_failure["error_type"], "ValueError")
        with self.assertRaisesRegex(ValueError, "poisoned"):
            adapter.select(self.context, root_id="new", selection_seed=0)

    def test_reference_passes_only_canonical_public_request_and_remaining_ceiling(self):
        factory = ShowdownWorkerFactory("weights", SHA, str(showdown_root()), self.contract.showdown_source_sha256)
        with patch("pokezero.mcts_eval.paper_reference_parallel.ParallelTrajectorySearch") as pool_type:
            pool = pool_type.return_value
            legal = next(i for i,x in enumerate(self.context.observation.legal_action_mask) if x)
            pool.search.return_value = replace(work(), result=replace(work().result, action=f"action:{legal}"))
            adapter = PublicModelSearchAdapter(SearchConfiguration("reference", seconds=1., workers=20),
                checkpoint_contract=self.contract, showdown_root=str(showdown_root()), reference_factory=factory)
            row = adapter.select(self.context, root_id="fixture:ref", selection_seed=4)
            request, _ = pool.search.call_args.args
            self.assertNotIn("private_canary", request.observation.metadata)
            self.assertFalse(request.state.replay.requests)
            self.assertIsNone(request.state.belief_engine.set_source)
            self.assertEqual(pool.search.call_args.kwargs["battle_id"], "search-over-raw-root:fixture:ref")
            self.assertTrue(0 < pool.search.call_args.kwargs["deadline_seconds"] <= 1.)
            self.assertEqual(row["action"], legal)
            adapter.close()
            adapter.close()
            pool.close.assert_called_once()

    def test_reference_alternative_leaf_requires_matching_worker_evidence(self):
        factory = ShowdownWorkerFactory("weights", SHA, str(showdown_root()), self.contract.showdown_source_sha256)
        with patch("pokezero.mcts_eval.paper_reference_parallel.ParallelTrajectorySearch") as pool_type:
            pool_type.return_value.search.return_value = work()
            adapter = PublicModelSearchAdapter(SearchConfiguration("reference", leaf="raw_rollout", workers=20),
                checkpoint_contract=self.contract, showdown_root=str(showdown_root()), reference_factory=factory)
            self.addCleanup(adapter.close)
            with self.assertRaisesRegex(ValueError, "authoritative worker evidence"):
                adapter.select(self.context, root_id="fixture:raw-leaf", selection_seed=5)
            self.assertTrue(adapter._poisoned)
            self.assertIsNone(adapter.last_failure.get("action"))

    def test_incumbent_receives_no_opponent_observation_and_resets_root(self):
        legal = next(i for i,x in enumerate(self.context.observation.legal_action_mask) if x)
        native = SimpleNamespace(reset=lambda: None, select_action_with_context=lambda context, rng: PolicyDecision(legal, "fixture"))
        from unittest.mock import Mock
        native.reset = Mock()
        native.select_action_with_context = Mock(return_value=PolicyDecision(legal, "fixture"))
        decider = Mock()
        decider._snapshot_stats.return_value = dict(depth_reached_histogram={}, fallback_decisions=0,
            prior_fallbacks=0, total_iterations=0, model_evals=0)
        decider._changed_depth.return_value = 0
        with patch("pokezero.mcts_eval.search_over_raw_adapters._incumbent_runtime", return_value=(decider, native, "config")), \
                patch("pokezero.mcts_eval.policy_opponent_profile.validate_selection") as validate:
            adapter = PublicModelSearchAdapter(SearchConfiguration("incumbent", seconds=3.),
                checkpoint_contract=self.contract, showdown_root=str(showdown_root()))
            adapter.select(self.context, root_id="fixture:inc", selection_seed=7)
            seen = native.select_action_with_context.call_args.args[0]
            self.assertEqual(set(seen.requested_observations), {"p1"})
            self.assertEqual(set(seen.requested_legal_action_masks), {"p1"})
            self.assertNotIn("private_canary", seen.observation.metadata)
            native.reset.assert_called_once()
            self.assertEqual(validate.call_args.kwargs["deadline_ms"], 3000)
            from pokezero.engine_search import EngineSearchFallbackError
            from pokezero.mcts_eval.policy_opponent_profile import fallback_refusal_diagnostic
            deadline = dict(schema="engine-search-fallback-refusal-v1", diagnostic_only_not_policy_input=True,
                reason="model_time_budget_no_completed_worlds",
                engine_mcts={"time_budget": {"exhausted": True, "native_invocations": []}})
            failure = EngineSearchFallbackError("strict refusal", diagnostic=deadline)
            native.select_action_with_context.side_effect = failure
            with self.assertRaises(EngineSearchFallbackError):
                adapter.select(self.context, root_id="fixture:deadline", selection_seed=8)
            saved = adapter.last_failure
            self.assertEqual(saved["status"], "UNCERTAIN_REFUSED")
            self.assertFalse(saved["retry_authorized"])
            self.assertEqual(saved["engine_fallback_diagnostic"], deadline)
            self.assertNotIn("native_diagnostic", saved)
            self.assertNotIn("action", saved)
            validate.assert_called_once()  # A refused selection is never validated.
            with self.assertRaisesRegex(ValueError, "poisoned"):
                adapter.select(self.context, root_id="fixture:no-retry", selection_seed=9)
            self.assertEqual(native.select_action_with_context.call_count, 2)
            saved["engine_fallback_diagnostic"]["engine_mcts"]["time_budget"]["exhausted"] = False
            self.assertTrue(fallback_refusal_diagnostic(failure)["engine_mcts"]["time_budget"]["exhausted"])
            adapter.close()
            decider.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
