"""Real raw-forward and fail-closed three-arm measurement boundary tests."""

from copy import deepcopy
from dataclasses import replace
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.manifest import SearchConfig
from pokezero.mcts_eval.policy_opponent_profile import (
    ARMS, MODES, _RawReplayPolicy, make_profile_decider, profile_root,
    root_seed, validate_selection,
)
from pokezero.mcts_eval.resolver import ContractError
from pokezero.mcts_eval.source_root_replay import SourceRootReplayError


MASK = (True, True) + (False,) * 7
CONFIG = SearchConfig(depth=2, sims=16, batch=8, worlds=4)
ROOT = SimpleNamespace(decision_id="a" * 64, seed=17, acting_player="p1",
                       turn_index=9, current_legal_action_mask=MASK)


def telemetry(arm, mode, opponent_seed):
    row = dict(root_action="move 2", fallbacks=0, prior_fallbacks=0,
               invalid_actions=0, model_evals=1, total_iterations=0,
               max_depth_reached=0, engine_mcts={})
    if arm == "raw_policy":
        row["raw_policy"] = dict(selector="deterministic_masked_argmax", model_evals=1,
            action_index=1, policy_distribution=[.1, .9] + [0.] * 7)
        return row
    row.update(total_iterations=64, max_depth_reached=2)
    engine = row["engine_mcts"] = dict(leaf_eval="model", worlds_constructed=4,
        worlds_searched=4, aggregated_choices_basis="full_budget", override={
            "root_allocation": dict(worlds=4, prior_authority=True, prior_cause=None,
                arms=[dict(action_index=index, visit_share=value, model_prior=value,
                           reported_prior=value) for index, value in enumerate((.1, .9))])})
    engine["joint_actions"] = dict(scope="per_native_invocation_without_belief_reweighting",
        native_invocations=[dict(world_seed=17, belief_multiplicity=4, witness=dict(
            schema="completed-joint-actions-v1", scope="single_native_tree",
            basis="finalized_backups_not_reservations", root_completed_traversals=64,
            root_distinct_pairs=2, root_side_one_options=2, root_side_two_options=1,
            root_side_one_visits=[6, 58], root_side_two_visits=[64],
            root_side_one_moves=["alpha", "beta"], root_side_two_moves=["reply"],
            root_side_one_report_order=[1, 0], root_side_two_report_order=[0],
            root_pairs=[dict(side_one_option=0, side_two_option=0, completed_visits=6),
                        dict(side_one_option=1, side_two_option=0, completed_visits=58)],
            tree_distinct_node_pairs=3, tree_completed_selections=80))])
    if arm == "own_policy_opponent_mcts":
        engine["policy_opponent"] = dict(mode="own_policy_callback", seed_root=opponent_seed,
            seed_derivation="sha256-domain-separated-v1", native_invocations=[
                dict(evaluations=1, provider_calls=2, samples=64)])
    if mode == "matched_deadline":
        engine["time_budget"] = dict(scope="whole_model_decision", requested_ms=1000,
            native_batch_guard_ms=64, native_invocations=[
                dict(requested_iterations=64, completed_iterations=64, remaining_iterations=0)])
    return row


def validate(row, arm="incumbent_mcts", mode="fixed_work"):
    return validate_selection(row, arm=arm, mode=mode, config=CONFIG, mask=MASK,
        opponent_seed=71, deadline_ms=1000, native_batch_guard_ms=64)


class ProfileContractTests(unittest.TestCase):
    def test_root_domains_are_stable_separate_and_strict(self):
        first = root_seed(71, "a" * 64, purpose="decision")
        self.assertEqual(first, root_seed(71, "a" * 64, purpose="decision"))
        self.assertEqual(len({first, root_seed(71, "a" * 64, purpose="opponent"),
            root_seed(72, "a" * 64, purpose="decision"),
            root_seed(71, "b" * 64, purpose="decision")}), 4)
        for seed in (None, True, -1, 2**64):
            with self.subTest(seed=seed), self.assertRaises(ContractError):
                root_seed(seed, "a" * 64, purpose="decision")
        for identity, purpose in (("A" * 64, "decision"), ("a" * 63, "decision"), ("a" * 64, "arm")):
            with self.assertRaises(ContractError):
                root_seed(71, identity, purpose=purpose)

    def test_factory_only_changes_the_registered_arm_and_deadline(self):
        with patch("pokezero.mcts_eval.policy_opponent_profile._LiveEngineTimingDecider") as engine:
            for mode in MODES:
                make_profile_decider(object(), "/showdown", arm="own_policy_opponent_mcts",
                    mode=mode, opponent_seed=71, deadline_ms=1000, native_batch_guard_ms=64)
                kwargs = engine.call_args.kwargs
                self.assertTrue(kwargs["policy_opponent"])
                self.assertTrue(kwargs["record_joint_actions"])
                self.assertTrue(kwargs["model_priors"])
                self.assertFalse(kwargs["use_opponent_priors"])
                self.assertEqual(kwargs["policy_opponent_seed"], 71)
                self.assertEqual(kwargs["model_world_workers"], 1)
                self.assertEqual(kwargs["model_decision_time_ms"], 1000 if mode == "matched_deadline" else None)
                self.assertEqual(kwargs["model_native_batch_guard_ms"], 64 if mode == "matched_deadline" else 0)
            make_profile_decider(object(), "/showdown", arm="incumbent_mcts",
                mode="fixed_work", opponent_seed=71, deadline_ms=1000, native_batch_guard_ms=64)
            self.assertFalse(engine.call_args.kwargs["policy_opponent"])
            self.assertIsNone(engine.call_args.kwargs["policy_opponent_seed"])

    def test_factory_refuses_invalid_registration_before_initialization(self):
        arguments = dict(arm="incumbent_mcts", mode="fixed_work", opponent_seed=71,
                         deadline_ms=1000, native_batch_guard_ms=64)
        for override in (dict(arm="other"), dict(mode="other"), dict(opponent_seed=True),
                         dict(deadline_ms=True), dict(native_batch_guard_ms=1000)):
            with patch("pokezero.mcts_eval.policy_opponent_profile._LiveEngineTimingDecider") as factory:
                with self.assertRaises(ContractError):
                    make_profile_decider(object(), "/showdown", **{**arguments, **override})
                factory.assert_not_called()

    def test_actual_arm_witnesses_are_required(self):
        for arm in ARMS:
            for mode in MODES:
                validate(telemetry(arm, mode, 71), arm, mode)
        row = telemetry("incumbent_mcts", "fixed_work", 71)
        row["total_iterations"] -= 1
        with self.assertRaisesRegex(ContractError, "actual native iterations"):
            validate(row)
        row = telemetry("own_policy_opponent_mcts", "fixed_work", 71)
        del row["engine_mcts"]["policy_opponent"]
        with self.assertRaisesRegex(ContractError, "opponent witness"):
            validate(row, "own_policy_opponent_mcts")

    def test_partial_deadline_is_not_mislabelled_exact_work(self):
        row = telemetry("incumbent_mcts", "matched_deadline", 71)
        row["total_iterations"] = 32
        row["engine_mcts"]["aggregated_choices_basis"] = "deadline_prefix"
        row["engine_mcts"]["time_budget"]["native_invocations"][0].update(
            completed_iterations=32, remaining_iterations=32)
        joint = row["engine_mcts"]["joint_actions"]["native_invocations"][0]["witness"]
        joint.update(root_completed_traversals=32, root_side_one_visits=[3, 29], root_side_two_visits=[32])
        for pair in joint["root_pairs"]:
            pair["completed_visits"] //= 2
        validate(row, mode="matched_deadline")
        with self.assertRaisesRegex(ContractError, "exact allocation"):
            validate(row)
        row["total_iterations"] -= 1
        with self.assertRaisesRegex(ContractError, "actual native iterations"):
            validate(row, mode="matched_deadline")

    def test_raw_search_leakage_invalid_mass_and_wrong_argmax_refuse(self):
        base = telemetry("raw_policy", "fixed_work", 71)
        for change in (dict(total_iterations=1), dict(engine_mcts={"leaf_eval": "model"}),
                       dict(model_evals=0), dict(max_depth_reached=1)):
            with self.assertRaises(ContractError):
                validate({**base, **change}, "raw_policy")
        for values in ([.1, .9, .1] + [0.] * 6, [float("nan")] * 9, [0.] * 9):
            row = deepcopy(base)
            row["raw_policy"]["policy_distribution"] = values
            with self.assertRaises(ContractError):
                validate(row, "raw_policy")
        row = deepcopy(base)
        row["raw_policy"]["action_index"] = 0
        with self.assertRaisesRegex(ContractError, "selected action"):
            validate(row, "raw_policy")

    def test_full_root_surface_and_sampler_work_cannot_be_faked(self):
        for mutate in (lambda row: row["engine_mcts"].update(worlds_constructed=3),
                       lambda row: row["engine_mcts"]["override"]["root_allocation"]["arms"].pop(),
                       lambda row: row["engine_mcts"]["override"]["root_allocation"]["arms"][0].update(model_prior=-.1),
                       lambda row: row.update(fallbacks=True)):
            row = telemetry("incumbent_mcts", "fixed_work", 71)
            mutate(row)
            with self.assertRaises(ContractError):
                validate(row)
        row = telemetry("own_policy_opponent_mcts", "fixed_work", 71)
        row["engine_mcts"]["policy_opponent"]["native_invocations"][0]["evaluations"] = 3
        with self.assertRaisesRegex(ContractError, "opponent work"):
            validate(row, "own_policy_opponent_mcts")
        # A collapsed tree's multiplicity remains four, but its compute ledger
        # counts 64 traversals once, not 256. Reserving, reweighting, duplicating,
        # omitting, or inventing pair evidence must fail closed.
        for mutation in (
            lambda engine: engine.pop("joint_actions"),
            lambda engine: engine["joint_actions"].update(scope="belief_weighted"),
            lambda engine: engine["joint_actions"]["native_invocations"][0].update(belief_multiplicity=True),
            lambda engine: engine["joint_actions"]["native_invocations"].append(
                deepcopy(engine["joint_actions"]["native_invocations"][0])),
        ):
            row = telemetry("incumbent_mcts", "fixed_work", 71)
            mutation(row["engine_mcts"])
            with self.assertRaises(ContractError):
                validate(row)
        for change in (dict(basis="collected_reservations"), dict(root_completed_traversals=256),
                       dict(root_distinct_pairs=True), dict(root_distinct_pairs=1),
                       dict(tree_completed_selections=1), dict(root_side_one_visits=[58, 6]),
                       dict(root_pairs=[]), dict(root_side_two_options=0),
                       dict(root_side_one_moves=["alpha"]), dict(root_side_one_report_order=[0, 1]),
                       dict(root_side_two_report_order=[True])):
            row = telemetry("incumbent_mcts", "fixed_work", 71)
            row["engine_mcts"]["joint_actions"]["native_invocations"][0]["witness"].update(change)
            with self.assertRaises(ContractError):
                validate(row)


class ProfileBoundaryTests(unittest.TestCase):
    def run_root(self, factory, clock, ordinal=0, on_row=None):
        with patch("pokezero.mcts_eval.policy_opponent_profile.source_bound_replay_prefix",
                   return_value=SimpleNamespace(public_action_rounds=("source-only",), repairs=())):
            return profile_root(ROOT, source_records=(), contract=object(), showdown_root="/showdown",
                config=CONFIG, seed=71, root_ordinal=ordinal, source_requested_players=("p1", "p2"),
                decider_factory=factory, clock=clock, on_row=on_row)

    def test_timer_excludes_preparation_and_rotates_arm_order(self):
        now = [0.]
        calls = []
        closed = []
        persisted = []
        def factory(contract, showdown, **kwargs):
            now[0] += 10  # model/export initialization, deliberately untimed
            arm, mode = kwargs["arm"], kwargs["mode"]
            calls.append((mode, arm))
            class Decider:
                def prepare_public_decision(self, record, config, **replay):
                    self_replay = replay
                    now[0] += 20  # source prefix replay, deliberately untimed
                    self_case.assertEqual(self_replay["public_action_rounds"], ("source-only",))
                    self_case.assertEqual(self_replay["source_requested_players"], ("p1", "p2"))
                    def decide():
                        now[0] += .25
                        return telemetry(arm, mode, kwargs["opponent_seed"])
                    return decide
                def close(self):
                    closed.append((mode, arm))
            return Decider()
        self_case = self
        def persist(mode, arm, row):
            self.assertEqual(closed[-1], (mode, arm))
            persisted.append((mode, arm, row["decision_wall_seconds"]))
        result = self.run_root(factory, lambda: now[0], ordinal=1, on_row=persist)
        self.assertEqual(result["state"], "COMPLETE")
        self.assertEqual(calls, [(MODES[0], arm) for arm in ARMS[1:] + ARMS[:1]] +
                         [(MODES[1], arm) for arm in ARMS[2:] + ARMS[:2]])
        self.assertEqual(calls, closed)
        self.assertEqual(persisted, [(mode, arm, .25) for mode, arm in calls])
        self.assertIn("PENDING", result["qualification"])
        for mode in MODES:
            self.assertEqual(set(result["modes"][mode]), set(ARMS))
            self.assertTrue(all(row["decision_wall_seconds"] == .25
                                for row in result["modes"][mode].values()))

    def test_refusal_keeps_other_arms_and_failed_telemetry_without_redraw(self):
        now = [0.]
        calls = []
        def factory(contract, showdown, **kwargs):
            arm, mode = kwargs["arm"], kwargs["mode"]
            calls.append((mode, arm))
            class Decider:
                def prepare_public_decision(self, record, config, **replay):
                    def decide():
                        now[0] += .125
                        row = telemetry(arm, mode, kwargs["opponent_seed"])
                        if arm == "own_policy_opponent_mcts":
                            row["prior_fallbacks"] = 1
                        return row
                    return decide
                def close(self): pass
            return Decider()
        result = self.run_root(factory, lambda: now[0])
        self.assertEqual(result["state"], "REFUSED")
        self.assertEqual(len(calls), 6)
        for mode in MODES:
            failed = result["modes"][mode]["own_policy_opponent_mcts"]
            self.assertEqual(failed["phase"], "validation")
            self.assertEqual(failed["telemetry"]["prior_fallbacks"], 1)
            self.assertEqual(failed["decision_wall_seconds"], .125)
            self.assertEqual(result["modes"][mode]["raw_policy"]["state"], "COMPLETE")

    def test_source_prefix_refusal_runs_no_candidate_and_keeps_identity(self):
        factory = unittest.mock.Mock()
        with patch("pokezero.mcts_eval.policy_opponent_profile.source_bound_replay_prefix",
                   side_effect=SourceRootReplayError("unresolved_public_event_for_non_source_player")):
            result = profile_root(ROOT, source_records=(), contract=object(), showdown_root="/showdown",
                config=CONFIG, seed=71, root_ordinal=0, source_requested_players=("p1", "p2"),
                decider_factory=factory)
        self.assertEqual(result["decision_id"], ROOT.decision_id)
        self.assertEqual(result["refusal"]["phase"], "source_prefix")
        self.assertEqual(result["modes"], {})
        factory.assert_not_called()

    def test_durable_sink_failure_stops_before_selecting_another_arm(self):
        calls = []
        def factory(contract, showdown, **kwargs):
            calls.append((kwargs["mode"], kwargs["arm"]))
            class Decider:
                def prepare_public_decision(self, record, config, **replay):
                    return lambda: telemetry(kwargs["arm"], kwargs["mode"], kwargs["opponent_seed"])
                def close(self): pass
            return Decider()
        def fail_sink(*_args):
            raise FileExistsError("existing receipt must not be overwritten")
        with self.assertRaises(FileExistsError):
            self.run_root(factory, lambda: 0., on_row=fail_sink)
        self.assertEqual(len(calls), 1)


class RawForwardTests(unittest.TestCase):
    def policy(self, **overrides):
        import torch
        from pokezero.neural_policy import (TransformerSoftmaxPolicy, TransformerTrainingConfig,
                                           TransformerTrainingResult)
        from test_policy_opponent import config
        cfg = config()
        class Model:
            calls = 0
            def eval(self): return self
            def __call__(self, **kwargs):
                self.calls += 1
                return SimpleNamespace(policy_logits=torch.tensor([[1., 2., 0., 0., -3., 0., 0., 0., 0.]]),
                                       value=torch.tensor([.37]))
        return TransformerSoftmaxPolicy(model=Model(), result=TransformerTrainingResult(
            model_config=cfg, training_config=TransformerTrainingConfig(window_size=1), epochs=()),
            **overrides)

    def observation(self):
        from test_engine_world import _dex
        from test_policy_opponent_view import VOCAB, view
        return view().observation(category_vocab=VOCAB, dex=_dex())

    def test_distribution_is_opt_in_and_uses_only_the_existing_forward(self):
        policy = self.policy()
        observation = self.observation()
        default = policy.select_action(observation, rng=random.Random(7))
        self.assertNotIn("policy_distribution", default.metadata)
        adapter = _RawReplayPolicy(policy)
        measured = adapter.select_action_with_context(SimpleNamespace(observation=observation), rng=random.Random(7))
        self.assertEqual(policy.model.calls, 2)  # one default + one measured
        self.assertEqual(measured.action_index, default.action_index)
        self.assertEqual(measured.value_estimate, default.value_estimate)
        self.assertEqual(adapter.stats.model_evals, 1)
        self.assertEqual(adapter.stats.total_iterations, 0)
        self.assertGreater(adapter.stats.model_wall_seconds, 0)
        self.assertAlmostEqual(sum(measured.metadata["raw_policy"]["policy_distribution"]), 1., places=6)

    def test_repeated_independent_roots_do_not_accumulate_model_history(self):
        policy = self.policy()
        adapter = _RawReplayPolicy(policy)
        observation = self.observation()
        decisions = [adapter.select_action_with_context(SimpleNamespace(observation=observation),
                                                       rng=random.Random(7)) for _ in range(3)]
        self.assertEqual(adapter.stats.model_evals, 3)
        self.assertEqual(policy.model.calls, 3)
        self.assertTrue(all(len(history) == 1 for history in policy._history_by_player.values()))
        self.assertEqual(decisions[0].metadata, decisions[-1].metadata)

    def test_raw_arm_refuses_nonlocal_or_noncanonical_selector(self):
        for kwargs in (dict(deterministic=False), dict(exploration_epsilon=.1),
                       dict(sampling_temperature=2.), dict(family_gated_selection=True),
                       dict(forward_fn=lambda tensors: None)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ContractError):
                _RawReplayPolicy(self.policy(**kwargs))
        policy = self.policy()
        policy.result = replace(policy.result, model_config=replace(policy.result.model_config, window_size=2))
        with self.assertRaisesRegex(ContractError, "one-snapshot"):
            _RawReplayPolicy(policy)
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            self.policy(record_policy_distribution="yes")


if __name__ == "__main__":
    unittest.main()
