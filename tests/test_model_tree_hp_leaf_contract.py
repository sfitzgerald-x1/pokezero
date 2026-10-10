"""Fail-closed valuation-only HP boundary; native execution is tested separately."""
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

from pokezero.engine_search import (EngineMctsConfig, EngineSearchWitnessError,
    native_search_args, require_model_hp_leaf_witness, validate_native_hp_leaf_witness)
from pokezero.mcts_eval.search_over_raw_adapters import _incumbent_runtime


def config(**changes):
    return EngineMctsConfig(**{**dict(leaf_eval="model", model_path="model", checkpoint_path="champion",
        tables_path="tables", model_priors=True, strict_fallbacks=True, search_sims=32,
        search_batch=16, search_depth=6, early_stop=False), **changes})


def witness():
    return dict(schema="pokezero.model-tree-hp-leaf.v1", mode="hp_fraction",
        value_frame="side_one_absolute", model_forwards_retained=True,
        scope="per_native_invocation_without_belief_reweighting", hp_leaf_rows_priced=12,
        native_invocations=[dict(world_seed=7, belief_multiplicity=4,
            completed_iterations=16, model_evals=13, hp_leaf_rows_priced=12)])


class HpLeafContractTests(unittest.TestCase):
    def test_mode_off_preserves_native_call_and_hp_materializes_existing_slots(self):
        record = dict(state_str="state", ctx_json="{}", seed=7, side_key="side_one")
        fold = object()
        def args(cfg, **kw):
            return native_search_args(cfg, record, tables_json="tables", root_inputs="root",
                rust_fold=fold, early_stop_min_sims=0, **kw)
        baseline = args(config())
        self.assertEqual(len(baseline), 12)
        hp = args(config(model_leaf_override="hp_fraction"))
        self.assertEqual(hp[:12], baseline)
        self.assertEqual(hp[12:18], [0, True, False, None, False, "hp_fraction"])
        self.assertEqual(len(hp), 24)
        timed = args(config(model_leaf_override="hp_fraction"), time_budget_ms=800)
        self.assertEqual(timed[:-1], hp)
        self.assertEqual(timed[-1], 800)

    def test_hp_requires_fixed_strict_model_tree_with_champion_priors(self):
        config(model_leaf_override="hp_fraction")
        for changes in (dict(leaf_eval="hp_fraction"), dict(model_priors=False),
                dict(strict_fallbacks=False), dict(early_stop=True), dict(depth_min=1),
                dict(worlds_min=1), dict(rollout_leaf_eval=True), dict(rollout_leaf_shadow=True),
                dict(policy_opponent=True, policy_opponent_seed=1), dict(model_leaf_override="raw_rollout")):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "model_leaf_override"):
                config(**{**dict(model_leaf_override="hp_fraction"), **changes})

    def test_native_work_cannot_be_missing_reflected_or_silently_uniform_rollout(self):
        good = dict(model_leaf_override="hp_fraction", hp_leaf_value_frame="side_one_absolute",
            hp_leaf_model_forwards_retained=True, hp_leaf_rows_priced=12, model_evals=13, iterations=16)
        validate_native_hp_leaf_witness(good)
        for changes in (dict(model_leaf_override="model"), dict(hp_leaf_value_frame="self_relative"),
                dict(hp_leaf_model_forwards_retained=False), dict(hp_leaf_rows_priced=True),
                dict(hp_leaf_rows_priced=-1), dict(hp_leaf_rows_priced=14), dict(rollout_leaf_mode="rollout")):
            with self.subTest(changes=changes), self.assertRaises(EngineSearchWitnessError):
                validate_native_hp_leaf_witness({**good, **changes})
        for key in good:
            incomplete = dict(good)
            del incomplete[key]
            with self.subTest(key=key), self.assertRaises(EngineSearchWitnessError):
                validate_native_hp_leaf_witness(incomplete)

    def test_invocations_are_unweighted_compute_not_duplicate_belief_rows(self):
        metadata = {"engine_mcts": {"model_leaf_override": witness()}}
        require_model_hp_leaf_witness(metadata, model_leaf_override="hp_fraction")
        changed = deepcopy(metadata)
        changed["engine_mcts"]["model_leaf_override"]["hp_leaf_rows_priced"] *= 4
        with self.assertRaisesRegex(EngineSearchWitnessError, "conserve"):
            require_model_hp_leaf_witness(changed, model_leaf_override="hp_fraction")

    def test_ordinary_model_cannot_accept_an_hp_witness_or_hp_accept_no_witness(self):
        require_model_hp_leaf_witness({"engine_mcts": {}}, model_leaf_override=None)
        with self.assertRaisesRegex(EngineSearchWitnessError, "unrequested"):
            require_model_hp_leaf_witness({"engine_mcts": {"model_leaf_override": witness()}},
                model_leaf_override=None)
        with self.assertRaisesRegex(EngineSearchWitnessError, "missing"):
            require_model_hp_leaf_witness({"engine_mcts": {}}, model_leaf_override="hp_fraction")

    def test_malformed_invocation_and_negative_totals_refuse(self):
        for changes in (dict(world_seed=-1), dict(world_seed=2**64), dict(belief_multiplicity=0),
                dict(completed_iterations=True), dict(model_evals=11), dict(hp_leaf_rows_priced=-1)):
            changed = witness()
            changed["native_invocations"][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(EngineSearchWitnessError):
                require_model_hp_leaf_witness({"engine_mcts": {"model_leaf_override": changed}},
                    model_leaf_override="hp_fraction")

    def test_runtime_does_not_switch_the_incumbent_tree_or_allocation(self):
        decider = Mock()
        decider._policy_for.return_value = "native"
        with patch("pokezero.mcts_eval.policy_opponent_profile.make_profile_decider", return_value=decider) as make:
            result = _incumbent_runtime("contract", "showdown", 3., leaf="hp_fraction")
        self.assertEqual(make.call_args.kwargs["model_leaf_override"], "hp_fraction")
        self.assertEqual(make.call_args.kwargs["arm"], "incumbent_mcts")
        self.assertEqual(make.call_args.kwargs["deadline_ms"], 3000)
        cfg = result[2]
        self.assertEqual((cfg.depth, cfg.sims, cfg.batch, cfg.worlds), (6, 4096, 16, 4))

    def test_unknown_leaf_runtime_fails_before_model_load(self):
        with patch("pokezero.mcts_eval.policy_opponent_profile.make_profile_decider") as make:
            with self.assertRaisesRegex(ValueError, "implemented model-tree"):
                _incumbent_runtime(None, "showdown", 10., leaf="unknown")
        make.assert_not_called()


if __name__ == "__main__":
    unittest.main()
