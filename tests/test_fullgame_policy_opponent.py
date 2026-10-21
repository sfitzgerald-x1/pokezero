"""Full-game own-policy wiring only; no model, native search or game is run."""
from pathlib import Path
import unittest
from unittest.mock import patch

from pokezero import foulplay_bridge as bridge


def config(**changes):
    values = dict(policy_mode="engine-mcts", opponent_policy_mode="raw", showdown_root=Path("showdown"),
                  checkpoint=Path("champion.pt"), engine_model_path=Path("model_ts.pt"),
                  engine_tables_path=Path("tables.json"), engine_sims=256,
                  engine_worlds=4, engine_batch=16, engine_depth=6,
                  allow_search_fallback=False, engine_policy_opponent=True,
                  engine_policy_opponent_seed=3829160000)
    values.update(changes)
    return bridge.ControlledFoulPlayConfig(**values)


class FullgamePolicyOpponentTests(unittest.TestCase):
    def test_default_is_off_and_unseeded(self):
        c = bridge.ControlledFoulPlayConfig(checkpoint=Path("champion.pt"), showdown_root=Path("showdown"))
        self.assertFalse(c.engine_policy_opponent)
        self.assertIsNone(c.engine_policy_opponent_seed)

    def test_cli_exposes_explicit_fields(self):
        args = bridge.build_arg_parser().parse_args([
            "--checkpoint", "champion.pt", "--policy-mode", "engine-mcts",
            "--engine-model-path", "model_ts.pt", "--engine-tables-path", "tables.json",
            "--engine-policy-opponent", "--engine-policy-opponent-seed", "12",
            "--no-search-fallback"])
        c = bridge._config_from_args(args)
        self.assertTrue(c.engine_policy_opponent)
        self.assertEqual(c.engine_policy_opponent_seed, 12)

    def test_incompatible_or_ambiguous_configs_fail_before_collection(self):
        invalid = [dict(engine_policy_opponent=1), dict(engine_policy_opponent_seed=None),
                   dict(engine_policy_opponent_seed=True), dict(engine_policy_opponent_seed=-1),
                   dict(engine_policy_opponent_seed=2**64), dict(engine_opponent_priors=True),
                   dict(engine_model_priors=False), dict(allow_search_fallback=True),
                   dict(engine_early_stop=True), dict(engine_worlds_min=1),
                   dict(policy_mode="raw", opponent_policy_mode="foul-play"),
                   dict(engine_policy_opponent=False)]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                config(**changes)

    def test_raw_opponent_clears_search_only_sampling_fields(self):
        c = bridge._opponent_seat_config(config())
        self.assertEqual(c.policy_mode, "raw")
        self.assertFalse(c.engine_policy_opponent)
        self.assertIsNone(c.engine_policy_opponent_seed)

    def test_constructor_receives_real_engine_configuration(self):
        # Keep the actual EngineMctsConfig validation, mock only model/source creation.
        with patch("pokezero.engine_search.EngineMctsPolicy") as factory, \
                patch.object(bridge, "load_showdown_dex_cached", return_value=object()), \
                patch.object(bridge, "load_gen3_randbat_source_cached", return_value=object()):
            bridge._build_policy(config=config(), model=None, result=None,
                value_model=None, value_result=None, env_config=None,
                rollout_config=None, policy_id="champion")
        built = factory.call_args.kwargs
        self.assertTrue(built["config"].policy_opponent)
        self.assertEqual(built["config"].policy_opponent_seed, 3829160000)
        self.assertFalse(built["config"].use_opponent_priors)
        self.assertTrue(built["config"].strict_fallbacks)
        self.assertIn("-policy-opponent", built["policy_id"])

    def test_saved_configuration_distinguishes_own_policy_from_auxiliary_head(self):
        result = bridge.ControlledFoulPlayBenchmarkResult(config=config(), policy_id="champion", games=())
        saved = result.to_dict()["engine_mcts"]
        self.assertTrue(saved["policy_opponent"])
        self.assertEqual(saved["policy_opponent_seed"], 3829160000)
        self.assertFalse(saved["opponent_priors"])


if __name__ == "__main__":
    unittest.main()
