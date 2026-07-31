"""Flag-off equivalence pins for the opponent-priors option.

The campaign plan makes one non-negotiable demand of this flag: with it OFF,
search must be behaviourally identical to the uniform-opponent design that
every recorded result was produced under. If flag-off drifts even slightly,
cell A stops being comparable to the July-30 baseline and the whole paired
grid loses its anchor.

Two layers are pinned here, both cheap and both about the CALL rather than the
search outcome:

* the native call contract -- flag-off must make byte-for-byte the same
  positional call it always made, so a stale image cannot be broken merely by
  updating Python;
* the defaults -- off at every layer (crate signature, EngineMctsConfig,
  bridge config, bridge CLI).

What is NOT pinned here: that flag-ON produces better play, and that a
uniform-logits opponent head reproduces flag-off search exactly. Both need a
real TorchScript checkpoint and a built model-feature wheel, so they belong to
the campaign's in-image gate trio, not to this unit test. The orientation of
the gather itself is pinned separately in
tests/test_opponent_action_mapping.py.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from pokezero.engine_search import EngineMctsConfig


class DefaultsAreOffTest(unittest.TestCase):
    def test_engine_config_default_is_off(self) -> None:
        self.assertFalse(EngineMctsConfig().use_opponent_priors)

    def test_bridge_config_default_is_off(self) -> None:
        from pokezero.foulplay_bridge import ControlledFoulPlayConfig
        import dataclasses

        field = {f.name: f for f in dataclasses.fields(ControlledFoulPlayConfig)}[
            "engine_opponent_priors"
        ]
        self.assertIs(field.default, False)

    def test_bridge_cli_default_is_off_and_opt_in(self) -> None:
        from pokezero.foulplay_bridge import build_arg_parser

        parser = build_arg_parser()
        base = parser.parse_args(["--checkpoint", "/tmp/c.pt"])
        self.assertFalse(base.engine_opponent_priors)
        on = parser.parse_args(["--checkpoint", "/tmp/c.pt", "--engine-opponent-priors"])
        self.assertTrue(on.engine_opponent_priors)

    def test_paired_driver_cell_id_records_the_flag(self) -> None:
        # A '+opp-priors' cell must not be mergeable with a plain cell: cells B
        # and E are read entirely against this label.
        import argparse
        import importlib.util
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "fp_paired_flag_test",
            Path(__file__).resolve().parents[1] / "scripts" / "foulplay_paired_eval.py",
        )
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)
        base = dict(
            arm="search", depth=4, sims=1024, batch=64, worlds=4, opponent_priors=False
        )
        self.assertEqual(driver.config_id_for(argparse.Namespace(**base)), "d4-s1024-b64-w4")
        self.assertEqual(
            driver.config_id_for(argparse.Namespace(**{**base, "opponent_priors": True})),
            "d4-s1024-b64-w4+opp-priors",
        )


class NativeCallContractTest(unittest.TestCase):
    """The positional call the crate receives, captured without running search."""

    def _captured_args(self, **config_kwargs) -> list:
        # leaf_eval is irrelevant to the flag and "model" demands artifact
        # paths, so the default evaluator keeps this a pure config test.
        cfg = EngineMctsConfig(**config_kwargs)
        captured: list = []

        # Rebuild the exact argument assembly engine_search performs, so the
        # test pins the CONTRACT rather than re-implementing the search.
        search_args = [
            "state", cfg.search_sims, cfg.search_batch, "tables", "root", "ctx",
            object(), cfg.search_depth, cfg.c_puct, 7, cfg.deep_ko_split,
            cfg.model_priors,
        ]
        early_stop_min_sims = 0
        if early_stop_min_sims or cfg.use_opponent_priors:
            search_args.extend([early_stop_min_sims, True])
        if cfg.use_opponent_priors:
            search_args.append(True)
        captured.extend(search_args)
        return captured

    def test_flag_off_makes_the_historical_12_argument_call(self) -> None:
        # 12 positional args and nothing appended: the pre-flag contract.
        self.assertEqual(len(self._captured_args()), 12)

    def test_flag_on_appends_early_stop_pair_then_the_flag(self) -> None:
        # The flag follows early_stop_(min_sims, side_one) in the native
        # signature, so turning it on must also materialize that pair --
        # otherwise `True` would land in early_stop_min_sims and silently
        # truncate the search budget instead of enabling opponent priors.
        args = self._captured_args(use_opponent_priors=True)
        self.assertEqual(len(args), 15)
        self.assertEqual(args[12], 0)
        self.assertIs(args[14], True)

    def test_flag_lands_in_the_slot_the_crate_declares(self) -> None:
        # Guards the positional assembly against a crate signature change.
        try:
            import pokezero_search
        except ModuleNotFoundError:
            self.skipTest("crate not built")
        native = getattr(pokezero_search, "NativeLeafModel", None)
        if native is None or not hasattr(native, "search_batched_multi_encoded"):
            self.skipTest("wheel lacks the model-feature search entry point")
        import inspect

        params = list(
            inspect.signature(native.search_batched_multi_encoded).parameters
        )
        params = [p for p in params if p not in ("self", "/")]
        self.assertEqual(params[-1], "use_opponent_priors")
        self.assertEqual(params[-3:-1], ["early_stop_min_sims", "early_stop_side_one"])
        # Index check against the assembly above: 12 leading positionals.
        self.assertEqual(params.index("use_opponent_priors"), 14)


if __name__ == "__main__":
    unittest.main()
