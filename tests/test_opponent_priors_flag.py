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

from pokezero.engine_search import EngineMctsConfig, native_search_args


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

    def test_fpu_defaults_to_the_legacy_flat_urgency_everywhere(self) -> None:
        # Same standing as the opponent flag above, one layer per assertion:
        # a default that drifted on ONE of these three would make the campaign's
        # "flag-off is the historical search" claim false at whichever layer the
        # shard actually configures.
        import dataclasses

        from pokezero.foulplay_bridge import ControlledFoulPlayConfig, build_arg_parser

        self.assertIsNone(EngineMctsConfig().fpu_reduction)
        field = {f.name: f for f in dataclasses.fields(ControlledFoulPlayConfig)}[
            "engine_fpu_reduction"
        ]
        self.assertIsNone(field.default)
        parser = build_arg_parser()
        self.assertIsNone(
            parser.parse_args(["--checkpoint", "/tmp/c.pt"]).engine_fpu_reduction
        )
        self.assertEqual(
            parser.parse_args(
                ["--checkpoint", "/tmp/c.pt", "--engine-fpu-reduction", "0.2"]
            ).engine_fpu_reduction,
            0.2,
        )

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
            arm="search", depth=4, sims=1024, batch=64, worlds=4,
            opponent_priors=False, checkpoint="/c/k0.pt",
        )
        self.assertEqual(
            driver.config_id_for(argparse.Namespace(**base)), "d4-s1024-b64-w4@k0"
        )
        self.assertEqual(
            driver.config_id_for(argparse.Namespace(**{**base, "opponent_priors": True})),
            "d4-s1024-b64-w4+opp-priors@k0",
        )


# A sentinel for the FoldState handle: asserted by IDENTITY below, so the
# contract pins WHICH object reaches slot 6 and not merely that something did.
FOLD = object()


class NativeCallContractTest(unittest.TestCase):
    """The positional call the crate receives, captured without running search.

    These call `engine_search.native_search_args` -- the REAL assembly the
    search uses. An earlier version of this class rebuilt the list inside the
    test and asserted against its own copy, which passed in both worlds:
    review deleted `config.model_priors` from the production assembly (which
    also shifts `use_opponent_priors` two slots into `early_stop_min_sims`,
    truncating the search budget) and this file still reported 6 passed,
    1 skipped.
    """

    def _captured_args(
        self,
        early_stop_min_sims: int = 0,
        sims: int | None = None,
        **config_kwargs,
    ) -> list:
        # leaf_eval is irrelevant to the flag and "model" demands artifact
        # paths, so the default evaluator keeps this a pure config test.
        cfg = EngineMctsConfig(**config_kwargs)
        record = {
            "state_str": "state",
            "ctx_json": "ctx",
            "seed": 7,
            "side_key": "side_one",
        }
        return native_search_args(
            cfg,
            record,
            tables_json="tables",
            root_inputs="root",
            rust_fold=FOLD,
            early_stop_min_sims=early_stop_min_sims,
            sims=sims,
        )

    def test_the_twelve_leading_positionals_are_the_pre_flag_contract(self) -> None:
        # Not just the COUNT: a dropped argument shifts everything after it,
        # and a length check alone cannot see a reordering.
        cfg = EngineMctsConfig()
        args = self._captured_args()
        self.assertEqual(
            args[:12],
            [
                "state", cfg.search_sims, cfg.search_batch, "tables", "root",
                "ctx", FOLD, cfg.search_depth, cfg.c_puct, 7,
                cfg.deep_ko_split, cfg.model_priors,
            ],
        )
        self.assertIs(args[6], FOLD, "the fold handle must reach slot 6 itself")

    def test_the_sims_override_replaces_slot_one_and_nothing_else(self) -> None:
        # #1009 collapses duplicate belief worlds into one deeper search by
        # passing multiplicity x the per-world budget. That value reaches the
        # crate through slot 1 of THIS list. The collapse counters
        # (`worlds_collapsed`, the weighted samples) are pinned in
        # tests/test_engine_search.py, but they all still read correctly if the
        # override is dropped on the way to the native call -- every collapsed
        # world would simply search at the unscaled budget, silently, and the
        # depth ladder would read a number bought with N times less compute.
        cfg = EngineMctsConfig()
        default = self._captured_args()
        self.assertEqual(default[1], cfg.search_sims)

        scaled = self._captured_args(sims=cfg.search_sims * 3)
        self.assertEqual(scaled[1], cfg.search_sims * 3)
        self.assertEqual(
            scaled[:1] + scaled[2:],
            default[:1] + default[2:],
            "the override must move slot 1 and no other argument",
        )

    def test_early_stop_alone_appends_the_pair_and_not_the_flag(self) -> None:
        args = self._captured_args(early_stop_min_sims=64)
        self.assertEqual(len(args), 14)
        self.assertEqual(args[12], 64)
        self.assertIs(args[13], True)

    def test_side_two_reports_side_one_false_in_the_early_stop_pair(self) -> None:
        cfg = EngineMctsConfig()
        record = {
            "state_str": "state", "ctx_json": "ctx", "seed": 7,
            "side_key": "side_two",
        }
        args = native_search_args(
            cfg, record, tables_json="t", root_inputs="r", rust_fold=FOLD,
            early_stop_min_sims=64,
        )
        self.assertIs(args[13], False)

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

    def test_fpu_alone_materializes_the_two_flags_it_sits_behind(self) -> None:
        # `fpu_reduction` is one slot past `use_opponent_priors`, which is two
        # slots past the early-stop pair. Appending the float without the two
        # arguments in front of it would land 0.2 in `early_stop_min_sims` --
        # a 0-sim floor, silently -- or in `use_opponent_priors`, turning the
        # opponent head on in a cell whose whole point is that it is off.
        args = self._captured_args(fpu_reduction=0.2)
        self.assertEqual(len(args), 16)
        self.assertEqual(args[12], 0)
        self.assertIs(args[13], True)
        self.assertIs(args[14], False, "the opponent flag must stay off")
        self.assertEqual(args[15], 0.2)

    def test_both_flags_on_appends_both_in_signature_order(self) -> None:
        args = self._captured_args(use_opponent_priors=True, fpu_reduction=0.3)
        self.assertEqual(len(args), 16)
        self.assertIs(args[14], True)
        self.assertEqual(args[15], 0.3)

    def test_fpu_none_leaves_every_other_arm_byte_for_byte(self) -> None:
        # The bit-identity claim at the call-assembly layer: the default config
        # and an explicitly-None one must produce the SAME list, and it must be
        # the historical 12.
        self.assertEqual(self._captured_args(fpu_reduction=None), self._captured_args())
        self.assertEqual(len(self._captured_args(fpu_reduction=None)), 12)

    def test_an_out_of_range_fpu_reduction_is_refused_by_the_config(self) -> None:
        for bad in (-0.1, 1.5):
            with self.assertRaises(ValueError):
                EngineMctsConfig(fpu_reduction=bad)

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
        self.assertEqual(params[-1], "fpu_reduction")
        self.assertEqual(
            params[-4:-1],
            ["early_stop_min_sims", "early_stop_side_one", "use_opponent_priors"],
        )
        # Index check against the assembly above: 12 leading positionals.
        self.assertEqual(params.index("use_opponent_priors"), 14)
        self.assertEqual(params.index("fpu_reduction"), 15)


if __name__ == "__main__":
    unittest.main()
