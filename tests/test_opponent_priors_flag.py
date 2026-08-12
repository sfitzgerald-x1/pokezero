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

from pathlib import Path
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
            opponent_priors=False, engine_fpu_reduction=None, engine_c_puct=None,
            engine_oracle_belief=False,
            # config_id_for reads its knobs by DIRECT attribute access, on purpose:
            # a Namespace predating a knob must raise rather than be handed the
            # default id and merged into the control. So adding a knob to the
            # driver legitimately breaks a hand-built Namespace here, and the fix
            # is to name the knob -- not to soften the driver to getattr.
            engine_early_stop=False, engine_early_stop_min_sims=None,
            checkpoint="/c/k0.pt",
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


#: The override-telemetry tests are the only ones here that cannot use the
#: default evaluator: the config REFUSES that flag outside `leaf_eval='model'`,
#: because the measurement needs root priors no other leaf evaluator computes and
#: would otherwise report a silent zero.
_MODEL_CONFIG = {
    "leaf_eval": "model",
    "model_path": "m.pt",
    "checkpoint_path": "c.pt",
    "tables_path": "t.json",
}

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

    def test_override_telemetry_alone_materializes_the_three_slots_it_sits_behind(
        self,
    ) -> None:
        # `override_telemetry` is one slot past `fpu_reduction`, which is one past
        # `use_opponent_priors`, which is two past the early-stop pair. The
        # dangerous shift is one slot: `True` landing in `fpu_reduction` is
        # ACCEPTED by the crate's validator (1.0 is in range) and changes
        # selection, so a pure-telemetry flag would silently become a search
        # change -- and the flag-off differential would still pass, because it
        # never sets the flag.
        args = self._captured_args(**_MODEL_CONFIG, override_telemetry=True)
        self.assertEqual(len(args), 17)
        self.assertEqual(args[12], 0, "early_stop_min_sims at its default")
        self.assertIs(args[13], True, "early_stop_side_one for a side_one record")
        self.assertIs(args[14], False, "the opponent flag must stay off")
        self.assertIsNone(args[15], "fpu_reduction must be its own default, not True")
        self.assertIs(args[16], True)

    def test_override_telemetry_does_not_disturb_the_knobs_in_front_of_it(self) -> None:
        # Every earlier slot keeps the value it has without the flag, so a cell
        # measured with the telemetry on is the same search as the cell without.
        with_flag = self._captured_args(
            **_MODEL_CONFIG,
            override_telemetry=True,
            use_opponent_priors=True,
            fpu_reduction=0.3,
        )
        without = self._captured_args(
            **_MODEL_CONFIG, use_opponent_priors=True, fpu_reduction=0.3
        )
        self.assertEqual(with_flag[:16], without)
        self.assertEqual(len(with_flag), 17)

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
        self.assertEqual(params[-1], "arm_priors")
        self.assertEqual(
            params[-5:-1],
            [
                "early_stop_min_sims",
                "early_stop_side_one",
                "use_opponent_priors",
                "fpu_reduction",
            ],
        )
        # Index check against the assembly above: 12 leading positionals.
        self.assertEqual(params.index("use_opponent_priors"), 14)
        self.assertEqual(params.index("fpu_reduction"), 15)
        self.assertEqual(params.index("arm_priors"), 16)


class OverrideTelemetryIsObservationalTest(unittest.TestCase):
    """`override_telemetry` must not change the SEARCH, only what it reports.

    This is the claim `scripts/foulplay_paired_eval.search_config_id` rests on
    when it keeps the flag OUT of config_id: telemetry-on and telemetry-off are
    one cell, so a shard that measured an override rate pools with a banked shard
    that did not. If the flag perturbed selection, plan §2 would be measuring a
    different engine than every other stage while reporting the same cell id --
    a wrong number, not an error.

    Turning it on materializes four positionals it sits behind, and THAT is the
    only way it could leak into the search. So the pins here are:

    * every materialized slot carries the crate's OWN DECLARED DEFAULT, read off
      the installed wheel rather than hardcoded -- so materializing the slot is a
      no-op against the call the crate would have made for itself;
    * the single exception, `early_stop_side_one` on a side_two record, is
      report-only in the crate. `model.rs` guards the in-loop early-stop check
      with `early_stop_min_sims > 0` (which is 0 here), and the one unguarded
      `root_visit_lock` call after the loop assigns only
      `early_stop_leader_visits` / `early_stop_runner_up_visits`, two REPORT
      fields that `engine_search` deliberately does not absorb (see the
      `root_visit_gap_sum` comment, which names this exact seat hazard). It also
      flips the report's `early_stop_side` string.
    * `arm_priors` itself reaches only `multiply_report_json` ->
      `stats_to_json(.., with_prior)`, which appends one `"prior"` key per root
      arm entry and changes nothing else. Pinned crate-side by
      `tree.rs::arm_priors_only_adds_a_reported_column`.

    NOT pinned here, and honestly: an end-to-end on/off replay of one real search
    input. That needs a TorchScript leaf artifact and a real checkpoint, which is
    the in-image gate's job -- the same boundary the opponent-priors flag's own
    on/off replay was run at (see the note in scripts/foulplay_paired_eval.py).
    """

    _SLOTS = (
        (12, "early_stop_min_sims"),
        (13, "early_stop_side_one"),
        (14, "use_opponent_priors"),
        (15, "fpu_reduction"),
        (16, "arm_priors"),
    )

    def _crate_defaults(self) -> dict[str, object]:
        try:
            import pokezero_search
        except ModuleNotFoundError:
            self.skipTest("crate not built")
        native = getattr(pokezero_search, "NativeLeafModel", None)
        if native is None or not hasattr(native, "search_batched_multi_encoded"):
            self.skipTest("wheel lacks the model-feature search entry point")
        import inspect

        parameters = inspect.signature(native.search_batched_multi_encoded).parameters
        return {name: parameters[name].default for _, name in self._SLOTS}

    def _args(self, side_key: str, **config_kwargs) -> list:
        cfg = EngineMctsConfig(**_MODEL_CONFIG, **config_kwargs)
        record = {
            "state_str": "state", "ctx_json": "ctx", "seed": 7, "side_key": side_key,
        }
        return native_search_args(
            cfg, record, tables_json="tables", root_inputs="root", rust_fold=FOLD,
            early_stop_min_sims=0,
        )

    def test_the_materialized_slots_carry_the_crates_own_defaults(self) -> None:
        defaults = self._crate_defaults()
        args = self._args("side_one", override_telemetry=True)
        for index, name in self._SLOTS:
            if name == "arm_priors":
                # The one slot that is SUPPOSED to differ from the default.
                self.assertIs(args[index], True, name)
                self.assertIs(defaults[name], False, "crate default must stay off")
                continue
            self.assertEqual(
                args[index], defaults[name],
                f"slot {index} ({name}) must equal the crate's own default, or "
                "materializing it to reach arm_priors changes the search",
            )

    def test_the_only_slot_that_can_differ_is_the_report_only_early_stop_side(
        self,
    ) -> None:
        # A side_two record passes early_stop_side_one=False where the crate's
        # default is True. Named here rather than left to be discovered: with
        # early_stop_min_sims at 0 the crate cannot early-stop at all, so the
        # value reaches only the report's `early_stop_side` /
        # `early_stop_leader_visits` / `early_stop_runner_up_visits` fields.
        defaults = self._crate_defaults()
        args = self._args("side_two", override_telemetry=True)
        differing = [
            name for index, name in self._SLOTS
            if name != "arm_priors" and args[index] != defaults[name]
        ]
        self.assertEqual(differing, ["early_stop_side_one"])
        self.assertEqual(args[12], 0, "no early-stop floor, so the side is inert")

    def test_engine_search_does_not_absorb_the_report_fields_that_slot_moves(
        self,
    ) -> None:
        # The complement of the test above: if a future revision started reading
        # the crate's leader/runner-up pair, the seat-dependent value WOULD reach
        # a number the campaign publishes, and telemetry would stop being free.
        # `engine_search` derives its visit gap from the acting seat's own entries
        # for exactly this reason.
        source = (
            Path(__file__).resolve().parents[1] / "src" / "pokezero" / "engine_search.py"
        ).read_text(encoding="utf-8")
        for field in ("early_stop_leader_visits", "early_stop_runner_up_visits"):
            reads = [
                line for line in source.splitlines()
                if field in line and not line.lstrip().startswith("#")
            ]
            self.assertEqual(reads, [], f"{field} is now read: {reads}")
        # Positive control for the grep above: a field the module DOES read must
        # be found by the identical query, or an empty result means nothing.
        control = [
            line for line in source.splitlines()
            if "early_stopped" in line and not line.lstrip().startswith("#")
        ]
        self.assertTrue(control, "the control field was not found; the query is broken")

    def test_every_earlier_slot_is_byte_for_byte_the_flag_off_call(self) -> None:
        # The whole-list form of the claim, at both seats and against a tuned
        # cell: the telemetry-on call is the telemetry-off call plus its own
        # trailing True, and nothing in front of it moves.
        for side_key in ("side_one", "side_two"):
            for knobs in ({}, {"use_opponent_priors": True, "fpu_reduction": 0.3}):
                with self.subTest(side_key=side_key, knobs=sorted(knobs)):
                    on = self._args(side_key, override_telemetry=True, **knobs)
                    off = self._args(side_key, **knobs)
                    padded = off + [None] * (17 - len(off))
                    padded[12] = 0
                    padded[13] = side_key == "side_one"
                    padded[14] = bool(knobs.get("use_opponent_priors", False))
                    padded[15] = knobs.get("fpu_reduction")
                    padded[16] = True
                    self.assertEqual(on, padded)


if __name__ == "__main__":
    unittest.main()
