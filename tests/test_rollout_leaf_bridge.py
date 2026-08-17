"""CLI -> config -> EngineMctsConfig plumbing for the ROLLOUT-LEAF arbiter arm.

The arm itself, and its fidelity gate, live in `tests/test_rollout_model_priors.py`.
What is pinned HERE is the wiring that lets the arm be RUN IN A GAME at all --
before this, nothing carried `rollout_leaf_eval` from argv down into
`EngineMctsConfig`, so the seam existed and was unreachable from the bridge.

Four properties, one class each:

* DEFAULTS ARE OFF at every layer (crate config, bridge dataclass, bridge CLI,
  paired-driver CLI). This is what keeps every banked result's configuration
  byte-unchanged: an arm that defaulted on would silently reprice the leaves of
  every cell in the campaign.
* THE NATIVE CALL is byte-for-byte the pre-seam one with the arm off, and gains
  exactly the seam's seven positionals with it on.
* THE FLAG REACHES `EngineMctsConfig`, and the CPU-budget fence is REACHABLE
  through the bridge layer rather than only from the library.
* THE CELL ID RECORDS THE ARM, so a rollout cell cannot be merged with its
  value-head twin -- which would average the experiment into its own control.

Program rule observed throughout: a check that cannot read False certifies
nothing. Every guard below is exercised on an input that makes it FAIL as well
as one that makes it pass, and the failing input is named in each test.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from pokezero.engine_search import EngineMctsConfig, EngineMctsPolicy, native_search_args
from pokezero.foulplay_bridge import ControlledFoulPlayConfig, build_arg_parser

REPO_ROOT = Path(__file__).resolve().parents[1]

_SPEC = importlib.util.spec_from_file_location(
    "foulplay_paired_eval_rollout_test",
    REPO_ROOT / "scripts" / "foulplay_paired_eval.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_DRIVER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DRIVER)

#: The bridge's own defaults for the six rollout knobs. Named once so a drift in
#: any single layer shows up as a mismatch rather than being copied into the
#: expectation by hand.
BRIDGE_ROLLOUT_DEFAULTS = {
    "engine_rollout_leaf": False,
    "engine_rollout_count": 32,
    "engine_rollout_max_plies": 200,
    "engine_rollout_policy": "uniform",
    "engine_rollout_seed": 0,
    "engine_rollout_threads": 1,
    "engine_rollout_threads_cpu_budget_ack": False,
}

FOLD = object()

_MODEL_CONFIG = {
    "leaf_eval": "model",
    "model_path": "m.pt",
    "checkpoint_path": "c.pt",
    "tables_path": "t.json",
}


class DefaultsAreOffTest(unittest.TestCase):
    """OFF at every layer, so no recorded result's configuration moves."""

    def test_engine_config_default_is_off(self) -> None:
        cfg = EngineMctsConfig()
        self.assertFalse(cfg.rollout_leaf_eval)
        # The knobs too: a default R or cap that drifted would change what an
        # arm-ON cell means without changing any flag anyone typed.
        self.assertEqual(cfg.rollout_count, 32)
        self.assertEqual(cfg.rollout_max_plies, 200)
        self.assertEqual(cfg.rollout_policy, "uniform")
        self.assertEqual(cfg.rollout_seed, 0)
        self.assertEqual(cfg.rollout_threads, 1)
        self.assertFalse(cfg.rollout_threads_cpu_budget_ack)

    def test_bridge_config_defaults_are_off(self) -> None:
        fields = {f.name: f for f in dataclasses.fields(ControlledFoulPlayConfig)}
        for name, expected in BRIDGE_ROLLOUT_DEFAULTS.items():
            with self.subTest(field=name):
                self.assertIn(
                    name,
                    fields,
                    f"{name} is missing from ControlledFoulPlayConfig, so the CLI "
                    "flag cannot reach the search",
                )
                self.assertEqual(fields[name].default, expected)

    def test_bridge_cli_default_is_off_and_the_flag_turns_it_on(self) -> None:
        """FAILING INPUT: the second half. If the parser ignored the flag, or if
        `store_true` were mistyped as a value option, the ON assertion reads
        False -- so this cannot pass merely by everything defaulting off.
        """
        parser = build_arg_parser()
        base = parser.parse_args(["--checkpoint", "/tmp/c.pt"])
        for name, expected in BRIDGE_ROLLOUT_DEFAULTS.items():
            with self.subTest(flag=name):
                self.assertEqual(getattr(base, name), expected)
        on = parser.parse_args(
            [
                "--checkpoint", "/tmp/c.pt",
                "--engine-rollout-leaf",
                "--engine-rollout-count", "8",
                "--engine-rollout-max-plies", "400",
                "--engine-rollout-policy", "uniform",
                "--engine-rollout-seed", "11",
                "--engine-rollout-threads", "3",
                "--engine-rollout-threads-cpu-budget-ack",
            ]
        )
        self.assertTrue(on.engine_rollout_leaf)
        self.assertEqual(on.engine_rollout_count, 8)
        self.assertEqual(on.engine_rollout_max_plies, 400)
        self.assertEqual(on.engine_rollout_seed, 11)
        self.assertEqual(on.engine_rollout_threads, 3)
        self.assertTrue(on.engine_rollout_threads_cpu_budget_ack)

    def test_paired_driver_cli_default_is_off_and_the_flag_turns_it_on(self) -> None:
        parser = _DRIVER.build_parser()
        required = [
            "--checkpoint", "/tmp/c.pt", "--showdown-root", "/tmp/sd",
            "--arm", "search", "--seed-start", "1", "--pairs", "1",
            "--out", "/tmp/shard.json",
        ]
        base = parser.parse_args(required)
        self.assertFalse(base.engine_rollout_leaf)
        self.assertEqual(base.engine_rollout_count, 32)
        self.assertEqual(base.engine_rollout_max_plies, 200)
        self.assertEqual(base.engine_rollout_policy, "uniform")
        self.assertEqual(base.engine_rollout_seed, 0)
        self.assertEqual(base.engine_rollout_threads, 1)
        self.assertFalse(base.engine_rollout_threads_cpu_budget_ack)
        on = parser.parse_args(
            required + ["--engine-rollout-leaf", "--engine-rollout-count", "8"]
        )
        self.assertTrue(on.engine_rollout_leaf)
        self.assertEqual(on.engine_rollout_count, 8)


class NativeCallContractTest(unittest.TestCase):
    """The positional list the crate receives, from the REAL assembly.

    Calls `engine_search.native_search_args` rather than rebuilding the list, for
    the reason the opponent-priors version of this class records: a test that
    rebuilds the contract asserts against its own copy and passes in both worlds.
    """

    def _args(self, **config_kwargs) -> list:
        cfg = EngineMctsConfig(**{**_MODEL_CONFIG, **config_kwargs})
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
            early_stop_min_sims=0,
            sims=None,
        )

    def test_arm_off_makes_the_pre_seam_call_byte_for_byte(self) -> None:
        """Flag-off equivalence, and it must hold with the KNOBS SET.

        The bridge now passes all seven rollout values into `EngineMctsConfig`
        unconditionally, so this is the assertion that makes that safe: with
        `rollout_leaf_eval` off, no value of R/cap/policy/seed/threads may reach
        the crate or shift any earlier slot.

        FAILING INPUT: the `assertNotEqual` below. If `native_search_args` had
        appended the seam unconditionally, or the widening cascade materialized a
        slot it should not, the two lists would already differ here.
        """
        baseline = self._args()
        with_knobs = self._args(
            rollout_count=8,
            rollout_max_plies=400,
            rollout_seed=99,
            rollout_threads=1,
        )
        self.assertEqual(
            baseline,
            with_knobs,
            "with the arm OFF the rollout knobs must not reach the crate at all",
        )
        # ... and the same assembly DOES move when the arm is on, so the equality
        # above is a property of the flag and not of a helper that ignores config.
        self.assertNotEqual(baseline, self._args(rollout_leaf_eval=True))

    def test_the_seam_appends_exactly_the_seven_positionals_in_order(self) -> None:
        """The seam's own slots, and the four it must materialize behind it.

        Asking for the seam must ALSO materialize the four conditional slots in
        front of it (plus the pair that `early_stop_min_sims` gates), because the
        crate reads positionally: 12 base + 5 materialized + 7 seam = 24.

        FAILING INPUT: the exact index of `"rollout"`. If any earlier slot were
        dropped from the widening chain, `"rollout"` slides down into
        `fpu_reduction` (15) or `arm_priors` (16) and every assertion below moves
        -- which is precisely the cascade the assembly's comment warns about, and
        the reason this is pinned by index rather than by `in`.
        """
        off = self._args()
        on = self._args(
            rollout_leaf_eval=True,
            rollout_count=8,
            rollout_max_plies=400,
            rollout_policy="uniform",
            rollout_seed=11,
            rollout_threads=1,
        )
        self.assertEqual(len(off), 12, "the pre-seam call is the 12 base positionals")
        self.assertEqual(len(on), 24)
        # The 12 base positionals are untouched by the seam.
        self.assertEqual(on[:12], off)
        # The five slots materialized to REACH the seam, each at its config value
        # and never a placeholder: writing True into `arm_priors` to reach the seam
        # would silently switch the arm-name column on for every rollout cell, and
        # a truthy string in `use_opponent_priors` would turn the opponent head on
        # by accident.
        self.assertEqual(
            on[12:17],
            [0, True, False, None, False],
            "early_stop_min_sims, early_stop_side_one, use_opponent_priors, "
            "fpu_reduction, arm_priors",
        )
        # THE SEAM, outermost and last.
        self.assertEqual(on[17:], ["rollout", 8, 400, "uniform", 11, 1, False])
        self.assertEqual(on[17], "rollout", "the mode must not land in an earlier slot")

    def test_a_bad_policy_name_is_refused_rather_than_substituted(self) -> None:
        """FAILING INPUT: 'uniform' constructs fine, 'greedy' must not."""
        EngineMctsConfig(**_MODEL_CONFIG, rollout_leaf_eval=True, rollout_policy="uniform")
        with self.assertRaises(ValueError) as caught:
            EngineMctsConfig(
                **_MODEL_CONFIG, rollout_leaf_eval=True, rollout_policy="greedy"
            )
        self.assertIn("rollout_policy", str(caught.exception))


def _bridge_config(**overrides) -> ControlledFoulPlayConfig:
    base = dict(
        checkpoint=Path("/tmp/ckpt.pt"),
        showdown_root=Path("/tmp/showdown"),
        policy_mode="engine-mcts",
        engine_model_path=Path("/tmp/model_ts.pt"),
        engine_tables_path=Path("/tmp/tables.json"),
    )
    base.update(overrides)
    return ControlledFoulPlayConfig(**base)


class FlagReachesEngineConfigTest(unittest.TestCase):
    """`_build_policy` must hand the arm to `EngineMctsConfig`.

    Everything heavy is patched away -- the dex, the candidate-set source and the
    policy class itself -- because the property under test is the ARGUMENT, not
    the search. What is deliberately NOT patched is `EngineMctsConfig`: the real
    dataclass is constructed, so its validator runs and the refusal tests below
    are refusals of the actual production object.
    """

    def _captured_config(self, **overrides) -> EngineMctsConfig:
        captured: dict[str, EngineMctsConfig] = {}

        class _FakePolicy:
            def __init__(self, **kwargs):
                captured["config"] = kwargs["config"]

        with mock.patch("pokezero.engine_search.EngineMctsPolicy", _FakePolicy), \
             mock.patch("pokezero.foulplay_bridge.load_showdown_dex_cached",
                        return_value=object()), \
             mock.patch("pokezero.foulplay_bridge.load_gen3_randbat_source_cached",
                        return_value=object()):
            from pokezero.foulplay_bridge import _build_policy

            _build_policy(
                config=_bridge_config(**overrides),
                model=object(),
                result=object(),
                value_model=object(),
                value_result=object(),
                env_config=object(),
                rollout_config=object(),
                policy_id="pid",
            )
        return captured["config"]

    def test_the_arm_and_all_six_knobs_reach_engine_mcts_config(self) -> None:
        """FAILING INPUT: the arm-OFF half. Both halves are asserted from the
        same builder, so a plumb that hardcoded either value would fail one.
        """
        off = self._captured_config()
        self.assertFalse(off.rollout_leaf_eval)

        on = self._captured_config(
            engine_rollout_leaf=True,
            engine_rollout_count=8,
            engine_rollout_max_plies=400,
            engine_rollout_policy="uniform",
            engine_rollout_seed=11,
            engine_rollout_threads=1,
        )
        self.assertTrue(on.rollout_leaf_eval)
        self.assertEqual(on.rollout_count, 8)
        self.assertEqual(on.rollout_max_plies, 400)
        self.assertEqual(on.rollout_policy, "uniform")
        self.assertEqual(on.rollout_seed, 11)
        self.assertEqual(on.rollout_threads, 1)

    def test_the_cpu_budget_ack_refusal_fires_through_the_bridge_layer(self) -> None:
        """The fence must be REACHABLE from the CLI, not merely present in the library.

        The hazard is stated on `EngineMctsConfig.rollout_threads`: the opponent
        is time-budgeted and thinks concurrently on the same host, so cores taken
        here weaken it in the direction that flatters this arm. If the bridge
        dropped the ack on the floor, threads>1 would be accepted unfenced -- or,
        worse, an ack the operator DID pass would be lost and the fence would
        refuse a legitimate run.

        FAILING INPUT: both directions are asserted. Without the ack the build
        must raise; with it, the same threads value must construct.
        """
        with self.assertRaises(ValueError) as caught:
            self._captured_config(
                engine_rollout_leaf=True,
                engine_rollout_threads=4,
                engine_rollout_threads_cpu_budget_ack=False,
            )
        message = str(caught.exception)
        self.assertIn("rollout_threads_cpu_budget_ack", message)
        # The real values must appear in the refusal, so the operator sees what
        # was actually configured rather than a generic sentence.
        self.assertIn("rollout_threads=4", message)

        acked = self._captured_config(
            engine_rollout_leaf=True,
            engine_rollout_threads=4,
            engine_rollout_threads_cpu_budget_ack=True,
        )
        self.assertEqual(acked.rollout_threads, 4)
        self.assertTrue(acked.rollout_threads_cpu_budget_ack)

    def test_threads_above_one_are_inert_and_unfenced_with_the_arm_off(self) -> None:
        """The fence's SCOPE, stated because the bridge passes threads always.

        `EngineMctsConfig`'s threads/ack refusal sits inside
        `if self.rollout_leaf_eval`, so `rollout_threads > 1` with the arm OFF is
        accepted and inert -- nothing on the value-head path reads it. That is the
        correct scope (threads only spend cores when rollouts run), but it is worth
        pinning rather than assuming, because the bridge now forwards the value
        unconditionally and a reader could reasonably expect the fence to fire.

        FAILING INPUT: the arm-ON assertion in the test above is the same
        threads/ack pair and DOES refuse, so this pair of tests brackets the fence
        instead of merely asserting one side of it.
        """
        off = self._captured_config(
            engine_rollout_threads=4,
            engine_rollout_threads_cpu_budget_ack=False,
        )
        self.assertFalse(off.rollout_leaf_eval)
        self.assertEqual(off.rollout_threads, 4)

    def test_the_arm_is_refused_outside_the_model_leaf(self) -> None:
        """The seam lives only on the encoded model path.

        FAILING INPUT: `leaf_eval='model'` (the bridge's own value) constructs;
        `hp_fraction_crate` must refuse rather than ignore the flag, because a
        cell banked as "oracle-leaf with model priors" that actually ran the
        handcrafted leaf is an ABSENT input reading as a wrong one.
        """
        self.assertTrue(self._captured_config(engine_rollout_leaf=True).rollout_leaf_eval)
        with self.assertRaises(ValueError) as caught:
            EngineMctsConfig(leaf_eval="hp_fraction_crate", rollout_leaf_eval=True)
        self.assertIn("requires leaf_eval='model'", str(caught.exception))


def _cell_args(**overrides) -> argparse.Namespace:
    base = dict(
        arm="search", depth=4, sims=1024, batch=64, worlds=4,
        opponent_priors=False, engine_fpu_reduction=None, engine_c_puct=None,
        engine_oracle_belief=False, engine_early_stop=False,
        engine_depth_min=None, engine_worlds_min=None,
        engine_early_stop_min_sims=None,
        opponent_policy_mode="foul-play", opponent_engine_depth=None,
        opponent_engine_sims=None,
        engine_rollout_leaf=False, engine_rollout_count=32,
        engine_rollout_max_plies=200, engine_rollout_policy="uniform",
        checkpoint="/c/k0.pt",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class CellIdentityTest(unittest.TestCase):
    """A rollout cell must never be mergeable with its value-head twin."""

    def test_the_cell_id_records_the_arm(self) -> None:
        """FAILING INPUT: the arm-off id. It must stay byte-identical to the id
        this cell had before the arm existed, so every banked shard still pools;
        if the fragment were unconditional, this first assertion fails.
        """
        self.assertEqual(_DRIVER.config_id_for(_cell_args()), "d4-s1024-b64-w4@k0")
        self.assertEqual(
            _DRIVER.config_id_for(
                _cell_args(engine_rollout_leaf=True, engine_rollout_count=8,
                           engine_rollout_max_plies=200)
            ),
            "d4-s1024-b64-w4+rollout8p200@k0",
        )

    def test_r_and_the_cap_split_the_cell(self) -> None:
        """Three pricers, three ids. R is the estimator's variance and the cap
        decides whether the leaf is an oracle at all, so none of these may pool.
        """
        ids = {
            _DRIVER.config_id_for(
                _cell_args(engine_rollout_leaf=True, engine_rollout_count=r,
                           engine_rollout_max_plies=cap)
            )
            for r, cap in ((8, 200), (32, 200), (8, 20))
        }
        self.assertEqual(len(ids), 3, f"R/cap must split the cell, got {ids}")

    def test_it_composes_with_the_other_semantic_knobs(self) -> None:
        self.assertEqual(
            _DRIVER.config_id_for(
                _cell_args(opponent_priors=True, engine_rollout_leaf=True,
                           engine_rollout_count=8, engine_rollout_max_plies=200)
            ),
            "d4-s1024-b64-w4+opp-priors+rollout8p200@k0",
        )

    def test_a_namespace_predating_the_arm_raises_rather_than_pools(self) -> None:
        """The direct-attribute-access contract, on the sharpest knob it has.

        A rollout shard misfiled into the value-head control does not merely
        corrupt a delta -- it averages the arm into what it is measured against,
        which inverts the program's conclusion. So an argv that never saw the flag
        must raise here rather than be handed the control's id.

        FAILING INPUT: a Namespace WITH the knob renders an id (asserted above),
        so this is not a test that everything raises.
        """
        stale = _cell_args()
        del stale.engine_rollout_leaf
        with self.assertRaises(AttributeError):
            _DRIVER.config_id_for(stale)

    def test_the_builder_refuses_the_arm_without_r_or_the_cap(self) -> None:
        """FAILING INPUT: both present renders; either missing must refuse rather
        than substitute this module's default, which is how the driver's builder
        and the report's reference builder drift apart silently.
        """
        self.assertTrue(
            _DRIVER.search_config_id(
                depth=4, sims=8, batch=1, worlds=1, tag="k0",
                rollout_leaf=True, rollout_count=8, rollout_max_plies=200,
            )
        )
        for missing in ({"rollout_count": None}, {"rollout_max_plies": None}):
            with self.subTest(missing=missing):
                with self.assertRaises(ValueError):
                    _DRIVER.search_config_id(
                        depth=4, sims=8, batch=1, worlds=1, tag="k0",
                        rollout_leaf=True,
                        **{"rollout_count": 8, "rollout_max_plies": 200, **missing},
                    )

    def test_the_merger_builds_the_identical_id(self) -> None:
        """Lockstep with `foulplay_power_report.cid_of`.

        The shared docstring records that these two builders drifting is a SILENT
        failure -- the reference matches no shard, so the non-starvation rule
        reports clean while firing on nothing -- and that it has already happened
        once. FAILING INPUT: the id built from the campaign-cell dict must equal
        the id built from the Namespace; a fragment added to one only fails here.
        """
        source = (REPO_ROOT / "scripts" / "foulplay_power_report.py").read_text()
        for kwarg in ("rollout_leaf=", "rollout_count=", "rollout_max_plies=",
                      "rollout_policy="):
            with self.subTest(kwarg=kwarg):
                self.assertIn(
                    kwarg,
                    source,
                    "foulplay_power_report.cid_of does not pass "
                    f"{kwarg} so a rollout cell's reference id would match no shard",
                )
        cell = dict(depth=4, sims=1024, batch=64, worlds=4, tag="k0",
                    rollout_leaf=True, rollout_count=8, rollout_max_plies=200)
        self.assertEqual(
            _DRIVER.search_config_id(**cell),
            _DRIVER.config_id_for(
                _cell_args(engine_rollout_leaf=True, engine_rollout_count=8,
                           engine_rollout_max_plies=200)
            ),
        )


class ChildArgvTest(unittest.TestCase):
    """What the paired driver actually tells the bridge."""

    def _argv(self, **overrides) -> list[str]:
        from test_foulplay_paired_eval import args as _full_args  # noqa: PLC0415

        return _DRIVER.bridge_argv(_full_args(**overrides), seat="p1")

    def test_arm_off_leaves_the_child_argv_byte_identical(self) -> None:
        """FAILING INPUT: the arm-on comparison. An unconditional forward would
        make the first assertion fail, and a forward that never happened would
        make the second.
        """
        off = self._argv()
        self.assertNotIn("--engine-rollout-leaf", off)
        for flag in ("--engine-rollout-count", "--engine-rollout-max-plies",
                     "--engine-rollout-policy", "--engine-rollout-seed",
                     "--engine-rollout-threads"):
            self.assertNotIn(flag, off)
        self.assertIn("--engine-rollout-leaf", self._argv(engine_rollout_leaf=True))

    def test_the_arm_reaches_the_child_with_every_knob(self) -> None:
        argv = self._argv(
            engine_rollout_leaf=True,
            engine_rollout_count=8,
            engine_rollout_max_plies=400,
            engine_rollout_policy="uniform",
            engine_rollout_seed=11,
            engine_rollout_threads=1,
        )
        self.assertIn("--engine-rollout-leaf", argv)
        for flag, value in (
            ("--engine-rollout-count", "8"),
            ("--engine-rollout-max-plies", "400"),
            ("--engine-rollout-policy", "uniform"),
            ("--engine-rollout-seed", "11"),
            ("--engine-rollout-threads", "1"),
        ):
            with self.subTest(flag=flag):
                self.assertEqual(argv[argv.index(flag) + 1], value)
        # And the child's parser accepts every one of them, which is the half a
        # string comparison cannot cover: a forwarded flag the bridge does not
        # define would fail the run, not this assertion, without it.
        parsed = build_arg_parser().parse_args(
            ["--checkpoint", "/tmp/c.pt"]
            + [a for a in argv if a.startswith("--engine-rollout")
               or (argv.index(a) > 0 and argv[argv.index(a) - 1].startswith("--engine-rollout"))]
        )
        self.assertTrue(parsed.engine_rollout_leaf)
        self.assertEqual(parsed.engine_rollout_count, 8)
        self.assertEqual(parsed.engine_rollout_max_plies, 400)

    def test_the_ack_travels_only_when_set(self) -> None:
        """FAILING INPUT: the without-ack case must NOT carry the flag, or the
        fence becomes unreachable through this driver and threads>1 is accepted
        unfenced on every cell.
        """
        without = self._argv(engine_rollout_leaf=True, engine_rollout_threads=4)
        self.assertNotIn("--engine-rollout-threads-cpu-budget-ack", without)
        withack = self._argv(
            engine_rollout_leaf=True, engine_rollout_threads=4,
            engine_rollout_threads_cpu_budget_ack=True,
        )
        self.assertIn("--engine-rollout-threads-cpu-budget-ack", withack)


class RuntimeWitnessTest(unittest.TestCase):
    """The telemetry that says the seam RAN, as opposed to having been requested.

    This program has already paid for the lesson that an image can build clean
    and be wrong at runtime, and the seam's positionals are appended as the
    OUTERMOST slots -- so an extension that predates them ignores what it does
    not recognise and produces a search that looks configured and is not. These
    fields are how a run answers that question about itself.
    """

    def test_absent_is_not_zero_on_the_fallback_fraction(self) -> None:
        """`0.0` is the ORACLE claim, so an un-engaged seam must not render it.

        `rollout_fallback_fraction == 0.0` means every rollout reached a terminal,
        i.e. the leaf is a real oracle rather than a blend of an oracle and a
        handcrafted cap value. A shard that ran no rollouts at all would produce
        exactly that number from a guarded `0/0`.

        FAILING INPUT: the second half. With rollouts recorded, a genuine 0.0 must
        still be reportable -- so this is not a test that the field is always None.
        """
        from pokezero.engine_search import EngineMctsStats

        idle = EngineMctsStats().to_dict()
        self.assertIsNone(idle["rollout_fallback_fraction"])
        self.assertIsNone(idle["rollout_terminal_fraction"])
        self.assertIsNone(idle["rollout_mean_plies"])
        self.assertEqual(idle["rollout_leaf_modes"], {})

        oracle = EngineMctsStats()
        oracle.rollouts_run = 800
        oracle.rollout_terminal_hits = 800
        oracle.rollout_leaf_modes = Counter({"rollout": 1})
        payload = oracle.to_dict()
        self.assertEqual(payload["rollout_fallback_fraction"], 0.0)
        self.assertEqual(payload["rollout_terminal_fraction"], 1.0)

        blend = EngineMctsStats()
        blend.rollouts_run = 1000
        blend.rollout_terminal_hits = 900
        blend.rollout_cap_hits = 100
        self.assertEqual(blend.to_dict()["rollout_fallback_fraction"], 0.1)

    def test_the_per_decision_row_reads_the_report_not_the_record(self) -> None:
        """THE REGRESSION GUARD, and it is a regression that actually happened.

        `world_runs` holds RECORD dicts with the crate's report nested under
        `record["report"]`. A first version of this helper read the rollout keys
        off the record's own top level: every unit expectation about the shard
        aggregate still passed (those are absorbed from `report` at a different
        seam), and the per-decision block silently vanished from all 65 decisions
        of a real game. It was found by capturing a played game's metadata, not by
        a test -- so the shape is pinned here.

        FAILING INPUT: `flat` below is exactly the shape the bug assumed. It must
        yield None, or this guard would pass against the broken reader too.
        """
        nested = [{"side_key": "side_one", "report": {
            "rollout_leaf_mode": "rollout",
            "rollouts_run": 16,
            "rollout_terminal_hits": 14,
            "rollout_cap_hits": 2,
            "rollout_dead_ends": 0,
            "rollout_encode_skipped": 3,
            "leaves_priced": 2,
        }}]
        row = EngineMctsPolicy._rollout_decision_row(nested)
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["rollout_leaf_modes"], ["rollout"])
        self.assertEqual(row["rollouts_run"], 16)
        self.assertEqual(row["rollout_encode_skipped"], 3)
        self.assertEqual(row["rollout_fallback_fraction"], 0.125)
        self.assertEqual(row["world_records"], 1)

        # The keys spread on the RECORD instead of the report -- the bug's shape.
        flat = [{"side_key": "side_one", "rollout_leaf_mode": "rollout",
                 "rollouts_run": 16, "report": {"model_evals": 1}}]
        self.assertIsNone(
            EngineMctsPolicy._rollout_decision_row(flat),
            "the helper must read report[...] and not the record's top level",
        )

    def test_no_seam_means_no_per_decision_block_at_all(self) -> None:
        """A value-head decision's metadata must be byte for byte what it was."""
        self.assertIsNone(
            EngineMctsPolicy._rollout_decision_row(
                [{"side_key": "side_one", "report": {"model_evals": 4}}]
            )
        )

    def test_a_decision_whose_worlds_disagree_renders_both_modes(self) -> None:
        """FAILING INPUT: a single-mode decision renders one entry, so this does
        not pass merely by listing everything. A decision that ran the seam on one
        world and the value head on another must not read as either regime.
        """
        rows = [
            {"report": {"rollout_leaf_mode": "rollout", "rollouts_run": 8,
                        "rollout_terminal_hits": 8}},
            {"report": {"rollout_leaf_mode": "model_value", "rollouts_run": 0}},
        ]
        row = EngineMctsPolicy._rollout_decision_row(rows)
        assert row is not None
        self.assertEqual(row["rollout_leaf_modes"], ["model_value", "rollout"])
        self.assertEqual(row["world_records"], 2)


if __name__ == "__main__":
    unittest.main()
