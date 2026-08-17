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
from typing import Any
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

    def test_the_driver_and_the_helper_agree_on_the_fragment(self) -> None:
        """`config_id_for` must render what `search_config_id` renders.

        An earlier revision of the opponent fragment added it to the helper but not
        to `config_id_for`, the only production caller -- so the helper worked in
        isolation while every real shard still rendered the pooled id, and the
        tests were green because they called the helper directly. This calls BOTH.
        """
        cell = dict(depth=4, sims=1024, batch=64, worlds=4, tag="k0",
                    rollout_leaf=True, rollout_count=8, rollout_max_plies=200)
        self.assertEqual(
            _DRIVER.search_config_id(**cell),
            _DRIVER.config_id_for(
                _cell_args(engine_rollout_leaf=True, engine_rollout_count=8,
                           engine_rollout_max_plies=200)
            ),
        )


class RawArmRefusesTheArmTest(unittest.TestCase):
    """`--arm raw --engine-rollout-leaf`, on the PAIRED DRIVER.

    The bridge refuses `engine_rollout_leaf` outside `policy_mode='engine-mcts'`
    (`ArgvReachesTheConfigTest.test_the_arm_is_refused_where_it_would_reach_nothing`)
    -- but the paired driver forwards the whole rollout block inside
    `if args.arm != "raw"`, so the raw arm sends the child `--policy-mode raw` and
    NONE of the seam's flags. The bridge's refusal is therefore unreachable
    through this driver, and what survived was the driver's own shard body writing
    `"rollout_leaf": true` into a `raw@<tag>` shard.

    That shard is the DENOMINATOR of every paired delta. A false witness there is
    worse than one on the treatment arm: it makes the comparison silently
    self-referential rather than merely noisy. Refused in `config_id_for`, beside
    the sibling opponent refusal, before any game runs.
    """

    def test_the_raw_arm_refuses_the_rollout_leaf(self) -> None:
        """FAILING INPUT: `--arm raw --engine-rollout-leaf`.

        Demonstrated as the two states of ONE assertion pair, so neither a refusal
        that never fires nor one that fires on every raw cell passes:

        * `arm=raw, engine_rollout_leaf=True`  -> SystemExit
        * `arm=raw, engine_rollout_leaf=False` -> `raw@k0`, byte-unchanged
        """
        with self.assertRaises(SystemExit) as caught:
            _DRIVER.config_id_for(_cell_args(arm="raw", engine_rollout_leaf=True,
                                             engine_rollout_count=8,
                                             engine_rollout_max_plies=200))
        message = str(caught.exception)
        self.assertIn("--arm raw", message)
        self.assertIn("--engine-rollout-leaf", message)
        # The ordinary raw anchor is untouched: every banked raw shard must still
        # render the id it has always rendered.
        self.assertEqual(
            _DRIVER.config_id_for(_cell_args(arm="raw", engine_rollout_leaf=False)),
            "raw@k0",
        )

    def test_the_refusal_precedes_the_shard_and_the_child(self) -> None:
        """It must fire before anything is written, not be caught downstream.

        `config_id_for` is called at the top of `main` -- before the model config
        is loaded, before `run_seat` spawns a child, and before the shard body is
        built -- which is why the refusal belongs there. Pinned on the argv
        builder too: the child argv for this invocation carries `--policy-mode raw`
        and no rollout flag at all, which is precisely why the bridge's refusal
        cannot see it.
        """
        from test_foulplay_paired_eval import args as _full_args  # noqa: PLC0415

        raw_with_arm = _full_args(arm="raw", engine_rollout_leaf=True,
                                  engine_rollout_count=8,
                                  engine_rollout_max_plies=200)
        argv = _DRIVER.bridge_argv(raw_with_arm, seat="p1")
        self.assertIn("raw", argv)
        self.assertEqual(
            [flag for flag in argv if flag.startswith("--engine-rollout")],
            [],
            "the driver sends the raw arm no rollout flag, so the bridge's refusal "
            "is structurally unreachable and this driver must refuse itself",
        )
        # ... and `main` refuses before it can do anything else. The parser is real,
        # so this is the actual command line a campaign would have queued.
        with self.assertRaises(SystemExit):
            _DRIVER.config_id_for(_DRIVER.build_parser().parse_args([
                "--checkpoint", "/c/k0.pt", "--showdown-root", "/s",
                "--arm", "raw", "--seed-start", "1", "--pairs", "2",
                "--depth", "4", "--sims", "64", "--batch", "16", "--worlds", "1",
                "--checkpoint-tag", "k0", "--out", "/o.json",
                "--engine-rollout-leaf",
            ]))


class ShardBodyMatchesTheCellIdTest(unittest.TestCase):
    """The driver's shard-body witness, and the guard that reads it against the id.

    Two records of the same fact leave every shard -- the cell **id** the merger
    keys on, and the **body** a reader of a single shard sees -- built by
    different code from different inputs. A disagreement is individually plausible
    on each side, so it is refused rather than reconciled.

    The previous revision had no test on the body at all, and no guard that could
    see a body/id disagreement: the body was an inline literal inside `main`,
    reachable only by running a whole paired campaign.
    """

    @staticmethod
    def _args(**overrides):
        from test_foulplay_paired_eval import args as _full_args  # noqa: PLC0415

        return _full_args(**overrides)

    def test_the_body_records_every_knob_including_the_two_the_id_omits(self) -> None:
        """FAILING INPUT: the seed and the thread count are set to NON-defaults.

        Both are deliberately outside the id (a replicate axis and a knob proven
        value-invariant crate-side), so this body is their only record and a
        rollout draw cannot be reproduced without them. A body that dropped them,
        or echoed the dataclass default instead of the argv, fails here.
        """
        body = _DRIVER.rollout_body_fields(
            self._args(engine_rollout_leaf=True, engine_rollout_count=8,
                       engine_rollout_max_plies=200, engine_rollout_policy="uniform",
                       engine_rollout_seed=1234, engine_rollout_threads=1),
            "d4-s1024-b64-w4+rollout8p200@k0",
        )
        self.assertEqual(body, {
            "rollout_leaf": True,
            "rollout_count": 8,
            "rollout_max_plies": 200,
            "rollout_policy": "uniform",
            "rollout_seed": 1234,
            "rollout_threads": 1,
        })
        # Arm off: the body still records the knobs (they are inert), and says so.
        off = _DRIVER.rollout_body_fields(self._args(), "d4-s1024-b64-w4@k0")
        self.assertFalse(off["rollout_leaf"])

    def test_a_body_that_contradicts_the_id_is_refused_in_both_directions(self) -> None:
        """THE GUARD, on the two disagreements it exists to catch.

        FAILING INPUT, twice over -- and each direction is a different corruption:

        * body ON / id without the fragment: the shard pools into the value-head
          CONTROL while claiming to be the arm. This is the exact shape
          `--arm raw --engine-rollout-leaf` produced (`raw@k0` + `rollout_leaf:
          true`).
        * body OFF / id WITH the fragment: the reverse misfile, averaging a
          value-head game into the arm's own cell.

        The agreeing pairs in `test_the_body_records_every_knob...` above are what
        stops this passing for a guard that refuses everything.
        """
        with self.assertRaises(SystemExit) as body_on_id_off:
            _DRIVER.rollout_body_fields(
                self._args(engine_rollout_leaf=True, engine_rollout_count=8,
                           engine_rollout_max_plies=200),
                "raw@k0",
            )
        self.assertIn("rollout_leaf=True", str(body_on_id_off.exception))
        self.assertIn("raw@k0", str(body_on_id_off.exception))

        with self.assertRaises(SystemExit) as body_off_id_on:
            _DRIVER.rollout_body_fields(
                self._args(engine_rollout_leaf=False),
                "d4-s1024-b64-w4+rollout8p200@k0",
            )
        self.assertIn("rollout_leaf=False", str(body_off_id_on.exception))

    def test_the_fragment_is_matched_not_substring_searched(self) -> None:
        """`+rollout` alone must not satisfy the id side.

        A future fragment named `+rolloutpolicy` or a cell tag containing the word
        would make a substring check read True for an id that carries no arm, and
        the guard would then certify the disagreement it exists to catch. Matched
        as `+rollout<R>p<cap>`.

        FAILING INPUT: an id carrying the literal `+rollout` and no R/cap, with the
        body ON -- which must still refuse.
        """
        with self.assertRaises(SystemExit):
            _DRIVER.rollout_body_fields(
                self._args(engine_rollout_leaf=True, engine_rollout_count=8,
                           engine_rollout_max_plies=200),
                "d4-s1024-b64-w4+rolloutish@k0",
            )
        # The real renderings, including the policy suffix, are accepted.
        for cid in ("d4-s1024-b64-w4+rollout8p200@k0",
                    "d4-s1024-b64-w4+opp-priors+rollout32p20-greedy@k1"):
            with self.subTest(cid=cid):
                self.assertTrue(
                    _DRIVER.rollout_body_fields(
                        self._args(engine_rollout_leaf=True, engine_rollout_count=8,
                                   engine_rollout_max_plies=200),
                        cid,
                    )["rollout_leaf"]
                )

    def test_main_builds_the_shard_body_through_this_helper(self) -> None:
        """THE CALL SITE. A guard that is never called certifies nothing.

        `main` cannot be driven in a unit test (it fingerprints the engine build,
        loads a checkpoint's model config and spawns a child per seat), so the
        call is pinned structurally: the shard body's rollout keys must come from
        `rollout_body_fields(args, config_id)` -- splatted, unconditional, and
        passed BOTH records -- and no rollout key may be re-introduced as a
        literal beside it.

        FAILING INPUT: replacing the splat with the six inline literals it
        replaced, or dropping `config_id` from the call, fails this.
        """
        import ast  # noqa: PLC0415
        import inspect  # noqa: PLC0415

        source = inspect.getsource(_DRIVER.main)
        tree = ast.parse(source.lstrip() if source.startswith(" ") else source)
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "rollout_body_fields"
        ]
        self.assertEqual(len(calls), 1, "main must build the body through the guard once")
        self.assertEqual(
            [ast.unparse(argument) for argument in calls[0].args],
            ["args", "config_id"],
            "the guard must be handed BOTH of the shard's own records -- this run's "
            "`args` and this shard's `config_id`. A literal id in either position "
            "makes the comparison self-satisfying, which is the failure mode the "
            "guard exists to prevent one layer down",
        )
        # Splatted into the report dict, so the attachment carries no logic.
        splats = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Dict) and any(key is None for key in node.keys)
        ]
        self.assertTrue(
            any(
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "rollout_body_fields"
                for node in splats
                for key, value in zip(node.keys, node.values)
                if key is None
            ),
            "the report dict must splat rollout_body_fields(...) directly",
        )
        # And the literals it replaced are gone, so there is exactly one writer.
        for key in ("rollout_leaf", "rollout_count", "rollout_max_plies",
                    "rollout_policy", "rollout_seed", "rollout_threads"):
            with self.subTest(key=key):
                self.assertNotIn(
                    f'"{key}":', source,
                    f"{key} must come from rollout_body_fields, not from a second "
                    "literal in main that the guard never sees",
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
        # R=4 DELIBERATELY, because the fixture's --engine-worlds is also 4: any
        # value-locating logic that searches argv by VALUE instead of by position
        # matches the wrong token here. R=8 alone would hide that.
        for count in (8, 4):
            with self.subTest(R=count):
                self._assert_every_knob_reaches_the_child(count)

    def _assert_every_knob_reaches_the_child(self, count: int) -> None:
        argv = self._argv(
            engine_rollout_leaf=True,
            engine_rollout_count=count,
            engine_rollout_max_plies=400,
            engine_rollout_policy="uniform",
            engine_rollout_seed=11,
            engine_rollout_threads=1,
        )
        self.assertIn("--engine-rollout-leaf", argv)
        for flag, value in (
            ("--engine-rollout-count", str(count)),
            ("--engine-rollout-max-plies", "400"),
            ("--engine-rollout-policy", "uniform"),
            ("--engine-rollout-seed", "11"),
            ("--engine-rollout-threads", "1"),
        ):
            with self.subTest(flag=flag):
                # By POSITION of the FLAG, which is unambiguous; the hazard is
                # locating a VALUE by `argv.index(value)`.
                self.assertEqual(argv[argv.index(flag) + 1], value)
        # And the child's parser accepts every one of them, which is the half a
        # string comparison cannot cover: a forwarded flag the bridge does not
        # define would fail the run, not this assertion, without it.
        # Sliced BY POSITION, never by `argv.index(value)`: index() finds the
        # FIRST occurrence of a value, so the moment a rollout knob's value
        # collides with an earlier token (set R to 4 and it matches
        # `--engine-worlds 4`) the filter silently drops it and the parse fails
        # for a reason that has nothing to do with the property under test.
        rollout_slice: list[str] = []
        for i, token in enumerate(argv):
            if token.startswith("--engine-rollout"):
                rollout_slice.append(token)
            elif (
                i
                and argv[i - 1].startswith("--engine-rollout")
                and not token.startswith("--")
            ):
                # Only a VALUE, so a boolean flag such as
                # --engine-rollout-threads-cpu-budget-ack does not drag the next
                # unrelated token in behind it.
                rollout_slice.append(token)
        parsed = build_arg_parser().parse_args(
            ["--checkpoint", "/tmp/c.pt", "--policy-mode", "engine-mcts",
             "--engine-model-path", "/tmp/m.pt",
             "--engine-tables-path", "/tmp/t.json"] + rollout_slice
        )
        self.assertTrue(parsed.engine_rollout_leaf)
        self.assertEqual(parsed.engine_rollout_count, count)
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


class ArgvReachesTheConfigTest(unittest.TestCase):
    """argv -> `ControlledFoulPlayConfig`, the layer the other tests skip.

    `FlagReachesEngineConfigTest` builds a `ControlledFoulPlayConfig` DIRECTLY and
    checks it reaches `EngineMctsConfig`, so it cannot see `_config_from_args` at
    all. Review demonstrated the gap: hardcoding `engine_rollout_leaf=False` in
    `_config_from_args` -- i.e. the CLI flag never reaching the config, the exact
    gap this change exists to close -- survived the entire suite.
    """

    def _parse(self, extra: list[str]) -> argparse.Namespace:
        return build_arg_parser().parse_args(
            ["--checkpoint", "/tmp/c.pt", "--policy-mode", "engine-mcts",
             "--engine-model-path", "/tmp/m.pt",
             "--engine-tables-path", "/tmp/t.json"] + extra
        )

    def test_the_whole_chain_from_argv_to_the_config(self) -> None:
        """FAILING INPUT: the arm-off half. Asserted from the same function, so a
        hardcoded value on either side fails one of the two.
        """
        from pokezero.foulplay_bridge import _config_from_args

        off = _config_from_args(self._parse([]))
        self.assertFalse(off.engine_rollout_leaf)

        on = _config_from_args(self._parse([
            "--engine-rollout-leaf",
            "--engine-rollout-count", "8",
            "--engine-rollout-max-plies", "400",
            "--engine-rollout-policy", "uniform",
            "--engine-rollout-seed", "11",
            "--engine-rollout-threads", "3",
            "--engine-rollout-threads-cpu-budget-ack",
        ]))
        self.assertTrue(on.engine_rollout_leaf)
        self.assertEqual(on.engine_rollout_count, 8)
        self.assertEqual(on.engine_rollout_max_plies, 400)
        self.assertEqual(on.engine_rollout_policy, "uniform")
        self.assertEqual(on.engine_rollout_seed, 11)
        self.assertEqual(on.engine_rollout_threads, 3)
        self.assertTrue(on.engine_rollout_threads_cpu_budget_ack)

    def test_the_arm_is_refused_where_it_would_reach_nothing(self) -> None:
        """`policy_mode != 'engine-mcts'` must refuse, like every sibling flag.

        The seam is the native search's leaf evaluator, so under 'raw' the flag
        reaches nothing -- while the shard body would still echo
        `rollout_leaf: true` for the cell. That is a false witness on the only
        record of what ran, and it was reachable from the paired driver as
        `--arm raw --engine-rollout-leaf`.

        FAILING INPUT: engine-mcts constructs (asserted above), so this is not a
        test that every mode raises.
        """
        from pokezero.foulplay_bridge import _config_from_args

        args = build_arg_parser().parse_args(
            ["--checkpoint", "/tmp/c.pt", "--policy-mode", "raw",
             "--engine-rollout-leaf"]
        )
        with self.assertRaises(ValueError) as caught:
            _config_from_args(args)
        self.assertIn("engine_rollout_leaf", str(caught.exception))
        self.assertIn("engine-mcts", str(caught.exception))

    def test_a_non_engine_opponent_seat_does_not_inherit_the_arm(self) -> None:
        """The derived opponent config sets `policy_mode = opponent_policy_mode`.

        So without `engine_rollout_leaf` in `_ENGINE_ONLY_FIELDS` the new refusal
        above would break a LEGITIMATE head-to-head run -- rollout pokezero seat
        against a raw opponent, which is exactly the configuration the first game
        was played in. The two changes only work together.

        FAILING INPUT: the engine-mcts opponent below KEEPS the arm, so this does
        not pass by clearing unconditionally.
        """
        from pokezero.foulplay_bridge import _opponent_seat_config

        base = _bridge_config(
            engine_rollout_leaf=True,
            opponent_policy_mode="raw",
        )
        raw_opponent = _opponent_seat_config(base)
        self.assertEqual(raw_opponent.policy_mode, "raw")
        self.assertFalse(
            raw_opponent.engine_rollout_leaf,
            "a raw opponent seat must not carry the arm, or building it raises",
        )
        # An engine-mcts opponent legitimately DOES inherit it: both seats price
        # leaves by rollout. That is a different experiment from
        # rollout-vs-value-head, which is why the shard witnesses it per seat.
        # `opponent_engine_depth` differs so this is a budget-vs-budget pairing
        # rather than a mirror match, which the config refuses outright.
        engine_opponent = _opponent_seat_config(
            _bridge_config(engine_rollout_leaf=True,
                           opponent_policy_mode="engine-mcts",
                           opponent_engine_depth=2)
        )
        self.assertTrue(engine_opponent.engine_rollout_leaf)


class ShardTelemetryWitnessTest(unittest.TestCase):
    """The shard body must say whether the arm was ASKED FOR, per seat.

    Review demonstrated that hardcoding `"rollout_leaf": False` in the engine
    telemetry block survived the suite. That block is the only record a reader of
    one shard has.
    """

    def _engine_block(self, **overrides) -> dict[str, Any]:
        from pokezero.foulplay_bridge import ControlledFoulPlayBenchmarkResult

        result = ControlledFoulPlayBenchmarkResult(
            config=_bridge_config(**overrides), policy_id="pid", games=(),
        )
        return result.to_dict()["engine_mcts"]

    def test_the_block_records_the_arm_and_its_knobs(self) -> None:
        """FAILING INPUT: the arm-off block must read False, the arm-on block True,
        from the same builder.
        """
        off = self._engine_block()
        self.assertFalse(off["rollout_leaf"])
        on = self._engine_block(
            engine_rollout_leaf=True,
            engine_rollout_count=8,
            engine_rollout_max_plies=400,
            engine_rollout_seed=11,
            engine_rollout_threads=2,
            engine_rollout_threads_cpu_budget_ack=True,
        )
        self.assertTrue(on["rollout_leaf"])
        self.assertEqual(on["rollout_count"], 8)
        self.assertEqual(on["rollout_max_plies"], 400)
        self.assertEqual(on["rollout_policy"], "uniform")
        # Outside config_id on purpose, so the body is their ONLY record.
        self.assertEqual(on["rollout_seed"], 11)
        self.assertEqual(on["rollout_threads"], 2)

    def test_the_opponent_seat_is_witnessed_separately(self) -> None:
        """An engine-mcts opponent inherits the arm, so the shard must say so.

        FAILING INPUT: the raw-opponent case must read False for the opponent seat
        while the pokezero seat reads True -- a single shared field cannot satisfy
        both halves.
        """
        from pokezero.foulplay_bridge import ControlledFoulPlayBenchmarkResult

        # `opponent_engine_depth` on the engine case only, so it is a
        # budget-vs-budget pairing rather than the mirror match the config refuses.
        for mode, extra, expected in (
            ("raw", {}, False),
            ("engine-mcts", {"opponent_engine_depth": 2}, True),
        ):
            with self.subTest(opponent=mode):
                payload = ControlledFoulPlayBenchmarkResult(
                    config=_bridge_config(
                        engine_rollout_leaf=True,
                        opponent_policy_mode=mode,
                        **extra,
                    ),
                    policy_id="pid",
                    games=(),
                ).to_dict()
                opponent_block = payload["opponent_engine_mcts"]
                self.assertIsNotNone(opponent_block)
                self.assertEqual(
                    opponent_block["rollout_leaf"], expected,
                    f"the {mode} opponent seat's rollout witness is wrong",
                )
                # The POKEZERO seat is on in both cases, so the two fields are
                # genuinely independent rather than one value rendered twice.
                self.assertTrue(payload["engine_mcts"]["rollout_leaf"])


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
        self.assertEqual(row["searches"], 1)

        # The keys spread on the RECORD instead of the report -- the bug's shape.
        flat = [{"side_key": "side_one", "rollout_leaf_mode": "rollout",
                 "rollouts_run": 16, "report": {"model_evals": 1}}]
        self.assertIsNone(
            EngineMctsPolicy._rollout_decision_row(flat),
            "the helper must read report[...] and not the record's top level",
        )

    @staticmethod
    def _policy_shell() -> Any:
        """An `EngineMctsPolicy` with only `.stats`, for the absorption tests.

        `_absorb_rollout_report` touches nothing else, and constructing a real
        policy needs a dex, a candidate-set source and a TorchScript artifact --
        none of which the absorption reads.
        """
        from pokezero.engine_search import EngineMctsStats

        shell = EngineMctsPolicy.__new__(EngineMctsPolicy)
        shell.stats = EngineMctsStats()
        return shell

    def test_the_shard_absorption_reads_the_report_and_populates_the_witness(self) -> None:
        """The shard-level witness, which is this change's headline claim.

        FAILING INPUT: a report with NO `rollout_leaf_mode` must leave every
        counter at zero and the mode set empty, while a report with one must
        populate all of them -- so neither an absorption that never fires nor one
        that fires unconditionally passes.
        """
        shell = self._policy_shell()
        self.assertFalse(
            shell._absorb_rollout_report({"model_evals": 7}),
            "a pre-seam report must not register as an engaged seam",
        )
        self.assertEqual(shell.stats.rollout_leaf_modes, Counter())
        self.assertEqual(shell.stats.rollouts_run, 0)

        self.assertTrue(shell._absorb_rollout_report({
            "rollout_leaf_mode": "rollout",
            "rollouts_run": 64,
            "rollout_plies": 1280,
            "rollout_terminal_hits": 60,
            "rollout_cap_hits": 4,
            "rollout_dead_ends": 0,
            "leaves_priced": 8,
            "rollout_encode_skipped": 3,
        }))
        s = shell.stats
        self.assertEqual(s.rollout_leaf_modes, Counter({"rollout": 1}))
        self.assertEqual(s.rollout_leaf_world_records, 1)
        # Each counter asserted individually and at a DISTINCT value, so a
        # copy-paste that read the wrong report key cannot pass.
        self.assertEqual(s.rollouts_run, 64)
        self.assertEqual(s.rollout_plies, 1280)
        self.assertEqual(s.rollout_terminal_hits, 60)
        self.assertEqual(s.rollout_cap_hits, 4)
        self.assertEqual(s.rollout_dead_ends, 0)
        self.assertEqual(s.rollout_leaves_priced, 8)
        self.assertEqual(s.rollout_encode_skipped, 3)

        # ... and it ACCUMULATES per invocation, which is the documented unit.
        shell._absorb_rollout_report({
            "rollout_leaf_mode": "rollout", "rollouts_run": 64,
            "rollout_terminal_hits": 64,
        })
        self.assertEqual(shell.stats.rollouts_run, 128)
        self.assertEqual(shell.stats.rollout_leaf_world_records, 2)

    def test_every_absorbed_counter_survives_to_dict(self) -> None:
        """The payload must carry what was absorbed.

        FAILING INPUT: every value below is distinct and non-zero, so a hardcoded
        constant or a duplicated key fails on at least one.
        """
        shell = self._policy_shell()
        shell._absorb_rollout_report({
            "rollout_leaf_mode": "rollout",
            "rollouts_run": 100,
            "rollout_plies": 5500,
            "rollout_terminal_hits": 91,
            "rollout_cap_hits": 9,
            "rollout_dead_ends": 0,
            "leaves_priced": 12,
            "rollout_encode_skipped": 5,
        })
        payload = shell.stats.to_dict()
        self.assertEqual(payload["rollout_leaf_modes"], {"rollout": 1})
        self.assertEqual(payload["rollout_leaf_world_records"], 1)
        self.assertEqual(payload["rollouts_run"], 100)
        self.assertEqual(payload["rollout_plies"], 5500)
        self.assertEqual(payload["rollout_terminal_hits"], 91)
        self.assertEqual(payload["rollout_cap_hits"], 9)
        self.assertEqual(payload["rollout_dead_ends"], 0)
        self.assertEqual(payload["rollout_leaves_priced"], 12)
        self.assertEqual(payload["rollout_encode_skipped"], 5)
        self.assertAlmostEqual(payload["rollout_fallback_fraction"], 0.09)
        self.assertAlmostEqual(payload["rollout_terminal_fraction"], 0.91)
        self.assertAlmostEqual(payload["rollout_mean_plies"], 55.0)

    def test_a_dead_end_is_refused_and_is_not_a_term_in_the_fallback_fraction(self) -> None:
        """`rollout_dead_ends`, which review found pinned to zero by construction.

        It was a second term in `rollout_fallback_fraction`, and four mutants on
        that term survived the whole suite -- because the term cannot read non-zero.
        The reason is now machine-checked crate-side
        (`rollout.rs::get_all_options_never_yields_an_empty_option_vector`, which
        fails when the pokezero fidelity patch to gen3's `get_all_options` is
        removed): every exit from that function backfills an empty option vector,
        so the engine cannot present a position with no legal continuation that it
        did not call over.

        So the field is no longer a term. It is an assertion, and this is the test
        that it can read non-zero and does something when it does.

        FAILING INPUTS, both halves asserted here:
          * `rollout_dead_ends=1` -> refused, and nothing is absorbed
          * `rollout_dead_ends=0` -> absorbed, and the fraction is cap/run exactly
        """
        shell = self._policy_shell()
        with self.assertRaises(ValueError) as caught:
            shell._absorb_rollout_report({
                "rollout_leaf_mode": "rollout", "rollouts_run": 10,
                "rollout_terminal_hits": 8, "rollout_cap_hits": 1,
                "rollout_dead_ends": 1, "leaves_priced": 2,
            })
        self.assertIn("rollout_dead_ends=1", str(caught.exception))
        self.assertIn("could not step", str(caught.exception))

        # The fraction is cap/run -- NOT (cap + dead)/run, and not terminal-derived.
        # 3/12 = 0.25 is distinct from every other ratio these numbers can form
        # (terminal 9/12 = 0.75, (cap+dead)/run would be 0.25 too if dead were
        # absorbed, which is why dead is 0 here and the refusal above carries that
        # half of the assertion).
        clean = self._policy_shell()
        self.assertTrue(clean._absorb_rollout_report({
            "rollout_leaf_mode": "rollout", "rollouts_run": 12,
            "rollout_terminal_hits": 9, "rollout_cap_hits": 3,
            "rollout_dead_ends": 0, "leaves_priced": 4,
        }))
        payload = clean.stats.to_dict()
        self.assertEqual(payload["rollout_dead_ends"], 0)
        self.assertAlmostEqual(payload["rollout_fallback_fraction"], 0.25)
        self.assertAlmostEqual(payload["rollout_terminal_fraction"], 0.75)

    def test_the_fallback_fraction_does_not_read_the_dead_end_counter(self) -> None:
        """A STRUCTURAL guard, and it says so, because this one has no behavioural
        signature and pretending otherwise would be the very defect under review.

        Re-adding `+ self.rollout_dead_ends` to `rollout_fallback_fraction` is an
        EQUIVALENT mutation: `_absorb_rollout_report` refuses a non-zero reading, so
        the counter is provably always 0 and the two expressions cannot differ on
        any input. Mutation testing therefore reports that mutant as surviving, and
        it is right to -- it is exactly the "term that cannot read non-zero" the
        review objected to, restored.

        So the decision is pinned where it lives: the fraction's numerator reads the
        cap counter and does not read the dead-end counter. That is a claim about the
        source, asserted against the source, rather than a behavioural test dressed
        up as one.

        FAILING INPUT: re-adding the term, or swapping cap for terminal, both land
        here (the second also fails behaviourally, above).
        """
        import ast  # noqa: PLC0415
        import inspect  # noqa: PLC0415
        import textwrap  # noqa: PLC0415

        from pokezero.engine_search import EngineMctsStats

        tree = ast.parse(textwrap.dedent(inspect.getsource(EngineMctsStats.to_dict)))
        fraction = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if (
                        isinstance(key, ast.Constant)
                        and key.value == "rollout_fallback_fraction"
                    ):
                        fraction = value
        self.assertIsNotNone(fraction, "the fallback fraction must be in `to_dict`")
        read = {
            node.attr for node in ast.walk(fraction) if isinstance(node, ast.Attribute)
        }
        self.assertIn("rollout_cap_hits", read)
        self.assertIn("rollouts_run", read)
        self.assertNotIn(
            "rollout_dead_ends", read,
            "the dead-end counter is an invariant witness, refused in "
            "`_absorb_rollout_report` and provably zero, so it must not be a term in "
            "a fraction that is read as an estimand",
        )

    def test_the_decision_row_fraction_is_cap_over_run_too(self) -> None:
        """The per-decision copy must agree with the shard copy.

        FAILING INPUT: `rollout_cap_hits` and `rollout_terminal_hits` are DISTINCT
        and neither equals half, so a row that summed the wrong fields, or that
        re-added a `dead` term, lands on a different number.
        """
        row = EngineMctsPolicy._rollout_decision_row([
            {"report": {"rollout_leaf_mode": "rollout", "rollouts_run": 20,
                        "rollout_terminal_hits": 17, "rollout_cap_hits": 3,
                        "rollout_dead_ends": 0, "leaves_priced": 5}}
        ])
        assert row is not None
        self.assertAlmostEqual(row["rollout_fallback_fraction"], 0.15)
        self.assertEqual(row["rollout_dead_ends"], 0)

    def test_the_absorption_is_CALLED_on_the_world_report_path(self) -> None:
        """THE CALL, not the method. This change's headline claim depends on it.

        Extracting `_absorb_rollout_report` made the METHOD testable and moved the
        untested boundary exactly one line out: review demonstrated that replacing
        the CALL with `pass` passes the whole suite, and that on a real game the
        resulting shard reads `rollout_leaf: true` with `rollout_leaf_modes {}` and
        `rollouts_run 0` -- byte-identical to the value-head twin -- while 16,908
        rollouts actually ran. Absence of a call is not mutation-testable inside
        the method, so it is pinned here, following the sibling
        `_rollout_metadata_fields`.

        Structural rather than a substring search, and that is the point: the call
        must be an UNCONDITIONAL statement in the same function body that absorbs
        `model_evals` -- the per-invocation world-report path. A substring check
        passes for `if False: self._absorb_rollout_report(report)`; this does not.

        `_search_model` cannot be driven in a unit test (it needs a dex, a
        candidate-set source, a TorchScript artifact and a live battle), which is
        why the inline block was unreachable in the first place.

        FAILING INPUTS, all four demonstrated:
          * the call replaced with `pass`                -> no such Call node
          * the call deleted outright                    -> no such Call node
          * the call wrapped in `if False:`              -> not statement-level
          * the call handed something other than `report` -> argument mismatch
        """
        import ast  # noqa: PLC0415
        import inspect  # noqa: PLC0415
        import textwrap  # noqa: PLC0415

        tree = ast.parse(
            textwrap.dedent(inspect.getsource(EngineMctsPolicy._search_model))
        )
        def owned(function: ast.AST):
            """Nodes belonging to `function` itself, not to a function nested in it.

            `ast.walk` crosses function boundaries, so an unqualified walk reports
            `_search_model` as an absorber merely because `run_world` is defined
            inside it -- and the guard would then be aimed one level too high,
            where the call is not.
            """
            stack = list(ast.iter_child_nodes(function))
            while stack:
                node = stack.pop()
                yield node
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                         ast.Lambda, ast.ClassDef)):
                    stack.extend(ast.iter_child_nodes(node))

        absorbers = [
            node for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(
                isinstance(inner, ast.AugAssign)
                and isinstance(inner.target, ast.Attribute)
                and inner.target.attr == "model_evals"
                for inner in owned(node)
            )
        ]
        self.assertEqual(
            len(absorbers), 1,
            "one world-report absorber is assumed; if `model_evals` moved, this "
            "guard is pointing at the wrong function and must be re-aimed",
        )
        # Statement level in the absorber's own body: `for`/`with` wrappers are the
        # ones the absorption legitimately sits under today (none), and a
        # conditional is exactly what must NOT be accepted.
        statement_calls = [
            node.value for node in absorbers[0].body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "_absorb_rollout_report"
        ]
        self.assertEqual(
            len(statement_calls), 1,
            "the world-report absorber must call self._absorb_rollout_report(report) "
            "unconditionally -- replacing it with `pass`, deleting it, or guarding it "
            "behind a conditional all land here",
        )
        call = statement_calls[0]
        self.assertEqual(
            [ast.unparse(argument) for argument in call.args], ["report"],
            "the crate's REPORT must be what is absorbed -- not the config, which "
            "says only what was asked for, and not the record, whose top level does "
            "not carry the rollout keys",
        )
        self.assertEqual(ast.unparse(call.func), "self._absorb_rollout_report")

    def test_the_metadata_fields_helper_is_what_the_decision_splats(self) -> None:
        """The attachment, isolated so it is reachable at all.

        FAILING INPUT: the no-seam case must yield `{}` -- not `{"rollout": None}`
        -- because a flag-off decision's metadata must be byte for byte unchanged.
        """
        self.assertEqual(
            EngineMctsPolicy._rollout_metadata_fields(
                [{"report": {"model_evals": 3}}]
            ),
            {},
        )
        fields = EngineMctsPolicy._rollout_metadata_fields(
            [{"report": {"rollout_leaf_mode": "rollout", "rollouts_run": 8,
                         "rollout_terminal_hits": 8}}]
        )
        self.assertEqual(list(fields), ["rollout"])
        self.assertEqual(fields["rollout"]["rollout_leaf_modes"], ["rollout"])
        # And the attachment site carries no logic of its own beyond the splat, so
        # this helper IS the behaviour.
        import inspect

        source = inspect.getsource(EngineMctsPolicy._search_model)
        self.assertIn("**self._rollout_metadata_fields(world_runs),", source)

    def test_collapsed_twin_worlds_are_counted_once(self) -> None:
        """THE COLLAPSE OVER-COUNT, found in review.

        Duplicate belief draws are searched ONCE and the same report object is
        written into every twin's record, so summing over `world_runs` multiplied
        every work counter by the collapse multiplicity -- 512 rollouts reported
        for 128 actually run.

        FAILING INPUT: the two records below carry the SAME `_collapse_key`, so a
        reader that does not dedupe reports double. The distinct-key case
        immediately after must still sum, so this is not satisfied by always
        taking one record.
        """
        report = {
            "rollout_leaf_mode": "rollout", "rollouts_run": 128,
            "rollout_terminal_hits": 120, "rollout_cap_hits": 8,
            "rollout_dead_ends": 0, "leaves_priced": 4,
            "rollout_encode_skipped": 2,
        }
        twins = [
            {"report": dict(report), "_collapse_key": "k", "_collapse_multiplicity": 2},
            {"report": dict(report), "_collapse_key": "k", "_collapse_multiplicity": 2},
        ]
        row = EngineMctsPolicy._rollout_decision_row(twins)
        assert row is not None
        self.assertEqual(row["searches"], 1, "one search, not two records")
        self.assertEqual(row["rollouts_run"], 128, "the work must not be doubled")
        self.assertEqual(row["leaves_priced"], 4)
        self.assertEqual(row["rollout_encode_skipped"], 2)
        # The worlds those searches stood for are still reported, because belief
        # breadth is a per-world quantity -- and it must not be squared either.
        self.assertEqual(row["worlds_represented"], 2)

        # Two INDEPENDENT searches must sum, or the dedupe has eaten real work.
        independent = [
            {"report": dict(report), "_collapse_key": "a", "_collapse_multiplicity": 1},
            {"report": dict(report), "_collapse_key": "b", "_collapse_multiplicity": 1},
        ]
        row2 = EngineMctsPolicy._rollout_decision_row(independent)
        assert row2 is not None
        self.assertEqual(row2["searches"], 2)
        self.assertEqual(row2["rollouts_run"], 256)
        self.assertEqual(row2["worlds_represented"], 2)

        # Records with NO collapse key are independent by identity, never merged
        # into one another by a shared default.
        bare = [{"report": dict(report)}, {"report": dict(report)}]
        row3 = EngineMctsPolicy._rollout_decision_row(bare)
        assert row3 is not None
        self.assertEqual(row3["searches"], 2)
        self.assertEqual(row3["rollouts_run"], 256)

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
        self.assertEqual(row["searches"], 2)


if __name__ == "__main__":
    unittest.main()
