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

        `_refuse_rollout_dead_ends` touches nothing else, and constructing a real
        policy needs a dex, a candidate-set source and a TorchScript artifact --
        none of which the absorption reads.
        """
        from pokezero.engine_search import EngineMctsStats

        shell = EngineMctsPolicy.__new__(EngineMctsPolicy)
        shell.stats = EngineMctsStats()
        return shell



    def test_a_dead_end_is_refused_and_is_not_a_term_in_the_fallback_fraction(self) -> None:
        """`rollout_dead_ends`, which review found pinned to zero by construction.

        The reason is machine-checked crate-side
        (`rollout.rs::get_all_options_never_yields_an_empty_option_vector`, which
        fails when the pokezero fidelity patch to gen3's `get_all_options` is
        removed): every exit from that function backfills an empty option vector, so
        the engine cannot present a position with no legal continuation that it did
        not call over. So the field is not a term to be summed; it is an assertion,
        and this is the test that it can read non-zero and does something when it does.

        RETARGETED FROM `_absorb_rollout_report`, WHICH THIS BRANCH NO LONGER HAS.
        The absorber accumulated the whole rollout ledger into the shard stats, and
        #1271 absorbs the same report into the same counters on the same path -- two
        writers of one surface, which would double every rollout counter on every arm
        shard. The ACCUMULATION is deleted here and #1271 lands first; the REFUSAL is
        not #1271's and stays, as `_refuse_rollout_dead_ends`. #1271's witness guard
        refuses an unbalanced partition and a fraction that is not its own quotient,
        and neither is the same statement as "dead ends must be zero" -- its shard
        schema's `cap + dead` numerator exists precisely so a non-zero count is
        REPORTED rather than refused. Dropping this with the absorber would have lost
        a refusal nothing else in either tree makes.

        FAILING INPUTS, both halves:
          * `rollout_dead_ends=1` -> refused, by value and with the cause named;
          * `rollout_dead_ends=0` -> accepted, so the refusal is not refusing
            every report it is handed.
        """
        shell = self._policy_shell()
        with self.assertRaises(ValueError) as caught:
            shell._refuse_rollout_dead_ends({
                "rollout_leaf_mode": "rollout", "rollouts_run": 10,
                "rollout_terminal_hits": 8, "rollout_cap_hits": 1,
                "rollout_dead_ends": 1, "leaves_priced": 2,
            })
        self.assertIn("rollout_dead_ends=1", str(caught.exception))
        self.assertIn("could not step", str(caught.exception))
        # AND IT ABSORBS NOTHING -- which is now a property of the method's whole
        # body rather than of where the check sits inside it. FAILING INPUT: any
        # accumulation re-added here lands on one of these.
        self.assertEqual(dict(shell.stats.rollout_leaf_modes), {})
        self.assertEqual(shell.stats.rollout_leaf_worlds, 0)
        self.assertEqual(shell.stats.rollouts_run, 0)
        self.assertEqual(shell.stats.rollout_terminal_hits, 0)
        self.assertEqual(shell.stats.rollout_cap_hits, 0)
        self.assertEqual(shell.stats.rollout_leaves_priced, 0)

        clean = self._policy_shell()
        clean._refuse_rollout_dead_ends({
            "rollout_leaf_mode": "rollout", "rollouts_run": 12,
            "rollout_terminal_hits": 9, "rollout_cap_hits": 3,
            "rollout_dead_ends": 0, "leaves_priced": 4,
        })
        self.assertEqual(clean.stats.rollouts_run, 0)
        # A PRE-SEAM REPORT IS NOT REFUSED EITHER, so the guard keyed off the REPORT
        # (not off `config.rollout_leaf_eval`) still tolerates an older wheel.
        clean._refuse_rollout_dead_ends({"model_evals": 7, "rollout_dead_ends": 3})

    def test_this_branch_emits_NO_shard_level_rollout_block(self) -> None:
        """B2. The merge order, asserted in the tree rather than described in a body.

        This branch's `EngineMctsStats.to_dict` used to emit thirteen rollout keys
        UNCONDITIONALLY and -- having adopted #1271's constant -- stamp them
        `rollout_leaf_schema: 2`. On a FLAG-OFF shard that is v1's shape wearing v2's
        stamp: zeroed counts, `None` quotients, and an absent world count that a v2
        reader reads as "the arm engaged zero worlds" rather than as an absence. It
        also stamped a version NO READER IN THIS TREE can check, because both readers
        (`require_rollout_leaf_shard_schema`,
        `require_rollout_leaf_document_schema`) are #1271's.

        So the block is deleted here and #1271 LANDS FIRST. The consequence is stated
        rather than hidden: on this branch alone there is no shard-level rollout
        telemetry at all. That is honest -- an absent block is an absence -- whereas
        the deleted one asserted a schema nothing could read.
        """
        from pokezero.engine_search import EngineMctsStats

        stats = EngineMctsStats()
        stats.rollout_leaf_modes["rollout"] = 4
        stats.rollout_leaf_worlds = 4
        stats.rollouts_run = 64
        payload = stats.to_dict()
        emitted = sorted(k for k in payload if str(k).startswith("rollout"))
        self.assertEqual(
            emitted,
            [],
            "this branch must not write the shard-level rollout block; #1271 owns it "
            f"and emits it CONDITIONALLY. Found: {emitted}",
        )
        self.assertNotIn("rollout_leaf_schema", payload)
        # AND THE FLAG-OFF PAYLOAD IS THE PRE-SEAM ONE, which is the contract the
        # unconditional writer could not satisfy by construction.
        self.assertEqual(
            sorted(k for k in EngineMctsStats().to_dict() if str(k).startswith("rollout")),
            [],
        )

    def test_the_crate_only_path_refuses_a_dead_end_FAIL_CLOSED(self) -> None:
        """The second search path, and the placement that makes its refusal safe.

        `_search_rollout_crate` is the sequential crate-only driver;
        `_refuse_rollout_dead_ends` never runs on it, so a dead-end refusal on the
        model path alone would leave two search paths reading one crate counter by
        two rules. Both refuse now.

        THE PLACEMENT IS THE BEHAVIOUR, which is why it is asserted structurally
        rather than by the presence of the text. Inside the world loop's `try`, a
        raise costs a COUNTED world failure and a fallback decision -- fail-closed
        and visible in `world_failures`. Below the `except`, the identical code
        crashes the whole decision. The first revision of this fix put it below the
        handler and asserted *in its own comment* that it was inside; the comment
        was the only thing that made it look safe. This test is what that comment
        was pretending to be.

        `_search_rollout_crate` needs a live battle and a native extension, so it
        cannot be driven from a unit test -- the same reason the absorption block
        was unreachable before it was extracted.

        FAILING INPUTS: the check deleted; the check moved out of the `try` (below
        the handler, i.e. crash instead of fallback); the check reading the config
        instead of the report.
        """
        import ast  # noqa: PLC0415
        import inspect  # noqa: PLC0415
        import textwrap  # noqa: PLC0415

        tree = ast.parse(
            textwrap.dedent(inspect.getsource(EngineMctsPolicy._search_rollout_crate))
        )
        tries = [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
        self.assertTrue(tries, "the world loop must still guard the native call")

        def raises_on_dead_ends(statements) -> bool:
            for node in statements:
                for inner in ast.walk(node):
                    if not isinstance(inner, ast.Raise):
                        continue
                    test_source = ast.unparse(node)
                    if (
                        "rollout_dead_ends" in test_source
                        and isinstance(node, ast.If)
                        and "report" in ast.unparse(node.test)
                    ):
                        return True
            return False

        guarded = [handler for handler in tries if raises_on_dead_ends(handler.body)]
        self.assertEqual(
            len(guarded), 1,
            "exactly one `try` body must raise on a non-zero rollout_dead_ends read "
            "off the REPORT. Deleting the check, or moving it below the `except` "
            "where it would crash the decision instead of costing a counted world "
            "failure, both land here",
        )
        # ... and the handler it sits under is the one that counts the failure, so
        # the fail-closed path is the one that receives it.
        handler_source = "".join(
            ast.unparse(handler) for handler in guarded[0].handlers
        )
        self.assertIn("crate_search_rollout", handler_source)
        self.assertIn("world_failure_reasons", handler_source)


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


class AgreedShardWitnessShapeTest(unittest.TestCase):
    """ONE SURFACE, ONE SHAPE, ONE VERSION STAMP -- pinned on the WRITER side.

    This branch and `phase1/rollout-model-priors` (#1271) were both widening the
    shard's `policy_stats` with a rollout-leaf witness, four incompatible ways at
    once and under a byte-identical `schema_version`:

        name      `rollout_leaf_world_records`   vs  `rollout_leaf_worlds`
        unit      `+= 1` (per invocation)        vs  `+= weight` (per world)
        presence  UNCONDITIONAL                  vs  conditional (absent when off)
        modes     `+= 1`                         vs  `+= weight`

    Nothing read the witness, so no metric was wrong -- it is worse in kind:
    whichever landed second would silently re-schema the four shards this arm has
    already banked, and a reader trusting the version string could not tell which
    shape it held. Same defect the deploy repo fixed in `b30c30e`, one layer down.

    THREE OF THE FOUR AXES WERE SETTLED by adopting #1271's schema (v2) -- name,
    unit, and the mode tally's unit -- plus a `rollout_leaf_schema` stamp inside the
    block, which is what makes "written before the rename" and "written by a writer
    that dropped a field" different artifacts. This class pins that decision so it
    cannot silently drift back.

    THE FOURTH AXIS, PRESENCE, WAS NOT SETTLED THERE, and is settled here: this
    writer emits the whole block UNCONDITIONALLY, and that is asserted. It is the
    axis that matters most to this campaign, because a refusal built on "absent is
    not false" cannot coexist with a sibling writer that encodes "arm off" as an
    absent key -- and `test_absent_is_not_zero_on_the_fallback_fraction` and the
    ABSENT-IS-NOT-FALSE note in `foulplay_power_report` are both built on it. The
    banked value-head twin carries `rollout_leaf_modes: {}`, `rollouts_run: 0` and
    `rollout_fallback_fraction: null` explicitly present; under a conditional writer
    those keys vanish and an arm-off run becomes indistinguishable from a shard
    written before the arm existed.

    The reader half -- refuse a shard whose schema this code did not write, rather
    than coerce it -- is `test_foulplay_power_report.ShardWitnessSchemaRefusalTest`.
    """

    @staticmethod
    def _declared() -> tuple:
        return tuple(_DRIVER.ROLLOUT_WITNESS_KEYS)




    def test_the_SUPERSEDED_field_name_is_not_emitted(self) -> None:
        """`rollout_leaf_world_records` was this branch's name and lost.

        It must not exist here under any spelling -- a field that exists but is
        unpublished is a merge waiting to republish it, and two writers of one
        surface is the entire thing being prevented.

        FAILING INPUT: the rename reverted.
        """
        from pokezero.engine_search import EngineMctsStats  # noqa: PLC0415

        stats = EngineMctsStats()
        for superseded in _DRIVER.ROLLOUT_WITNESS_SUPERSEDED_KEYS:
            self.assertNotIn(superseded, stats.to_dict())
            self.assertFalse(
                hasattr(stats, superseded),
                f"{superseded} is the superseded name for this counter; it must not "
                "exist here, or a merge can silently republish it",
            )



    def test_the_crate_only_paths_fraction_reads_both_counters_STRUCTURALLY(
        self,
    ) -> None:
        """The FOURTH writer, and this one has no behavioural signature -- said
        plainly, because pretending otherwise is the defect this whole round is about.

        `_search_rollout_crate` refuses a non-zero `rollout_dead_ends` inside its own
        world loop and `continue`s before accumulating, so `dead_ends` is provably 0
        by the time its decision metadata is built. `(cap + dead)/denom` and
        `cap/denom` therefore cannot differ on any input this path accepts, and a
        mutant reverting it is EQUIVALENT -- measured: it survived the suite, and it
        was right to.

        So the decision is pinned where it lives, against the source, and labelled as
        such. The behavioural writers are covered by the test above.

        FAILING INPUT: dropping the `dead_ends` term, or swapping `cap_hits` for
        `terminal_hits`, both land here.
        """
        import ast  # noqa: PLC0415
        import inspect  # noqa: PLC0415
        import textwrap  # noqa: PLC0415

        tree = ast.parse(textwrap.dedent(
            inspect.getsource(EngineMctsPolicy._search_rollout_crate)
        ))
        found = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if (isinstance(key, ast.Constant)
                            and key.value == "rollout_fallback_fraction"):
                        found = value
        self.assertIsNotNone(
            found, "the crate-only path must publish a fallback fraction"
        )
        names = {n.id for n in ast.walk(found) if isinstance(n, ast.Name)}
        self.assertIn("cap_hits", names)
        self.assertIn(
            "dead_ends", names,
            "the fraction must be the quotient of its OWN partition, agreeing with "
            "the shard aggregate, the model path's row and the crate -- one published "
            "field name, one rule",
        )
        self.assertNotIn("terminal_hits", names)


class WorldReportPathBehaviourTest(unittest.TestCase):
    """THE WITNESS AS A DATUM. A structural assertion about code is not an
    assertion about behaviour, and this class is the behavioural half.

    Review round 2 established the gap by measurement, and the AST pin it was about
    IS GONE FROM THIS BRANCH -- deleted with the absorber it pinned. Recorded here
    rather than quietly dropped, because the finding generalises past the code that
    occasioned it.

    The pin required an unconditional statement-level
    `self._absorb_rollout_report(report)` in the same body that absorbs `model_evals`,
    taking `report`. It killed five mutants -- and FOUR inert placements satisfied it
    and survived the whole suite:

      * `return report` hoisted above the call;
      * `report = {}` rebound before the call;
      * `report` stripped of its `rollout*` keys -- which reproduces the original
        symptom exactly, `rollout_leaf: true` with `rollout_leaf_modes {}` and
        `rollouts_run 0` while tens of thousands of rollouts ran;
      * the absorber shadowed by an instance attribute.

    So the pin proved the call was WRITTEN, not that it RAN ON REAL DATA. Reversing to
    a behavioural test was the right call, and the FOLLOW-THROUGH WAS INCOMPLETE: the
    reversal's own justification -- "#1271 makes the term testable at the writer
    instead" -- held at TWO of the three sites that carry the `cap + dead` rule. The
    shard reader and the shard writer each had a non-zero-dead fixture; the
    per-decision witness guard did not, so `expected = (cap + dead) -> cap` survived
    there. #1271 now carries that third fixture
    (`test_the_DEAD_END_term_of_the_fallback_numerator_is_pinned`), which is what makes
    the sentence true at all three.

    Nothing in CI covers the class either: ruff runs only as `python-floor-syntax`, a
    target-version parse check, so not even an unreachable-code rule is in effect. And
    NO TEST FILE ON EITHER BRANCH IS NAMED IN ANY WORKFLOW -- see the CI note in the
    PR body.

    What closes it is running the path and reading the counters. That became
    possible when the world-report closure became `_run_world_report`, a bound
    method taking its captured state explicitly: `native` is a stub whose
    `search_batched_multi_encoded` returns a JSON report, everything else is the
    real production code -- the real `native_search_args` assembly, the real
    `json.loads`, the real absorption, the real `return`.

    Every test below names the failing input it would read False on.
    """

    #: A report shaped like the crate's, with DISTINCT non-zero values so a
    #: counter that read the wrong key cannot pass, and large enough that a
    #: partially-absorbed report is distinguishable from an unabsorbed one.
    ROLLOUT_REPORT = {
        "iterations": 4160,
        "model_evals": 8997,
        "max_depth_reached": 4,
        "rollout_leaf_mode": "rollout",
        "rollouts_run": 71832,
        "rollout_plies": 3935252,
        "rollout_terminal_hits": 70529,
        "rollout_cap_hits": 1303,
        "rollout_dead_ends": 0,
        "leaves_priced": 8979,
        "rollout_encode_skipped": 47,
    }
    #: The same search with no seam: a wheel that predates the rollout positionals
    #: ignores what it does not recognise, so the config says the arm ran and the
    #: report says nothing. The witness must stay empty.
    PRE_SEAM_REPORT = {"iterations": 3648, "model_evals": 8185, "max_depth_reached": 4}

    RECORD = {"state_str": "state", "ctx_json": "ctx", "seed": 7,
              "side_key": "side_one"}

    class _StubNative:
        """The crate seam, and only it. Records what it was handed."""

        def __init__(self, payload: dict) -> None:
            self.payload = payload
            self.calls: list[tuple] = []

        def search_batched_multi_encoded(self, *args) -> str:
            import json as _json  # noqa: PLC0415

            self.calls.append(args)
            return _json.dumps(self.payload)

    def _policy(self) -> Any:
        from pokezero.engine_search import EngineMctsStats  # noqa: PLC0415

        policy = EngineMctsPolicy.__new__(EngineMctsPolicy)
        policy.stats = EngineMctsStats()
        policy._tables_json = "tables"
        return policy

    def _run(self, payload: dict, *, weight: int = 1, **config_kwargs) -> tuple:
        policy = self._policy()
        native = self._StubNative(payload)
        config = EngineMctsConfig(
            **_MODEL_CONFIG, rollout_leaf_eval=True, rollout_count=8,
            **config_kwargs,
        )
        returned = policy._run_world_report(
            dict(self.RECORD), 0, weight=weight,
            config=config, native=native, root_inputs="root", rust_fold=FOLD,
        )
        return policy, native, returned




    def test_a_dead_end_COSTS_A_COUNTED_WORLD_FAILURE_not_a_crash(self) -> None:
        """The model path's refusal, and the placement claim about it.

        `_refuse_rollout_dead_ends` raises on a non-zero `rollout_dead_ends`, and the
        comment beside that raise used to say it "costs a COUNTED world failure and
        a fallback decision". On `_search_rollout_crate` that is true -- the refusal
        is inside the world loop's `try`. On THIS path it was false: the absorption
        runs after this method's only `try` closes, so the ValueError escaped every
        handler, and there is no outer try around `_search_model` (the source says
        so itself, in `_absorb_aborted_lossy_subcases`), so it went straight out of
        `decide()` and crashed the decision. Found by running this path, not by
        reading it -- which is the whole argument for this class existing.

        FAILING INPUTS: an uncaught raise propagates and fails the `assertIsNone`
        (measured: that is what the tree did before the handler was added); a
        refusal that was deleted, short-circuited, or keyed off the config returns
        the report and records no failure, and fails on the other two assertions.
        """
        policy, _native, returned = self._run(
            {**self.ROLLOUT_REPORT, "rollout_dead_ends": 3}, weight=2
        )
        # Fail-CLOSED: the world is dropped, so nothing it priced reaches the
        # aggregate.
        self.assertIsNone(returned)
        # ... and COUNTED, in the same taxonomy and with the same weight as the
        # native call's own aborts.
        reasons = dict(policy.stats.world_failure_reasons)
        matching = [key for key in reasons if "rollout_dead_ends=3" in key]
        self.assertEqual(
            len(matching), 1,
            f"the dead end must be counted as a world failure; got {reasons}",
        )
        self.assertTrue(matching[0].startswith("crate_search: "), matching[0])
        self.assertEqual(reasons[matching[0]], 2, "counted per DRAW, like its siblings")
        # The witness must not advertise the refused world's work.
        self.assertEqual(policy.stats.rollouts_run, 0)
        self.assertEqual(dict(policy.stats.rollout_leaf_modes), {})

    def test_an_aborting_native_call_is_still_counted_and_absorbs_nothing(self) -> None:
        """The pre-existing abort path, asserted so this class covers the whole
        method rather than only its new tail.

        FAILING INPUT: an absorption hoisted above the `try`'s `return None` would
        populate the witness for a world that never produced a report.
        """
        policy = self._policy()

        class Exploding:
            def search_batched_multi_encoded(self, *args):
                raise RuntimeError("native exploded")

        returned = policy._run_world_report(
            dict(self.RECORD), 0,
            config=EngineMctsConfig(**_MODEL_CONFIG, rollout_leaf_eval=True,
                                    rollout_count=8),
            native=Exploding(), root_inputs="root", rust_fold=FOLD,
        )
        self.assertIsNone(returned)
        self.assertEqual(dict(policy.stats.rollout_leaf_modes), {})
        self.assertTrue(
            any("native exploded" in key
                for key in policy.stats.world_failure_reasons),
            dict(policy.stats.world_failure_reasons),
        )

    def test_search_model_binds_the_path_by_forwarding_partial(self) -> None:
        """The residual boundary, kept to one line and pinned.

        Extracting the closure moved the untested boundary from "is the absorption
        called" to "is the extracted method called with the world's arguments". That
        line is a `functools.partial`, which FORWARDS positionals rather than
        re-listing them -- a hand-written forwarder can drop one silently. This
        asserts the binding is that shape, and names the captured state.

        FAILING INPUTS: a forwarder that dropped `weight` or `sims`; a binding that
        passed a different callable; a partial that captured a literal instead of
        the world's `native`.
        """
        import ast  # noqa: PLC0415
        import inspect  # noqa: PLC0415
        import textwrap  # noqa: PLC0415

        tree = ast.parse(
            textwrap.dedent(inspect.getsource(EngineMctsPolicy._search_model))
        )
        bindings = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "run_world"
                    for t in node.targets)
        ]
        self.assertEqual(len(bindings), 1, "one `run_world` binding is assumed")
        call = bindings[0].value
        self.assertIsInstance(call, ast.Call)
        self.assertEqual(ast.unparse(call.func), "functools.partial")
        self.assertEqual(
            [ast.unparse(a) for a in call.args], ["self._run_world_report"],
            "the partial must bind the real world-report method",
        )
        self.assertEqual(
            {kw.arg: ast.unparse(kw.value) for kw in call.keywords},
            {"config": "config", "native": "native",
             "root_inputs": "root_inputs", "rust_fold": "rust_fold"},
            "the captured state must be the world loop's own, not a literal",
        )


class CrateOnlyPathBehaviourTest(unittest.TestCase):
    """`_search_rollout_crate`'s dead-end refusal, RUN rather than parsed.

    The refusal was pinned by an AST assertion that the `try` body holds an
    `ast.If` containing a `Raise`, whose unparse mentions `rollout_dead_ends` and
    whose test mentions `report`. Review found the same blind spot as the
    absorption's:

        if False and int(report.get("rollout_dead_ends") or 0):
            raise ValueError(...)

    satisfies all four conditions -- the `If` is there, the `Raise` is there, both
    names are there -- and can never fire. It SURVIVED.

    So the refusal is now asserted by driving the world loop against a stub crate
    and reading what came out. The `try`-placement property is observable that way
    too, and better: inside the handler a dead end costs a counted world failure
    and a fallback, below it the decision crashes -- and those are two different
    outcomes, not two different shapes.
    """

    REPORT = {
        "side_one": [{"move": "move:tackle", "visits": 40},
                     {"move": "move:growl", "visits": 10}],
        "max_depth_reached": 3,
        "iterations": 512,
        "rollouts_run": 640,
        "rollout_plies": 12800,
        "rollout_terminal_hits": 600,
        "rollout_cap_hits": 40,
        "rollout_dead_ends": 0,
        "leaves_priced": 80,
    }

    def _drive(self, report: dict) -> tuple:
        """Run the real world loop over one world against a stub crate."""
        import json as _json  # noqa: PLC0415
        import types  # noqa: PLC0415
        from pokezero.engine_search import EngineMctsStats  # noqa: PLC0415

        policy = EngineMctsPolicy.__new__(EngineMctsPolicy)
        policy.stats = EngineMctsStats()
        # `rollout_crate` is the SEQUENTIAL arm and does not set
        # `rollout_leaf_eval` -- that flag names the model-path seam, and the config
        # validator refuses the pair. This path is reached by `leaf_eval` alone.
        policy._config = EngineMctsConfig(
            leaf_eval="rollout_crate", search_sims=64, search_depth=4,
            rollout_count=8,
        )
        policy.policy_id = "test"
        fell_back: list[str] = []
        policy._fallback = lambda ctx, rng, reason: fell_back.append(reason)  # type: ignore[assignment]
        policy._map_choices = lambda ctx, aggregated: 0  # type: ignore[assignment]

        world = types.SimpleNamespace(slot_sides={"p1": "side_one"})
        state = types.SimpleNamespace(to_string=lambda: "state")
        context = types.SimpleNamespace(player_id="p1")

        import pokezero_search  # noqa: PLC0415

        with mock.patch.object(
            pokezero_search, "puct_search_multi_rollout",
            lambda *a, **k: _json.dumps(report),
        ):
            decision = policy._search_rollout_crate(
                context, [(world, state)], __import__("random").Random(1)
            )
        return policy, decision, fell_back

    def test_a_clean_report_is_searched_and_priced(self) -> None:
        """The control, so the refusal below cannot pass by firing always.

        FAILING INPUT: a refusal with an inverted predicate refuses this one.
        """
        policy, decision, fell_back = self._drive(dict(self.REPORT))
        self.assertEqual(fell_back, [], "a clean report must not fall back")
        assert decision is not None
        engine = decision.metadata["engine_mcts"]
        self.assertEqual(engine["rollouts_run"], 640)
        self.assertEqual(engine["rollout_dead_ends"], 0)
        # Cap hits alone, agreeing with the shard-level copy: 40/640.
        self.assertAlmostEqual(engine["rollout_fallback_fraction"], 0.0625)
        self.assertEqual(policy.stats.worlds_searched, 1)
        self.assertEqual(dict(policy.stats.world_failure_reasons), {})

    def test_the_crate_only_path_COUNTS_a_dead_end_and_falls_back(self) -> None:
        """The refusal, and its fail-closed placement, both as behaviour.

        FAILING INPUTS, every one of them measured:
          * `if False and int(report.get("rollout_dead_ends") or 0):` -- the mutant
            that SURVIVED the AST pin -- prices the world, so `worlds_searched`
            reads 1 and nothing falls back;
          * the check deleted, or its predicate replaced by `False`: the same;
          * the check keyed off the config rather than the report: the same;
          * the check moved BELOW the `except` (the original unsafe revision): the
            ValueError escapes and this test errors instead of reading a counted
            failure.
        """
        policy, decision, fell_back = self._drive(
            {**self.REPORT, "rollout_dead_ends": 7}
        )
        # Fail-closed: the only world aborted, so the decision falls back -- and it
        # falls back through the COUNTED path, not through an exception.
        self.assertEqual(fell_back, ["crate_search_failed"])
        self.assertEqual(policy.stats.worlds_searched, 0)
        reasons = dict(policy.stats.world_failure_reasons)
        matching = [key for key in reasons if "rollout_dead_ends=7" in key]
        self.assertEqual(len(matching), 1, f"got {reasons}")
        self.assertTrue(
            matching[0].startswith("crate_search_rollout: "),
            f"the failure must land in this path's own taxonomy: {matching[0]}",
        )
        # And nothing the refused world priced reaches the aggregate.
        self.assertEqual(policy.stats.total_iterations, 0)


if __name__ == "__main__":
    unittest.main()
