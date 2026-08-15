"""Gates for the seed-paired FoulPlay driver.

The driver's job is to make the paired delta (search - raw) scoreable, so the
things pinned here are the ones whose failure would produce a plausible NUMBER
rather than an error:

* the cell identity a shard is merged under (`config_id`);
* the seed join, which must fail rather than mis-align;
* the score key, which must fail rather than read every game as a loss;
* the opponent-priors label, which must match what the bridge actually ran;
* the opponent definition (FoulPlay's own search budget) and the thread pin,
  both of which silently change opponent strength if they drift.

Pure functions only -- no bridge subprocess, no checkpoint, no cluster.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]

_SPEC = importlib.util.spec_from_file_location(
    "foulplay_paired_eval_test", REPO_ROOT / "scripts" / "foulplay_paired_eval.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_DRIVER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DRIVER)


def args(**overrides) -> argparse.Namespace:
    base = dict(
        checkpoint="/tmp/ckpt.pt",
        showdown_root="/tmp/showdown",
        device="cuda",
        arm="search",
        seed_start=7800000,
        pairs=200,
        depth=4,
        sims=1024,
        batch=64,
        worlds=4,
        opponent_priors=False,
        engine_fpu_reduction=None,
        engine_c_puct=None,
        engine_override_telemetry=False,
        engine_oracle_belief=False,
        opponent_journal=None,
        engine_early_stop=False,
        engine_depth_min=None,
        engine_worlds_min=None,
        engine_early_stop_min_sims=None,
        # Head-to-head knobs. In the shared fixture rather than per-test, so the direct
        # attribute reads in config_id_for stay direct: the design is that a Namespace
        # predating a knob RAISES rather than being handed the control's id, which
        # test_a_namespace_predating_the_knobs_raises_rather_than_pools pins.
        opponent_policy_mode="foul-play",
        opponent_engine_depth=None,
        opponent_engine_sims=None,
        engine_model_path=None,
        engine_tables_path=None,
        foulplay_root=None,
        foulplay_python=None,
        out="/tmp/shard.json",
        skip_build_check=True,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def game(seed: int, *, score: float = 1.0, won: bool = True) -> dict:
    return {
        "seed": seed,
        "pokezero_won": won,
        "pokezero_score": score,
        "tied": False,
        "capped": False,
    }


class ConfigIdTest(unittest.TestCase):
    def test_search_cell_id_matches_the_campaign_grid(self) -> None:
        # Checkpoint-qualified: cells A (k0) and G (k1) run the SAME search
        # config, so an unqualified id would merge them -- and cell G's entire
        # job is the checkpoint contrast.
        self.assertEqual(_DRIVER.config_id_for(args()), "d4-s1024-b64-w4@ckpt")

    def test_depth_ladder_cells_are_distinguishable(self) -> None:
        # The Tier-2 ladder differs from Tier 1 only in depth/sims; if those did
        # not reach config_id, a d6 shard would merge into the d4 cell.
        self.assertEqual(
            _DRIVER.config_id_for(args(depth=6, sims=4096)), "d6-s4096-b64-w4@ckpt"
        )

    def test_raw_arm_is_search_config_independent_but_checkpoint_qualified(self) -> None:
        # One raw arm PER CHECKPOINT pairs with every search cell on that
        # checkpoint, so its id must not carry search axes (or it cannot be
        # reused) but must carry the checkpoint (or R0 and R1 pool, and the raw
        # arm is the denominator of every paired delta).
        self.assertEqual(_DRIVER.config_id_for(args(arm="raw")), "raw@ckpt")
        self.assertEqual(
            _DRIVER.config_id_for(args(arm="raw", depth=6, worlds=16)), "raw@ckpt"
        )
        self.assertNotEqual(
            _DRIVER.config_id_for(args(arm="raw", checkpoint="/c/k0.pt")),
            _DRIVER.config_id_for(args(arm="raw", checkpoint="/c/k1.pt")),
        )


class SelectionKnobIdentityTest(unittest.TestCase):
    """The selection knobs must SPLIT a cell, and must split nothing else.

    Two cells differing only in fpu_reduction or c_puct would otherwise carry
    one config_id, and every downstream read is keyed on that string: the
    merger would pool a stage-2 arm into its own control and report the pooled
    mean as the arm's delta. Pooling produces a number, so nothing errors.

    The other half is the reason the suffixes are conditional: the banked
    depth-panel and axis-study shards were written before either knob existed,
    and they must stay mergeable with an untuned cell rendered today.
    """

    def test_the_default_string_did_not_move(self) -> None:
        # Byte-identical to the pre-knob id, asserted as a LITERAL rather than
        # against a re-derivation, so a change to the builder cannot agree with
        # itself. This is the same string as ConfigIdTest above and that is the
        # point -- it is the compatibility claim, spelled out where it is made.
        self.assertEqual(_DRIVER.config_id_for(args()), "d4-s1024-b64-w4@ckpt")

    def test_each_knob_alone_changes_the_id(self) -> None:
        default = _DRIVER.config_id_for(args())
        fpu = _DRIVER.config_id_for(args(engine_fpu_reduction=0.2))
        cpuct = _DRIVER.config_id_for(args(engine_c_puct=0.8))
        self.assertEqual(fpu, "d4-s1024-b64-w4-fpu0.2@ckpt")
        self.assertEqual(cpuct, "d4-s1024-b64-w4-c0.8@ckpt")
        self.assertEqual(len({default, fpu, cpuct}), 3)

    def test_the_knobs_compose_in_the_plans_order(self) -> None:
        # The plan's deliverable names the config d{}-s{}-b{}-w{}-op{}-fpu{}-c{}.
        self.assertEqual(
            _DRIVER.config_id_for(
                args(opponent_priors=True, engine_fpu_reduction=0.2, engine_c_puct=2.0)
            ),
            "d4-s1024-b64-w4+opp-priors-fpu0.2-c2@ckpt",
        )

    def test_two_fpu_values_do_not_pool(self) -> None:
        # The stage-2 panel is r in {0.1, 0.2, 0.3} on one base cell, so these
        # three ids are the only thing keeping the three arms apart.
        ids = {_DRIVER.config_id_for(args(engine_fpu_reduction=r))
               for r in (0.1, 0.2, 0.3)}
        self.assertEqual(len(ids), 3, ids)

    def test_fpu_zero_is_a_setting_not_an_absence(self) -> None:
        # Some(0.0) prices an unvisited arm at the parent mean; None is the flat
        # 0.5. A truthiness test here would merge that arm into the control.
        self.assertEqual(
            _DRIVER.config_id_for(args(engine_fpu_reduction=0.0)),
            "d4-s1024-b64-w4-fpu0@ckpt",
        )
        self.assertNotEqual(
            _DRIVER.config_id_for(args(engine_fpu_reduction=0.0)),
            _DRIVER.config_id_for(args()),
        )

    def test_an_explicit_default_c_puct_lands_on_the_banked_control(self) -> None:
        # Stage 3 reads {0.8, 1.1, 2.0} against the stage-2 winner, which was run
        # before c_puct was a knob and so carries no suffix. A cell that spells
        # 1.4 out must pool with it -- it IS the same search -- or the control
        # becomes an empty cell.
        self.assertEqual(_DRIVER.BRIDGE_DEFAULT_C_PUCT, 1.4)
        self.assertEqual(
            _DRIVER.config_id_for(args(engine_c_puct=1.4)),
            _DRIVER.config_id_for(args()),
        )

    def test_the_same_value_typed_two_ways_is_one_cell(self) -> None:
        # A campaign JSON may carry 2 or 2.0 for the same arm.
        self.assertEqual(
            _DRIVER.config_id_for(args(engine_c_puct=2)),
            _DRIVER.config_id_for(args(engine_c_puct=2.0)),
        )

    def test_the_raw_arm_ignores_the_knobs(self) -> None:
        # The raw arm runs no search, and one raw shard per checkpoint is the
        # denominator of every tuned cell's delta. Suffixing it would strand it.
        self.assertEqual(
            _DRIVER.config_id_for(
                args(arm="raw", engine_fpu_reduction=0.2, engine_c_puct=0.8)
            ),
            "raw@ckpt",
        )

    def test_a_namespace_predating_the_knobs_raises_rather_than_pools(self) -> None:
        # Read directly, not through getattr: an old caller must fail loudly
        # instead of being handed the control's id.
        stale = args()
        del stale.engine_fpu_reduction
        with self.assertRaises(AttributeError):
            _DRIVER.config_id_for(stale)

    def test_the_merger_builds_the_identical_id(self) -> None:
        """The report's campaign-side builder must agree with the driver's.

        Two builders drifting does not error: depth_reference is populated with
        ids that match no shard, `depth_rule_applied` still reports true, and
        the non-starvation rule never fires. That is a recorded past failure of
        this exact pair, on the checkpoint tag.
        """
        import sys

        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from foulplay_paired_eval import search_config_id

        cell = {"checkpoint": "k1", "arm": "search", "depth": 8, "sims": 2048,
                "batch": 64, "worlds": 1, "fpu_reduction": 0.2, "c_puct": 0.8}
        self.assertEqual(
            search_config_id(
                depth=cell["depth"], sims=cell["sims"], batch=cell["batch"],
                worlds=cell["worlds"], tag=cell["checkpoint"],
                opponent_priors=bool(cell.get("opponent_priors")),
                fpu_reduction=cell.get("fpu_reduction"),
                c_puct=cell.get("c_puct"),
            ),
            _DRIVER.config_id_for(
                args(checkpoint="/c/t.pt", checkpoint_tag="k1", depth=8, sims=2048,
                     batch=64, worlds=1, engine_fpu_reduction=0.2, engine_c_puct=0.8)
            ),
        )


class SelectionKnobForwardingTest(unittest.TestCase):
    """A knob that reaches nothing is this harness's recurring defect.

    So these assert the CHILD ARGV, from a real parse where possible: the
    bridge's `--engine-fpu-reduction` / `--engine-c-puct` are the only path from
    this driver into `EngineMctsConfig`, and an unforwarded flag produces a
    complete, plausible shard measured at the default.
    """

    def test_both_knobs_reach_the_child_when_set(self) -> None:
        argv = _DRIVER.bridge_argv(
            args(engine_fpu_reduction=0.2, engine_c_puct=0.8), seat="p1"
        )
        self.assertEqual(argv[argv.index("--engine-fpu-reduction") + 1], "0.2")
        self.assertEqual(argv[argv.index("--engine-c-puct") + 1], "0.8")

    def test_unset_knobs_leave_the_child_argv_unchanged(self) -> None:
        # The compatibility claim at the argv level: an untuned cell must hand
        # the bridge exactly what it handed it before the knobs existed, so it
        # inherits the crate's flat 0.5 and c_puct 1.4.
        argv = _DRIVER.bridge_argv(args(), seat="p1")
        self.assertNotIn("--engine-fpu-reduction", argv)
        self.assertNotIn("--engine-c-puct", argv)

    def test_fpu_zero_is_forwarded(self) -> None:
        argv = _DRIVER.bridge_argv(args(engine_fpu_reduction=0.0), seat="p1")
        self.assertEqual(argv[argv.index("--engine-fpu-reduction") + 1], "0.0")

    def test_the_raw_arm_carries_neither(self) -> None:
        argv = _DRIVER.bridge_argv(
            args(arm="raw", engine_fpu_reduction=0.2, engine_c_puct=0.8), seat="p1"
        )
        self.assertNotIn("--engine-fpu-reduction", argv)
        self.assertNotIn("--engine-c-puct", argv)

    def test_the_real_cli_yields_the_dests_bridge_argv_reads(self) -> None:
        # CLI -> Namespace -> child argv, on one real parse. Every other test
        # here hand-builds a Namespace, so a renamed add_argument would leave
        # the suite green while a real shard died inside bridge_argv.
        ns = _DRIVER.build_parser().parse_args([
            "--checkpoint", "/tmp/ckpt.pt", "--showdown-root", "/tmp/showdown",
            "--arm", "search", "--seed-start", "1", "--pairs", "2",
            "--out", "/tmp/shard.json",
            "--engine-fpu-reduction", "0.3", "--engine-c-puct", "1.1",
        ])
        self.assertEqual(ns.engine_fpu_reduction, 0.3)
        self.assertEqual(ns.engine_c_puct, 1.1)
        argv = _DRIVER.bridge_argv(ns, seat="p2")
        self.assertEqual(argv[argv.index("--engine-fpu-reduction") + 1], "0.3")
        self.assertEqual(argv[argv.index("--engine-c-puct") + 1], "1.1")
        self.assertEqual(_DRIVER.config_id_for(ns), "d4-s1024-b64-w4-fpu0.3-c1.1@ckpt")

    def test_the_cli_defaults_are_both_none(self) -> None:
        ns = _DRIVER.build_parser().parse_args([
            "--checkpoint", "/tmp/ckpt.pt", "--showdown-root", "/tmp/showdown",
            "--arm", "search", "--seed-start", "1", "--pairs", "2",
            "--out", "/tmp/shard.json",
        ])
        self.assertIsNone(ns.engine_fpu_reduction)
        self.assertIsNone(ns.engine_c_puct)

    def test_the_bridge_declares_both_flags(self) -> None:
        """The far end of the chain, asserted against the bridge's real parser.

        Without this the driver could emit a flag no bridge accepts, and the
        failure would be a dead shard at game 0 rather than a wrong number --
        but it would be a dead shard per cell, discovered on the cluster.
        """
        from pokezero.foulplay_bridge import build_arg_parser

        dests = {a.dest for a in build_arg_parser()._actions}
        self.assertIn("engine_fpu_reduction", dests)
        self.assertIn("engine_c_puct", dests)
        parsed = build_arg_parser().parse_args([
            "--checkpoint", "/tmp/c.pt",
            "--engine-fpu-reduction", "0.2", "--engine-c-puct", "0.8",
        ])
        self.assertEqual(parsed.engine_fpu_reduction, 0.2)
        self.assertEqual(parsed.engine_c_puct, 0.8)


class DynamicRangeIdentityTest(unittest.TestCase):
    """Dynamic depth/world RANGES, which the owner specified as "min N max M".

    `--depth` and `--worlds` remain the maxima; the floors turn each axis dynamic.
    The id renders the range, so a cell reads as the budget policy it is.
    """

    def test_the_fixed_id_is_unchanged_when_no_floor_is_set(self) -> None:
        # The compatibility claim, as a literal: every banked fixed-budget cell
        # must keep byte-for-byte the id it already has.
        self.assertEqual(_DRIVER.config_id_for(args()), "d4-s1024-b64-w4@ckpt")

    def test_a_depth_range_renders_as_a_range(self) -> None:
        # "depth 3 min 6 max, everything else the same"
        self.assertEqual(
            _DRIVER.config_id_for(args(depth=6, engine_depth_min=3)),
            "d3-6-s1024-b64-w4@ckpt",
        )

    def test_a_worlds_range_renders_as_a_range(self) -> None:
        # "worlds 2 min 16 max, depth 2 min 6 max"
        self.assertEqual(
            _DRIVER.config_id_for(
                args(depth=6, engine_depth_min=2, worlds=16, engine_worlds_min=2)),
            "d2-6-s1024-b64-w2-16@ckpt",
        )

    def test_two_ranges_sharing_a_cap_do_not_pool(self) -> None:
        # d3-6 and d5-6 are different budget policies with the same maximum. If
        # the id ignored the floor they would merge and each would be reported as
        # the other's control.
        self.assertNotEqual(
            _DRIVER.config_id_for(args(depth=6, engine_depth_min=3)),
            _DRIVER.config_id_for(args(depth=6, engine_depth_min=5)),
        )

    def test_a_floor_equal_to_the_cap_pools_with_the_fixed_cell(self) -> None:
        # min == max IS the fixed budget, so it must land on the fixed id rather
        # than acquire a cell of its own with no control behind it.
        self.assertEqual(
            _DRIVER.config_id_for(args(depth=4, engine_depth_min=4)),
            _DRIVER.config_id_for(args()),
        )

    def test_both_floors_reach_the_child(self) -> None:
        argv = _DRIVER.bridge_argv(
            args(depth=6, engine_depth_min=3, worlds=16, engine_worlds_min=2), seat="p1")
        self.assertEqual(argv[argv.index("--engine-depth-min") + 1], "3")
        self.assertEqual(argv[argv.index("--engine-worlds-min") + 1], "2")

    def test_unset_floors_leave_the_child_argv_unchanged(self) -> None:
        argv = _DRIVER.bridge_argv(args(), seat="p1")
        self.assertNotIn("--engine-depth-min", argv)
        self.assertNotIn("--engine-worlds-min", argv)

    def test_the_report_builder_stays_in_lockstep(self) -> None:
        report = (REPO_ROOT / "scripts" / "foulplay_power_report.py").read_text()
        self.assertIn('depth_min=cell.get("depth_min")', report)
        self.assertIn('worlds_min=cell.get("worlds_min")', report)


class DynamicBudgetPassthroughTest(unittest.TestCase):
    """The dynamic per-decision budget (docs/dynamic-search-budget-plan-20260812.md).

    The stop rule itself is pre-existing and was LATENT: `EngineMctsConfig.early_stop`
    defaulted False and no harness could reach it, which `events.rs:2388` states in as
    many words. What is new is reachability, so what is pinned here is the chain and
    the cell identity -- NOT the rule, which `tree.rs::root_visit_lock_is_strict_about_
    remaining_simulations` already covers.

    Unlike the override-telemetry flag this one DOES change the search: it changes how
    many simulations a decision receives. So it must SPLIT the cell, and the test that
    matters most is the one asserting it does.
    """

    def test_the_flag_reaches_the_child_when_set(self) -> None:
        self.assertIn(
            "--engine-early-stop",
            _DRIVER.bridge_argv(args(engine_early_stop=True), seat="p1"),
        )

    def test_unset_leaves_the_child_argv_unchanged(self) -> None:
        argv = _DRIVER.bridge_argv(args(), seat="p1")
        self.assertNotIn("--engine-early-stop", argv)
        self.assertNotIn("--engine-early-stop-min-sims", argv)

    def test_the_floor_is_forwarded_only_with_the_switch(self) -> None:
        argv = _DRIVER.bridge_argv(
            args(engine_early_stop=True, engine_early_stop_min_sims=256), seat="p1")
        self.assertIn("--engine-early-stop-min-sims", argv)
        self.assertEqual(argv[argv.index("--engine-early-stop-min-sims") + 1], "256")

    def test_the_raw_arm_carries_it_not(self) -> None:
        # The raw arm runs no search, so there is no budget to make dynamic.
        self.assertNotIn(
            "--engine-early-stop",
            _DRIVER.bridge_argv(args(arm="raw", engine_early_stop=True), seat="p1"),
        )

    def test_it_IS_part_of_config_id(self) -> None:
        # The opposite of the telemetry flag, and the reason is the measurement:
        # early-stop-on against the same config off is the whole experiment, so
        # pooling them would erase it.
        self.assertNotEqual(
            _DRIVER.config_id_for(args(engine_early_stop=True)),
            _DRIVER.config_id_for(args()),
        )
        self.assertEqual(
            _DRIVER.config_id_for(args(engine_early_stop=True)),
            "d4-s1024-b64-w4+early-stop@ckpt",
        )

    def test_the_default_floor_pools_with_an_unset_floor(self) -> None:
        # A cell that spells out the bridge default must land on the same id as a
        # cell that leaves it unset, or one budget policy acquires two cells.
        self.assertEqual(
            _DRIVER.config_id_for(args(
                engine_early_stop=True,
                engine_early_stop_min_sims=_DRIVER.BRIDGE_DEFAULT_EARLY_STOP_MIN_SIMS)),
            _DRIVER.config_id_for(args(engine_early_stop=True)),
        )

    def test_two_floors_do_not_pool(self) -> None:
        ids = {
            _DRIVER.config_id_for(args(engine_early_stop=True,
                                       engine_early_stop_min_sims=n))
            for n in (128, 256, 512)
        }
        self.assertEqual(len(ids), 3, ids)

    def test_the_id_builders_stay_in_lockstep(self) -> None:
        # scripts/foulplay_power_report.cid_of builds this same id from a campaign
        # cell dict, and search_config_id's own docstring warns that the two
        # drifting apart is a SILENT failure -- the reference matches no shard, so
        # the rule that depends on it never fires. It has happened once already.
        import inspect

        sig = inspect.signature(_DRIVER.search_config_id).parameters
        self.assertIn("early_stop", sig)
        self.assertIn("early_stop_min_sims", sig)
        report = (REPO_ROOT / "scripts" / "foulplay_power_report.py").read_text()
        self.assertIn("early_stop=bool(cell.get(\"early_stop\"))", report)
        self.assertIn("early_stop_min_sims=cell.get(\"early_stop_min_sims\")", report)

    def test_the_bridge_declares_both_flags(self) -> None:
        # Read FROM the bridge rather than retyped: a divergence renders a shard
        # that dies on its own argument after the pod has claimed its GPUs.
        bridge = (REPO_ROOT / "src" / "pokezero" / "foulplay_bridge.py").read_text()
        self.assertIn('"--engine-early-stop"', bridge)
        self.assertIn('"--engine-early-stop-min-sims"', bridge)


class OpponentJournalPassthroughTest(unittest.TestCase):
    """H4 was CANNOT RUN because the moves it needs were recorded and then dropped.

    The bridge decodes FoulPlay's choice every round in every mode but "off", but
    the DEFAULT mode "addressed" retains only the prefix up to a battle's last
    refusal address and returns an empty tuple when a battle has none. On a healthy
    shard almost no battle has one, so this campaign's own s2048 canary produced a
    journal header reading `recorded_decisions: 35` beside `emitted_decisions: 0`
    and `games_with_journal: 0`. Nothing errored; the measurement simply was not
    there. Only "full" emits the moves, and before this the driver could not ask
    for it -- the same one-layer-up gap that would have killed the override probe.

    Observational, so it is pinned in both directions like the telemetry flag: it
    must REACH the bridge, and it must NOT enter config_id.
    """

    def test_the_mode_reaches_the_child_when_set(self) -> None:
        argv = _DRIVER.bridge_argv(args(opponent_journal="full"), seat="p1")
        self.assertIn("--opponent-journal", argv)
        self.assertEqual(argv[argv.index("--opponent-journal") + 1], "full")

    def test_unset_leaves_the_child_argv_unchanged(self) -> None:
        self.assertNotIn("--opponent-journal", _DRIVER.bridge_argv(args(), seat="p1"))

    def test_the_raw_arm_carries_it_too(self) -> None:
        # NOT scoped to search arms: FoulPlay's own choice exists in every arm, and
        # raw is a comparator in the value-gap campaign rather than a spectator.
        # This is the one passthrough here that must survive arm == "raw".
        self.assertIn(
            "--opponent-journal",
            _DRIVER.bridge_argv(args(arm="raw", opponent_journal="full"), seat="p1"),
        )

    def test_the_real_cli_parses_it_and_forwards_it(self) -> None:
        ns = _DRIVER.build_parser().parse_args([
            "--checkpoint", "/tmp/ckpt.pt", "--showdown-root", "/tmp/showdown",
            "--arm", "search", "--seed-start", "1", "--pairs", "2",
            "--out", "/tmp/shard.json", "--opponent-journal", "full",
        ])
        self.assertEqual(ns.opponent_journal, "full")
        self.assertIn("--opponent-journal", _DRIVER.bridge_argv(ns, seat="p2"))

    def test_the_cli_default_is_unset_not_addressed(self) -> None:
        # Unset must mean "say nothing", not "say the bridge default out loud":
        # an explicit --opponent-journal addressed would change the child argv of
        # every banked cell and break byte-identity with the shards already on disk.
        ns = _DRIVER.build_parser().parse_args([
            "--checkpoint", "/tmp/ckpt.pt", "--showdown-root", "/tmp/showdown",
            "--arm", "search", "--seed-start", "1", "--pairs", "2",
            "--out", "/tmp/shard.json",
        ])
        self.assertIsNone(ns.opponent_journal)

    def test_a_bad_mode_is_refused_by_the_parser(self) -> None:
        with self.assertRaises(SystemExit):
            _DRIVER.build_parser().parse_args([
                "--checkpoint", "/tmp/ckpt.pt", "--showdown-root", "/tmp/showdown",
                "--arm", "search", "--seed-start", "1", "--pairs", "2",
                "--out", "/tmp/shard.json", "--opponent-journal", "everything",
            ])

    def test_the_bridge_declares_the_flag_and_the_modes_agree(self) -> None:
        # The driver's choices must be the bridge's modes, read from the bridge
        # rather than retyped -- a divergence here renders a shard that dies on
        # its own argument after the pod is scheduled.
        from pokezero.foulplay_bridge import OPPONENT_JOURNAL_MODES

        action = next(
            a for a in _DRIVER.build_parser()._actions
            if a.dest == "opponent_journal"
        )
        self.assertEqual(tuple(action.choices), tuple(OPPONENT_JOURNAL_MODES))

    def test_it_is_not_part_of_config_id(self) -> None:
        # Emission-only: recording is unconditional and retention is decided after
        # the battle is over, so two cells differing only in journal mode are the
        # same search and MUST pool.
        self.assertEqual(
            _DRIVER.config_id_for(args(opponent_journal="full")),
            _DRIVER.config_id_for(args()),
        )
        # And structurally: the id builder takes no journal parameter at all, so
        # there is no way for a future caller to thread one in by accident.
        import inspect

        self.assertNotIn(
            "journal",
            str(inspect.signature(_DRIVER.search_config_id)),
        )


class OverrideTelemetryPassthroughTest(unittest.TestCase):
    """The value-gap plan's §2 measurement cannot be switched on from a shard.

    That was the state of this driver at the commit that ADDED the telemetry: the
    bridge grew `--engine-override-telemetry` and the config field defaulting to
    False, and nothing here could reach it -- so a campaign shard would have
    emitted no telemetry at all and §2 would have come back empty, after the
    GPU-hours were spent. Same shape as the stage-2 death on
    `--engine-fpu-reduction`, one layer up.

    The flag is OBSERVATIONAL, so it is pinned in two directions at once: it must
    REACH the bridge, and it must NOT reach config_id.
    """

    def test_the_flag_reaches_the_child_when_set(self) -> None:
        self.assertIn(
            "--engine-override-telemetry",
            _DRIVER.bridge_argv(args(engine_override_telemetry=True), seat="p1"),
        )

    def test_unset_leaves_the_child_argv_unchanged(self) -> None:
        self.assertNotIn(
            "--engine-override-telemetry", _DRIVER.bridge_argv(args(), seat="p1")
        )

    def test_the_raw_arm_carries_it_not(self) -> None:
        # The raw arm runs no search, so there is no override to measure.
        self.assertNotIn(
            "--engine-override-telemetry",
            _DRIVER.bridge_argv(
                args(arm="raw", engine_override_telemetry=True), seat="p1"
            ),
        )

    def test_the_real_cli_parses_it_and_forwards_it(self) -> None:
        # CLI -> Namespace -> child argv on one real parse, for the same reason
        # the selection knobs get this test: a hand-built Namespace cannot see a
        # renamed add_argument.
        ns = _DRIVER.build_parser().parse_args([
            "--checkpoint", "/tmp/ckpt.pt", "--showdown-root", "/tmp/showdown",
            "--arm", "search", "--seed-start", "1", "--pairs", "2",
            "--out", "/tmp/shard.json", "--engine-override-telemetry",
        ])
        self.assertTrue(ns.engine_override_telemetry)
        self.assertIn(
            "--engine-override-telemetry", _DRIVER.bridge_argv(ns, seat="p2")
        )

    def test_the_cli_default_is_off(self) -> None:
        ns = _DRIVER.build_parser().parse_args([
            "--checkpoint", "/tmp/ckpt.pt", "--showdown-root", "/tmp/showdown",
            "--arm", "search", "--seed-start", "1", "--pairs", "2",
            "--out", "/tmp/shard.json",
        ])
        self.assertFalse(ns.engine_override_telemetry)

    def test_the_bridge_declares_and_parses_the_flag(self) -> None:
        """The far end of the chain. A flag no bridge accepts is a dead shard."""
        from pokezero.foulplay_bridge import build_arg_parser

        self.assertIn(
            "engine_override_telemetry", {a.dest for a in build_arg_parser()._actions}
        )
        parsed = build_arg_parser().parse_args(
            ["--checkpoint", "/tmp/c.pt", "--engine-override-telemetry"]
        )
        self.assertTrue(parsed.engine_override_telemetry)
        self.assertFalse(
            build_arg_parser().parse_args(
                ["--checkpoint", "/tmp/c.pt"]
            ).engine_override_telemetry
        )

    def test_the_bridge_config_carries_it_into_the_engine_config(self) -> None:
        # The last hop: bridge CLI -> ControlledFoulPlayConfig ->
        # EngineMctsConfig.override_telemetry. Asserted on the FIELD NAMES rather
        # than by running a search, which needs a checkpoint.
        import dataclasses

        from pokezero.engine_search import EngineMctsConfig
        from pokezero.foulplay_bridge import ControlledFoulPlayConfig

        field = {f.name: f for f in dataclasses.fields(ControlledFoulPlayConfig)}[
            "engine_override_telemetry"
        ]
        self.assertIs(field.default, False)
        self.assertIs(EngineMctsConfig().override_telemetry, False)

    def test_the_flag_is_deliberately_absent_from_config_id(self) -> None:
        """Observational, so telemetry-on and telemetry-off are ONE cell.

        Plan §1/H1 reads the production override rate by turning the instrument on
        in an axis-study cell, and that read needs those shards to pool with the
        banked ones. The behavioural claim underneath this is pinned in
        tests/test_opponent_priors_flag.py::OverrideTelemetryIsObservationalTest
        and rust/pokezero-search/src/tree.rs.
        """
        self.assertEqual(
            _DRIVER.config_id_for(args(engine_override_telemetry=True)),
            _DRIVER.config_id_for(args()),
        )
        self.assertEqual(
            _DRIVER.config_id_for(args(engine_override_telemetry=True)),
            "d4-s1024-b64-w4@ckpt",
        )
        # Positive control on the same builder: a knob that DOES change the
        # search still splits the cell, so the equality above is a statement
        # about this flag and not about a builder that ignores everything.
        self.assertNotEqual(
            _DRIVER.config_id_for(args(engine_fpu_reduction=0.2)),
            _DRIVER.config_id_for(args()),
        )

    def test_the_shard_body_witnesses_the_instrument(self) -> None:
        # Since config_id omits it, the report body is the ONLY place a reader can
        # learn whether the rate in this shard has a denominator. Asserted against
        # the module source, because building a real report needs two bridge runs.
        source = (
            Path(__file__).resolve().parents[1] / "scripts" / "foulplay_paired_eval.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"override_telemetry": bool(args.engine_override_telemetry)', source)


class OracleBeliefPassthroughTest(unittest.TestCase):
    """§4a's arm: search the TRUE hidden state, same config and seeds.

    Unlike the telemetry flag this one CHANGES the search -- it replaces every
    sampled belief world with the true completion -- so it must split the cell.
    Its sampled twin is the same cell with the flag off, which is why the
    fragment has to exist: pooled, the plan's centerpiece figure would be the
    average of the two arms it is trying to contrast.
    """

    def test_the_flag_splits_the_cell(self) -> None:
        self.assertEqual(
            _DRIVER.config_id_for(args(engine_oracle_belief=True)),
            "d4-s1024-b64-w4+oracle-belief@ckpt",
        )
        self.assertNotEqual(
            _DRIVER.config_id_for(args(engine_oracle_belief=True)),
            _DRIVER.config_id_for(args()),
        )

    def test_off_leaves_every_banked_id_byte_identical(self) -> None:
        # The compatibility half, same as every knob before it.
        self.assertEqual(_DRIVER.config_id_for(args()), "d4-s1024-b64-w4@ckpt")
        self.assertEqual(
            _DRIVER.config_id_for(
                args(opponent_priors=True, engine_fpu_reduction=0.3, engine_c_puct=0.8)
            ),
            "d4-s1024-b64-w4+opp-priors-fpu0.3-c0.8@ckpt",
        )

    def test_it_composes_with_the_selection_knobs(self) -> None:
        self.assertEqual(
            _DRIVER.config_id_for(
                args(opponent_priors=True, engine_oracle_belief=True,
                     engine_fpu_reduction=0.3, engine_c_puct=0.8)
            ),
            "d4-s1024-b64-w4+opp-priors+oracle-belief-fpu0.3-c0.8@ckpt",
        )

    def test_the_merger_builds_the_identical_id(self) -> None:
        # foulplay_power_report.cid_of imports search_config_id, so the only way
        # the two can drift is a keyword the campaign side forgets to pass -- and
        # a drifted id matches no shard while reporting success.
        import sys

        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from foulplay_paired_eval import search_config_id

        cell = {"checkpoint": "k0", "arm": "search", "depth": 8, "sims": 16384,
                "batch": 64, "worlds": 1, "oracle_belief": True}
        self.assertEqual(
            search_config_id(
                depth=cell["depth"], sims=cell["sims"], batch=cell["batch"],
                worlds=cell["worlds"], tag=cell["checkpoint"],
                opponent_priors=bool(cell.get("opponent_priors")),
                fpu_reduction=cell.get("fpu_reduction"),
                c_puct=cell.get("c_puct"),
                oracle_belief=bool(cell.get("oracle_belief")),
            ),
            _DRIVER.config_id_for(
                args(checkpoint="/c/t.pt", checkpoint_tag="k0", depth=8, sims=16384,
                     batch=64, worlds=1, engine_oracle_belief=True)
            ),
        )

    def test_the_report_side_passes_the_keyword(self) -> None:
        # cid_of is a closure inside main(), so it cannot be called directly here.
        # The source check is the cheap stand-in for the drift that has already
        # happened once on this pair (the checkpoint tag).
        source = (
            Path(__file__).resolve().parents[1] / "scripts" / "foulplay_power_report.py"
        ).read_text(encoding="utf-8")
        self.assertIn('oracle_belief=bool(cell.get("oracle_belief"))', source)

    def test_the_flag_reaches_the_child_only_on_a_search_arm(self) -> None:
        self.assertIn(
            "--engine-oracle-belief",
            _DRIVER.bridge_argv(args(engine_oracle_belief=True), seat="p1"),
        )
        self.assertNotIn(
            "--engine-oracle-belief", _DRIVER.bridge_argv(args(), seat="p1")
        )
        self.assertNotIn(
            "--engine-oracle-belief",
            _DRIVER.bridge_argv(args(arm="raw", engine_oracle_belief=True), seat="p1"),
        )

    def test_the_real_cli_parses_it_and_forwards_it(self) -> None:
        ns = _DRIVER.build_parser().parse_args([
            "--checkpoint", "/tmp/ckpt.pt", "--showdown-root", "/tmp/showdown",
            "--arm", "search", "--seed-start", "1", "--pairs", "2",
            "--out", "/tmp/shard.json", "--engine-oracle-belief",
        ])
        self.assertTrue(ns.engine_oracle_belief)
        self.assertIn("--engine-oracle-belief", _DRIVER.bridge_argv(ns, seat="p2"))
        self.assertEqual(
            _DRIVER.config_id_for(ns), "d4-s1024-b64-w4+oracle-belief@ckpt"
        )

    def test_the_bridge_declares_and_parses_the_flag(self) -> None:
        from pokezero.foulplay_bridge import build_arg_parser

        self.assertIn(
            "engine_oracle_belief", {a.dest for a in build_arg_parser()._actions}
        )
        parsed = build_arg_parser().parse_args(
            ["--checkpoint", "/tmp/c.pt", "--engine-oracle-belief"]
        )
        self.assertTrue(parsed.engine_oracle_belief)
        self.assertFalse(
            build_arg_parser().parse_args(
                ["--checkpoint", "/tmp/c.pt"]
            ).engine_oracle_belief
        )


class FoulplayPathTest(unittest.TestCase):
    """The driver must be able to point the bridge at a relocated foul-play.

    The bridge defaults to a repo-relative checkout and has no environment
    fallback, so a deployment that ships foul-play elsewhere cannot reach it. A
    whole campaign probe died on that: every shard raised FileNotFoundError for
    .../third_party/foul-play/.venv/bin/python while the image had it at
    /opt/foul-play.
    """

    def test_paths_are_forwarded_when_given(self) -> None:
        argv = _DRIVER.bridge_argv(
            args(foulplay_root="/opt/foul-play",
                 foulplay_python="/opt/foul-play/.venv/bin/python"),
            seat="p1",
        )
        self.assertEqual(argv[argv.index("--foulplay-root") + 1], "/opt/foul-play")
        self.assertEqual(
            argv[argv.index("--foulplay-python") + 1], "/opt/foul-play/.venv/bin/python"
        )

    def test_omitted_paths_leave_the_bridge_default_alone(self) -> None:
        # Passing an empty value would override the bridge default with nothing.
        argv = _DRIVER.bridge_argv(args(), seat="p1")
        self.assertNotIn("--foulplay-root", argv)
        self.assertNotIn("--foulplay-python", argv)

    def test_empty_path_does_not_override_the_bridge_default(self) -> None:
        # Truthiness, not `is not None`: forwarding "" hands the bridge Path("")
        # and overrides its default with nothing. Review found this case asserted
        # only in a comment, so the `is not None` mutant survived.
        argv = _DRIVER.bridge_argv(args(foulplay_root="", foulplay_python=""), seat="p1")
        self.assertNotIn("--foulplay-root", argv)
        self.assertNotIn("--foulplay-python", argv)

    def test_the_real_cli_yields_the_dests_bridge_argv_reads(self) -> None:
        """Pin CLI -> Namespace -> bridge_argv, not bridge_argv alone.

        Every other test here hand-builds a Namespace, so deleting or renaming an
        add_argument left the suite green while a real run would die with
        AttributeError inside bridge_argv -- the exact "only shows up in
        production" failure these flags exist to prevent. Found by review.
        """
        parser = _DRIVER.build_parser()
        ns = parser.parse_args([
            "--checkpoint", "/tmp/ckpt.pt", "--showdown-root", "/tmp/showdown",
            "--arm", "raw", "--seed-start", "1", "--pairs", "2",
            "--out", "/tmp/shard.json",
            "--foulplay-root", "/opt/foul-play",
            "--foulplay-python", "/opt/foul-play/.venv/bin/python",
        ])
        self.assertEqual(ns.foulplay_root, "/opt/foul-play")
        self.assertEqual(ns.foulplay_python, "/opt/foul-play/.venv/bin/python")
        # And they survive the trip through bridge_argv on a real parse.
        built = _DRIVER.bridge_argv(ns, seat="p1")
        self.assertEqual(built[built.index("--foulplay-root") + 1], "/opt/foul-play")

    def test_the_cli_defaults_leave_both_paths_unset(self) -> None:
        ns = _DRIVER.build_parser().parse_args([
            "--checkpoint", "/tmp/ckpt.pt", "--showdown-root", "/tmp/showdown",
            "--arm", "raw", "--seed-start", "1", "--pairs", "2",
            "--out", "/tmp/shard.json",
        ])
        self.assertIsNone(ns.foulplay_root)
        self.assertIsNone(ns.foulplay_python)
        self.assertNotIn("--foulplay-root", _DRIVER.bridge_argv(ns, seat="p1"))

    def test_the_required_argument_set_is_pinned(self) -> None:
        """N5/N6: nothing pinned which arguments are required.

        The refactor provably dropped no required=True (review diffed every
        parser action, 19 vs 19), but the tests did not PIN the set, so a later
        change could quietly make --out or --checkpoint optional and a shard
        would run with a default it should never have. Cheap to pin now that the
        parser is reachable; it was unreachable before the extraction.
        """
        required = {a.dest for a in _DRIVER.build_parser()._actions if a.required}
        self.assertEqual(
            required,
            {"checkpoint", "showdown_root", "arm", "seed_start", "pairs", "out"},
        )

    def test_main_parses_through_build_parser(self) -> None:
        """N8: build_parser() is tested, but nothing asserted main() USES it.

        A divergent re-inlined parser in main() would leave every other test
        green. main() is one line today, so drift takes deliberate effort -- but
        the assertion is two lines.
        """
        calls = []
        real = _DRIVER.build_parser

        def spy():
            calls.append(1)
            return real()

        _DRIVER.build_parser = spy
        try:
            with self.assertRaises(SystemExit):
                _DRIVER.main([])  # no required args -> argparse exits 2
        finally:
            _DRIVER.build_parser = real
        self.assertEqual(len(calls), 1, "main() did not parse through build_parser()")

    def test_paths_are_forwarded_on_both_arms_and_seats(self) -> None:
        # The raw arm drives the same opponent, so it needs the same path.
        for arm in ("search", "raw"):
            for seat in ("p1", "p2"):
                with self.subTest(arm=arm, seat=seat):
                    argv = _DRIVER.bridge_argv(
                        args(arm=arm, foulplay_root="/opt/foul-play"), seat=seat
                    )
                    self.assertIn("--foulplay-root", argv)


class OpponentDefinitionTest(unittest.TestCase):
    def test_foulplay_budget_is_pinned_across_every_arm(self) -> None:
        # Part of the opponent definition, not a tuning knob: an arm that faced a
        # different FoulPlay budget is measuring a different opponent.
        self.assertEqual(_DRIVER.FOULPLAY_SEARCH_TIME_MS, 1000)
        for arm in ("search", "raw"):
            with self.subTest(arm=arm):
                argv = _DRIVER.bridge_argv(args(arm=arm), seat="p1")
                self.assertIn("--search-time-ms", argv)
                self.assertEqual(argv[argv.index("--search-time-ms") + 1], "1000")

    def test_thread_pin_is_the_full_family_set(self) -> None:
        # The July-30 jobs used OMP/MKL=2; unpinned BLAS in a CPU-capped pod
        # weakens the FoulPlay side specifically.
        self.assertEqual(
            set(_DRIVER.THREAD_PIN_ENV),
            {
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "POKEZERO_TORCH_NUM_THREADS", "POKEZERO_TORCH_NUM_INTEROP_THREADS",
                "OMP_DYNAMIC",
            },
        )
        self.assertEqual(_DRIVER.THREAD_PIN_ENV["OMP_NUM_THREADS"], "1")


class BridgeArgvTest(unittest.TestCase):
    def test_search_arm_passes_every_engine_axis(self) -> None:
        argv = _DRIVER.bridge_argv(args(depth=6, sims=2048, worlds=8), seat="p2")
        self.assertEqual(argv[argv.index("--policy-mode") + 1], "engine-mcts")
        self.assertEqual(argv[argv.index("--engine-depth") + 1], "6")
        self.assertEqual(argv[argv.index("--engine-sims") + 1], "2048")
        self.assertEqual(argv[argv.index("--engine-worlds") + 1], "8")
        self.assertEqual(argv[argv.index("--pokezero-player") + 1], "p2")

    def test_raw_arm_passes_no_engine_axes(self) -> None:
        argv = _DRIVER.bridge_argv(args(arm="raw"), seat="p1")
        self.assertEqual(argv[argv.index("--policy-mode") + 1], "raw")
        for flag in ("--engine-depth", "--engine-sims", "--engine-worlds"):
            self.assertNotIn(flag, argv)

    def test_both_seats_share_one_seed_band(self) -> None:
        # Within-seed pairing: the seats must face the SAME seeds, or the seat
        # comparison silently becomes a team comparison.
        p1 = _DRIVER.bridge_argv(args(), seat="p1")
        p2 = _DRIVER.bridge_argv(args(), seat="p2")
        self.assertEqual(
            p1[p1.index("--seed-start") + 1], p2[p2.index("--seed-start") + 1]
        )
        self.assertEqual(p1[p1.index("--games") + 1], p2[p2.index("--games") + 1])


class SeedJoinTest(unittest.TestCase):
    def test_rows_are_keyed_by_seed_not_position(self) -> None:
        summary = {"game_results": [game(7800002), game(7800000), game(7800001)]}
        rows = _DRIVER.per_seed_outcomes(summary, "p1")
        self.assertEqual(sorted(rows), [7800000, 7800001, 7800002])
        self.assertEqual(rows[7800002]["seed"], 7800002)

    def test_missing_score_key_is_fatal_not_a_zero(self) -> None:
        # The regression that matters: defaulting this to 0.0 would read as a
        # perfect-loss arm, i.e. a huge and entirely fake paired delta.
        broken = {"game_results": [{"seed": 1, "pokezero_won": True}]}
        with self.assertRaises(SystemExit) as caught:
            _DRIVER.per_seed_outcomes(broken, "p1")
        self.assertIn("pokezero_score", str(caught.exception))

    def test_scores_are_read_through_verbatim(self) -> None:
        summary = {"game_results": [game(1, score=0.5, won=False)]}
        self.assertEqual(_DRIVER.per_seed_outcomes(summary, "p1")[1]["score"], 0.5)


class SeatBlockTest(unittest.TestCase):
    def test_latency_gate_field_is_surfaced_separately_from_policy_timing(self) -> None:
        summary = {
            "completed_games": 200,
            "engine_mcts": {
                "search_wall_per_searched_decision": 4.2,
                "fallback_rate": 0.008,
                "policy_stats": {"depth_reached_mean": 3.1, "depth_reached_max": 4},
            },
            "policy_timing": {"average_elapsed_seconds": 3.8, "p95_elapsed_seconds": 6.2},
        }
        block = _DRIVER.seat_block(summary, "p1")
        # Both walls present and NOT conflated: the gate reads the first.
        self.assertEqual(block["search_wall_per_searched_decision"], 4.2)
        self.assertEqual(block["wall_per_decision_mean"], 3.8)
        self.assertEqual(block["wall_per_decision_p95"], 6.2)
        self.assertEqual(block["depth_reached_mean"], 3.1)

    def test_a_dynamic_cell_surfaces_the_per_decision_wall_as_well(self) -> None:
        # THREE walls on a ladder cell, and the reason: the gate field is
        # per-RUNG there. `searched_decisions` is charged once per `_search_model`
        # call and a ladder calls it once per rung, so a cell measured at 2,224 rungs
        # against 1,062 decisions reported 4.24 s when the true per-decision cost was
        # 8.88 s -- published once as a 23% saving when it was a 2x regression.
        # Hoisted out of `policy_stats` so the analysis cannot reach for the per-rung
        # figure by habit.
        summary = {
            "completed_games": 200,
            "engine_mcts": {
                "search_wall_per_searched_decision": 4.24,
                "policy_stats": {
                    "search_wall_per_ladder_decision": 8.88,
                    "ladder_rungs_per_decision": 2.094,
                },
            },
        }
        block = _DRIVER.seat_block(summary, "p1")
        self.assertEqual(block["search_wall_per_searched_decision"], 4.24)
        self.assertEqual(block["search_wall_per_ladder_decision"], 8.88)
        self.assertAlmostEqual(block["ladder_rungs_per_decision"], 2.094)

    def test_a_fixed_cell_reports_no_ladder_wall_rather_than_a_zero(self) -> None:
        # None, not 0.0: a fixed cell has no rungs, and a 0.0 would read as "free".
        summary = {
            "completed_games": 200,
            "engine_mcts": {"search_wall_per_searched_decision": 12.51, "policy_stats": {}},
        }
        block = _DRIVER.seat_block(summary, "p1")
        self.assertEqual(block["search_wall_per_searched_decision"], 12.51)
        self.assertIsNone(block["search_wall_per_ladder_decision"])
        self.assertIsNone(block["ladder_rungs_per_decision"])

    def test_raw_arm_block_survives_absent_engine_telemetry(self) -> None:
        block = _DRIVER.seat_block({"completed_games": 200, "wins": 90}, "p2")
        self.assertIsNone(block["search_wall_per_searched_decision"])
        self.assertIsNone(block["depth_reached_mean"])
        self.assertEqual(block["games"], 200)


class OpponentPriorsLiftedTest(unittest.TestCase):
    """Cells B/E were refused until the opponent map's ordering was verified.

    LIFTED 2026-08-11. This class used to assert the refusal; it now asserts the
    contract the refusal was protecting, so the flag cannot regress into being
    accepted-but-inert. A wrong mapping does not fail -- it reports a confident
    paired delta from permuted priors, and the campaign reads that as "opponent
    priors do not help" -- so the guard is kept, pointed the other way.
    """

    def test_opponent_priors_are_no_longer_refused(self) -> None:
        # The refusal was a SystemExit carrying "REFUSED" raised in main()
        # before any game. Assert the string is gone from the driver rather than
        # invoking main(), which needs torch and a real checkpoint.
        source = (REPO_ROOT / "scripts" / "foulplay_paired_eval.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("REFUSED pending review", source)

    def test_the_flag_actually_reaches_the_bridge(self) -> None:
        # The half that matters now: accepted AND forwarded. An accepted flag
        # that never reaches the bridge would run the whole grid with uniform
        # opponent priors and report it as the opponent-priors arm.
        argv = _DRIVER.bridge_argv(args(opponent_priors=True), seat="p1")
        self.assertIn("--engine-opponent-priors", argv)

    def test_the_cell_identity_records_the_flag(self) -> None:
        # An opponent-priors cell must not pool with its own control.
        self.assertEqual(
            _DRIVER.config_id_for(args(opponent_priors=True)),
            "d4-s1024-b64-w4+opp-priors@ckpt",
        )
        self.assertNotEqual(
            _DRIVER.config_id_for(args(opponent_priors=True)),
            _DRIVER.config_id_for(args()),
        )

    def test_the_default_path_is_unaffected(self) -> None:
        # The refusal must not touch the nine cells that do not use the flag.
        argv = _DRIVER.bridge_argv(args(), seat="p1")
        self.assertNotIn("--engine-opponent-priors", argv)
        self.assertEqual(_DRIVER.config_id_for(args()), "d4-s1024-b64-w4@ckpt")

    def test_raw_arm_never_carries_the_flag(self) -> None:
        argv = _DRIVER.bridge_argv(args(arm="raw", opponent_priors=True), seat="p1")
        self.assertNotIn("--engine-opponent-priors", argv)


class NoDuplicateDefinitionsTest(unittest.TestCase):
    """Catches an editing accident that Python silently tolerates.

    A bad amend spliced a second copy of this module's helpers and `main` into
    the file. Python keeps the LAST definition, so the module still imported and
    every test passed -- while the surviving `main` was the stale one and the
    corrected code sat above it as dead lines. The commit message reported the
    fix as landed.

    There is no linter configured in this repo and no lint job in CI, so an
    F811 would not be reported by anything. This asserts it directly, for the
    two campaign scripts whose `main` is the actual entry point the launcher
    shells out to.
    """

    def _duplicate_defs(self, relative):
        import ast
        from collections import Counter

        tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))
        names = [
            node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        return {name: n for name, n in Counter(names).items() if n > 1}

    def test_paired_eval_defines_each_top_level_name_once(self) -> None:
        self.assertEqual(self._duplicate_defs("scripts/foulplay_paired_eval.py"), {})

    def test_power_report_defines_each_top_level_name_once(self) -> None:
        self.assertEqual(self._duplicate_defs("scripts/foulplay_power_report.py"), {})

    def test_the_lift_records_its_evidence(self) -> None:
        # The refusal named what would lift it. Now that it is lifted, the
        # evidence must stay at the site so the next reader can RE-CHECK rather
        # than trust -- a lift whose justification has been deleted is
        # indistinguishable from a lift nobody justified.
        source = (REPO_ROOT / "scripts" / "foulplay_paired_eval.py").read_text(
            encoding="utf-8"
        )
        body = source[source.index("def main("):]
        self.assertIn("LIFTED 2026-08-11", body)
        # the three merged reviews
        for pr in ("#1194", "#1192", "#1207"):
            self.assertIn(pr, body, f"lift no longer cites {pr}")
        # the real-checkpoint evidence, which is the gap the refusal named
        self.assertIn("iteration-1563", body)
        self.assertIn("prior_fallbacks == 0", body)
        # the determinism control, without which the comparison proves nothing
        self.assertIn("IDENTICAL", body)
        self.assertNotIn("second switch onward", body,
                         "stale round-4 rationale is back in the live guard")




class HeadToHeadCellIdentityTest(unittest.TestCase):
    """The opponent must be part of the cell identity.

    Demonstrated in review: without it, a banked vs-foul-play shard and a head-to-head
    vs-raw shard both render `d4-s512-b8-w16@c`, and foulplay_power_report.collect_rows
    merges them into ONE pooled win rate with no warning. Two experiments against different
    opponents, at different scales, with different nulls, become a single number. If their
    seed bands overlap it is worse -- the conflicting-scores check hard-exits the report.
    """

    def _f(self, **kw):
        import importlib.util as u
        from pathlib import Path
        sp = u.spec_from_file_location(
            "pe", Path(__file__).resolve().parents[1] / "scripts" / "foulplay_paired_eval.py")
        m = u.module_from_spec(sp); sp.loader.exec_module(m)
        base = dict(depth=4, sims=512, batch=8, worlds=16, tag="c")
        base.update(kw)
        return m.search_config_id(**base)

    def _through_production(self, *extra):
        """config_id_for + build_parser -- the path a real shard takes.

        The previous revision tested search_config_id DIRECTLY and passed while the bug was
        live: the fragment was added to the helper but config_id_for, its only production
        caller, never forwarded it, so every real shard still rendered the pooled id. A test
        that does not traverse the production entry point cannot see that.
        """
        import importlib.util as u
        from pathlib import Path
        sp = u.spec_from_file_location(
            "pe", Path(__file__).resolve().parents[1] / "scripts" / "foulplay_paired_eval.py")
        m = u.module_from_spec(sp); sp.loader.exec_module(m)
        base = ["--checkpoint", "/c", "--showdown-root", "/s", "--arm", "search",
                "--seed-start", "1", "--pairs", "5", "--depth", "4", "--sims", "512",
                "--batch", "8", "--worlds", "16", "--checkpoint-tag", "k0", "--out", "/o.json"]
        return m.config_id_for(m.build_parser().parse_args(base + list(extra)))

    def test_the_PRODUCTION_id_carries_the_opponent(self) -> None:
        default = self._through_production()
        vs_raw = self._through_production("--opponent-policy-mode", "raw")
        vs_d6 = self._through_production(
            "--opponent-policy-mode", "engine-mcts",
            "--opponent-engine-depth", "6", "--opponent-engine-sims", "16384")
        self.assertEqual(default, "d4-s512-b8-w16@k0", "banked ids must not move")
        self.assertNotEqual(vs_raw, default)
        self.assertNotEqual(vs_d6, default)
        self.assertNotEqual(vs_raw, vs_d6)
        self.assertIn("+vs-raw", vs_raw)
        self.assertIn("vs-engine-mcts-d6-s16384", vs_d6)

    def test_the_report_builder_stays_in_lockstep_with_the_driver(self) -> None:
        """Both builders must move together.

        The shared docstring warns that drift is a SILENT failure -- the reference simply
        matches no shard -- and it has already happened once on the checkpoint tag.
        """
        import importlib.util as u
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        sp = u.spec_from_file_location("pr", root / "scripts" / "foulplay_power_report.py")
        pr = u.module_from_spec(sp); sp.loader.exec_module(pr)
        src = (root / "scripts" / "foulplay_power_report.py").read_text()
        for name in ("opponent_policy_mode", "opponent_engine_depth", "opponent_engine_sims"):
            self.assertIn(name, src,
                          f"{name} must reach the report's cid_of or the two ids drift")

    def test_the_banked_vs_foulplay_id_is_byte_identical(self) -> None:
        """Load-bearing: every banked shard must stay mergeable across this change."""
        self.assertEqual(self._f(), "d4-s512-b8-w16@c")
        self.assertEqual(self._f(opponent_policy_mode="foul-play"), "d4-s512-b8-w16@c")

    def test_a_head_to_head_cell_gets_a_distinct_id(self) -> None:
        self.assertNotEqual(self._f(opponent_policy_mode="raw"), self._f())
        self.assertIn("+vs-raw", self._f(opponent_policy_mode="raw"))

    def test_two_budgets_against_each_other_carry_BOTH_budgets(self) -> None:
        cid = self._f(depth=3, sims=2048, batch=16, worlds=1,
                      opponent_policy_mode="engine-mcts",
                      opponent_engine_depth=6, opponent_engine_sims=16384)
        self.assertIn("d3-s2048", cid)
        self.assertIn("vs-engine-mcts-d6-s16384", cid)

    def test_cells_differing_only_in_the_opponent_budget_do_not_pool(self) -> None:
        a = self._f(opponent_policy_mode="engine-mcts", opponent_engine_depth=6,
                    opponent_engine_sims=16384)
        b = self._f(opponent_policy_mode="engine-mcts", opponent_engine_depth=2,
                    opponent_engine_sims=1024)
        self.assertNotEqual(a, b, "different opponents are different experiments")


if __name__ == "__main__":
    unittest.main()
