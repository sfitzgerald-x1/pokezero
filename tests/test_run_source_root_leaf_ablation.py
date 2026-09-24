from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


def _runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_source_root_leaf_ablation.py"
    spec = importlib.util.spec_from_file_location("source_root_leaf_ablation_test_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _selection(*, rollout_witness=None):
    return {
        "root_action": "move 1",
        "total_iterations": 4096,
        "model_evals": 1024,
        "max_depth_reached": 6,
        "root_allocation": {
            "worlds": 4,
            "arms": [
                {"move": "move 1", "visit_share": .6, "reported_prior": .7, "model_prior": .7, "q": .1},
                {"move": "move 2", "visit_share": .4, "reported_prior": .3, "model_prior": .3, "q": .0},
            ],
        },
        "rollout_leaf": rollout_witness,
    }


class SourceRootLeafAblationRunnerTest(unittest.TestCase):
    def test_target_roster_is_exact_and_has_no_duplicate_address(self) -> None:
        runner = _runner()
        self.assertEqual(len(runner.TARGETS), 11)
        self.assertEqual(len(set(runner.TARGETS)), 11)
        self.assertEqual(
            [(target.seed, target.seat, target.turn_index) for target in runner.TARGETS],
            [
                (2026092004, "p1", 19), (2026092004, "p1", 20),
                (2026092005, "p1", 21), (2026092005, "p1", 25),
                (2026092005, "p2", 1), (2026092005, "p2", 6),
                (2026092006, "p1", 2), (2026092006, "p1", 3),
                (2026092007, "p1", 9), (2026092007, "p1", 19),
                (2026092007, "p2", 9),
            ],
        )

    def test_source_paths_are_targeted_not_a_full_corpus_glob(self) -> None:
        runner = _runner()
        root = runner.SourceRoot(2026092005, "p2", 6)
        with tempfile.TemporaryDirectory() as temporary:
            source_root = Path(temporary)
            target = (
                source_root / "seeds" / "seed-2026092005" / "public-decision-records"
                / "seed-2026092005-p2"
            )
            target.mkdir(parents=True)
            path = target / "turn-006-a.json"
            path.write_text("{}", encoding="utf-8")
            self.assertEqual(runner._source_wrapper_path(source_root, root), path)
            (target / "turn-006-b.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(runner.AblationError, "2 records"):
                runner._source_wrapper_path(source_root, root)

    def test_parser_freezes_cuda_only_model_device(self) -> None:
        runner = _runner()
        required = [
            "--checkpoint", "checkpoint.pt",
            "--expected-checkpoint-sha256", "a" * 64,
            "--showdown-root", "/showdown",
            "--source-root", "/r4",
            "--out-root", "/out",
        ]
        self.assertEqual(runner._parse_args(required).model_device, "cuda")
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                runner._parse_args([*required, "--model-device", "cpu"])
            with self.assertRaises(SystemExit):
                runner._parse_args([*required, "--shard-count", "4", "--shard-index", "4"])
            with self.assertRaises(SystemExit):
                runner._parse_args([*required, "--finalize-only", "--shard-count", "4"])
        sharded = runner._parse_args([*required, "--shard-count", "4", "--shard-index", "3"])
        self.assertEqual((sharded.shard_index, sharded.shard_count), (3, 4))

    def test_decision_seed_is_stable_and_root_specific(self) -> None:
        runner = _runner()
        first = SimpleNamespace(decision_id="a")
        second = SimpleNamespace(decision_id="b")
        self.assertEqual(runner._decision_seed(first), runner._decision_seed(first))
        self.assertNotEqual(runner._decision_seed(first), runner._decision_seed(second))

    def test_control_projection_covers_selection_allocation_and_work(self) -> None:
        runner = _runner()
        first = _selection()
        second = _selection()
        self.assertEqual(runner._control_projection(first), runner._control_projection(second))
        second["root_allocation"]["arms"][0]["visit_share"] = .5
        self.assertNotEqual(runner._control_projection(first), runner._control_projection(second))

    def test_completed_root_refuses_control_drift(self) -> None:
        runner = _runner()
        root = runner.SourceRoot(2026092004, "p1", 19)
        record = SimpleNamespace(to_dict=lambda: {"record": "bound"})
        payload = {
            "schema_version": runner.SCHEMA_VERSION,
            "state": "COMPLETE",
            "source": root.to_dict(),
            "record_sha256": runner._sha256(record.to_dict()),
            "arms": {
                "model_control_a": {"selection": _selection()},
                "model_control_b": {"selection": _selection()},
                "rollout_leaf": {"selection": _selection(rollout_witness={"priced": 1})},
            },
        }
        with mock.patch.object(runner, "require_rollout_leaf_witness") as witness:
            runner._validate_completed_root(payload, root=root, record=record)
            self.assertEqual(witness.call_count, 3)
            self.assertEqual(
                [call.kwargs["rollout_leaf_eval"] for call in witness.call_args_list],
                [False, False, True],
            )
            payload["arms"]["model_control_b"]["selection"]["root_action"] = "move 2"
            with self.assertRaisesRegex(runner.AblationError, "controls disagree"):
                runner._validate_completed_root(payload, root=root, record=record)

    def test_contract_freezes_leaf_and_search_axes(self) -> None:
        runner = _runner()
        self.assertEqual(runner.SEARCH, {"depth": 6, "sims": 4096, "batch": 16, "worlds": 4})
        self.assertEqual(runner.ARMS, ("model_control_a", "model_control_b", "rollout_leaf"))
        self.assertTrue(runner.ROLLOUT["rollout_threads_cpu_budget_ack"])

    def test_historical_baseline_requires_the_r4_fixed_work_config(self) -> None:
        runner = _runner()
        self.assertEqual(
            runner._historical_config_projection(dict(runner.HISTORICAL_MODEL_CONFIG)),
            runner.HISTORICAL_MODEL_CONFIG,
        )
        altered = dict(runner.HISTORICAL_MODEL_CONFIG)
        altered["model_priors"] = False
        with self.assertRaisesRegex(runner.AblationError, "historical fixed-work"):
            runner._historical_config_projection(altered)

    def test_operational_error_is_retryable_not_a_terminal_nonpass(self) -> None:
        runner = _runner()
        with tempfile.TemporaryDirectory() as temporary:
            out_root = Path(temporary) / "out"
            required = [
                "--checkpoint", "checkpoint.pt",
                "--expected-checkpoint-sha256", "a" * 64,
                "--showdown-root", "/showdown",
                "--source-root", "/r4",
                "--out-root", str(out_root),
            ]
            with mock.patch.object(runner, "_run", side_effect=runner.AblationError("transient")):
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(runner.main(required), 1)
            self.assertFalse((out_root / "NONPASS.json").exists())
            self.assertTrue((out_root / "last_error.json").exists())
