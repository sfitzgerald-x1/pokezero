from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
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


_BRANCH_REASONS = (
    "empty_action_map",
    "unmapped_action",
    "action_index_out_of_range",
    "invalid_mapped_mass",
    "missing_model_head_row",
    "decision_arm_count_mismatch",
)


def _live_branch_prior():
    reasons = {reason: 0 for reason in _BRANCH_REASONS}
    zero_witness = {
        seat: {
            name: 0
            for name in (
                "nodes",
                "move_arms",
                "move_arms_engine_missing",
                "move_arms_present_but_illegal",
                "move_arms_order_unavailable",
                "move_arms_unexplained",
                "switch_arms",
                "none_arms",
            )
        }
        for seat in ("acting", "opponent")
    }
    return {
        "prior_fallbacks": 0,
        "branch_prior_fallbacks": {
            "schema_version": "pokezero.engine-mcts.branch-prior-fallbacks.v1",
            "native_invocations": 1,
            "belief_worlds": 4,
            "branch_prior_fallbacks": 0,
            "reason_counts": reasons,
            "unclassified_branch_prior_fallbacks": 0,
            "reason_ledger_complete": True,
            "unmapped_action_witness": zero_witness,
            "events": [{
                "native_invocation": 1,
                "belief_records": 1,
                "collapse_multiplicity": 1,
                "branch_prior_fallbacks": 0,
                "reason_counts": dict(reasons),
                "unmapped_action_witness": zero_witness,
            }],
        },
    }


def _selection(*, rollout_witness=None):
    return {
        "root_action": "move 1",
        "total_iterations": 4096,
        "model_evals": 1024,
        "max_depth_reached": 6,
        "root_allocation": {
            "worlds": 4,
            "arms": [
                {"move": "move 1", "action_index": 0, "visit_share": .6, "reported_prior": .7, "model_prior": .7, "q": .1},
                {"move": "move 2", "action_index": 1, "visit_share": .4, "reported_prior": .3, "model_prior": .3, "q": .0},
            ],
        },
        "rollout_leaf": rollout_witness,
        "live_branch_prior": _live_branch_prior(),
    }


class SourceRootLeafAblationRunnerTest(unittest.TestCase):
    def test_target_roster_is_exact_and_has_no_duplicate_address(self) -> None:
        runner = _runner()
        self.assertEqual(len(runner.TARGETS), 16)
        self.assertEqual(len(set(runner.TARGETS)), 16)
        self.assertEqual(
            [(target.seed, target.seat, target.turn_index) for target in runner.TARGETS],
            [
                (2026092004, "p1", 19), (2026092004, "p1", 20),
                (2026092005, "p1", 21), (2026092005, "p1", 25),
                (2026092005, "p2", 1), (2026092005, "p2", 6),
                (2026092006, "p1", 2), (2026092006, "p1", 3),
                (2026092007, "p1", 9), (2026092007, "p1", 19),
                (2026092007, "p2", 9),
                (2026092006, "p2", 118), (2026092006, "p2", 119),
                (2026092006, "p2", 124), (2026092006, "p2", 126),
                (2026092006, "p2", 127),
            ],
        )
        self.assertEqual(
            runner.FALLBACK_TARGETS,
            frozenset(runner.TARGETS[-5:]),
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

    def test_historical_fallback_witness_is_bound_to_the_exact_raw_selection(self) -> None:
        runner = _runner()
        root = runner.SourceRoot(2026092006, "p2", 119)
        record = SimpleNamespace(
            decision_id="d" * 64,
            acting_player="p2",
            turn_index=119,
            recorded_action_index=3,
        )
        with tempfile.TemporaryDirectory() as temporary:
            source_root = Path(temporary)
            directory = (
                source_root / "seeds" / "seed-2026092006" / "branch-prior-fallback-ledgers"
                / "seed-2026092006-p2"
            )
            directory.mkdir(parents=True)
            path = directory / "turn-119-witness.json"
            payload = {
                "public_decision": {
                    "decision_id": record.decision_id,
                    "acting_player": record.acting_player,
                    "turn_index": record.turn_index,
                    "recorded_action_index": record.recorded_action_index,
                },
                "branch_prior_fallbacks": {
                    "branch_prior_fallbacks": 7,
                    "reason_ledger_complete": True,
                    "unclassified_branch_prior_fallbacks": 0,
                    "reason_counts": {"unmapped_action": 7, "empty_action_map": 0},
                },
                "selection": {
                    "model_argmax": 3,
                    "search_argmax": 3,
                    "model_override": False,
                    "root_allocation": {"arms": []},
                },
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            witness = runner._load_historical_fallback(source_root, root, record)
            self.assertEqual(witness["branch_prior_fallbacks"], 7)
            self.assertEqual(witness["selection"], payload["selection"])
            payload["selection"]["search_argmax"] = 6
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(runner.AblationError, "historical raw selection"):
                runner._load_historical_fallback(source_root, root, record)

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
                runner._parse_args([*required, "--finalize-only", "--shard-count", "4", "--shard-index", "1"])
        sharded = runner._parse_args([*required, "--shard-count", "4", "--shard-index", "3"])
        self.assertEqual((sharded.shard_index, sharded.shard_count), (3, 4))
        finalizer = runner._parse_args([*required, "--finalize-only", "--shard-count", "4"])
        self.assertEqual((finalizer.shard_index, finalizer.shard_count), (0, 4))

    def test_parser_binds_optional_source_repair_identities_exactly(self) -> None:
        runner = _runner()
        required = [
            "--checkpoint", "checkpoint.pt",
            "--expected-checkpoint-sha256", "a" * 64,
            "--showdown-root", "/showdown",
            "--source-root", "/r4",
            "--out-root", "/out",
            "--expected-source-commit", "b" * 40,
            "--expected-engine-fingerprint", "c" * 64,
        ]
        args = runner._parse_args(required)
        self.assertEqual(args.expected_source_commit, "b" * 40)
        self.assertEqual(args.expected_engine_fingerprint, "c" * 64)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                runner._parse_args([*required[:-2], "--expected-engine-fingerprint", "invalid"])
            with self.assertRaises(SystemExit):
                runner._parse_args([*required[:-4], "--expected-engine-fingerprint", "c" * 64])
            with self.assertRaises(SystemExit):
                runner._parse_args([*required[:-4], "--expected-source-commit", "b" * 40])

    def test_execution_runtime_requires_explicit_repaired_source_and_engine_identities(self) -> None:
        runner = _runner()
        source_commit = "b" * 40
        repaired_engine = "c" * 64
        historical_engine = "d" * 64
        showdown = "e" * 64
        fake_engine = ModuleType("engine_build_fingerprint")
        fake_engine.assert_fresh = lambda: None
        fake_engine.compute_fingerprint = lambda: {"fingerprint": repaired_engine}
        args = SimpleNamespace(
            expected_source_commit=source_commit,
            expected_engine_fingerprint=repaired_engine,
            showdown_root="/showdown",
        )
        historical = {
            "historical_runtime": {
                "showdown_source": {"content_sha256": showdown},
                "engine_fingerprint": historical_engine,
            }
        }
        with (
            mock.patch.object(runner, "_source_provenance", return_value={"commit": source_commit}),
            mock.patch.object(runner, "_showdown_source_provenance", return_value={"content_sha256": showdown}),
            mock.patch.dict(sys.modules, {"engine_build_fingerprint": fake_engine}),
        ):
            runtime = runner._execution_runtime(args, historical)
        self.assertEqual(runtime["engine_identity_mode"], "source_repair")
        self.assertEqual(runtime["historical_engine_fingerprint"], historical_engine)
        self.assertEqual(runtime["expected_engine_fingerprint"], repaired_engine)

        incomplete = SimpleNamespace(
            expected_source_commit=source_commit,
            expected_engine_fingerprint=None,
            showdown_root="/showdown",
        )
        with self.assertRaisesRegex(runner.AblationError, "requires both explicitly bound"):
            runner._execution_runtime(incomplete, historical)

        args.expected_source_commit = "f" * 40
        with mock.patch.object(runner, "_source_provenance", return_value={"commit": source_commit}):
            with self.assertRaisesRegex(runner.AblationError, "explicitly bound source-repair commit"):
                runner._execution_runtime(args, historical)

    def test_decision_seed_is_stable_and_root_specific(self) -> None:
        runner = _runner()
        first = SimpleNamespace(decision_id="a")
        second = SimpleNamespace(decision_id="b")
        self.assertEqual(runner._decision_seed(first), runner._decision_seed(first))
        self.assertNotEqual(runner._decision_seed(first), runner._decision_seed(second))
        self.assertNotEqual(runner._decision_seed(first), runner._rollout_seed(first))
        self.assertEqual(runner._rollout_seed(first), runner._rollout_seed(first))

    def test_control_projection_covers_selection_allocation_and_work(self) -> None:
        runner = _runner()
        first = _selection()
        second = _selection()
        self.assertEqual(runner._control_projection(first), runner._control_projection(second))
        second["root_allocation"]["arms"][0]["visit_share"] = .5
        self.assertNotEqual(runner._control_projection(first), runner._control_projection(second))

    def test_selection_rejects_missing_duplicate_or_out_of_range_action_identity(self) -> None:
        runner = _runner()
        with mock.patch.object(runner, "require_rollout_leaf_witness"):
            missing = _selection()
            del missing["root_allocation"]["arms"][0]["action_index"]
            with self.assertRaisesRegex(runner.AblationError, "action indices"):
                runner._validate_persisted_selection(missing, rollout_leaf_eval=False)
            duplicate = _selection()
            duplicate["root_allocation"]["arms"][1]["action_index"] = 0
            with self.assertRaisesRegex(runner.AblationError, "not unique"):
                runner._validate_persisted_selection(duplicate, rollout_leaf_eval=False)
            out_of_range = _selection()
            out_of_range["root_allocation"]["arms"][1]["action_index"] = runner.ACTION_COUNT
            with self.assertRaisesRegex(runner.AblationError, "action indices"):
                runner._validate_persisted_selection(out_of_range, rollout_leaf_eval=False)

    def test_persisted_selection_refuses_nonzero_or_incomplete_live_branch_prior_ledger(self) -> None:
        runner = _runner()
        with mock.patch.object(runner, "require_rollout_leaf_witness"):
            nonzero = _selection()
            nonzero["live_branch_prior"]["branch_prior_fallbacks"]["branch_prior_fallbacks"] = 1
            nonzero["live_branch_prior"]["branch_prior_fallbacks"]["reason_counts"]["unmapped_action"] = 1
            nonzero["live_branch_prior"]["branch_prior_fallbacks"]["events"][0]["branch_prior_fallbacks"] = 1
            nonzero["live_branch_prior"]["branch_prior_fallbacks"]["events"][0]["reason_counts"]["unmapped_action"] = 1
            with self.assertRaisesRegex(runner.AblationError, "reports a fallback|zero decomposition"):
                runner._validate_persisted_selection(nonzero, rollout_leaf_eval=False)
            incomplete = _selection()
            del incomplete["live_branch_prior"]["branch_prior_fallbacks"]["reason_counts"]["unmapped_action"]
            with self.assertRaisesRegex(runner.AblationError, "complete zero decomposition"):
                runner._validate_persisted_selection(incomplete, rollout_leaf_eval=False)

    def test_persisted_selection_accepts_native_shaped_zero_unmapped_action_witness(self) -> None:
        runner = _runner()
        with mock.patch.object(runner, "require_rollout_leaf_witness"):
            runner._validate_persisted_selection(_selection(), rollout_leaf_eval=False)
            corrupted = _selection()
            corrupted["live_branch_prior"]["branch_prior_fallbacks"]["events"][0][
                "unmapped_action_witness"
            ]["acting"]["nodes"] = 1
            with self.assertRaisesRegex(runner.AblationError, "not all zero"):
                runner._validate_persisted_selection(corrupted, rollout_leaf_eval=False)

    def test_completed_root_refuses_control_drift(self) -> None:
        runner = _runner()
        root = runner.SourceRoot(2026092004, "p1", 19)
        record = SimpleNamespace(to_dict=lambda: {"record": "bound"})
        payload = {
            "schema_version": runner.SCHEMA_VERSION,
            "state": "COMPLETE",
            "manifest_sha256": "m" * 64,
            "source": root.to_dict(),
            "record_sha256": runner._sha256(record.to_dict()),
            "historical_branch_prior_fallback": None,
            "arms": {
                "model_control_a": {"selection": _selection()},
                "model_control_b": {"selection": _selection()},
                "rollout_leaf": {"selection": _selection(rollout_witness={"priced": 1})},
            },
        }
        with mock.patch.object(runner, "require_rollout_leaf_witness") as witness:
            runner._validate_completed_root(
                payload, root=root, record=record, historical_fallback=None, manifest_sha256="m" * 64
            )
            self.assertEqual(witness.call_count, 3)
            self.assertEqual(
                [call.kwargs["rollout_leaf_eval"] for call in witness.call_args_list],
                [False, False, True],
            )
            payload["arms"]["model_control_b"]["selection"]["root_action"] = "move 2"
            with self.assertRaisesRegex(runner.AblationError, "controls disagree"):
                runner._validate_completed_root(
                    payload, root=root, record=record, historical_fallback=None, manifest_sha256="m" * 64
                )

    def test_completed_root_refuses_a_foreign_manifest(self) -> None:
        runner = _runner()
        root = runner.SourceRoot(2026092004, "p1", 19)
        record = SimpleNamespace(to_dict=lambda: {"record": "bound"})
        payload = {
            "schema_version": runner.SCHEMA_VERSION,
            "state": "COMPLETE",
            "manifest_sha256": "old" * 21 + "x",
            "source": root.to_dict(),
            "record_sha256": runner._sha256(record.to_dict()),
            "historical_branch_prior_fallback": None,
            "arms": {
                "model_control_a": {"selection": _selection()},
                "model_control_b": {"selection": _selection()},
                "rollout_leaf": {"selection": _selection(rollout_witness={"priced": 1})},
            },
        }
        with mock.patch.object(runner, "require_rollout_leaf_witness"):
            with self.assertRaisesRegex(runner.AblationError, "does not bind this run manifest"):
                runner._validate_completed_root(
                    payload, root=root, record=record, historical_fallback=None,
                    manifest_sha256="new" * 21 + "x"
                )

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

    def test_sharded_worker_requires_serial_prepare(self) -> None:
        runner = _runner()
        with tempfile.TemporaryDirectory() as temporary:
            out_root = Path(temporary) / "out"
            with self.assertRaisesRegex(runner.AblationError, "prepare-only"):
                runner._prepare_root(
                    out_root, {"schema_version": "test"}, resume=True, require_existing_manifest=True
                )
            runner._prepare_root(
                out_root, {"schema_version": "test"}, resume=False, require_existing_manifest=False
            )
            runner._prepare_root(
                out_root, {"schema_version": "test"}, resume=True, require_existing_manifest=True
            )
