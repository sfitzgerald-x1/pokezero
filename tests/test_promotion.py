import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from pokezero.eval_cli import (
    OPPONENT_POOL_SNAPSHOT_SCHEMA_VERSION,
    PROMOTION_ARCHIVE_INTEGRITY_SCHEMA_VERSION,
    PROMOTION_RETENTION_APPLY_SCHEMA_VERSION,
    PROMOTION_RETENTION_PLAN_SCHEMA_VERSION,
    main as eval_cli_main,
)
from pokezero.evaluation import PromotionGateConfig
from pokezero.linear_policy import LinearPolicyModel, save_linear_model
from pokezero.opponents import historical_opponent_policy_specs
from pokezero.promotion import (
    PROMOTION_REGISTRY_SCHEMA_VERSION,
    NEURAL_SELFPLAY_SOURCE_TYPE,
    _promotion_registry_lock,
    load_promotion_registry,
    record_promotion,
    verify_promotion_registry,
)
from pokezero.selfplay import SELFPLAY_RUN_SCHEMA_VERSION


class PromotionRegistryTest(unittest.TestCase):
    def test_load_missing_registry_returns_empty_registry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = load_promotion_registry(Path(temp_dir) / "promotions.json")

        self.assertEqual(registry.entries, ())
        self.assertIsNone(registry.latest)

    def test_promotion_registry_lock_warns_when_fcntl_is_unavailable(self) -> None:
        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "fcntl":
                raise ImportError("fcntl unavailable")
            return real_import(name, globals, locals, fromlist, level)

        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = Path(temp_dir) / "promotions.json"

            with patch("builtins.__import__", side_effect=fake_import):
                with self.assertWarnsRegex(RuntimeWarning, "requires fcntl"):
                    with _promotion_registry_lock(registry_path):
                        pass

    def test_record_promotion_writes_gate_passing_checkpoint_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = temp_path / "run" / "manifest.json"
            registry_path = temp_path / "promotions.json"
            write_manifest(manifest_path, selfplay_manifest())

            result = record_promotion(
                manifest_path,
                registry_path=registry_path,
                config=passing_gate_config(),
                label="smoke-promote",
                notes="first accepted checkpoint",
                promoted_at="2026-06-02T00:00:00Z",
            )
            registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
            loaded = load_promotion_registry(registry_path)

        self.assertTrue(result.recorded)
        self.assertEqual(registry_payload["schema_version"], PROMOTION_REGISTRY_SCHEMA_VERSION)
        self.assertEqual(registry_payload["latest_policy_id"], "linear-selfplay-test-iter-0001")
        self.assertEqual(len(loaded.entries), 1)
        self.assertEqual(loaded.latest.label if loaded.latest is not None else None, "smoke-promote")
        self.assertEqual(loaded.latest.gate_result["passed"] if loaded.latest is not None else None, True)

    def test_record_promotion_can_copy_checkpoint_to_artifact_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = temp_path / "run" / "manifest.json"
            registry_path = temp_path / "promotions.json"
            artifact_dir = temp_path / "promoted-checkpoints"
            manifest = selfplay_manifest()
            write_manifest(manifest_path, manifest)
            source_checkpoint = write_checkpoint_for_manifest(temp_path, manifest)

            result = record_promotion(
                manifest_path,
                registry_path=registry_path,
                artifact_dir=artifact_dir,
                config=passing_gate_config(),
                promoted_at="2026-06-02T00:00:00Z",
            )
            loaded = load_promotion_registry(registry_path)
            managed_checkpoint_path = Path(result.entry.checkpoint_path if result.entry else "")
            managed_checkpoint_exists = managed_checkpoint_path.exists()
            managed_checkpoint_payload = json.loads(managed_checkpoint_path.read_text(encoding="utf-8"))
            source_checkpoint_text = source_checkpoint.read_text(encoding="utf-8")
            managed_checkpoint_text = managed_checkpoint_path.read_text(encoding="utf-8")

        self.assertTrue(result.recorded)
        self.assertTrue(managed_checkpoint_exists)
        self.assertEqual(len(result.entry.checkpoint_sha256 if result.entry else ""), 64)
        self.assertEqual(managed_checkpoint_payload["policy_id"], "linear-selfplay-test-iter-0001")
        self.assertEqual(managed_checkpoint_path.parent, artifact_dir)
        self.assertEqual(result.entry.source_checkpoint_path if result.entry else None, "run/iteration-0001/linear-policy.json")
        self.assertEqual(result.entry.checkpoint_path if result.entry else None, str(managed_checkpoint_path))
        self.assertEqual(loaded.latest.checkpoint_path if loaded.latest else None, str(managed_checkpoint_path))
        self.assertEqual(loaded.checkpoint_policy_specs(), (f"linear:{managed_checkpoint_path}",))
        self.assertEqual(source_checkpoint_text, managed_checkpoint_text)

    def test_registry_policy_specs_preserve_neural_checkpoint_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = temp_path / "promotions.json"
            checkpoint_path = temp_path / "transformer-policy.pt"
            checkpoint_path.write_text("checkpoint", encoding="utf-8")
            write_manifest(
                registry_path,
                {
                    "schema_version": PROMOTION_REGISTRY_SCHEMA_VERSION,
                    "registry_path": str(registry_path),
                    "latest_policy_id": "entity-test-iter-0001",
                    "latest_checkpoint_path": str(checkpoint_path),
                    "entries": [
                        {
                            "sequence": 1,
                            "policy_id": "entity-test-iter-0001",
                            "checkpoint_path": str(checkpoint_path),
                            "manifest_path": "runs/neural/manifest.json",
                            "source_type": NEURAL_SELFPLAY_SOURCE_TYPE,
                            "source_iteration": 1,
                            "promoted_at": "2026-06-02T00:00:00Z",
                            "label": None,
                            "notes": None,
                            "gate_result": {"passed": True},
                        }
                    ],
                },
            )

            registry = load_promotion_registry(registry_path)

        self.assertEqual(registry.checkpoint_policy_specs(), (f"neural:{checkpoint_path}",))
        self.assertEqual(registry.latest.checkpoint_policy_spec if registry.latest else None, f"neural:{checkpoint_path}")

    def test_registry_opponent_pool_preview_matches_selfplay_selection_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = write_registry_with_entries(Path(temp_dir), count=4)

            registry = load_promotion_registry(registry_path)
            latest_spec = registry.entries[-1].checkpoint_policy_spec

        self.assertEqual(
            registry.opponent_pool_policy_specs(max_historical_opponents=2),
            historical_opponent_policy_specs(
                registry.checkpoint_policy_specs(),
                current_policy_spec=None,
                max_historical_opponents=2,
            ),
        )
        self.assertEqual(
            registry.opponent_pool_policy_specs(
                max_historical_opponents=2,
                current_policy_spec=latest_spec,
            ),
            historical_opponent_policy_specs(
                registry.checkpoint_policy_specs(),
                current_policy_spec=latest_spec,
                max_historical_opponents=2,
            ),
        )
        self.assertEqual(registry.opponent_pool_policy_specs(max_historical_opponents=0), ())
        with self.assertRaisesRegex(ValueError, "non-negative"):
            registry.opponent_pool_policy_specs(max_historical_opponents=-1)

    def test_registry_opponent_pool_preview_supports_spread_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = write_registry_with_entries(Path(temp_dir), count=5)

            registry = load_promotion_registry(registry_path)
            latest_spec = registry.entries[-1].checkpoint_policy_spec
            specs = registry.checkpoint_policy_specs()

        self.assertEqual(
            registry.opponent_pool_policy_specs(
                max_historical_opponents=2,
                current_policy_spec=latest_spec,
                selection_mode="spread",
            ),
            (specs[0], specs[3]),
        )

    def test_historical_opponent_policy_specs_excludes_current_checkpoint_by_resolved_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            previous_checkpoint = temp_path / "previous-linear.json"
            current_checkpoint = temp_path / "current-linear.json"
            previous_spec = f"linear:{previous_checkpoint.resolve(strict=False)}"
            current_absolute_spec = f"linear:{current_checkpoint.resolve(strict=False)}"
            current_relative_spec = f"linear:{os.path.relpath(current_checkpoint, Path.cwd())}"

            selected = historical_opponent_policy_specs(
                (previous_spec, current_absolute_spec),
                current_policy_spec=current_relative_spec,
                max_historical_opponents=2,
            )

        self.assertEqual(selected, (previous_spec,))

    def test_verify_promotion_registry_can_require_loadable_neural_policy_specs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = temp_path / "promotions.json"
            checkpoint_path = temp_path / "transformer-policy.pt"
            checkpoint_path.write_text("checkpoint", encoding="utf-8")
            write_manifest(
                registry_path,
                {
                    "schema_version": PROMOTION_REGISTRY_SCHEMA_VERSION,
                    "registry_path": str(registry_path),
                    "latest_policy_id": "entity-test-iter-0001",
                    "latest_checkpoint_path": str(checkpoint_path),
                    "entries": [
                        {
                            "sequence": 1,
                            "policy_id": "entity-test-iter-0001",
                            "checkpoint_path": str(checkpoint_path),
                            "manifest_path": "runs/neural/manifest.json",
                            "source_type": NEURAL_SELFPLAY_SOURCE_TYPE,
                            "source_iteration": 1,
                            "promoted_at": "2026-06-02T00:00:00Z",
                            "label": None,
                            "notes": None,
                            "gate_result": {"passed": True},
                        }
                    ],
                },
            )

            with patch(
                "pokezero.neural_policy.load_transformer_policy",
                return_value=SimpleNamespace(policy_id="entity-test-iter-0001"),
            ) as load:
                result = verify_promotion_registry(registry_path, verify_loadable=True)

        self.assertTrue(result.passed)
        self.assertEqual(result.verified_loadable_count, 1)
        load.assert_called_once_with(
            checkpoint_path.resolve(strict=False),
            deterministic=False,
            exploration_epsilon=0.0,
            sampling_temperature=1.0,
            family_gated_selection=False,
        )

    def test_record_promotion_prefers_manifest_relative_checkpoint_over_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with tempfile.TemporaryDirectory() as cwd_dir:
                temp_path = Path(temp_dir)
                manifest_path = temp_path / "run" / "manifest.json"
                registry_path = temp_path / "promotions.json"
                artifact_dir = temp_path / "promoted-checkpoints"
                manifest = selfplay_manifest()
                write_manifest(manifest_path, manifest)
                write_checkpoint_for_manifest(temp_path, manifest, policy_id="correct-checkpoint")
                write_checkpoint_for_manifest(Path(cwd_dir), manifest, policy_id="wrong-cwd-checkpoint")

                previous_cwd = Path.cwd()
                try:
                    os.chdir(cwd_dir)
                    result = record_promotion(
                        manifest_path,
                        registry_path=registry_path,
                        artifact_dir=artifact_dir,
                        config=passing_gate_config(),
                    )
                    managed_checkpoint_path = Path(result.entry.checkpoint_path if result.entry else "")
                    managed_checkpoint_payload = json.loads(managed_checkpoint_path.read_text(encoding="utf-8"))
                finally:
                    os.chdir(previous_cwd)

        self.assertEqual(managed_checkpoint_payload["policy_id"], "correct-checkpoint")

    def test_record_promotion_does_not_write_failed_gate(self) -> None:
        manifest = selfplay_manifest()
        manifest["iterations"][0]["collection_metrics"]["capped_games"] = 5
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = temp_path / "run" / "manifest.json"
            registry_path = temp_path / "promotions.json"
            write_manifest(manifest_path, manifest)

            result = record_promotion(
                manifest_path,
                registry_path=registry_path,
                config=passing_gate_config(),
            )
            registry_exists = registry_path.exists()

        self.assertFalse(result.recorded)
        self.assertFalse(registry_exists)

    def test_record_promotion_rejects_duplicate_checkpoint_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = temp_path / "run" / "manifest.json"
            registry_path = temp_path / "promotions.json"
            write_manifest(manifest_path, selfplay_manifest())
            record_promotion(
                manifest_path,
                registry_path=registry_path,
                config=passing_gate_config(),
            )

            with self.assertRaisesRegex(ValueError, "already promoted"):
                record_promotion(
                    manifest_path,
                    registry_path=registry_path,
                    config=passing_gate_config(),
                )

            loaded = load_promotion_registry(registry_path)

        self.assertEqual(len(loaded.entries), 1)

    def test_record_promotion_allows_same_policy_id_with_distinct_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            first_manifest_path = temp_path / "run-a" / "manifest.json"
            second_manifest_path = temp_path / "run-b" / "manifest.json"
            registry_path = temp_path / "promotions.json"
            first_manifest = selfplay_manifest()
            second_manifest = selfplay_manifest()
            set_manifest_checkpoint(second_manifest, "run-b/iteration-0001/linear-policy.json")
            write_manifest(first_manifest_path, first_manifest)
            write_manifest(second_manifest_path, second_manifest)

            record_promotion(
                first_manifest_path,
                registry_path=registry_path,
                config=passing_gate_config(),
            )
            record_promotion(
                second_manifest_path,
                registry_path=registry_path,
                config=passing_gate_config(),
            )
            loaded = load_promotion_registry(registry_path)

        self.assertEqual(len(loaded.entries), 2)
        self.assertEqual(
            [entry.policy_id for entry in loaded.entries],
            ["linear-selfplay-test-iter-0001", "linear-selfplay-test-iter-0001"],
        )
        self.assertEqual(
            [entry.checkpoint_path for entry in loaded.entries],
            ["run/iteration-0001/linear-policy.json", "run-b/iteration-0001/linear-policy.json"],
        )

    def test_record_promotion_serializes_concurrent_artifact_promotions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            first_manifest_path = temp_path / "run-a" / "manifest.json"
            second_manifest_path = temp_path / "run-b" / "manifest.json"
            registry_path = temp_path / "promotions.json"
            artifact_dir = temp_path / "promoted-checkpoints"
            first_manifest = selfplay_manifest()
            second_manifest = selfplay_manifest()
            set_manifest_identity(
                first_manifest,
                policy_id="linear-concurrent-a",
                checkpoint_path="run-a/iteration-0001/linear-policy.json",
            )
            set_manifest_identity(
                second_manifest,
                policy_id="linear-concurrent-b",
                checkpoint_path="run-b/iteration-0001/linear-policy.json",
            )
            write_manifest(first_manifest_path, first_manifest)
            write_manifest(second_manifest_path, second_manifest)
            write_checkpoint_for_manifest(temp_path, first_manifest)
            write_checkpoint_for_manifest(temp_path, second_manifest)

            processes = [
                start_promotion_subprocess(first_manifest_path, registry_path, artifact_dir),
                start_promotion_subprocess(second_manifest_path, registry_path, artifact_dir),
            ]
            results = [process.communicate(timeout=30) for process in processes]
            return_codes = [process.returncode for process in processes]
            loaded = load_promotion_registry(registry_path)
            artifact_paths = [Path(entry.checkpoint_path or "") for entry in loaded.entries]
            artifact_exists = [path.exists() for path in artifact_paths]
            artifact_parents = [path.parent for path in artifact_paths]

            self.assertEqual(return_codes, [0, 0], results)
            self.assertEqual([entry.sequence for entry in loaded.entries], [1, 2])
            self.assertEqual(
                sorted(entry.policy_id for entry in loaded.entries),
                ["linear-concurrent-a", "linear-concurrent-b"],
            )
            self.assertEqual(len({entry.checkpoint_path for entry in loaded.entries}), 2)
            self.assertEqual(artifact_exists, [True, True])
            self.assertEqual(artifact_parents, [artifact_dir, artifact_dir])

    def test_eval_cli_gate_defaults_incumbent_from_registry_latest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            incumbent_manifest = selfplay_manifest()
            set_manifest_identity(
                incumbent_manifest,
                policy_id="linear-incumbent",
                checkpoint_path="run-incumbent/iteration-0001/linear-policy.json",
            )
            incumbent_manifest_path = temp_path / "run-incumbent" / "manifest.json"
            registry_path = temp_path / "promotions.json"
            write_manifest(incumbent_manifest_path, incumbent_manifest)
            record_promotion(
                incumbent_manifest_path,
                registry_path=registry_path,
                config=passing_gate_config(),
            )

            candidate_manifest = selfplay_manifest()
            set_manifest_identity(
                candidate_manifest,
                policy_id="linear-candidate",
                checkpoint_path="run-candidate/iteration-0001/linear-policy.json",
            )
            add_benchmark_head_to_head(
                candidate_manifest,
                first_policy_id="linear-candidate",
                second_policy_id="linear-incumbent",
                first_policy_wins=18,
                second_policy_wins=2,
                capped_games=0,
            )
            candidate_manifest_path = temp_path / "run-candidate" / "manifest.json"
            write_manifest(candidate_manifest_path, candidate_manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "gate",
                        str(candidate_manifest_path),
                        "--registry",
                        str(registry_path),
                        "--min-benchmark-win-rate",
                        "0.60",
                        "--min-benchmark-games",
                        "20",
                        "--min-incumbent-games",
                        "20",
                        "--max-collection-capped-rate",
                        "0.20",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["gate_mode"], "absolute_floor+incumbent_delta")
        self.assertEqual(payload["incumbent_policy_id"], "linear-incumbent")
        self.assertEqual(payload["incumbent_games"], 20)

    def test_eval_cli_promote_and_promotions_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = temp_path / "run" / "manifest.json"
            registry_path = temp_path / "promotions.json"
            write_manifest(manifest_path, selfplay_manifest())

            with patch("sys.stdout", new_callable=io.StringIO) as promote_stdout:
                promote_exit = eval_cli_main(
                    [
                        "promote",
                        str(manifest_path),
                        "--registry",
                        str(registry_path),
                        "--min-benchmark-win-rate",
                        "0.60",
                        "--min-benchmark-games",
                        "20",
                        "--max-collection-capped-rate",
                        "0.20",
                        "--label",
                        "candidate-a",
                        "--json",
                    ]
                )
            promote_payload = json.loads(promote_stdout.getvalue())

            with patch("sys.stdout", new_callable=io.StringIO) as list_stdout:
                list_exit = eval_cli_main(["promotions", "--registry", str(registry_path), "--json"])
            list_payload = json.loads(list_stdout.getvalue())

        self.assertEqual(promote_exit, 0)
        self.assertTrue(promote_payload["recorded"])
        self.assertEqual(promote_payload["entry"]["label"], "candidate-a")
        self.assertEqual(list_exit, 0)
        self.assertEqual(list_payload["entries"][0]["policy_id"], "linear-selfplay-test-iter-0001")

    def test_eval_cli_promotions_json_defaults_to_excluding_latest_promoted_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = write_registry_with_entries(Path(temp_dir), count=3)
            registry = load_promotion_registry(registry_path)
            latest_spec = registry.latest_selection_checkpoint_policy_spec()

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "2",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            payload["opponent_pool_policy_specs"],
            list(
                historical_opponent_policy_specs(
                    registry.selection_checkpoint_policy_specs(),
                    current_policy_spec=latest_spec,
                    max_historical_opponents=2,
                )
            ),
        )
        self.assertEqual(payload["opponent_pool_excluded_current_policy_spec"], latest_spec)
        self.assertIsNone(payload["opponent_pool_verified"])
        self.assertEqual(len(payload["entry_statuses"]), 3)
        self.assertEqual(payload["entry_statuses"][0]["selected_as"], ["opponent_pool"])
        self.assertEqual(payload["entry_statuses"][0]["opponent_pool_status"], "selected")
        self.assertIsNone(payload["entry_statuses"][0]["opponent_pool_skip_reason"])
        self.assertEqual(payload["entry_statuses"][-1]["selected_as"], ["latest"])
        self.assertEqual(payload["entry_statuses"][-1]["opponent_pool_status"], "excluded_current_policy")
        self.assertEqual(payload["entry_statuses"][-1]["opponent_pool_skip_reason"], "matches_current_policy")
        self.assertEqual(payload["entry_statuses"][-1]["verification_status"], "not_verified")
        self.assertEqual(payload["entry_statuses"][-1]["checkpoint_exists"], "not_verified")
        self.assertEqual(payload["opponent_pool_requested_size"], 2)
        self.assertEqual(payload["opponent_pool_selected_size"], 2)
        self.assertEqual(payload["opponent_pool_available_size"], 2)
        self.assertIsNone(payload["opponent_pool_required_size"])
        self.assertTrue(payload["opponent_pool_requirement_passed"])

    def test_eval_cli_promotions_json_uses_spread_opponent_pool_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = write_registry_with_entries(Path(temp_dir), count=5)
            registry = load_promotion_registry(registry_path)
            latest_spec = registry.latest_selection_checkpoint_policy_spec()

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "2",
                        "--historical-opponent-selection",
                        "spread",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            payload["opponent_pool_policy_specs"],
            list(
                historical_opponent_policy_specs(
                    registry.selection_checkpoint_policy_specs(),
                    current_policy_spec=latest_spec,
                    max_historical_opponents=2,
                    selection_mode="spread",
                )
            ),
        )
        self.assertEqual(payload["historical_opponent_selection"], "spread")
        self.assertEqual(payload["opponent_pool_snapshot"]["historical_opponent_selection"], "spread")

    def test_eval_cli_promotions_json_can_require_opponent_pool_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = write_registry_with_entries(Path(temp_dir), count=3)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "2",
                        "--require-opponent-pool-size",
                        "2",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["opponent_pool_selected_size"], 2)
        self.assertEqual(payload["opponent_pool_available_size"], 2)
        self.assertEqual(payload["opponent_pool_required_size"], 2)
        self.assertTrue(payload["opponent_pool_requirement_passed"])

    def test_eval_cli_promotions_json_marks_entries_outside_requested_pool_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = write_registry_with_entries(Path(temp_dir), count=4)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "2",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            [status["opponent_pool_status"] for status in payload["entry_statuses"]],
            [
                "available_outside_requested_size",
                "selected",
                "selected",
                "excluded_current_policy",
            ],
        )
        self.assertEqual(
            payload["entry_statuses"][0]["opponent_pool_skip_reason"],
            "outside_requested_pool_size",
        )

    def test_eval_cli_promotions_json_can_report_lifecycle_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = write_registry_with_entries(Path(temp_dir), count=4)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "2",
                        "--lifecycle",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())
            lifecycle = payload["lifecycle_summary"]

        self.assertEqual(exit_code, 0)
        self.assertEqual(lifecycle["total_entries"], 4)
        self.assertTrue(lifecycle["opponent_pool_requested"])
        self.assertFalse(lifecycle["verification_enabled"])
        self.assertEqual(lifecycle["latest_count"], 1)
        self.assertEqual(lifecycle["selected_opponent_pool_count"], 2)
        self.assertEqual(lifecycle["selected_opponent_pool_unhealthy_count"], 0)
        self.assertEqual(lifecycle["selection_eligible_count"], 4)
        self.assertEqual(lifecycle["stale_available_count"], 1)
        self.assertEqual(lifecycle["excluded_current_policy_count"], 1)
        self.assertEqual(lifecycle["unselectable_count"], 0)
        self.assertEqual(lifecycle["failed_verification_count"], 0)
        self.assertEqual(lifecycle["registry_level_failed_verification_count"], 0)
        self.assertEqual(
            lifecycle["opponent_pool_status_counts"],
            {
                "available_outside_requested_size": 1,
                "excluded_current_policy": 1,
                "selected": 2,
            },
        )
        self.assertEqual(lifecycle["verification_status_counts"], {"not_verified": 4})

    def test_eval_cli_promotions_lifecycle_marks_pool_not_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = write_registry_with_entries(Path(temp_dir), count=2)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--lifecycle",
                        "--json",
                    ]
                )
            lifecycle = json.loads(stdout.getvalue())["lifecycle_summary"]

        self.assertEqual(exit_code, 0)
        self.assertFalse(lifecycle["opponent_pool_requested"])
        self.assertEqual(lifecycle["selected_opponent_pool_count"], 0)
        self.assertEqual(lifecycle["stale_available_count"], 0)
        self.assertEqual(lifecycle["opponent_pool_status_counts"], {"not_requested": 2})

    def test_eval_cli_promotions_json_marks_entries_without_selection_checkpoint_as_unselectable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = write_registry_with_entries(temp_path, count=3)
            payload = json.loads(registry_path.read_text(encoding="utf-8"))
            payload["entries"][0]["checkpoint_path"] = None
            write_manifest(registry_path, payload)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "2",
                        "--json",
                    ]
                )
            result = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(result["entry_statuses"][0]["opponent_pool_status"], "unselectable")
        self.assertEqual(result["entry_statuses"][0]["opponent_pool_skip_reason"], "missing_selection_checkpoint")

    def test_eval_cli_promotions_text_can_report_lifecycle_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = write_registry_with_entries(temp_path, count=3)
            payload = json.loads(registry_path.read_text(encoding="utf-8"))
            payload["entries"][0]["checkpoint_path"] = None
            write_manifest(registry_path, payload)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "2",
                        "--lifecycle",
                    ]
                )

        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn("lifecycle_summary:", output)
        self.assertIn("- total_entries: 3", output)
        self.assertIn("- opponent_pool_requested: True", output)
        self.assertIn("- selected_opponent_pool_count: 1", output)
        self.assertIn("- selected_opponent_pool_unhealthy_count: 0", output)
        self.assertIn("- selection_eligible_count: 2", output)
        self.assertIn("- unselectable_count: 1", output)
        self.assertIn("opponent_pool_status_counts:", output)
        self.assertIn("- unselectable: 1", output)
        self.assertIn("- selected: 1", output)

    def test_eval_cli_promotions_rejects_required_pool_size_above_requested_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = write_registry_with_entries(Path(temp_dir), count=10)

            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "3",
                        "--require-opponent-pool-size",
                        "5",
                    ]
                )

        self.assertEqual(exit_code, 1)
        self.assertIn("--require-opponent-pool-size cannot exceed --opponent-pool-size", stderr.getvalue())

    def test_eval_cli_promotions_json_fails_when_required_pool_is_too_small(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = write_registry_with_entries(Path(temp_dir), count=2)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "2",
                        "--require-opponent-pool-size",
                        "2",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["opponent_pool_selected_size"], 1)
        self.assertEqual(payload["opponent_pool_available_size"], 1)
        self.assertEqual(payload["opponent_pool_required_size"], 2)
        self.assertFalse(payload["opponent_pool_requirement_passed"])

    def test_eval_cli_promotions_required_pool_size_still_fails_when_verification_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = write_registry_with_entries(temp_path, count=3)
            (temp_path / "checkpoint-1.json").unlink()

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "2",
                        "--require-opponent-pool-size",
                        "2",
                        "--verify",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 2)
        self.assertTrue(payload["opponent_pool_requirement_passed"])
        self.assertFalse(payload["verification"]["passed"])
        self.assertIn("checkpoint_exists", failed_verification_check_names_from_payload(payload["verification"]))

    def test_eval_cli_promotions_can_verify_selected_opponent_pool_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = write_registry_with_entries(temp_path, count=4)
            (temp_path / "checkpoint-1.json").unlink()

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "2",
                        "--require-opponent-pool-size",
                        "2",
                        "--verify",
                        "--verify-opponent-pool-only",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        outside_status = next(status for status in payload["entry_statuses"] if status["sequence"] == 1)
        selected_statuses = [
            status
            for status in payload["entry_statuses"]
            if "opponent_pool" in status["selected_as"]
        ]
        self.assertEqual(exit_code, 0)
        self.assertFalse(payload["verification"]["passed"])
        self.assertFalse(payload["opponent_pool_verified"])
        self.assertTrue(payload["selected_opponent_pool_verified"])
        self.assertTrue(payload["opponent_pool_current_policy_verified"])
        self.assertTrue(payload["opponent_pool_registry_level_verified"])
        self.assertTrue(payload["opponent_pool_preflight_verified"])
        self.assertEqual(payload["opponent_pool_verification_exit_scope"], "opponent_pool_plus_current")
        self.assertEqual(outside_status["checkpoint_exists"], "fail")
        self.assertTrue(selected_statuses)
        self.assertTrue(all(not status["failed_checks"] for status in selected_statuses))

    def test_eval_cli_promotions_can_write_verified_opponent_pool_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = write_registry_with_entries(temp_path, count=3)
            snapshot_path = temp_path / "pool-snapshot.json"

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "2",
                        "--require-opponent-pool-size",
                        "2",
                        "--verify",
                        "--verify-opponent-pool-only",
                        "--write-opponent-pool",
                        str(snapshot_path),
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["opponent_pool_snapshot_path"], str(snapshot_path))
        self.assertEqual(payload["opponent_pool_snapshot"], snapshot)
        self.assertEqual(snapshot["schema_version"], OPPONENT_POOL_SNAPSHOT_SCHEMA_VERSION)
        self.assertEqual(snapshot["registry_path"], str(registry_path.resolve(strict=False)))
        self.assertEqual(snapshot["requested_size"], 2)
        self.assertEqual(snapshot["required_size"], 2)
        self.assertEqual(snapshot["selected_size"], 2)
        self.assertEqual(snapshot["available_size"], 2)
        self.assertTrue(snapshot["requirement_passed"])
        self.assertTrue(snapshot["verification_enabled"])
        self.assertEqual(snapshot["verification_exit_scope"], "opponent_pool_plus_current")
        self.assertTrue(snapshot["preflight_verified"])
        self.assertTrue(snapshot["selected_opponent_pool_verified"])
        self.assertEqual(snapshot["preflight_verified"], payload["opponent_pool_preflight_verified"])
        self.assertEqual(snapshot["current_policy_verified"], payload["opponent_pool_current_policy_verified"])
        self.assertEqual(snapshot["registry_level_verified"], payload["opponent_pool_registry_level_verified"])
        self.assertEqual(snapshot["policy_specs"], payload["opponent_pool_policy_specs"])
        self.assertEqual([entry["sequence"] for entry in snapshot["selected_entries"]], [1, 2])
        self.assertTrue(all(entry["verification_status"] == "partial" for entry in snapshot["selected_entries"]))
        self.assertTrue(all(entry["checkpoint_exists"] == "pass" for entry in snapshot["selected_entries"]))
        self.assertTrue(all(entry["failed_checks"] == [] for entry in snapshot["selected_entries"]))
        self.assertTrue(all("opponent_pool_status" not in entry for entry in snapshot["selected_entries"]))

    def test_eval_cli_promotions_writes_snapshot_when_required_pool_size_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = write_registry_with_entries(temp_path, count=2)
            snapshot_path = temp_path / "pool-snapshot.json"

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "2",
                        "--require-opponent-pool-size",
                        "2",
                        "--write-opponent-pool",
                        str(snapshot_path),
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 2)
        self.assertFalse(payload["opponent_pool_requirement_passed"])
        self.assertFalse(snapshot["requirement_passed"])
        self.assertEqual(snapshot["selected_size"], 1)
        self.assertEqual(snapshot["required_size"], 2)

    def test_eval_cli_promotions_writes_snapshot_when_verification_preflight_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = write_registry_with_entries(temp_path, count=4)
            snapshot_path = temp_path / "pool-snapshot.json"
            (temp_path / "checkpoint-2.json").unlink()

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "2",
                        "--require-opponent-pool-size",
                        "2",
                        "--verify",
                        "--verify-opponent-pool-only",
                        "--write-opponent-pool",
                        str(snapshot_path),
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 2)
        self.assertFalse(payload["opponent_pool_preflight_verified"])
        self.assertFalse(snapshot["preflight_verified"])
        self.assertTrue(snapshot["requirement_passed"])
        self.assertEqual(snapshot["verification_exit_scope"], "opponent_pool_plus_current")
        failed_selected_entry = next(entry for entry in snapshot["selected_entries"] if entry["sequence"] == 2)
        self.assertEqual(failed_selected_entry["verification_status"], "fail")
        self.assertEqual(failed_selected_entry["checkpoint_exists"], "fail")
        self.assertIn("checkpoint_exists", failed_selected_entry["failed_checks"])

    def test_eval_cli_promotions_selected_opponent_pool_verification_fails_for_broken_selected_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = write_registry_with_entries(temp_path, count=4)
            (temp_path / "checkpoint-2.json").unlink()

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "2",
                        "--require-opponent-pool-size",
                        "2",
                        "--verify",
                        "--verify-opponent-pool-only",
                        "--lifecycle",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())
            lifecycle = payload["lifecycle_summary"]

        selected_status = next(status for status in payload["entry_statuses"] if status["sequence"] == 2)
        self.assertEqual(exit_code, 2)
        self.assertFalse(payload["verification"]["passed"])
        self.assertFalse(payload["opponent_pool_verified"])
        self.assertFalse(payload["selected_opponent_pool_verified"])
        self.assertTrue(payload["opponent_pool_current_policy_verified"])
        self.assertTrue(payload["opponent_pool_registry_level_verified"])
        self.assertFalse(payload["opponent_pool_preflight_verified"])
        self.assertEqual(payload["opponent_pool_verification_exit_scope"], "opponent_pool_plus_current")
        self.assertEqual(selected_status["opponent_pool_status"], "selected")
        self.assertEqual(selected_status["checkpoint_exists"], "fail")
        self.assertIn("checkpoint_exists", selected_status["failed_checks"])
        self.assertTrue(lifecycle["verification_enabled"])
        self.assertEqual(lifecycle["failed_verification_count"], 1)
        self.assertEqual(lifecycle["selected_opponent_pool_unhealthy_count"], 1)
        self.assertEqual(lifecycle["registry_level_failed_verification_count"], 0)
        self.assertEqual(lifecycle["verification_status_counts"]["fail"], 1)

    def test_eval_cli_promotions_selected_pool_only_fails_empty_selected_pool(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = write_registry_with_entries(temp_path, count=1)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "3",
                        "--verify",
                        "--verify-opponent-pool-only",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 2)
        self.assertTrue(payload["verification"]["passed"])
        self.assertFalse(payload["selected_opponent_pool_verified"])
        self.assertTrue(payload["opponent_pool_current_policy_verified"])
        self.assertTrue(payload["opponent_pool_registry_level_verified"])
        self.assertFalse(payload["opponent_pool_preflight_verified"])
        self.assertEqual(payload["opponent_pool_selected_size"], 0)

    def test_eval_cli_promotions_selected_pool_only_still_gates_excluded_current_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = write_registry_with_entries(temp_path, count=4)
            (temp_path / "checkpoint-4.json").unlink()

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "2",
                        "--require-opponent-pool-size",
                        "2",
                        "--verify",
                        "--verify-opponent-pool-only",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        latest_status = next(status for status in payload["entry_statuses"] if status["sequence"] == 4)
        self.assertEqual(exit_code, 2)
        self.assertFalse(payload["verification"]["passed"])
        self.assertTrue(payload["selected_opponent_pool_verified"])
        self.assertFalse(payload["opponent_pool_current_policy_verified"])
        self.assertTrue(payload["opponent_pool_registry_level_verified"])
        self.assertFalse(payload["opponent_pool_preflight_verified"])
        self.assertEqual(latest_status["selected_as"], ["latest"])
        self.assertEqual(latest_status["opponent_pool_status"], "excluded_current_policy")
        self.assertEqual(latest_status["checkpoint_exists"], "fail")

    def test_eval_cli_promotions_selected_pool_only_still_gates_registry_level_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = write_registry_with_entries(temp_path, count=4)
            payload = json.loads(registry_path.read_text(encoding="utf-8"))
            payload["entries"][3]["sequence"] = 5
            write_manifest(registry_path, payload)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "2",
                        "--require-opponent-pool-size",
                        "2",
                        "--verify",
                        "--verify-opponent-pool-only",
                        "--lifecycle",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())
            lifecycle = payload["lifecycle_summary"]

        self.assertEqual(exit_code, 2)
        self.assertFalse(payload["verification"]["passed"])
        self.assertTrue(payload["selected_opponent_pool_verified"])
        self.assertTrue(payload["opponent_pool_current_policy_verified"])
        self.assertFalse(payload["opponent_pool_registry_level_verified"])
        self.assertFalse(payload["opponent_pool_preflight_verified"])
        self.assertIn("sequence_contiguous", failed_verification_check_names_from_payload(payload["verification"]))
        self.assertEqual(lifecycle["failed_verification_count"], 0)
        self.assertEqual(lifecycle["registry_level_failed_verification_count"], 1)

    def test_eval_cli_promotions_selected_pool_only_marks_external_current_policy_not_applicable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = write_registry_with_entries(temp_path, count=4)
            external_current_spec = f"linear:{temp_path / 'external-current.json'}"

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "2",
                        "--current-policy-spec",
                        external_current_spec,
                        "--verify",
                        "--verify-opponent-pool-only",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["verification"]["passed"])
        self.assertTrue(payload["selected_opponent_pool_verified"])
        self.assertIsNone(payload["opponent_pool_current_policy_verified"])
        self.assertTrue(payload["opponent_pool_registry_level_verified"])
        self.assertTrue(payload["opponent_pool_preflight_verified"])
        self.assertTrue(
            all(status["opponent_pool_status"] != "excluded_current_policy" for status in payload["entry_statuses"])
        )

    def test_eval_cli_promotions_verify_keeps_registry_scoped_opponent_pool_verified_without_new_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = write_registry_with_entries(temp_path, count=4)
            (temp_path / "checkpoint-1.json").unlink()

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "2",
                        "--verify",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 2)
        self.assertFalse(payload["verification"]["passed"])
        self.assertFalse(payload["opponent_pool_verified"])
        self.assertTrue(payload["selected_opponent_pool_verified"])
        self.assertTrue(payload["opponent_pool_registry_level_verified"])
        self.assertTrue(payload["opponent_pool_preflight_verified"])
        self.assertEqual(payload["opponent_pool_verification_exit_scope"], "registry")

    def test_eval_cli_promotions_verify_opponent_pool_only_requires_verify_and_pool_preview(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = write_registry_with_entries(Path(temp_dir), count=2)

            with patch("sys.stderr", new_callable=io.StringIO) as missing_verify:
                missing_verify_exit = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "1",
                        "--verify-opponent-pool-only",
                    ]
                )
            with patch("sys.stderr", new_callable=io.StringIO) as missing_pool:
                missing_pool_exit = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--verify",
                        "--verify-opponent-pool-only",
                    ]
                )

        self.assertEqual(missing_verify_exit, 1)
        self.assertIn("--verify-opponent-pool-only requires --verify", missing_verify.getvalue())
        self.assertEqual(missing_pool_exit, 1)
        self.assertIn("--verify-opponent-pool-only requires --opponent-pool-size", missing_pool.getvalue())

    def test_eval_cli_promotions_write_opponent_pool_requires_pool_preview(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = write_registry_with_entries(Path(temp_dir), count=2)
            snapshot_path = Path(temp_dir) / "pool-snapshot.json"

            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--write-opponent-pool",
                        str(snapshot_path),
                    ]
                )

        self.assertEqual(exit_code, 1)
        self.assertIn("--write-opponent-pool requires --opponent-pool-size", stderr.getvalue())
        self.assertFalse(snapshot_path.exists())

    def test_eval_cli_promotions_retention_plan_requires_pool_preview(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = write_registry_with_entries(Path(temp_dir), count=2)

            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--retention-plan",
                    ]
                )

        self.assertEqual(exit_code, 1)
        self.assertIn("--retention-plan requires --opponent-pool-size", stderr.getvalue())

    def test_eval_cli_promotions_json_reports_retention_plan_for_stale_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = write_registry_with_entries(Path(temp_dir), count=4)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "2",
                        "--retention-plan",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        plan = payload["retention_plan"]
        self.assertEqual(plan["schema_version"], PROMOTION_RETENTION_PLAN_SCHEMA_VERSION)
        self.assertFalse(plan["verification_enabled"])
        self.assertEqual(plan["selected_opponent_pool_size"], 2)
        self.assertEqual(plan["available_opponent_pool_size"], 3)
        self.assertEqual(plan["summary"]["verify_before_cleanup_count"], 1)
        self.assertEqual(plan["summary"]["cleanup_candidate_count"], 0)
        self.assertEqual(plan["summary"]["retain_count"], 3)
        entries_by_sequence = {entry["sequence"]: entry for entry in plan["entries"]}
        self.assertEqual(entries_by_sequence[1]["recommended_action"], "verify_before_cleanup")
        self.assertEqual(
            entries_by_sequence[1]["recommendation_reason"],
            "stale_outside_requested_pool_unverified",
        )
        self.assertEqual(entries_by_sequence[2]["recommended_action"], "retain")
        self.assertEqual(entries_by_sequence[3]["recommended_action"], "retain")
        self.assertEqual(entries_by_sequence[4]["recommended_action"], "retain")
        self.assertEqual(entries_by_sequence[4]["recommendation_reason"], "latest_promotion")

    def test_eval_cli_promotions_partial_verified_retention_plan_requires_more_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = write_registry_with_entries(Path(temp_dir), count=4)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "2",
                        "--retention-plan",
                        "--verify",
                        "--verify-opponent-pool-only",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        plan = payload["retention_plan"]
        self.assertTrue(plan["verification_enabled"])
        self.assertTrue(plan["registry_level_verification_passed"])
        entries_by_sequence = {entry["sequence"]: entry for entry in plan["entries"]}
        self.assertEqual(entries_by_sequence[1]["recommended_action"], "verify_before_cleanup")
        self.assertEqual(
            entries_by_sequence[1]["recommendation_reason"],
            "stale_outside_requested_pool_partially_verified",
        )
        self.assertEqual(entries_by_sequence[1]["verification_status"], "partial")
        self.assertEqual(plan["summary"]["cleanup_candidate_count"], 0)
        self.assertEqual(plan["summary"]["manual_review_count"], 0)

    def test_eval_cli_promotions_full_verified_retention_plan_marks_cleanup_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = write_managed_registry_with_entries(Path(temp_dir), count=4)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "2",
                        "--retention-plan",
                        "--verify",
                        "--verify-loadable",
                        "--verify-opponent-pool-only",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        plan = payload["retention_plan"]
        self.assertTrue(plan["verification_enabled"])
        self.assertTrue(plan["registry_level_verification_passed"])
        entries_by_sequence = {entry["sequence"]: entry for entry in plan["entries"]}
        self.assertEqual(entries_by_sequence[1]["recommended_action"], "cleanup_candidate")
        self.assertEqual(entries_by_sequence[1]["recommendation_reason"], "stale_outside_requested_pool")
        self.assertEqual(entries_by_sequence[1]["verification_status"], "pass")
        self.assertEqual(entries_by_sequence[1]["checksum"], "pass")
        self.assertEqual(entries_by_sequence[1]["loadable"], "pass")
        self.assertEqual(plan["summary"]["cleanup_candidate_count"], 1)
        self.assertEqual(plan["summary"]["manual_review_count"], 0)

    def test_eval_cli_promotions_retention_plan_marks_broken_stale_entry_for_manual_review(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = write_registry_with_entries(temp_path, count=4)
            (temp_path / "checkpoint-1.json").unlink()

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "2",
                        "--retention-plan",
                        "--verify",
                        "--verify-opponent-pool-only",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertFalse(payload["verification"]["passed"])
        self.assertTrue(payload["opponent_pool_preflight_verified"])
        entries_by_sequence = {entry["sequence"]: entry for entry in payload["retention_plan"]["entries"]}
        self.assertEqual(entries_by_sequence[1]["recommended_action"], "manual_review")
        self.assertEqual(entries_by_sequence[1]["recommendation_reason"], "verification_failed")
        self.assertIn("checkpoint_exists", entries_by_sequence[1]["failed_checks"])
        self.assertEqual(payload["retention_plan"]["summary"]["manual_review_count"], 1)

    def test_eval_cli_promotions_retention_plan_blocks_cleanup_candidate_on_registry_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = write_managed_registry_with_entries(temp_path, count=4)
            payload = json.loads(registry_path.read_text(encoding="utf-8"))
            payload["entries"][-1]["sequence"] = 5
            write_manifest(registry_path, payload)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "2",
                        "--retention-plan",
                        "--verify",
                        "--verify-loadable",
                        "--verify-opponent-pool-only",
                        "--json",
                    ]
                )
            result = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 2)
        self.assertFalse(result["verification"]["passed"])
        self.assertFalse(result["retention_plan"]["registry_level_verification_passed"])
        entries_by_sequence = {entry["sequence"]: entry for entry in result["retention_plan"]["entries"]}
        self.assertEqual(entries_by_sequence[1]["recommended_action"], "manual_review")
        self.assertEqual(entries_by_sequence[1]["recommendation_reason"], "registry_verification_failed")
        self.assertEqual(result["retention_plan"]["summary"]["cleanup_candidate_count"], 0)
        self.assertEqual(result["retention_plan"]["summary"]["manual_review_count"], 1)

    def test_eval_cli_promotions_apply_retention_plan_requires_verified_retention_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = write_managed_registry_with_entries(Path(temp_dir), count=4)

            with patch("sys.stderr", new_callable=io.StringIO) as missing_plan_stderr:
                missing_plan_exit = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "2",
                        "--apply-retention-plan",
                        "--verify",
                        "--verify-loadable",
                    ]
                )
            with patch("sys.stderr", new_callable=io.StringIO) as missing_verify_stderr:
                missing_verify_exit = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "2",
                        "--retention-plan",
                        "--apply-retention-plan",
                    ]
                )

        self.assertEqual(missing_plan_exit, 1)
        self.assertIn("--apply-retention-plan requires --retention-plan", missing_plan_stderr.getvalue())
        self.assertEqual(missing_verify_exit, 1)
        self.assertIn("--apply-retention-plan requires --verify --verify-loadable", missing_verify_stderr.getvalue())

    def test_eval_cli_promotions_apply_retention_plan_dry_run_does_not_move_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = write_managed_registry_with_entries(temp_path, count=4)
            registry_before = load_promotion_registry(registry_path)
            stale_artifact_path = Path(registry_before.entries[0].checkpoint_path or "")

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "2",
                        "--retention-plan",
                        "--apply-retention-plan",
                        "--verify",
                        "--verify-loadable",
                        "--verify-opponent-pool-only",
                        "--json",
                    ]
                )
            result = json.loads(stdout.getvalue())
            registry_after = load_promotion_registry(registry_path)
            stale_artifact_exists_after = stale_artifact_path.exists()

        self.assertEqual(exit_code, 0)
        apply_payload = result["retention_apply"]
        self.assertEqual(apply_payload["schema_version"], PROMOTION_RETENTION_APPLY_SCHEMA_VERSION)
        self.assertTrue(apply_payload["dry_run"])
        self.assertEqual(apply_payload["summary"]["planned_count"], 1)
        self.assertEqual(apply_payload["summary"]["applied_count"], 0)
        entries_by_sequence = {entry["sequence"]: entry for entry in apply_payload["entries"]}
        self.assertEqual(entries_by_sequence[1]["apply_action"], "archive")
        self.assertEqual(entries_by_sequence[1]["apply_status"], "planned")
        self.assertTrue(stale_artifact_exists_after)
        self.assertEqual(registry_after.entries[0].checkpoint_path, registry_before.entries[0].checkpoint_path)

    def test_eval_cli_promotions_apply_retention_plan_archives_managed_artifact_and_rewrites_registry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = write_managed_registry_with_entries(temp_path, count=4)
            archive_dir = temp_path / "archive"
            registry_before = load_promotion_registry(registry_path)
            stale_entry_before = registry_before.entries[0]
            stale_artifact_path = Path(stale_entry_before.checkpoint_path or "")

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "2",
                        "--retention-plan",
                        "--apply-retention-plan",
                        "--retention-apply-confirm",
                        "archive",
                        "--retention-archive-dir",
                        str(archive_dir),
                        "--verify",
                        "--verify-loadable",
                        "--verify-opponent-pool-only",
                        "--json",
                    ]
                )
            result = json.loads(stdout.getvalue())
            registry_after = load_promotion_registry(registry_path)
            verification = verify_promotion_registry(registry_path, verify_loadable=True)
            registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
            archive_path = Path(result["retention_apply"]["entries"][0]["archive_path"])
            stale_artifact_exists_after = stale_artifact_path.exists()
            archive_exists_after = archive_path.exists()

        self.assertEqual(exit_code, 0)
        apply_payload = result["retention_apply"]
        self.assertFalse(apply_payload["dry_run"])
        self.assertEqual(apply_payload["summary"]["applied_count"], 1)
        self.assertEqual(apply_payload["summary"]["planned_count"], 0)
        entries_by_sequence = {entry["sequence"]: entry for entry in apply_payload["entries"]}
        archived_entry = entries_by_sequence[1]
        self.assertEqual(archived_entry["apply_status"], "applied")
        self.assertFalse(stale_artifact_exists_after)
        self.assertTrue(archive_exists_after)
        self.assertEqual(registry_after.entries[0].checkpoint_path, str(archive_path))
        self.assertEqual(registry_after.entries[0].source_checkpoint_path, stale_entry_before.source_checkpoint_path)
        self.assertTrue(verification.passed)
        self.assertEqual(
            registry_payload["entries"][0]["retention_archived_from_checkpoint_path"],
            stale_entry_before.checkpoint_path,
        )
        self.assertIn("retention_archived_at", registry_payload["entries"][0])

    def test_eval_cli_promotions_apply_retention_plan_is_idempotent_for_archived_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = write_managed_registry_with_entries(temp_path, count=4)
            archive_dir = temp_path / "archive"
            first_args = [
                "promotions",
                "--registry",
                str(registry_path),
                "--opponent-pool-size",
                "2",
                "--retention-plan",
                "--apply-retention-plan",
                "--retention-apply-confirm",
                "archive",
                "--retention-archive-dir",
                str(archive_dir),
                "--verify",
                "--verify-loadable",
                "--verify-opponent-pool-only",
                "--json",
            ]

            with patch("sys.stdout", new_callable=io.StringIO) as first_stdout:
                first_exit = eval_cli_main(first_args)
            first_payload = json.loads(first_stdout.getvalue())
            first_archive_path = first_payload["retention_apply"]["entries"][0]["archive_path"]

            with patch("sys.stdout", new_callable=io.StringIO) as second_stdout:
                second_exit = eval_cli_main(first_args)
            second_payload = json.loads(second_stdout.getvalue())
            registry_after = load_promotion_registry(registry_path)

        self.assertEqual(first_exit, 0)
        self.assertEqual(second_exit, 0)
        second_entries = {entry["sequence"]: entry for entry in second_payload["retention_plan"]["entries"]}
        self.assertEqual(second_entries[1]["recommended_action"], "retain")
        self.assertEqual(second_entries[1]["recommendation_reason"], "already_archived")
        self.assertEqual(second_payload["retention_apply"]["summary"]["applied_count"], 0)
        self.assertEqual(registry_after.entries[0].checkpoint_path, first_archive_path)

    def test_eval_cli_promotions_archive_integrity_reports_verified_archived_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = write_managed_registry_with_entries(temp_path, count=4)
            archive_dir = temp_path / "archive"
            archive_args = [
                "promotions",
                "--registry",
                str(registry_path),
                "--opponent-pool-size",
                "2",
                "--retention-plan",
                "--apply-retention-plan",
                "--retention-apply-confirm",
                "archive",
                "--retention-archive-dir",
                str(archive_dir),
                "--verify",
                "--verify-loadable",
                "--verify-opponent-pool-only",
                "--json",
            ]

            with patch("sys.stdout", new_callable=io.StringIO) as archive_stdout:
                archive_exit = eval_cli_main(archive_args)
            archive_payload = json.loads(archive_stdout.getvalue())
            archive_path = archive_payload["retention_apply"]["entries"][0]["archive_path"]

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--archive-integrity",
                        "--verify",
                        "--verify-loadable",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(archive_exit, 0)
        self.assertEqual(exit_code, 0)
        integrity = payload["archive_integrity"]
        self.assertEqual(integrity["schema_version"], PROMOTION_ARCHIVE_INTEGRITY_SCHEMA_VERSION)
        self.assertTrue(integrity["verification_enabled"])
        self.assertTrue(integrity["verify_loadable_enabled"])
        self.assertTrue(integrity["archive_integrity_passed"])
        self.assertEqual(integrity["summary"]["archived_entry_count"], 1)
        self.assertEqual(integrity["summary"]["archived_verified_count"], 1)
        self.assertEqual(integrity["summary"]["archived_healthy_count"], 1)
        self.assertEqual(integrity["summary"]["archived_incomplete_count"], 0)
        self.assertEqual(integrity["summary"]["archived_failed_count"], 0)
        self.assertEqual(integrity["summary"]["checkpoint_exists_counts"], {"pass": 1})
        self.assertEqual(integrity["summary"]["checksum_counts"], {"pass": 1})
        self.assertEqual(integrity["summary"]["loadable_counts"], {"pass": 1})
        self.assertEqual(integrity["entries"][0]["sequence"], 1)
        self.assertEqual(integrity["entries"][0]["checkpoint_path"], archive_path)
        self.assertIsNotNone(integrity["entries"][0]["retention_archived_at"])

    def test_eval_cli_promotions_require_archive_integrity_fails_broken_archived_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = write_managed_registry_with_entries(temp_path, count=4)
            archive_dir = temp_path / "archive"
            archive_args = [
                "promotions",
                "--registry",
                str(registry_path),
                "--opponent-pool-size",
                "2",
                "--retention-plan",
                "--apply-retention-plan",
                "--retention-apply-confirm",
                "archive",
                "--retention-archive-dir",
                str(archive_dir),
                "--verify",
                "--verify-loadable",
                "--verify-opponent-pool-only",
                "--json",
            ]

            with patch("sys.stdout", new_callable=io.StringIO) as archive_stdout:
                archive_exit = eval_cli_main(archive_args)
            archive_payload = json.loads(archive_stdout.getvalue())
            archive_path = Path(archive_payload["retention_apply"]["entries"][0]["archive_path"])
            archive_path.unlink()

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "2",
                        "--verify",
                        "--verify-loadable",
                        "--verify-opponent-pool-only",
                        "--archive-integrity",
                        "--require-archive-integrity",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(archive_exit, 0)
        self.assertEqual(exit_code, 2)
        self.assertTrue(payload["opponent_pool_preflight_verified"])
        integrity = payload["archive_integrity"]
        self.assertFalse(integrity["archive_integrity_passed"])
        self.assertEqual(integrity["summary"]["archived_entry_count"], 1)
        self.assertEqual(integrity["summary"]["archived_failed_count"], 1)
        self.assertEqual(integrity["summary"]["checkpoint_exists_counts"], {"fail": 1})
        self.assertIn("checkpoint_exists", integrity["entries"][0]["failed_checks"])

    def test_eval_cli_promotions_archive_integrity_marks_partial_verification_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = write_managed_registry_with_entries(temp_path, count=4)
            archive_dir = temp_path / "archive"
            archive_args = [
                "promotions",
                "--registry",
                str(registry_path),
                "--opponent-pool-size",
                "2",
                "--retention-plan",
                "--apply-retention-plan",
                "--retention-apply-confirm",
                "archive",
                "--retention-archive-dir",
                str(archive_dir),
                "--verify",
                "--verify-loadable",
                "--verify-opponent-pool-only",
                "--json",
            ]

            with patch("sys.stdout", new_callable=io.StringIO):
                archive_exit = eval_cli_main(archive_args)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--archive-integrity",
                        "--verify",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(archive_exit, 0)
        self.assertEqual(exit_code, 0)
        integrity = payload["archive_integrity"]
        self.assertFalse(integrity["verify_loadable_enabled"])
        self.assertFalse(integrity["archive_integrity_passed"])
        self.assertEqual(integrity["summary"]["archived_verified_count"], 0)
        self.assertEqual(integrity["summary"]["archived_incomplete_count"], 1)
        self.assertEqual(integrity["summary"]["archived_failed_count"], 0)
        self.assertEqual(integrity["summary"]["loadable_counts"], {"not_checked": 1})

    def test_eval_cli_promotions_archive_integrity_text_prints_archived_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = write_managed_registry_with_entries(temp_path, count=4)
            archive_dir = temp_path / "archive"
            archive_args = [
                "promotions",
                "--registry",
                str(registry_path),
                "--opponent-pool-size",
                "2",
                "--retention-plan",
                "--apply-retention-plan",
                "--retention-apply-confirm",
                "archive",
                "--retention-archive-dir",
                str(archive_dir),
                "--verify",
                "--verify-loadable",
                "--verify-opponent-pool-only",
                "--json",
            ]

            with patch("sys.stdout", new_callable=io.StringIO):
                archive_exit = eval_cli_main(archive_args)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--archive-integrity",
                        "--verify",
                        "--verify-loadable",
                    ]
                )
            output = stdout.getvalue()

        self.assertEqual(archive_exit, 0)
        self.assertEqual(exit_code, 0)
        self.assertIn("archive_integrity:", output)
        self.assertIn("- archive_integrity: PASS", output)
        self.assertIn("- archived_entry_count: 1", output)
        self.assertIn("- archived_verified_count: 1", output)
        self.assertIn("archive_integrity_entries:", output)
        self.assertIn("verification=pass exists=pass checksum=pass loadable=pass", output)
        self.assertIn("archived_from=", output)

    def test_eval_cli_promotions_require_archive_integrity_requires_loadable_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = write_registry_with_entries(Path(temp_dir), count=1)

            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--archive-integrity",
                        "--require-archive-integrity",
                        "--verify",
                        "--json",
                    ]
                )

        self.assertEqual(exit_code, 1)
        self.assertIn("--require-archive-integrity requires --verify --verify-loadable", stderr.getvalue())

    def test_eval_cli_promotions_apply_retention_plan_skips_failed_verification_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = write_managed_registry_with_entries(temp_path, count=4)
            registry_before = load_promotion_registry(registry_path)
            stale_artifact_path = Path(registry_before.entries[0].checkpoint_path or "")
            stale_artifact_path.unlink()

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "2",
                        "--retention-plan",
                        "--apply-retention-plan",
                        "--retention-apply-confirm",
                        "archive",
                        "--retention-archive-dir",
                        str(temp_path / "archive"),
                        "--verify",
                        "--verify-loadable",
                        "--verify-opponent-pool-only",
                        "--json",
                    ]
                )
            result = json.loads(stdout.getvalue())
            archive_files = list((temp_path / "archive").glob("*"))

        self.assertEqual(exit_code, 0)
        entries_by_sequence = {entry["sequence"]: entry for entry in result["retention_apply"]["entries"]}
        self.assertEqual(entries_by_sequence[1]["apply_action"], "skip")
        self.assertEqual(entries_by_sequence[1]["apply_reason"], "not_cleanup_candidate")
        self.assertEqual(result["retention_apply"]["summary"]["applied_count"], 0)
        self.assertEqual(archive_files, [])

    def test_eval_cli_promotions_apply_retention_plan_skips_non_managed_cleanup_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = write_managed_registry_with_entries(temp_path, count=4)
            payload = json.loads(registry_path.read_text(encoding="utf-8"))
            payload["entries"][0]["source_checkpoint_path"] = payload["entries"][0]["checkpoint_path"]
            write_manifest(registry_path, payload)
            registry_before = load_promotion_registry(registry_path)
            stale_artifact_path = Path(registry_before.entries[0].checkpoint_path or "")

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "2",
                        "--retention-plan",
                        "--apply-retention-plan",
                        "--retention-apply-confirm",
                        "archive",
                        "--retention-archive-dir",
                        str(temp_path / "archive"),
                        "--verify",
                        "--verify-loadable",
                        "--verify-opponent-pool-only",
                        "--json",
                    ]
                )
            result = json.loads(stdout.getvalue())
            stale_artifact_exists_after = stale_artifact_path.exists()

        self.assertEqual(exit_code, 0)
        plan_entries = {entry["sequence"]: entry for entry in result["retention_plan"]["entries"]}
        self.assertEqual(plan_entries[1]["recommended_action"], "cleanup_candidate")
        apply_entries = {entry["sequence"]: entry for entry in result["retention_apply"]["entries"]}
        self.assertEqual(apply_entries[1]["apply_action"], "skip")
        self.assertEqual(apply_entries[1]["apply_reason"], "not_managed_artifact_copy")
        self.assertTrue(stale_artifact_exists_after)

    def test_eval_cli_promotions_apply_retention_plan_refuses_confirmed_archive_on_failing_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = write_managed_registry_with_entries(temp_path, count=4)
            payload = json.loads(registry_path.read_text(encoding="utf-8"))
            payload["entries"][-1]["sequence"] = 5
            write_manifest(registry_path, payload)
            original_checkpoint_path = Path(payload["entries"][0]["checkpoint_path"])

            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "2",
                        "--retention-plan",
                        "--apply-retention-plan",
                        "--retention-apply-confirm",
                        "archive",
                        "--retention-archive-dir",
                        str(temp_path / "archive"),
                        "--verify",
                        "--verify-loadable",
                        "--verify-opponent-pool-only",
                        "--json",
                    ]
                )
            archive_files = list((temp_path / "archive").glob("*"))
            original_checkpoint_exists_after = original_checkpoint_path.exists()

        self.assertEqual(exit_code, 1)
        self.assertIn("requires a passing promotions preflight", stderr.getvalue())
        self.assertTrue(original_checkpoint_exists_after)
        self.assertEqual(archive_files, [])

    def test_eval_cli_promotions_apply_retention_plan_rolls_back_moved_artifact_on_registry_rewrite_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = write_managed_registry_with_entries(temp_path, count=4)
            registry_before = load_promotion_registry(registry_path)
            stale_artifact_path = Path(registry_before.entries[0].checkpoint_path or "")
            archive_dir = temp_path / "archive"

            with patch(
                "pokezero.eval_cli._rewrite_retention_archived_checkpoint_paths",
                side_effect=OSError("simulated registry write failure"),
            ):
                with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                    exit_code = eval_cli_main(
                        [
                            "promotions",
                            "--registry",
                            str(registry_path),
                            "--opponent-pool-size",
                            "2",
                            "--retention-plan",
                            "--apply-retention-plan",
                            "--retention-apply-confirm",
                            "archive",
                            "--retention-archive-dir",
                            str(archive_dir),
                            "--verify",
                            "--verify-loadable",
                            "--verify-opponent-pool-only",
                            "--json",
                        ]
                    )
            archive_files = list(archive_dir.glob("*"))
            registry_after = load_promotion_registry(registry_path)
            stale_artifact_exists_after = stale_artifact_path.exists()

        self.assertEqual(exit_code, 1)
        self.assertIn("simulated registry write failure", stderr.getvalue())
        self.assertTrue(stale_artifact_exists_after)
        self.assertEqual(archive_files, [])
        self.assertEqual(registry_after.entries[0].checkpoint_path, registry_before.entries[0].checkpoint_path)

    def test_eval_cli_promotions_json_can_override_current_policy_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = write_registry_with_entries(Path(temp_dir), count=3)
            registry = load_promotion_registry(registry_path)
            middle_spec = registry.selection_checkpoint_policy_spec_for_entry(registry.entries[1])

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "2",
                        "--current-policy-spec",
                        middle_spec or "",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            payload["opponent_pool_policy_specs"],
            list(
                historical_opponent_policy_specs(
                    registry.selection_checkpoint_policy_specs(),
                    current_policy_spec=middle_spec,
                    max_historical_opponents=2,
                )
            ),
        )
        self.assertEqual(payload["opponent_pool_excluded_current_policy_spec"], middle_spec)
        self.assertEqual(payload["entry_statuses"][1]["opponent_pool_status"], "excluded_current_policy")
        self.assertEqual(payload["entry_statuses"][1]["opponent_pool_skip_reason"], "matches_current_policy")

    def test_eval_cli_promotions_json_marks_exact_duplicate_opponent_pool_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = write_registry_with_entries(temp_path, count=3)
            payload = json.loads(registry_path.read_text(encoding="utf-8"))
            payload["entries"][0]["checkpoint_path"] = payload["entries"][1]["checkpoint_path"]
            write_manifest(registry_path, payload)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "1",
                        "--json",
                    ]
                )
            result = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(result["entry_statuses"][0]["selected_as"], [])
        self.assertEqual(result["entry_statuses"][0]["opponent_pool_status"], "available_outside_requested_size")
        self.assertEqual(result["entry_statuses"][1]["selected_as"], ["opponent_pool"])
        self.assertEqual(result["entry_statuses"][1]["opponent_pool_status"], "selected")
        self.assertEqual(result["entry_statuses"][2]["selected_as"], ["latest"])
        self.assertEqual(result["entry_statuses"][2]["opponent_pool_status"], "excluded_current_policy")

    def test_eval_cli_promotions_text_prints_opponent_pool_preview(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = write_registry_with_entries(temp_path, count=2)
            snapshot_path = temp_path / "pool-snapshot.json"
            registry = load_promotion_registry(registry_path)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "1",
                        "--write-opponent-pool",
                        str(snapshot_path),
                    ]
                )
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn("opponent_pool_policy_specs:", output)
        self.assertIn("historical_opponent_selection: recent", output)
        self.assertIn("opponent_pool_selected_size: 1", output)
        self.assertIn("opponent_pool_available_size: 1", output)
        self.assertIn(f"opponent_pool_snapshot: {snapshot_path}", output)
        self.assertIn(registry.selection_checkpoint_policy_spec_for_entry(registry.entries[0]) or "", output)
        self.assertNotIn(f"- {registry.selection_checkpoint_policy_spec_for_entry(registry.entries[-1])}", output)
        self.assertIn("status=not_verified", output)
        self.assertIn("selected=opponent_pool", output)
        self.assertIn("selected=latest", output)
        self.assertIn("pool=selected", output)
        self.assertIn("pool=excluded_current_policy", output)
        self.assertIn("pass --verify", output)
        self.assertEqual(snapshot["historical_opponent_selection"], "recent")
        self.assertEqual(snapshot["policy_specs"], [registry.selection_checkpoint_policy_spec_for_entry(registry.entries[0])])

    def test_eval_cli_promotions_text_prints_retention_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = write_registry_with_entries(Path(temp_dir), count=3)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "1",
                        "--retention-plan",
                    ]
                )

        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn("retention_plan:", output)
        self.assertIn("- verify_before_cleanup_count: 1", output)
        self.assertIn("retention_entries:", output)
        self.assertIn("action=verify_before_cleanup", output)
        self.assertIn("reason=stale_outside_requested_pool_unverified", output)

    def test_eval_cli_promotions_text_omits_pool_status_without_pool_preview(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = write_registry_with_entries(Path(temp_dir), count=1)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(["promotions", "--registry", str(registry_path)])

        self.assertEqual(exit_code, 0)
        self.assertNotIn("pool=", stdout.getvalue())

    def test_eval_cli_promotions_text_reports_failed_required_pool_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = write_registry_with_entries(Path(temp_dir), count=2)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "2",
                        "--require-opponent-pool-size",
                        "2",
                    ]
                )

        self.assertEqual(exit_code, 2)
        output = stdout.getvalue()
        self.assertIn("opponent_pool_selected_size: 1", output)
        self.assertIn("opponent_pool_required_size: 2", output)
        self.assertIn("opponent_pool_requirement: FAIL", output)

    def test_eval_cli_promotions_verify_json_marks_partial_status_for_unchecked_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = write_registry_with_entries(Path(temp_dir), count=1)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--verify",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["entry_statuses"][0]["verification_status"], "partial")
        self.assertEqual(payload["entry_statuses"][0]["checkpoint_path_present"], "pass")
        self.assertEqual(payload["entry_statuses"][0]["checkpoint_exists"], "pass")
        self.assertEqual(payload["entry_statuses"][0]["checksum"], "not_checked")
        self.assertEqual(payload["entry_statuses"][0]["loadable"], "not_checked")

    def test_eval_cli_promotions_rejects_current_policy_without_pool_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = write_registry_with_entries(Path(temp_dir), count=1)

            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--current-policy-spec",
                        "linear:checkpoint.json",
                    ]
                )

        self.assertEqual(exit_code, 1)
        self.assertIn("--current-policy-spec requires --opponent-pool-size", stderr.getvalue())

    def test_eval_cli_promotions_rejects_required_pool_size_without_pool_preview(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = write_registry_with_entries(Path(temp_dir), count=1)

            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--require-opponent-pool-size",
                        "1",
                    ]
                )

        self.assertEqual(exit_code, 1)
        self.assertIn("--require-opponent-pool-size requires --opponent-pool-size", stderr.getvalue())

    def test_eval_cli_promote_can_copy_checkpoint_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = temp_path / "run" / "manifest.json"
            registry_path = temp_path / "promotions.json"
            artifact_dir = temp_path / "artifact-store"
            manifest = selfplay_manifest()
            write_manifest(manifest_path, manifest)
            write_checkpoint_for_manifest(temp_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promote",
                        str(manifest_path),
                        "--registry",
                        str(registry_path),
                        "--artifact-dir",
                        str(artifact_dir),
                        "--min-benchmark-win-rate",
                        "0.60",
                        "--min-benchmark-games",
                        "20",
                        "--max-collection-capped-rate",
                        "0.20",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())
            managed_checkpoint = Path(payload["entry"]["checkpoint_path"])
            managed_checkpoint_exists = managed_checkpoint.exists()

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["entry"]["source_checkpoint_path"], "run/iteration-0001/linear-policy.json")
        self.assertEqual(len(payload["entry"]["checkpoint_sha256"]), 64)
        self.assertEqual(managed_checkpoint.parent, artifact_dir)
        self.assertTrue(managed_checkpoint_exists)

    def test_verify_promotion_registry_passes_for_managed_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = temp_path / "run" / "manifest.json"
            registry_path = temp_path / "promotions.json"
            artifact_dir = temp_path / "artifact-store"
            manifest = selfplay_manifest()
            write_manifest(manifest_path, manifest)
            write_checkpoint_for_manifest(temp_path, manifest)
            record_promotion(
                manifest_path,
                registry_path=registry_path,
                artifact_dir=artifact_dir,
                config=passing_gate_config(),
            )

            result = verify_promotion_registry(registry_path)

        self.assertTrue(result.passed)
        self.assertEqual(result.entry_count, 1)
        self.assertEqual(result.checked_checkpoint_count, 1)

    def test_verify_promotion_registry_can_require_loadable_policy_specs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = temp_path / "run" / "manifest.json"
            registry_path = temp_path / "promotions.json"
            manifest = selfplay_manifest()
            write_manifest(manifest_path, manifest)
            write_valid_linear_checkpoint_for_manifest(temp_path, manifest)
            record_promotion(
                manifest_path,
                registry_path=registry_path,
                artifact_dir=temp_path / "artifact-store",
                config=passing_gate_config(),
            )

            result = verify_promotion_registry(registry_path, verify_loadable=True)

        self.assertTrue(result.passed)
        self.assertEqual(result.verified_loadable_count, 1)
        check_names = {check.name for check in result.checks}
        self.assertIn("checkpoint_policy_loadable", check_names)
        self.assertIn("checkpoint_policy_id", check_names)

    def test_verify_promotion_registry_fails_unloadable_policy_spec(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint_path = temp_path / "linear-policy.json"
            checkpoint_path.write_text("{}", encoding="utf-8")
            registry_path = temp_path / "promotions.json"
            write_manifest(registry_path, promotion_registry_payload(checkpoint_path=str(checkpoint_path)))

            result = verify_promotion_registry(registry_path, verify_loadable=True)

        self.assertFalse(result.passed)
        self.assertIn("checkpoint_policy_loadable", failed_verification_check_names(result))

    def test_verify_promotion_registry_fails_policy_id_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint_path = temp_path / "linear-policy.json"
            save_linear_model(
                checkpoint_path,
                LinearPolicyModel.initialized(feature_count=16, window_size=1, policy_id="different-policy"),
            )
            registry_path = temp_path / "promotions.json"
            write_manifest(registry_path, promotion_registry_payload(checkpoint_path=str(checkpoint_path)))

            result = verify_promotion_registry(registry_path, verify_loadable=True)

        self.assertFalse(result.passed)
        self.assertIn("checkpoint_policy_id", failed_verification_check_names(result))

    def test_verify_promotion_registry_fails_missing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = temp_path / "promotions.json"
            write_manifest(registry_path, promotion_registry_payload(checkpoint_path="missing-checkpoint.json"))

            result = verify_promotion_registry(registry_path)

        self.assertFalse(result.passed)
        self.assertIn("checkpoint_exists", failed_verification_check_names(result))

    def test_verify_promotion_registry_resolves_manifest_relative_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = temp_path / "run" / "manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text("{}", encoding="utf-8")
            manifest_relative_checkpoint = manifest_path.parent / "iteration-0001" / "linear-policy.json"
            manifest_relative_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            manifest_relative_checkpoint.write_text("{}", encoding="utf-8")
            registry_path = temp_path / "promotions.json"
            write_manifest(
                registry_path,
                promotion_registry_payload(
                    checkpoint_path="iteration-0001/linear-policy.json",
                    manifest_path=str(manifest_path),
                ),
            )

            result = verify_promotion_registry(registry_path)

        self.assertTrue(result.passed)
        self.assertNotIn("checkpoint_exists", failed_verification_check_names(result))

    def test_verify_promotion_registry_resolves_registry_relative_checkpoint_from_any_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with tempfile.TemporaryDirectory() as cwd_dir:
                temp_path = Path(temp_dir)
                registry_path = temp_path / "runs" / "promotions.json"
                checkpoint_path = registry_path.parent / "promoted" / "linear-policy.json"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_path.write_text("{}", encoding="utf-8")
                write_manifest(
                    registry_path,
                    promotion_registry_payload(
                        checkpoint_path="promoted/linear-policy.json",
                        manifest_path="selfplay/manifest.json",
                    ),
                )

                previous_cwd = Path.cwd()
                try:
                    os.chdir(cwd_dir)
                    result = verify_promotion_registry(registry_path)
                finally:
                    os.chdir(previous_cwd)

        self.assertTrue(result.passed)

    def test_registry_selection_specs_from_relative_registry_are_absolute(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = temp_path / "runs" / "promotions.json"
            checkpoint_path = registry_path.parent / "promoted" / "linear-policy.json"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text("{}", encoding="utf-8")
            write_manifest(
                registry_path,
                promotion_registry_payload(
                    checkpoint_path="promoted/linear-policy.json",
                    manifest_path="selfplay/manifest.json",
                ),
            )

            previous_cwd = Path.cwd()
            try:
                os.chdir(temp_path)
                registry = load_promotion_registry(Path("runs/promotions.json"))
                specs = registry.selection_checkpoint_policy_specs()
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(registry.path, registry_path.resolve(strict=False))
        self.assertEqual(specs, (f"linear:{checkpoint_path.resolve(strict=False)}",))

    def test_verify_promotion_registry_prefers_registry_relative_checkpoint_over_cwd_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with tempfile.TemporaryDirectory() as cwd_dir:
                temp_path = Path(temp_dir)
                registry_path = temp_path / "runs" / "promotions.json"
                good_checkpoint = registry_path.parent / "promoted" / "linear-policy.json"
                bad_checkpoint = Path(cwd_dir) / "promoted" / "linear-policy.json"
                good_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                bad_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                save_linear_model(
                    good_checkpoint,
                    LinearPolicyModel.initialized(
                        feature_count=16,
                        window_size=1,
                        policy_id="linear-selfplay-test-iter-0001",
                    ),
                )
                save_linear_model(
                    bad_checkpoint,
                    LinearPolicyModel.initialized(
                        feature_count=16,
                        window_size=1,
                        policy_id="wrong-cwd-policy",
                    ),
                )
                write_manifest(
                    registry_path,
                    promotion_registry_payload(
                        checkpoint_path="promoted/linear-policy.json",
                        manifest_path="selfplay/manifest.json",
                    ),
                )

                previous_cwd = Path.cwd()
                try:
                    os.chdir(cwd_dir)
                    result = verify_promotion_registry(registry_path, verify_loadable=True)
                finally:
                    os.chdir(previous_cwd)

        self.assertTrue(result.passed)

    def test_eval_cli_promotions_json_uses_resolved_selection_specs_for_pool_preview(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = temp_path / "runs" / "promotions.json"
            first_checkpoint = registry_path.parent / "promoted" / "first.json"
            second_checkpoint = registry_path.parent / "promoted" / "second.json"
            first_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            first_checkpoint.write_text("{}", encoding="utf-8")
            second_checkpoint.write_text("{}", encoding="utf-8")
            payload = promotion_registry_payload(
                checkpoint_path="promoted/first.json",
                manifest_path="selfplay-a/manifest.json",
            )
            second_entry = dict(payload["entries"][0])
            second_entry["sequence"] = 2
            second_entry["policy_id"] = "linear-selfplay-test-iter-0002"
            second_entry["checkpoint_path"] = "promoted/second.json"
            second_entry["manifest_path"] = "selfplay-b/manifest.json"
            payload["entries"].append(second_entry)
            write_manifest(registry_path, payload)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "1",
                        "--json",
                    ]
                )
            result = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(result["opponent_pool_policy_specs"], [f"linear:{first_checkpoint.resolve(strict=False)}"])
        self.assertEqual(
            result["opponent_pool_excluded_current_policy_spec"],
            f"linear:{second_checkpoint.resolve(strict=False)}",
        )
        self.assertEqual(result["entry_statuses"][0]["selected_as"], ["opponent_pool"])
        self.assertEqual(
            result["entry_statuses"][0]["selection_checkpoint_policy_spec"],
            f"linear:{first_checkpoint.resolve(strict=False)}",
        )
        self.assertEqual(result["entry_statuses"][1]["selected_as"], ["latest"])
        self.assertEqual(
            result["entry_statuses"][1]["selection_checkpoint_policy_spec"],
            f"linear:{second_checkpoint.resolve(strict=False)}",
        )

    def test_eval_cli_promotions_json_excludes_relative_current_policy_spec(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = temp_path / "runs" / "promotions.json"
            first_checkpoint = registry_path.parent / "promoted" / "first.json"
            second_checkpoint = registry_path.parent / "promoted" / "second.json"
            first_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            first_checkpoint.write_text("{}", encoding="utf-8")
            second_checkpoint.write_text("{}", encoding="utf-8")
            payload = promotion_registry_payload(
                checkpoint_path="promoted/first.json",
                manifest_path="selfplay-a/manifest.json",
            )
            second_entry = dict(payload["entries"][0])
            second_entry["sequence"] = 2
            second_entry["policy_id"] = "linear-selfplay-test-iter-0002"
            second_entry["checkpoint_path"] = "promoted/second.json"
            second_entry["manifest_path"] = "selfplay-b/manifest.json"
            payload["entries"].append(second_entry)
            write_manifest(registry_path, payload)
            current_policy_spec = f"linear:{os.path.relpath(second_checkpoint, Path.cwd())}"

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--opponent-pool-size",
                        "2",
                        "--current-policy-spec",
                        current_policy_spec,
                        "--json",
                    ]
                )
            result = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(result["opponent_pool_policy_specs"], [f"linear:{first_checkpoint.resolve(strict=False)}"])
        self.assertEqual(result["opponent_pool_excluded_current_policy_spec"], current_policy_spec)
        self.assertEqual(result["entry_statuses"][0]["opponent_pool_status"], "selected")
        self.assertEqual(result["entry_statuses"][1]["opponent_pool_status"], "excluded_current_policy")
        self.assertEqual(result["entry_statuses"][1]["opponent_pool_skip_reason"], "matches_current_policy")

    def test_eval_cli_promotions_verify_json_marks_missing_checkpoint_path_as_failed_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = temp_path / "promotions.json"
            payload = promotion_registry_payload(checkpoint_path="checkpoint.json")
            payload["entries"][0]["checkpoint_path"] = None
            write_manifest(registry_path, payload)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(["promotions", "--registry", str(registry_path), "--verify", "--json"])
            result = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 2)
        self.assertEqual(result["entry_statuses"][0]["verification_status"], "fail")
        self.assertEqual(result["entry_statuses"][0]["checkpoint_path_present"], "fail")
        self.assertEqual(result["entry_statuses"][0]["checkpoint_exists"], "fail")
        self.assertIn("checkpoint_path_present", result["entry_statuses"][0]["failed_checks"])

    def test_verify_promotion_registry_fails_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = temp_path / "run" / "manifest.json"
            registry_path = temp_path / "promotions.json"
            artifact_dir = temp_path / "artifact-store"
            manifest = selfplay_manifest()
            write_manifest(manifest_path, manifest)
            write_checkpoint_for_manifest(temp_path, manifest)
            record_promotion(
                manifest_path,
                registry_path=registry_path,
                artifact_dir=artifact_dir,
                config=passing_gate_config(),
            )
            loaded = load_promotion_registry(registry_path)
            managed_checkpoint = Path(loaded.latest.checkpoint_path if loaded.latest else "")
            managed_checkpoint.write_text(json.dumps({"policy_id": "tampered"}, indent=2), encoding="utf-8")

            result = verify_promotion_registry(registry_path)

        self.assertFalse(result.passed)
        self.assertIn("checkpoint_sha256", failed_verification_check_names(result))

    def test_verify_promotion_registry_can_require_checksum_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint_path = temp_path / "linear-policy.json"
            checkpoint_path.write_text("{}", encoding="utf-8")
            registry_path = temp_path / "promotions.json"
            write_manifest(registry_path, promotion_registry_payload(checkpoint_path=str(checkpoint_path)))

            result = verify_promotion_registry(registry_path, require_checksums=True)

        self.assertFalse(result.passed)
        self.assertIn("checkpoint_sha256_present", failed_verification_check_names(result))

    def test_verify_promotion_registry_fails_non_contiguous_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            first_checkpoint = temp_path / "first.json"
            second_checkpoint = temp_path / "second.json"
            first_checkpoint.write_text("{}", encoding="utf-8")
            second_checkpoint.write_text("{}", encoding="utf-8")
            registry_path = temp_path / "promotions.json"
            payload = promotion_registry_payload(checkpoint_path=str(first_checkpoint))
            second_entry = dict(payload["entries"][0])
            second_entry["sequence"] = 3
            second_entry["checkpoint_path"] = str(second_checkpoint)
            payload["entries"].append(second_entry)
            write_manifest(registry_path, payload)

            result = verify_promotion_registry(registry_path)

        self.assertFalse(result.passed)
        self.assertIn("sequence_contiguous", failed_verification_check_names(result))

    def test_eval_cli_promotions_verify_json_returns_nonzero_for_broken_registry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = temp_path / "promotions.json"
            write_manifest(registry_path, promotion_registry_payload(checkpoint_path="missing-checkpoint.json"))

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(["promotions", "--registry", str(registry_path), "--verify", "--json"])
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 2)
        self.assertFalse(payload["verification"]["passed"])
        self.assertIn("checkpoint_exists", failed_verification_check_names_from_payload(payload["verification"]))
        self.assertEqual(payload["entry_statuses"][0]["verification_status"], "fail")
        self.assertEqual(payload["entry_statuses"][0]["checkpoint_exists"], "fail")
        self.assertIn("checkpoint_exists", payload["entry_statuses"][0]["failed_checks"])

    def test_eval_cli_promotions_verify_can_require_checksum_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint_path = temp_path / "linear-policy.json"
            checkpoint_path.write_text("{}", encoding="utf-8")
            registry_path = temp_path / "promotions.json"
            write_manifest(registry_path, promotion_registry_payload(checkpoint_path=str(checkpoint_path)))

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--verify",
                        "--require-checksum",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 2)
        self.assertIn("checkpoint_sha256_present", failed_verification_check_names_from_payload(payload["verification"]))

    def test_eval_cli_promotions_verify_can_require_loadable_policy_specs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = temp_path / "run" / "manifest.json"
            registry_path = temp_path / "promotions.json"
            manifest = selfplay_manifest()
            write_manifest(manifest_path, manifest)
            write_valid_linear_checkpoint_for_manifest(temp_path, manifest)
            record_promotion(
                manifest_path,
                registry_path=registry_path,
                artifact_dir=temp_path / "artifact-store",
                config=passing_gate_config(),
            )

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--verify",
                        "--verify-loadable",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["verification"]["verified_loadable_count"], 1)
        self.assertEqual(payload["entry_statuses"][0]["verification_status"], "pass")
        self.assertEqual(payload["entry_statuses"][0]["checkpoint_exists"], "pass")
        self.assertEqual(payload["entry_statuses"][0]["loadable"], "pass")
        self.assertEqual(payload["entry_statuses"][0]["policy_id_matches"], "pass")
        check_names = {check["name"] for check in payload["verification"]["checks"]}
        self.assertIn("checkpoint_policy_loadable", check_names)

    def test_eval_cli_promotions_verify_loadable_requires_verify(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = Path(temp_dir) / "promotions.json"

            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                exit_code = eval_cli_main(
                    [
                        "promotions",
                        "--registry",
                        str(registry_path),
                        "--verify-loadable",
                    ]
                )

        self.assertEqual(exit_code, 1)
        self.assertIn("--verify-loadable requires --verify", stderr.getvalue())


def selfplay_manifest() -> dict:
    return {
        "schema_version": SELFPLAY_RUN_SCHEMA_VERSION,
        "run_dir": "run",
        "latest_checkpoint_path": "run/iteration-0001/linear-policy.json",
        "iterations": [
            {
                "schema_version": SELFPLAY_RUN_SCHEMA_VERSION,
                "iteration": 1,
                "checkpoint_path": "run/iteration-0001/linear-policy.json",
                "collection_metrics": collection_metrics(games=10, capped_games=1),
                "training": {"model": {"policy_id": "linear-selfplay-test-iter-0001"}},
                "benchmark": benchmark_payload(
                    policy_id="linear-selfplay-test-iter-0001",
                    wins=13,
                    losses=7,
                    capped_games=1,
                ),
            }
        ],
    }


def set_manifest_identity(manifest: dict, *, policy_id: str, checkpoint_path: str) -> None:
    manifest["latest_checkpoint_path"] = checkpoint_path
    iteration = manifest["iterations"][0]
    iteration["checkpoint_path"] = checkpoint_path
    iteration["training"]["model"]["policy_id"] = policy_id
    iteration["benchmark"] = benchmark_payload(
        policy_id=policy_id,
        wins=13,
        losses=7,
        capped_games=1,
    )


def set_manifest_checkpoint(manifest: dict, checkpoint_path: str) -> None:
    manifest["latest_checkpoint_path"] = checkpoint_path
    manifest["iterations"][0]["checkpoint_path"] = checkpoint_path


def add_benchmark_head_to_head(
    manifest: dict,
    *,
    first_policy_id: str,
    second_policy_id: str,
    first_policy_wins: int,
    second_policy_wins: int,
    capped_games: int,
) -> None:
    games = first_policy_wins + second_policy_wins
    manifest["iterations"][0]["benchmark"]["head_to_heads"].append(
        {
            "label": f"{first_policy_id} vs {second_policy_id}",
            "first_policy_id": first_policy_id,
            "second_policy_id": second_policy_id,
            "games": games,
            "first_policy_wins": first_policy_wins,
            "second_policy_wins": second_policy_wins,
            "ties": 0,
            "capped_games": capped_games,
            "first_policy_win_rate": first_policy_wins / games,
            "second_policy_win_rate": second_policy_wins / games,
        }
    )


def write_registry_with_entries(temp_path: Path, *, count: int) -> Path:
    registry_path = temp_path / "promotions.json"
    entries = []
    for sequence in range(1, count + 1):
        checkpoint_path = (temp_path / f"checkpoint-{sequence}.json").resolve(strict=False)
        checkpoint_path.write_text("{}", encoding="utf-8")
        entries.append(
            {
                "sequence": sequence,
                "policy_id": f"linear-selfplay-test-iter-{sequence:04d}",
                "checkpoint_path": str(checkpoint_path),
                "manifest_path": f"runs/selfplay-{sequence}/manifest.json",
                "source_type": SELFPLAY_RUN_SCHEMA_VERSION,
                "source_iteration": sequence,
                "promoted_at": "2026-06-02T00:00:00Z",
                "label": None,
                "notes": None,
                "gate_result": {"passed": True},
            }
        )
    write_manifest(
        registry_path,
        {
            "schema_version": PROMOTION_REGISTRY_SCHEMA_VERSION,
            "registry_path": str(registry_path),
            "latest_policy_id": entries[-1]["policy_id"] if entries else None,
            "latest_checkpoint_path": entries[-1]["checkpoint_path"] if entries else None,
            "entries": entries,
        },
    )
    return registry_path


def write_managed_registry_with_entries(temp_path: Path, *, count: int) -> Path:
    registry_path = temp_path / "promotions.json"
    artifact_dir = temp_path / "artifact-store"
    for sequence in range(1, count + 1):
        manifest = selfplay_manifest()
        policy_id = f"linear-selfplay-test-iter-{sequence:04d}"
        checkpoint_path = f"run-{sequence}/iteration-0001/linear-policy.json"
        set_manifest_identity(manifest, policy_id=policy_id, checkpoint_path=checkpoint_path)
        manifest_path = temp_path / f"run-{sequence}" / "manifest.json"
        write_manifest(manifest_path, manifest)
        write_valid_linear_checkpoint_for_manifest(temp_path, manifest, policy_id=policy_id)
        record_promotion(
            manifest_path,
            registry_path=registry_path,
            artifact_dir=artifact_dir,
            config=passing_gate_config(),
            promoted_at="2026-06-02T00:00:00Z",
        )
    return registry_path


def write_checkpoint_for_manifest(temp_path: Path, manifest: dict, *, policy_id: str | None = None) -> Path:
    checkpoint_path = temp_path / str(manifest["latest_checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_policy_id = policy_id or manifest["iterations"][0]["training"]["model"]["policy_id"]
    checkpoint_path.write_text(json.dumps({"policy_id": checkpoint_policy_id}, indent=2), encoding="utf-8")
    return checkpoint_path


def write_valid_linear_checkpoint_for_manifest(temp_path: Path, manifest: dict, *, policy_id: str | None = None) -> Path:
    checkpoint_path = temp_path / str(manifest["latest_checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_policy_id = policy_id or manifest["iterations"][0]["training"]["model"]["policy_id"]
    save_linear_model(
        checkpoint_path,
        LinearPolicyModel.initialized(feature_count=16, window_size=1, policy_id=checkpoint_policy_id),
    )
    return checkpoint_path


def benchmark_payload(*, policy_id: str, wins: int, losses: int, capped_games: int) -> dict:
    games = wins + losses
    return {
        "format_id": "gen3randombattle",
        "max_decision_rounds": 250,
        "games_per_matchup": games,
        "head_to_heads": [
            {
                "label": f"{policy_id} vs random-legal",
                "first_policy_id": policy_id,
                "second_policy_id": "random-legal",
                "games": games,
                "first_policy_wins": wins,
                "second_policy_wins": losses,
                "ties": 0,
                "capped_games": capped_games,
                "first_policy_win_rate": wins / games,
                "second_policy_win_rate": losses / games,
            }
        ],
        "matchups": [],
    }


def collection_metrics(*, games: int, capped_games: int) -> dict:
    return {
        "games": games,
        "elapsed_seconds": 1.0,
        "total_decision_rounds": games,
        "total_simulator_turns": games,
        "p1_wins": games - capped_games,
        "p2_wins": 0,
        "ties": 0,
        "capped_games": capped_games,
    }


def passing_gate_config() -> PromotionGateConfig:
    return PromotionGateConfig(
        min_benchmark_win_rate=0.60,
        min_benchmark_games=20,
        max_collection_capped_rate=0.20,
    )


def start_promotion_subprocess(manifest_path: Path, registry_path: Path, artifact_dir: Path) -> subprocess.Popen:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(src_path) if not existing_pythonpath else f"{src_path}{os.pathsep}{existing_pythonpath}"
    )
    code = (
        "from pathlib import Path\n"
        "import sys\n"
        "from pokezero.evaluation import PromotionGateConfig\n"
        "from pokezero.promotion import record_promotion\n"
        "record_promotion(\n"
        "    Path(sys.argv[1]),\n"
        "    registry_path=Path(sys.argv[2]),\n"
        "    artifact_dir=Path(sys.argv[3]),\n"
        "    config=PromotionGateConfig(\n"
        "        min_benchmark_win_rate=0.60,\n"
        "        min_benchmark_games=20,\n"
        "        max_collection_capped_rate=0.20,\n"
        "    ),\n"
        ")\n"
    )
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            code,
            str(manifest_path),
            str(registry_path),
            str(artifact_dir),
        ],
        cwd=repo_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def promotion_registry_payload(*, checkpoint_path: str, manifest_path: str = "run/manifest.json") -> dict:
    return {
        "schema_version": PROMOTION_REGISTRY_SCHEMA_VERSION,
        "registry_path": "promotions.json",
        "latest_policy_id": "linear-selfplay-test-iter-0001",
        "latest_checkpoint_path": checkpoint_path,
        "entries": [
            {
                "sequence": 1,
                "policy_id": "linear-selfplay-test-iter-0001",
                "checkpoint_path": checkpoint_path,
                "manifest_path": manifest_path,
                "source_type": SELFPLAY_RUN_SCHEMA_VERSION,
                "source_iteration": 1,
                "promoted_at": "2026-06-02T00:00:00Z",
                "label": None,
                "notes": None,
                "gate_result": {"passed": True},
            }
        ],
    }


def failed_verification_check_names(result) -> set[str]:
    return {check.name for check in result.checks if not check.passed}


def failed_verification_check_names_from_payload(payload: dict) -> set[str]:
    return {check["name"] for check in payload["checks"] if not check["passed"]}


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
