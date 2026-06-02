import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pokezero.eval_cli import main as eval_cli_main
from pokezero.evaluation import PromotionGateConfig
from pokezero.promotion import (
    PROMOTION_REGISTRY_SCHEMA_VERSION,
    load_promotion_registry,
    record_promotion,
)
from pokezero.selfplay import SELFPLAY_RUN_SCHEMA_VERSION


class PromotionRegistryTest(unittest.TestCase):
    def test_load_missing_registry_returns_empty_registry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = load_promotion_registry(Path(temp_dir) / "promotions.json")

        self.assertEqual(registry.entries, ())
        self.assertIsNone(registry.latest)

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
        self.assertEqual(managed_checkpoint_payload["policy_id"], "linear-selfplay-test-iter-0001")
        self.assertEqual(managed_checkpoint_path.parent, artifact_dir)
        self.assertEqual(result.entry.source_checkpoint_path if result.entry else None, "run/iteration-0001/linear-policy.json")
        self.assertEqual(result.entry.checkpoint_path if result.entry else None, str(managed_checkpoint_path))
        self.assertEqual(loaded.latest.checkpoint_path if loaded.latest else None, str(managed_checkpoint_path))
        self.assertEqual(loaded.checkpoint_policy_specs(), (f"linear:{managed_checkpoint_path}",))
        self.assertEqual(source_checkpoint_text, managed_checkpoint_text)

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
        self.assertEqual(managed_checkpoint.parent, artifact_dir)
        self.assertTrue(managed_checkpoint_exists)


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


def write_checkpoint_for_manifest(temp_path: Path, manifest: dict) -> Path:
    checkpoint_path = temp_path / str(manifest["latest_checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    policy_id = manifest["iterations"][0]["training"]["model"]["policy_id"]
    checkpoint_path.write_text(json.dumps({"policy_id": policy_id}, indent=2), encoding="utf-8")
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


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
