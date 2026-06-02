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

    def test_record_promotion_rejects_duplicate_policy_by_default(self) -> None:
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
