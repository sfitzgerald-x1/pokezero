from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from pokezero.foulplay_collision_capture import async_main, build_collision_capture_arg_parser
from pokezero.foulplay_bridge import ControlledFoulPlayConfig, run_controlled_foulplay_benchmark
from pokezero.observation import OBSERVATION_SCHEMA_VERSION_V3
from pokezero.policy import RandomLegalPolicy


class FoulPlayCollisionCaptureParserTest(unittest.TestCase):
    def test_parser_requires_compact_output_and_defaults_to_raw_policy(self) -> None:
        args = build_collision_capture_arg_parser().parse_args(
            ["--checkpoint", "checkpoint.pt", "--out", "collision-sketch.jsonl"]
        )

        self.assertEqual(args.checkpoint, Path("checkpoint.pt"))
        self.assertEqual(args.out, Path("collision-sketch.jsonl"))
        self.assertEqual(args.policy_mode, "raw")

    def test_parser_does_not_expose_search_policy_mode(self) -> None:
        with self.assertRaises(SystemExit):
            build_collision_capture_arg_parser().parse_args(
                ["--checkpoint", "checkpoint.pt", "--out", "collision-sketch.jsonl", "--policy-mode", "root-puct"]
            )

    def test_parser_accepts_explicit_v3_random_legal_audit_driver(self) -> None:
        args = build_collision_capture_arg_parser().parse_args(
            [
                "--capture-driver",
                "random-legal",
                "--observation-schema",
                "v3",
                "--out",
                "collision-sketch.jsonl",
            ]
        )

        self.assertEqual(args.capture_driver, "random-legal")
        self.assertIsNone(args.checkpoint)
        self.assertEqual(args.observation_schema, "v3")

    def test_random_legal_driver_uses_v3_without_checkpoint_weights(self) -> None:
        config = ControlledFoulPlayConfig(
            checkpoint=None,
            showdown_root=Path("/showdown"),
            policy_mode="raw",
            capture_driver="random-legal",
            audit_observation_schema="v3",
        )
        expected = object()
        with (
            patch("pokezero.foulplay_bridge._validate_external_paths"),
            patch("pokezero.foulplay_bridge.gen3_category_vocabulary", return_value=object()),
            patch("pokezero.foulplay_bridge.load_showdown_dex_cached", return_value=object()),
            patch(
                "pokezero.foulplay_bridge._run_controlled_foulplay_games",
                new_callable=AsyncMock,
                return_value=expected,
            ) as runner,
        ):
            actual = asyncio.run(run_controlled_foulplay_benchmark(config))

        self.assertIs(actual, expected)
        kwargs = runner.await_args.kwargs
        self.assertIsInstance(kwargs["policy"], RandomLegalPolicy)
        self.assertEqual(kwargs["policy_id"], "audit-random-legal")
        self.assertEqual(kwargs["observation_spec"].schema_version, OBSERVATION_SCHEMA_VERSION_V3)
        self.assertIsNone(kwargs["checkpoint_sha256"])

    def test_random_legal_driver_rejects_checkpoint_and_non_v3_schema(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not provide a checkpoint"):
            ControlledFoulPlayConfig(
                checkpoint=Path("checkpoint.pt"),
                showdown_root=Path("/showdown"),
                policy_mode="raw",
                capture_driver="random-legal",
                audit_observation_schema="v3",
            )
        with self.assertRaisesRegex(ValueError, "requires audit_observation_schema='v3'"):
            ControlledFoulPlayConfig(
                checkpoint=None,
                showdown_root=Path("/showdown"),
                policy_mode="raw",
                capture_driver="random-legal",
                audit_observation_schema=None,
            )

    def test_summary_path_cannot_replace_compact_output(self) -> None:
        with self.assertRaises(SystemExit):
            asyncio.run(
                async_main(
                    [
                        "--checkpoint",
                        "checkpoint.pt",
                        "--out",
                        "collision-sketch.jsonl",
                        "--summary-out",
                        "collision-sketch.jsonl",
                    ]
                )
            )


if __name__ == "__main__":
    unittest.main()
