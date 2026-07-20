from __future__ import annotations

import asyncio
import unittest
from pathlib import Path

from pokezero.foulplay_collision_capture import async_main, build_collision_capture_arg_parser


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
