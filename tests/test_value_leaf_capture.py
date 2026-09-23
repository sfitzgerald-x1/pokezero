from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import AsyncMock, patch

from pokezero.value_leaf_capture import (
    _source_binding_from_manifest,
    async_main,
    build_value_leaf_capture_arg_parser,
)


def _binding(manifest_sha256: str) -> dict[str, object]:
    return {
        "checkpoint_sha256": "a" * 64,
        "checkpoint_iteration": 9375,
        "source_commit": "b" * 40,
        "source_tree_sha256": "c" * 64,
        "source_image_digest": "d" * 64,
        "observation_schema_version": "v3",
        "collection_manifest_sha256": manifest_sha256,
    }


class ValueLeafCaptureCliTest(unittest.TestCase):
    def test_exposes_source_binding_and_one_game_oracle_controls(self) -> None:
        parser = build_value_leaf_capture_arg_parser()
        args = parser.parse_args(
            [
                "--checkpoint", "/tmp/champion.pt",
                "--out", "/tmp/value-leaves",
                "--source-binding-manifest", "/tmp/manifest.json",
                "--source-binding-receipt-sha256", "d" * 64,
                "--games", "1",
                "--live-continuation-oracle",
            ]
        )
        self.assertEqual(args.policy_mode, "raw")
        self.assertEqual(args.games, 1)
        self.assertTrue(args.live_continuation_oracle)
        self.assertEqual(args.source_binding_manifest, Path("/tmp/manifest.json"))

    def test_reads_only_binding_sealed_by_its_manifest_digest(self) -> None:
        with TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            document = {"source_binding": _binding("0" * 64)}
            raw = json.dumps(document, sort_keys=True).encode("utf-8")
            digest = hashlib.sha256(raw).hexdigest()
            manifest_path.write_bytes(raw)
            self.assertEqual(
                _source_binding_from_manifest(
                    manifest_path, expected_receipt_sha256=digest
                ),
                document["source_binding"],
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                _source_binding_from_manifest(
                    manifest_path, expected_receipt_sha256="f" * 64
                )

    def test_rejects_summary_path_that_would_overwrite_completed_cache(self) -> None:
        with self.assertRaisesRegex(SystemExit, "2"):
            asyncio.run(
                async_main(
                    [
                        "--checkpoint", "/tmp/champion.pt",
                        "--showdown-root", "/tmp/showdown",
                        "--out", "/tmp/value-leaves",
                        "--summary-out", "/tmp/value-leaves",
                        "--source-binding-manifest", "/tmp/manifest.json",
                        "--source-binding-receipt-sha256", "d" * 64,
                        "--games", "1",
                        "--live-continuation-oracle",
                    ]
                )
            )

    def test_summary_write_failure_does_not_turn_completed_cache_into_failed_command(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            document = {"source_binding": _binding("0" * 64)}
            raw = json.dumps(document, sort_keys=True).encode("utf-8")
            manifest = root / "manifest.json"
            manifest.write_bytes(raw)
            receipt_sha = hashlib.sha256(raw).hexdigest()
            result = type(
                "Result",
                (),
                {
                    "to_dict": lambda self: {"cache": {"capture_count": 1}},
                    "cache": type("Cache", (), {"capture_count": 1})(),
                },
            )()
            with (
                patch(
                    "pokezero.value_leaf_capture.capture_live_foulplay_value_leaf_training_cache",
                    new=AsyncMock(return_value=result),
                ),
                patch(
                    "pokezero.value_leaf_capture._write_json",
                    side_effect=OSError("sidecar unavailable"),
                ),
            ):
                self.assertEqual(
                    asyncio.run(
                        async_main(
                            [
                                "--checkpoint", str(root / "checkpoint.pt"),
                                "--showdown-root", str(root / "showdown"),
                                "--out", str(root / "value-leaves"),
                                "--summary-out", str(root / "summary.json"),
                                "--source-binding-manifest", str(manifest),
                                "--source-binding-receipt-sha256", receipt_sha,
                                "--games", "1",
                                "--live-continuation-oracle",
                            ]
                        )
                    ),
                    0,
                )


if __name__ == "__main__":
    unittest.main()
