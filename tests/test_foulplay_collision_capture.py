from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pokezero.deep_line_audit import PROTOCOL_SIGNATURE_SCHEMA_VERSION
from pokezero.foulplay_collision_capture import (
    _protocol_census_provenance,
    _resume_protocol_signature_census,
    async_main,
    build_collision_capture_arg_parser,
)
from pokezero.foulplay_bridge import (
    ControlledFoulPlayBenchmarkResult,
    ControlledFoulPlayCollisionSketchResult,
    ControlledFoulPlayConfig,
    capture_controlled_foulplay_collision_sketch,
    run_controlled_foulplay_benchmark,
)
from pokezero.env import TerminalState
from pokezero.observation import OBSERVATION_SCHEMA_VERSION_V3
from pokezero.policy import RandomLegalPolicy


class FoulPlayCollisionCaptureParserTest(unittest.TestCase):
    def test_zero_game_capture_scope_has_no_invalid_seed_range(self) -> None:
        provenance = _protocol_census_provenance(
            source_hash="source-hash",
            command_arguments=(),
            seed_start=17,
            games=0,
            capture_driver="random-legal",
            max_decision_rounds=61,
        )

        self.assertIsNone(provenance["execution_scope"]["seed_range"])

    def test_collision_summary_exposes_count_only_protocol_census(self) -> None:
        config = ControlledFoulPlayConfig(
            checkpoint=None,
            showdown_root=Path("/showdown"),
            policy_mode="raw",
            capture_driver="random-legal",
            audit_observation_schema="v3",
        )
        result = ControlledFoulPlayCollisionSketchResult(
            benchmark=ControlledFoulPlayBenchmarkResult(config=config, policy_id="audit-random-legal", games=()),
            output_path=Path("collision-sketch.jsonl"),
            pool_id="fixture",
            checkpoint_sha256=None,
            belief_set_source_hash="source-hash",
            observation_schema_version=OBSERVATION_SCHEMA_VERSION_V3,
            captured_games=1,
            skipped_capped_games=0,
            skipped_tied_games=0,
            captured_decisions=2,
            resumed_decisions=0,
            captured_new_decisions=2,
            recovered_trailing_partial=False,
            protocol_signature_schema_version=PROTOCOL_SIGNATURE_SCHEMA_VERSION,
            protocol_signatures={"move:protect": 1},
            protocol_signature_game_ids=("a" * 64,),
        ).to_dict()

        self.assertEqual(result["protocol_signature_schema_version"], PROTOCOL_SIGNATURE_SCHEMA_VERSION)
        self.assertEqual(result["protocol_signatures"], {"move:protect": 1})
        self.assertEqual(result["protocol_signature_game_ids"], ["a" * 64])
        self.assertNotIn("protocol_lines", result)

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

    def test_summary_stamps_protocol_census_provenance(self) -> None:
        config = ControlledFoulPlayConfig(
            checkpoint=None,
            showdown_root=Path("/showdown"),
            policy_mode="raw",
            capture_driver="random-legal",
            audit_observation_schema="v3",
        )
        collision_result = ControlledFoulPlayCollisionSketchResult(
            benchmark=ControlledFoulPlayBenchmarkResult(config=config, policy_id="audit-random-legal", games=()),
            output_path=Path("collision-sketch.jsonl"),
            pool_id="fixture",
            checkpoint_sha256=None,
            belief_set_source_hash="source-hash",
            observation_schema_version=OBSERVATION_SCHEMA_VERSION_V3,
            captured_games=1,
            skipped_capped_games=0,
            skipped_tied_games=0,
            captured_decisions=2,
            resumed_decisions=0,
            captured_new_decisions=2,
            recovered_trailing_partial=False,
            protocol_signature_schema_version=PROTOCOL_SIGNATURE_SCHEMA_VERSION,
            protocol_signatures={"move:protect": 1},
            protocol_signature_game_ids=("a" * 64,),
        )
        writes: list[dict] = []
        with (
            patch(
                "pokezero.foulplay_collision_capture.capture_controlled_foulplay_collision_sketch",
                new_callable=AsyncMock,
                return_value=collision_result,
            ),
            patch(
                "pokezero.foulplay_collision_capture.load_gen3_randbat_source_cached",
                return_value=SimpleNamespace(metadata=SimpleNamespace(source_hash="source-hash")),
            ),
            patch("pokezero.foulplay_collision_capture.public_repo_commit", return_value="a" * 40),
            patch("pokezero.foulplay_collision_capture._write_json", side_effect=lambda _path, payload: writes.append(payload)),
        ):
            exit_code = asyncio.run(
                async_main(
                    [
                        "--capture-driver",
                        "random-legal",
                        "--observation-schema",
                        "v3",
                        "--showdown-root",
                        "/showdown",
                        "--games",
                        "3",
                        "--seed-start",
                        "17",
                        "--max-decision-rounds",
                        "61",
                        "--out",
                        "collision-sketch.jsonl",
                        "--summary-out",
                        "summary.json",
                    ]
                )
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(writes), 1)
        provenance = writes[0]["audit_provenance"]
        self.assertEqual(provenance["showdown_source_hash"], "source-hash")
        self.assertEqual(provenance["observation_schema"], OBSERVATION_SCHEMA_VERSION_V3)
        self.assertEqual(provenance["public_repo_commit"], "a" * 40)
        self.assertEqual(provenance["command"][1:3], ["--capture-driver", "random-legal"])
        self.assertEqual(
            provenance["execution_scope"],
            {
                "seed_range": {"start": 17, "end": 19, "count": 3},
                "capture_driver": "random-legal",
                "max_decision_rounds": 61,
            },
        )

    def test_protocol_only_capture_allows_no_observation_schema(self) -> None:
        """Tied/capped captures still publish their count-only census safely."""

        config = ControlledFoulPlayConfig(
            checkpoint=None,
            showdown_root=Path("/showdown"),
            policy_mode="raw",
            capture_driver="random-legal",
            audit_observation_schema="v3",
        )
        collision_result = ControlledFoulPlayCollisionSketchResult(
            benchmark=ControlledFoulPlayBenchmarkResult(config=config, policy_id="audit-random-legal", games=()),
            output_path=Path("collision-sketch.jsonl"),
            pool_id="fixture",
            checkpoint_sha256=None,
            belief_set_source_hash="source-hash",
            observation_schema_version=None,
            captured_games=0,
            skipped_capped_games=1,
            skipped_tied_games=1,
            captured_decisions=0,
            resumed_decisions=0,
            captured_new_decisions=0,
            recovered_trailing_partial=False,
            protocol_signature_schema_version=PROTOCOL_SIGNATURE_SCHEMA_VERSION,
            protocol_signatures={"cant:recharge": 2},
            protocol_signature_game_ids=("a" * 64,),
        )
        writes: list[dict] = []
        with (
            patch(
                "pokezero.foulplay_collision_capture.capture_controlled_foulplay_collision_sketch",
                new_callable=AsyncMock,
                return_value=collision_result,
            ),
            patch(
                "pokezero.foulplay_collision_capture.load_gen3_randbat_source_cached",
                return_value=SimpleNamespace(metadata=SimpleNamespace(source_hash="source-hash")),
            ),
            patch("pokezero.foulplay_collision_capture.public_repo_commit", return_value="a" * 40),
            patch("pokezero.foulplay_collision_capture._write_json", side_effect=lambda _path, payload: writes.append(payload)),
        ):
            exit_code = asyncio.run(
                async_main(
                    [
                        "--capture-driver",
                        "random-legal",
                        "--observation-schema",
                        "v3",
                        "--showdown-root",
                        "/showdown",
                        "--out",
                        "collision-sketch.jsonl",
                        "--summary-out",
                        "summary.json",
                    ]
                )
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(writes[0]["protocol_signatures"], {"cant:recharge": 2})
        self.assertIsNone(writes[0]["collision_sketch_capture"]["observation_schema_version"])

    def test_resume_protocol_signature_census_requires_atomic_progress_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sketch_path = root / "sketch.jsonl"
            summary_path = root / "summary.json"
            provenance = {
                "schema_version": "pokezero.protocol-capture-provenance.v1",
                "public_repo_commit": "a" * 40,
                "showdown_source_hash": "source-hash",
                "observation_schema": OBSERVATION_SCHEMA_VERSION_V3,
                "image_digest": "fixture-image",
                "execution_scope": {
                    "seed_range": {"start": 17, "end": 19, "count": 3},
                    "capture_driver": "random-legal",
                    "max_decision_rounds": 61,
                },
            }
            sketch_path.touch()
            summary_path.write_text(
                json.dumps(
                    {
                        "protocol_signature_schema_version": PROTOCOL_SIGNATURE_SCHEMA_VERSION,
                        "protocol_signatures": {"move:protect": 3},
                        "protocol_signature_game_ids": ["b" * 64],
                        "collision_sketch_capture": {"out": str(sketch_path), "pool_id": "fixture"},
                        "audit_provenance": {**provenance, "recorded_at": "2026-07-20T00:00:00+00:00"},
                    }
                ),
                encoding="utf-8",
            )

            counts, game_ids = _resume_protocol_signature_census(
                sketch_path=sketch_path,
                summary_path=summary_path,
                expected_provenance=provenance,
                expected_pool_id="fixture",
            )
            self.assertEqual(counts, {"move:protect": 3})
            self.assertEqual(game_ids, ("b" * 64,))

            sketch_path.unlink()
            with self.assertRaisesRegex(ValueError, "has no matching collision sketch"):
                _resume_protocol_signature_census(
                    sketch_path=sketch_path,
                    summary_path=summary_path,
                    expected_provenance=provenance,
                    expected_pool_id="fixture",
                )

            sketch_path.touch()
            summary_path.unlink()
            with self.assertRaisesRegex(ValueError, "without a resumable protocol census summary"):
                _resume_protocol_signature_census(
                    sketch_path=sketch_path,
                    summary_path=summary_path,
                    expected_provenance=provenance,
                    expected_pool_id="fixture",
                )

    def test_resume_protocol_signature_census_rejects_provenance_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sketch_path = root / "sketch.jsonl"
            summary_path = root / "summary.json"
            provenance = {
                "schema_version": "pokezero.protocol-capture-provenance.v1",
                "public_repo_commit": "a" * 40,
                "showdown_source_hash": "source-hash",
                "observation_schema": OBSERVATION_SCHEMA_VERSION_V3,
                "image_digest": "fixture-image",
                "execution_scope": {
                    "seed_range": {"start": 17, "end": 19, "count": 3},
                    "capture_driver": "random-legal",
                    "max_decision_rounds": 61,
                },
            }
            sketch_path.touch()
            summary_path.write_text(
                json.dumps(
                    {
                        "protocol_signature_schema_version": PROTOCOL_SIGNATURE_SCHEMA_VERSION,
                        "protocol_signatures": {"move:protect": 3},
                        "protocol_signature_game_ids": ["b" * 64],
                        "collision_sketch_capture": {"out": str(sketch_path), "pool_id": "fixture"},
                        "audit_provenance": {
                            **provenance,
                            "image_digest": "different-image",
                            "recorded_at": "2026-07-20T00:00:00+00:00",
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "incompatible source, schema, or image provenance"):
                _resume_protocol_signature_census(
                    sketch_path=sketch_path,
                    summary_path=summary_path,
                    expected_provenance=provenance,
                    expected_pool_id="fixture",
                )

    def test_resume_protocol_signature_census_rejects_pool_id_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sketch_path = root / "sketch.jsonl"
            summary_path = root / "summary.json"
            provenance = {
                "schema_version": "pokezero.protocol-capture-provenance.v1",
                "public_repo_commit": "a" * 40,
                "showdown_source_hash": "source-hash",
                "observation_schema": OBSERVATION_SCHEMA_VERSION_V3,
                "image_digest": "fixture-image",
            }
            sketch_path.touch()
            summary_path.write_text(
                json.dumps(
                    {
                        "protocol_signature_schema_version": PROTOCOL_SIGNATURE_SCHEMA_VERSION,
                        "protocol_signatures": {"move:protect": 3},
                        "protocol_signature_game_ids": ["b" * 64],
                        "collision_sketch_capture": {"out": str(sketch_path), "pool_id": "old-pool"},
                        "audit_provenance": {**provenance, "recorded_at": "2026-07-20T00:00:00+00:00"},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "does not match this sketch"):
                _resume_protocol_signature_census(
                    sketch_path=sketch_path,
                    summary_path=summary_path,
                    expected_provenance=provenance,
                    expected_pool_id="new-pool",
                )

    def test_resume_protocol_signature_census_rejects_execution_scope_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sketch_path = root / "sketch.jsonl"
            summary_path = root / "summary.json"
            provenance = {
                "schema_version": "pokezero.protocol-capture-provenance.v1",
                "public_repo_commit": "a" * 40,
                "showdown_source_hash": "source-hash",
                "observation_schema": OBSERVATION_SCHEMA_VERSION_V3,
                "image_digest": "fixture-image",
                "execution_scope": {
                    "seed_range": {"start": 17, "end": 19, "count": 3},
                    "capture_driver": "random-legal",
                    "max_decision_rounds": 61,
                },
            }
            sketch_path.touch()
            summary_path.write_text(
                json.dumps(
                    {
                        "protocol_signature_schema_version": PROTOCOL_SIGNATURE_SCHEMA_VERSION,
                        "protocol_signatures": {"move:protect": 3},
                        "protocol_signature_game_ids": ["b" * 64],
                        "collision_sketch_capture": {"out": str(sketch_path), "pool_id": "fixture"},
                        "audit_provenance": {
                            **provenance,
                            "execution_scope": {
                                **provenance["execution_scope"],
                                "seed_range": {"start": 20, "end": 22, "count": 3},
                            },
                            "recorded_at": "2026-07-20T00:00:00+00:00",
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "incompatible source, schema, or image provenance"):
                _resume_protocol_signature_census(
                    sketch_path=sketch_path,
                    summary_path=summary_path,
                    expected_provenance=provenance,
                    expected_pool_id="fixture",
                )

    def test_protocol_signature_census_deduplicates_a_replayed_game(self) -> None:
        config = ControlledFoulPlayConfig(
            checkpoint=None,
            showdown_root=Path("/showdown"),
            policy_mode="raw",
            capture_driver="random-legal",
            audit_observation_schema="v3",
        )
        trajectory = SimpleNamespace(
            terminal=TerminalState(winner=None, turn_count=4, capped=False),
            metadata={
                "protocol_signature_schema_version": PROTOCOL_SIGNATURE_SCHEMA_VERSION,
                "protocol_signatures": {"move:protect": 3},
            },
            seed=123,
        )

        async def replay_same_game(*_args, trajectory_callback, **_kwargs):
            trajectory_callback(trajectory)
            trajectory_callback(trajectory)
            return ControlledFoulPlayBenchmarkResult(config=config, policy_id="audit-random-legal", games=())

        progress: list[dict] = []
        with tempfile.TemporaryDirectory() as directory:
            with patch("pokezero.foulplay_bridge.run_controlled_foulplay_benchmark", new=replay_same_game):
                result = asyncio.run(
                    capture_controlled_foulplay_collision_sketch(
                        config,
                        out_path=Path(directory) / "collision-sketch.jsonl",
                        capture_progress_callback=progress.append,
                    )
                )

        self.assertEqual(result.protocol_signatures, {"move:protect": 3})
        self.assertEqual(len(result.protocol_signature_game_ids), 1)
        self.assertEqual(progress[0]["status"], "running")
        self.assertEqual(progress[0]["protocol_signatures"], {})
        self.assertEqual(progress[0]["protocol_signature_game_ids"], [])


if __name__ == "__main__":
    unittest.main()
