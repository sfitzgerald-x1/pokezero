"""Safety and provenance tests for the source-bound qualification runner."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "mcts_deadline_qualification_runner_under_test",
    ROOT / "scripts" / "run_mcts_deadline_qualification.py",
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _arguments(out_root: Path, *, resume: bool = False) -> list[str]:
    args = [
        "--checkpoint", "/checkpoint.pt",
        "--expected-checkpoint-sha256", "a" * 64,
        "--showdown-root", "/showdown",
        "--corpus", "/corpus.jsonl",
        "--expected-corpus-sha256", "b" * 64,
        "--expected-corpus-file-sha256", "c" * 64,
        "--source-receipt", "/source-receipt.json",
        "--expected-showdown-source-sha256", "d" * 64,
        "--out-root", str(out_root),
    ]
    if resume:
        args.append("--resume")
    return args


class FakeRecord:
    def __init__(self, index: int) -> None:
        self.decision_id = f"decision-{index}"
        self.battle_id = f"battle-{index}"
        self.seat = "p1"
        self.turn_index = index

    def to_payload(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "battle_id": self.battle_id,
            "seat": self.seat,
            "turn_index": self.turn_index,
            "identity": "frozen-public-input",
        }


class DeadlineQualificationRunnerSafetyTest(unittest.TestCase):
    def test_dirty_git_source_is_refused(self) -> None:
        source = ROOT / "pyproject.toml"
        completed = types.SimpleNamespace(stdout="d" * 40 + "\n")
        dirty = types.SimpleNamespace(stdout=" M src/pokezero/engine_search.py\n")
        with (
            mock.patch.object(runner, "_execution_source_files", return_value=[source]),
            mock.patch.object(runner.subprocess, "run", side_effect=(completed, dirty)),
        ):
            with self.assertRaisesRegex(runner.DeadlineQualificationError, "checkout is dirty"):
                runner._active_source_provenance()

    def test_image_without_git_validates_against_content_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image_root = Path(temporary) / "image"
            for relative in runner.REQUIRED_RECEIPT_FILES:
                target = image_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(f"# {relative}\n", encoding="utf-8")
            (image_root / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
            source_files = []
            for directory, patterns in (
                (image_root / "src" / "pokezero", ("*.py",)),
                (image_root / "scripts", ("*.py", "*.mjs")),
                (image_root / "rust" / "pokezero-search", ("*",)),
            ):
                if directory.is_dir():
                    for pattern in patterns:
                        source_files.extend(path for path in directory.rglob(pattern) if path.is_file())
            source_files.append(image_root / "pyproject.toml")
            hashes = {
                path.relative_to(image_root).as_posix(): runner.sha256_file(path)
                for path in source_files
            }
            receipt = {
                "schema_version": runner.SOURCE_RECEIPT_SCHEMA_VERSION,
                "complete": True,
                "immutable_image": "registry.example/pokezero@sha256:" + "e" * 64,
                "source_commit": "d" * 40,
                "execution_tree_sha256": runner._hash_source_files(image_root, source_files),
                "engine_fingerprint": "f" * 64,
                "source_files_sha256": hashes,
            }
            receipt_path = image_root / "receipt.json"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with (
                mock.patch.object(runner, "ROOT", image_root),
                mock.patch.dict(os.environ, {"POKEZERO_COMMIT": "d" * 40}, clear=False),
            ):
                loaded, active = runner._verify_source_receipt(str(receipt_path))
            self.assertEqual(loaded["immutable_image"], receipt["immutable_image"])
            self.assertEqual(active["tree_status"], "explicit_commit_without_git")
            self.assertEqual(active["execution_tree_sha256"], receipt["execution_tree_sha256"])

    def test_stale_installed_native_engine_is_refused(self) -> None:
        receipt = {
            "source_files_sha256": {
                "src/pokezero/engine_search.py": runner.REVIEWED_ENGINE_SEARCH_SHA256,
            },
            "engine_fingerprint": runner.REVIEWED_ENGINE_FINGERPRINT,
        }
        with (
            mock.patch.object(runner, "sha256_file", return_value=runner.REVIEWED_ENGINE_SEARCH_SHA256),
            mock.patch.object(runner, "assert_fresh", side_effect=SystemExit(1)),
        ):
            with self.assertRaisesRegex(runner.DeadlineQualificationError, "freshness"):
                runner._deadline_mechanics_evidence(receipt)

    def test_showdown_runtime_hash_drift_is_refused_before_checkpoint_resolution(self) -> None:
        with mock.patch.object(
            runner,
            "_showdown_source_provenance",
            return_value={"content_sha256": "e" * 64, "git_commit": "f" * 40, "git_clean": True},
        ):
            with self.assertRaisesRegex(runner.DeadlineQualificationError, "Showdown runtime differs"):
                runner._verify_showdown_source("/showdown", "d" * 64)

    def test_reused_unit_with_same_id_but_different_corpus_record_is_refused(self) -> None:
        record = FakeRecord(0)
        payload = {"decision_id": record.decision_id, "corpus_record_sha256": "0" * 64}
        with self.assertRaisesRegex(runner.DeadlineQualificationError, "identity differs"):
            runner._validate_reused_decision(
                payload,
                record=record,
                target=Path("/evidence/decision.json"),
                requirements=runner.DeadlineQualificationRequirements(expected_decisions=1),
            )

    def test_existing_terminal_root_is_never_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_root = Path(temporary) / "existing"
            out_root.mkdir()
            terminal = {"state": "PASS", "sentinel": "preserve-me"}
            (out_root / "PASS.json").write_text(json.dumps(terminal), encoding="utf-8")
            self.assertEqual(runner.main(_arguments(out_root)), 2)
            self.assertEqual(json.loads((out_root / "PASS.json").read_text()), terminal)
            self.assertFalse((out_root / "NONPASS.json").exists())

    def test_mismatched_resume_does_not_append_a_terminal_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_root = Path(temporary) / "resume"
            out_root.mkdir()
            running = {"state": "RUNNING", "sentinel": "other-contract"}
            (out_root / "RUNNING.json").write_text(json.dumps(running), encoding="utf-8")
            (out_root / "MANIFEST.json").write_text(json.dumps({"other": "contract"}), encoding="utf-8")
            self.assertEqual(runner.main(_arguments(out_root, resume=True)), 2)
            self.assertEqual(json.loads((out_root / "RUNNING.json").read_text()), running)
            self.assertFalse((out_root / "NONPASS.json").exists())

    def test_new_root_persists_preflight_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_root = Path(temporary) / "new"
            self.assertEqual(runner.main(_arguments(out_root)), 2)
            terminal = json.loads((out_root / "NONPASS.json").read_text())
            self.assertEqual(terminal["state"], "NONPASS")
            self.assertIn("source receipt", terminal["error"])
            self.assertFalse((out_root / "RUNNING.json").exists())

    def test_complete_replay_writes_source_and_runtime_bound_pass(self) -> None:
        """The executable path, not just the verifier, must publish a real PASS."""

        class FakeContract:
            def to_manifest(self):
                return {"checkpoint_sha256": "a" * 64, "policy_id": "fake"}

        class FakeDecider:
            initialized_with = None

            def __init__(self, *_args, **kwargs):
                type(self).initialized_with = dict(kwargs)

            def prepare(self, record, _config):
                index = int(record.decision_id.rsplit("-", 1)[-1])
                prefix = index == 0
                completed = 128 if prefix else 256

                def decide():
                    return {
                        "root_action": "move 1",
                        "invalid_actions": 0,
                        "engine_mcts": {
                            "leaf_eval": "model",
                            "worlds_constructed": 4,
                            "worlds_searched": 4,
                            "time_budget": {
                                "scope": "whole_model_decision",
                                "requested_ms": 1000,
                                "deadline_elapsed_ms": 1005.0 if prefix else 900.0,
                                "deadline_overshoot_ms": 5.0 if prefix else 0.0,
                                "exhausted": prefix,
                                "worlds_budget_skipped": 0,
                                "native_invocations": [{
                                    "status": "completed",
                                    "multiplicity": 1,
                                    "requested_iterations": 256,
                                    "completed_iterations": completed,
                                    "remaining_iterations": 256 - completed,
                                    "time_budget_ms": 900 if prefix else 1000,
                                    "time_budget_elapsed_ms": 901.0 if prefix else 900.0,
                                    "time_budget_batch_overshoot_ms": 1.0 if prefix else 0.0,
                                    "time_budget_exhausted": prefix,
                                    "root_visits": {"side_one": completed, "side_two": completed},
                                }],
                            },
                        },
                    }

                return decide

            def close(self):
                return None

        records = [FakeRecord(index) for index in range(16)]
        corpus_manifest = types.SimpleNamespace(corpus_sha256="b" * 64)
        receipt = {
            "immutable_image": "registry.example/pokezero@sha256:" + "e" * 64,
            "source_commit": "f" * 40,
        }
        active_source = {"commit": "f" * 40, "execution_tree_sha256": "1" * 64}
        mechanics = {"engine_search_sha256": runner.REVIEWED_ENGINE_SEARCH_SHA256}
        showdown = {"content_sha256": "d" * 64, "git_commit": "2" * 40, "git_clean": True}
        with tempfile.TemporaryDirectory() as temporary:
            out_root = Path(temporary) / "new"
            with (
                mock.patch.object(runner, "_verify_source_receipt", return_value=(receipt, active_source)),
                mock.patch.object(runner, "_deadline_mechanics_evidence", return_value=mechanics),
                mock.patch.object(runner, "_showdown_source_provenance", return_value=showdown),
                mock.patch.object(runner, "sha256_file", return_value="c" * 64),
                mock.patch.object(runner, "read_corpus", return_value=(corpus_manifest, records)),
                mock.patch.object(runner, "validate_representative_timing_panel", return_value={"ok": True}),
                mock.patch.object(runner, "resolve_checkpoint_contract", return_value=FakeContract()) as resolver,
                mock.patch.object(runner, "_LiveEngineTimingDecider", FakeDecider),
            ):
                self.assertEqual(runner.main(_arguments(out_root)), 0)

            terminal = json.loads((out_root / "PASS.json").read_text())
            self.assertEqual(terminal["state"], "PASS")
            self.assertEqual(terminal["summary"]["decision_count"], 16)
            self.assertEqual(terminal["summary"]["native_prefix_count"], 1)
            self.assertFalse(terminal["manifest"]["search_config"]["model_priors"])
            self.assertFalse(terminal["manifest"]["search_config"]["use_opponent_priors"])
            self.assertEqual(terminal["manifest"]["source_receipt"], receipt)
            self.assertEqual(terminal["manifest"]["showdown_source"], showdown)
            resolver.assert_called_once_with(
                "/checkpoint.pt",
                expected_sha256="a" * 64,
                model_device="cpu",
                showdown_root="/showdown",
                showdown_source_sha256="d" * 64,
                expected_showdown_source_sha256="d" * 64,
            )
            self.assertEqual(
                FakeDecider.initialized_with,
                {
                    "model_decision_time_ms": 1000,
                    "model_priors": False,
                    "use_opponent_priors": False,
                },
            )
            self.assertEqual(len(list((out_root / "decisions").glob("*.json"))), 16)
            self.assertTrue((out_root / "DEADLINE_QUALIFICATION_PASS.json").is_file())
            self.assertFalse((out_root / "RUNNING.json").exists())


if __name__ == "__main__":
    unittest.main()
