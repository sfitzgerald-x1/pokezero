"""Safety checks around terminalizing a source-bound qualification root."""

from __future__ import annotations

import importlib.util
import json
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
        "--expected-source-commit", "d" * 40,
        "--expected-deadline-source-commit", "e" * 40,
        "--out-root", str(out_root),
    ]
    if resume:
        args.append("--resume")
    return args


class DeadlineQualificationRunnerSafetyTest(unittest.TestCase):
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
            self.assertIn("source commit", terminal["error"])
            self.assertFalse((out_root / "RUNNING.json").exists())

    def test_complete_replay_writes_all_durable_units_and_pass_marker(self) -> None:
        """The executable path, not just the verifier, must publish a real PASS."""

        class FakeContract:
            def to_manifest(self):
                return {"checkpoint_sha256": "a" * 64, "policy_id": "fake"}

        class FakeDecider:
            initialized_with = None

            def __init__(self, *_args, **_kwargs):
                self.closed = False
                type(self).initialized_with = dict(_kwargs)

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
                                "native_invocations": [
                                    {
                                        "status": "completed",
                                        "multiplicity": 1,
                                        "requested_iterations": 256,
                                        "completed_iterations": completed,
                                        "remaining_iterations": 256 - completed,
                                        "time_budget_ms": 900 if prefix else 1000,
                                        "time_budget_elapsed_ms": 901.0 if prefix else 900.0,
                                        "time_budget_batch_overshoot_ms": 1.0 if prefix else 0.0,
                                        "time_budget_exhausted": prefix,
                                        "root_visits": {
                                            "side_one": completed,
                                            "side_two": completed,
                                        },
                                    }
                                ],
                            },
                        },
                    }

                return decide

            def close(self):
                self.closed = True

        records = [
            types.SimpleNamespace(
                decision_id=f"decision-{index}",
                battle_id=f"battle-{index}",
                seat="p1",
                turn_index=index,
            )
            for index in range(16)
        ]
        corpus_manifest = types.SimpleNamespace(corpus_sha256="b" * 64)
        with tempfile.TemporaryDirectory() as temporary:
            out_root = Path(temporary) / "new"
            args = _arguments(out_root)
            args[args.index("--expected-source-commit") + 1] = "source"
            with (
                mock.patch.object(runner, "_source_commit", return_value="source"),
                mock.patch.object(runner, "_require_clean_source"),
                mock.patch.object(runner, "sha256_file", return_value="c" * 64),
                mock.patch.object(runner, "read_corpus", return_value=(corpus_manifest, records)),
                mock.patch.object(runner, "validate_representative_timing_panel", return_value={"ok": True}),
                mock.patch.object(runner, "resolve_checkpoint_contract", return_value=FakeContract()),
                mock.patch.object(
                    runner,
                    "_deadline_mechanics_source",
                    return_value={"commit": "deadline", "paths": {"engine": "blob"}},
                ),
                mock.patch.object(runner, "_LiveEngineTimingDecider", FakeDecider),
            ):
                self.assertEqual(runner.main(args), 0)

            terminal = json.loads((out_root / "PASS.json").read_text())
            self.assertEqual(terminal["state"], "PASS")
            self.assertEqual(terminal["summary"]["decision_count"], 16)
            self.assertEqual(terminal["summary"]["native_prefix_count"], 1)
            self.assertFalse(terminal["manifest"]["search_config"]["model_priors"])
            self.assertFalse(terminal["manifest"]["search_config"]["use_opponent_priors"])
            self.assertEqual(terminal["manifest"]["deadline_mechanics_source"]["commit"], "deadline")
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
