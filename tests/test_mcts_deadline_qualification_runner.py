"""Safety checks around terminalizing a source-bound qualification root."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


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


if __name__ == "__main__":
    unittest.main()
