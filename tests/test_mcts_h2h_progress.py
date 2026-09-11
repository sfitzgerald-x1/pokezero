"""Focused checks for the MCTS-vs-MCTS mutable liveness checkpoint."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_SPEC = importlib.util.spec_from_file_location("mcts_h2h_progress_test", ROOT / "scripts" / "mcts_mcts_h2h.py")
assert _SPEC is not None and _SPEC.loader is not None
DRIVER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = DRIVER
_SPEC.loader.exec_module(DRIVER)


def _payload(*, event: str, seed: int) -> dict[str, object]:
    return {
        "schema_version": DRIVER.PROGRESS_SCHEMA_VERSION,
        "event": event,
        "seed": seed,
        "candidate_seat": "p1",
    }


class MctsH2hProgressTest(unittest.TestCase):
    def test_progress_checkpoint_replaces_only_its_compatible_previous_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "progress" / "current.json"

            DRIVER._write_progress_json(path, _payload(event="game_started", seed=1))
            DRIVER._write_progress_json(path, _payload(event="decision_committed", seed=1))

            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["event"], "decision_committed")

    def test_progress_checkpoint_refuses_a_malformed_existing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "progress" / "current.json"
            path.parent.mkdir()
            path.write_text("not json\n", encoding="utf-8")

            with self.assertRaisesRegex(DRIVER.HeadToHeadError, "unreadable progress checkpoint"):
                DRIVER._write_progress_json(path, _payload(event="game_started", seed=1))
            self.assertEqual(path.read_text(encoding="utf-8"), "not json\n")

    def test_progress_checkpoint_refuses_an_unexpected_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "other.json"

            with self.assertRaisesRegex(DRIVER.HeadToHeadError, "unexpected path"):
                DRIVER._write_progress_json(path, _payload(event="game_started", seed=1))
