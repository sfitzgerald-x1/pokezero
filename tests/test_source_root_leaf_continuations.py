"""Focused contract tests for the source-root leaf continuation runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "source_root_leaf_continuations_test", ROOT / "scripts" / "run_source_root_leaf_continuations.py"
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def _payload(*, leaked_opponent: bool = False, capped: bool = False) -> dict[str, object]:
    outcomes = []
    for label, action in (("model_leaf", 1), ("rollout_leaf", 2)):
        outcomes.append(
            {
                "action_label": label,
                "action_index": action,
                "continuation": {
                    "terminal": {"winner": "p1", "turn_count": 9, "capped": capped},
                },
            }
        )
    grid: dict[str, object] = {
        "schema_version": "pokezero.sealed-root-action-grid.v2",
        "opponent_action_held_fixed": True,
        "actions": [
            {"action_label": "model_leaf", "action_index": 1},
            {"action_label": "rollout_leaf", "action_index": 2},
        ],
        "continuation_targets": [
            {
                "target": target,
                "trials": [
                    {"continuation_rng_seed": seed, "outcomes": outcomes}
                    for seed in RUNNER.CONTINUATION_RNG_SEEDS
                ],
            }
            for target in RUNNER.CONTINUATION_TARGETS
        ],
    }
    if leaked_opponent:
        grid["opponent_action"] = 3
    return {
        "schema_version": RUNNER.SCHEMA_VERSION,
        "state": "COMPLETE",
        "source": {"seed": 1, "seat": "p1", "turn_index": 2},
        "manifest_sha256": "m" * 64,
        "grid": grid,
    }


class ContinuationContractTest(unittest.TestCase):
    def test_completed_root_accepts_complete_paired_grid(self) -> None:
        RUNNER._validate_completed_root(
            _payload(),
            root=RUNNER.SourceRoot(1, "p1", 2),
            manifest_sha256="m" * 64,
        )

    def test_completed_root_refuses_opponent_action_leak(self) -> None:
        with self.assertRaisesRegex(RUNNER.ContinuationError, "leaks"):
            RUNNER._validate_completed_root(
                _payload(leaked_opponent=True),
                root=RUNNER.SourceRoot(1, "p1", 2),
                manifest_sha256="m" * 64,
            )

    def test_completed_root_refuses_capped_suffix(self) -> None:
        with self.assertRaisesRegex(RUNNER.ContinuationError, "incomplete or capped"):
            RUNNER._validate_completed_root(
                _payload(capped=True),
                root=RUNNER.SourceRoot(1, "p1", 2),
                manifest_sha256="m" * 64,
            )

    def test_source_histories_keep_each_seat_order_and_append_current(self) -> None:
        replay = type(
            "Replay",
            (),
            {
                "replay_observations": {
                    1: {"p1": "p1-1", "p2": "p2-1"},
                    0: {"p1": "p1-0"},
                }
            },
        )()
        self.assertEqual(
            RUNNER._source_histories(replay, {"p1": "p1-current", "p2": "p2-current"}),
            {"p1": ("p1-0", "p1-1", "p1-current"), "p2": ("p2-1", "p2-current")},
        )

    def test_terminal_writer_never_replaces(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "terminal.json"
            RUNNER._write_create_only_json(path, {"one": 1})
            with self.assertRaisesRegex(RUNNER.ContinuationError, "refusing to replace"):
                RUNNER._write_create_only_json(path, {"two": 2})
            self.assertEqual(json.loads(path.read_text()), {"one": 1})


if __name__ == "__main__":
    unittest.main()
