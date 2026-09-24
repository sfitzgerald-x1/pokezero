"""Focused contract tests for the source-root leaf continuation runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from dataclasses import dataclass


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "source_root_leaf_continuations_test", ROOT / "scripts" / "run_source_root_leaf_continuations.py"
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


@dataclass(frozen=True)
class _Record:
    seed: int = 1
    battle_id: str = "battle-1"
    turn_index: int = 2

    def to_dict(self) -> dict[str, object]:
        return {"seed": self.seed, "battle_id": self.battle_id, "turn_index": self.turn_index}


RECORD = _Record()
LEAF = {
    "arms": {
        "model_control_a": {"selection": {"root_action": "move 1"}},
        "rollout_leaf": {"selection": {"root_action": "move 2"}},
    }
}
EXPECTED_ACTIONS = {"model_leaf": 1, "rollout_leaf": 2}


def _payload(*, leaked_opponent: bool = False, capped: bool = False, source_hash: str | None = None) -> dict[str, object]:
    outcomes = []
    for label, action in (("model_leaf", 1), ("rollout_leaf", 2)):
        outcomes.append(
            {
                "action_label": label,
                "action_index": action,
                "continuation": {
                    "decision_round_count": 5,
                    "terminal_after_fixed_joint_step": False,
                    "terminal": {"winner": "p1", "turn_count": 9, "capped": capped},
                    "initial_max_continuation_decision_rounds": RUNNER.INITIAL_MAX_CONTINUATION_DECISION_ROUNDS,
                    "effective_max_continuation_decision_rounds": RUNNER.INITIAL_MAX_CONTINUATION_DECISION_ROUNDS,
                    "cap_retry": False,
                },
            }
        )
    grid: dict[str, object] = {
        "schema_version": "pokezero.sealed-root-action-grid.v2",
        "source_battle_id": RECORD.battle_id,
        "source_seed": RECORD.seed,
        "source_decision_round": RECORD.turn_index,
        "subject_player": "p1",
        "opponent_player": "p2",
        "opponent_action_held_fixed": True,
        "actions": [
            {"action_label": "model_leaf", "action_index": 1},
            {"action_label": "rollout_leaf", "action_index": 2},
        ],
        "search_evidence": {
            "model_leaf_choice": "move 1",
            "rollout_leaf_choice": "move 2",
            "selection_changed": True,
            "leaf_only_intervention": True,
        },
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
        "source_record_sha256": source_hash or RUNNER._sha256(RECORD.to_dict()),
        "leaf_complete_sha256": RUNNER._sha256(LEAF),
        "grid": grid,
    }


class ContinuationContractTest(unittest.TestCase):
    def test_completed_root_accepts_complete_paired_grid(self) -> None:
        RUNNER._validate_completed_root(
            _payload(),
            root=RUNNER.SourceRoot(1, "p1", 2),
            record=RECORD,
            leaf_payload=LEAF,
            expected_actions=EXPECTED_ACTIONS,
            manifest_sha256="m" * 64,
        )

    def test_completed_root_refuses_opponent_action_leak(self) -> None:
        with self.assertRaisesRegex(RUNNER.ContinuationError, "leaks"):
            RUNNER._validate_completed_root(
                _payload(leaked_opponent=True),
                root=RUNNER.SourceRoot(1, "p1", 2),
                record=RECORD,
                leaf_payload=LEAF,
                expected_actions=EXPECTED_ACTIONS,
                manifest_sha256="m" * 64,
            )

    def test_completed_root_refuses_capped_suffix(self) -> None:
        with self.assertRaisesRegex(RUNNER.ContinuationError, "incomplete or capped"):
            RUNNER._validate_completed_root(
                _payload(capped=True),
                root=RUNNER.SourceRoot(1, "p1", 2),
                record=RECORD,
                leaf_payload=LEAF,
                expected_actions=EXPECTED_ACTIONS,
                manifest_sha256="m" * 64,
            )

    def test_completed_root_refuses_source_record_drift(self) -> None:
        with self.assertRaisesRegex(RUNNER.ContinuationError, "source record drifted"):
            RUNNER._validate_completed_root(
                _payload(source_hash="x" * 64),
                root=RUNNER.SourceRoot(1, "p1", 2),
                record=RECORD,
                leaf_payload=LEAF,
                expected_actions=EXPECTED_ACTIONS,
                manifest_sha256="m" * 64,
            )

    def test_completed_root_refuses_missing_terminal_winner(self) -> None:
        payload = _payload()
        outcome = payload["grid"]["continuation_targets"][0]["trials"][0]["outcomes"][0]
        del outcome["continuation"]["terminal"]["winner"]
        with self.assertRaisesRegex(RUNNER.ContinuationError, "winner is invalid"):
            RUNNER._validate_completed_root(
                payload,
                root=RUNNER.SourceRoot(1, "p1", 2),
                record=RECORD,
                leaf_payload=LEAF,
                expected_actions=EXPECTED_ACTIONS,
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

    def test_terminal_writer_reuses_only_identical_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "terminal.json"
            RUNNER._write_create_only_json(path, {"one": 1})
            RUNNER._write_or_require_identical_json(path, {"one": 1})
            with self.assertRaisesRegex(RUNNER.ContinuationError, "differs"):
                RUNNER._write_or_require_identical_json(path, {"two": 2})


if __name__ == "__main__":
    unittest.main()
