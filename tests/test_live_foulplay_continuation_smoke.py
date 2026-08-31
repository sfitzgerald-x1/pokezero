from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

from pokezero.env import TerminalState
from pokezero.trajectory import BattleTrajectory


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "live_foulplay_continuation_smoke.py"


def _module():
    spec = importlib.util.spec_from_file_location("live_foulplay_continuation_smoke", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trajectory(proof: object) -> BattleTrajectory:
    trajectory = BattleTrajectory(
        battle_id="controlled-foulplay-118000000",
        format_id="gen3randombattle",
        seed=118_000_000,
        metadata={"controlled_foulplay_bridge": True, "live_foulplay_continuation_smoke": proof},
    )
    trajectory.record_terminal(TerminalState(winner="p1", turn_count=8, capped=False))
    return trajectory


class LiveFoulPlayContinuationSmokeTest(unittest.TestCase):
    def test_extracts_a_validated_live_continuation_proof(self) -> None:
        module = _module()
        proof = module.proof_from_trajectory(
            _trajectory(
                {
                    "source_request_sha256": {"p1": "a", "p2": "b"},
                    "snapshot_request_sha256": {"p1": "c", "p2": "d"},
                    "first_restored_joint_step": {"p1": 1, "p2": 2},
                    "actual_foulplay_choice": "move 3",
                    "decoded_actual_foulplay_action": 2,
                    "continuation_policy_mode": "raw",
                    "full_state_snapshot_scope": "scorer-only",
                    "source_decision_round": 2,
                    "continuation": {
                        "decision_round_count": 2,
                        "terminal": {"winner": "p2", "turn_count": 9, "capped": False},
                    },
                }
            )
        )
        self.assertEqual(proof["first_restored_joint_step"], {"p1": 1, "p2": 2})

    def test_rejects_a_missing_or_capped_continuation_proof(self) -> None:
        module = _module()
        with self.assertRaisesRegex(module.ContinuationSmokeError, "lacks a continuation proof"):
            module.proof_from_trajectory(_trajectory(None))
        with self.assertRaisesRegex(module.ContinuationSmokeError, "capped"):
            module.proof_from_trajectory(
                _trajectory(
                    {
                        "source_request_sha256": {"p1": "a", "p2": "b"},
                        "snapshot_request_sha256": {"p1": "c", "p2": "d"},
                        "first_restored_joint_step": {"p1": 1, "p2": 2},
                        "actual_foulplay_choice": "move 3",
                        "decoded_actual_foulplay_action": 2,
                        "continuation_policy_mode": "raw",
                        "full_state_snapshot_scope": "scorer-only",
                        "source_decision_round": 2,
                        "continuation": {
                            "decision_round_count": 1,
                            "terminal": {"winner": None, "turn_count": 8, "capped": True},
                        },
                    }
                )
            )

    def test_refuses_to_replace_an_existing_receipt(self) -> None:
        module = _module()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            module._write_new_json(path, {"first": True})
            with self.assertRaises(FileExistsError):
                module._write_new_json(path, {"second": True})


if __name__ == "__main__":
    unittest.main()
