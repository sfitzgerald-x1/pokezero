import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pokezero.collection import RolloutRecord, read_rollout_records
from pokezero.dataset import iter_training_examples
from pokezero.env import TerminalState
from pokezero.observation import ObservationPerspective, ObservationSpec, PokeZeroObservationV0
from pokezero.replay_import import (
    DEFAULT_REPLAY_POLICY_ID,
    NORMALIZED_REPLAY_SCHEMA_VERSION,
    import_replay_files,
    normalized_replay_payload_from_rollout_record,
    rollout_record_from_normalized_replay,
)
from pokezero.replay_import_cli import main as replay_import_cli_main
from pokezero.trajectory import BattleTrajectory, TrajectoryStep


MASK = (True, False, False, False, False, False, False, False, False)


def observation(player_id: str, showdown_slot: str) -> PokeZeroObservationV0:
    spec = ObservationSpec(categorical_feature_count=1, numeric_feature_count=1)
    return PokeZeroObservationV0(
        categorical_ids=tuple((index,) for index in range(spec.token_count)),
        numeric_features=tuple((float(index),) for index in range(spec.token_count)),
        token_type_ids=tuple(0 for _ in range(spec.token_count)),
        attention_mask=tuple(True for _ in range(spec.token_count)),
        legal_action_mask=MASK,
        perspective=ObservationPerspective.from_showdown_slot(player_id, showdown_slot),
        metadata={"source": "normalized-replay"},
    )


def rollout_record() -> RolloutRecord:
    trajectory = BattleTrajectory(
        battle_id="replay-battle-1",
        format_id="gen3randombattle",
        seed=77,
        metadata={"source": "fixture"},
    )
    trajectory.append(
        TrajectoryStep(
            player_id="p1",
            turn_index=0,
            observation=observation("p1", "p1"),
            legal_action_mask=MASK,
            action_index=0,
            reward=0.0,
            opponent_action_index=0,
            action_probability=1.0,
            metadata={"raw_choice": "move 1"},
        )
    )
    trajectory.append(
        TrajectoryStep(
            player_id="p2",
            turn_index=0,
            observation=observation("p2", "p2"),
            legal_action_mask=MASK,
            action_index=0,
            reward=0.0,
            opponent_action_index=0,
            action_probability=1.0,
            metadata={"raw_choice": "move 1"},
        )
    )
    trajectory.record_terminal(TerminalState(winner="p1", turn_count=1))
    return RolloutRecord(
        battle_id=trajectory.battle_id,
        seed=trajectory.seed,
        format_id=trajectory.format_id,
        policy_ids={"p1": "replay:p1", "p2": "replay:p2"},
        decision_round_count=1,
        elapsed_seconds=0.2,
        terminal=trajectory.terminal,
        trajectory=trajectory,
    )


class ReplayImportTest(unittest.TestCase):
    def test_import_replay_files_writes_trainable_rollout_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "replay.json"
            output_path = temp_path / "rollouts.jsonl"
            input_path.write_text(
                json.dumps(normalized_replay_payload_from_rollout_record(rollout_record())),
                encoding="utf-8",
            )

            result = import_replay_files((input_path,), output_path=output_path)
            records = read_rollout_records(output_path)
            examples = list(iter_training_examples((output_path,)))

        self.assertEqual(result.records_written, 1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].battle_id, "replay-battle-1")
        self.assertEqual(records[0].policy_ids, {"p1": "replay:p1", "p2": "replay:p2"})
        self.assertEqual(records[0].decision_round_count, 1)
        self.assertEqual(records[0].terminal.winner, "p1")
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0].action_index, 0)

    def test_default_policy_ids_are_applied_for_recorded_players(self) -> None:
        payload = normalized_replay_payload_from_rollout_record(rollout_record())
        del payload["policy_ids"]

        record = rollout_record_from_normalized_replay(payload)

        self.assertEqual(record.policy_ids, {"p1": DEFAULT_REPLAY_POLICY_ID, "p2": DEFAULT_REPLAY_POLICY_ID})

    def test_import_rejects_illegal_recorded_action(self) -> None:
        payload = normalized_replay_payload_from_rollout_record(rollout_record())
        payload["steps"][0]["action_index"] = 1

        with self.assertRaisesRegex(ValueError, "action_index must be legal"):
            rollout_record_from_normalized_replay(payload)

    def test_import_rejects_unknown_normalized_replay_schema(self) -> None:
        payload = normalized_replay_payload_from_rollout_record(rollout_record())
        payload["schema_version"] = "other"

        with self.assertRaisesRegex(ValueError, "Unsupported normalized replay schema"):
            rollout_record_from_normalized_replay(payload)

    def test_replay_import_cli_import_writes_json_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "replay.json"
            output_path = temp_path / "rollouts.jsonl"
            input_path.write_text(
                json.dumps(normalized_replay_payload_from_rollout_record(rollout_record())),
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with patch("sys.stdout", stdout):
                exit_code = replay_import_cli_main(
                    [
                        "import",
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                        "--json",
                    ]
                )

            summary = json.loads(stdout.getvalue())
            records = read_rollout_records(output_path)

        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["records_written"], 1)
        self.assertEqual(summary["output_path"], str(output_path))
        self.assertEqual(len(records), 1)

    def test_normalized_replay_payload_uses_current_schema_version(self) -> None:
        payload = normalized_replay_payload_from_rollout_record(rollout_record())

        self.assertEqual(payload["schema_version"], NORMALIZED_REPLAY_SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
