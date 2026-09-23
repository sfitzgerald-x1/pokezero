from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pokezero.actions import ACTION_COUNT
from pokezero.dataset import (
    TrajectoryDatasetConfig,
    concat_training_caches,
    iter_training_cache_batches,
    write_training_cache_from_examples,
)
from pokezero.live_foulplay_continuation import LIVE_FOULPLAY_SUCCESSOR_CAPTURE_SCHEMA_VERSION
from pokezero.observation import PokeZeroObservationV0
from pokezero.value_leaf_training import (
    VALUE_LEAF_TRAINING_CACHE_SCHEMA_VERSION,
    source_bound_successor_capture_callback,
    value_leaf_training_example,
    write_value_leaf_training_cache,
)


def _observation() -> PokeZeroObservationV0:
    return PokeZeroObservationV0(
        categorical_ids=((0,),),
        numeric_features=((0.0,),),
        token_type_ids=(0,),
        attention_mask=(True,),
        legal_action_mask=tuple(index in {2, 7} for index in range(ACTION_COUNT)),
    )


def _capture(
    *, winner: str | None = "p1", terminal_fixed_step: bool = False,
    source_binding: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": LIVE_FOULPLAY_SUCCESSOR_CAPTURE_SCHEMA_VERSION,
        "source_battle_id": "live-123",
        "format_id": "gen3randombattle",
        "source_seed": 123,
        "source_decision_round": 4,
        "source_request_sha256": {"p1": "a", "p2": "b"},
        "snapshot_request_sha256": {"p1": "c", "p2": "d"},
        "pokezero_player": "p1",
        "foulplay_player": "p2",
        "first_restored_joint_step": {"p1": 7, "p2": 4},
        "successor_observation": _observation(),
        "source_binding": source_binding or _binding(),
        "continuation": {
            "decision_round_count": 0 if terminal_fixed_step else 3,
            "terminal_after_fixed_joint_step": terminal_fixed_step,
            "terminal": {"winner": winner, "turn_count": 8, "capped": False},
        },
    }


def _binding() -> dict[str, object]:
    return {
        "checkpoint_sha256": "a" * 64,
        "checkpoint_iteration": 9375,
        "source_commit": "b" * 40,
        "source_tree_sha256": "c" * 64,
        "source_image_digest": "d" * 64,
        "observation_schema_version": _observation().schema_version,
        "collection_manifest_sha256": "e" * 64,
    }


class ValueLeafTrainingTest(unittest.TestCase):
    def test_materializes_successor_with_pokezero_relative_terminal_value(self) -> None:
        win = value_leaf_training_example(_capture(winner="p1"))
        tie = value_leaf_training_example(_capture(winner=None))
        loss = value_leaf_training_example(_capture(winner="p2"))
        self.assertEqual((win.return_value, tie.return_value, loss.return_value), (1.0, 0.0, -1.0))
        self.assertEqual(win.action_index, 2)
        self.assertEqual(win.window_size, 1)
        self.assertEqual(win.step_metadata["value_leaf_training"]["policy_target"], "not-used-value-only")

    def test_rejects_terminal_fixed_step_that_has_no_successor_leaf(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-terminal fixed joint step"):
            value_leaf_training_example(_capture(terminal_fixed_step=True))

    def test_cache_is_source_bound_and_value_only(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "value-leaves"
            summary = write_value_leaf_training_cache(
                captures=(_capture(winner="p1"),),
                output_path=path,
                source_binding=_binding(),
            )
            self.assertEqual(summary.value_target_counts, {"win": 1, "tie": 0, "loss": 0})
            metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            contract = metadata["value_leaf_training"]
            self.assertEqual(contract["schema_version"], VALUE_LEAF_TRAINING_CACHE_SCHEMA_VERSION)
            self.assertEqual(contract["compatible_objectives"], ["value-only"])
            self.assertEqual(contract["required_window_size"], 1)
            self.assertEqual(contract["source_binding"], _binding())
            self.assertEqual(
                metadata["training_objective_contract"]["compatible_objectives"], ["value-only"]
            )
            batch = next(iter_training_cache_batches(path, batch_size=1))
            self.assertEqual(batch.returns, (1.0,))
            with self.assertRaisesRegex(ValueError, "compatible only with value-only"):
                next(
                    iter_training_cache_batches(
                        path, batch_size=1, objective="behavior-cloning"
                    )
                )

    def test_rejects_wrong_window_or_schema_binding(self) -> None:
        with TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "window_size=1"):
                write_value_leaf_training_cache(
                    captures=(_capture(),),
                    output_path=Path(temp_dir) / "bad-window",
                    source_binding=_binding(),
                    dataset_config=TrajectoryDatasetConfig(window_size=2),
                )
            binding = _binding()
            binding["observation_schema_version"] = "wrong-schema"
            with self.assertRaisesRegex(ValueError, "schema does not match"):
                write_value_leaf_training_cache(
                    captures=(_capture(source_binding=binding),),
                    output_path=Path(temp_dir) / "bad-schema",
                    source_binding=binding,
                )

    def test_rejects_capture_without_matching_sealed_binding(self) -> None:
        with TemporaryDirectory() as temp_dir:
            mismatched = _binding()
            mismatched["checkpoint_iteration"] = 9376
            with self.assertRaisesRegex(ValueError, "does not match"):
                write_value_leaf_training_cache(
                    captures=(_capture(source_binding=mismatched),),
                    output_path=Path(temp_dir) / "mismatched",
                    source_binding=_binding(),
                )

    def test_capture_binding_wrapper_refuses_override_and_attaches_every_receipt(self) -> None:
        captured: list[dict[str, object]] = []
        callback = source_bound_successor_capture_callback(
            source_binding=_binding(), sink=captured.append
        )
        raw_capture = _capture()
        raw_capture.pop("source_binding")
        callback(raw_capture)
        self.assertEqual(captured[0]["source_binding"], _binding())
        with self.assertRaisesRegex(ValueError, "already has a source binding"):
            callback(_capture())

    def test_concat_preserves_value_only_contract_and_aggregates_same_binding(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first"
            second = root / "second"
            merged = root / "merged"
            write_value_leaf_training_cache(
                captures=(_capture(winner="p1"),), output_path=first, source_binding=_binding()
            )
            second_capture = _capture(winner="p2")
            second_capture["source_seed"] = 124
            write_value_leaf_training_cache(
                captures=(second_capture,), output_path=second, source_binding=_binding()
            )
            concat_training_caches((first, second), merged)
            metadata = json.loads((merged / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["value_leaf_training"]["capture_count"], 2)
            self.assertEqual(
                metadata["value_leaf_training"]["value_target_counts"],
                {"win": 1, "tie": 0, "loss": 1},
            )
            with self.assertRaisesRegex(ValueError, "compatible only with value-only"):
                next(iter_training_cache_batches(merged, batch_size=2, objective="ppo"))

    def test_concat_rejects_uncontracted_or_differently_bound_cache(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            special = root / "special"
            ordinary = root / "ordinary"
            other = root / "other"
            write_value_leaf_training_cache(
                captures=(_capture(),), output_path=special, source_binding=_binding()
            )
            write_training_cache_from_examples(
                (value_leaf_training_example(_capture()),),
                ordinary,
                config=TrajectoryDatasetConfig(window_size=1),
            )
            with self.assertRaisesRegex(ValueError, "present on only one input"):
                concat_training_caches((ordinary, special), root / "bad-mixed")
            alternate_binding = _binding()
            alternate_binding["source_commit"] = "f" * 40
            other_capture = _capture(source_binding=alternate_binding)
            other_capture["source_seed"] = 124
            write_value_leaf_training_cache(
                captures=(other_capture,), output_path=other, source_binding=alternate_binding
            )
            with self.assertRaisesRegex(ValueError, "value-leaf source_binding mismatch"):
                concat_training_caches((special, other), root / "bad-binding")


if __name__ == "__main__":
    unittest.main()
