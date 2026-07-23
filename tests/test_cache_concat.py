"""concat_training_caches: the shard fan-in primitive.

Oracle: concatenating cache dirs must be BYTE-IDENTICAL to a single cache
written over the same records in the same order — same arrays, same metadata.
Exercised with window_size=2 (nonzero window_indices must be offset, zero
padding entries must not be) and deliberately different categorical compaction
widths between parts (the narrower part must be zero-padded, documented as
semantically identity).
"""

from __future__ import annotations

from pathlib import Path
import json
import shutil
import tempfile
import unittest

from pokezero.collection import RolloutRecord, write_rollout_record
from pokezero.dataset import (
    TrajectoryDatasetConfig,
    concat_training_caches,
    write_training_cache_from_rollouts,
)
from pokezero.env import TerminalState
from pokezero.observation import ObservationSpec, PokeZeroObservationV0
from pokezero.trajectory import BattleTrajectory, TrajectoryStep

try:
    import numpy  # noqa: F401

    NUMPY = True
except Exception:  # pragma: no cover
    NUMPY = False

LEGAL_TWO_ACTION_MASK = (True, True, False, False, False, False, False, False, False)
SPEC = ObservationSpec(categorical_feature_count=3, numeric_feature_count=1)


def observation(value: int, *, cat_density: int = 1) -> PokeZeroObservationV0:
    """cat_density controls how many of the 3 categorical slots are nonzero,
    which drives the cache's global compaction width."""
    cats = tuple(value + i if i < cat_density else 0 for i in range(3))
    return PokeZeroObservationV0(
        categorical_ids=tuple(cats for _ in range(SPEC.token_count)),
        numeric_features=tuple((float(value),) for _ in range(SPEC.token_count)),
        token_type_ids=tuple(0 for _ in range(SPEC.token_count)),
        attention_mask=tuple(True for _ in range(SPEC.token_count)),
        legal_action_mask=LEGAL_TWO_ACTION_MASK,
    )


def rollout_record(seed: int, *, cat_density: int = 1, turns: int = 4) -> RolloutRecord:
    trajectory = BattleTrajectory(battle_id=f"concat-{seed}", format_id="gen3randombattle", seed=seed)
    for turn_index in range(turns):
        action_index = (turn_index + seed) % 2
        trajectory.append(
            TrajectoryStep(
                player_id="p1",
                turn_index=turn_index,
                observation=observation(action_index + 1 + seed, cat_density=cat_density),
                legal_action_mask=LEGAL_TWO_ACTION_MASK,
                action_index=action_index,
                opponent_action_index=1 - action_index,
            )
        )
    trajectory.record_terminal(TerminalState(winner="p1", turn_count=turns))
    return RolloutRecord(
        battle_id=trajectory.battle_id,
        seed=trajectory.seed,
        format_id=trajectory.format_id,
        policy_ids={"p1": "fixture"},
        decision_round_count=turns,
        elapsed_seconds=0.1,
        terminal=trajectory.terminal,
        trajectory=trajectory,
    )


def write_cache(root: Path, name: str, records, *, config: TrajectoryDatasetConfig) -> Path:
    jsonl = root / f"{name}.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            write_rollout_record(handle, record)
    cache = root / name
    write_training_cache_from_rollouts(jsonl, cache, config=config)
    return cache


@unittest.skipUnless(NUMPY, "requires numpy")
class ConcatOracleTests(unittest.TestCase):
    def assert_caches_byte_identical(self, left: Path, right: Path) -> None:
        left_files = sorted(p.name for p in left.iterdir())
        right_files = sorted(p.name for p in right.iterdir())
        self.assertEqual(left_files, right_files)
        for name in left_files:
            self.assertEqual(
                (left / name).read_bytes(), (right / name).read_bytes(), f"file differs: {name}"
            )

    def test_concat_matches_single_write_with_width_and_window_offsets(self) -> None:
        config = TrajectoryDatasetConfig(window_size=2)
        rec_a = rollout_record(1, cat_density=1)
        rec_b1 = rollout_record(10, cat_density=3)  # wider compaction than A
        rec_b2 = rollout_record(20, cat_density=2)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_a = write_cache(root, "part-a", [rec_a], config=config)
            cache_b = write_cache(root, "part-b", [rec_b1, rec_b2], config=config)
            oracle = write_cache(root, "oracle", [rec_a, rec_b1, rec_b2], config=config)
            merged = root / "merged"
            summary = concat_training_caches((cache_a, cache_b), merged)
            self.assert_caches_byte_identical(oracle, merged)
            oracle_meta = json.loads((oracle / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(summary.record_count, oracle_meta["record_count"])
            self.assertEqual(summary.example_count, oracle_meta["example_count"])

    def test_concat_three_parts_matches_single_write(self) -> None:
        config = TrajectoryDatasetConfig(window_size=1)
        records = [rollout_record(seed, cat_density=1 + seed % 3) for seed in (1, 2, 3)]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parts = [write_cache(root, f"part-{i}", [rec], config=config) for i, rec in enumerate(records)]
            oracle = write_cache(root, "oracle", records, config=config)
            merged = root / "merged"
            concat_training_caches(parts, merged)
            self.assert_caches_byte_identical(oracle, merged)

    def test_config_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_a = write_cache(root, "a", [rollout_record(1)], config=TrajectoryDatasetConfig(window_size=1))
            cache_b = write_cache(root, "b", [rollout_record(2)], config=TrajectoryDatasetConfig(window_size=2))
            with self.assertRaisesRegex(ValueError, "dataset_config"):
                concat_training_caches((cache_a, cache_b), root / "merged")

    def test_array_set_mismatch_fails_closed(self) -> None:
        config = TrajectoryDatasetConfig(window_size=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_a = write_cache(root, "a", [rollout_record(1)], config=config)
            cache_b = write_cache(root, "b", [rollout_record(2)], config=config)
            (cache_b / "turn_indices.npy").unlink()
            with self.assertRaisesRegex(ValueError, "array set"):
                concat_training_caches((cache_a, cache_b), root / "merged")

    def test_single_part_concat_is_a_copy(self) -> None:
        config = TrajectoryDatasetConfig(window_size=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_a = write_cache(root, "a", [rollout_record(1)], config=config)
            merged = root / "merged"
            concat_training_caches((cache_a,), merged)
            self.assert_caches_byte_identical(cache_a, merged)

    def test_refuses_existing_output_without_overwrite(self) -> None:
        config = TrajectoryDatasetConfig(window_size=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_a = write_cache(root, "a", [rollout_record(1)], config=config)
            merged = root / "merged"
            concat_training_caches((cache_a,), merged)
            with self.assertRaises(FileExistsError):
                concat_training_caches((cache_a,), merged)
            shutil.rmtree(merged)


if __name__ == "__main__":
    unittest.main()
