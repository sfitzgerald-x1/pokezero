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
    BELIEF_FIDELITY_DEGENERATE,
    BELIEF_FIDELITY_FULL,
    BELIEF_FIDELITY_KEY,
    COLLECTION_ENV_ENGINE,
    COLLECTION_ENV_KEY,
    COLLECTION_ENV_SHOWDOWN,
    TrajectoryDatasetConfig,
    concat_training_caches,
    write_training_cache_from_rollouts,
    write_training_cache_streaming,
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


def write_env_cache(
    root: Path,
    name: str,
    records,
    *,
    config: TrajectoryDatasetConfig,
    collection_env: str,
    belief_fidelity: str,
) -> Path:
    """A cache stamped with an explicit collection environment."""
    jsonl = root / f"{name}.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            write_rollout_record(handle, record)
    cache = root / name
    write_training_cache_streaming(
        jsonl,
        cache,
        config=config,
        collection_env=collection_env,
        belief_fidelity=belief_fidelity,
    )
    return cache


def strip_provenance(cache: Path) -> Path:
    """Make a cache look like a pre-provenance production shard."""
    path = cache / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata.pop(COLLECTION_ENV_KEY, None)
    metadata.pop(BELIEF_FIDELITY_KEY, None)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cache


def metadata_of(cache: Path) -> dict:
    return json.loads((cache / "metadata.json").read_text(encoding="utf-8"))


@unittest.skipUnless(NUMPY, "requires numpy")
class CollectionEnvProvenanceTests(unittest.TestCase):
    """collection_env is a fail-closed concat gate; belief_fidelity is recorded only.

    The engine-env backend writes shards that are schema-identical to Showdown's,
    so nothing in the arrays can catch a mix. The stamp is the only guard, and the
    back-compat rule is load-bearing: every production shard predates the field,
    so a MISSING key must behave exactly like "showdown".
    """

    def setUp(self) -> None:
        self.config = TrajectoryDatasetConfig(window_size=1)
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.addCleanup(self._temp.cleanup)

    def showdown_cache(self, name: str, seed: int) -> Path:
        return write_env_cache(
            self.root,
            name,
            [rollout_record(seed)],
            config=self.config,
            collection_env=COLLECTION_ENV_SHOWDOWN,
            belief_fidelity=BELIEF_FIDELITY_FULL,
        )

    def engine_cache(self, name: str, seed: int) -> Path:
        return write_env_cache(
            self.root,
            name,
            [rollout_record(seed)],
            config=self.config,
            collection_env=COLLECTION_ENV_ENGINE,
            belief_fidelity=BELIEF_FIDELITY_DEGENERATE,
        )

    # -- stamping ---------------------------------------------------------

    def test_writers_stamp_both_provenance_fields(self) -> None:
        showdown = metadata_of(self.showdown_cache("stamp-showdown", 1))
        engine = metadata_of(self.engine_cache("stamp-engine", 2))
        self.assertEqual(showdown[COLLECTION_ENV_KEY], COLLECTION_ENV_SHOWDOWN)
        self.assertEqual(showdown[BELIEF_FIDELITY_KEY], BELIEF_FIDELITY_FULL)
        self.assertEqual(engine[COLLECTION_ENV_KEY], COLLECTION_ENV_ENGINE)
        self.assertEqual(engine[BELIEF_FIDELITY_KEY], BELIEF_FIDELITY_DEGENERATE)

    def test_default_stamp_is_showdown(self) -> None:
        cache = write_cache(self.root, "default", [rollout_record(1)], config=self.config)
        metadata = metadata_of(cache)
        self.assertEqual(metadata[COLLECTION_ENV_KEY], COLLECTION_ENV_SHOWDOWN)
        self.assertEqual(metadata[BELIEF_FIDELITY_KEY], BELIEF_FIDELITY_FULL)

    def test_unknown_env_rejected_at_write(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown collection_env"):
            write_env_cache(
                self.root,
                "bogus",
                [rollout_record(1)],
                config=self.config,
                collection_env="shodown",
                belief_fidelity=BELIEF_FIDELITY_FULL,
            )

    # -- the four concat cases --------------------------------------------

    def test_showdown_plus_showdown_concatenates(self) -> None:
        a = self.showdown_cache("sd-a", 1)
        b = self.showdown_cache("sd-b", 2)
        merged = self.root / "merged-sd-sd"
        concat_training_caches((a, b), merged)
        metadata = metadata_of(merged)
        self.assertEqual(metadata[COLLECTION_ENV_KEY], COLLECTION_ENV_SHOWDOWN)
        self.assertEqual(metadata[BELIEF_FIDELITY_KEY], BELIEF_FIDELITY_FULL)

    def test_legacy_unstamped_plus_showdown_concatenates(self) -> None:
        """The back-compat rule: every shard in production today has no key."""
        legacy = strip_provenance(self.showdown_cache("legacy", 1))
        self.assertNotIn(COLLECTION_ENV_KEY, metadata_of(legacy))
        fresh = self.showdown_cache("fresh", 2)
        merged = self.root / "merged-legacy-sd"
        concat_training_caches((legacy, fresh), merged)
        metadata = metadata_of(merged)
        # Normalized on the way out too: a merge of legacy Showdown shards is a
        # Showdown shard and says so, even when the FIRST part was unstamped.
        self.assertEqual(metadata[COLLECTION_ENV_KEY], COLLECTION_ENV_SHOWDOWN)
        self.assertEqual(metadata[BELIEF_FIDELITY_KEY], BELIEF_FIDELITY_FULL)

    def test_legacy_plus_legacy_concatenates(self) -> None:
        a = strip_provenance(self.showdown_cache("legacy-a", 1))
        b = strip_provenance(self.showdown_cache("legacy-b", 2))
        merged = self.root / "merged-legacy-legacy"
        concat_training_caches((a, b), merged)
        self.assertEqual(metadata_of(merged)[COLLECTION_ENV_KEY], COLLECTION_ENV_SHOWDOWN)

    def test_engine_plus_engine_concatenates(self) -> None:
        a = self.engine_cache("eng-a", 1)
        b = self.engine_cache("eng-b", 2)
        merged = self.root / "merged-eng-eng"
        concat_training_caches((a, b), merged)
        metadata = metadata_of(merged)
        self.assertEqual(metadata[COLLECTION_ENV_KEY], COLLECTION_ENV_ENGINE)
        self.assertEqual(metadata[BELIEF_FIDELITY_KEY], BELIEF_FIDELITY_DEGENERATE)

    def test_engine_plus_showdown_fails_closed(self) -> None:
        engine = self.engine_cache("mix-eng", 1)
        showdown = self.showdown_cache("mix-sd", 2)
        with self.assertRaises(ValueError) as caught:
            concat_training_caches((showdown, engine), self.root / "merged-mixed")
        message = str(caught.exception)
        # The message must name both paths and both env values — a bare
        # "not concatenable" would send the reader hunting through shard dirs.
        self.assertIn(str(showdown), message)
        self.assertIn(str(engine), message)
        self.assertIn(COLLECTION_ENV_SHOWDOWN, message)
        self.assertIn(COLLECTION_ENV_ENGINE, message)
        self.assertFalse((self.root / "merged-mixed").exists())

    def test_showdown_plus_engine_fails_closed_in_either_order(self) -> None:
        engine = self.engine_cache("order-eng", 1)
        showdown = self.showdown_cache("order-sd", 2)
        with self.assertRaisesRegex(ValueError, COLLECTION_ENV_KEY):
            concat_training_caches((engine, showdown), self.root / "merged-order")

    def test_legacy_plus_engine_fails_closed(self) -> None:
        """Legacy normalizes to showdown, so it must refuse engine too."""
        legacy = strip_provenance(self.showdown_cache("legacy-mix", 1))
        engine = self.engine_cache("legacy-mix-eng", 2)
        with self.assertRaisesRegex(ValueError, COLLECTION_ENV_ENGINE):
            concat_training_caches((legacy, engine), self.root / "merged-legacy-eng")

    def test_belief_fidelity_is_recorded_not_gated(self) -> None:
        """Same env, differing fidelity: merges (not a gate), records null (not a lie)."""
        a = self.engine_cache("fid-a", 1)
        b = write_env_cache(
            self.root,
            "fid-b",
            [rollout_record(2)],
            config=self.config,
            collection_env=COLLECTION_ENV_ENGINE,
            belief_fidelity=BELIEF_FIDELITY_FULL,
        )
        merged = self.root / "merged-fidelity"
        concat_training_caches((a, b), merged)
        metadata = metadata_of(merged)
        self.assertEqual(metadata[COLLECTION_ENV_KEY], COLLECTION_ENV_ENGINE)
        self.assertIsNone(metadata[BELIEF_FIDELITY_KEY])


if __name__ == "__main__":
    unittest.main()
