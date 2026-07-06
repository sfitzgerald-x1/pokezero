import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.observation import OBSERVATION_SCHEMA_VERSION_V2
from pokezero.opponents import (
    checkpoint_policy_spec_observation_schema,
    current_family_checkpoint_policy_specs,
    current_family_historical_opponent_policy_specs,
    historical_opponent_policy_specs,
    is_current_family_checkpoint_policy_spec,
    opponent_pool_policy_specs,
)


class OpponentPoolTest(unittest.TestCase):
    def test_excludes_current_policy_by_default(self) -> None:
        pool = opponent_pool_policy_specs(
            fixed_policy_specs=("simple-legal", "scripted-teacher"),
            checkpoint_history=("neural:/runs/iter-0001.pt",),
            current_policy_spec="neural:/runs/iter-0002.pt",
            max_historical_opponents=3,
        )
        # No mirror by default; current policy is never an opponent.
        self.assertEqual(pool, ("simple-legal", "scripted-teacher", "neural:/runs/iter-0001.pt"))

    def test_mirror_match_appends_current_policy(self) -> None:
        pool = opponent_pool_policy_specs(
            fixed_policy_specs=("simple-legal", "scripted-teacher"),
            checkpoint_history=(),
            current_policy_spec="neural:/runs/iter-0002.pt",
            max_historical_opponents=3,
            include_current_policy=True,
        )
        # current-vs-current self-play is available from the start.
        self.assertEqual(
            pool, ("simple-legal", "scripted-teacher", "neural:/runs/iter-0002.pt")
        )

    def test_mirror_match_does_not_duplicate_existing_identity(self) -> None:
        # The current policy identity is already in the pool (here via a fixed spec, since
        # history always excludes the current identity) -> mirror must not add a duplicate.
        pool = opponent_pool_policy_specs(
            fixed_policy_specs=("simple-legal", "neural:/runs/iter-0002.pt"),
            checkpoint_history=(),
            current_policy_spec="neural:/runs/iter-0002.pt",
            max_historical_opponents=3,
            include_current_policy=True,
        )
        self.assertEqual(pool.count("neural:/runs/iter-0002.pt"), 1)
        self.assertEqual(pool, ("simple-legal", "neural:/runs/iter-0002.pt"))

    def test_spread_historical_selection_samples_across_history(self) -> None:
        selected = historical_opponent_policy_specs(
            (
                "neural:/runs/iter-0001.pt",
                "neural:/runs/iter-0002.pt",
                "neural:/runs/iter-0003.pt",
                "neural:/runs/iter-0004.pt",
                "neural:/runs/iter-0005.pt",
            ),
            current_policy_spec=None,
            max_historical_opponents=3,
            selection_mode="spread",
        )

        self.assertEqual(
            selected,
            (
                "neural:/runs/iter-0001.pt",
                "neural:/runs/iter-0003.pt",
                "neural:/runs/iter-0005.pt",
            ),
        )

    def test_spread_historical_selection_excludes_current_before_sampling(self) -> None:
        selected = historical_opponent_policy_specs(
            (
                "neural:/runs/iter-0001.pt",
                "neural:/runs/iter-0002.pt",
                "neural:/runs/iter-0003.pt",
                "neural:/runs/iter-0004.pt",
            ),
            current_policy_spec="neural:/runs/iter-0004.pt",
            max_historical_opponents=2,
            selection_mode="spread",
        )

        self.assertEqual(selected, ("neural:/runs/iter-0001.pt", "neural:/runs/iter-0003.pt"))

    def test_spread_historical_selection_single_slot_keeps_latest(self) -> None:
        selected = historical_opponent_policy_specs(
            (
                "neural:/runs/iter-0001.pt",
                "neural:/runs/iter-0002.pt",
                "neural:/runs/iter-0003.pt",
            ),
            current_policy_spec=None,
            max_historical_opponents=1,
            selection_mode="spread",
        )

        self.assertEqual(selected, ("neural:/runs/iter-0003.pt",))

    def test_rejects_unknown_historical_selection_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "selection mode"):
            historical_opponent_policy_specs(
                ("neural:/runs/iter-0001.pt",),
                current_policy_spec=None,
                max_historical_opponents=1,
                selection_mode="unknown",
            )

    def test_current_family_filter_accepts_supported_v2_checkpoint(self) -> None:
        with TemporaryDirectory() as temp_dir:
            checkpoint = _write_checkpoint_stub(Path(temp_dir) / "v2.json", OBSERVATION_SCHEMA_VERSION_V2)
            spec = f"linear:{checkpoint}"

            self.assertEqual(checkpoint_policy_spec_observation_schema(spec), OBSERVATION_SCHEMA_VERSION_V2)
            self.assertTrue(is_current_family_checkpoint_policy_spec(spec))
            self.assertEqual(current_family_checkpoint_policy_specs((spec,)), (spec,))

    def test_current_family_filter_accepts_supported_neural_checkpoint_metadata(self) -> None:
        spec = "neural:/runs/current-family.pt?sample=true"
        config = SimpleNamespace(observation_schema_version=OBSERVATION_SCHEMA_VERSION_V2)

        with patch("pokezero.neural_policy.load_transformer_model_config", return_value=config) as load_config:
            self.assertEqual(checkpoint_policy_spec_observation_schema(spec), OBSERVATION_SCHEMA_VERSION_V2)
            self.assertTrue(is_current_family_checkpoint_policy_spec(spec))

        self.assertEqual(load_config.call_count, 2)
        for call in load_config.call_args_list:
            self.assertEqual(call.args, (Path("/runs/current-family.pt"),))

    def test_current_family_filter_rejects_legacy_checkpoint_by_default(self) -> None:
        with TemporaryDirectory() as temp_dir:
            legacy = _write_checkpoint_stub(Path(temp_dir) / "legacy.json", "pokezero.observation.v1")
            spec = f"linear:{legacy}"

            with self.assertRaisesRegex(ValueError, "legacy or unreadable checkpoint opponents"):
                current_family_checkpoint_policy_specs((spec,))

    def test_current_family_filter_rejects_v2_no_belief_path(self) -> None:
        with TemporaryDirectory() as temp_dir:
            checkpoint = _write_checkpoint_stub(
                Path(temp_dir) / "pokezero-no-belief-gen3-1m.json",
                OBSERVATION_SCHEMA_VERSION_V2,
            )
            spec = f"linear:{checkpoint}"

            with self.assertRaisesRegex(ValueError, "legacy comparison family"):
                current_family_checkpoint_policy_specs((spec,))

    def test_current_family_filter_allows_no_belief_parent_directory_for_current_filename(self) -> None:
        with TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "belief-vs-no-belief-ablation"
            checkpoint_dir.mkdir()
            checkpoint = _write_checkpoint_stub(checkpoint_dir / "belief-v2-500k.json", OBSERVATION_SCHEMA_VERSION_V2)
            spec = f"linear:{checkpoint}"

            self.assertEqual(current_family_checkpoint_policy_specs((spec,)), (spec,))

    def test_current_family_filter_rejects_v2_no_belief_sidecar_metadata(self) -> None:
        with TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "current-family.pt"
            checkpoint.write_bytes(b"not a real torch checkpoint")
            checkpoint.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "name": "pokezero-no-belief-gen3-2m",
                        "input_family": "no-belief",
                    }
                ),
                encoding="utf-8",
            )
            spec = f"neural:{checkpoint}?sample=true"
            config = SimpleNamespace(observation_schema_version=OBSERVATION_SCHEMA_VERSION_V2)

            with patch("pokezero.neural_policy.load_transformer_model_config", return_value=config):
                with self.assertRaisesRegex(ValueError, "legacy comparison family"):
                    current_family_checkpoint_policy_specs((spec,))

    def test_current_family_filter_can_drop_legacy_checkpoint_from_mixed_history(self) -> None:
        with TemporaryDirectory() as temp_dir:
            current = _write_checkpoint_stub(Path(temp_dir) / "current.json", OBSERVATION_SCHEMA_VERSION_V2)
            legacy = _write_checkpoint_stub(Path(temp_dir) / "legacy.json", "pokezero.observation.v1")
            current_spec = f"linear:{current}"
            legacy_spec = f"linear:{legacy}"

            self.assertEqual(
                current_family_checkpoint_policy_specs((legacy_spec, current_spec), legacy_mode="drop"),
                (current_spec,),
            )

    def test_current_family_historical_selection_filters_before_spread_sampling(self) -> None:
        with TemporaryDirectory() as temp_dir:
            legacy = _write_checkpoint_stub(Path(temp_dir) / "legacy.json", "pokezero.observation.v1")
            first = _write_checkpoint_stub(Path(temp_dir) / "first.json", OBSERVATION_SCHEMA_VERSION_V2)
            second = _write_checkpoint_stub(Path(temp_dir) / "second.json", OBSERVATION_SCHEMA_VERSION_V2)
            legacy_spec = f"linear:{legacy}"
            first_spec = f"linear:{first}"
            second_spec = f"linear:{second}"

            selected = current_family_historical_opponent_policy_specs(
                (legacy_spec, first_spec, second_spec),
                current_policy_spec=None,
                max_historical_opponents=2,
                selection_mode="spread",
                legacy_mode="drop",
            )

            self.assertEqual(selected, (first_spec, second_spec))


def _write_checkpoint_stub(path: Path, observation_schema_version: str) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "pokezero.linear_policy.v1",
                "observation_schema_version": observation_schema_version,
            }
        ),
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    unittest.main()
