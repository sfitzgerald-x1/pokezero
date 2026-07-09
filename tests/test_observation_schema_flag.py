"""--observation-schema {v2.1,v2.2}: the v2.2 fresh-selection latch.

#516's checkpoint-driven resolution could adopt an existing v2.2 checkpoint but nothing
could START one. The flag SETS the schema on a fresh train/collect (when omitted, the
current default spec applies — v2.2 since the 2026-07-08 promotion); on resume/adoption
the checkpoint wins and an explicitly disagreeing flag hard-fails (mask-conflict
semantics). Collection stamps the encoding schema into cache metadata; the trainer
cross-checks it both directions, with the legacy-absence asymmetry (no field = pre-v2.2
collector).
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pokezero.observation import (
    OBSERVATION_SCHEMA_VERSION_V2_1,
    OBSERVATION_SCHEMA_VERSION_V2_2,
)
from pokezero.showdown import (
    V2_1_REPLAY_OBSERVATION_SPEC,
    V2_2_REPLAY_OBSERVATION_SPEC,
    observation_schema_version_from_choice,
)


def _integration_root():
    from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT

    root = Path(os.environ.get("POKEZERO_SHOWDOWN_ROOT") or DEFAULT_SHOWDOWN_ROOT)
    if not (root / "dist" / "sim" / "index.js").exists():
        return None
    if shutil.which("node") is None:
        return None
    return root


class SchemaChoiceHelperTest(unittest.TestCase):
    def test_choices_map_to_full_versions_and_none_passes_through(self) -> None:
        self.assertIsNone(observation_schema_version_from_choice(None))
        self.assertEqual(
            observation_schema_version_from_choice("v2.1"), OBSERVATION_SCHEMA_VERSION_V2_1
        )
        self.assertEqual(
            observation_schema_version_from_choice("v2.2"), OBSERVATION_SCHEMA_VERSION_V2_2
        )
        with self.assertRaises(ValueError):
            observation_schema_version_from_choice("v2")  # legacy mode: checkpoint-driven only


class SchemaFlagLatchUnitTest(unittest.TestCase):
    def test_resume_with_conflicting_flag_hard_fails_both_directions(self) -> None:
        from pokezero.neural_cli import _require_schema_flag_agrees_with_checkpoint

        v2_1_checkpoint = SimpleNamespace(observation_schema_version=OBSERVATION_SCHEMA_VERSION_V2_1)
        v2_2_checkpoint = SimpleNamespace(observation_schema_version=OBSERVATION_SCHEMA_VERSION_V2_2)
        # No flag: checkpoint wins silently.
        _require_schema_flag_agrees_with_checkpoint(
            SimpleNamespace(observation_schema=None), v2_2_checkpoint
        )
        # Agreeing flag: no-op.
        _require_schema_flag_agrees_with_checkpoint(
            SimpleNamespace(observation_schema="v2.2"), v2_2_checkpoint
        )
        with self.assertRaisesRegex(ValueError, "cannot change across a resume"):
            _require_schema_flag_agrees_with_checkpoint(
                SimpleNamespace(observation_schema="v2.2"), v2_1_checkpoint
            )
        with self.assertRaisesRegex(ValueError, "cannot change across a resume"):
            _require_schema_flag_agrees_with_checkpoint(
                SimpleNamespace(observation_schema="v2.1"), v2_2_checkpoint
            )

    def test_cache_schema_cross_check_and_legacy_asymmetry(self) -> None:
        from pokezero.neural_cli import _require_cache_observation_schema_matches

        v2_1_model = SimpleNamespace(observation_schema_version=OBSERVATION_SCHEMA_VERSION_V2_1)
        v2_2_model = SimpleNamespace(observation_schema_version=OBSERVATION_SCHEMA_VERSION_V2_2)
        with tempfile.TemporaryDirectory() as tmp:
            legacy = Path(tmp) / "legacy-cache"
            legacy.mkdir()
            (legacy / "metadata.json").write_text(json.dumps({"record_count": 1}))
            v2_2_cache = Path(tmp) / "v2-2-cache"
            v2_2_cache.mkdir()
            (v2_2_cache / "metadata.json").write_text(
                json.dumps({"observation_schema": OBSERVATION_SCHEMA_VERSION_V2_2})
            )
            v2_1_cache = Path(tmp) / "v2-1-cache"
            v2_1_cache.mkdir()
            (v2_1_cache / "metadata.json").write_text(
                json.dumps({"observation_schema": OBSERVATION_SCHEMA_VERSION_V2_1})
            )

            # Matches pass.
            _require_cache_observation_schema_matches([v2_1_cache], v2_1_model)
            _require_cache_observation_schema_matches([v2_2_cache], v2_2_model)
            # Cross-schema fails BOTH directions.
            with self.assertRaisesRegex(ValueError, "cross-schema"):
                _require_cache_observation_schema_matches([v2_2_cache], v2_1_model)
            with self.assertRaisesRegex(ValueError, "cross-schema"):
                _require_cache_observation_schema_matches([v2_1_cache], v2_2_model)
            # Legacy asymmetry: absent field passes v2.1 (indistinguishable from today),
            # refuses v2.2 (turn-merged rows cannot come from a legacy collector).
            _require_cache_observation_schema_matches([legacy], v2_1_model)
            with self.assertRaisesRegex(ValueError, "legacy"):
                _require_cache_observation_schema_matches([legacy], v2_2_model)


@unittest.skipUnless(_integration_root() is not None, "requires built Showdown checkout and node")
class SchemaFlagEndToEndTest(unittest.TestCase):
    """Fresh collect + train at --observation-schema v2.2, end to end through the real
    CLIs and the real BattleStream env; the flag-less default path stamps v2.2/155
    (post-flip), and the explicit v2.1 flag still stamps v2.1/140."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = _integration_root()

    def _collect(self, out: Path, *extra: str) -> None:
        from pokezero.rollout_cli import main as rollout_main

        exit_code = rollout_main(
            [
                "collect-selfplay-training-cache",
                "--games", "1",
                "--out", str(out),
                "--seed-start", "23",
                "--showdown-root", str(self.root),
                *extra,
            ]
        )
        self.assertEqual(exit_code, 0)

    def test_collect_and_train_v2_2_end_to_end_and_cross_checks(self) -> None:
        from pokezero.neural_cli import main as neural_main
        from pokezero.neural_policy import load_transformer_model_config

        with tempfile.TemporaryDirectory() as tmp:
            v2_2_cache = Path(tmp) / "cache-v2-2"
            v2_1_cache = Path(tmp) / "cache-v2-1"
            default_cache = Path(tmp) / "cache-default"

            # Fresh v2.2 collect: schema + census recorded in cache metadata.
            self._collect(v2_2_cache, "--observation-schema", "v2.2")
            metadata = json.loads((v2_2_cache / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["observation_schema"], OBSERVATION_SCHEMA_VERSION_V2_2)
            self.assertEqual(
                metadata["observation_shapes"]["numeric_features"][-1],
                V2_2_REPLAY_OBSERVATION_SPEC.numeric_feature_count,
            )

            # Explicit v2.1 collect: v2.1 stamp, v2.1 census — the pre-flip schema stays
            # a first-class explicit selection.
            self._collect(v2_1_cache, "--observation-schema", "v2.1")
            v2_1_metadata = json.loads(
                (v2_1_cache / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                v2_1_metadata["observation_schema"], OBSERVATION_SCHEMA_VERSION_V2_1
            )
            self.assertEqual(
                v2_1_metadata["observation_shapes"]["numeric_features"][-1],
                V2_1_REPLAY_OBSERVATION_SPEC.numeric_feature_count,
            )

            # Default (flag-less) collect: stamps the CURRENT default — v2.2 since the
            # 2026-07-08 promotion (this assertion is deliberately about the default).
            self._collect(default_cache)
            default_metadata = json.loads(
                (default_cache / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                default_metadata["observation_schema"], OBSERVATION_SCHEMA_VERSION_V2_2
            )
            self.assertEqual(
                default_metadata["observation_shapes"]["numeric_features"][-1],
                V2_2_REPLAY_OBSERVATION_SPEC.numeric_feature_count,
            )

            # Cross-check hard-fails both directions through the real CLI (run BEFORE
            # the successful train, which consumes + garbage-collects its cache).
            import contextlib
            import io

            for data, schema_args, expectation in (
                (v2_1_cache, [], "cross-schema"),  # v2.2 default train on a v2.1 cache
                (v2_2_cache, ["--observation-schema", "v2.1"], "cross-schema"),
            ):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    exit_code = neural_main(
                        [
                            "train",
                            "--data", str(data),
                            "--out", str(Path(tmp) / "should-not-exist.pt"),
                            "--showdown-root", str(self.root),
                            *schema_args,
                            "--epochs", "1",
                            "--batch-size", "8",
                            "--embedding-dim", "16",
                            "--layers", "0",
                            "--attention-heads", "1",
                            "--feedforward-dim", "16",
                        ]
                    )
                self.assertEqual(exit_code, 1)
                self.assertRegex(stderr.getvalue(), expectation)

            # Fresh v2.2 train on the v2.2 cache: the checkpoint stamps the schema,
            # the widths, and the turn-merged vocabulary.
            checkpoint = Path(tmp) / "v2-2.pt"
            train_args = [
                "train",
                "--data", str(v2_2_cache),
                "--out", str(checkpoint),
                "--showdown-root", str(self.root),
                "--observation-schema", "v2.2",
                "--epochs", "1",
                "--batch-size", "8",
                "--embedding-dim", "16",
                "--layers", "0",
                "--attention-heads", "1",
                "--feedforward-dim", "16",
            ]
            self.assertEqual(neural_main(train_args), 0)
            stamped = load_transformer_model_config(checkpoint)
            self.assertEqual(stamped.observation_schema_version, OBSERVATION_SCHEMA_VERSION_V2_2)
            self.assertEqual(
                stamped.numeric_feature_count,
                V2_2_REPLAY_OBSERVATION_SPEC.numeric_feature_count,
            )
            self.assertEqual(
                stamped.categorical_feature_count,
                V2_2_REPLAY_OBSERVATION_SPEC.categorical_feature_count,
            )
            self.assertTrue(any(token.startswith("tt_phase:") for token in stamped.category_vocab))




if __name__ == "__main__":
    unittest.main()
