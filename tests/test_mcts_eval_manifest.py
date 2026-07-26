"""Manifest identities + checkpoint contract resolution (plan deliverables 1 and 5).

The identity split is the load-bearing property: reducing concurrency must NOT
invalidate checkpoint materialization, validation, smoke, or corpus artifacts
(they are keyed by experiment_id), while every execution-scoped stage must see a
new identity (execution_id).
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from pokezero.mcts_eval import ContractError, export_reuse_key, sha256_file
from pokezero.mcts_eval.manifest import (
    DEFAULT_BATCH,
    MatrixManifest,
    ResourceProfile,
    SearchConfig,
    default_lattice,
)
from pokezero.mcts_eval.resolver import (
    TABLES_SCHEMA_VERSION,
    CheckpointContract,
    validate_encoder_tables,
)


def _contract(**overrides) -> CheckpointContract:
    values = dict(
        checkpoint_path="/shared/ckpt.pt",
        checkpoint_sha256="a" * 64,
        policy_id="foundation-midscale-iter-1438",
        schema_version="pokezero.observation.v3",
        token_count=87,
        categorical_feature_count=51,
        numeric_feature_count=169,
        transition_token_count=64,
        architecture={"embedding_dim": 512, "transformer_layers": 3},
        feature_masks={"transition_token_budget": 64},
        model_device="cpu",
        showdown_source_sha256="b" * 64,
    )
    values.update(overrides)
    return CheckpointContract(**values)


def _manifest(**overrides) -> MatrixManifest:
    values = dict(
        checkpoint_manifest=_contract().to_manifest(),
        configs=default_lattice(depths=(2, 4), sims=(512, 1024)),
    )
    values.update(overrides)
    return MatrixManifest(**values)


class SearchConfigTest(unittest.TestCase):
    def test_config_id_is_stable_and_descriptive(self) -> None:
        config = SearchConfig(depth=8, sims=4096)
        self.assertEqual(config.config_id, "d8-s4096-b16-w4-local")

    def test_inference_mode_is_part_of_identity(self) -> None:
        local = SearchConfig(depth=8, sims=4096)
        served = SearchConfig(depth=8, sims=4096, inference_mode="served")
        self.assertNotEqual(local.config_id, served.config_id)

    def test_batch_above_sims_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch"):
            SearchConfig(depth=2, sims=8, batch=16)

    def test_lattice_is_deterministic_and_complete(self) -> None:
        lattice = default_lattice()
        self.assertEqual(len(lattice), 25)  # plan A3: 5 depths x 5 sims
        self.assertEqual(lattice, default_lattice())
        self.assertTrue(all(c.batch == DEFAULT_BATCH and c.worlds == 4 for c in lattice))


class ManifestIdentityTest(unittest.TestCase):
    def test_resource_profile_changes_execution_not_experiment(self) -> None:
        base = _manifest()
        scaled = base.with_resource_profile(ResourceProfile(concurrency=8))
        self.assertEqual(base.experiment_id, scaled.experiment_id)
        self.assertNotEqual(base.execution_id, scaled.execution_id)

    def test_matrix_change_changes_experiment(self) -> None:
        base = _manifest()
        wider = _manifest(configs=default_lattice(depths=(2, 4, 6), sims=(512, 1024)))
        self.assertNotEqual(base.experiment_id, wider.experiment_id)

    def test_checkpoint_change_changes_experiment(self) -> None:
        base = _manifest()
        other = _manifest(checkpoint_manifest=_contract(checkpoint_sha256="c" * 64).to_manifest())
        self.assertNotEqual(base.experiment_id, other.experiment_id)

    def test_identities_are_reproducible_across_instances(self) -> None:
        self.assertEqual(_manifest().experiment_id, _manifest().experiment_id)
        self.assertEqual(_manifest().execution_id, _manifest().execution_id)

    def test_duplicate_config_rejected(self) -> None:
        duplicated = (SearchConfig(depth=2, sims=512), SearchConfig(depth=2, sims=512))
        with self.assertRaisesRegex(ValueError, "duplicate config_id"):
            _manifest(configs=duplicated)

    def test_payload_round_trips_as_json(self) -> None:
        payload = _manifest().to_payload()
        self.assertEqual(json.loads(json.dumps(payload))["experiment_id"], payload["experiment_id"])


class ExportReuseKeyTest(unittest.TestCase):
    def test_key_changes_with_every_meaning_bearing_input(self) -> None:
        base = export_reuse_key(_contract())
        self.assertNotEqual(base, export_reuse_key(_contract(checkpoint_sha256="c" * 64)))
        self.assertNotEqual(base, export_reuse_key(_contract(model_device="cuda")))
        self.assertNotEqual(base, export_reuse_key(_contract(showdown_source_sha256="c" * 64)))
        self.assertNotEqual(base, export_reuse_key(_contract(transition_token_count=16)))
        self.assertNotEqual(base, export_reuse_key(_contract(exporter_revision="other")))

    def test_key_is_stable_for_identical_contracts(self) -> None:
        self.assertEqual(export_reuse_key(_contract()), export_reuse_key(_contract()))


class EncoderTableValidationTest(unittest.TestCase):
    def _tables(self, path: Path, **layout_overrides) -> Path:
        layout = {
            "schema_version": "pokezero.observation.v3",
            "token_count": 87,
            "categorical_feature_count": 51,
            "numeric_feature_count": 169,
        }
        layout.update(layout_overrides)
        path.write_text(
            json.dumps({"schema_version": TABLES_SCHEMA_VERSION, "layout": layout}), encoding="utf-8"
        )
        return path

    def test_matching_tables_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tables = self._tables(Path(temp_dir) / "tables.json")
            validate_encoder_tables(_contract(), tables)  # no raise

    def test_root_leaf_width_disagreement_is_terminal(self) -> None:
        # The trimmed-region failure mode: leaf tables describe a different width
        # than the checkpoint's root encode.
        with tempfile.TemporaryDirectory() as temp_dir:
            tables = self._tables(Path(temp_dir) / "tables.json", token_count=39)
            with self.assertRaisesRegex(ContractError, "root/leaf observation contract"):
                validate_encoder_tables(_contract(), tables)

    def test_schema_disagreement_is_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tables = self._tables(
                Path(temp_dir) / "tables.json", schema_version="pokezero.observation.v2.2"
            )
            with self.assertRaisesRegex(ContractError, "root/leaf observation contract"):
                validate_encoder_tables(_contract(), tables)

    def test_unknown_artifact_schema_is_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tables.json"
            path.write_text(json.dumps({"schema_version": "nope", "layout": {}}), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "unknown artifact schema"):
                validate_encoder_tables(_contract(), path)


class Sha256FileTest(unittest.TestCase):
    def test_hashes_file_contents(self) -> None:
        import hashlib

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "blob.bin"
            path.write_bytes(b"pokezero" * 1000)
            self.assertEqual(sha256_file(path), hashlib.sha256(b"pokezero" * 1000).hexdigest())



class EngineMctsPolicySpecTest(unittest.TestCase):
    """A lattice cell's identity IS its configuration, so the spec must never
    fill in a default for a field that would mislabel a timing/strength row."""

    def _factory(self, spec: str):
        from pokezero.collection import policy_factory_from_spec

        return policy_factory_from_spec(spec)

    def test_required_options_enforced(self) -> None:
        for spec, missing in (
            ("engine-mcts:/c.pt", "depth"),
            ("engine-mcts:/c.pt?depth=8", "sims"),
            ("engine-mcts:/c.pt?depth=8&sims=4096", "model"),
            ("engine-mcts:/c.pt?depth=8&sims=4096&model=/m.pt", "tables"),
        ):
            with self.assertRaisesRegex(ValueError, missing):
                self._factory(spec)

    def test_missing_checkpoint_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "checkpoint path"):
            self._factory("engine-mcts:?depth=8&sims=4096&model=/m.pt&tables=/t.json")

    def test_unknown_option_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown engine-mcts"):
            self._factory(
                "engine-mcts:/c.pt?depth=8&sims=4096&model=/m.pt&tables=/t.json&earlystop=1"
            )

    def test_full_spec_builds_lazily(self) -> None:
        # Construction must not require the artifacts to exist (the controller
        # materializes them); only invocation does.
        factory = self._factory(
            "engine-mcts:/c.pt?depth=8&sims=4096&batch=16&worlds=4&model=/m.pt&tables=/t.json"
        )
        self.assertTrue(callable(factory))

if __name__ == "__main__":
    unittest.main()
