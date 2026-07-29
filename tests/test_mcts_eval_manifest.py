"""Manifest identities + checkpoint contract resolution (plan deliverables 1 and 5).

The identity split is the load-bearing property: reducing concurrency must NOT
invalidate checkpoint materialization, validation, smoke, or corpus artifacts
(they are keyed by experiment_id), while every execution-scoped stage must see a
new identity (execution_id).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

# `scripts/` is not a package, so the exporter is imported by module name. Resolve
# it from THIS FILE rather than the cwd: a relative sys.path entry only works when
# the runner happens to start at the repo root, and when it does not the affected
# tests raise ModuleNotFoundError, which reads as an environment problem rather
# than as coverage that silently stopped running.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _add_scripts_to_path() -> None:
    scripts = os.path.join(ROOT, "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)


# Done at import time, not inside a helper: the tests below import the exporter at
# the top of the test body, before any helper of theirs has run.
_add_scripts_to_path()

from pokezero.mcts_eval import ContractError, export_reuse_key, sha256_file  # noqa: E402
from pokezero.mcts_eval.manifest import (  # noqa: E402
    DEFAULT_BATCH,
    MatrixManifest,
    ResourceProfile,
    SearchConfig,
    default_lattice,
)
from pokezero.mcts_eval.resolver import (  # noqa: E402
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
        category_vocab=("alpha", "beta", "gamma"),
        architecture={"embedding_dim": 512, "transformer_layers": 3},
        feature_masks={
            "transition_token_budget": 64,
            "exact_state": True,
            "opponent_tendency_stats_block": True,
            "tier2_residuals": True,
            "tier2_investment": True,
        },
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
    def _tables(self, path: Path, *, masks: dict | None = None, **layout_overrides) -> Path:
        layout = {
            "schema_version": "pokezero.observation.v3",
            "token_count": 87,
            "categorical_feature_count": 51,
            "numeric_feature_count": 169,
            "default_feature_masks": {
                "exact_state": True,
                "stats_block": True,
                "tier2_residuals": True,
                "tier2_investment": True,
                "transition_token_budget": 64,
            },
        }
        vocab = {"tokens": list(_contract().category_vocab)}
        if masks is not None:
            layout["default_feature_masks"] = {**layout["default_feature_masks"], **masks}
        layout.update(layout_overrides)
        path.write_text(
            json.dumps(
                {"schema_version": TABLES_SCHEMA_VERSION, "vocab": vocab, "layout": layout}
            ),
            encoding="utf-8",
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

    def test_feature_mask_disagreement_is_terminal(self) -> None:
        # Regression: the shipped exporter emitted `tier2_investment: False` (the
        # dataclass default) for every checkpoint, while every trained checkpoint
        # carries True. Shape agreed, so this passed the width/schema checks and
        # ran — the crate skipped the investment slots at every leaf while the
        # Python root encode wrote them.
        with tempfile.TemporaryDirectory() as temp_dir:
            tables = self._tables(
                Path(temp_dir) / "tables.json", masks={"tier2_investment": False}
            )
            with self.assertRaisesRegex(ContractError, "tier2_investment"):
                validate_encoder_tables(_contract(), tables)

    def test_history_budget_disagreement_is_terminal(self) -> None:
        # Regression: a budget-0 (Markov) checkpoint was handed tables claiming the
        # full 64-token history region, so the crate filled 64 synthesized history
        # tokens the model was trained to never attend to.
        with tempfile.TemporaryDirectory() as temp_dir:
            tables = self._tables(Path(temp_dir) / "tables.json")  # budget 64
            contract = _contract(
                feature_masks={
                    "transition_token_budget": 0,
                    "exact_state": True,
                    "opponent_tendency_stats_block": True,
                    "tier2_residuals": True,
                    "tier2_investment": True,
                }
            )
            with self.assertRaisesRegex(ContractError, "transition_token_budget"):
                validate_encoder_tables(contract, tables)

    def test_zero_budget_tables_accepted_for_zero_budget_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tables = self._tables(
                Path(temp_dir) / "tables.json", masks={"transition_token_budget": 0}
            )
            contract = _contract(
                feature_masks={
                    "transition_token_budget": 0,
                    "exact_state": True,
                    "opponent_tendency_stats_block": True,
                    "tier2_residuals": True,
                    "tier2_investment": True,
                }
            )
            validate_encoder_tables(contract, tables)  # no raise

    def test_missing_mask_block_is_terminal(self) -> None:
        # Tables predating the masks block cannot be shown to agree, so they must
        # not be adopted silently.
        with tempfile.TemporaryDirectory() as temp_dir:
            tables = self._tables(Path(temp_dir) / "tables.json", default_feature_masks={})
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


class EngineMctsPolicyModeTest(unittest.TestCase):
    """policy_mode='engine-mcts' is how a lattice cell becomes a STRENGTH row:
    the same frozen search configuration that produced a timing row must drive
    the FoulPlay games, or the two axes of the frontier describe different
    searches."""

    def _config(self, **overrides):
        from pathlib import Path as P

        from pokezero.foulplay_bridge import ControlledFoulPlayConfig

        values = dict(
            checkpoint=P("/shared/ckpt.pt"),
            showdown_root=P("/opt/pokemon-showdown"),
            policy_mode="engine-mcts",
            engine_model_path=P("/artifacts/model_ts.pt"),
            engine_tables_path=P("/artifacts/encoder_tables.json"),
        )
        values.update(overrides)
        return ControlledFoulPlayConfig(**values)

    def test_engine_mcts_requires_exported_artifacts(self) -> None:
        with self.assertRaisesRegex(ValueError, "engine_model_path"):
            self._config(engine_model_path=None)
        with self.assertRaisesRegex(ValueError, "engine_tables_path"):
            self._config(engine_tables_path=None)

    def test_unknown_policy_mode_names_engine_mcts(self) -> None:
        with self.assertRaisesRegex(ValueError, "engine-mcts"):
            self._config(policy_mode="bogus")

    def test_search_axes_round_trip(self) -> None:
        config = self._config(engine_depth=8, engine_sims=8192, engine_batch=256, engine_worlds=4)
        self.assertEqual(
            (config.engine_depth, config.engine_sims, config.engine_batch, config.engine_worlds),
            (8, 8192, 256, 4),
        )

    def test_axes_match_the_lattice_config_id(self) -> None:
        # A strength row must be joinable to its timing row on config_id.
        cell = SearchConfig(depth=8, sims=8192, batch=256, worlds=4)
        config = self._config(
            engine_depth=cell.depth, engine_sims=cell.sims,
            engine_batch=cell.batch, engine_worlds=cell.worlds,
        )
        self.assertEqual(
            f"d{config.engine_depth}-s{config.engine_sims}-b{config.engine_batch}"
            f"-w{config.engine_worlds}-local",
            cell.config_id,
        )


class MaterializationGateTest(unittest.TestCase):
    """The FoulPlay bridge must decide by CAPABILITY, not by policy class.

    Gating on isinstance(RootPUCTSearchPolicy) handed EngineMctsPolicy a None
    materialization state, so engine search fell back to uniform-legal on every
    decision — 0/20 against the raw policy's 10/20, with no error raised. A
    silent capability mismatch is the most expensive kind of bug in a strength
    study because it reads as a scientific result.
    """

    def test_engine_policy_declares_the_requirement(self) -> None:
        from pokezero.engine_search import EngineMctsPolicy

        self.assertTrue(
            getattr(EngineMctsPolicy, "requires_public_materialization_state", False),
            "EngineMctsPolicy must declare it needs a materialized public state",
        )

    def test_bridge_gates_on_capability_not_class(self) -> None:
        import inspect

        from pokezero import foulplay_bridge

        source = inspect.getsource(foulplay_bridge)
        gate = source[source.index("public_materialization_state = ("):][:1200]
        self.assertIn("requires_public_materialization_state", gate)


class TrimmedEncoderTablesTest(unittest.TestCase):
    """Region-trimmed checkpoints must get tables matching THEIR width.

    The exporter derived its layout from the schema default (87 tokens), so a
    trimmed 39-token checkpoint produced tables the model could not consume and
    the root/leaf contract check refused the run — trimmed models could not go
    through the crate at all.
    """

    def _specs(self):
        import dataclasses

        _add_scripts_to_path()
        from pokezero.showdown import OBSERVATION_SCHEMA_VERSION_V3, observation_spec_for_schema

        full = observation_spec_for_schema(OBSERVATION_SCHEMA_VERSION_V3)
        return full, dataclasses.replace(full, transition_token_count=16)

    def test_layout_follows_the_trimmed_spec(self) -> None:
        import export_encoder_tables as exporter

        from pokezero.showdown import OBSERVATION_SCHEMA_VERSION_V3

        full, trimmed = self._specs()
        default_layout = exporter._layout_payload(OBSERVATION_SCHEMA_VERSION_V3)
        trimmed_layout = exporter._layout_payload(OBSERVATION_SCHEMA_VERSION_V3, spec=trimmed)
        self.assertEqual(default_layout["token_count"], full.token_count)
        self.assertEqual(trimmed_layout["token_count"], trimmed.token_count)
        self.assertLess(trimmed_layout["token_count"], default_layout["token_count"])

    def test_offsets_before_the_transition_tail_are_unchanged(self) -> None:
        # The transition region is the LAST block, so trimming it must not move
        # any earlier token offset — that is what keeps the tables valid.
        import export_encoder_tables as exporter

        from pokezero.showdown import OBSERVATION_SCHEMA_VERSION_V3

        _, trimmed = self._specs()
        default_layout = exporter._layout_payload(OBSERVATION_SCHEMA_VERSION_V3)
        trimmed_layout = exporter._layout_payload(OBSERVATION_SCHEMA_VERSION_V3, spec=trimmed)
        self.assertEqual(default_layout["token_offsets"], trimmed_layout["token_offsets"])

    def test_trimmed_tables_satisfy_the_contract_guard(self) -> None:
        # End to end: tables built from a trimmed spec must pass the same
        # root/leaf validation that rejected the schema-default ones.
        import json
        import tempfile

        import export_encoder_tables as exporter

        from pokezero.showdown import OBSERVATION_SCHEMA_VERSION_V3

        from pokezero.observation import ObservationFeatureMasks

        _, trimmed = self._specs()
        # The masks a real checkpoint carries, not the dataclass defaults — passing
        # only the spec is exactly the hole that shipped wrong tables to every run.
        masks = ObservationFeatureMasks(
            transition_token_budget=trimmed.transition_token_count,
            tier2_investment=True,
        )
        layout = exporter._layout_payload(
            OBSERVATION_SCHEMA_VERSION_V3, spec=trimmed, masks=masks
        )
        vocab = {"tokens": list(_contract().category_vocab)}
        contract = _contract(
            token_count=trimmed.token_count,
            transition_token_count=trimmed.transition_token_count,
            numeric_feature_count=layout["numeric_feature_count"],
            categorical_feature_count=layout["categorical_feature_count"],
            feature_masks={
                "transition_token_budget": trimmed.transition_token_count,
                "exact_state": masks.exact_state,
                "opponent_tendency_stats_block": masks.opponent_tendency_stats_block,
                "tier2_residuals": masks.tier2_residuals,
                "tier2_investment": masks.tier2_investment,
            },
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tables.json"
            path.write_text(
                json.dumps(
                    {"schema_version": TABLES_SCHEMA_VERSION, "vocab": vocab, "layout": layout}
                )
            )
            validate_encoder_tables(contract, path)  # must not raise

    def test_schema_default_masks_are_rejected_for_a_real_checkpoint(self) -> None:
        # The shipped path, pinned as a failure: build tables WITHOUT the
        # checkpoint's masks and the guard must now refuse them.
        import json
        import tempfile

        import export_encoder_tables as exporter

        from pokezero.showdown import OBSERVATION_SCHEMA_VERSION_V3

        layout = exporter._layout_payload(OBSERVATION_SCHEMA_VERSION_V3)
        contract = _contract(
            numeric_feature_count=layout["numeric_feature_count"],
            categorical_feature_count=layout["categorical_feature_count"],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tables.json"
            path.write_text(json.dumps({"schema_version": TABLES_SCHEMA_VERSION, "layout": layout}))
            with self.assertRaisesRegex(ContractError, "tier2_investment"):
                validate_encoder_tables(contract, path)


class EncoderTableVocabPolarityTest(unittest.TestCase):
    """The vocabulary must be checked against the CHECKPOINT, never against the build.

    The enumeration is a positional list and the model's embedding rows were learned
    against the positions in force at training time, so the checkpoint is the only
    authority for what a categorical id means. A build-anchored check does not merely
    miss things: it inverts, rejecting the artifact that matches the model and
    accepting one that silently shifts rows.
    """

    # 2026-07-29, both 5M checkpoints: trained on 1216 tokens while the build had
    # grown to 1217 by inserting `volatile:solarbeam` at index 1204, renumbering the
    # 13 volatiles after it. Scaled down, same shape.
    TRAINED = ("aaa", "volatile:stockpile", "volatile:substitute")
    BUILD = ("aaa", "volatile:solarbeam", "volatile:stockpile", "volatile:substitute")

    def _tables(self, path: Path, tokens) -> Path:
        path.write_text(
            json.dumps(
                {
                    "schema_version": TABLES_SCHEMA_VERSION,
                    "vocab": {"tokens": list(tokens)},
                    "layout": {
                        "schema_version": "pokezero.observation.v3",
                        "token_count": 87,
                        "categorical_feature_count": 51,
                        "numeric_feature_count": 169,
                        "default_feature_masks": {
                            "exact_state": True,
                            "stats_block": True,
                            "tier2_residuals": True,
                            "tier2_investment": True,
                            "transition_token_budget": 64,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        return path

    def _contract_trained(self):
        return _contract(category_vocab=self.TRAINED)

    def test_tables_matching_the_trained_vocab_are_accepted(self) -> None:
        # The cached k64 tables. A build-anchored check REJECTED these -- they were
        # correct, and regenerating them was what introduced the defect.
        with tempfile.TemporaryDirectory() as temp_dir:
            tables = self._tables(Path(temp_dir) / "t.json", self.TRAINED)
            validate_encoder_tables(self._contract_trained(), tables)  # no raise

    def test_fresh_build_export_against_an_older_checkpoint_is_terminal(self) -> None:
        # The tables this agent regenerated for k0. A build-anchored check ACCEPTED
        # these; they shift every volatile row the model learned after the insert.
        with tempfile.TemporaryDirectory() as temp_dir:
            tables = self._tables(Path(temp_dir) / "t.json", self.BUILD)
            with self.assertRaisesRegex(ContractError, "vocab.tokens"):
                validate_encoder_tables(self._contract_trained(), tables)

    def test_the_error_names_the_inserted_token_and_where_it_shifts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tables = self._tables(Path(temp_dir) / "t.json", self.BUILD)
            with self.assertRaises(ContractError) as caught:
                validate_encoder_tables(self._contract_trained(), tables)
            message = str(caught.exception)
            self.assertIn("index 1", message)  # first divergence
            self.assertIn("volatile:solarbeam", message)  # not in the trained vocabulary

    def test_vocab_is_in_the_reuse_key(self) -> None:
        # So tables exported against another enumeration resolve to a different path
        # and can never be adopted for this checkpoint, whatever wrote them.
        self.assertNotEqual(
            export_reuse_key(self._contract_trained()),
            export_reuse_key(_contract(category_vocab=self.BUILD)),
        )


class TrimmedEncoderTablesTest(unittest.TestCase):
    """Region-trimmed checkpoints must get tables matching THEIR width.

    The exporter derived its layout from the schema default (87 tokens), so a
    trimmed 39-token checkpoint produced tables the model could not consume and
    the root/leaf contract check refused the run — trimmed models could not go
    through the crate at all.
    """

    def _specs(self):
        import dataclasses

        _add_scripts_to_path()
        from pokezero.showdown import OBSERVATION_SCHEMA_VERSION_V3, observation_spec_for_schema

        full = observation_spec_for_schema(OBSERVATION_SCHEMA_VERSION_V3)
        return full, dataclasses.replace(full, transition_token_count=16)

    def test_layout_follows_the_trimmed_spec(self) -> None:
        import export_encoder_tables as exporter

        from pokezero.showdown import OBSERVATION_SCHEMA_VERSION_V3

        full, trimmed = self._specs()
        default_layout = exporter._layout_payload(OBSERVATION_SCHEMA_VERSION_V3)
        trimmed_layout = exporter._layout_payload(OBSERVATION_SCHEMA_VERSION_V3, spec=trimmed)
        self.assertEqual(default_layout["token_count"], full.token_count)
        self.assertEqual(trimmed_layout["token_count"], trimmed.token_count)
        self.assertLess(trimmed_layout["token_count"], default_layout["token_count"])

    def test_offsets_before_the_transition_tail_are_unchanged(self) -> None:
        # The transition region is the LAST block, so trimming it must not move
        # any earlier token offset — that is what keeps the tables valid.
        import export_encoder_tables as exporter

        from pokezero.showdown import OBSERVATION_SCHEMA_VERSION_V3

        _, trimmed = self._specs()
        default_layout = exporter._layout_payload(OBSERVATION_SCHEMA_VERSION_V3)
        trimmed_layout = exporter._layout_payload(OBSERVATION_SCHEMA_VERSION_V3, spec=trimmed)
        self.assertEqual(default_layout["token_offsets"], trimmed_layout["token_offsets"])

    def test_trimmed_tables_satisfy_the_contract_guard(self) -> None:
        # End to end: tables built from a trimmed spec must pass the same
        # root/leaf validation that rejected the schema-default ones.
        import json
        import tempfile

        import export_encoder_tables as exporter

        from pokezero.showdown import OBSERVATION_SCHEMA_VERSION_V3

        from pokezero.observation import ObservationFeatureMasks

        _, trimmed = self._specs()
        # The masks a real checkpoint carries, not the dataclass defaults — passing
        # only the spec is exactly the hole that shipped wrong tables to every run.
        masks = ObservationFeatureMasks(
            transition_token_budget=trimmed.transition_token_count,
            tier2_investment=True,
        )
        layout = exporter._layout_payload(
            OBSERVATION_SCHEMA_VERSION_V3, spec=trimmed, masks=masks
        )
        vocab = {"tokens": list(_contract().category_vocab)}
        contract = _contract(
            token_count=trimmed.token_count,
            transition_token_count=trimmed.transition_token_count,
            numeric_feature_count=layout["numeric_feature_count"],
            categorical_feature_count=layout["categorical_feature_count"],
            feature_masks={
                "transition_token_budget": trimmed.transition_token_count,
                "exact_state": masks.exact_state,
                "opponent_tendency_stats_block": masks.opponent_tendency_stats_block,
                "tier2_residuals": masks.tier2_residuals,
                "tier2_investment": masks.tier2_investment,
            },
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tables.json"
            path.write_text(
                json.dumps(
                    {"schema_version": TABLES_SCHEMA_VERSION, "vocab": vocab, "layout": layout}
                )
            )
            validate_encoder_tables(contract, path)  # must not raise

    def test_schema_default_masks_are_rejected_for_a_real_checkpoint(self) -> None:
        # The shipped path, pinned as a failure: build tables WITHOUT the
        # checkpoint's masks and the guard must now refuse them.
        import json
        import tempfile

        import export_encoder_tables as exporter

        from pokezero.showdown import OBSERVATION_SCHEMA_VERSION_V3

        layout = exporter._layout_payload(OBSERVATION_SCHEMA_VERSION_V3)
        contract = _contract(
            numeric_feature_count=layout["numeric_feature_count"],
            categorical_feature_count=layout["categorical_feature_count"],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tables.json"
            path.write_text(json.dumps({"schema_version": TABLES_SCHEMA_VERSION, "layout": layout}))
            with self.assertRaisesRegex(ContractError, "tier2_investment"):
                validate_encoder_tables(contract, path)
