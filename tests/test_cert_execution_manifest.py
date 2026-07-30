"""Tests for file-backed certification execution manifest production."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import cert_execution_manifest as manifest  # noqa: E402
import cert_sweep_readout as readout  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExecutionManifestProducerTests(unittest.TestCase):
    def _source_commit(self) -> str:
        return subprocess.check_output(
            ("git", "-C", str(ROOT), "rev-parse", "HEAD"), text=True
        ).strip()

    def _write(self, path: Path, text: str) -> Path:
        path.write_text(text, encoding="utf-8")
        return path

    def _inputs(self, root: Path) -> dict[str, Path]:
        source = self._source_commit()
        fingerprint = "b" * 64
        image = "c" * 40
        report = self._write(
            root / "shard.json",
            json.dumps(
                {
                    "acceptance_eligible": True,
                    "boundaries_full_round": 1,
                    "boundaries_measured": 1,
                    "build_check": "gated",
                    "counters": {
                        "boundaries_full_round": 1,
                        "boundaries_measured": 1,
                        "engine_error": 0,
                        "transition:diverged": 0,
                        "transition:matched": 1,
                    },
                    "divergence_classes": {},
                    "engine_errors": 0,
                    "games": 1,
                    "repros": [],
                    "repro_retention": {
                        "keep_repro": 40,
                        "repros_retained": 0,
                        "transitions_diverged": 0,
                    },
                    "seeds": {"min": 1000, "max": 1000, "distinct": 1},
                    "transitions_diverged": 0,
                    "transitions_matched": 1,
                    "checkpoint_provenance": {
                        "complete": True,
                        "records_with_provenance": 1,
                    },
                }
            ),
        )
        checkpoint = self._write(
            root / "shard.jsonl",
            json.dumps(
                {
                    "schema": "engine-transition-differential/1",
                    "build_check": "gated",
                    "seed": 1000,
                    "counters": {
                        "boundaries_full_round": 1,
                        "boundaries_measured": 1,
                        "engine_error": 0,
                        "transition:diverged": 0,
                        "transition:matched": 1,
                    },
                    "repros": [],
                    "provenance": {
                        "source_commit": source,
                        "engine_fingerprint": fingerprint,
                        "image_commit": image,
                    },
                }
            ) + "\n",
        )
        engine_artifacts = {}
        for name in ("poke_engine", "pokezero_search"):
            module = self._write(root / f"{name}.so", name)
            engine_artifacts[name] = {
                "module_path": str(module),
                "module_sha256": _sha256(module),
            }
        stamp = self._write(
            root / "engine-stamp.json",
            json.dumps(
                {
                    "schema": "pokezero-engine-build/2",
                    "fingerprint": fingerprint,
                    "artifacts": engine_artifacts,
                }
            ),
        )
        behavior = self._write(root / "behavior.log", "".join(f"[p{n}] PASS\n" for n in range(9)))
        branch = self._write(root / "branch.log", "[search-crate-branch-events] PASS mapped events\n")
        return {
            "contract": self._write(root / "contract.json", "{}\n"),
            "report": report,
            "checkpoint": checkpoint,
            "marker": self._write(root / "shard.complete", "complete\n"),
            "stamp": stamp,
            "behavior": behavior,
            "branch": branch,
        }

    def _final_contract(self, paths: dict[str, Path]) -> None:
        source = self._source_commit()
        paths["contract"].write_text(
            json.dumps(
                {
                    "registered_before_launch": True,
                    "requires_execution_contract": True,
                    "launch_registration": {
                        "fresh_measurements_inspected_before_registration": 0,
                        "coordinator_go": True,
                        "engine_patch_count": 1,
                    },
                    "certification_gates": {
                        "expected_shards": 1,
                        "expected_games": 1,
                        "seed_blocks": [{"start": 1000, "games": 1}],
                        "minimum_coverage_measured_fraction": 0.97,
                        "required_build_check": "gated",
                        "required_matcher": "strict",
                        "required_repros_per_game": 40,
                        "required_keep_repro": 40,
                        "required_behavioral_probe_passes": 9,
                        "required_source_commit": source,
                        "required_image_commit": "c" * 40,
                        "required_engine_fingerprint": "b" * 64,
                        "required_readout_sha256": _sha256(ROOT / "scripts" / "cert_sweep_readout.py"),
                        "required_execution_manifest_producer_sha256": _sha256(
                            ROOT / "scripts" / "cert_execution_manifest.py"
                        ),
                    },
                    "pre_registered_family_rate_table": {
                        "documented_families": {},
                        "new_mechanisms_post_fix": {},
                    },
                    "predicted_class_rates_10k": {},
                }
            ),
            encoding="utf-8",
        )

    def test_producer_hashes_logs_and_freezes_content_addressed_blobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._inputs(root)
            output = root / "manifest.json"
            payload = manifest.produce_manifest(
                contract=paths["contract"],
                readout=ROOT / "scripts" / "cert_sweep_readout.py",
                output=output,
                reports=[paths["report"]],
                checkpoints=[paths["checkpoint"]],
                completion_markers=[paths["marker"]],
                behavioral_logs=[paths["behavior"]],
                branch_logs=[paths["branch"]],
                aggregate_behavioral_log=paths["behavior"],
                aggregate_branch_log=paths["branch"],
                engine_stamp=paths["stamp"],
            )
            persisted = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload, persisted)
            self.assertEqual(payload["schema"], "engine-cert-execution-manifest/2")
            self.assertEqual(manifest.validate_execution_manifest_schema(payload), [])
            self.assertEqual(
                payload["aggregate_provenance"]["behavioral_probes"]["sha256"],
                _sha256(paths["behavior"]),
            )
            self.assertEqual(payload["shards"][0]["checkpoint"]["records"], 1)
            self.assertEqual(payload["shards"][0]["image_commit"], "c" * 40)
            self.assertEqual(
                _sha256(Path(payload["contract_blob"]["path"])), _sha256(paths["contract"])
            )

    def test_producer_rejects_checkpoint_shorter_than_reported_game_population(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._inputs(root)
            report = json.loads(paths["report"].read_text(encoding="utf-8"))
            report["games"] = 2
            report["seeds"] = {"min": 1000, "max": 1001, "distinct": 2}
            report["checkpoint_provenance"]["records_with_provenance"] = 2
            paths["report"].write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checkpoint record population"):
                manifest.produce_manifest(
                    contract=paths["contract"],
                    readout=ROOT / "scripts" / "cert_sweep_readout.py",
                    output=root / "manifest.json",
                    reports=[paths["report"]],
                    checkpoints=[paths["checkpoint"]],
                    completion_markers=[paths["marker"]],
                    behavioral_logs=[paths["behavior"]],
                    branch_logs=[paths["branch"]],
                    aggregate_behavioral_log=paths["behavior"],
                    aggregate_branch_log=paths["branch"],
                    engine_stamp=paths["stamp"],
                )

    def test_producer_rejects_mixed_checkpoint_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._inputs(root)
            row = json.loads(paths["checkpoint"].read_text())
            row["provenance"]["engine_fingerprint"] = "d" * 64
            paths["checkpoint"].write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "engine fingerprint"):
                manifest.produce_manifest(
                    contract=paths["contract"],
                    readout=ROOT / "scripts" / "cert_sweep_readout.py",
                    output=root / "manifest.json",
                    reports=[paths["report"]],
                    checkpoints=[paths["checkpoint"]],
                    completion_markers=[paths["marker"]],
                    behavioral_logs=[paths["behavior"]],
                    branch_logs=[paths["branch"]],
                    aggregate_behavioral_log=paths["behavior"],
                    aggregate_branch_log=paths["branch"],
                    engine_stamp=paths["stamp"],
                )

    def test_producer_rejects_checkpoint_counter_and_repro_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._inputs(root)
            row = json.loads(paths["checkpoint"].read_text(encoding="utf-8"))
            row["counters"]["transition:diverged"] = 1
            row["repros"] = [{"seed": 1000, "step": 1, "reason": "fixture"}]
            paths["checkpoint"].write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not bind checkpoint content"):
                manifest.produce_manifest(
                    contract=paths["contract"],
                    readout=ROOT / "scripts" / "cert_sweep_readout.py",
                    output=root / "manifest.json",
                    reports=[paths["report"]],
                    checkpoints=[paths["checkpoint"]],
                    completion_markers=[paths["marker"]],
                    behavioral_logs=[paths["behavior"]],
                    branch_logs=[paths["branch"]],
                    aggregate_behavioral_log=paths["behavior"],
                    aggregate_branch_log=paths["branch"],
                    engine_stamp=paths["stamp"],
                )

    def test_final_contract_producer_uses_injected_public_commit_without_git(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._inputs(root)
            self._final_contract(paths)
            with patch.object(manifest, "public_repo_commit", return_value=self._source_commit()):
                payload = manifest.produce_manifest(
                    contract=paths["contract"],
                    readout=ROOT / "scripts" / "cert_sweep_readout.py",
                    output=root / "manifest.json",
                    reports=[paths["report"]],
                    checkpoints=[paths["checkpoint"]],
                    completion_markers=[paths["marker"]],
                    behavioral_logs=[paths["behavior"]],
                    branch_logs=[paths["branch"]],
                    aggregate_behavioral_log=paths["behavior"],
                    aggregate_branch_log=paths["branch"],
                    engine_stamp=paths["stamp"],
                    repo_root=Path("/no-git-image"),
                )
        self.assertEqual(payload["source"]["commit"], self._source_commit())

    def test_no_git_producer_fails_closed_without_injected_public_commit(self) -> None:
        with patch.object(manifest, "public_repo_commit", return_value=None):
            with self.assertRaisesRegex(ValueError, "cannot resolve public source commit"):
                manifest._source_checkout(Path("/no-git-image"))

    def test_final_contract_schema_rejects_missing_predicted_class_table(self) -> None:
        contract = {
            "registered_before_launch": True,
            "requires_execution_contract": True,
            "certification_gates": {},
            "pre_registered_family_rate_table": {},
        }
        errors = manifest.validate_final_contract_schema(contract)
        self.assertIn("final contract predicted_class_rates_10k is not an object", errors)

    def test_gates_bearing_contract_cannot_take_the_legacy_producer_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._inputs(root)
            self._final_contract(paths)
            contract = json.loads(paths["contract"].read_text(encoding="utf-8"))
            contract["registered_before_launch"] = False
            contract["requires_execution_contract"] = False
            paths["contract"].write_text(json.dumps(contract), encoding="utf-8")
            errors = manifest.validate_final_contract_schema(contract)
            self.assertIn("final contract registered_before_launch is not true", errors)
            self.assertIn("final contract requires_execution_contract is not true", errors)
            with self.assertRaisesRegex(ValueError, "registered_before_launch is not true"):
                manifest.produce_manifest(
                    contract=paths["contract"],
                    readout=ROOT / "scripts" / "cert_sweep_readout.py",
                    output=root / "manifest.json",
                    reports=[paths["report"]],
                    checkpoints=[paths["checkpoint"]],
                    completion_markers=[paths["marker"]],
                    behavioral_logs=[paths["behavior"]],
                    branch_logs=[paths["branch"]],
                    aggregate_behavioral_log=paths["behavior"],
                    aggregate_branch_log=paths["branch"],
                    engine_stamp=paths["stamp"],
                )

    def test_malformed_certification_gates_cannot_take_the_legacy_producer_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._inputs(root)
            contract = {"certification_gates": []}
            paths["contract"].write_text(json.dumps(contract), encoding="utf-8")
            errors = manifest.validate_final_contract_schema(contract)
            self.assertIn("final contract registered_before_launch is not true", errors)
            self.assertIn("final contract requires_execution_contract is not true", errors)
            self.assertIn("final contract has no certification_gates object", errors)
            with self.assertRaisesRegex(ValueError, "has no certification_gates object"):
                manifest.produce_manifest(
                    contract=paths["contract"],
                    readout=ROOT / "scripts" / "cert_sweep_readout.py",
                    output=root / "manifest.json",
                    reports=[paths["report"]],
                    checkpoints=[paths["checkpoint"]],
                    completion_markers=[paths["marker"]],
                    behavioral_logs=[paths["behavior"]],
                    branch_logs=[paths["branch"]],
                    aggregate_behavioral_log=paths["behavior"],
                    aggregate_branch_log=paths["branch"],
                    engine_stamp=paths["stamp"],
                )

    def test_final_contract_rejects_predicted_class_shapes_before_production(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._inputs(root)
            self._final_contract(paths)
            contract = json.loads(paths["contract"].read_text(encoding="utf-8"))
            contract["predicted_class_rates_10k"] = {
                "scalar": 42,
                "missing": {},
                "unordered": {
                    "expected_10k": 1.0,
                    "wilson95_count_10k": [3.0, 1.0],
                },
                "negative": {
                    "expected_10k": -1.0,
                    "wilson95_count_10k": [0.0, 1.0],
                },
            }
            paths["contract"].write_text(json.dumps(contract), encoding="utf-8")
            errors = manifest.validate_final_contract_schema(contract)
            self.assertTrue(any("predicted class 'scalar' is not an object" in error for error in errors))
            self.assertTrue(any("predicted class 'missing' expected_10k" in error for error in errors))
            self.assertTrue(any("predicted class 'unordered' wilson95_count_10k" in error for error in errors))
            self.assertTrue(any("predicted class 'negative' expected_10k" in error for error in errors))
            with self.assertRaisesRegex(ValueError, "predicted class 'scalar' is not an object"):
                manifest.produce_manifest(
                    contract=paths["contract"],
                    readout=ROOT / "scripts" / "cert_sweep_readout.py",
                    output=root / "manifest.json",
                    reports=[paths["report"]],
                    checkpoints=[paths["checkpoint"]],
                    completion_markers=[paths["marker"]],
                    behavioral_logs=[paths["behavior"]],
                    branch_logs=[paths["branch"]],
                    aggregate_behavioral_log=paths["behavior"],
                    aggregate_branch_log=paths["branch"],
                    engine_stamp=paths["stamp"],
                )

    def test_legacy_producer_rejects_present_malformed_predicted_class_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._inputs(root)
            paths["contract"].write_text(
                json.dumps({"predicted_class_rates_10k": {"scalar": 42}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "predicted class 'scalar' is not an object"):
                manifest.produce_manifest(
                    contract=paths["contract"],
                    readout=ROOT / "scripts" / "cert_sweep_readout.py",
                    output=root / "manifest.json",
                    reports=[paths["report"]],
                    checkpoints=[paths["checkpoint"]],
                    completion_markers=[paths["marker"]],
                    behavioral_logs=[paths["behavior"]],
                    branch_logs=[paths["branch"]],
                    aggregate_behavioral_log=paths["behavior"],
                    aggregate_branch_log=paths["branch"],
                    engine_stamp=paths["stamp"],
                )

    def test_final_contract_schema_vocabulary_tracks_shared_validator(self) -> None:
        schema = json.loads(
            (ROOT / "schemas" / "engine-cert-final-contract.schema.json").read_text(encoding="utf-8")
        )
        documented = schema["properties"]["pre_registered_family_rate_table"]["properties"][
            "documented_families"
        ]
        names = documented["propertyNames"]["anyOf"][0]["enum"]
        self.assertEqual(set(names), manifest.EMITTABLE_DOCUMENTED_FAMILIES)
        limit_names = documented["propertyNames"]["anyOf"][1]["enum"]
        self.assertEqual(set(limit_names), manifest.EMITTABLE_LIMIT_FAMILIES)
        prediction = schema["properties"]["predicted_class_rates_10k"]["additionalProperties"]
        self.assertEqual(prediction["required"], ["expected_10k", "wilson95_count_10k"])
        self.assertEqual(prediction["properties"]["expected_10k"]["minimum"], 0)
        self.assertEqual(
            prediction["properties"]["wilson95_count_10k"]["items"]["minimum"], 0
        )

    def test_final_contract_rejects_schema_vocabulary_and_rate_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._inputs(root)
            self._final_contract(paths)
            contract = json.loads(paths["contract"].read_text(encoding="utf-8"))
            contract["pre_registered_family_rate_table"]["documented_families"] = {
                "limit:typo": {"wilson95_rate": [0.0, 1.0]},
            }
            errors = manifest.validate_final_contract_schema(contract)
        self.assertTrue(
            any(
                "limit:typo" in error and "cannot be emitted" in error
                for error in errors
            )
        )

    def test_final_contract_reserves_coverage_upper_rate_for_substitute_limit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._inputs(root)
            self._final_contract(paths)
            contract = json.loads(paths["contract"].read_text(encoding="utf-8"))
            contract["pre_registered_family_rate_table"]["documented_families"] = {
                "I1_cap_state_shape": {
                    "upper_rate": 0.031,
                    "upper_rate_basis": "coverage_budget",
                }
            }
            errors = manifest.validate_final_contract_schema(contract)
        self.assertIn(
            "final contract documented family 'I1_cap_state_shape' cannot use a "
            "coverage-budget upper_rate",
            errors,
        )

    def test_final_contract_rejects_invalid_direct_rate_despite_count_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._inputs(root)
            self._final_contract(paths)
            contract = json.loads(paths["contract"].read_text(encoding="utf-8"))
            contract["pre_registered_family_rate_table"] = {
                "calibration_boundaries": 100,
                "documented_families": {
                    "I1_cap_state_shape": {
                        "wilson95_rate": [0.5, 2.0],
                        "wilson95": [1, 2],
                    }
                },
                "new_mechanisms_post_fix": {},
            }
            errors = manifest.validate_final_contract_schema(contract)
        self.assertIn(
            "final contract documented family 'I1_cap_state_shape' has an invalid "
            "wilson95_rate interval",
            errors,
        )

    def test_final_contract_rejects_boolean_zero_measurement_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._inputs(root)
            self._final_contract(paths)
            output = root / "manifest.json"
            contract = json.loads(paths["contract"].read_text(encoding="utf-8"))
            contract["launch_registration"][
                "fresh_measurements_inspected_before_registration"
            ] = False
            paths["contract"].write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "fresh_measurements_inspected_before_registration is not zero",
            ):
                manifest.produce_manifest(
                    contract=paths["contract"],
                    readout=ROOT / "scripts" / "cert_sweep_readout.py",
                    output=output,
                    reports=[paths["report"]],
                    checkpoints=[paths["checkpoint"]],
                    completion_markers=[paths["marker"]],
                    behavioral_logs=[paths["behavior"]],
                    branch_logs=[paths["branch"]],
                    aggregate_behavioral_log=paths["behavior"],
                    aggregate_branch_log=paths["branch"],
                    engine_stamp=paths["stamp"],
                )
            self.assertFalse(output.exists())

    def test_final_contract_schema_rejects_consumer_required_gate_before_production(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._inputs(root)
            self._final_contract(paths)
            contract = json.loads(paths["contract"].read_text(encoding="utf-8"))
            contract["certification_gates"].pop("required_matcher")
            paths["contract"].write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "required_matcher"):
                manifest.produce_manifest(
                    contract=paths["contract"],
                    readout=ROOT / "scripts" / "cert_sweep_readout.py",
                    output=root / "manifest.json",
                    reports=[paths["report"]],
                    checkpoints=[paths["checkpoint"]],
                    completion_markers=[paths["marker"]],
                    behavioral_logs=[paths["behavior"]],
                    branch_logs=[paths["branch"]],
                    aggregate_behavioral_log=paths["behavior"],
                    aggregate_branch_log=paths["branch"],
                    engine_stamp=paths["stamp"],
                )

    def test_final_contract_accepts_count_interval_with_calibration_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._inputs(root)
            self._final_contract(paths)
            contract = json.loads(paths["contract"].read_text(encoding="utf-8"))
            table = contract["pre_registered_family_rate_table"]
            table["calibration_boundaries"] = 100
            table["documented_families"] = {
                "I1_cap_state_shape": {
                    "wilson95": [1, 3],
                    "prediction_interval_rate": [0.0, 0.05],
                }
            }
            table["new_mechanisms_post_fix"] = {
                "recharge": {
                    "predicted_next": 0,
                    "classifier_outcome": "UNATTRIBUTED",
                    "exclusion_counter": "recharge_turn_residual_gap",
                }
            }
        self.assertEqual(manifest.validate_final_contract_schema(contract), [])

    def test_final_contract_rejects_family_rate_shapes_the_readout_cannot_honor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._inputs(root)
            self._final_contract(paths)
            contract = json.loads(paths["contract"].read_text(encoding="utf-8"))
            table = contract["pre_registered_family_rate_table"]
            table["documented_families"] = {
                "I1_cap_state_shape": {"wilson95": [1, 3]},
                "I2_matcher_accounting": {
                    "wilson95_rate": [0.01, 0.03],
                    "prediction_interval_rate": [0.8, 0.2],
                },
            }
            table["new_mechanisms_post_fix"] = {
                "bad-zero": {
                    "predicted_next": 1,
                    "classifier_outcome": "ATTRIBUTED",
                    "exclusion_counter": "not-a-classifier-counter",
                }
            }
            paths["contract"].write_text(json.dumps(contract), encoding="utf-8")
            errors = manifest.validate_final_contract_schema(contract)
            self.assertTrue(any("I1_cap_state_shape" in error and "wilson95" in error for error in errors))
            self.assertTrue(any("I2_matcher_accounting" in error and "prediction_interval_rate" in error for error in errors))
            self.assertTrue(any("bad-zero" in error and "not registered at zero" in error for error in errors))
            self.assertTrue(any("bad-zero" in error and "classifier_outcome" in error for error in errors))
            self.assertTrue(any("bad-zero" in error and "exclusion_counter" in error for error in errors))
            with self.assertRaisesRegex(ValueError, "final contract schema validation failed"):
                manifest.produce_manifest(
                    contract=paths["contract"],
                    readout=ROOT / "scripts" / "cert_sweep_readout.py",
                    output=root / "manifest.json",
                    reports=[paths["report"]],
                    checkpoints=[paths["checkpoint"]],
                    completion_markers=[paths["marker"]],
                    behavioral_logs=[paths["behavior"]],
                    branch_logs=[paths["branch"]],
                    aggregate_behavioral_log=paths["behavior"],
                    aggregate_branch_log=paths["branch"],
                    engine_stamp=paths["stamp"],
                )

    def test_checked_in_final_contract_matches_current_launch_vocabulary(self) -> None:
        contract = json.loads(
            (ROOT / "reports" / "c15_resweep_spec.json").read_text(encoding="utf-8")
        )
        calibration = json.loads(
            (ROOT / "reports" / "c24_final_classifier_c14_calibration.json").read_text(
                encoding="utf-8"
            )
        )
        c14 = json.loads(
            (ROOT / "reports" / "c14_cert_sweep_readout.json").read_text(
                encoding="utf-8"
            )
        )
        c15 = json.loads(
            (ROOT / "reports" / "c15_instrument_coverage.json").read_text(
                encoding="utf-8"
            )
        )
        c17 = json.loads(
            (ROOT / "reports" / "c17_substitute_retained_verification.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest.validate_final_contract_schema(contract), [])
        gates = contract["certification_gates"]
        self.assertEqual(
            gates["required_readout_sha256"],
            _sha256(ROOT / "scripts" / "cert_sweep_readout.py"),
        )
        self.assertEqual(
            gates["required_execution_manifest_producer_sha256"],
            _sha256(ROOT / "scripts" / "cert_execution_manifest.py"),
        )
        table = contract["pre_registered_family_rate_table"]
        self.assertEqual(
            set(table["documented_families"]),
            set(manifest.EMITTABLE_DOCUMENTED_FAMILIES)
            | set(manifest.EMITTABLE_LIMIT_FAMILIES),
        )
        self.assertEqual(
            {
                entry["exclusion_counter"]
                for entry in table["new_mechanisms_post_fix"].values()
            },
            set(manifest.EMITTABLE_EXCLUSION_COUNTERS),
        )
        self.assertEqual(
            table["calibration_boundaries"],
            c14["aggregate"]["boundaries_measured"],
        )
        self.assertEqual(
            calibration["source_evidence"]["boundaries_measured"],
            c14["aggregate"]["boundaries_measured"],
        )
        self.assertEqual(
            calibration["source_evidence"]["coverage_measured_fraction"],
            c14["coverage_measured_fraction"],
        )
        c14_raw_counts = {
            divergence_class: entry["observed"]
            for divergence_class, entry in c14[
                "per_class_observed_vs_predicted"
            ].items()
        }
        self.assertEqual(
            {
                divergence_class: entry["expected_10k"]
                for divergence_class, entry in contract[
                    "predicted_class_rates_10k"
                ].items()
            },
            c14_raw_counts,
        )
        self.assertEqual(calibration["raw_class_archive_counts"], c14_raw_counts)
        self.assertEqual(len(contract["predicted_class_rates_10k"]), 61)
        source_counts = {
            "limit:roll_divergent_lethality": c15["family_attribution"][
                "limit:roll_divergent_lethality"
            ],
            "I1_cap_state_shape": c15["family_attribution"]["I1_cap_state_shape"],
            "LS_capped_lethal_shape": c15["family_attribution"][
                "LS_capped_lethal_shape"
            ],
            "limit:world_sample_drag_target": c15["family_attribution"][
                "limit:world_sample_drag_target"
            ],
            "I3_roll_inherited": c15["family_attribution"]["I3_roll_inherited"],
            "I6_sleeptalk_callee_union": c15["family_attribution"][
                "I6_sleeptalk_callee_union"
            ],
            "I4_attribution_tie": c15["family_attribution"]["I4_attribution_tie"],
            "I5_boundary_truncation": c15["family_attribution"][
                "I5_boundary_truncation"
            ],
            "I2_matcher_accounting": c15["family_attribution"][
                "I2_matcher_accounting"
            ],
            "LS_crit_arm_pairing_echo": c15["family_attribution"][
                "LS_crit_arm_pairing_echo"
            ],
            "LS_confusion_fan": c15["family_attribution"]["LS_confusion_fan"],
            "LS_structural_arm_echo": c15["unattributed_named_shapes"][
                "structural component-count mismatch without a sibling engine"
            ],
        }
        self.assertEqual(
            calibration["registered_family_source_counts"], source_counts
        )
        calibration_boundaries = c14["aggregate"]["boundaries_measured"]
        expected_intervals = {}
        for family, count in source_counts.items():
            lower_rate, upper_rate = readout.wilson(count, calibration_boundaries)
            lower = round(lower_rate * calibration_boundaries)
            upper = round(upper_rate * calibration_boundaries)
            expected_intervals[family] = [
                lower
                if family
                in {
                    "limit:roll_divergent_lethality",
                    "limit:world_sample_drag_target",
                }
                else 0,
                upper,
            ]
        self.assertEqual(
            {
                family: entry["wilson95"]
                for family, entry in table["documented_families"].items()
                if "wilson95" in entry
            },
            expected_intervals,
        )
        self.assertEqual(
            calibration["registered_family_count_intervals"], expected_intervals
        )
        substitute = table["documented_families"][
            "limit:world_substitute_health_unknown"
        ]
        substitute_evidence = calibration["registered_non_empirical_upper_rates"][
            "limit:world_substitute_health_unknown"
        ]
        self.assertEqual(substitute["upper_rate"], substitute_evidence["upper_rate"])
        self.assertEqual(
            substitute["upper_rate_basis"], substitute_evidence["basis"]
        )
        self.assertEqual(c17["summary"]["identities"], 13)
        self.assertEqual(
            sum(row["at_limit"] - row["before_limit"] for row in c17["identities"]),
            c17["summary"]["identities"],
        )
        fingerprint = json.loads(
            subprocess.check_output(
                [sys.executable, "scripts/engine_build_fingerprint.py", "--print"],
                cwd=ROOT,
                text=True,
            )
        )
        self.assertEqual(
            gates["required_engine_fingerprint"], fingerprint["fingerprint"]
        )
        self.assertEqual(
            fingerprint["count"],
            contract["launch_registration"]["engine_patch_count"],
        )


if __name__ == "__main__":
    unittest.main()
