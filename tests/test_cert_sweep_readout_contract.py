"""Fail-closed pins for the certification execution/readout contract."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT / "scripts"))

import cert_sweep_readout as readout  # noqa: E402


class CertificationContractTests(unittest.TestCase):
    image_commit = "c" * 40
    engine_fingerprint = "b" * 64

    def setUp(self) -> None:
        self.source_commit = subprocess.check_output(
            ("git", "-C", os.fspath(ROOT), "rev-parse", "HEAD"), text=True
        ).strip()

    def _file(self, path: Path, text: str | None = None) -> dict[str, str]:
        if text is not None:
            path.write_text(text, encoding="utf-8")
        return {"path": os.fspath(path), "sha256": readout._sha256(path)}

    def _probe_log(self, path: Path, *, branch: bool = False) -> dict:
        if branch:
            evidence = self._file(path, "[search-crate-branch-events] PASS mapped events\n")
            evidence["passed"] = True
            return evidence
        evidence = self._file(path, "".join(f"[probe-{index}] PASS\n" for index in range(9)))
        evidence.update({"passed": 9, "total": 9})
        return evidence

    def _stamp(self, root: Path) -> dict:
        artifacts = {}
        for name in ("poke_engine", "pokezero_search"):
            module = root / f"{name}.so"
            artifacts[name] = {
                "module_path": os.fspath(module),
                "module_sha256": readout._sha256(Path(self._file(module, name)["path"])),
            }
        stamp = root / "engine-stamp.json"
        stamp.write_text(
            json.dumps(
                {
                    "schema": "pokezero-engine-build/2",
                    "fingerprint": self.engine_fingerprint,
                    "artifacts": artifacts,
                }
            ),
            encoding="utf-8",
        )
        return self._file(stamp)

    def _shard(self, path: Path, seed: int, *, measured: int = 100) -> None:
        payload = {
            "acceptance_eligible": True,
            "boundaries_full_round": 100,
            "boundaries_measured": measured,
            "build_check": "gated",
            "counters": {
                "boundaries_full_round": 100,
                "boundaries_measured": measured,
                "skip:world_unsupported:test_fixture": 2,
            },
            "divergence_classes": {},
            "engine_errors": 0,
            "games": 1,
            "matcher": "strict",
            "repro_retention": {
                "repros_per_game": 40,
                "keep_repro": 1000,
                "repros_retained": 0,
                "transitions_diverged": 0,
                "repros_complete": True,
            },
            "repros": [],
            "checkpoint_provenance": {
                "records_with_provenance": 1,
                "complete": True,
            },
            "seeds": {"min": seed, "max": seed, "distinct": 1},
            "transitions_diverged": 0,
            "transitions_matched": measured,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _contract(self, *, minimum_coverage: float = 0.97) -> dict:
        readout_sha = readout._sha256(Path(readout.__file__))
        return {
            "registered_before_launch": True,
            "requires_execution_contract": True,
            "launch_registration": {
                "fresh_measurements_inspected_before_registration": 0,
                "coordinator_go": True,
                "engine_patch_count": 1,
            },
            "certification_gates": {
                "expected_shards": 2,
                "expected_games": 2,
                "seed_blocks": [
                    {"start": 1000, "games": 1},
                    {"start": 2000, "games": 1},
                ],
                "minimum_coverage_measured_fraction": minimum_coverage,
                "required_build_check": "gated",
                "required_matcher": "strict",
                "required_repros_per_game": 40,
                "required_keep_repro": 1000,
                "required_behavioral_probe_passes": 9,
                "required_source_commit": self.source_commit,
                "required_image_commit": self.image_commit,
                "required_engine_fingerprint": self.engine_fingerprint,
                "required_readout_sha256": readout_sha,
                "required_execution_manifest_producer_sha256": readout._sha256(
                    ROOT / "scripts" / "cert_execution_manifest.py"
                ),
            },
            "pre_registered_family_rate_table": {
                "documented_families": {},
                "new_mechanisms_post_fix": {},
            },
            "predicted_class_rates_10k": {},
        }

    def _checkpoint(self, root: Path, seed: int) -> dict:
        path = root / f"checkpoint-{seed}.jsonl"
        path.write_text(
            json.dumps(
                {
                    "schema": "engine-transition-differential/1",
                    "build_check": "gated",
                    "seed": seed,
                    "provenance": {
                        "source_commit": self.source_commit,
                        "engine_fingerprint": self.engine_fingerprint,
                        "image_commit": self.image_commit,
                    },
                }
            ) + "\n",
            encoding="utf-8",
        )
        evidence = self._file(path)
        evidence.update({
            "records": 1,
            "resume_provenance": {
                "source_commit": self.source_commit,
                "engine_fingerprint": self.engine_fingerprint,
                "image_commit": self.image_commit,
            },
        })
        return evidence

    def _manifest(self, paths: list[Path], contract_path: Path, *, complete: bool = True) -> dict:
        root = contract_path.parent
        marker_evidence = []
        for index in range(2):
            marker = root / f"complete-{index}"
            if complete:
                marker_evidence.append(self._file(marker, "complete\n"))
            else:
                marker_evidence.append({"path": os.fspath(marker), "sha256": "0" * 64})
        behavior = [self._probe_log(root / f"behavior-{index}.log") for index in range(2)]
        branch = [self._probe_log(root / f"branch-{index}.log", branch=True) for index in range(2)]
        return {
            "schema": "engine-cert-execution-manifest/2",
            "producer": {
                "path": os.fspath(ROOT / "scripts" / "cert_execution_manifest.py"),
                "sha256": readout._sha256(ROOT / "scripts" / "cert_execution_manifest.py"),
            },
            "source": {"commit": self.source_commit, "checkout": os.fspath(ROOT)},
            "contract_blob": self._file(root / "contract-blob.json", contract_path.read_text()),
            "readout_blob": {
                "path": os.fspath(Path(readout.__file__)),
                "sha256": readout._sha256(Path(readout.__file__)),
            },
            "engine_provenance": {
                "fingerprint": self.engine_fingerprint,
                "stamp": self._stamp(root),
            },
            "aggregate_provenance": {
                "behavioral_probes": self._probe_log(root / "aggregate-behavior.log"),
                "branch_events_probe": self._probe_log(root / "aggregate-branch.log", branch=True),
            },
            "shards": [
                {
                    "seed_start": 1000 if index == 0 else 2000,
                    "report": self._file(paths[index], paths[index].read_text()),
                    "checkpoint": self._checkpoint(root, 1000 if index == 0 else 2000),
                    "completion_marker": marker_evidence[index],
                    "image_commit": self.image_commit,
                    "behavioral_probes": behavior[index],
                    "branch_events_probe": branch[index],
                }
                for index in range(2)
            ],
        }

    def _runtime(self) -> dict:
        return {
            "source_commit": self.source_commit,
            "checkout": os.fspath(ROOT),
            "readout_path": os.fspath(Path(readout.__file__)),
            "readout_sha256": readout._sha256(Path(readout.__file__)),
        }

    def _run(self, root: Path, *, contract: dict | None = None, manifest: dict | None = None) -> dict:
        paths = [root / "shard-0.json", root / "shard-1.json"]
        contract_path = root / "contract.json"
        manifest_path = root / "manifest.json"
        output_path = root / "readout.json"
        contract_path.write_text(json.dumps(contract or self._contract()), encoding="utf-8")
        manifest_path.write_text(
            json.dumps(manifest or self._manifest(paths, contract_path)), encoding="utf-8"
        )
        with patch.object(readout, "_current_runtime_provenance", return_value=self._runtime()):
            exit_code = readout.main(
                [
                    "--shards",
                    *(os.fspath(path) for path in paths),
                    "--prediction",
                    os.fspath(contract_path),
                    "--execution-manifest",
                    os.fspath(manifest_path),
                    "--json",
                    os.fspath(output_path),
                ]
            )
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(exit_code, 0 if payload["verdict"] == "PASS" else 1)
        return payload

    def test_complete_registered_execution_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / "shard-0.json", root / "shard-1.json"]
            self._shard(paths[0], 1000)
            self._shard(paths[1], 2000)
            payload = self._run(root)
        self.assertEqual(payload["verdict"], "PASS")
        self.assertEqual(payload["gate_failures"], [])
        self.assertEqual(payload["enforcement_status"], "enforced")
        self.assertEqual(payload["contract_evidence"]["distinct_seed_total"], 2)
        self.assertEqual(payload["skip_counters"]["skip:world_unsupported:test_fixture"], 4)

    def test_failure_writes_json_and_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / "shard-0.json", root / "shard-1.json"]
            self._shard(paths[0], 1000)
            self._shard(paths[1], 2000)
            contract = self._contract()
            contract_path = root / "contract.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            payload = self._run(
                root,
                contract=contract,
                manifest=self._manifest(paths, contract_path, complete=False),
            )
        self.assertEqual(payload["verdict"], "FAIL")
        self.assertTrue(any("completion marker" in failure for failure in payload["gate_failures"]))

    def test_registered_contract_cannot_fall_back_to_legacy_mode(self) -> None:
        failures, evidence = readout._contract_gates(
            paths=[],
            shards=[],
            contract={"registered_before_launch": True},
            contract_path=Path(readout.__file__),
            execution_manifest=None,
            coverage=1.0,
            aggregate={"games": 0},
            legacy_opt_out=True,
        )
        self.assertEqual(len(failures), 1)
        self.assertEqual(evidence["enforcement_status"], "refused-final-contract")

    def test_unregistered_legacy_requires_two_explicit_opt_ins(self) -> None:
        contract = {"legacy_contract_opt_out": True}
        failures, evidence = readout._contract_gates(
            paths=[], shards=[], contract=contract, contract_path=Path(readout.__file__),
            execution_manifest=None, coverage=1.0, aggregate={"games": 0}, legacy_opt_out=True,
        )
        self.assertEqual(failures, [])
        self.assertEqual(evidence["enforcement_status"], "legacy-opt-out")
        failures, _ = readout._contract_gates(
            paths=[], shards=[], contract=contract, contract_path=Path(readout.__file__),
            execution_manifest=None, coverage=1.0, aggregate={"games": 0}, legacy_opt_out=False,
        )
        self.assertTrue(failures)

    def test_aggregate_and_branch_probe_provenance_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / "shard-0.json", root / "shard-1.json"]
            self._shard(paths[0], 1000)
            self._shard(paths[1], 2000)
            contract = self._contract()
            contract_path = root / "contract.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            manifest = self._manifest(paths, contract_path)
            manifest.pop("aggregate_provenance")
            manifest["shards"][0]["checkpoint"]["sha256"] = "0" * 64
            payload = self._run(root, contract=contract, manifest=manifest)
        self.assertEqual(payload["verdict"], "FAIL")
        self.assertIn("execution manifest has no aggregate provenance", payload["gate_failures"])
        self.assertTrue(any("checkpoint SHA-256" in failure for failure in payload["gate_failures"]))

    def test_seed_bool_and_coverage_over_one_fail_gracefully(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / "shard-0.json", root / "shard-1.json"]
            self._shard(paths[0], 1000, measured=101)
            self._shard(paths[1], 2000)
            payload = json.loads(paths[1].read_text())
            payload["seeds"]["min"] = True
            paths[1].write_text(json.dumps(payload), encoding="utf-8")
            output = self._run(root)
        self.assertEqual(output["verdict"], "FAIL")
        self.assertTrue(any("coverage" in failure for failure in output["gate_failures"]))
        self.assertTrue(any("malformed seed summary" in failure for failure in output["gate_failures"]))

    def test_wilson_lower_bound_is_advisory_for_unobserved_family(self) -> None:
        contract = self._contract()
        contract["pre_registered_family_rate_table"]["documented_families"] = {
            "I1_cap_state_shape": {"wilson95_rate": [0.01, 0.03]}
        }
        failures, evidence = readout._family_rate_gates({}, contract, boundaries_measured=100)
        self.assertEqual(failures, [])
        self.assertEqual(evidence["families"]["I1_cap_state_shape"]["observed"], 0)
        self.assertEqual(evidence["families"]["I1_cap_state_shape"]["lower_rate_advisory"], 0.01)

    def test_pre_registered_prediction_lower_bound_is_binding(self) -> None:
        contract = self._contract()
        contract["pre_registered_family_rate_table"]["documented_families"] = {
            "I1_cap_state_shape": {
                "wilson95_rate": [0.01, 0.03],
                "prediction_interval_rate": [0.01, 0.03],
            }
        }
        failures, _ = readout._family_rate_gates({}, contract, boundaries_measured=100)
        self.assertEqual(
            failures,
            ["registered family 'I1_cap_state_shape' rate 0 is below pre-registered prediction lower rate 0.01"],
        )

    def test_wilson_upper_bound_remains_binding(self) -> None:
        contract = self._contract()
        contract["pre_registered_family_rate_table"]["documented_families"] = {
            "I1_cap_state_shape": {"wilson95_rate": [0.01, 0.03]}
        }
        failures, _ = readout._family_rate_gates(
            {"I1_cap_state_shape": 4}, contract, boundaries_measured=100
        )
        self.assertEqual(
            failures,
            ["registered family 'I1_cap_state_shape' rate 0.04 exceeds registered upper rate 0.03"],
        )

    def test_checkpoint_requires_every_game_record_and_complete_report_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = self._checkpoint(root, 1000)
            failures: list[str] = []
            readout._checkpoint_provenance(
                checkpoint,
                label="two-game checkpoint",
                failures=failures,
                required_source=self.source_commit,
                required_fingerprint=self.engine_fingerprint,
                required_image=self.image_commit,
                expected_seed_range=(1000, 1001),
                expected_records=2,
                expected_distinct_seeds=2,
            )
        self.assertIn("two-game checkpoint has 1 checkpoint records, expected 2 for its shard", failures)
        self.assertIn("two-game checkpoint has 1 distinct checkpoint seeds, expected 2 for its shard", failures)

    def test_incomplete_shard_checkpoint_provenance_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / "shard-0.json", root / "shard-1.json"]
            self._shard(paths[0], 1000)
            self._shard(paths[1], 2000)
            first = json.loads(paths[0].read_text(encoding="utf-8"))
            first["checkpoint_provenance"]["complete"] = False
            paths[0].write_text(json.dumps(first), encoding="utf-8")
            payload = self._run(root)
        self.assertEqual(payload["verdict"], "FAIL")
        self.assertIn("shard-0.json: checkpoint_provenance is not complete", payload["gate_failures"])

    def test_consumer_rejects_manifest_missing_per_shard_probe_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / "shard-0.json", root / "shard-1.json"]
            self._shard(paths[0], 1000)
            self._shard(paths[1], 2000)
            contract_path = root / "contract.json"
            contract = self._contract()
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            manifest = self._manifest(paths, contract_path)
            manifest["shards"][0]["behavioral_probes"].pop("total")
            payload = self._run(root, contract=contract, manifest=manifest)
        self.assertEqual(payload["verdict"], "FAIL")
        self.assertTrue(any("behavioral_probes is missing required field 'total'" in failure
                            for failure in payload["gate_failures"]))

    def test_missing_aggregate_scalar_never_defaults_to_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / "shard-0.json", root / "shard-1.json"]
            self._shard(paths[0], 1000)
            self._shard(paths[1], 2000)
            first = json.loads(paths[0].read_text(encoding="utf-8"))
            first.pop("engine_errors")
            paths[0].write_text(json.dumps(first), encoding="utf-8")
            payload = self._run(root)
        self.assertEqual(payload["verdict"], "FAIL")
        self.assertIn("shard-0.json: missing or malformed aggregate scalar 'engine_errors'", payload["gate_failures"])

    def test_malformed_first_seed_summary_cannot_mislabel_second_shard_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / "shard-0.json", root / "shard-1.json"]
            self._shard(paths[0], 1000)
            self._shard(paths[1], 2000)
            first = json.loads(paths[0].read_text(encoding="utf-8"))
            first["seeds"]["min"] = True
            paths[0].write_text(json.dumps(first), encoding="utf-8")
            contract = self._contract()
            contract_path = root / "contract.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            manifest = self._manifest(paths, contract_path)
            manifest["shards"][1]["checkpoint"]["sha256"] = "0" * 64
            payload = self._run(root, contract=contract, manifest=manifest)
        self.assertTrue(any(failure.startswith("shard-1.json checkpoint SHA-256")
                            for failure in payload["gate_failures"]))
        self.assertFalse(any(failure.startswith("shard-0.json checkpoint SHA-256")
                             for failure in payload["gate_failures"]))

    def test_runtime_provenance_uses_injected_public_commit_without_git(self) -> None:
        with patch.object(readout, "public_repo_commit", return_value=self.source_commit):
            runtime = readout._current_runtime_provenance()
            checkout = readout._checkout_commit(Path("/no-git-image"))
        self.assertEqual(runtime["source_commit"], self.source_commit)
        self.assertEqual(checkout, self.source_commit)

    def test_contract_producer_hash_is_checked_against_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / "shard-0.json", root / "shard-1.json"]
            self._shard(paths[0], 1000)
            self._shard(paths[1], 2000)
            contract = self._contract()
            contract_path = root / "contract.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            manifest = self._manifest(paths, contract_path)
            manifest["producer"]["sha256"] = "0" * 64
            payload = self._run(root, contract=contract, manifest=manifest)
        self.assertEqual(payload["verdict"], "FAIL")
        self.assertTrue(any("execution manifest producer SHA-256 does not match its artifact" in failure
                            for failure in payload["gate_failures"]))

    def test_predicted_zero_requires_an_emittable_exclusion_counter(self) -> None:
        contract = self._contract()
        contract["pre_registered_family_rate_table"]["new_mechanisms_post_fix"] = {
            "typed_hidden_power_thaw": {"predicted_next": 0}
        }
        failures, _ = readout._family_rate_gates({}, contract, boundaries_measured=100)
        self.assertEqual(
            failures,
            ["post-fix mechanism 'typed_hidden_power_thaw' has no emittable classifier exclusion counter"],
        )

    def test_predicted_zero_exclusion_counter_is_enforced(self) -> None:
        contract = self._contract()
        contract["pre_registered_family_rate_table"]["new_mechanisms_post_fix"] = {
            "recharge": {
                "predicted_next": 0,
                "classifier_outcome": "UNATTRIBUTED",
                "exclusion_counter": "recharge_turn_residual_gap",
            }
        }
        failures, _ = readout._family_rate_gates(
            {}, contract, boundaries_measured=100,
            exclusion_counts={"recharge_turn_residual_gap": 1},
        )
        self.assertEqual(
            failures,
            [
                "predicted-zero post-fix mechanism 'recharge' observed 1 times "
                "through exclusion counter 'recharge_turn_residual_gap'"
            ],
        )

    def test_duplicate_repro_cannot_replace_missing_identity(self) -> None:
        rows = [{"seed": 1000, "step": 7}, {"seed": 1000, "step": 7}]
        shards = [{"seeds": {"min": 1000, "max": 1000, "distinct": 1}}]
        self.assertEqual(
            readout._repro_integrity_gates(rows, shards),
            ["retained repro population contains 1 duplicate seed/step identities"],
        )

    def test_repro_must_belong_to_its_own_shard_band(self) -> None:
        rows = [{"seed": 2000, "step": 7, "_cert_shard_seed_start": 1000}]
        shards = [
            {"seeds": {"min": 1000, "max": 1000, "distinct": 1}},
            {"seeds": {"min": 2000, "max": 2000, "distinct": 1}},
        ]
        self.assertEqual(
            readout._repro_integrity_gates(rows, shards),
            ["retained repro identity (2000, 7) is outside its shard seed band"],
        )


if __name__ == "__main__":
    unittest.main()
