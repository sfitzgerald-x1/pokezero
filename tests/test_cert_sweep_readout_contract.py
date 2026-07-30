"""Fail-closed pins for the certification execution/readout contract."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT / "scripts"))

import cert_sweep_readout as readout  # noqa: E402


class CertificationContractTests(unittest.TestCase):
    source_commit = "a" * 40
    engine_fingerprint = "b" * 64

    def _shard(self, path: Path, seed: int, *, measured: int = 100) -> None:
        payload = {
            "acceptance_eligible": True,
            "boundaries_full_round": 100,
            "boundaries_measured": measured,
            "build_check": "gated",
            "counters": {
                "boundaries_full_round": 100,
                "boundaries_measured": measured,
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
            "seeds": {"min": seed, "max": seed, "distinct": 1},
            "transitions_diverged": 0,
            "transitions_matched": measured,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _contract(self, *, minimum_coverage: float = 0.97) -> dict:
        return {
            "registered_before_launch": True,
            "launch_registration": {
                "fresh_measurements_inspected_before_registration": 0,
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
                "required_engine_fingerprint": self.engine_fingerprint,
                "required_readout_sha256": readout._sha256(Path(readout.__file__)),
            },
            "pre_registered_family_rate_table": {
                "documented_families": {},
                "new_mechanisms_post_fix": {},
            },
            "predicted_class_rates_10k": {},
        }

    def _manifest(
        self,
        paths: list[Path],
        contract_path: Path,
        *,
        complete: bool = True,
    ) -> dict:
        return {
            "schema": "engine-cert-execution-manifest/1",
            "source_commit": self.source_commit,
            "image_commit": "c" * 40,
            "engine_fingerprint": self.engine_fingerprint,
            "readout_sha256": readout._sha256(Path(readout.__file__)),
            "contract_sha256": readout._sha256(contract_path),
            "shards": [
                {
                    "seed_start": 1000 if index == 0 else 2000,
                    "report_sha256": readout._sha256(path),
                    "complete_marker": complete,
                    "behavioral_probes": {"passed": 9, "total": 9},
                    "source_commit": self.source_commit,
                    "engine_fingerprint": self.engine_fingerprint,
                }
                for index, path in enumerate(paths)
            ],
        }

    def _run(
        self,
        root: Path,
        *,
        contract: dict | None = None,
        manifest: dict | None = None,
    ) -> dict:
        paths = [root / "shard-0.json", root / "shard-1.json"]
        contract_path = root / "contract.json"
        manifest_path = root / "manifest.json"
        output_path = root / "readout.json"
        contract_path.write_text(
            json.dumps(contract or self._contract()), encoding="utf-8"
        )
        manifest_path.write_text(
            json.dumps(manifest or self._manifest(paths, contract_path)),
            encoding="utf-8",
        )
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
        self.assertEqual(exit_code, 0)
        return json.loads(output_path.read_text(encoding="utf-8"))

    def test_complete_registered_execution_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / "shard-0.json", root / "shard-1.json"]
            self._shard(paths[0], 1000)
            self._shard(paths[1], 2000)
            payload = self._run(root)
        self.assertEqual(payload["verdict"], "PASS")
        self.assertEqual(payload["gate_failures"], [])
        self.assertEqual(payload["contract_evidence"]["distinct_seed_total"], 2)

    def test_missing_completion_marker_fails(self) -> None:
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
        self.assertTrue(
            any("completion marker" in failure for failure in payload["gate_failures"])
        )

    def test_seed_block_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._shard(root / "shard-0.json", 1000)
            self._shard(root / "shard-1.json", 1000)
            payload = self._run(root)
        self.assertEqual(payload["verdict"], "FAIL")
        self.assertIn(
            "observed seed blocks do not exactly match the registered blocks",
            payload["gate_failures"],
        )

    def test_coverage_below_registered_floor_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._shard(root / "shard-0.json", 1000, measured=90)
            self._shard(root / "shard-1.json", 2000, measured=90)
            payload = self._run(root, contract=self._contract(minimum_coverage=0.95))
        self.assertEqual(payload["verdict"], "FAIL")
        self.assertTrue(
            any("below registered floor" in failure for failure in payload["gate_failures"])
        )

    def test_unregistered_family_fails(self) -> None:
        failures, _ = readout._family_rate_gates(
            {"new_broad_echo": 1}, self._contract()
        )
        self.assertEqual(
            failures, ["attributed family 'new_broad_echo' was not pre-registered"]
        )

    def test_registered_family_upper_bound_is_enforced(self) -> None:
        contract = self._contract()
        contract["pre_registered_family_rate_table"]["documented_families"] = {
            "known_limit": {"wilson95": [1, 3]}
        }
        failures, _ = readout._family_rate_gates({"known_limit": 4}, contract)
        self.assertEqual(
            failures,
            [
                "attributed family 'known_limit' count 4 exceeds "
                "registered upper bound 3"
            ],
        )


if __name__ == "__main__":
    unittest.main()
