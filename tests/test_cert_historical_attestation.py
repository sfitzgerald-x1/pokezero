"""Regression guards for immutable, historical certification evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
C25_ATTESTATION_SOURCE_COMMIT = "a3e16f2a49cf55197f026e912c8afa31fc5334ac"
C25_ATTESTATION_BLOB_SHA = "9cc357a9872f6b5fc633e97e97fb85165077e8f7"
C26_SUCCESSOR_ARTIFACTS = (
    "reports/c26_current_engine_resweep_spec.json",
    "reports/c26_current_engine_calibration.json",
    "reports/c26_current_engine_attestation.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HistoricalCertificationAttestationTests(unittest.TestCase):
    def test_c25_attestation_matches_its_immutable_git_blob(self) -> None:
        attestation_path = ROOT / "reports" / "c25_cert_contract_attestation.json"
        attested = subprocess.check_output(
            (
                "git",
                "-C",
                str(ROOT),
                "show",
                f"{C25_ATTESTATION_SOURCE_COMMIT}:{attestation_path.relative_to(ROOT)}",
            )
        )

        self.assertEqual(attestation_path.read_bytes(), attested)
        blob_sha = subprocess.check_output(
            (
                "git",
                "-C",
                str(ROOT),
                "rev-parse",
                f"{C25_ATTESTATION_SOURCE_COMMIT}:{attestation_path.relative_to(ROOT)}",
            ),
            text=True,
        ).strip()
        self.assertEqual(blob_sha, C25_ATTESTATION_BLOB_SHA)

    def test_c15_contract_matches_its_c25_attested_git_blob(self) -> None:
        attestation_path = ROOT / "reports" / "c25_cert_contract_attestation.json"
        attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
        contract_path = ROOT / attestation["contract_path"]
        current_contract = contract_path.read_bytes()
        attested_contract = subprocess.check_output(
            (
                "git",
                "-C",
                str(ROOT),
                "show",
                f"{attestation['contract_source_commit']}:{attestation['contract_path']}",
            )
        )

        self.assertEqual(current_contract, attested_contract)
        self.assertEqual(
            hashlib.sha256(current_contract).hexdigest(),
            attestation["contract_sha256"],
        )

    def test_active_source_lifecycle_stays_non_launchable(self) -> None:
        lifecycle_path = ROOT / "reports" / "certification_contract_lifecycle.json"
        if not lifecycle_path.is_file():
            # A source-freeze marker may disappear only once all successor
            # evidence exists; otherwise its removal would silently unpin the
            # current build before registration completes.
            for relative in C26_SUCCESSOR_ARTIFACTS:
                self.assertTrue((ROOT / relative).is_file(), relative)
            return

        lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
        self.assertEqual(
            lifecycle["schema_version"],
            "pokezero.engine-cert-contract-lifecycle/v1",
        )
        self.assertIs(lifecycle["launchable"], False)
        self.assertIn(
            lifecycle["stage"],
            # attested_fail is a real terminal state, added when C32 ran to
            # completion and failed. A negative result is evidence the program is
            # designed to produce, so the lifecycle has to be able to express it
            # rather than looking forever pending.
            {"build_source", "contract_registered_attestation_pending", "attested_fail"},
        )
        for relative in lifecycle["required_absent_artifacts"]:
            self.assertFalse((ROOT / relative).exists(), relative)
        if lifecycle["stage"] == "attested_fail":
            attestation = ROOT / lifecycle["registered_sweep_attestation_path"]
            self.assertTrue(attestation.is_file())
            import json as _json
            payload = _json.loads(attestation.read_text(encoding="utf-8"))
            self.assertEqual(payload["verdict"], "FAIL")
            self.assertEqual(lifecycle["result"]["verdict"], "FAIL")
            self.assertIs(lifecycle["launchable"], False)
            return
        if lifecycle["stage"] == "build_source":
            for field in (
                "registered_contract_path",
                "registered_contract_sha256",
                "registered_calibration_path",
            ):
                self.assertNotIn(field, lifecycle)
            identity = lifecycle["source_code_identity"]
            self.assertEqual(
                identity["readout_sha256"],
                _sha256(ROOT / "scripts" / "cert_sweep_readout.py"),
            )
            self.assertEqual(
                identity["execution_manifest_producer_sha256"],
                _sha256(ROOT / "scripts" / "cert_execution_manifest.py"),
            )
            fingerprint = json.loads(
                subprocess.check_output(
                    [sys.executable, "scripts/engine_build_fingerprint.py", "--print"],
                    cwd=ROOT,
                    text=True,
                )
            )
            self.assertEqual(
                identity["engine_fingerprint"], fingerprint["fingerprint"]
            )
            self.assertIs(type(identity["engine_patch_count"]), int)
            self.assertGreater(identity["engine_patch_count"], 0)
            self.assertEqual(identity["engine_patch_count"], fingerprint["count"])
            return

        self.assertEqual(
            lifecycle["stage"], "contract_registered_attestation_pending"
        )
        contract_path = ROOT / lifecycle["registered_contract_path"]
        self.assertEqual(
            lifecycle["registered_contract_sha256"],
            _sha256(contract_path),
        )
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        calibration_path = ROOT / lifecycle["registered_calibration_path"]
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        self.assertEqual(
            calibration["source_evidence"]["current_classifier_source_commit"],
            contract["certification_gates"]["required_source_commit"],
        )


if __name__ == "__main__":
    unittest.main()
