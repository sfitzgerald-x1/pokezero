"""Registration invariants for the C26 certification contract.

C15 has an equivalent guard in ``tests/test_cert_execution_manifest.py``; that
one protects attested history and must keep passing untouched. This one protects
the live registration: the contract has to pin a real build-source freeze, agree
with the calibration it was derived from, reserve seeds nothing else has used,
and stop short of claiming a sweep that has not run.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The ACTIVE registration. C26's artifacts stay as history and are guarded by
# LifecycleTests below; these invariants follow whichever contract is current.
CONTRACT_PATH = ROOT / "reports" / "c32_current_engine_resweep_spec.json"
C26_CONTRACT_PATH = ROOT / "reports" / "c26_current_engine_resweep_spec.json"
CALIBRATION_PATH = ROOT / "reports" / "c32_current_engine_calibration.json"
C26_CALIBRATION_PATH = ROOT / "reports" / "c26_current_engine_calibration.json"
LIFECYCLE_PATH = ROOT / "reports" / "certification_contract_lifecycle.json"
ATTESTATION_PATH = ROOT / "reports" / "c32_current_engine_attestation.json"
C26_ATTESTATION_PATH = ROOT / "reports" / "c26_current_engine_attestation.json"

# Every seed band certification work has already consumed or reserved in public
# evidence. The C26 blocks must not touch any of them.
PUBLIC_CONSUMED_SEED_RANGES = (
    (2000000, 2701249),   # C14 sweep archive, ledger Appendix Z12
    (2800000, 3501249),   # C15 registered blocks
    (16000000, 16701249),  # C26 blocks: burned, not recycled, though never swept
    (17000000, 17000999),  # C26 diagnostic probe band, reserved outward
)


def _show(commit: str, relative: str) -> bytes:
    return subprocess.check_output(("git", "-C", str(ROOT), "show", f"{commit}:{relative}"))


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _calibration() -> dict:
    return json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))


class ContractSchemaTests(unittest.TestCase):
    def test_contract_passes_the_shared_final_contract_schema(self) -> None:
        try:
            manifest = importlib.import_module("cert_execution_manifest")
        except BaseException as error:  # noqa: BLE001
            self.skipTest(f"cert_execution_manifest is unavailable ({error})")
        self.assertEqual(manifest.validate_final_contract_schema(_contract()), [])


class SourceIdentityTests(unittest.TestCase):
    def test_contract_pins_a_real_build_source_freeze(self) -> None:
        contract = _contract()
        gates = contract["certification_gates"]
        source = gates["required_source_commit"]
        self.assertEqual(source, contract["launch_registration"]["public_source_parent_commit"])
        self.assertEqual(source, gates["required_image_commit"])

        lifecycle = json.loads(_show(source, "reports/certification_contract_lifecycle.json"))
        self.assertEqual(lifecycle["stage"], "build_source")
        self.assertIs(lifecycle["launchable"], False)
        # The pinned source must not already contain any successor artifact, or
        # the contract would be pinning a commit that pre-judged its own result.
        for relative in lifecycle["required_absent_artifacts"]:
            absent = subprocess.run(
                ("git", "-C", str(ROOT), "cat-file", "-e", f"{source}:{relative}"),
                capture_output=True,
            )
            self.assertNotEqual(absent.returncode, 0, relative)

        identity = lifecycle["source_code_identity"]
        self.assertEqual(gates["required_engine_fingerprint"], identity["engine_fingerprint"])
        self.assertEqual(gates["required_readout_sha256"], identity["readout_sha256"])
        self.assertEqual(
            gates["required_execution_manifest_producer_sha256"],
            identity["execution_manifest_producer_sha256"],
        )
        self.assertEqual(
            contract["launch_registration"]["engine_patch_count"],
            identity["engine_patch_count"],
        )

    def test_registered_hashes_are_the_bytes_at_the_pinned_source(self) -> None:
        gates = _contract()["certification_gates"]
        source = gates["required_source_commit"]
        for relative, registered in (
            ("scripts/cert_sweep_readout.py", gates["required_readout_sha256"]),
            (
                "scripts/cert_execution_manifest.py",
                gates["required_execution_manifest_producer_sha256"],
            ),
        ):
            self.assertEqual(
                hashlib.sha256(_show(source, relative)).hexdigest(), registered, relative
            )

    def test_c26_source_identity_is_superseded_not_silently_stale(self) -> None:
        """C26 is now historical, and that has to be visible rather than implied.

        This used to assert the working tree still ran the registered bytes,
        which made C26 the ACTIVE registration. C27 moved the engine fingerprint
        and C28 moved the readout, so it no longer is. The registered hashes at
        the pinned commit stay immutable evidence -- checked above -- but the
        working tree has deliberately diverged, exactly as C15's did, and a
        successor registration is owed before any sweep runs.
        """

        gates = json.loads(C26_CONTRACT_PATH.read_text(encoding="utf-8"))[
            "certification_gates"
        ]
        current_readout = hashlib.sha256(
            (ROOT / "scripts" / "cert_sweep_readout.py").read_bytes()
        ).hexdigest()
        self.assertNotEqual(
            current_readout,
            gates["required_readout_sha256"],
            "the working tree matches C26 again; either C28 was reverted or the "
            "successor registration is missing",
        )


class CalibrationAgreementTests(unittest.TestCase):
    def test_calibration_speaks_for_the_registered_source(self) -> None:
        contract, calibration = _contract(), _calibration()
        evidence = calibration["source_evidence"]
        gates = contract["certification_gates"]
        self.assertEqual(
            evidence["current_classifier_source_commit"], gates["required_source_commit"]
        )
        self.assertEqual(
            evidence["current_engine_fingerprint"], gates["required_engine_fingerprint"]
        )
        self.assertEqual(
            evidence["current_classifier_readout_sha256"], gates["required_readout_sha256"]
        )
        self.assertEqual(evidence["fresh_measurements_inspected"], 0)
        self.assertEqual(evidence["archive_role"], "historical_calibration_only")
        self.assertEqual(
            contract["launch_registration"]["calibration_evidence"],
            str(CALIBRATION_PATH.relative_to(ROOT)),
        )

    def test_every_registered_bound_comes_from_the_calibration(self) -> None:
        contract, calibration = _contract(), _calibration()
        table = contract["pre_registered_family_rate_table"]
        self.assertEqual(
            table["calibration_boundaries"], calibration["calibration_boundaries"]
        )
        derived = dict(calibration["registered_family_count_intervals"])
        budget = calibration["registered_non_empirical_upper_rates"][
            "limit:world_substitute_health_unknown"
        ]
        for family, prediction in table["documented_families"].items():
            if family == "limit:world_substitute_health_unknown":
                self.assertEqual(
                    prediction["upper_full_round_rate"], budget["upper_full_round_rate"]
                )
                self.assertEqual(prediction["upper_rate_basis"], budget["upper_rate_basis"])
                self.assertEqual(
                    prediction["risk_budget_rationale"], budget["risk_budget_rationale"]
                )
                continue
            self.assertEqual(prediction["wilson95"], derived.pop(family), family)
        self.assertEqual(derived, {}, "a calibrated family was left unregistered")

    def test_predicted_classes_are_the_calibrated_classes(self) -> None:
        contract, calibration = _contract(), _calibration()
        counts = calibration["current_engine_class_counts"]
        intervals = calibration["predicted_class_count_intervals"]
        predicted = contract["predicted_class_rates_10k"]
        self.assertEqual(set(predicted), set(counts))
        for name, prediction in predicted.items():
            self.assertEqual(prediction["expected_10k"], counts[name], name)
            self.assertEqual(prediction["wilson95_count_10k"], intervals[name], name)

    def test_the_archival_unattributed_residue_is_declared_not_absorbed(self) -> None:
        """A registered-zero counter with archival residue must be visible."""

        contract, calibration = _contract(), _calibration()
        residue = calibration["current_engine_exclusion_counts"]
        mechanisms = contract["pre_registered_family_rate_table"]["new_mechanisms_post_fix"]
        for name, prediction in mechanisms.items():
            self.assertEqual(prediction["predicted_next"], 0, name)
            self.assertEqual(
                prediction["archival_residue_on_frozen_build"],
                int(residue.get(name, 0)),
                name,
            )
        declared = [entry for entry in contract["gated_on"] if entry.startswith("DECLARED RISK")]
        if any(prediction["archival_residue_on_frozen_build"] for prediction in mechanisms.values()):
            self.assertTrue(
                declared,
                "archival residue exists but the contract declares no registration risk",
            )


class SeedReservationTests(unittest.TestCase):
    def test_blocks_are_eight_fresh_non_overlapping_1250_game_blocks(self) -> None:
        gates = _contract()["certification_gates"]
        blocks = gates["seed_blocks"]
        self.assertEqual(len(blocks), gates["expected_shards"])
        self.assertEqual(sum(block["games"] for block in blocks), gates["expected_games"])
        previous_end: int | None = None
        for block in sorted(blocks, key=lambda item: item["start"]):
            self.assertEqual(block["games"], 1250)
            if previous_end is not None:
                self.assertGreater(block["start"], previous_end)
            previous_end = block["start"] + block["games"] - 1

    def test_blocks_are_disjoint_from_publicly_consumed_seeds(self) -> None:
        blocks = _contract()["certification_gates"]["seed_blocks"]
        for block in blocks:
            start, end = block["start"], block["start"] + block["games"] - 1
            for low, high in PUBLIC_CONSUMED_SEED_RANGES:
                self.assertFalse(
                    start <= high and low <= end,
                    f"block {start}-{end} overlaps consumed range {low}-{high}",
                )

    def test_c15_blocks_are_not_reused(self) -> None:
        c15 = json.loads(
            (ROOT / "reports" / "c15_resweep_spec.json").read_text(encoding="utf-8")
        )
        historical = {block["start"] for block in c15["certification_gates"]["seed_blocks"]}
        historical |= {
            block["start"]
            for block in json.loads(C26_CONTRACT_PATH.read_text(encoding="utf-8"))[
                "certification_gates"
            ]["seed_blocks"]
        }
        current = {block["start"] for block in _contract()["certification_gates"]["seed_blocks"]}
        self.assertEqual(historical & current, set())


class RetentionTests(unittest.TestCase):
    def test_keep_repro_covers_the_structural_worst_case_shard(self) -> None:
        """A shard that diverges on every retained boundary must still be complete."""

        gates = _contract()["certification_gates"]
        worst_case = max(block["games"] for block in gates["seed_blocks"]) * gates[
            "required_repros_per_game"
        ]
        self.assertGreaterEqual(gates["required_keep_repro"], worst_case)


class ContractAttestationTests(unittest.TestCase):
    """The CONTRACT attestation, which is not the sweep-result attestation."""

    PATH = ROOT / "reports" / "c32_cert_contract_attestation.json"

    def _attestation(self) -> dict:
        if not self.PATH.is_file():
            self.skipTest("the active contract attestation is not registered yet")
        return json.loads(self.PATH.read_text(encoding="utf-8"))

    def test_attests_the_registered_contract_bytes(self) -> None:
        attestation = self._attestation()
        self.assertEqual(
            attestation["schema_version"],
            "pokezero.engine-cert-contract-attestation/v1",
        )
        self.assertEqual(attestation["contract_path"], str(CONTRACT_PATH.relative_to(ROOT)))
        attested = _show(attestation["contract_source_commit"], attestation["contract_path"])
        self.assertEqual(attested, CONTRACT_PATH.read_bytes())
        self.assertEqual(
            hashlib.sha256(attested).hexdigest(), attestation["contract_sha256"]
        )

    def test_attestation_is_not_circular(self) -> None:
        """It pins the registration commit, which cannot contain it."""

        attestation = self._attestation()
        present = subprocess.run(
            ("git", "-C", str(ROOT), "cat-file", "-e",
             f"{attestation['contract_source_commit']}:"
             f"{self.PATH.relative_to(ROOT)}"),
            capture_output=True,
        )
        self.assertNotEqual(present.returncode, 0)

    def test_attestation_agrees_with_the_launch_identity(self) -> None:
        attestation = self._attestation()
        gates = _contract()["certification_gates"]
        self.assertEqual(attestation["launch_source_commit"], gates["required_source_commit"])
        self.assertEqual(attestation["launch_source_commit"], gates["required_image_commit"])
        self.assertEqual(
            attestation["required_engine_fingerprint"],
            gates["required_engine_fingerprint"],
        )
        self.assertEqual(attestation["registration"]["fresh_c26_measurements_inspected"], 0)

    def test_the_sweep_result_attestation_is_a_different_artifact(self) -> None:
        """A contract attestation must never be mistaken for a sweep result."""

        self._attestation()
        self.assertNotEqual(self.PATH, ATTESTATION_PATH)
        self.assertFalse(ATTESTATION_PATH.exists())
        lifecycle = json.loads(LIFECYCLE_PATH.read_text(encoding="utf-8"))
        self.assertNotIn(
            str(self.PATH.relative_to(ROOT)), lifecycle["required_absent_artifacts"]
        )


class LifecycleTests(unittest.TestCase):
    def test_c26_is_superseded_without_ever_having_been_executed(self) -> None:
        """C26 was registered, held, and then superseded — all three recorded.

        This used to assert the lifecycle still pointed at C26. It no longer
        does: C27/C31 moved the engine and C28/C29/C30 moved the readout, so a
        new build source is frozen. What must stay true is that C26's own
        artifacts survive as evidence, that the supersession is explicit rather
        than implied, and that its sweep attestation never appeared — because the
        sweep was deliberately never run.
        """

        lifecycle = json.loads(LIFECYCLE_PATH.read_text(encoding="utf-8"))
        superseded = lifecycle["supersedes"]
        self.assertEqual(superseded["registration"], "C26")
        self.assertEqual(superseded["status"], "registered_never_executed")
        self.assertEqual(superseded["contract"], str(C26_CONTRACT_PATH.relative_to(ROOT)))
        self.assertEqual(
            superseded["calibration"], str(C26_CALIBRATION_PATH.relative_to(ROOT))
        )
        for relative in (superseded["contract"], superseded["calibration"],
                         superseded["contract_attestation"], superseded["diagnosis"]):
            self.assertTrue((ROOT / relative).is_file(), relative)
        # The SWEEP attestation is the one that must never have appeared.
        self.assertFalse(C26_ATTESTATION_PATH.exists())
        self.assertNotIn(
            str(C26_ATTESTATION_PATH.relative_to(ROOT)),
            lifecycle["required_absent_artifacts"],
            "C26's sweep attestation is no longer pending; it is abandoned",
        )


if __name__ == "__main__":
    unittest.main()
