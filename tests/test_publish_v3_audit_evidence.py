from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "publish_v3_audit_evidence.py"
SPEC = importlib.util.spec_from_file_location("publish_v3_audit_evidence", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
PUBLISHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PUBLISHER)


IMAGE = "registry.example.invalid/pokezero@sha256:" + "a" * 64
COMMIT = "b" * 40
SOURCE_HASH = "c" * 16


def provenance(*, command: list[str], seed_range=None, shard=None, execution_scope=None):
    payload = {
        "recorded_at": "2026-07-21T00:00:00+00:00",
        "public_repo_commit": COMMIT,
        "showdown_source_hash": SOURCE_HASH,
        "observation_schema": "pokezero.observation.v3",
        "image_digest": IMAGE,
        "command": command,
    }
    if seed_range is not None:
        payload["seed_range"] = seed_range
    if shard is not None:
        payload["shard"] = shard
    if execution_scope is None:
        script = Path(command[0]).name
        execution_scope = {
            "coverage_enumeration_audit.py": {"seed_range": seed_range, "shard": shard},
            "deep_line_audit.py": {
                "seed_range": seed_range,
                "max_rounds": 250,
                "scenario_names": [],
                "protocol_fixtures": False,
            },
            "silent_mutation_audit.py": {
                "seed_range": seed_range,
                "max_rounds": 120,
                "scenario_names": [],
            },
            "encoding_collision_audit.py": {
                "decision_range": {"start": 0, "limit": 100000, "end_exclusive": 100000},
                "input_kind": "collision-sketch",
                "input_artifact_count": 1,
            },
            "protocol_emission_inventory.py": {
                "input_audit_count": 0,
                "seed_range": None,
                "shard": None,
            },
        }[script]
    payload["execution_scope"] = execution_scope
    return payload


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def aggregate_identity(value: dict) -> dict:
    return {
        "public_repo_commit": value["public_repo_commit"],
        "showdown_source_hash": value["showdown_source_hash"],
        "observation_schema": value["observation_schema"],
        "image_digest": value["image_digest"],
    }


class PublishV3AuditEvidenceTests(unittest.TestCase):
    def _coverage_root(self, root: Path) -> Path:
        coverage = root / "coverage"
        stages = {}
        for stage in ("static", "depth"):
            command = [
                "scripts/coverage_enumeration_audit.py",
                "--exact-variants",
                "--observation-schema",
                "v3",
                "--no-universal-lane",
                "--shard",
                "0/1",
                "--json",
                "/shared/private/audit.json",
            ]
            depth_rounds = 0 if stage == "static" else 8
            if depth_rounds:
                command.extend(("--depth-rounds", str(depth_rounds)))
            coverage_provenance = provenance(
                command=command,
                seed_range={"start": 1, "end": 3},
                shard={"index": 0, "count": 1},
            )
            audit = {
                "schema_version": "pokezero.deep-line-audit.v1",
                "protocol_signature_schema_version": "pokezero.protocol-signature-census.v2",
                "finding_count": 0,
                "decisions_checked": 10,
                "coverage_execution": {"depth_rounds": depth_rounds, "failure_artifact_count": 0},
                "audit_provenance": coverage_provenance,
            }
            write_json(coverage / stage / "shards" / "shard-00" / "audit.json", audit)
            stages[stage] = {
                "schema_version": "pokezero.coverage-audit-stage.v1",
                "status": "complete",
                "stage": stage,
                "shards": 1,
                "finding_count": 0,
                "decisions_checked": 10,
                "failure_artifact_count": 0,
                "coverage_complete": True,
                "uncovered": {
                    "species": [],
                    "ability_pairs": [],
                    "moves": [],
                    "items": [],
                    "variants": [],
                },
                "audit_provenance": coverage_provenance,
            }
            write_json(coverage / stage / "summary.json", stages[stage])
            write_json(
                coverage / stage / "ledger-merged.json",
                {
                    "complete": True,
                    "uncovered": {
                        "species": [],
                        "ability_pairs": [],
                        "moves": [],
                        "items": [],
                        "variants": [],
                    },
                    "audit_provenance": aggregate_identity(coverage_provenance),
                },
            )
        party_audit = {
            "schema_version": "pokezero.deep-line-audit.v1",
            "protocol_signature_schema_version": "pokezero.protocol-signature-census.v2",
            "finding_count": 0,
            "decisions_checked": 4,
            "audit_provenance": provenance(
                command=[
                    "scripts/deep_line_audit.py",
                    "--observation-schema",
                    "v3",
                    "--random-games",
                    "0",
                    "--scenarios",
                    "--interaction-registry",
                    "--protocol-fixtures",
                    "--json",
                    "/shared/private/party.json",
                ]
            ),
        }
        write_json(coverage / "party" / "audit.json", party_audit)
        stages["party"] = {
            "schema_version": "pokezero.coverage-audit-party.v1",
            "status": "complete",
            "finding_count": 0,
            "decisions_checked": 4,
            "audit_provenance": party_audit["audit_provenance"],
        }
        write_json(coverage / "party" / "summary.json", stages["party"])
        write_json(
            coverage / "complete.json",
            {
                "schema_version": "pokezero.coverage-audit-job.v1",
                "status": "clean",
                "terminal_stage": "full",
                "image_digest": IMAGE,
                "stages": stages,
            },
        )
        return coverage

    def _silent_root(self, root: Path) -> Path:
        silent = root / "silent"
        write_json(
            silent / "complete.json",
            {
                "schema_version": "pokezero.silent-mutation-audit-job.v1",
                "status": "clean",
                "image_digest": IMAGE,
                "steps_audited": 42,
                "silent_candidate_count": 0,
                "audit_provenance": provenance(
                    command=[
                        "scripts/silent_mutation_audit.py",
                        "--observation-schema",
                        "v3",
                        "--random-games",
                        "8",
                        "--max-rounds",
                        "120",
                        "--interaction-registry",
                        "--json",
                        "/shared/private/silent.json",
                    ]
                ),
            },
        )
        write_json(
            silent / "audit.json",
            {
                "schema_version": "pokezero.silent-mutation-audit.v1",
                "steps_audited": 42,
                "silent_candidate_count": 0,
                "audit_provenance": provenance(
                    command=[
                        "scripts/silent_mutation_audit.py",
                        "--observation-schema",
                        "v3",
                        "--random-games",
                        "8",
                        "--max-rounds",
                        "120",
                        "--interaction-registry",
                        "--json",
                        "/shared/private/silent.json",
                    ]
                ),
            },
        )
        return silent

    def _collision_root(self, root: Path) -> Path:
        collision = root / "collision"
        write_json(
            collision / "audit" / "collision-audit.json",
            {
                "schema_version": "pokezero.encoding-collision-audit.v1",
                "expected_observation_schema": "pokezero.observation.v3",
                "model_input_numeric_dtype": "float32",
                "records_scanned": 100000,
                "input_group_count": 3,
                "collision_group_count": 3,
                "actionable_collision_group_count": 0,
                "audit_provenance": provenance(
                    command=[
                        "scripts/encoding_collision_audit.py",
                        "--max-decisions",
                        "100000",
                        "--out",
                        "/shared/private/collision.json",
                    ]
                ),
            },
        )
        write_json(
            collision / "controller" / "complete.json",
            {
                "schema_version": "pokezero.collision-audit-controller-complete.v1",
                "status": "clean",
                "image_digest": IMAGE,
                "audit_path": "/shared/private/collision.json",
                "audit_sha256": hashlib.sha256(
                    (collision / "audit" / "collision-audit.json").read_bytes()
                ).hexdigest(),
            },
        )
        return collision

    def _inventory_root(self, root: Path) -> Path:
        inventory = root / "inventory"
        write_json(
            inventory / "inventory.json",
            {
                "schema_version": "pokezero.protocol-emission-inventory.v2",
                "engine_emittable": {"tag_count": 8},
                "consumer_dispatch": {"tag_count": 7},
                "differential": {
                    "observed_but_unconsumed": [],
                    "observed_but_unconsumed_unclassified": [],
                    "observed_signatures_without_semantic_coverage": [],
                    "emittable_but_unobserved": [],
                    "consumer_not_emittable": [],
                },
                "observed": {
                    "tag_count": 7,
                    "audit_provenance": [
                        {
                            "path": "/shared/private/observed-audit.json",
                            "audit_provenance": provenance(
                                command=["scripts/deep_line_audit.py", "--json", "/shared/private/observed.json"]
                            ),
                        }
                    ],
                },
                "audit_provenance": provenance(
                    command=["scripts/protocol_emission_inventory.py", "--out", "/shared/private/inventory.json"]
                ),
            },
        )
        write_json(
            inventory / "complete.json",
            {
                "schema_version": "pokezero.protocol-inventory-job.v1",
                "status": "clean",
                "image_digest": IMAGE,
                "observed_tag_count": 7,
                "observed_but_unconsumed_count": 0,
                "observed_but_unconsumed_unclassified_count": 0,
                "inventory_sha256": hashlib.sha256(
                    (inventory / "inventory.json").read_bytes()
                ).hexdigest(),
            },
        )
        return inventory

    def test_publishes_whitelisted_provenance_without_private_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "public" / "summary.json"
            payload = PUBLISHER.publish(
                coverage_root=self._coverage_root(root),
                silent_root=self._silent_root(root),
                collision_root=self._collision_root(root),
                inventory_root=self._inventory_root(root),
                output=output,
            )
            serialized = output.read_text(encoding="utf-8")

        self.assertEqual(payload["schema_version"], "pokezero.v3-audit-public-evidence.v2")
        self.assertEqual(payload["provenance"]["image_digest"], "sha256:" + "a" * 64)
        self.assertEqual(payload["layers"]["encoding_collision"]["records_scanned"], 100000)
        self.assertEqual(payload["layers"]["encoding_collision"]["input_group_count"], 3)
        self.assertNotIn("registry.example.invalid", serialized)
        self.assertNotIn("/shared/private", serialized)
        self.assertIn("<artifact-path>", serialized)

    def test_rejects_mixed_provenance_before_writing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            collision = self._collision_root(root)
            audit_path = collision / "audit" / "collision-audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["audit_provenance"]["showdown_source_hash"] = "d" * 16
            write_json(audit_path, audit)
            complete_path = collision / "controller" / "complete.json"
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            complete["audit_sha256"] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
            write_json(complete_path, complete)
            output = root / "public" / "summary.json"

            with self.assertRaisesRegex(ValueError, "mixed audit provenance"):
                PUBLISHER.publish(
                    coverage_root=self._coverage_root(root),
                    silent_root=self._silent_root(root),
                    collision_root=collision,
                    inventory_root=self._inventory_root(root),
                    output=output,
                )
            self.assertFalse(output.exists())

    def test_rejects_artifact_missing_execution_scope_before_writing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            silent = self._silent_root(root)
            audit_path = silent / "audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            del audit["audit_provenance"]["execution_scope"]
            write_json(audit_path, audit)
            output = root / "public" / "summary.json"

            with self.assertRaisesRegex(ValueError, "missing execution scope"):
                PUBLISHER.publish(
                    coverage_root=self._coverage_root(root),
                    silent_root=silent,
                    collision_root=self._collision_root(root),
                    inventory_root=self._inventory_root(root),
                    output=output,
                )
            self.assertFalse(output.exists())

    def test_rejects_tampered_collision_artifact_before_writing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            collision = self._collision_root(root)
            audit_path = collision / "audit" / "collision-audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["records_scanned"] = 99999
            write_json(audit_path, audit)
            output = root / "public" / "summary.json"

            with self.assertRaisesRegex(ValueError, "does not authenticate"):
                PUBLISHER.publish(
                    coverage_root=self._coverage_root(root),
                    silent_root=self._silent_root(root),
                    collision_root=collision,
                    inventory_root=self._inventory_root(root),
                    output=output,
                )
            self.assertFalse(output.exists())

    def test_rejects_collision_report_without_model_input_group_count(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            collision = self._collision_root(root)
            audit_path = collision / "audit" / "collision-audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit.pop("input_group_count")
            write_json(audit_path, audit)
            complete_path = collision / "controller" / "complete.json"
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            complete["audit_sha256"] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
            write_json(complete_path, complete)

            with self.assertRaisesRegex(ValueError, "input_group_count"):
                PUBLISHER.publish(
                    coverage_root=self._coverage_root(root),
                    silent_root=self._silent_root(root),
                    collision_root=collision,
                    inventory_root=self._inventory_root(root),
                    output=root / "public" / "summary.json",
                )

    def test_rejects_coverage_summary_that_disagrees_with_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            coverage = self._coverage_root(root)
            summary_path = coverage / "depth" / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["finding_count"] = 1
            write_json(summary_path, summary)
            complete_path = coverage / "complete.json"
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            complete["stages"]["depth"] = summary
            write_json(complete_path, complete)

            with self.assertRaisesRegex(ValueError, "differs from its constituent"):
                PUBLISHER.publish(
                    coverage_root=coverage,
                    silent_root=self._silent_root(root),
                    collision_root=self._collision_root(root),
                    inventory_root=self._inventory_root(root),
                    output=root / "public" / "summary.json",
                )

    def test_rejects_depth_stage_without_the_bounded_depth_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            coverage = self._coverage_root(root)
            audit_path = coverage / "depth" / "shards" / "shard-00" / "audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            command = audit["audit_provenance"]["command"]
            index = command.index("--depth-rounds")
            del command[index : index + 2]
            audit["coverage_execution"]["depth_rounds"] = 0
            write_json(audit_path, audit)
            summary_path = coverage / "depth" / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["audit_provenance"] = audit["audit_provenance"]
            write_json(summary_path, summary)
            complete_path = coverage / "complete.json"
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            complete["stages"]["depth"] = summary
            write_json(complete_path, complete)

            with self.assertRaisesRegex(ValueError, "--depth-rounds"):
                PUBLISHER.publish(
                    coverage_root=coverage,
                    silent_root=self._silent_root(root),
                    collision_root=self._collision_root(root),
                    inventory_root=self._inventory_root(root),
                    output=root / "public" / "summary.json",
                )

    def test_rejects_incomplete_party_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            coverage = self._coverage_root(root)
            summary_path = coverage / "party" / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["status"] = "running"
            write_json(summary_path, summary)
            complete_path = coverage / "complete.json"
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            complete["stages"]["party"] = summary
            write_json(complete_path, complete)

            with self.assertRaisesRegex(ValueError, "party summary is not complete"):
                PUBLISHER.publish(
                    coverage_root=coverage,
                    silent_root=self._silent_root(root),
                    collision_root=self._collision_root(root),
                    inventory_root=self._inventory_root(root),
                    output=root / "public" / "summary.json",
                )

    def test_rejects_tampered_inventory_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inventory = self._inventory_root(root)
            inventory_path = inventory / "inventory.json"
            payload = json.loads(inventory_path.read_text(encoding="utf-8"))
            payload["observed"]["tag_count"] = 8
            write_json(inventory_path, payload)

            with self.assertRaisesRegex(ValueError, "does not authenticate"):
                PUBLISHER.publish(
                    coverage_root=self._coverage_root(root),
                    silent_root=self._silent_root(root),
                    collision_root=self._collision_root(root),
                    inventory_root=inventory,
                    output=root / "public" / "summary.json",
                )

    def test_rejects_non_numeric_seed_range_before_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            coverage = self._coverage_root(root)
            audit_path = coverage / "static" / "shards" / "shard-00" / "audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["audit_provenance"]["seed_range"] = {"start": "/shared/private", "end": 3}
            write_json(audit_path, audit)

            with self.assertRaisesRegex(ValueError, "seed start"):
                PUBLISHER.publish(
                    coverage_root=coverage,
                    silent_root=self._silent_root(root),
                    collision_root=self._collision_root(root),
                    inventory_root=self._inventory_root(root),
                    output=root / "public" / "summary.json",
                )

    def test_rejects_private_command_flag_before_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            coverage = self._coverage_root(root)
            audit_path = coverage / "static" / "shards" / "shard-00" / "audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["audit_provenance"]["command"].extend(("--namespace", "private-cluster"))
            write_json(audit_path, audit)

            with self.assertRaisesRegex(ValueError, "unrecognized flag"):
                PUBLISHER.publish(
                    coverage_root=coverage,
                    silent_root=self._silent_root(root),
                    collision_root=self._collision_root(root),
                    inventory_root=self._inventory_root(root),
                    output=root / "public" / "summary.json",
                )

    def test_rejects_unsafe_uncovered_atom_before_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            coverage = self._coverage_root(root)
            summary_path = coverage / "static" / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["uncovered"]["moves"] = ["/shared/private"]
            write_json(summary_path, summary)
            complete_path = coverage / "complete.json"
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            complete["stages"]["static"] = summary
            write_json(complete_path, complete)

            with self.assertRaisesRegex(ValueError, "unsafe uncovered atoms"):
                PUBLISHER.publish(
                    coverage_root=coverage,
                    silent_root=self._silent_root(root),
                    collision_root=self._collision_root(root),
                    inventory_root=self._inventory_root(root),
                    output=root / "public" / "summary.json",
                )

    def test_rejects_clean_terminal_with_nonzero_candidate_counter(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            silent = self._silent_root(root)
            audit_path = silent / "audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["silent_candidate_count"] = 1
            write_json(audit_path, audit)
            complete_path = silent / "complete.json"
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            complete["silent_candidate_count"] = 1
            write_json(complete_path, complete)

            with self.assertRaisesRegex(ValueError, "disagrees with its validated counters"):
                PUBLISHER.publish(
                    coverage_root=self._coverage_root(root),
                    silent_root=silent,
                    collision_root=self._collision_root(root),
                    inventory_root=self._inventory_root(root),
                    output=root / "public" / "summary.json",
                )

    def test_rejects_zero_work_silent_mutation_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            silent = self._silent_root(root)
            audit_path = silent / "audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["steps_audited"] = 0
            write_json(audit_path, audit)
            complete_path = silent / "complete.json"
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            complete["steps_audited"] = 0
            write_json(complete_path, complete)

            with self.assertRaisesRegex(ValueError, "checked no state transitions"):
                PUBLISHER.publish(
                    coverage_root=self._coverage_root(root),
                    silent_root=silent,
                    collision_root=self._collision_root(root),
                    inventory_root=self._inventory_root(root),
                    output=root / "public" / "summary.json",
                )

    def test_rejects_collision_audit_below_the_fixed_100k_floor(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            collision = self._collision_root(root)
            audit_path = collision / "audit" / "collision-audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["records_scanned"] = 99_999
            write_json(audit_path, audit)
            complete_path = collision / "controller" / "complete.json"
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            complete["audit_sha256"] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
            write_json(complete_path, complete)

            with self.assertRaisesRegex(ValueError, "99999 < 100000"):
                PUBLISHER.publish(
                    coverage_root=self._coverage_root(root),
                    silent_root=self._silent_root(root),
                    collision_root=collision,
                    inventory_root=self._inventory_root(root),
                    output=root / "public" / "summary.json",
                )

    def test_rejects_clean_inventory_with_semantic_coverage_gap(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inventory = self._inventory_root(root)
            inventory_path = inventory / "inventory.json"
            payload = json.loads(inventory_path.read_text(encoding="utf-8"))
            payload["differential"]["observed_signatures_without_semantic_coverage"] = [
                {"signature": "-activate|p1a: Test|move: Example"}
            ]
            write_json(inventory_path, payload)
            complete_path = inventory / "complete.json"
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            complete["inventory_sha256"] = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
            write_json(complete_path, complete)

            with self.assertRaisesRegex(ValueError, "disagrees with its validated counters"):
                PUBLISHER.publish(
                    coverage_root=self._coverage_root(root),
                    silent_root=self._silent_root(root),
                    collision_root=self._collision_root(root),
                    inventory_root=inventory,
                    output=root / "public" / "summary.json",
                )

    def test_rejects_clean_inventory_with_unresolved_emission_or_consumer_deltas(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inventory = self._inventory_root(root)
            inventory_path = inventory / "inventory.json"
            payload = json.loads(inventory_path.read_text(encoding="utf-8"))
            payload["differential"]["observed_but_unconsumed"] = [{"tag": "-singleturn", "count": 4}]
            payload["differential"]["emittable_but_unobserved"] = ["-fail"]
            payload["differential"]["consumer_not_emittable"] = ["switch"]
            write_json(inventory_path, payload)
            complete_path = inventory / "complete.json"
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            complete["observed_but_unconsumed_count"] = 1
            complete["inventory_sha256"] = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
            write_json(complete_path, complete)

            with self.assertRaisesRegex(ValueError, "disagrees with its validated counters"):
                PUBLISHER.publish(
                    coverage_root=self._coverage_root(root),
                    silent_root=self._silent_root(root),
                    collision_root=self._collision_root(root),
                    inventory_root=inventory,
                    output=root / "public" / "summary.json",
                )

    def test_rejects_mixed_observed_census_provenance_before_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inventory = self._inventory_root(root)
            inventory_path = inventory / "inventory.json"
            payload = json.loads(inventory_path.read_text(encoding="utf-8"))
            payload["observed"]["audit_provenance"][0]["audit_provenance"]["showdown_source_hash"] = "d" * 16
            write_json(inventory_path, payload)
            complete_path = inventory / "complete.json"
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            complete["inventory_sha256"] = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
            write_json(complete_path, complete)

            with self.assertRaisesRegex(ValueError, "mixed audit provenance"):
                PUBLISHER.publish(
                    coverage_root=self._coverage_root(root),
                    silent_root=self._silent_root(root),
                    collision_root=self._collision_root(root),
                    inventory_root=inventory,
                    output=root / "public" / "summary.json",
                )

    def test_rejects_missing_coverage_failure_counter_before_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            coverage = self._coverage_root(root)
            audit_path = coverage / "static" / "shards" / "shard-00" / "audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit.pop("coverage_execution")
            write_json(audit_path, audit)

            with self.assertRaisesRegex(ValueError, "no coverage execution ledger"):
                PUBLISHER.publish(
                    coverage_root=coverage,
                    silent_root=self._silent_root(root),
                    collision_root=self._collision_root(root),
                    inventory_root=self._inventory_root(root),
                    output=root / "public" / "summary.json",
                )

    def test_rejects_missing_coverage_summary_failure_counter_before_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            coverage = self._coverage_root(root)
            summary_path = coverage / "depth" / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary.pop("failure_artifact_count")
            write_json(summary_path, summary)
            complete_path = coverage / "complete.json"
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            complete["stages"]["depth"] = summary
            write_json(complete_path, complete)

            with self.assertRaisesRegex(ValueError, "depth failure_artifact_count"):
                PUBLISHER.publish(
                    coverage_root=coverage,
                    silent_root=self._silent_root(root),
                    collision_root=self._collision_root(root),
                    inventory_root=self._inventory_root(root),
                    output=root / "public" / "summary.json",
                )


if __name__ == "__main__":
    unittest.main()
