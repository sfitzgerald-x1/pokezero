from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from pokezero.protocol_emission_inventory import (
    _SIGNATURE_COVERAGE,
    _signature_coverage,
    build_protocol_inventory,
    discover_consumer_dispatches,
    discover_engine_emissions,
    load_observed_audit_provenance,
    load_observed_signatures,
    require_expected_observed_audit_provenance,
)


class ProtocolEmissionInventoryTests(unittest.TestCase):
    def test_direct_signature_registry_matches_discovered_consumer_tags(self) -> None:
        consumers = discover_consumer_dispatches(Path(__file__).resolve().parents[1])
        for coverage in _SIGNATURE_COVERAGE:
            if coverage.coverage == "direct":
                self.assertIn(coverage.signature.split(":", 1)[0], consumers)

    def test_cli_allows_static_inventory_without_observed_audits(self) -> None:
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "protocol_emission_inventory.py"
        spec = importlib.util.spec_from_file_location("protocol_emission_inventory_cli", script_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        payload = {
            "observed": {"audit_provenance": [], "tag_count": 0},
            "differential": {
                "observed_but_unconsumed": [],
                "observed_signatures_without_direct_consumer": [],
                "observed_signatures_without_semantic_coverage": [],
            },
            "engine_emittable": {"tag_count": 1},
            "consumer_dispatch": {"tag_count": 1},
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "inventory.json"
            with (
                patch.object(module, "load_gen3_randbat_source_cached", return_value=SimpleNamespace(
                    metadata=SimpleNamespace(source_hash="source-hash")
                )),
                patch.object(module, "build_protocol_inventory", return_value=payload),
                patch.object(module, "public_repo_commit", return_value="a" * 40),
            ):
                self.assertEqual(
                    module.main((
                        "--showdown-root", str(temporary),
                        "--observation-schema", "v3",
                        "--out", str(output),
                    )),
                    0,
                )
            self.assertTrue(output.is_file())
            written = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                written["audit_provenance"]["execution_scope"],
                {"input_audit_count": 0, "seed_range": None, "shard": None},
            )

    def _fixture_roots(self, root: Path) -> tuple[Path, Path]:
        showdown = root / "showdown"
        public = root / "public"
        (showdown / "sim").mkdir(parents=True)
        (showdown / "data" / "mods" / "gen3").mkdir(parents=True)
        (showdown / "data").mkdir(exist_ok=True)
        (showdown / "sim" / "battle.ts").write_text(
            "this.add('move', pokemon, move);\nthis.battle.add('cant', pokemon, 'slp');\n",
            encoding="utf-8",
        )
        (showdown / "data" / "mods" / "gen3" / "moves.ts").write_text(
            "this.add('-fail', pokemon);\nthis.add('-activate', pokemon, 'move: Bide');\n",
            encoding="utf-8",
        )
        (showdown / "data" / "moves.ts").write_text(
            "// this.add('-commented-out', pokemon);\n"
            "const sample = \"this.add('-inside-string', pokemon)\";\n"
            "this.add('-miss', pokemon);\n",
            encoding="utf-8",
        )
        for relative, text in {
            "src/pokezero/showdown.py": "if event_type == 'move':\n    pass\n",
            "src/pokezero/transitions.py": "if event_type in {'-fail', 'cant'}:\n    pass\n",
            "src/pokezero/belief.py": "if event.event_type == '-activate':\n    pass\n",
            "src/pokezero/public_action_capture.py": "if event_type == 'switch':\n    pass\n",
            "src/pokezero/turn_merged.py": "if event_type == '-fail':\n    pass\n",
        }.items():
            path = public / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return showdown, public

    def test_engine_inventory_records_gen3_and_shared_source_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            showdown, _public = self._fixture_roots(Path(temporary))
            emissions = discover_engine_emissions(showdown)

        self.assertEqual(set(emissions), {"-activate", "-fail", "-miss", "cant", "move"})
        self.assertEqual(emissions["-fail"][0].evidence, "gen3-module")
        self.assertEqual(emissions["move"][0].evidence, "shared-simulator")
        self.assertNotIn("-commented-out", emissions)
        self.assertNotIn("-inside-string", emissions)

    def test_canonical_differential_detects_subtype_mismatch_and_dynamic_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            showdown, public = self._fixture_roots(Path(temporary))
            report = build_protocol_inventory(showdown_root=showdown, public_root=public)

        engine_signatures = {
            row["signature"] for row in report["engine_emittable"]["canonical_signatures"]
        }
        self.assertIn("-activate:bide", engine_signatures)
        self.assertIn("cant:slp", engine_signatures)
        self.assertNotIn("move", engine_signatures)
        self.assertFalse(report["engine_emittable"]["canonical_complete"])
        self.assertEqual(
            [
                (row["tag"], row["reason"])
                for row in report["engine_emittable"]["unresolved_emissions"]
            ],
            [("move", "dynamic-payload")],
        )

        consumer_signatures = {
            row["signature"]: row["kind"]
            for row in report["consumer_dispatch"]["canonical_signatures"]
        }
        self.assertEqual(consumer_signatures["cant:*"], "pattern")
        self.assertEqual(consumer_signatures["-activate:protect"], "exact")
        mismatch = report["differential"]["emittable_signatures_without_consumer"]
        self.assertIn("-activate:bide", mismatch)
        self.assertNotIn("cant:slp", mismatch)

    def test_dynamic_tag_is_reported_without_source_expression(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            showdown, public = self._fixture_roots(root)
            (showdown / "sim" / "dynamic.ts").write_text(
                "this.add(eventType, pokemon, privatePayload);\n",
                encoding="utf-8",
            )
            report = build_protocol_inventory(showdown_root=showdown, public_root=public)

        dynamic = [
            row
            for row in report["engine_emittable"]["unresolved_emissions"]
            if row["reason"] == "dynamic-tag"
        ]
        self.assertEqual(len(dynamic), 1)
        self.assertIsNone(dynamic[0]["tag"])
        self.assertNotIn("expression", dynamic[0])
        self.assertEqual(dynamic[0]["source_location"]["path"], "sim/dynamic.ts")

    def test_differential_keeps_observed_frequency_and_consumer_scope_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            showdown, public = self._fixture_roots(root)
            observed = root / "observed.json"
            observed.write_text(
                json.dumps({
                    "protocol_signatures": {
                        "-activate:protect": 2,
                        "-fieldactivate:perishsong": 1,
                        "-mystery": 5,
                        "-mustrecharge": 1,
                        "-singleturn:protect": 3,
                    },
                    "protocol_signature_schema_version": "pokezero.protocol-signature-census.v2",
                    "audit_provenance": {
                        "observation_schema": "pokezero.observation.v3",
                        "showdown_source_hash": "fixture-source",
                        "image_digest": "fixture-image",
                    },
                }),
                encoding="utf-8",
            )
            report = build_protocol_inventory(
                showdown_root=showdown,
                public_root=public,
                observed_audits=(observed,),
            )

        self.assertEqual(report["engine_emittable"]["tag_count"], 5)
        self.assertEqual(report["consumer_dispatch"]["tag_count"], 5)
        self.assertEqual(report["observed"]["signature_count"], 5)
        self.assertEqual(
            report["differential"]["observed_but_unconsumed"],
            [
                {"tag": "-mystery", "count": 5},
                {"tag": "-singleturn", "count": 3},
                {"tag": "-fieldactivate", "count": 1},
                {"tag": "-mustrecharge", "count": 1},
            ],
        )
        coverage = {row["signature"]: row for row in report["observed"]["signature_coverage"]}
        self.assertEqual(coverage["-activate:protect"]["coverage"], "direct")
        self.assertEqual(coverage["-singleturn:protect"]["coverage"], "semantic-alias")
        self.assertEqual(coverage["-fieldactivate:perishsong"]["coverage"], "semantic-alias")
        self.assertEqual(coverage["-mustrecharge"]["coverage"], "semantic-alias")
        self.assertEqual(coverage["-mystery"]["coverage"], "unclassified")
        self.assertEqual(
            [row["signature"] for row in report["differential"]["observed_signatures_without_semantic_coverage"]],
            ["-mystery"],
        )
        self.assertEqual(
            report["differential"]["observed_but_unconsumed_unclassified"],
            [{
                "signature": "-mystery",
                "tag": "-mystery",
                "count": 5,
                "sources": [str(observed)],
                "coverage": "unclassified",
            }],
        )
        self.assertIn("-miss", report["differential"]["emittable_but_unobserved"])
        self.assertIn("switch", report["differential"]["consumer_not_emittable"])
        self.assertEqual(report["observed"]["audit_provenance"][0]["audit_provenance"]["image_digest"], "fixture-image")

    def test_signature_coverage_records_recharge_as_a_semantic_alias(self) -> None:
        self.assertEqual(_signature_coverage("-mustrecharge").coverage, "semantic-alias")
        self.assertEqual(_signature_coverage("move:protect").coverage, "direct")
        self.assertEqual(_signature_coverage("-start:perish3").coverage, "direct")

    def test_inventory_keeps_focus_punch_charge_as_an_unclassified_o_minus_c_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            showdown, public = self._fixture_roots(root)
            observed = root / "observed.json"
            observed.write_text(
                json.dumps(
                    {
                        "protocol_signatures": {"debug": 4, "-singleturn:focuspunch": 2},
                        "protocol_signature_schema_version": "pokezero.protocol-signature-census.v2",
                        "audit_provenance": {
                            "observation_schema": "pokezero.observation.v3",
                            "showdown_source_hash": "fixture-source",
                            "image_digest": "fixture-image",
                        },
                    }
                ),
                encoding="utf-8",
            )
            report = build_protocol_inventory(
                showdown_root=showdown,
                public_root=public,
                observed_audits=(observed,),
            )

        coverage = {row["signature"]: row for row in report["observed"]["signature_coverage"]}
        self.assertEqual(coverage["debug"]["coverage"], "non-model")
        self.assertEqual(coverage["-singleturn:focuspunch"]["coverage"], "unclassified")
        self.assertEqual(
            report["differential"]["observed_but_unconsumed_unclassified"],
            [{
                "signature": "-singleturn:focuspunch",
                "tag": "-singleturn",
                "count": 2,
                "sources": [str(observed)],
                "coverage": "unclassified",
            }],
        )

    def test_consumer_discovery_only_counts_event_type_comparisons(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _showdown, public = self._fixture_roots(Path(temporary))
            path = public / "src" / "pokezero" / "showdown.py"
            path.write_text(
                "label = '-not-a-dispatch'\nif event_type == 'move':\n    pass\n",
                encoding="utf-8",
            )
            consumers = discover_consumer_dispatches(public)

        self.assertIn("move", consumers)
        self.assertNotIn("-not-a-dispatch", consumers)

    def test_observed_signature_loader_rejects_negative_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.json"
            path.write_text(json.dumps({"protocol_signatures": {"-fail": -1}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "negative count"):
                load_observed_signatures((path,))

    def test_observed_provenance_loader_rejects_missing_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "missing.json"
            path.write_text(json.dumps({"protocol_signatures": {}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing audit_provenance"):
                load_observed_audit_provenance((path,))

    def test_observed_provenance_loader_rejects_legacy_signature_spelling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.json"
            path.write_text(
                json.dumps({
                    "protocol_signatures": {"-activate:moveprotect": 1},
                    "audit_provenance": {"observation_schema": "pokezero.observation.v3"},
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "protocol signature schema"):
                load_observed_audit_provenance((path,))

    def test_observed_provenance_requires_current_full_identity(self) -> None:
        provenance = {
            "public_repo_commit": "a" * 40,
            "showdown_source_hash": "source-hash",
            "observation_schema": "pokezero.observation.v3",
            "image_digest": "example.invalid/pokezero@sha256:" + ("b" * 64),
        }
        entries = ({"audit_provenance": provenance},)

        self.assertEqual(
            require_expected_observed_audit_provenance(entries, expected=provenance), provenance
        )
        with self.assertRaisesRegex(ValueError, "differs from this inventory run"):
            require_expected_observed_audit_provenance(
                entries,
                expected={**provenance, "image_digest": "example.invalid/pokezero@sha256:" + ("c" * 64)},
            )


if __name__ == "__main__":
    unittest.main()
