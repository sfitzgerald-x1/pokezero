from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from pokezero.protocol_emission_inventory import (
    build_protocol_inventory,
    discover_consumer_dispatches,
    discover_engine_emissions,
    load_observed_audit_provenance,
    load_observed_signatures,
)


class ProtocolEmissionInventoryTests(unittest.TestCase):
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
            "// this.add('-commented-out', pokemon);\nthis.add('-miss', pokemon);\n",
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
                        "-singleturn:protect": 3,
                    },
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
        self.assertEqual(report["observed"]["signature_count"], 4)
        self.assertEqual(
            report["differential"]["observed_but_unconsumed"],
            [
                {"tag": "-mystery", "count": 5},
                {"tag": "-singleturn", "count": 3},
                {"tag": "-fieldactivate", "count": 1},
            ],
        )
        coverage = {row["signature"]: row for row in report["observed"]["signature_coverage"]}
        self.assertEqual(coverage["-activate:protect"]["coverage"], "direct")
        self.assertEqual(coverage["-singleturn:protect"]["coverage"], "semantic-alias")
        self.assertEqual(coverage["-fieldactivate:perishsong"]["coverage"], "semantic-alias")
        self.assertEqual(coverage["-mystery"]["coverage"], "unclassified")
        self.assertEqual(
            [row["signature"] for row in report["differential"]["observed_signatures_without_semantic_coverage"]],
            ["-mystery"],
        )
        self.assertIn("-miss", report["differential"]["emittable_but_unobserved"])
        self.assertIn("switch", report["differential"]["consumer_not_emittable"])
        self.assertEqual(report["observed"]["audit_provenance"][0]["audit_provenance"]["image_digest"], "fixture-image")

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


if __name__ == "__main__":
    unittest.main()
