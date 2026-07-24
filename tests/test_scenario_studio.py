from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT
from pokezero.scenario_studio.catalog import ScenarioCatalog, validate_scenario
from pokezero.scenario_studio.domain import EndgameScenario, ScenarioValidationError
from pokezero.scenario_studio.server import ScenarioStudioHTTPServer
from pokezero.scenario_studio.service import ScenarioStudioService
from pokezero.scenario_studio.storage import ScenarioRepository


SHOWDOWN_READY = (DEFAULT_SHOWDOWN_ROOT / "dist" / "sim" / "index.js").is_file() and shutil.which("node")
ENDGAME_SCENARIO_DIR = Path(__file__).resolve().parents[1] / "scenarios" / "endgame"


def scenario_payload(catalog: ScenarioCatalog) -> dict[str, object]:
    """A small source-composed scenario whose values are all catalog-derived."""

    p1 = catalog.pokemon_for_variant("arcanine-2-variant-3")
    p2 = catalog.pokemon_for_variant("blastoise-3-variant-16")
    p1 = replace(p1, current_hp=max(1, p1.max_hp // 3))
    p2_moves = list(p2.moves)
    p2_moves[0] = replace(p2_moves[0], pp=max(0, p2_moves[0].max_pp - 2))
    p2 = replace(p2, moves=tuple(p2_moves))
    return {
        "schema_version": "endgame-scenario-v1",
        "scenario_id": "source-composed-smoke",
        "title": "Source-composed smoke",
        "description": "A small scenario used to exercise the authoring contract.",
        "tags": ["smoke", "priority"],
        "format_id": "gen3customgame",
        "source_format_id": "gen3randombattle",
        "seed": 1701,
        "provenance": {"randbat_source_hash": catalog.source_hash, "replay_proven": False},
        "knowledge_mode": "fully_revealed",
        "perspective": "p1",
        "side_to_move": "p1",
        "teams": {
            "p1": {
                "construction_mode": "source-composed",
                "generated_team_seed": None,
                "active_slot": 0,
                "pokemon": [p1.to_payload()],
            },
            "p2": {
                "construction_mode": "source-composed",
                "generated_team_seed": None,
                "active_slot": 0,
                "pokemon": [p2.to_payload()],
            },
        },
        "objective": {
            "kind": "best_move",
            "expected_root_actions": ["move extremespeed"],
            "principal_variation": [],
            "max_plies": 1,
            "verification": {"status": "unverified", "engine": None, "artifact": None},
        },
        "author_notes": "Synthetic fully revealed authoring fixture, not a reachability claim.",
    }


class ScenarioRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.catalog = ScenarioCatalog(showdown_root=DEFAULT_SHOWDOWN_ROOT)
        self.payload = scenario_payload(self.catalog)
        self.scenario = EndgameScenario.from_payload(self.payload)
        self.repository = ScenarioRepository(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_canonical_round_trip_is_stable(self) -> None:
        encoded = self.scenario.canonical_json()

        restored = EndgameScenario.from_payload_json(encoded)

        self.assertEqual(restored.canonical_json(), encoded)
        self.assertEqual(json.loads(encoded)["schema_version"], "endgame-scenario-v1")

    def test_repository_rejects_traversal_and_preserves_prior_file_on_failed_overwrite(self) -> None:
        self.repository.save("source-composed-smoke", self.scenario)
        original = (Path(self.temp_dir.name) / "source-composed-smoke.json").read_text(encoding="utf-8")

        with self.assertRaises(ScenarioValidationError):
            self.repository.load("../outside")
        with mock.patch("pokezero.scenario_studio.storage.os.replace", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.repository.save("source-composed-smoke", replace(self.scenario, title="new title"))

        self.assertEqual((Path(self.temp_dir.name) / "source-composed-smoke.json").read_text(encoding="utf-8"), original)

    def test_catalog_rejects_source_hash_mismatch(self) -> None:
        invalid = dict(self.payload)
        invalid["provenance"] = {"randbat_source_hash": "wrong", "replay_proven": False}

        with self.assertRaisesRegex(ScenarioValidationError, "does not match current"):
            validate_scenario(EndgameScenario.from_payload(invalid), self.catalog)

    def test_domain_rejects_unpinned_scenario_formats(self) -> None:
        invalid = dict(self.payload)
        invalid["format_id"] = "gen3randombattle"

        with self.assertRaisesRegex(ScenarioValidationError, "gen3customgame"):
            EndgameScenario.from_payload(invalid)


@unittest.skipUnless(SHOWDOWN_READY, "requires node and built Pokemon Showdown checkout")
class ScenarioStudioIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.service = ScenarioStudioService(showdown_root=DEFAULT_SHOWDOWN_ROOT, scenario_dir=self.temp_dir.name)
        self.payload = scenario_payload(self.service.catalog)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_generated_side_with_hidden_power_culling_validates_through_showdown(self) -> None:
        payload = dict(self.payload)
        payload["teams"] = {
            "p1": self.service.generate_team(seed=1701),
            "p2": self.service.generate_team(seed=1702),
        }

        result = self.service.validate_payload(payload)

        unown = next(pokemon for pokemon in result["scenario"]["teams"]["p1"]["pokemon"] if pokemon["species"] == "Unown")
        self.assertEqual([move["id"] for move in unown["moves"]], ["hiddenpowerbug"])
        self.assertTrue(result["validation"]["set_valid"])
        self.assertTrue(result["validation"]["state_consistent"])
        self.assertTrue(result["legal_actions"]["p1"])

    def test_committed_endgame_suite_contains_ten_distinct_showdown_materializations(self) -> None:
        paths = sorted(ENDGAME_SCENARIO_DIR.glob("*.json"))
        self.assertGreaterEqual(len(paths), 10)
        scenario_ids: set[str] = set()
        tags: set[str] = set()
        for path in paths:
            raw = path.read_text(encoding="utf-8")
            scenario_model = EndgameScenario.from_payload_json(raw)
            self.assertEqual(raw, scenario_model.canonical_json(), path.name)
            result = self.service.validate_payload(scenario_model.to_payload())
            scenario = result["scenario"]
            scenario_ids.add(scenario["scenario_id"])
            tags.update(scenario["tags"])
            self.assertTrue(result["validation"]["set_valid"], path.name)
            self.assertTrue(result["validation"]["state_consistent"], path.name)
            self.assertTrue(result["legal_actions"]["p1"], path.name)
        self.assertEqual(len(scenario_ids), len(paths))
        self.assertGreaterEqual(len(tags), 10)

    def test_save_load_and_root_evaluation_keep_source_and_checkpoint_provenance(self) -> None:
        saved = self.service.save("source-composed-smoke", self.payload)
        self.assertEqual(self.service.load("source-composed-smoke")["scenario"], saved["scenario"])
        model_config = SimpleNamespace(
            policy_id="scenario-test-policy",
            observation_schema_version="test-schema",
            transition_token_budget=32,
        )
        result = SimpleNamespace(model_config=model_config)

        def action_priors(*, observations, **_kwargs):
            return tuple(float(index) for index in range(len(observations[0].legal_action_mask)))

        with (
            mock.patch("pokezero.scenario_studio.service.load_transformer_checkpoint", return_value=(object(), result)),
            mock.patch("pokezero.scenario_studio.service.feature_masks_from_model_config", return_value=object()),
            mock.patch("pokezero.scenario_studio.service.observation_spec_from_model_config", return_value=object()),
            mock.patch(
                "pokezero.scenario_studio.service.env_config_with_checkpoint_masks",
                side_effect=lambda config, *_args, **_kwargs: config,
            ),
            mock.patch("pokezero.scenario_studio.service.evaluate_transformer_action_priors", side_effect=action_priors),
        ):
            evaluation = self.service.evaluate_root(self.payload, checkpoint_path="scenario-test.pt")

        self.assertEqual(evaluation["checkpoint"]["policy_id"], "scenario-test-policy")
        self.assertEqual(evaluation["checkpoint"]["transition_token_budget"], 32)
        self.assertTrue(evaluation["synthetic_history"])
        self.assertEqual(evaluation["actions"][0]["rank"], 1)
        self.assertGreater(evaluation["actions"][0]["probability"], evaluation["actions"][-1]["probability"])

    def test_loopback_api_serves_editor_and_returns_structured_validation_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "loopback"):
            ScenarioStudioHTTPServer(("0.0.0.0", 0), self.service)
        server = ScenarioStudioHTTPServer(("127.0.0.1", 0), self.service)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            with urlopen(f"{base_url}/", timeout=10) as response:
                page = response.read().decode("utf-8")
            self.assertIn("Endgame Scenario Studio", page)

            with urlopen(f"{base_url}/api/catalog", timeout=10) as response:
                catalog = json.loads(response.read())
            self.assertEqual(catalog["source_hash"], self.service.catalog.source_hash)

            request = Request(
                f"{base_url}/api/scenarios/source-composed-smoke",
                data=json.dumps({"scenario": self.payload}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="PUT",
            )
            with urlopen(request, timeout=20) as response:
                saved = json.loads(response.read())
            self.assertEqual(saved["slug"], "source-composed-smoke")

            invalid = Request(
                f"{base_url}/api/validate",
                data=b'{"scenario": {}}',
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(HTTPError) as raised:
                urlopen(invalid, timeout=10)
            with raised.exception as error:
                self.assertEqual(error.code, 400)
                body = json.loads(error.read())
                self.assertEqual(body["error"]["code"], "validation_error")
                self.assertNotIn("Traceback", body["error"]["message"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=10)
