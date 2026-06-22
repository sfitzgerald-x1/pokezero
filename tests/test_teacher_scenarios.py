import os
from pathlib import Path
import random
import unittest

from pokezero.dex import load_showdown_dex, showdown_dex_from_payload
from pokezero.policy import PolicyDecision, ScriptedTeacherPolicy
from pokezero.teacher_scenarios import (
    TEACHER_SCENARIO_PREFLIGHT_SCHEMA_VERSION,
    run_teacher_scenario_preflight,
    teacher_scenario_ids,
)


class TeacherScenarioPreflightTest(unittest.TestCase):
    def test_default_scenarios_pass_against_scripted_teacher(self) -> None:
        payload = run_teacher_scenario_preflight(
            policy=ScriptedTeacherPolicy(dex=teacher_scenario_dex()),
        )

        self.assertEqual(payload["schema_version"], TEACHER_SCENARIO_PREFLIGHT_SCHEMA_VERSION)
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["failed_count"], 0)
        self.assertEqual(payload["scenario_count"], len(teacher_scenario_ids()))
        self.assertGreaterEqual(payload["teacher_branch_counts"]["damaging_move"], 1)
        self.assertGreaterEqual(payload["teacher_branch_counts"]["switch"], 1)

    def test_default_scenarios_pass_against_real_showdown_dex_when_available(self) -> None:
        root = Path(
            os.environ.get(
                "POKEZERO_SHOWDOWN_ROOT",
                "/Users/scott/workspace/pokerena/vendor/pokemon-showdown",
            )
        )
        if not (root / "dist" / "data" / "moves.js").exists():
            self.skipTest("built Pokemon Showdown dex not available")
        try:
            dex = load_showdown_dex(root)
        except Exception as exc:  # noqa: BLE001 - optional source integration test.
            self.skipTest(f"built Pokemon Showdown dex not loadable: {exc}")

        payload = run_teacher_scenario_preflight(
            policy=ScriptedTeacherPolicy(dex=dex),
        )

        self.assertTrue(payload["passed"])
        self.assertEqual(payload["failed_count"], 0)
        self.assertEqual(payload["scenario_count"], len(teacher_scenario_ids()))

    def test_subset_runs_only_requested_scenarios(self) -> None:
        payload = run_teacher_scenario_preflight(
            policy=ScriptedTeacherPolicy(dex=teacher_scenario_dex()),
            scenario_ids=("status-no-effect-electric-immunity",),
        )

        self.assertTrue(payload["passed"])
        self.assertEqual(payload["scenario_count"], 1)
        self.assertEqual(payload["scenarios"][0]["id"], "status-no-effect-electric-immunity")
        self.assertEqual(payload["teacher_branch_counts"], {"status_no_effect": 1})

    def test_unknown_scenario_is_rejected_with_known_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown teacher scenario"):
            run_teacher_scenario_preflight(
                policy=ScriptedTeacherPolicy(dex=teacher_scenario_dex()),
                scenario_ids=("not-a-scenario",),
            )

    def test_wrong_policy_decision_is_reported_as_failed_scenario(self) -> None:
        payload = run_teacher_scenario_preflight(
            policy=AlwaysFirstBadPolicy(),
            scenario_ids=("damaging-super-effective",),
        )

        self.assertFalse(payload["passed"])
        self.assertEqual(payload["failed_count"], 1)
        scenario = payload["scenarios"][0]
        self.assertEqual(scenario["id"], "damaging-super-effective")
        self.assertEqual(scenario["failed_fields"], ["action_index", "teacher_branch", "teacher_reason"])

    def test_policy_error_is_reported_as_failed_scenario(self) -> None:
        payload = run_teacher_scenario_preflight(
            policy=ExplodingPolicy(),
            scenario_ids=("damaging-super-effective",),
        )

        self.assertFalse(payload["passed"])
        scenario = payload["scenarios"][0]
        self.assertIsNone(scenario["observed"])
        self.assertIn("RuntimeError: boom", scenario["error"])


class AlwaysFirstBadPolicy:
    policy_id = "bad"

    def select_action(self, observation, *, rng: random.Random) -> PolicyDecision:
        return PolicyDecision(
            action_index=0,
            policy_id=self.policy_id,
            metadata={
                "action_family": "move",
                "teacher_branch": "fallback",
                "teacher_reason": "wrong",
            },
        )


class ExplodingPolicy:
    policy_id = "explode"

    def select_action(self, observation, *, rng: random.Random) -> PolicyDecision:
        raise RuntimeError("boom")


def teacher_scenario_dex():
    return showdown_dex_from_payload(
        {
            "moves": {
                "flamethrower": {
                    "id": "flamethrower",
                    "name": "Flamethrower",
                    "type": "Fire",
                    "category": "Special",
                    "basePower": 90,
                    "accuracy": 100,
                },
                "shadowball": {
                    "id": "shadowball",
                    "name": "Shadow Ball",
                    "type": "Ghost",
                    "category": "Physical",
                    "basePower": 80,
                    "accuracy": 100,
                },
                "thunderwave": {
                    "id": "thunderwave",
                    "name": "Thunder Wave",
                    "type": "Electric",
                    "category": "Status",
                    "basePower": 0,
                    "accuracy": 100,
                    "status": "par",
                },
                "glare": {
                    "id": "glare",
                    "name": "Glare",
                    "type": "Normal",
                    "category": "Status",
                    "basePower": 0,
                    "accuracy": 100,
                    "status": "par",
                },
                "healbell": {
                    "id": "healbell",
                    "name": "Heal Bell",
                    "type": "Normal",
                    "category": "Status",
                    "basePower": 0,
                    "accuracy": 0,
                },
                "growl": {
                    "id": "growl",
                    "name": "Growl",
                    "type": "Normal",
                    "category": "Status",
                    "basePower": 0,
                    "accuracy": 100,
                    "boosts": {"atk": -1},
                },
                "tackle": {
                    "id": "tackle",
                    "name": "Tackle",
                    "type": "Normal",
                    "category": "Physical",
                    "basePower": 35,
                    "accuracy": 95,
                },
                "rapidspin": {
                    "id": "rapidspin",
                    "name": "Rapid Spin",
                    "type": "Normal",
                    "category": "Physical",
                    "basePower": 50,
                    "accuracy": 100,
                },
                "recover": {
                    "id": "recover",
                    "name": "Recover",
                    "type": "Normal",
                    "category": "Status",
                    "basePower": 0,
                    "accuracy": 0,
                    "heal": True,
                },
                "swordsdance": {
                    "id": "swordsdance",
                    "name": "Swords Dance",
                    "type": "Normal",
                    "category": "Status",
                    "basePower": 0,
                    "accuracy": 0,
                    "boosts": {"atk": 2},
                },
                "spikes": {
                    "id": "spikes",
                    "name": "Spikes",
                    "type": "Ground",
                    "category": "Status",
                    "basePower": 0,
                    "accuracy": 0,
                },
            },
            "species": {
                "charizard": {"id": "charizard", "name": "Charizard", "types": ["Fire", "Flying"]},
                "dusclops": {"id": "dusclops", "name": "Dusclops", "types": ["Ghost"]},
                "golem": {"id": "golem", "name": "Golem", "types": ["Rock", "Ground"]},
                "snorlax": {"id": "snorlax", "name": "Snorlax", "types": ["Normal"]},
                "starmie": {"id": "starmie", "name": "Starmie", "types": ["Water", "Psychic"]},
                "xatu": {"id": "xatu", "name": "Xatu", "types": ["Psychic", "Flying"]},
            },
            "typeChart": {
                "fire": {},
                "flying": {"Ground": 3},
                "ghost": {"Normal": 3, "Ghost": 1},
                "ground": {"Electric": 3},
                "normal": {},
                "psychic": {"Ghost": 1},
                "rock": {"Fire": 2, "Normal": 2},
                "water": {"Fire": 2, "Rock": 2},
            },
        }
    )


if __name__ == "__main__":
    unittest.main()
