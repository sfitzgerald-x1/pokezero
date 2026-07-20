from __future__ import annotations

from types import SimpleNamespace
import unittest

from pokezero.silent_mutation_audit import SilentMutationAuditReport, _has_protocol_backing, _pokemon_surface, _public_mutation_surface


def _snapshot(*, battle: dict, revealed: dict, protocol_lines: tuple[str, ...], turn: int = 1):
    return SimpleNamespace(
        bridge_snapshot={"battle": battle},
        replay=SimpleNamespace(public_revealed=revealed, turn_number=turn),
        protocol_lines=protocol_lines,
        belief_engine=SimpleNamespace(snapshot=lambda: SimpleNamespace(side=lambda _side: ())),
    )


def _battle(*, status: str, hp: int = 100, active: bool = True) -> dict:
    return {
        "field": {"weather": ""},
        "sides": [
            {
                "id": "p1",
                "pokemon": [
                    {
                        "species": "[Species:starmie]",
                        "isActive": active,
                        "fainted": False,
                        "hp": hp,
                        "maxhp": 100,
                        "status": status,
                        "boosts": {},
                        "volatiles": {},
                        "types": ["Water", "Psychic"],
                        "ability": "Natural Cure",
                        "item": "Leftovers",
                    }
                ],
                "sideConditions": {},
            },
            {"id": "p2", "pokemon": [], "sideConditions": {}},
        ],
    }


class SilentMutationAuditTests(unittest.TestCase):
    def test_surface_includes_revealed_public_fields_but_not_hidden_values(self) -> None:
        snapshot = _snapshot(
            battle=_battle(status="tox"),
            revealed={"p1": (SimpleNamespace(species="Starmie"),)},
            protocol_lines=(),
        )

        surface, ambiguous = _public_mutation_surface(snapshot)

        self.assertEqual(ambiguous, 0)
        self.assertEqual(set(surface["p1:starmie"]), {"active", "fainted", "hp", "status", "boosts", "volatiles", "types"})
        self.assertNotIn("ability", surface["p1:starmie"])
        self.assertNotIn("item", surface["p1:starmie"])

    def test_status_change_without_status_protocol_is_a_silent_candidate(self) -> None:
        before = _snapshot(
            battle=_battle(status="tox"),
            revealed={"p1": (SimpleNamespace(species="Starmie"),)},
            protocol_lines=("|switch|p1a: Starmie|Starmie, L80|100/100 tox",),
        )
        after = _snapshot(
            battle=_battle(status="", active=False),
            revealed={"p1": (SimpleNamespace(species="Starmie"),)},
            protocol_lines=before.protocol_lines + ("|switch|p1a: Blissey|Blissey, L80|100/100",),
            turn=2,
        )

        report = SilentMutationAuditReport()
        report.record_transition(before, after, game_id="natural-cure-shape")
        payload = report.to_json_dict()

        self.assertEqual(payload["silent_candidate_count"], 1)
        candidate = next(item for item in payload["aggregates"] if item["classification"] == "silent-candidate")
        self.assertEqual(candidate["entity"], "p1:starmie")
        self.assertEqual(candidate["field"], "status")
        self.assertNotIn("tox", str(candidate))
        self.assertNotIn("Natural Cure", str(candidate))

    def test_status_cure_with_protocol_backing_is_not_a_candidate(self) -> None:
        before = _snapshot(
            battle=_battle(status="tox"),
            revealed={"p1": (SimpleNamespace(species="Starmie"),)},
            protocol_lines=(),
        )
        after = _snapshot(
            battle=_battle(status=""),
            revealed={"p1": (SimpleNamespace(species="Starmie"),)},
            protocol_lines=("|-curestatus|p1a: Starmie|tox",),
            turn=2,
        )

        report = SilentMutationAuditReport()
        report.record_transition(before, after, game_id="public-cure")

        self.assertEqual(report.to_json_dict()["silent_candidate_count"], 0)
        self.assertEqual(report.to_json_dict()["classification_counts"]["protocol-backed"], 1)

    def test_switch_is_backing_for_outgoing_boost_and_volatile_resets(self) -> None:
        self.assertTrue(_has_protocol_backing("boosts", ("switch",)))
        self.assertTrue(_has_protocol_backing("volatiles", ("switch",)))

    def test_request_private_choice_lock_is_excluded_from_the_public_volatile_surface(self) -> None:
        surface = _pokemon_surface(
            {
                "isActive": True,
                "fainted": False,
                "hp": 100,
                "maxhp": 100,
                "status": "",
                "boosts": {},
                "volatiles": {"choicelock": "choiceband", "substitute": {}},
                "types": ["Water"],
            },
            public_volatiles=("substitute",),
        )

        self.assertEqual(surface["volatiles"], ("substitute",))


if __name__ == "__main__":
    unittest.main()
