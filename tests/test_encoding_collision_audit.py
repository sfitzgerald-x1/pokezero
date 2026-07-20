from __future__ import annotations

import unittest

from pokezero.encoding_collision_audit import EncodingCollisionAudit
from pokezero.observation import OBSERVATION_SCHEMA_VERSION_V3
from pokezero.public_decision_corpus import (
    PublicActionIdentifier,
    PublicDecisionRecord,
    PublicObservation,
    PublicResolvedActionRound,
    public_decision_id,
)


def _record(
    *,
    state: dict[str, object],
    player: str = "p1",
    turn: int = 2,
    rounds: tuple[PublicResolvedActionRound, ...] = (),
) -> PublicDecisionRecord:
    observation = PublicObservation(
        schema_version=OBSERVATION_SCHEMA_VERSION_V3,
        categorical_ids=((11, 12),),
        numeric_features=((0.25, 0.5),),
        token_type_ids=(1,),
        attention_mask=(True,),
        legal_action_mask=(True, False, False, False, False, False, False, False, False),
        acting_player_state=state,
    )
    prototype = PublicDecisionRecord(
        decision_id="pending",
        battle_id="collision-fixture",
        seed=123,
        format_id="gen3randombattle",
        acting_player=player,
        turn_index=turn,
        recorded_action_index=0,
        observation=observation,
        history=(),
        current_legal_action_mask=observation.legal_action_mask,
        public_resolved_action_rounds=rounds,
        public_belief_view={"candidate_count": 3},
    )
    return PublicDecisionRecord(**{**prototype.__dict__, "decision_id": public_decision_id(prototype)})


class EncodingCollisionAuditTest(unittest.TestCase):
    def test_reports_distinct_public_state_for_identical_model_input(self) -> None:
        audit = EncodingCollisionAudit()
        audit.add(_record(state={"turn_number": 2, "weather": "sunnyday"}))
        audit.add(_record(state={"turn_number": 2, "weather": "raindance"}))

        payload = audit.to_json_dict(corpus={"selected_content_sha256": "fixture"})

        self.assertEqual(payload["records_scanned"], 2)
        self.assertEqual(payload["collision_group_count"], 1)
        self.assertEqual(payload["actionable_collision_group_count"], 1)
        collision = payload["collision_groups"][0]
        self.assertEqual(collision["decision_kind"], "move-only")
        self.assertTrue(collision["actionable"])
        self.assertIn("acting_player_state.weather", collision["alternatives"][0]["difference_paths"])

    def test_keeps_seat_scopes_separate(self) -> None:
        audit = EncodingCollisionAudit()
        audit.add(_record(state={"turn_number": 2, "weather": "sunnyday"}, player="p1"))
        audit.add(_record(state={"turn_number": 2, "weather": "raindance"}, player="p2"))

        payload = audit.to_json_dict(corpus={})

        self.assertEqual(payload["collision_group_count"], 0)

    def test_transition_history_abstraction_is_labeled_not_hidden(self) -> None:
        audit = EncodingCollisionAudit()
        state = {"turn_number": 2, "weather": "sunnyday"}
        audit.add(_record(state=state))
        audit.add(
            _record(
                state=state,
                rounds=(
                    PublicResolvedActionRound(
                        turn_index=0,
                        actions={"p1": PublicActionIdentifier(kind="move", move_id="toxic")},
                    ),
                ),
            )
        )

        payload = audit.to_json_dict(corpus={})

        self.assertEqual(payload["collision_group_count"], 1)
        self.assertEqual(payload["actionable_collision_group_count"], 0)
        collision = payload["collision_groups"][0]
        self.assertFalse(collision["actionable"])
        self.assertEqual(collision["whitelist_classifications"], ["transition-window-truncation"])

    def test_rejects_non_v3_records(self) -> None:
        audit = EncodingCollisionAudit()
        record = _record(state={"turn_number": 2})
        legacy = PublicDecisionRecord(
            **{**record.__dict__, "observation": PublicObservation(**{**record.observation.__dict__, "schema_version": "v2.2"})}
        )

        with self.assertRaisesRegex(ValueError, "requires schema"):
            audit.add(legacy)


if __name__ == "__main__":
    unittest.main()
