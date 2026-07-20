from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pokezero.encoding_collision_audit import (
    CollisionSketchWriter,
    EncodingCollisionAudit,
    audit_collision_sketches,
    collision_sketch_manifest,
)
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

    def test_compact_sketch_reports_collision_without_model_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "collision-sketch.jsonl"
            manifest = collision_sketch_manifest(
                capture_manifest={
                    "opponent_legal_mask_mode": "hidden",
                    "root_noise": {"enabled": False, "root_dirichlet_alpha": None},
                }
            )
            with CollisionSketchWriter(path, manifest=manifest) as writer:
                writer.append_record(_record(state={"turn_number": 2, "weather": "sunnyday"}))
                writer.append_record(_record(state={"turn_number": 2, "weather": "raindance"}))
                writer.complete()

            rows = path.read_text(encoding="utf-8")
            self.assertNotIn("numeric_features", rows)
            self.assertNotIn("categorical_ids", rows)
            self.assertNotIn("acting_player_state", rows)
            payload = audit_collision_sketches([path])

        self.assertEqual(payload["records_scanned"], 2)
        self.assertEqual(payload["collision_group_count"], 1)
        self.assertEqual(payload["actionable_collision_group_count"], 1)
        collision = payload["collision_groups"][0]
        self.assertTrue(collision["requires_public_replay_hydration"])
        self.assertTrue(collision["actionable"])
        self.assertIn("seed", collision["base"]["locator"])

    def test_compact_sketch_resume_preserves_valid_records_and_recovers_partial_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "collision-sketch.jsonl"
            manifest = collision_sketch_manifest(
                capture_manifest={
                    "opponent_legal_mask_mode": "hidden",
                    "root_noise": {"enabled": False, "root_dirichlet_alpha": None},
                }
            )
            first = _record(state={"turn_number": 2, "weather": "sunnyday"})
            second = _record(state={"turn_number": 2, "weather": "raindance"})
            with CollisionSketchWriter(path, manifest=manifest) as writer:
                writer.append_record(first)
            with path.open("ab") as handle:
                handle.write(b'{"record_type":"sketch"')

            with CollisionSketchWriter(path, manifest=manifest, resume=True) as writer:
                self.assertEqual(writer.resumed_record_count, 1)
                self.assertTrue(writer.recovered_trailing_partial)
                self.assertEqual(writer.append_record(first), 0)
                self.assertEqual(writer.append_record(second), 1)
                self.assertEqual(writer.record_count, 2)
                writer.complete()

            payload = audit_collision_sketches([path])
        self.assertEqual(payload["records_scanned"], 2)

    def test_incomplete_sketch_is_resumable_but_rejected_by_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "collision-sketch.jsonl"
            manifest = collision_sketch_manifest(
                capture_manifest={
                    "opponent_legal_mask_mode": "hidden",
                    "root_noise": {"enabled": False, "root_dirichlet_alpha": None},
                }
            )
            with CollisionSketchWriter(path, manifest=manifest) as writer:
                writer.append_record(_record(state={"turn_number": 2, "weather": "sunnyday"}))

            with self.assertRaisesRegex(ValueError, "incomplete"):
                audit_collision_sketches([path])

            with CollisionSketchWriter(path, manifest=manifest, resume=True) as writer:
                self.assertEqual(writer.resumed_record_count, 1)
                writer.complete()

            with self.assertRaisesRegex(ValueError, "already complete"):
                CollisionSketchWriter(path, manifest=manifest, resume=True)

    def test_compact_locator_retains_the_actual_foulplay_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "collision-sketch.jsonl"
            manifest = collision_sketch_manifest(
                capture_manifest={
                    "opponent_legal_mask_mode": "hidden",
                    "root_noise": {"enabled": False, "root_dirichlet_alpha": None},
                }
            )
            with CollisionSketchWriter(path, manifest=manifest) as writer:
                writer.append_record(
                    _record(state={"turn_number": 2, "weather": "sunnyday"}),
                    foulplay_random_seed=987654,
                )
                writer.complete()

            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[1]["foulplay_random_seed"], 987654)

    def test_completion_hash_rejects_a_mutated_completed_sketch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "collision-sketch.jsonl"
            manifest = collision_sketch_manifest(
                capture_manifest={
                    "opponent_legal_mask_mode": "hidden",
                    "root_noise": {"enabled": False, "root_dirichlet_alpha": None},
                }
            )
            with CollisionSketchWriter(path, manifest=manifest) as writer:
                writer.append_record(_record(state={"turn_number": 2, "weather": "sunnyday"}))
                writer.complete()

            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            rows[1]["input_hash"] = "0" * 64
            path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "completion does not match"):
                audit_collision_sketches([path])


if __name__ == "__main__":
    unittest.main()
