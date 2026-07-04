import unittest

from pokezero.actions import ACTION_COUNT
from pokezero.observation import (
    LEGACY_OBSERVATION_SCHEMA_VERSIONS,
    OBSERVATION_SCHEMA_VERSION,
    STATS_TOKEN_COUNT,
    TRANSITION_TOKEN_COUNT,
    ObservationFeatureMasks,
    ObservationPerspective,
    ObservationSpec,
    PokeZeroObservationV0,
    opponent_showdown_slot,
)


class FakeArray:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    def __len__(self) -> int:
        return self.shape[0]


class ObservationSpecTest(unittest.TestCase):
    def test_token_count_matches_spec_v2_shape(self) -> None:
        spec = ObservationSpec(categorical_feature_count=2, numeric_feature_count=3)

        # v2 layout: field + self mons + opponent mons + action candidates + stats token(s)
        # + transition-token slots (the 24 recent-event tokens are gone).
        self.assertEqual(STATS_TOKEN_COUNT, 1)
        self.assertEqual(TRANSITION_TOKEN_COUNT, 128)
        self.assertEqual(
            spec.token_count,
            1 + 6 + 6 + ACTION_COUNT + STATS_TOKEN_COUNT + TRANSITION_TOKEN_COUNT,
        )

    def test_schema_version_is_v2_and_v1_is_legacy(self) -> None:
        self.assertEqual(OBSERVATION_SCHEMA_VERSION, "pokezero.observation.v2")
        self.assertIn("pokezero.observation.v1", LEGACY_OBSERVATION_SCHEMA_VERSIONS)

    def test_legacy_schema_version_is_refused_with_pinned_tag_message(self) -> None:
        spec = ObservationSpec(categorical_feature_count=2, numeric_feature_count=3)
        observation = PokeZeroObservationV0(
            categorical_ids=tuple((0, 0) for _ in range(spec.token_count)),
            numeric_features=tuple((0.0, 0.0, 0.0) for _ in range(spec.token_count)),
            token_type_ids=tuple(0 for _ in range(spec.token_count)),
            attention_mask=tuple(True for _ in range(spec.token_count)),
            legal_action_mask=tuple(True for _ in range(ACTION_COUNT)),
            schema_version="pokezero.observation.v1",
        )

        with self.assertRaisesRegex(ValueError, "pinned tag"):
            observation.validate(spec)

    def test_feature_masks_validate_transition_budget(self) -> None:
        masks = ObservationFeatureMasks(transition_token_budget=32)
        self.assertEqual(masks.transition_token_budget, 32)
        with self.assertRaisesRegex(ValueError, "transition_token_budget"):
            ObservationFeatureMasks(transition_token_budget=0)
        with self.assertRaisesRegex(ValueError, "transition_token_budget"):
            ObservationFeatureMasks(transition_token_budget=TRANSITION_TOKEN_COUNT + 1)

    def test_observation_validates_fixed_shape(self) -> None:
        spec = ObservationSpec(categorical_feature_count=2, numeric_feature_count=3)
        observation = PokeZeroObservationV0(
            categorical_ids=tuple((0, 0) for _ in range(spec.token_count)),
            numeric_features=tuple((0.0, 0.0, 0.0) for _ in range(spec.token_count)),
            token_type_ids=tuple(0 for _ in range(spec.token_count)),
            attention_mask=tuple(True for _ in range(spec.token_count)),
            legal_action_mask=tuple(True for _ in range(ACTION_COUNT)),
        )

        observation.validate(spec)

    def test_observation_rejects_wrong_legal_mask_size(self) -> None:
        spec = ObservationSpec(categorical_feature_count=2, numeric_feature_count=3)
        observation = PokeZeroObservationV0(
            categorical_ids=tuple((0, 0) for _ in range(spec.token_count)),
            numeric_features=tuple((0.0, 0.0, 0.0) for _ in range(spec.token_count)),
            token_type_ids=tuple(0 for _ in range(spec.token_count)),
            attention_mask=tuple(True for _ in range(spec.token_count)),
            legal_action_mask=(True,),
        )

        with self.assertRaisesRegex(ValueError, "legal_action_mask"):
            observation.validate(spec)

    def test_observation_accepts_array_like_shapes(self) -> None:
        spec = ObservationSpec(categorical_feature_count=2, numeric_feature_count=3)
        observation = PokeZeroObservationV0(
            categorical_ids=FakeArray((spec.token_count, 2)),
            numeric_features=FakeArray((spec.token_count, 3)),
            token_type_ids=FakeArray((spec.token_count,)),
            attention_mask=FakeArray((spec.token_count,)),
            legal_action_mask=FakeArray((ACTION_COUNT,)),
        )

        observation.validate(spec)


class ObservationPerspectiveTest(unittest.TestCase):
    def test_opponent_showdown_slot_maps_transport_sides(self) -> None:
        self.assertEqual(opponent_showdown_slot("p1"), "p2")
        self.assertEqual(opponent_showdown_slot("p2"), "p1")

    def test_perspective_from_showdown_slot_records_player_relative_metadata(self) -> None:
        perspective = ObservationPerspective.from_showdown_slot("agent-a", "p2")

        self.assertEqual(perspective.player_id, "agent-a")
        self.assertEqual(perspective.showdown_slot, "p2")
        self.assertEqual(perspective.opponent_showdown_slot, "p1")

    def test_perspective_rejects_ambiguous_or_invalid_slots(self) -> None:
        with self.assertRaisesRegex(ValueError, "must differ"):
            ObservationPerspective("agent-a", "p1", "p1")

        with self.assertRaisesRegex(ValueError, "showdown_slot"):
            ObservationPerspective.from_showdown_slot("agent-a", "left")

    def test_observation_accepts_perspective_debug_metadata(self) -> None:
        spec = ObservationSpec(categorical_feature_count=2, numeric_feature_count=3)
        observation = PokeZeroObservationV0(
            categorical_ids=tuple((0, 0) for _ in range(spec.token_count)),
            numeric_features=tuple((0.0, 0.0, 0.0) for _ in range(spec.token_count)),
            token_type_ids=tuple(0 for _ in range(spec.token_count)),
            attention_mask=tuple(True for _ in range(spec.token_count)),
            legal_action_mask=tuple(True for _ in range(ACTION_COUNT)),
            perspective=ObservationPerspective.from_showdown_slot("agent-a", "p2"),
        )

        observation.validate(spec)


if __name__ == "__main__":
    unittest.main()
