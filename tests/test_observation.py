import unittest

from pokezero.actions import ACTION_COUNT
from pokezero.observation import ObservationSpec, PokeZeroObservationV0


class FakeArray:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    def __len__(self) -> int:
        return self.shape[0]


class ObservationSpecTest(unittest.TestCase):
    def test_token_count_matches_first_iteration_shape(self) -> None:
        spec = ObservationSpec(categorical_feature_count=2, numeric_feature_count=3)

        self.assertEqual(spec.token_count, 1 + 6 + 6 + ACTION_COUNT + 24)

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


if __name__ == "__main__":
    unittest.main()
