"""Raw-tensor checks for the in-place v3 observation-layout cutover."""

from __future__ import annotations

import unittest
from dataclasses import replace

from pokezero.observation import OBSERVATION_SCHEMA_VERSION_V3
from pokezero.showdown import (
    NUMERIC_TM2_FAIL,
    NUMERIC_TT_DAMAGE_FRACTION,
    NUMERIC_TT_FAIL,
    TRANSITION_TOKEN_OFFSET,
    V2_2_REPLAY_OBSERVATION_SPEC,
    V3_DROPPED_LEGACY_NUMERIC_INDICES,
    V3_LEGACY_NUMERIC_FEATURE_COUNT,
    V3_NUMERIC_INDEX_BY_LEGACY_INDEX,
    V3_NUMERIC_LEGACY_INDEX_BY_NEW_INDEX,
    V3_REWRITTEN_LEGACY_NUMERIC_INDICES,
    V3_REPLAY_OBSERVATION_SPEC,
    _project_v3_numeric_rows,
    normalize_for_player,
    observation_from_player_state,
    parse_showdown_replay,
    v3_numeric_index,
)


class _TurnMergedVocabulary:
    """Minimal vocabulary for numeric-layout tests; categorical ids are irrelevant here."""

    def encode(self, value: str) -> int:
        return 0

    def is_enumerated(self, value: str) -> bool:
        return value == "tt_phase:turn"


_FAIL_LINES = [
    "|player|p1|Alice|",
    "|player|p2|Bob|",
    "|switch|p1a: Snorlax|Snorlax, L80|100/100",
    "|switch|p2a: Skarmory|Skarmory, L76|100/100",
    "|turn|1",
    "|move|p1a: Snorlax|Body Slam|p2a: Skarmory",
    "|-damage|p2a: Skarmory|70/100",
    "|move|p2a: Skarmory|Toxic|p1a: Snorlax",
    "|-status|p1a: Snorlax|tox",
    "|upkeep",
    "|turn|2",
    "|move|p2a: Skarmory|Toxic|p1a: Snorlax",
    "|-fail|p1a: Snorlax|tox",
    "|move|p1a: Snorlax|Body Slam|p2a: Skarmory",
    "|-damage|p2a: Skarmory|55/100",
    "|upkeep",
    "|turn|3",
    "|move|p1a: Snorlax|Body Slam|p2a: Skarmory",
    "|-damage|p2a: Skarmory|40/100",
    "|move|p2a: Skarmory|Toxic|p1a: Snorlax",
    "|-fail|p1a: Snorlax|tox",
    "|upkeep",
    "|turn|4",
]


class ObservationV3LayoutCutoverTest(unittest.TestCase):
    def _state(self):
        replay = parse_showdown_replay(_FAIL_LINES, battle_id="v3-layout-cutover")
        return normalize_for_player(
            replay,
            player_id="p1",
            configured_showdown_slot="p1",
            format_id="gen3randombattle",
            include_turn_merged=True,
        )

    def _encode(self, spec):
        observation = observation_from_player_state(
            self._state(), category_vocab=_TurnMergedVocabulary(), spec=spec
        )
        observation.validate(spec)
        return observation

    def test_projection_moves_every_writer_column_exactly_once(self) -> None:
        legacy_rows = [
            [float((row_index + 1) * 10_000 + column) for column in range(V3_LEGACY_NUMERIC_FEATURE_COUNT)]
            for row_index in range(3)
        ]
        projected = _project_v3_numeric_rows(legacy_rows)
        self.assertEqual(len(projected[0]), V3_REPLAY_OBSERVATION_SPEC.numeric_feature_count)
        for row_index, row in enumerate(projected):
            for new_index, legacy_index in enumerate(V3_NUMERIC_LEGACY_INDEX_BY_NEW_INDEX):
                self.assertEqual(row[new_index], legacy_rows[row_index][legacy_index])

    def test_legacy_v2_2_surface_is_fully_accounted_for(self) -> None:
        legacy_v2_2_indices = set(range(V2_2_REPLAY_OBSERVATION_SPEC.numeric_feature_count))
        carried = set(V3_NUMERIC_INDEX_BY_LEGACY_INDEX)
        self.assertEqual(
            legacy_v2_2_indices,
            (carried & legacy_v2_2_indices)
            | V3_DROPPED_LEGACY_NUMERIC_INDICES
            | V3_REWRITTEN_LEGACY_NUMERIC_INDICES,
        )
        self.assertEqual(V3_DROPPED_LEGACY_NUMERIC_INDICES, {24, 25, 35, 36, 48, 49, 50, 51, 52, 53, 54, 55, 103, 104})
        self.assertEqual(V3_REWRITTEN_LEGACY_NUMERIC_INDICES, {NUMERIC_TT_DAMAGE_FRACTION})

    def test_raw_v3_output_uses_the_grouped_layout(self) -> None:
        v2_2 = self._encode(V2_2_REPLAY_OBSERVATION_SPEC)
        v3 = self._encode(V3_REPLAY_OBSERVATION_SPEC)
        self.assertEqual(v3.schema_version, OBSERVATION_SCHEMA_VERSION_V3)
        self.assertEqual(len(v3.numeric_features[0]), 155)

        for v2_row, v3_row in zip(v2_2.numeric_features, v3.numeric_features):
            for legacy_index, new_index in V3_NUMERIC_INDEX_BY_LEGACY_INDEX.items():
                if legacy_index >= V2_2_REPLAY_OBSERVATION_SPEC.numeric_feature_count:
                    continue
                if legacy_index in V3_REWRITTEN_LEGACY_NUMERIC_INDICES:
                    continue
                self.assertEqual(v3_row[new_index], v2_row[legacy_index])

        self.assertEqual(v3.categorical_ids, v2_2.categorical_ids)
        self.assertEqual(v3.attention_mask, v2_2.attention_mask)
        self.assertEqual(v3.token_type_ids, v2_2.token_type_ids)

        # The fixture has both first- and second-sub-block fail events. These checks use the
        # physical v3 rows rather than the semantic compatibility view used by lifecycle tests.
        self.assertEqual(
            v3.numeric_features[TRANSITION_TOKEN_OFFSET + 2][v3_numeric_index(NUMERIC_TT_FAIL)],
            1.0,
        )
        self.assertEqual(
            v3.numeric_features[TRANSITION_TOKEN_OFFSET + 3][v3_numeric_index(NUMERIC_TM2_FAIL)],
            1.0,
        )

    def test_v3_refuses_a_narrower_public_census(self) -> None:
        narrow = replace(
            V3_REPLAY_OBSERVATION_SPEC,
            numeric_feature_count=V3_REPLAY_OBSERVATION_SPEC.numeric_feature_count - 1,
        )
        with self.assertRaisesRegex(ValueError, "requires at least 155 numeric columns"):
            observation_from_player_state(
                self._state(), category_vocab=_TurnMergedVocabulary(), spec=narrow
            )


if __name__ == "__main__":
    unittest.main()
