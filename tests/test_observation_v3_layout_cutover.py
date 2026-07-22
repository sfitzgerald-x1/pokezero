"""Raw-tensor checks for the in-place v3 observation-layout cutover."""

from __future__ import annotations

import unittest
from dataclasses import replace

from pokezero.observation import OBSERVATION_SCHEMA_VERSION_V3
from pokezero.showdown import (
    ACTION_CANDIDATE_TOKEN_OFFSET,
    FIELD_TOKEN_OFFSET,
    NUMERIC_TM2_DAMAGE_FRACTION,
    NUMERIC_TM2_FAIL,
    NUMERIC_SELF_SCREENS,
    NUMERIC_TOXIC_STAGE,
    NUMERIC_TT_ABS_TURN,
    NUMERIC_TT_DAMAGE_FRACTION,
    NUMERIC_TT_FAIL,
    OPPONENT_POKEMON_TOKEN_OFFSET,
    OPPONENT_TENDENCY_STATS_TOKEN_OFFSET,
    SELF_POKEMON_TOKEN_OFFSET,
    TRANSITION_TOKEN_OFFSET,
    V2_2_REPLAY_OBSERVATION_SPEC,
    V3_DROPPED_LEGACY_NUMERIC_INDICES,
    V3_NUMERIC_LAYOUT_GROUPS,
    V3_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT,
    V3_NUMERIC_INDEX_BY_LEGACY_INDEX,
    V3_NUMERIC_LEGACY_INDEX_BY_NEW_INDEX,
    V3_REWRITTEN_LEGACY_NUMERIC_INDICES,
    V3_REPLAY_OBSERVATION_SPEC,
    _project_v3_numeric_rows,
    numeric_index_if_present_for_schema,
    numeric_index_for_schema,
    normalize_for_player,
    observation_from_player_state,
    parse_showdown_replay,
    v3_numeric_index,
)

# This literal is intentionally independent of the production grouping table. It is the
# physical V3 schema manifest: changing the map's order must fail this test, not rewrite its
# own expected output. Values are historical writer indices in V3 public-column order.
_EXPECTED_V3_LEGACY_INDEX_BY_NEW_INDEX = (
    0, 1, 2, 3, 15, 33, 16, 17, 18, 19, 20, 21,
    26, 27, 28, 29, 30, 37, 38, 39, 40, 41, 42, 43,
    58, 59, 60, 61, 62, 137, 138, 139, 159, 160, 161, 162,
    163, 164, 165, 4, 5, 6, 7, 8, 9, 10, 11, 63,
    64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
    76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
    88, 89, 90, 91, 121, 122, 123, 124, 125, 126, 127, 128,
    129, 130, 131, 132, 133, 134, 135, 136, 12, 13, 14, 31,
    32, 34, 22, 23, 44, 45, 46, 47, 56, 57, 157, 158,
    166, 167, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
    102, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
    116, 117, 118, 119, 120, 140, 141, 142, 143, 144, 145, 146,
    147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 168,
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

    def _encode(self, spec, *, state=None):
        observation = observation_from_player_state(
            state or self._state(), category_vocab=_TurnMergedVocabulary(), spec=spec
        )
        observation.validate(spec)
        return observation

    def test_projection_moves_every_writer_column_exactly_once(self) -> None:
        legacy_rows = [
            [
                float((row_index + 1) * 10_000 + column)
                for column in range(V3_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT)
            ]
            for row_index in range(3)
        ]
        projected = _project_v3_numeric_rows(legacy_rows)
        self.assertEqual(len(projected[0]), V3_REPLAY_OBSERVATION_SPEC.numeric_feature_count)
        for row_index, row in enumerate(projected):
            for new_index, legacy_index in enumerate(V3_NUMERIC_LEGACY_INDEX_BY_NEW_INDEX):
                self.assertEqual(row[new_index], legacy_rows[row_index][legacy_index])

    def test_public_layout_matches_the_independent_physical_manifest(self) -> None:
        self.assertEqual(V3_NUMERIC_LEGACY_INDEX_BY_NEW_INDEX, _EXPECTED_V3_LEGACY_INDEX_BY_NEW_INDEX)

    def test_v3_shortens_only_the_history_tail(self) -> None:
        self.assertEqual(
            (
                FIELD_TOKEN_OFFSET,
                SELF_POKEMON_TOKEN_OFFSET,
                OPPONENT_POKEMON_TOKEN_OFFSET,
                ACTION_CANDIDATE_TOKEN_OFFSET,
                OPPONENT_TENDENCY_STATS_TOKEN_OFFSET,
                TRANSITION_TOKEN_OFFSET,
                V3_REPLAY_OBSERVATION_SPEC.token_count,
            ),
            (0, 1, 7, 13, 22, 23, 87),
        )
        self.assertEqual(V3_REPLAY_OBSERVATION_SPEC.transition_token_count, 64)
        self.assertEqual(V2_2_REPLAY_OBSERVATION_SPEC.transition_token_count, 128)
        self.assertEqual(V2_2_REPLAY_OBSERVATION_SPEC.token_count, 151)

    def test_v3_history_keeps_the_most_recent_64_turn_rows(self) -> None:
        state = self._state()
        template = state.turn_merged_tokens[-1]
        turn_merged_tokens = tuple(replace(template, turn=turn) for turn in range(1, 71))
        observation = self._encode(
            V3_REPLAY_OBSERVATION_SPEC,
            state=replace(state, turn_number=70, turn_merged_tokens=turn_merged_tokens),
        )

        history_mask = observation.attention_mask[TRANSITION_TOKEN_OFFSET:]
        self.assertEqual(len(history_mask), 64)
        self.assertEqual(sum(history_mask), 64)
        turn_column = v3_numeric_index(NUMERIC_TT_ABS_TURN)
        self.assertAlmostEqual(
            observation.numeric_features[TRANSITION_TOKEN_OFFSET][turn_column], 7 / 1000
        )
        self.assertAlmostEqual(observation.numeric_features[-1][turn_column], 70 / 1000)

    def test_numeric_group_boundaries_and_total_are_frozen(self) -> None:
        expected = (
            ("core", 0, 5, 6),
            ("pokemon_state", 6, 38, 33),
            ("belief", 39, 91, 53),
            ("action", 92, 97, 6),
            ("field", 98, 109, 12),
            ("tendency", 110, 120, 11),
            ("history", 121, 154, 34),
        )
        cursor = 0
        actual = []
        for name, legacy_indices in V3_NUMERIC_LAYOUT_GROUPS:
            start = cursor
            cursor += len(legacy_indices)
            actual.append((name, start, cursor - 1, len(legacy_indices)))
        self.assertEqual(tuple(actual), expected)
        self.assertEqual(cursor, V3_REPLAY_OBSERVATION_SPEC.numeric_feature_count)

    def test_schema_aware_numeric_lookup_maps_or_rejects(self) -> None:
        self.assertEqual(
            numeric_index_for_schema(
                V2_2_REPLAY_OBSERVATION_SPEC.schema_version, NUMERIC_TOXIC_STAGE
            ),
            NUMERIC_TOXIC_STAGE,
        )
        self.assertEqual(
            numeric_index_for_schema(
                V3_REPLAY_OBSERVATION_SPEC.schema_version, NUMERIC_TOXIC_STAGE
            ),
            17,
        )
        with self.assertRaisesRegex(ValueError, "dropped from v3"):
            numeric_index_for_schema(
                V3_REPLAY_OBSERVATION_SPEC.schema_version, NUMERIC_SELF_SCREENS
            )
        self.assertIsNone(
            numeric_index_if_present_for_schema(
                V3_REPLAY_OBSERVATION_SPEC.schema_version, NUMERIC_SELF_SCREENS
            )
        )
        with self.assertRaisesRegex(ValueError, "not part of v3"):
            numeric_index_if_present_for_schema(
                V3_REPLAY_OBSERVATION_SPEC.schema_version,
                V3_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT,
            )

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
        self.assertEqual(
            V3_REWRITTEN_LEGACY_NUMERIC_INDICES,
            {NUMERIC_TT_DAMAGE_FRACTION, NUMERIC_TM2_DAMAGE_FRACTION},
        )

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

        v3_rows = V3_REPLAY_OBSERVATION_SPEC.token_count
        self.assertEqual(v3.categorical_ids, v2_2.categorical_ids[:v3_rows])
        self.assertEqual(v3.attention_mask, v2_2.attention_mask[:v3_rows])
        self.assertEqual(v3.token_type_ids, v2_2.token_type_ids[:v3_rows])

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

    def test_second_sub_block_confusion_rewrite_is_declared_and_projected(self) -> None:
        state = self._state()
        token_index, token = next(
            (index, token)
            for index, token in enumerate(state.turn_merged_tokens)
            if token.second.status == "action"
        )
        rewritten_second = replace(
            token.second,
            damage_fraction=0.25,
            confusion_selfhit=True,
            confusion_selfhit_fraction=0.10,
        )
        rewritten_token = replace(token, second=rewritten_second)
        rewritten_state = replace(
            state,
            turn_merged_tokens=(
                *state.turn_merged_tokens[:token_index],
                rewritten_token,
                *state.turn_merged_tokens[token_index + 1 :],
            ),
        )
        v2_2 = self._encode(V2_2_REPLAY_OBSERVATION_SPEC, state=rewritten_state)
        v3 = self._encode(V3_REPLAY_OBSERVATION_SPEC, state=rewritten_state)
        row = TRANSITION_TOKEN_OFFSET + token_index
        self.assertEqual(v2_2.numeric_features[row][NUMERIC_TM2_DAMAGE_FRACTION], 0.25)
        self.assertEqual(
            v3.numeric_features[row][v3_numeric_index(NUMERIC_TM2_DAMAGE_FRACTION)],
            0.15,
        )

    def test_v3_refuses_a_noncanonical_public_census(self) -> None:
        for numeric_feature_count in (
            V3_REPLAY_OBSERVATION_SPEC.numeric_feature_count - 1,
            V3_REPLAY_OBSERVATION_SPEC.numeric_feature_count + 1,
        ):
            with self.subTest(numeric_feature_count=numeric_feature_count):
                invalid = replace(
                    V3_REPLAY_OBSERVATION_SPEC,
                    numeric_feature_count=numeric_feature_count,
                )
                with self.assertRaisesRegex(ValueError, "requires exactly 155 numeric columns"):
                    observation_from_player_state(
                        self._state(), category_vocab=_TurnMergedVocabulary(), spec=invalid
                    )


if __name__ == "__main__":
    unittest.main()
