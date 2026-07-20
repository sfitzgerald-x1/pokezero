"""Observation spec v3 tests (docs/observation_v3_spec.md).

Covers the schema-table wiring (fourth checkpoint-driven entry; v2.2 keeps the fresh
default), the appended v3 column layout, change 1 (the window-scoped ``-fail`` transition
bit, mirrored onto both turn-merged sub-blocks exactly like the miss bit), change 2 (the
public sleep-clause block bits on the field token, with the full lifecycle from the spec's
acceptance section), the v2.2 byte-identity invariant (a v2.2 encode of a fail-drawing log
is unchanged in shape and is a byte-prefix of the v3 encode), and the incremental-fold
twin's parity (handler + serialization round-trip, with the pre-v3 payload bytes
unchanged for fail-free games).
"""

import json
import os
import unittest
from pathlib import Path

from pokezero.observation import (
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION_V2_2,
    OBSERVATION_SCHEMA_VERSION_V3,
    SUPPORTED_OBSERVATION_SCHEMA_VERSIONS,
    TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS,
)
from pokezero.showdown import (
    FIELD_TOKEN_OFFSET,
    NUMERIC_SLEEP_CLAUSE_BLOCKS_OPP,
    NUMERIC_SLEEP_CLAUSE_BLOCKS_SELF,
    NUMERIC_TM2_FAIL,
    NUMERIC_TM2_MISS,
    NUMERIC_TT_FAIL,
    NUMERIC_TT_MISS,
    REPLAY_OBSERVATION_SPECS_BY_SCHEMA,
    TRANSITION_TOKEN_OFFSET,
    V2_2_REPLAY_OBSERVATION_SPEC,
    V3_REPLAY_OBSERVATION_SPEC,
    normalize_for_player,
    observation_from_player_state,
    observation_schema_version_from_choice,
    observation_spec_for_schema,
    parse_showdown_replay,
)
from pokezero.showdown import _ReplayParser
from pokezero.transitions import TOKEN_KIND_MOVE, extract_transition_tokens
from pokezero.turn_merged import extract_turn_merged_tokens

SHOWDOWN_ROOT = Path(
    os.environ.get("POKEZERO_SHOWDOWN_ROOT", "/Users/scott/workspace/pokerena/vendor/pokemon-showdown")
)

_LEADS = [
    "|player|p1|Alice|",
    "|player|p2|Bob|",
    "|switch|p1a: Snorlax|Snorlax, L80|100/100",
    "|switch|p2a: Skarmory|Skarmory, L76|100/100",
    "|turn|1",
]

# Turn 1: clean baseline. Turn 2: FIRST mover's status move fails (the ``-fail`` argument
# names the TARGET — the case the miss-style actor-side condition would drop, exercising
# the spec's window-scope rule). Turn 3: SECOND mover's status move fails.
_FAIL_LINES = _LEADS + [
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


class SchemaTableTest(unittest.TestCase):
    def test_v3_is_supported_but_not_the_default(self) -> None:
        self.assertIn(OBSERVATION_SCHEMA_VERSION_V3, SUPPORTED_OBSERVATION_SCHEMA_VERSIONS)
        self.assertIn(OBSERVATION_SCHEMA_VERSION_V3, REPLAY_OBSERVATION_SPECS_BY_SCHEMA)
        self.assertIs(
            observation_spec_for_schema(OBSERVATION_SCHEMA_VERSION_V3),
            V3_REPLAY_OBSERVATION_SPEC,
        )
        # v2.2 keeps the fresh-selection default: v3 launches only after the Rust fold
        # encoder mirrors it and the golden corpus regenerates (spec coordination section).
        self.assertEqual(OBSERVATION_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION_V2_2)
        # v3 shares v2.2's turn-merged transition surface.
        self.assertIn(OBSERVATION_SCHEMA_VERSION_V3, TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS)
        self.assertIn(OBSERVATION_SCHEMA_VERSION_V2_2, TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS)

    def test_v3_widths_append_four_numerics_to_the_v2_2_census(self) -> None:
        self.assertEqual(
            V3_REPLAY_OBSERVATION_SPEC.numeric_feature_count,
            V2_2_REPLAY_OBSERVATION_SPEC.numeric_feature_count + 4,
        )
        self.assertEqual(
            V3_REPLAY_OBSERVATION_SPEC.categorical_feature_count,
            V2_2_REPLAY_OBSERVATION_SPEC.categorical_feature_count,
        )
        self.assertEqual(
            V3_REPLAY_OBSERVATION_SPEC.token_count, V2_2_REPLAY_OBSERVATION_SPEC.token_count
        )

    def test_v3_column_layout(self) -> None:
        # The four appended columns start exactly at the v2.2 census end (155).
        self.assertEqual(NUMERIC_TT_FAIL, V2_2_REPLAY_OBSERVATION_SPEC.numeric_feature_count)
        self.assertEqual(NUMERIC_TM2_FAIL, NUMERIC_TT_FAIL + 1)
        self.assertEqual(NUMERIC_SLEEP_CLAUSE_BLOCKS_SELF, NUMERIC_TT_FAIL + 2)
        self.assertEqual(NUMERIC_SLEEP_CLAUSE_BLOCKS_OPP, NUMERIC_TT_FAIL + 3)
        self.assertEqual(
            V3_REPLAY_OBSERVATION_SPEC.numeric_feature_count,
            NUMERIC_SLEEP_CLAUSE_BLOCKS_OPP + 1,
        )

    def test_cli_choice_maps_to_v3(self) -> None:
        self.assertEqual(
            observation_schema_version_from_choice("v3"), OBSERVATION_SCHEMA_VERSION_V3
        )


class FailBitExtractionTest(unittest.TestCase):
    """Change 1 at the extraction layer: window-scoped, no side condition."""

    def _tokens(self, lines):
        replay = parse_showdown_replay(lines, battle_id="fail-extract")
        return extract_transition_tokens(replay, perspective_slot="p1")

    def test_target_named_fail_sets_the_actors_window(self) -> None:
        tokens = self._tokens(_FAIL_LINES)
        moves = [t for t in tokens if t.kind == TOKEN_KIND_MOVE]
        failed = [t for t in moves if t.fail]
        # Exactly the two failed Toxics — both Skarmory's, though ``-fail`` named Snorlax.
        self.assertEqual(len(failed), 2)
        for token in failed:
            self.assertEqual(token.actor_slot, "p2")
            self.assertEqual(token.action, "toxic")
            # Independent signals: a fail is not an accuracy miss.
            self.assertFalse(token.miss)
        # The successful turn-1 Toxic and every Body Slam stay unmarked.
        self.assertFalse(any(t.fail for t in moves if t not in failed))

    def test_no_fail_line_leaves_the_bit_unset(self) -> None:
        tokens = self._tokens(_FAIL_LINES[: len(_LEADS) + 5])  # through turn 1 only
        self.assertFalse(any(t.fail for t in tokens))

    def test_fail_lands_on_the_correct_merged_sub_block(self) -> None:
        replay = parse_showdown_replay(_FAIL_LINES, battle_id="fail-merged")
        merged = extract_turn_merged_tokens(replay, perspective_slot="p1")
        by_turn = {token.turn: token for token in merged if token.phase == "turn"}
        # Turn 2: Skarmory (fail) moved FIRST; turn 3: Skarmory (fail) moved SECOND.
        self.assertTrue(by_turn[2].first.fail)
        self.assertFalse(by_turn[2].second.fail)
        self.assertFalse(by_turn[3].first.fail)
        self.assertTrue(by_turn[3].second.fail)
        self.assertFalse(by_turn[1].first.fail)
        self.assertFalse(by_turn[1].second.fail)


class SleepClauseTrackerTest(unittest.TestCase):
    """Change 2 lifecycle at the public-parser layer (spec acceptance item 2)."""

    _INDUCED = _LEADS + [
        "|move|p1a: Snorlax|Lovely Kiss|p2a: Skarmory",
        "|-status|p2a: Skarmory|slp",
    ]

    def _state(self, lines, *, player="p1"):
        replay = parse_showdown_replay(lines, battle_id="sleep-clause")
        return normalize_for_player(
            replay, player_id=player, configured_showdown_slot=player
        )

    def test_induced_sleep_turns_the_inducers_bit_on(self) -> None:
        state = self._state(self._INDUCED)
        self.assertTrue(state.self_sleep_clause_blocks)
        self.assertFalse(state.opponent_sleep_clause_blocks)
        # The victim's perspective sees the symmetric bit.
        opponent_view = self._state(self._INDUCED, player="p2")
        self.assertFalse(opponent_view.self_sleep_clause_blocks)
        self.assertTrue(opponent_view.opponent_sleep_clause_blocks)

    def test_rest_does_not_engage_the_clause(self) -> None:
        lines = _LEADS + [
            "|move|p2a: Skarmory|Rest|p2a: Skarmory",
            "|-status|p2a: Skarmory|slp|[from] move: Rest",
        ]
        state = self._state(lines)
        self.assertFalse(state.self_sleep_clause_blocks)
        self.assertFalse(state.opponent_sleep_clause_blocks)

    def test_curestatus_clears_the_bit(self) -> None:
        state = self._state(self._INDUCED + ["|-curestatus|p2a: Skarmory|slp|[msg]"])
        self.assertFalse(state.self_sleep_clause_blocks)

    def test_faint_clears_the_bit(self) -> None:
        state = self._state(self._INDUCED + ["|faint|p2a: Skarmory"])
        self.assertFalse(state.self_sleep_clause_blocks)

    def test_switch_out_does_not_clear_the_bit(self) -> None:
        state = self._state(
            self._INDUCED + ["|switch|p2a: Starmie|Starmie, L76|100/100"]
        )
        self.assertTrue(state.self_sleep_clause_blocks)
        # ...and the benched sleeper's eventual Heal Bell cure (position-less ident) clears.
        cured = self._state(
            self._INDUCED
            + [
                "|switch|p2a: Starmie|Starmie, L76|100/100",
                "|-curestatus|p2: Skarmory|slp|[silent]",
            ]
        )
        self.assertFalse(cured.self_sleep_clause_blocks)

    def test_cureteam_clears_the_cured_sides_victims(self) -> None:
        # Aromatherapy: a single |-cureteam| line, no per-mon -curestatus (silent
        # clearStatus). The wake is public, so the tracked victim clears.
        state = self._state(
            self._INDUCED
            + [
                "|switch|p2a: Vileplume|Vileplume, L80|100/100",
                "|move|p2a: Vileplume|Aromatherapy|p2a: Vileplume",
                "|-cureteam|p2a: Vileplume|[from] move: Aromatherapy",
            ]
        )
        self.assertFalse(state.self_sleep_clause_blocks)

    def test_snapshot_round_trip_preserves_the_tracker(self) -> None:
        replay = parse_showdown_replay(self._INDUCED, battle_id="sleep-clause")
        self.assertEqual(replay.induced_sleep_victims.get("p1"), ("p2:skarmory",))
        resumed = _ReplayParser.from_snapshot(replay)
        resumed.feed(["|-curestatus|p2a: Skarmory|slp|[msg]"])
        self.assertEqual(resumed.snapshot().induced_sleep_victims.get("p1"), ())


@unittest.skipUnless(
    (SHOWDOWN_ROOT / "data" / "random-battles" / "gen3" / "sets.json").exists(),
    "requires a local Gen 3 Pokemon Showdown checkout",
)
class V3EncodeTest(unittest.TestCase):
    """Column-level v3 emission + the v2.2 byte-identity invariant."""

    @staticmethod
    def _vocab():
        from pokezero.randbat_vocab import gen3_category_vocabulary

        return gen3_category_vocabulary(SHOWDOWN_ROOT, include_turn_merged=True)

    def _state(self, lines, *, player="p1"):
        replay = parse_showdown_replay(lines, battle_id="v3-encode")
        return normalize_for_player(
            replay,
            player_id=player,
            configured_showdown_slot=player,
            format_id="gen3randombattle",
            include_turn_merged=True,
        )

    def _encode(self, state, spec):
        observation = observation_from_player_state(
            state, category_vocab=self._vocab(), spec=spec
        )
        observation.validate(spec)
        return observation

    def test_fail_columns_fill_for_both_sub_blocks_under_v3_only(self) -> None:
        state = self._state(_FAIL_LINES)
        observation = self._encode(state, V3_REPLAY_OBSERVATION_SPEC)
        rows = observation.numeric_features
        lead_row = TRANSITION_TOKEN_OFFSET  # lead pair
        turn_rows = {n: TRANSITION_TOKEN_OFFSET + n for n in (1, 2, 3)}
        # Turn 2: the failed Toxic moved FIRST; turn 3 it moved SECOND.
        self.assertEqual(rows[turn_rows[2]][NUMERIC_TT_FAIL], 1.0)
        self.assertEqual(rows[turn_rows[2]][NUMERIC_TM2_FAIL], 0.0)
        self.assertEqual(rows[turn_rows[3]][NUMERIC_TT_FAIL], 0.0)
        self.assertEqual(rows[turn_rows[3]][NUMERIC_TM2_FAIL], 1.0)
        # Clean rows carry neither bit; a fail is never a miss.
        for row_index in (lead_row, turn_rows[1]):
            self.assertEqual(rows[row_index][NUMERIC_TT_FAIL], 0.0)
            self.assertEqual(rows[row_index][NUMERIC_TM2_FAIL], 0.0)
        self.assertEqual(rows[turn_rows[2]][NUMERIC_TT_MISS], 0.0)
        self.assertEqual(rows[turn_rows[3]][NUMERIC_TM2_MISS], 0.0)

    def test_v2_2_encode_of_the_fail_log_is_unchanged_and_a_byte_prefix_of_v3(self) -> None:
        # The absolute invariant: the fail-drawing log's v2.2 encoding keeps its exact
        # shape (155 columns — the fail columns do not exist) and every shared surface is
        # byte-identical between the two encodes, so the v2.2 output cannot have moved.
        # (Cross-checked once against the pre-change encoder on main: byte-identical.)
        state = self._state(_FAIL_LINES)
        v2_2 = self._encode(state, V2_2_REPLAY_OBSERVATION_SPEC)
        v3 = self._encode(state, V3_REPLAY_OBSERVATION_SPEC)
        self.assertEqual(v2_2.schema_version, OBSERVATION_SCHEMA_VERSION_V2_2)
        self.assertEqual(v3.schema_version, OBSERVATION_SCHEMA_VERSION_V3)
        width = V2_2_REPLAY_OBSERVATION_SPEC.numeric_feature_count
        for row_index, (v22_row, v3_row) in enumerate(
            zip(v2_2.numeric_features, v3.numeric_features)
        ):
            self.assertEqual(len(v22_row), width)
            self.assertEqual(len(v3_row), width + 4)
            self.assertEqual(tuple(v22_row), tuple(v3_row[:width]), f"numeric row {row_index}")
        # No categorical additions: the rows agree everywhere.
        self.assertEqual(
            [tuple(row) for row in v2_2.categorical_ids],
            [tuple(row) for row in v3.categorical_ids],
        )
        self.assertEqual(v2_2.attention_mask, v3.attention_mask)
        self.assertEqual(v2_2.token_type_ids, v3.token_type_ids)

    def test_sleep_clause_bits_encode_on_the_field_token_under_v3_only(self) -> None:
        lines = _LEADS + [
            "|move|p1a: Snorlax|Lovely Kiss|p2a: Skarmory",
            "|-status|p2a: Skarmory|slp",
            "|upkeep",
            "|turn|2",
        ]
        inducer = self._encode(self._state(lines), V3_REPLAY_OBSERVATION_SPEC)
        field_row = inducer.numeric_features[FIELD_TOKEN_OFFSET]
        self.assertEqual(field_row[NUMERIC_SLEEP_CLAUSE_BLOCKS_SELF], 1.0)
        self.assertEqual(field_row[NUMERIC_SLEEP_CLAUSE_BLOCKS_OPP], 0.0)
        victim = self._encode(self._state(lines, player="p2"), V3_REPLAY_OBSERVATION_SPEC)
        victim_row = victim.numeric_features[FIELD_TOKEN_OFFSET]
        self.assertEqual(victim_row[NUMERIC_SLEEP_CLAUSE_BLOCKS_SELF], 0.0)
        self.assertEqual(victim_row[NUMERIC_SLEEP_CLAUSE_BLOCKS_OPP], 1.0)
        # Under v2.2 the columns do not exist (width) and the shared prefix is untouched.
        v2_2 = self._encode(self._state(lines), V2_2_REPLAY_OBSERVATION_SPEC)
        self.assertEqual(
            len(v2_2.numeric_features[FIELD_TOKEN_OFFSET]),
            V2_2_REPLAY_OBSERVATION_SPEC.numeric_feature_count,
        )
        self.assertEqual(
            tuple(v2_2.numeric_features[FIELD_TOKEN_OFFSET]),
            tuple(inducer.numeric_features[FIELD_TOKEN_OFFSET])[
                : V2_2_REPLAY_OBSERVATION_SPEC.numeric_feature_count
            ],
        )


class IncrementalFoldParityTest(unittest.TestCase):
    """The incremental fold twin (transitions_fold) mirrors the fail marker, and its
    serialized payload stays byte-stable for fail-free games (the committed golden
    corpus predates the field)."""

    def test_incremental_fold_matches_batch_on_a_fail_log(self) -> None:
        from pokezero.transitions_fold import FoldState

        replay = parse_showdown_replay(_FAIL_LINES, battle_id="fail-fold")
        batch = extract_transition_tokens(replay, perspective_slot="p1")
        _, products = FoldState.initial(perspective_slot="p1").advance(_FAIL_LINES)
        self.assertEqual(products.transition_tokens, batch)
        self.assertTrue(any(t.fail for t in products.transition_tokens))

    def test_payload_round_trip_carries_fail_and_omits_it_when_clean(self) -> None:
        from pokezero.transitions_fold import FoldState

        # Fail-free prefix: the payload must not mention the field at all (pre-v3 bytes).
        clean_state, _ = FoldState.initial(perspective_slot="p1").advance(
            _FAIL_LINES[: len(_LEADS) + 5]
        )
        self.assertNotIn('"fail"', json.dumps(clean_state.to_payload(), sort_keys=True))
        # Fail-carrying game: serialize -> resume -> identical payload and products.
        state, products = FoldState.initial(perspective_slot="p1").advance(_FAIL_LINES)
        canonical = json.dumps(state.to_payload(), sort_keys=True)
        self.assertIn('"fail": true', canonical)
        resumed = FoldState.from_payload(json.loads(canonical))
        self.assertEqual(json.dumps(resumed.to_payload(), sort_keys=True), canonical)
        self.assertEqual(resumed.products().transition_tokens, products.transition_tokens)


if __name__ == "__main__":
    unittest.main()
