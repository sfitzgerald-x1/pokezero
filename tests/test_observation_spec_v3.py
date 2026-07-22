"""Observation spec v3 tests (docs/observation_v3_spec.md).

Covers the schema-table wiring (fourth checkpoint-driven entry; v2.2 keeps the fresh
default), the grouped v3 column layout, change 1 (the window-scoped ``-fail`` transition
bit, mirrored onto both turn-merged sub-blocks exactly like the miss bit), change 2 (the
public sleep-clause block bits on the field token, with the full lifecycle from the spec's
acceptance section), v2.2 byte identity, and the v3 legacy-semantic compatibility view used
by the lifecycle fixtures. ``test_observation_v3_layout_cutover.py`` owns physical v3
permutation-map coverage. The incremental-fold twin retains handler + serialization
round-trip coverage, with pre-v3 payload bytes unchanged for fail-free games.
"""

import json
import os
import unittest
from dataclasses import replace
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
    NUMERIC_CONFUSION_TURNS,
    NUMERIC_ENCORE_TURNS,
    NUMERIC_GENDER_FEMALE,
    NUMERIC_GENDER_MALE,
    NUMERIC_MEANLOOK_TRAP,
    NUMERIC_OPP_WISH_PENDING,
    NUMERIC_OPP_WISH_TURNS,
    NUMERIC_SELF_WISH_PENDING,
    NUMERIC_SELF_WISH_TURNS,
    NUMERIC_SLEEP_CLAUSE_BLOCKS_OPP,
    NUMERIC_SLEEP_CLAUSE_BLOCKS_SELF,
    NUMERIC_STALL_COUNTER,
    NUMERIC_TM2_FAIL,
    NUMERIC_TM2_MISS,
    NUMERIC_TT_CONFUSION_SELFHIT,
    NUMERIC_TT_DAMAGE_FRACTION,
    NUMERIC_TT_FAIL,
    NUMERIC_TT_MISS,
    NUMERIC_WRAP_TRAP_TURNS,
    OPPONENT_POKEMON_TOKEN_OFFSET,
    REPLAY_OBSERVATION_SPECS_BY_SCHEMA,
    SELF_POKEMON_TOKEN_OFFSET,
    TRANSITION_TOKEN_OFFSET,
    V2_2_REPLAY_OBSERVATION_SPEC,
    V3_DROPPED_LEGACY_NUMERIC_INDICES,
    V3_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT,
    V3_NUMERIC_INDEX_BY_LEGACY_INDEX,
    V3_NUMERIC_LEGACY_INDEX_BY_NEW_INDEX,
    V3_REPLAY_OBSERVATION_SPEC,
    normalize_for_player,
    observation_from_player_state,
    observation_schema_version_from_choice,
    observation_spec_for_schema,
    parse_showdown_replay,
    v3_numeric_index,
)
from pokezero.showdown import _ReplayParser, _wish_pending, _wish_turns_remaining
from pokezero.transitions import TOKEN_KIND_MOVE, extract_transition_tokens
from pokezero.turn_merged import extract_turn_merged_tokens

SHOWDOWN_ROOT = Path(
    os.environ.get("POKEZERO_SHOWDOWN_ROOT", "/Users/scott/workspace/pokerena/vendor/pokemon-showdown")
)


def _legacy_v3_semantic_view(observation):
    """Expose named v3 features through their frozen writer positions for lifecycle tests.

    Product v3 rows are deliberately grouped into a new physical order. The cutover-map suite
    pins that physical order; this helper keeps the existing lifecycle fixtures focused on the
    semantic field they exercise rather than on an obsolete absolute position.
    """

    if observation.schema_version != OBSERVATION_SCHEMA_VERSION_V3:
        return observation
    rows = tuple(
        tuple(
            0.0
            if legacy_index in V3_DROPPED_LEGACY_NUMERIC_INDICES
            else row[V3_NUMERIC_INDEX_BY_LEGACY_INDEX[legacy_index]]
            for legacy_index in range(V3_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT)
        )
        for row in observation.numeric_features
    )
    return replace(observation, numeric_features=rows)

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

# ---- change 4: confusion turns-so-far. Signal Beam's 10% secondary is the ONLY gen3-randbats
# confusion source (venomoth carries it). Snorlax is confused on turn 1; it rides turns 2-3
# (elapsed 1, 2), snaps out via ``-end`` on turn 3, and turn 4 is clean again. ----
_CONFUSE_LEADS = [
    "|player|p1|Alice|",
    "|player|p2|Bob|",
    "|switch|p1a: Venomoth|Venomoth, L80|100/100",
    "|switch|p2a: Snorlax|Snorlax, L80|100/100",
    "|turn|1",
]
_CONFUSE_RIDE = _CONFUSE_LEADS + [
    "|move|p1a: Venomoth|Signal Beam|p2a: Snorlax",
    "|-damage|p2a: Snorlax|80/100",
    "|-start|p2a: Snorlax|confusion",
    "|upkeep",
    "|turn|2",
    "|-activate|p2a: Snorlax|confusion",
    "|upkeep",
    "|turn|3",
    "|-activate|p2a: Snorlax|confusion",
    "|-end|p2a: Snorlax|confusion",
    "|upkeep",
    "|turn|4",
]
# p1's OWN mon confused, with a |request| so the confused mon lands on a SELF-side token — the
# reveal-driven opponent token exercises the opponent write path, this the self write path.
_CONFUSE_SELF = [
    "|player|p1|Us|",
    "|player|p2|Them|",
    "|switch|p1a: Snorlax|Snorlax, L80|100/100",
    "|switch|p2a: Venomoth|Venomoth, L80|100/100",
    "|turn|1",
    "|move|p2a: Venomoth|Signal Beam|p1a: Snorlax",
    "|-damage|p1a: Snorlax|80/100",
    "|-start|p1a: Snorlax|confusion",
    "|upkeep",
    "|turn|2",
    '|request|{"active":[{"moves":[{"move":"Body Slam","id":"bodyslam"}]}],'
    '"side":{"id":"p1","name":"Us","pokemon":[{"ident":"p1a: Snorlax",'
    '"details":"Snorlax, L80","condition":"80/100","active":true}]}}',
]


# ---- change 5: encore turns-so-far. Wobbuffet is a gen3-randbats Encore carrier (16 total).
# Snorlax uses Body Slam turn 1, Wobbuffet locks it in with Encore (|-start|…|Encore); it rides
# turns 2-3 (elapsed 1, 2) repeating Body Slam, snaps out via ``-end`` on turn 3, turn 4 clean. ----
_ENCORE_LEADS = [
    "|player|p1|Alice|",
    "|player|p2|Bob|",
    "|switch|p1a: Wobbuffet|Wobbuffet, L80|100/100",
    "|switch|p2a: Snorlax|Snorlax, L80|100/100",
    "|turn|1",
]
_ENCORE_RIDE = _ENCORE_LEADS + [
    "|move|p2a: Snorlax|Body Slam|p1a: Wobbuffet",
    "|-damage|p1a: Wobbuffet|80/100",
    "|move|p1a: Wobbuffet|Encore|p2a: Snorlax",
    "|-start|p2a: Snorlax|Encore",
    "|upkeep",
    "|turn|2",
    "|move|p2a: Snorlax|Body Slam|p1a: Wobbuffet",
    "|-damage|p1a: Wobbuffet|60/100",
    "|upkeep",
    "|turn|3",
    "|move|p2a: Snorlax|Body Slam|p1a: Wobbuffet",
    "|-damage|p1a: Wobbuffet|40/100",
    "|-end|p2a: Snorlax|Encore",
    "|upkeep",
    "|turn|4",
]
# p1's OWN mon encored, with a |request| so the encored mon lands on a SELF-side token — the
# reveal-driven opponent token exercises the opponent write path, this the self write path.
_ENCORE_SELF = [
    "|player|p1|Us|",
    "|player|p2|Them|",
    "|switch|p1a: Snorlax|Snorlax, L80|100/100",
    "|switch|p2a: Wobbuffet|Wobbuffet, L80|100/100",
    "|turn|1",
    "|move|p1a: Snorlax|Body Slam|p2a: Wobbuffet",
    "|-damage|p2a: Wobbuffet|80/100",
    "|move|p2a: Wobbuffet|Encore|p1a: Snorlax",
    "|-start|p1a: Snorlax|Encore",
    "|upkeep",
    "|turn|2",
    '|request|{"active":[{"moves":[{"move":"Body Slam","id":"bodyslam"}]}],'
    '"side":{"id":"p1","name":"Us","pokemon":[{"ident":"p1a: Snorlax",'
    '"details":"Snorlax, L80","condition":"80/100","active":true}]}}',
]


# ---- change 6: Wrap (partial-trap) turns-so-far. Shuckle is the gen3-randbats pool's SOLE Wrap
# carrier (1 of 220). Shuckle pins Snorlax with Wrap (|-activate|…|move: Wrap|[of] …, no -start);
# the trap rides turns 2-3 (elapsed 1, 2) chipping via ``[from] Wrap`` residuals, expires via
# ``-end … Wrap [partiallytrapped]`` on turn 3, turn 4 clean. Wrap traps the TARGET. ----
_WRAP_LEADS = [
    "|player|p1|Alice|",
    "|player|p2|Bob|",
    "|switch|p1a: Shuckle|Shuckle, L80|100/100",
    "|switch|p2a: Snorlax|Snorlax, L80|100/100",
    "|turn|1",
]
_WRAP_RIDE = _WRAP_LEADS + [
    "|move|p1a: Shuckle|Wrap|p2a: Snorlax",
    "|-activate|p2a: Snorlax|move: Wrap|[of] p1a: Shuckle",
    "|-damage|p2a: Snorlax|94/100|[from] Wrap",
    "|upkeep",
    "|turn|2",
    "|move|p2a: Snorlax|Body Slam|p1a: Shuckle",
    "|-damage|p1a: Shuckle|80/100",
    "|-damage|p2a: Snorlax|88/100|[from] Wrap",
    "|upkeep",
    "|turn|3",
    "|move|p2a: Snorlax|Body Slam|p1a: Shuckle",
    "|-damage|p1a: Shuckle|60/100",
    "|-end|p2a: Snorlax|Wrap|[partiallytrapped]",
    "|upkeep",
    "|turn|4",
]
# p1's OWN mon trapped, with a |request| so the trapped mon lands on a SELF-side token — the
# reveal-driven opponent token exercises the opponent write path, this the self write path.
_WRAP_SELF = [
    "|player|p1|Us|",
    "|player|p2|Them|",
    "|switch|p1a: Snorlax|Snorlax, L80|100/100",
    "|switch|p2a: Shuckle|Shuckle, L80|100/100",
    "|turn|1",
    "|move|p2a: Shuckle|Wrap|p1a: Snorlax",
    "|-activate|p1a: Snorlax|move: Wrap|[of] p2a: Shuckle",
    "|-damage|p1a: Snorlax|94/100|[from] Wrap",
    "|upkeep",
    "|turn|2",
    '|request|{"active":[{"moves":[{"move":"Body Slam","id":"bodyslam"}]}],'
    '"side":{"id":"p1","name":"Us","pokemon":[{"ident":"p1a: Snorlax",'
    '"details":"Snorlax, L80","condition":"94/100","active":true}]}}',
]


# ---- change 9: Wish turns-to-land. p1's Vaporeon declares Wish on turn 1 (a per-SIDE slot
# condition); the field-token clock reads 1/2 at the turn-2 decision (the heal lands end of turn 2)
# and returns to 0 at turn 3 after the ``[from] move: Wish`` heal lands. Wish is a status move, so
# in normal play the FIRST post-declaration decision is turn 2 (1/2); the 2/2 value is only observed
# at an intra-turn decision on the declaration turn (exercised by the forced-switch wish-pass below).
_WISH_LEADS = [
    "|player|p1|Alice|",
    "|player|p2|Bob|",
    "|switch|p1a: Vaporeon|Vaporeon, L80|300/300",
    "|switch|p2a: Snorlax|Snorlax, L80|500/500",
    "|turn|1",
]
_WISH_RIDE = _WISH_LEADS + [
    "|move|p1a: Vaporeon|Wish|p1a: Vaporeon",
    "|move|p2a: Snorlax|Body Slam|p1a: Vaporeon",
    "|-damage|p1a: Vaporeon|250/300",
    "|upkeep",
    "|turn|2",
    "|move|p1a: Vaporeon|Protect|p1a: Vaporeon",
    "|-singleturn|p1a: Vaporeon|Protect",
    "|move|p2a: Snorlax|Body Slam|p1a: Vaporeon",
    "|-activate|p1a: Vaporeon|Protect",
    "|-heal|p1a: Vaporeon|400/300|[from] move: Wish",
    "|upkeep",
    "|turn|3",
]
# Wish-pass via a FORCED SWITCH on the declaration turn: Vaporeon Wishes turn 1, is KO'd the same
# turn, and p1 must switch. At the forced-switch |request| (turn_number still 1) the clock reads 2/2;
# the incoming Umbreon inherits the per-side slot and reads 1/2 at turn 2, then the heal lands on IT.
_WISH_PASS_FAINT = [
    "|player|p1|Alice|",
    "|player|p2|Bob|",
    "|switch|p1a: Vaporeon|Vaporeon, L80|100/100",
    "|switch|p2a: Snorlax|Snorlax, L80|500/500",
    "|turn|1",
    "|move|p1a: Vaporeon|Wish|p1a: Vaporeon",
    "|move|p2a: Snorlax|Body Slam|p1a: Vaporeon",
    "|-damage|p1a: Vaporeon|0 fnt",
    "|faint|p1a: Vaporeon",
    '|request|{"forceSwitch":[true],"side":{"id":"p1","name":"Alice","pokemon":['
    '{"ident":"p1a: Vaporeon","details":"Vaporeon, L80","condition":"0 fnt","active":true},'
    '{"ident":"p1: Umbreon","details":"Umbreon, L80","condition":"400/400","active":false}]}}',
]
# The turn-2 continuation after the forced switch resolves onto Umbreon.
_WISH_PASS_RIDE = _WISH_PASS_FAINT[:-1] + [
    "|switch|p1a: Umbreon|Umbreon, L80|400/400",
    "|upkeep",
    "|turn|2",
    "|move|p1a: Umbreon|Moonlight|p1a: Umbreon",
    "|move|p2a: Snorlax|Body Slam|p1a: Umbreon",
    "|-damage|p1a: Umbreon|300/400",
    "|-heal|p1a: Umbreon|500/400|[from] move: Wish",
    "|upkeep",
    "|turn|3",
]


def _through_turn(lines, turn):
    """The log prefix up to and including the ``|turn|<turn>`` decision boundary."""
    return lines[: lines.index(f"|turn|{turn}") + 1]


class SchemaTableTest(unittest.TestCase):
    def test_v3_is_supported_but_not_the_default(self) -> None:
        self.assertIn(OBSERVATION_SCHEMA_VERSION_V3, SUPPORTED_OBSERVATION_SCHEMA_VERSIONS)
        self.assertIn(OBSERVATION_SCHEMA_VERSION_V3, REPLAY_OBSERVATION_SPECS_BY_SCHEMA)
        self.assertIs(
            observation_spec_for_schema(OBSERVATION_SCHEMA_VERSION_V3),
            V3_REPLAY_OBSERVATION_SPEC,
        )
        # v2.2 keeps the fresh-selection default: v3 launches only after the fresh
        # golden corpus and EOC audit pass (spec coordination section).
        self.assertEqual(OBSERVATION_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION_V2_2)
        # v3 shares v2.2's turn-merged transition surface.
        self.assertIn(OBSERVATION_SCHEMA_VERSION_V3, TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS)
        self.assertIn(OBSERVATION_SCHEMA_VERSION_V2_2, TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS)

    def test_v3_widths_are_cutover_specific(self) -> None:
        # V3 removes fourteen source-unreachable numeric columns and reuses the resulting width
        # for its grouped layout. Matching v2.2's width is coincidental, not compatibility.
        self.assertEqual(V3_REPLAY_OBSERVATION_SPEC.numeric_feature_count, 155)
        self.assertEqual(
            V3_REPLAY_OBSERVATION_SPEC.numeric_feature_count,
            V2_2_REPLAY_OBSERVATION_SPEC.numeric_feature_count,
        )
        self.assertEqual(
            V3_REPLAY_OBSERVATION_SPEC.categorical_feature_count,
            V2_2_REPLAY_OBSERVATION_SPEC.categorical_feature_count,
        )
        self.assertEqual(
            V3_REPLAY_OBSERVATION_SPEC.token_count, V2_2_REPLAY_OBSERVATION_SPEC.token_count
        )

    def test_v3_column_layout(self) -> None:
        # The public map is the V3 spec artifact: every physical output index has exactly one
        # legacy writer source, and every legacy writer position is either mapped or explicitly
        # removed for reachability evidence.
        self.assertEqual(len(V3_NUMERIC_LEGACY_INDEX_BY_NEW_INDEX), 155)
        self.assertEqual(len(set(V3_NUMERIC_LEGACY_INDEX_BY_NEW_INDEX)), 155)
        self.assertEqual(
            set(V3_NUMERIC_LEGACY_INDEX_BY_NEW_INDEX) | V3_DROPPED_LEGACY_NUMERIC_INDICES,
            set(range(V3_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT)),
        )
        self.assertEqual(v3_numeric_index(NUMERIC_STALL_COUNTER), 32)
        self.assertEqual(v3_numeric_index(NUMERIC_CONFUSION_TURNS), 33)
        self.assertEqual(v3_numeric_index(NUMERIC_ENCORE_TURNS), 34)
        self.assertEqual(v3_numeric_index(NUMERIC_WRAP_TRAP_TURNS), 35)
        self.assertEqual(v3_numeric_index(NUMERIC_GENDER_MALE), 36)
        self.assertEqual(v3_numeric_index(NUMERIC_GENDER_FEMALE), 37)
        self.assertEqual(v3_numeric_index(NUMERIC_MEANLOOK_TRAP), 38)
        self.assertEqual(v3_numeric_index(NUMERIC_SELF_WISH_PENDING), 104)
        self.assertEqual(v3_numeric_index(NUMERIC_OPP_WISH_PENDING), 105)
        self.assertEqual(v3_numeric_index(NUMERIC_SLEEP_CLAUSE_BLOCKS_SELF), 106)
        self.assertEqual(v3_numeric_index(NUMERIC_SLEEP_CLAUSE_BLOCKS_OPP), 107)
        self.assertEqual(v3_numeric_index(NUMERIC_SELF_WISH_TURNS), 108)
        self.assertEqual(v3_numeric_index(NUMERIC_OPP_WISH_TURNS), 109)
        self.assertEqual(v3_numeric_index(NUMERIC_TT_DAMAGE_FRACTION), 121)
        self.assertEqual(v3_numeric_index(NUMERIC_TT_FAIL), 152)
        self.assertEqual(v3_numeric_index(NUMERIC_TM2_FAIL), 153)
        self.assertEqual(v3_numeric_index(NUMERIC_TT_CONFUSION_SELFHIT), 154)
        for legacy_index in V3_DROPPED_LEGACY_NUMERIC_INDICES:
            with self.assertRaisesRegex(ValueError, "dropped from v3"):
                v3_numeric_index(legacy_index)

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


class StallCounterTrackerTest(unittest.TestCase):
    """Change 3 lifecycle at the public-parser layer (spec acceptance item 3).

    One per-side counter = consecutive successful stall-move uses by that side's active mon.
    Mirrors the clause-lifecycle suite: increment across consecutive Protects; reset on each
    of the five causes; Endure shares the counter; both seats symmetric; snapshot round-trip.
    """

    def _protect(self, slot: str) -> list[str]:
        return [f"|move|{slot}a: X|Protect|{slot}a: X", f"|-singleturn|{slot}a: X|Protect", "|upkeep"]

    def _counter(self, lines, *, slot="p1"):
        return parse_showdown_replay(lines, battle_id="stall").stall_counter.get(slot, 0)

    def _state(self, lines, *, player="p1"):
        replay = parse_showdown_replay(lines, battle_id="stall")
        return normalize_for_player(replay, player_id=player, configured_showdown_slot=player)

    def test_counter_climbs_across_consecutive_protects(self) -> None:
        lines = list(_LEADS)
        for turn in range(1, 4):
            lines += self._protect("p1") + [f"|turn|{turn + 1}"]
            self.assertEqual(self._counter(lines), turn)

    def test_reset_on_failed_stall_move(self) -> None:
        # A failed Protect (the randomChance miss deletes the `stall` volatile) emits -fail and
        # no -singleturn — reset cause (1).
        lines = _LEADS + self._protect("p1") + [
            "|turn|2",
            "|move|p1a: X|Protect|p1a: X",
            "|-fail|p1a: X",
            "|turn|3",
        ]
        self.assertEqual(self._counter(lines), 0)

    def test_reset_on_non_stall_move(self) -> None:
        lines = _LEADS + self._protect("p1") + [
            "|turn|2",
            "|move|p1a: X|Spikes|p1a: X",
            "|turn|3",
        ]
        self.assertEqual(self._counter(lines), 0)

    def test_reset_on_cant(self) -> None:
        lines = _LEADS + self._protect("p1") + ["|turn|2", "|cant|p1a: X|par", "|turn|3"]
        self.assertEqual(self._counter(lines), 0)

    def test_reset_on_switch_out_and_drag(self) -> None:
        switched = _LEADS + self._protect("p1") + [
            "|turn|2",
            "|switch|p1a: Zapdos|Zapdos, L78|100/100",
        ]
        self.assertEqual(self._counter(switched), 0)
        dragged = _LEADS + self._protect("p1") + [
            "|turn|2",
            "|drag|p1a: Zapdos|Zapdos, L78|100/100",
        ]
        self.assertEqual(self._counter(dragged), 0)

    def test_reset_on_faint(self) -> None:
        lines = _LEADS + self._protect("p1") + ["|turn|2", "|faint|p1a: X", "|turn|3"]
        self.assertEqual(self._counter(lines), 0)

    def test_endure_shares_the_counter_with_protect(self) -> None:
        # Endure emits `-singleturn|SLOT|move: Endure`; it feeds the SAME streak as Protect.
        lines = _LEADS + [
            "|move|p1a: X|Endure|p1a: X",
            "|-singleturn|p1a: X|move: Endure",
            "|upkeep",
            "|turn|2",
        ] + self._protect("p1") + ["|turn|3"]
        self.assertEqual(self._counter(lines), 2)

    def test_both_seats_symmetric(self) -> None:
        # p2 stalls; the per-side scalars are symmetric across the two perspectives.
        lines = _LEADS + self._protect("p2") + [
            "|turn|2",
        ] + self._protect("p2") + ["|turn|3"]
        self.assertEqual(self._counter(lines, slot="p2"), 2)
        self.assertEqual(self._counter(lines, slot="p1"), 0)
        stalled_by_opp = self._state(lines, player="p1")
        self.assertEqual(stalled_by_opp.opponent_stall_counter, 2)
        self.assertEqual(stalled_by_opp.self_stall_counter, 0)
        stalled_view = self._state(lines, player="p2")
        self.assertEqual(stalled_view.self_stall_counter, 2)
        self.assertEqual(stalled_view.opponent_stall_counter, 0)

    def test_snapshot_round_trip_preserves_both_counters(self) -> None:
        lines = _LEADS + self._protect("p1") + ["|turn|2"] + self._protect("p1") + ["|turn|3"]
        # p2 also has a one-turn streak, so both sides are non-zero.
        lines = lines[: len(_LEADS)] + self._protect("p1") + ["|turn|2"] + self._protect("p1") \
            + self._protect("p2") + ["|turn|3"]
        replay = parse_showdown_replay(lines, battle_id="stall")
        self.assertEqual(replay.stall_counter.get("p1"), 2)
        self.assertEqual(replay.stall_counter.get("p2"), 1)
        resumed = _ReplayParser.from_snapshot(replay)
        # Resume and feed a p1 non-stall move: only p1 resets, p2's streak survives the round-trip.
        resumed.feed(["|move|p1a: X|Spikes|p1a: X", "|turn|4"])
        snap = resumed.snapshot()
        self.assertEqual(snap.stall_counter.get("p1"), 0)
        self.assertEqual(snap.stall_counter.get("p2"), 1)

    def test_pending_flag_round_trips_mid_action_window(self) -> None:
        # A snapshot taken between a stall |move| and its resolution restores the in-flight
        # flag, so a resumed -fail still resets (snapshot-vs-live convergence).
        mid = _LEADS + self._protect("p1") + ["|turn|2", "|move|p1a: X|Protect|p1a: X"]
        replay = parse_showdown_replay(mid, battle_id="stall")
        self.assertEqual(replay.stall_counter.get("p1"), 1)
        self.assertTrue(replay.stall_move_pending.get("p1"))
        resumed = _ReplayParser.from_snapshot(replay)
        resumed.feed(["|-fail|p1a: X"])
        self.assertEqual(resumed.snapshot().stall_counter.get("p1"), 0)


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
        return _legacy_v3_semantic_view(observation)

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

    def test_v2_2_fail_surface_matches_v3_legacy_semantics(self) -> None:
        # The raw v3 layout is a permutation. This test intentionally uses the legacy-semantic
        # view so lifecycle coverage stays focused on the fail bits; the cutover suite pins the
        # raw physical map and the v2.2 pristine suite pins legacy byte identity.
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
            self.assertEqual(len(v3_row), V3_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT)
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


@unittest.skipUnless(
    (SHOWDOWN_ROOT / "data" / "random-battles" / "gen3" / "sets.json").exists(),
    "requires a local Gen 3 Pokemon Showdown checkout",
)
class StallCounterEncodeTest(unittest.TestCase):
    """Change 3 at the encode layer: the counter lands on the ACTIVE mon token, rises
    1/8, 2/8, …, is written for both seats, exists under v3 only, and leaves the v2.2 prefix
    byte-identical (the cmp-against-pristine invariant applied to a Protect-heavy log)."""

    @staticmethod
    def _vocab():
        from pokezero.randbat_vocab import gen3_category_vocabulary

        return gen3_category_vocabulary(SHOWDOWN_ROOT, include_turn_merged=True)

    def _state(self, lines, *, player="p1"):
        replay = parse_showdown_replay(lines, battle_id="stall-encode")
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
        return _legacy_v3_semantic_view(observation)

    @staticmethod
    def _active_token(team, offset):
        for idx, mon in enumerate(team):
            if mon.active:
                return offset + idx
        raise AssertionError("no active mon in team")

    # Two consecutive Protects by the OPPONENT (populated from public reveals — no request
    # needed), viewed from the perspective whose opponent is the staller.
    def _opp_protect_lines(self, staller: str):
        other = "p2" if staller == "p1" else "p1"
        return _LEADS[:2] + [
            f"|switch|{staller}a: Skarmory|Skarmory, L76|100/100",
            f"|switch|{other}a: Snorlax|Snorlax, L80|100/100",
            "|turn|1",
            f"|move|{staller}a: Skarmory|Protect|{staller}a: Skarmory",
            f"|-singleturn|{staller}a: Skarmory|Protect",
            "|upkeep",
            "|turn|2",
            f"|move|{staller}a: Skarmory|Protect|{staller}a: Skarmory",
            f"|-singleturn|{staller}a: Skarmory|Protect",
            "|upkeep",
            "|turn|3",
        ]

    def test_counter_rises_on_the_opponent_active_token_both_seats(self) -> None:
        for staller, viewer in (("p1", "p2"), ("p2", "p1")):
            lines = self._opp_protect_lines(staller)
            state = self._state(lines, player=viewer)
            self.assertEqual(state.opponent_stall_counter, 2)
            obs = self._encode(state, V3_REPLAY_OBSERVATION_SPEC)
            tok = self._active_token(state.opponent_team, OPPONENT_POKEMON_TOKEN_OFFSET)
            # Two consecutive Protects -> 2/8 = 0.25 on the active token only.
            self.assertAlmostEqual(obs.numeric_features[tok][NUMERIC_STALL_COUNTER], 0.25)
            # Non-active opponent mons carry nothing.
            for idx in range(len(state.opponent_team)):
                token = OPPONENT_POKEMON_TOKEN_OFFSET + idx
                if token != tok:
                    self.assertEqual(obs.numeric_features[token][NUMERIC_STALL_COUNTER], 0.0)

    def test_counter_rises_one_eighth_then_two_eighths(self) -> None:
        base = self._opp_protect_lines("p1")
        after_one = base[:9] + ["|turn|2"]  # through the first Protect's -singleturn + upkeep
        state1 = self._state(after_one, player="p2")
        obs1 = self._encode(state1, V3_REPLAY_OBSERVATION_SPEC)
        tok1 = self._active_token(state1.opponent_team, OPPONENT_POKEMON_TOKEN_OFFSET)
        self.assertAlmostEqual(obs1.numeric_features[tok1][NUMERIC_STALL_COUNTER], 0.125)
        state2 = self._state(base, player="p2")
        obs2 = self._encode(state2, V3_REPLAY_OBSERVATION_SPEC)
        tok2 = self._active_token(state2.opponent_team, OPPONENT_POKEMON_TOKEN_OFFSET)
        self.assertAlmostEqual(obs2.numeric_features[tok2][NUMERIC_STALL_COUNTER], 0.25)

    def test_counter_on_the_self_active_token_via_request(self) -> None:
        # The self team is only known through the request; build one so the self active token
        # is populated and its v3 stall column fires.
        request = {
            "active": [{"moves": [{"move": "Protect", "id": "protect"}, {"move": "Drill Peck", "id": "drillpeck"}]}],
            "side": {
                "name": "p1",
                "id": "p1",
                "pokemon": [
                    {"ident": "p1: Skarmory", "details": "Skarmory, L76, M", "condition": "100/100", "active": True},
                    {"ident": "p1: Snorlax", "details": "Snorlax, L80, M", "condition": "100/100", "active": False},
                ],
            },
        }
        lines = _LEADS[:2] + [
            "|switch|p1a: Skarmory|Skarmory, L76, M|100/100",
            "|switch|p2a: Snorlax|Snorlax, L80|100/100",
            "|turn|1",
            "|move|p1a: Skarmory|Protect|p1a: Skarmory",
            "|-singleturn|p1a: Skarmory|Protect",
            "|upkeep",
            "|turn|2",
            "|move|p1a: Skarmory|Protect|p1a: Skarmory",
            "|-singleturn|p1a: Skarmory|Protect",
            "|upkeep",
            "|request|" + json.dumps(request, separators=(",", ":")),
            "|turn|3",
        ]
        state = self._state(lines, player="p1")
        self.assertEqual(state.self_stall_counter, 2)
        obs = self._encode(state, V3_REPLAY_OBSERVATION_SPEC)
        tok = self._active_token(state.self_team, SELF_POKEMON_TOKEN_OFFSET)
        self.assertAlmostEqual(obs.numeric_features[tok][NUMERIC_STALL_COUNTER], 0.25)

    def test_v2_2_protect_surface_matches_v3_legacy_semantics(self) -> None:
        # The raw v3 layout is a permutation, not a prefix. _encode deliberately returns the
        # legacy-semantic view for v3 so this lifecycle suite stays independent of physical
        # layout; the cutover suite verifies the actual v2.2-to-v3 permutation.
        lines = self._opp_protect_lines("p1")
        state = self._state(lines, player="p2")
        v2_2 = self._encode(state, V2_2_REPLAY_OBSERVATION_SPEC)
        v3 = self._encode(state, V3_REPLAY_OBSERVATION_SPEC)
        width = V2_2_REPLAY_OBSERVATION_SPEC.numeric_feature_count
        for row_index, (v22_row, v3_row) in enumerate(
            zip(v2_2.numeric_features, v3.numeric_features)
        ):
            self.assertEqual(len(v22_row), width)
            self.assertEqual(len(v3_row), V3_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT)
            self.assertEqual(tuple(v22_row), tuple(v3_row[:width]), f"numeric row {row_index}")
        self.assertEqual(
            [tuple(row) for row in v2_2.categorical_ids],
            [tuple(row) for row in v3.categorical_ids],
        )
        self.assertEqual(v2_2.attention_mask, v3.attention_mask)
        self.assertEqual(v2_2.token_type_ids, v3.token_type_ids)
        # The v3 stall column is the ONLY difference: exactly one active token carries it.
        tok = self._active_token(state.opponent_team, OPPONENT_POKEMON_TOKEN_OFFSET)
        self.assertAlmostEqual(v3.numeric_features[tok][NUMERIC_STALL_COUNTER], 0.25)


class IncrementalFoldParityTest(unittest.TestCase):
    """The incremental fold mirrors V3 history fields without changing clean payloads."""

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

    def test_incremental_fold_matches_batch_on_confusion_self_hit(self) -> None:
        from pokezero.transitions_fold import FoldState

        replay = parse_showdown_replay(_CONFUSE_SELFHIT_LINES, battle_id="confusion-fold")
        batch = extract_transition_tokens(replay, perspective_slot="p1")
        _, products = FoldState.initial(perspective_slot="p1").advance(
            _CONFUSE_SELFHIT_LINES
        )
        self.assertEqual(products.transition_tokens, batch)
        self.assertTrue(any(token.confusion_selfhit for token in products.transition_tokens))

    def test_payload_round_trip_carries_confusion_and_omits_it_when_clean(self) -> None:
        from pokezero.transitions_fold import FoldState

        clean, _ = FoldState.initial(perspective_slot="p1").advance(
            _CONFUSE_SELFHIT_LINES[: len(_LEADS) + 2]
        )
        self.assertNotIn(
            '"confusion_selfhit"', json.dumps(clean.to_payload(), sort_keys=True)
        )
        state, products = FoldState.initial(perspective_slot="p1").advance(
            _CONFUSE_SELFHIT_LINES
        )
        canonical = json.dumps(state.to_payload(), sort_keys=True)
        self.assertIn('"confusion_selfhit": true', canonical)
        resumed = FoldState.from_payload(json.loads(canonical))
        self.assertEqual(json.dumps(resumed.to_payload(), sort_keys=True), canonical)
        self.assertEqual(resumed.products().transition_tokens, products.transition_tokens)


class ConfusionElapsedTrackerTest(unittest.TestCase):
    """Change 4 lifecycle at the public-parser layer (spec acceptance item 4)."""

    def _replay(self, lines):
        return parse_showdown_replay(lines, battle_id="confusion")

    def test_counter_rises_each_turn_the_volatile_is_present(self) -> None:
        # 1 turn elapsed at turn 2, 2 at turn 3; the un-confused side never leaves 0.
        self.assertEqual(self._replay(_through_turn(_CONFUSE_RIDE, 2)).confusion_elapsed["p2"], 1)
        replay3 = self._replay(_through_turn(_CONFUSE_RIDE, 3))
        self.assertEqual(replay3.confusion_elapsed["p2"], 2)
        self.assertEqual(replay3.confusion_elapsed["p1"], 0)

    def test_end_confusion_resets_the_counter(self) -> None:
        # Turn 4 is past the turn-3 ``-end`` snap-out.
        self.assertEqual(self._replay(_through_turn(_CONFUSE_RIDE, 4)).confusion_elapsed["p2"], 0)

    def test_switch_out_resets_the_counter(self) -> None:
        lines = _CONFUSE_LEADS + [
            "|move|p1a: Venomoth|Signal Beam|p2a: Snorlax",
            "|-start|p2a: Snorlax|confusion",
            "|upkeep",
            "|turn|2",
            "|switch|p2a: Skarmory|Skarmory, L76|100/100",
            "|upkeep",
            "|turn|3",
        ]
        self.assertEqual(self._replay(lines).confusion_elapsed["p2"], 0)

    def test_faint_resets_the_counter(self) -> None:
        lines = _CONFUSE_LEADS + [
            "|move|p1a: Venomoth|Signal Beam|p2a: Snorlax",
            "|-start|p2a: Snorlax|confusion",
            "|upkeep",
            "|turn|2",
            "|faint|p2a: Snorlax",
            "|upkeep",
        ]
        self.assertEqual(self._replay(lines).confusion_elapsed["p2"], 0)

    def test_baton_pass_keeps_the_counter_on_the_inheritor(self) -> None:
        # Confusion is a Baton-Pass-copied volatile, so the switch-out reset is gated on the
        # volatile being absent: a BP that carried confusion keeps the counter climbing.
        lines = _CONFUSE_LEADS + [
            "|move|p1a: Venomoth|Signal Beam|p2a: Snorlax",
            "|-start|p2a: Snorlax|confusion",
            "|upkeep",
            "|turn|2",
            "|move|p2a: Snorlax|Baton Pass|p2a: Snorlax",
            "|switch|p2a: Smeargle|Smeargle, L83|100/100|[from] Baton Pass",
            "|upkeep",
            "|turn|3",
        ]
        replay = self._replay(lines)
        self.assertIn("confusion", replay.volatiles["p2"])
        self.assertEqual(replay.confusion_elapsed["p2"], 2)

    def test_snapshot_round_trip_preserves_elapsed(self) -> None:
        replay = self._replay(_through_turn(_CONFUSE_RIDE, 3))  # mid-confusion, elapsed 2
        self.assertEqual(replay.confusion_elapsed["p2"], 2)
        resumed = _ReplayParser.from_snapshot(replay)
        self.assertEqual(resumed.snapshot().confusion_elapsed["p2"], 2)
        # The reset still fires on the resumed tracker (state, not just the log prefix, carries it).
        resumed.feed(
            ["|-activate|p2a: Snorlax|confusion", "|-end|p2a: Snorlax|confusion", "|turn|4"]
        )
        self.assertEqual(resumed.snapshot().confusion_elapsed["p2"], 0)


@unittest.skipUnless(
    (SHOWDOWN_ROOT / "data" / "random-battles" / "gen3" / "sets.json").exists(),
    "requires a local Gen 3 Pokemon Showdown checkout",
)
class ConfusionEncodeTest(unittest.TestCase):
    """Change 4 at the encode layer: the column-pinned rise/reset on the confused mon's token,
    the reserved +4 sibling column staying zero, and the v2.2 byte-identity guard."""

    _RESERVED_STALL_COL = NUMERIC_STALL_COUNTER

    @staticmethod
    def _vocab():
        from pokezero.randbat_vocab import gen3_category_vocabulary

        return gen3_category_vocabulary(SHOWDOWN_ROOT, include_turn_merged=True)

    def _state(self, lines, *, player="p1"):
        replay = parse_showdown_replay(lines, battle_id="confusion-encode")
        return normalize_for_player(
            replay,
            player_id=player,
            configured_showdown_slot=player,
            format_id="gen3randombattle",
            include_turn_merged=True,
        )

    def _encode(self, state, spec):
        observation = observation_from_player_state(state, category_vocab=self._vocab(), spec=spec)
        observation.validate(spec)
        return _legacy_v3_semantic_view(observation)

    def _confusion_cells(self, observation):
        return [
            (index, row[NUMERIC_CONFUSION_TURNS])
            for index, row in enumerate(observation.numeric_features)
            if row[NUMERIC_CONFUSION_TURNS]
        ]

    def test_column_rises_then_resets_on_the_confused_opponent_token(self) -> None:
        from pokezero.showdown import OPPONENT_POKEMON_TOKEN_OFFSET

        for turn, want in ((2, 0.2), (3, 0.4)):
            observation = self._encode(
                self._state(_through_turn(_CONFUSE_RIDE, turn)), V3_REPLAY_OBSERVATION_SPEC
            )
            cells = self._confusion_cells(observation)
            # Column-position-pinned: exactly ONE token, at the opponent-active slot, in the
            # confusion column — and the reserved +4 stall column is untouched.
            self.assertEqual(len(cells), 1)
            self.assertEqual(cells[0][0], OPPONENT_POKEMON_TOKEN_OFFSET)
            self.assertAlmostEqual(cells[0][1], want)
            self.assertTrue(
                all(row[self._RESERVED_STALL_COL] == 0.0 for row in observation.numeric_features)
            )
        # Snap-out on turn 3 -> the column is empty at turn 4.
        observation = self._encode(
            self._state(_through_turn(_CONFUSE_RIDE, 4)), V3_REPLAY_OBSERVATION_SPEC
        )
        self.assertEqual(self._confusion_cells(observation), [])

    def test_column_fills_the_confused_self_active_token(self) -> None:
        # The self write path: p1's own confused Snorlax (via a request) carries the column.
        observation = self._encode(
            self._state(_CONFUSE_SELF, player="p1"), V3_REPLAY_OBSERVATION_SPEC
        )
        cells = self._confusion_cells(observation)
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0][0], SELF_POKEMON_TOKEN_OFFSET)
        self.assertAlmostEqual(cells[0][1], 0.2)  # 1 turn elapsed

    def test_switch_out_and_faint_zero_the_column(self) -> None:
        for tail in (
            ["|switch|p2a: Skarmory|Skarmory, L76|100/100", "|upkeep", "|turn|3"],
            ["|faint|p2a: Snorlax", "|switch|p2a: Skarmory|Skarmory, L76|100/100", "|upkeep", "|turn|3"],
        ):
            lines = _CONFUSE_LEADS + [
                "|move|p1a: Venomoth|Signal Beam|p2a: Snorlax",
                "|-start|p2a: Snorlax|confusion",
                "|upkeep",
                "|turn|2",
            ] + tail
            observation = self._encode(self._state(lines), V3_REPLAY_OBSERVATION_SPEC)
            self.assertEqual(self._confusion_cells(observation), [])

    def test_v2_2_confusion_surface_matches_v3_legacy_semantics(self) -> None:
        # NON-VACUOUS guard: at turn 3 the v3 encode DOES set the confusion column (0.4), so the
        # legacy-semantic compatibility assertion is meaningful.
        state = self._state(_through_turn(_CONFUSE_RIDE, 3))
        v2_2 = self._encode(state, V2_2_REPLAY_OBSERVATION_SPEC)
        v3 = self._encode(state, V3_REPLAY_OBSERVATION_SPEC)
        width = V2_2_REPLAY_OBSERVATION_SPEC.numeric_feature_count
        # The v3 encode is non-vacuous: the confusion column is actually populated.
        self.assertTrue(any(row[NUMERIC_CONFUSION_TURNS] for row in v3.numeric_features))
        # Under v2.2 the column does not exist, while its own output remains byte-identical.
        for row_index, (v22_row, v3_row) in enumerate(
            zip(v2_2.numeric_features, v3.numeric_features)
        ):
            self.assertEqual(len(v22_row), width)
            self.assertEqual(len(v3_row), V3_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT)
            self.assertEqual(tuple(v22_row), tuple(v3_row[:width]), f"numeric row {row_index}")
        self.assertEqual(
            [tuple(row) for row in v2_2.categorical_ids],
            [tuple(row) for row in v3.categorical_ids],
        )
        self.assertEqual(v2_2.attention_mask, v3.attention_mask)
        self.assertEqual(v2_2.token_type_ids, v3.token_type_ids)


class EncoreElapsedTrackerTest(unittest.TestCase):
    """Change 5 lifecycle at the public-parser layer (spec acceptance item 5)."""

    def _replay(self, lines):
        return parse_showdown_replay(lines, battle_id="encore")

    def test_counter_rises_each_turn_the_volatile_is_present(self) -> None:
        # 1 turn elapsed at turn 2, 2 at turn 3; the un-encored side never leaves 0.
        self.assertEqual(self._replay(_through_turn(_ENCORE_RIDE, 2)).encore_elapsed["p2"], 1)
        replay3 = self._replay(_through_turn(_ENCORE_RIDE, 3))
        self.assertEqual(replay3.encore_elapsed["p2"], 2)
        self.assertEqual(replay3.encore_elapsed["p1"], 0)

    def test_end_encore_resets_the_counter(self) -> None:
        # Turn 4 is past the turn-3 ``-end`` expiry.
        self.assertEqual(self._replay(_through_turn(_ENCORE_RIDE, 4)).encore_elapsed["p2"], 0)

    def test_switch_out_resets_the_counter(self) -> None:
        lines = _ENCORE_LEADS + [
            "|move|p2a: Snorlax|Body Slam|p1a: Wobbuffet",
            "|move|p1a: Wobbuffet|Encore|p2a: Snorlax",
            "|-start|p2a: Snorlax|Encore",
            "|upkeep",
            "|turn|2",
            "|switch|p2a: Skarmory|Skarmory, L76|100/100",
            "|upkeep",
            "|turn|3",
        ]
        self.assertEqual(self._replay(lines).encore_elapsed["p2"], 0)

    def test_drag_resets_the_counter(self) -> None:
        # Encore is noCopy: true, so a phazing |drag| (Whirlwind/Roar) also drops the volatile.
        lines = _ENCORE_LEADS + [
            "|move|p2a: Snorlax|Body Slam|p1a: Wobbuffet",
            "|move|p1a: Wobbuffet|Encore|p2a: Snorlax",
            "|-start|p2a: Snorlax|Encore",
            "|upkeep",
            "|turn|2",
            "|drag|p2a: Skarmory|Skarmory, L76|100/100",
            "|upkeep",
            "|turn|3",
        ]
        self.assertEqual(self._replay(lines).encore_elapsed["p2"], 0)

    def test_faint_resets_the_counter(self) -> None:
        lines = _ENCORE_LEADS + [
            "|move|p2a: Snorlax|Body Slam|p1a: Wobbuffet",
            "|move|p1a: Wobbuffet|Encore|p2a: Snorlax",
            "|-start|p2a: Snorlax|Encore",
            "|upkeep",
            "|turn|2",
            "|faint|p2a: Snorlax",
            "|upkeep",
        ]
        self.assertEqual(self._replay(lines).encore_elapsed["p2"], 0)

    def test_snapshot_round_trip_preserves_elapsed(self) -> None:
        replay = self._replay(_through_turn(_ENCORE_RIDE, 3))  # mid-encore, elapsed 2
        self.assertEqual(replay.encore_elapsed["p2"], 2)
        resumed = _ReplayParser.from_snapshot(replay)
        self.assertEqual(resumed.snapshot().encore_elapsed["p2"], 2)
        # The reset still fires on the resumed tracker (state, not just the log prefix, carries it).
        resumed.feed(["|-end|p2a: Snorlax|Encore", "|turn|4"])
        self.assertEqual(resumed.snapshot().encore_elapsed["p2"], 0)


@unittest.skipUnless(
    (SHOWDOWN_ROOT / "data" / "random-battles" / "gen3" / "sets.json").exists(),
    "requires a local Gen 3 Pokemon Showdown checkout",
)
class EncoreEncodeTest(unittest.TestCase):
    """Change 5 at the encode layer: the column-pinned rise/reset on the encored mon's token,
    the sibling stall/confusion columns staying zero, and the v2.2 byte-identity guard."""

    _STALL_COL = NUMERIC_STALL_COUNTER
    _CONFUSION_COL = NUMERIC_CONFUSION_TURNS

    @staticmethod
    def _vocab():
        from pokezero.randbat_vocab import gen3_category_vocabulary

        return gen3_category_vocabulary(SHOWDOWN_ROOT, include_turn_merged=True)

    def _state(self, lines, *, player="p1"):
        replay = parse_showdown_replay(lines, battle_id="encore-encode")
        return normalize_for_player(
            replay,
            player_id=player,
            configured_showdown_slot=player,
            format_id="gen3randombattle",
            include_turn_merged=True,
        )

    def _encode(self, state, spec):
        observation = observation_from_player_state(state, category_vocab=self._vocab(), spec=spec)
        observation.validate(spec)
        return _legacy_v3_semantic_view(observation)

    def _encore_cells(self, observation):
        return [
            (index, row[NUMERIC_ENCORE_TURNS])
            for index, row in enumerate(observation.numeric_features)
            if row[NUMERIC_ENCORE_TURNS]
        ]

    def test_column_rises_then_resets_on_the_encored_opponent_token(self) -> None:
        for turn, want in ((2, 1 / 6), (3, 2 / 6)):
            observation = self._encode(
                self._state(_through_turn(_ENCORE_RIDE, turn)), V3_REPLAY_OBSERVATION_SPEC
            )
            cells = self._encore_cells(observation)
            # Column-position-pinned: exactly ONE token, at the opponent-active slot, in the
            # encore column — and the sibling stall (+4) and confusion (+5) columns are untouched.
            self.assertEqual(len(cells), 1)
            self.assertEqual(cells[0][0], OPPONENT_POKEMON_TOKEN_OFFSET)
            self.assertAlmostEqual(cells[0][1], want)
            self.assertTrue(
                all(row[self._STALL_COL] == 0.0 for row in observation.numeric_features)
            )
            self.assertTrue(
                all(row[self._CONFUSION_COL] == 0.0 for row in observation.numeric_features)
            )
        # Expiry on turn 3 -> the column is empty at turn 4.
        observation = self._encode(
            self._state(_through_turn(_ENCORE_RIDE, 4)), V3_REPLAY_OBSERVATION_SPEC
        )
        self.assertEqual(self._encore_cells(observation), [])

    def test_column_fills_the_encored_self_active_token(self) -> None:
        # The self write path: p1's own encored Snorlax (via a request) carries the column.
        observation = self._encode(
            self._state(_ENCORE_SELF, player="p1"), V3_REPLAY_OBSERVATION_SPEC
        )
        cells = self._encore_cells(observation)
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0][0], SELF_POKEMON_TOKEN_OFFSET)
        self.assertAlmostEqual(cells[0][1], 1 / 6)  # 1 turn elapsed

    def test_switch_out_drag_and_faint_zero_the_column(self) -> None:
        for tail in (
            ["|switch|p2a: Skarmory|Skarmory, L76|100/100", "|upkeep", "|turn|3"],
            ["|drag|p2a: Skarmory|Skarmory, L76|100/100", "|upkeep", "|turn|3"],
            ["|faint|p2a: Snorlax", "|switch|p2a: Skarmory|Skarmory, L76|100/100", "|upkeep", "|turn|3"],
        ):
            lines = _ENCORE_LEADS + [
                "|move|p2a: Snorlax|Body Slam|p1a: Wobbuffet",
                "|move|p1a: Wobbuffet|Encore|p2a: Snorlax",
                "|-start|p2a: Snorlax|Encore",
                "|upkeep",
                "|turn|2",
            ] + tail
            observation = self._encode(self._state(lines), V3_REPLAY_OBSERVATION_SPEC)
            self.assertEqual(self._encore_cells(observation), [])

    def test_v2_2_encore_surface_matches_v3_legacy_semantics(self) -> None:
        # NON-VACUOUS guard: at turn 3 the v3 encode DOES set the encore column (2/6), so the
        # legacy-semantic compatibility assertion is meaningful.
        state = self._state(_through_turn(_ENCORE_RIDE, 3))
        v2_2 = self._encode(state, V2_2_REPLAY_OBSERVATION_SPEC)
        v3 = self._encode(state, V3_REPLAY_OBSERVATION_SPEC)
        width = V2_2_REPLAY_OBSERVATION_SPEC.numeric_feature_count
        # The v3 encode is non-vacuous: the encore column is actually populated.
        self.assertTrue(any(row[NUMERIC_ENCORE_TURNS] for row in v3.numeric_features))
        # Under v2.2 the column does not exist, while its own output remains byte-identical.
        for row_index, (v22_row, v3_row) in enumerate(
            zip(v2_2.numeric_features, v3.numeric_features)
        ):
            self.assertEqual(len(v22_row), width)
            self.assertEqual(len(v3_row), V3_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT)
            self.assertEqual(tuple(v22_row), tuple(v3_row[:width]), f"numeric row {row_index}")
        self.assertEqual(
            [tuple(row) for row in v2_2.categorical_ids],
            [tuple(row) for row in v3.categorical_ids],
        )
        self.assertEqual(v2_2.attention_mask, v3.attention_mask)
        self.assertEqual(v2_2.token_type_ids, v3.token_type_ids)


class WrapTrapElapsedTrackerTest(unittest.TestCase):
    """Change 6 lifecycle at the public-parser layer (spec acceptance item 6)."""

    def _replay(self, lines):
        return parse_showdown_replay(lines, battle_id="wrap")

    def test_counter_rises_each_turn_the_volatile_is_present(self) -> None:
        # 1 turn elapsed at turn 2, 2 at turn 3; the un-trapped side never leaves 0.
        self.assertEqual(self._replay(_through_turn(_WRAP_RIDE, 2)).wrap_trap_elapsed["p2"], 1)
        replay3 = self._replay(_through_turn(_WRAP_RIDE, 3))
        self.assertEqual(replay3.wrap_trap_elapsed["p2"], 2)
        self.assertEqual(replay3.wrap_trap_elapsed["p1"], 0)

    def test_end_wrap_resets_the_counter(self) -> None:
        # Turn 4 is past the turn-3 ``-end … Wrap [partiallytrapped]`` expiry.
        self.assertEqual(self._replay(_through_turn(_WRAP_RIDE, 4)).wrap_trap_elapsed["p2"], 0)

    def test_switch_out_resets_the_counter(self) -> None:
        # The trapped mon leaving its slot drops the (Baton-Pass-copyable) volatile; a plain switch
        # additionally guards the parse-loop reset path (drag/faint are the realistic exits).
        lines = _WRAP_LEADS + [
            "|move|p1a: Shuckle|Wrap|p2a: Snorlax",
            "|-activate|p2a: Snorlax|move: Wrap|[of] p1a: Shuckle",
            "|upkeep",
            "|turn|2",
            "|switch|p2a: Skarmory|Skarmory, L76|100/100",
            "|upkeep",
            "|turn|3",
        ]
        self.assertEqual(self._replay(lines).wrap_trap_elapsed["p2"], 0)

    def test_drag_resets_the_counter(self) -> None:
        # A phazing |drag| (Whirlwind/Roar) forces the trapped mon out and drops the volatile.
        lines = _WRAP_LEADS + [
            "|move|p1a: Shuckle|Wrap|p2a: Snorlax",
            "|-activate|p2a: Snorlax|move: Wrap|[of] p1a: Shuckle",
            "|upkeep",
            "|turn|2",
            "|drag|p2a: Skarmory|Skarmory, L76|100/100",
            "|upkeep",
            "|turn|3",
        ]
        self.assertEqual(self._replay(lines).wrap_trap_elapsed["p2"], 0)

    def test_faint_resets_the_counter(self) -> None:
        lines = _WRAP_LEADS + [
            "|move|p1a: Shuckle|Wrap|p2a: Snorlax",
            "|-activate|p2a: Snorlax|move: Wrap|[of] p1a: Shuckle",
            "|upkeep",
            "|turn|2",
            "|faint|p2a: Snorlax",
            "|upkeep",
        ]
        self.assertEqual(self._replay(lines).wrap_trap_elapsed["p2"], 0)

    def test_snapshot_round_trip_preserves_elapsed(self) -> None:
        replay = self._replay(_through_turn(_WRAP_RIDE, 3))  # mid-trap, elapsed 2
        self.assertEqual(replay.wrap_trap_elapsed["p2"], 2)
        resumed = _ReplayParser.from_snapshot(replay)
        self.assertEqual(resumed.snapshot().wrap_trap_elapsed["p2"], 2)
        # The reset still fires on the resumed tracker (state, not just the log prefix, carries it).
        resumed.feed(["|-end|p2a: Snorlax|Wrap|[partiallytrapped]", "|turn|4"])
        self.assertEqual(resumed.snapshot().wrap_trap_elapsed["p2"], 0)


@unittest.skipUnless(
    (SHOWDOWN_ROOT / "data" / "random-battles" / "gen3" / "sets.json").exists(),
    "requires a local Gen 3 Pokemon Showdown checkout",
)
class WrapTrapEncodeTest(unittest.TestCase):
    """Change 6 at the encode layer: the column-pinned rise/reset on the trapped mon's token,
    the sibling stall/confusion/encore columns staying zero, and the v2.2 byte-identity guard."""

    _STALL_COL = NUMERIC_STALL_COUNTER
    _CONFUSION_COL = NUMERIC_CONFUSION_TURNS
    _ENCORE_COL = NUMERIC_ENCORE_TURNS

    @staticmethod
    def _vocab():
        from pokezero.randbat_vocab import gen3_category_vocabulary

        return gen3_category_vocabulary(SHOWDOWN_ROOT, include_turn_merged=True)

    def _state(self, lines, *, player="p1"):
        replay = parse_showdown_replay(lines, battle_id="wrap-encode")
        return normalize_for_player(
            replay,
            player_id=player,
            configured_showdown_slot=player,
            format_id="gen3randombattle",
            include_turn_merged=True,
        )

    def _encode(self, state, spec):
        observation = observation_from_player_state(state, category_vocab=self._vocab(), spec=spec)
        observation.validate(spec)
        return _legacy_v3_semantic_view(observation)

    def _wrap_cells(self, observation):
        return [
            (index, row[NUMERIC_WRAP_TRAP_TURNS])
            for index, row in enumerate(observation.numeric_features)
            if row[NUMERIC_WRAP_TRAP_TURNS]
        ]

    def test_column_rises_then_resets_on_the_trapped_opponent_token(self) -> None:
        for turn, want in ((2, 1 / 5), (3, 2 / 5)):
            observation = self._encode(
                self._state(_through_turn(_WRAP_RIDE, turn)), V3_REPLAY_OBSERVATION_SPEC
            )
            cells = self._wrap_cells(observation)
            # Column-position-pinned: exactly ONE token, at the opponent-active slot, in the
            # wrap-trap column — and the sibling stall (+4), confusion (+5), encore (+6) columns
            # are untouched.
            self.assertEqual(len(cells), 1)
            self.assertEqual(cells[0][0], OPPONENT_POKEMON_TOKEN_OFFSET)
            self.assertAlmostEqual(cells[0][1], want)
            self.assertTrue(
                all(row[self._STALL_COL] == 0.0 for row in observation.numeric_features)
            )
            self.assertTrue(
                all(row[self._CONFUSION_COL] == 0.0 for row in observation.numeric_features)
            )
            self.assertTrue(
                all(row[self._ENCORE_COL] == 0.0 for row in observation.numeric_features)
            )
        # Expiry on turn 3 -> the column is empty at turn 4.
        observation = self._encode(
            self._state(_through_turn(_WRAP_RIDE, 4)), V3_REPLAY_OBSERVATION_SPEC
        )
        self.assertEqual(self._wrap_cells(observation), [])

    def test_column_fills_the_trapped_self_active_token(self) -> None:
        # The self write path: p1's own trapped Snorlax (via a request) carries the column.
        observation = self._encode(
            self._state(_WRAP_SELF, player="p1"), V3_REPLAY_OBSERVATION_SPEC
        )
        cells = self._wrap_cells(observation)
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0][0], SELF_POKEMON_TOKEN_OFFSET)
        self.assertAlmostEqual(cells[0][1], 1 / 5)  # 1 turn elapsed

    def test_switch_out_drag_and_faint_zero_the_column(self) -> None:
        for tail in (
            ["|switch|p2a: Skarmory|Skarmory, L76|100/100", "|upkeep", "|turn|3"],
            ["|drag|p2a: Skarmory|Skarmory, L76|100/100", "|upkeep", "|turn|3"],
            ["|faint|p2a: Snorlax", "|switch|p2a: Skarmory|Skarmory, L76|100/100", "|upkeep", "|turn|3"],
        ):
            lines = _WRAP_LEADS + [
                "|move|p1a: Shuckle|Wrap|p2a: Snorlax",
                "|-activate|p2a: Snorlax|move: Wrap|[of] p1a: Shuckle",
                "|upkeep",
                "|turn|2",
            ] + tail
            observation = self._encode(self._state(lines), V3_REPLAY_OBSERVATION_SPEC)
            self.assertEqual(self._wrap_cells(observation), [])

    def test_v2_2_wrap_surface_matches_v3_legacy_semantics(self) -> None:
        # NON-VACUOUS guard: at turn 3 the v3 encode DOES set the wrap column (2/5), so the
        # legacy-semantic compatibility assertion is meaningful.
        state = self._state(_through_turn(_WRAP_RIDE, 3))
        v2_2 = self._encode(state, V2_2_REPLAY_OBSERVATION_SPEC)
        v3 = self._encode(state, V3_REPLAY_OBSERVATION_SPEC)
        width = V2_2_REPLAY_OBSERVATION_SPEC.numeric_feature_count
        # The v3 encode is non-vacuous: the wrap-trap column is actually populated.
        self.assertTrue(any(row[NUMERIC_WRAP_TRAP_TURNS] for row in v3.numeric_features))
        # Under v2.2 the column does not exist, while its own output remains byte-identical.
        for row_index, (v22_row, v3_row) in enumerate(
            zip(v2_2.numeric_features, v3.numeric_features)
        ):
            self.assertEqual(len(v22_row), width)
            self.assertEqual(len(v3_row), V3_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT)
            self.assertEqual(tuple(v22_row), tuple(v3_row[:width]), f"numeric row {row_index}")
        self.assertEqual(
            [tuple(row) for row in v2_2.categorical_ids],
            [tuple(row) for row in v3.categorical_ids],
        )
        self.assertEqual(v2_2.attention_mask, v3.attention_mask)
        self.assertEqual(v2_2.token_type_ids, v3.token_type_ids)


class WishTurnsTrackerTest(unittest.TestCase):
    """Change 9 lifecycle at the public-parser layer (spec acceptance item 9): the per-side
    turns-to-land clock re-derived from ``wish_set_turns`` reads 2 the declaration turn, 1 the
    landing turn, 0 otherwise — and is nonzero on exactly the turns the v2.2 pending bit is set."""

    def _remaining(self, lines, slot="p1"):
        return _wish_turns_remaining(parse_showdown_replay(lines, battle_id="wish"), slot)

    def test_two_then_one_then_zero(self) -> None:
        # Mid declaration-turn (before |turn|2 is fed): 2. Turn 2 (the landing turn): 1. Turn 3
        # (after the heal landed and popped wish_set_turns): 0.
        mid = _WISH_LEADS + ["|move|p1a: Vaporeon|Wish|p1a: Vaporeon"]
        self.assertEqual(self._remaining(mid), 2)
        self.assertEqual(self._remaining(_through_turn(_WISH_RIDE, 2)), 1)
        self.assertEqual(self._remaining(_through_turn(_WISH_RIDE, 3)), 0)

    def test_zero_when_no_wish_pending(self) -> None:
        self.assertEqual(self._remaining(_WISH_LEADS), 0)
        # The un-wishing side never leaves 0.
        self.assertEqual(self._remaining(_through_turn(_WISH_RIDE, 2), slot="p2"), 0)

    def test_nonzero_exactly_when_pending_bit_is_set(self) -> None:
        # The v3 turns clock and the v2.2 pending predicate agree about presence at every boundary.
        for prefix in (_WISH_LEADS,
                       _WISH_LEADS + ["|move|p1a: Vaporeon|Wish|p1a: Vaporeon"],
                       _through_turn(_WISH_RIDE, 2),
                       _through_turn(_WISH_RIDE, 3)):
            replay = parse_showdown_replay(prefix, battle_id="wish")
            self.assertEqual(bool(_wish_turns_remaining(replay, "p1")), _wish_pending(replay, "p1"))

    def test_wish_pass_keeps_the_clock_on_the_side_across_a_switch(self) -> None:
        # Per-slot (keyed on side, not mon): the forced switch after the wisher faints keeps the
        # clock. Mid-turn-1 (forced-switch request) = 2; turn 2 on the inheritor = 1; turn 3 = 0.
        self.assertEqual(self._remaining(_WISH_PASS_FAINT), 2)
        self.assertEqual(self._remaining(_through_turn(_WISH_PASS_RIDE, 2)), 1)
        self.assertEqual(self._remaining(_through_turn(_WISH_PASS_RIDE, 3)), 0)

    def test_re_declared_wish_while_pending_does_not_re_arm(self) -> None:
        # Gen3: a Wish declared while one is already pending FAILS and must not reset the clock.
        lines = _through_turn(_WISH_RIDE, 2) + ["|move|p1a: Vaporeon|Wish|p1a: Vaporeon"]
        # Still keyed to the original turn-1 declaration: remaining stays 1 (not re-armed to 2).
        self.assertEqual(self._remaining(lines), 1)


@unittest.skipUnless(
    (SHOWDOWN_ROOT / "data" / "random-battles" / "gen3" / "sets.json").exists(),
    "requires a local Gen 3 Pokemon Showdown checkout",
)
class WishTurnsEncodeTest(unittest.TestCase):
    """Change 9 at the encode layer: the per-side clock lands on the FIELD token, reads 2/2 then
    1/2 then 0 across a Wish's life, on both seats, persists across a wish-pass switch, leaves the
    sibling v3 columns and the v2.2 pending bits alone, and keeps the v2.2 prefix byte-identical."""

    @staticmethod
    def _vocab():
        from pokezero.randbat_vocab import gen3_category_vocabulary

        return gen3_category_vocabulary(SHOWDOWN_ROOT, include_turn_merged=True)

    def _state(self, lines, *, player="p1"):
        replay = parse_showdown_replay(lines, battle_id="wish-encode")
        return normalize_for_player(
            replay,
            player_id=player,
            configured_showdown_slot=player,
            format_id="gen3randombattle",
            include_turn_merged=True,
        )

    def _encode(self, state, spec):
        observation = observation_from_player_state(state, category_vocab=self._vocab(), spec=spec)
        observation.validate(spec)
        return _legacy_v3_semantic_view(observation)

    def _field(self, observation):
        return observation.numeric_features[FIELD_TOKEN_OFFSET]

    def test_self_clock_reads_one_half_then_zero_on_the_field_token(self) -> None:
        # Turn 2 (the landing turn): the wishing side reads 1/2; the opponent column stays 0.
        turn2 = self._encode(self._state(_through_turn(_WISH_RIDE, 2)), V3_REPLAY_OBSERVATION_SPEC)
        self.assertAlmostEqual(self._field(turn2)[NUMERIC_SELF_WISH_TURNS], 0.5)
        self.assertEqual(self._field(turn2)[NUMERIC_OPP_WISH_TURNS], 0.0)
        # Only ONE token (the field token) carries the column.
        self.assertEqual(
            [i for i, row in enumerate(turn2.numeric_features) if row[NUMERIC_SELF_WISH_TURNS]],
            [FIELD_TOKEN_OFFSET],
        )
        # The v2.2 pending bit agrees (56 set), and the sibling v3 columns are untouched.
        self.assertEqual(self._field(turn2)[NUMERIC_SELF_WISH_PENDING], 1.0)
        for col in (NUMERIC_STALL_COUNTER, NUMERIC_CONFUSION_TURNS, NUMERIC_ENCORE_TURNS,
                    NUMERIC_WRAP_TRAP_TURNS, NUMERIC_MEANLOOK_TRAP):
            self.assertTrue(all(row[col] == 0.0 for row in turn2.numeric_features))
        # Turn 3: the Wish landed -> the column (and the pending bit) return to 0.
        turn3 = self._encode(self._state(_through_turn(_WISH_RIDE, 3)), V3_REPLAY_OBSERVATION_SPEC)
        self.assertEqual(self._field(turn3)[NUMERIC_SELF_WISH_TURNS], 0.0)
        self.assertEqual(self._field(turn3)[NUMERIC_SELF_WISH_PENDING], 0.0)

    def test_opponent_reads_the_column_from_the_other_seat(self) -> None:
        # From p2's perspective p1's Wish is the OPPONENT's: opp column set, self column clear.
        turn2 = self._encode(
            self._state(_through_turn(_WISH_RIDE, 2), player="p2"), V3_REPLAY_OBSERVATION_SPEC
        )
        self.assertAlmostEqual(self._field(turn2)[NUMERIC_OPP_WISH_TURNS], 0.5)
        self.assertEqual(self._field(turn2)[NUMERIC_SELF_WISH_TURNS], 0.0)

    def test_two_over_two_at_the_forced_switch_and_persists_to_the_wish_pass_recipient(self) -> None:
        # Intra-turn-1 forced-switch decision: the clock reads 2/2 (2 turns remaining) even though
        # the wisher just fainted — the per-side slot condition outlives the mon.
        faint = self._encode(self._state(_WISH_PASS_FAINT), V3_REPLAY_OBSERVATION_SPEC)
        self.assertAlmostEqual(self._field(faint)[NUMERIC_SELF_WISH_TURNS], 1.0)
        # Turn 2: the incoming Umbreon inherits the slot and reads 1/2 ("the heal lands on me").
        turn2 = self._encode(
            self._state(_through_turn(_WISH_PASS_RIDE, 2)), V3_REPLAY_OBSERVATION_SPEC
        )
        self.assertAlmostEqual(self._field(turn2)[NUMERIC_SELF_WISH_TURNS], 0.5)

    def test_v2_2_wish_surface_matches_v3_legacy_semantics(self) -> None:
        # NON-VACUOUS guard: at turn 2 the v3 encode DOES set the wish column (1/2), so the invariant
        # (v2.2 output unchanged; v3 carries the field through the legacy semantic view) is meaningful. The v2.2
        # pending bit (56) is present in BOTH encodes and stays byte-identical.
        state = self._state(_through_turn(_WISH_RIDE, 2))
        v2_2 = self._encode(state, V2_2_REPLAY_OBSERVATION_SPEC)
        v3 = self._encode(state, V3_REPLAY_OBSERVATION_SPEC)
        width = V2_2_REPLAY_OBSERVATION_SPEC.numeric_feature_count
        self.assertTrue(any(row[NUMERIC_SELF_WISH_TURNS] for row in v3.numeric_features))
        self.assertEqual(v2_2.numeric_features[FIELD_TOKEN_OFFSET][NUMERIC_SELF_WISH_PENDING], 1.0)
        for row_index, (v22_row, v3_row) in enumerate(
            zip(v2_2.numeric_features, v3.numeric_features)
        ):
            self.assertEqual(len(v22_row), width)
            self.assertEqual(len(v3_row), V3_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT)
            self.assertEqual(tuple(v22_row), tuple(v3_row[:width]), f"numeric row {row_index}")
        self.assertEqual(
            [tuple(row) for row in v2_2.categorical_ids],
            [tuple(row) for row in v3.categorical_ids],
        )
        self.assertEqual(v2_2.attention_mask, v3.attention_mask)
        self.assertEqual(v2_2.token_type_ids, v3.token_type_ids)


# Change 10 fixture: p2 Skarmory (faster) attacks p1 Snorlax for 0.17, then the SLOWER
# confused Snorlax self-hits for 0.10 with an UNTAGGED -damage and no |move|/|cant| line —
# so the fold folds 0.10 into Skarmory's move damage (0.27) exactly as the change brief
# reproduces. Skarmory (base speed 70) outspeeds Snorlax (30), so it is the FIRST sub-block.
_CONFUSE_SELFHIT_LINES = _LEADS + [
    "|move|p2a: Skarmory|Drill Peck|p1a: Snorlax",
    "|-damage|p1a: Snorlax|83/100",  # opponent move = 0.17
    "|-activate|p1a: Snorlax|confusion",
    "|-damage|p1a: Snorlax|73/100",  # confused self-hit = 0.10 (untagged)
    "|upkeep",
    "|turn|2",
]


@unittest.skipUnless(
    (SHOWDOWN_ROOT / "data" / "random-battles" / "gen3" / "sets.json").exists(),
    "requires a local Gen 3 Pokemon Showdown checkout",
)
class ConfusionSelfHitEncodeTest(unittest.TestCase):
    """Change 10 at the encode layer: v2.2 keeps the folded damage (0.27) byte-identically;
    v3 subtracts the confused defender's self-hit back out (0.17) and sets the flag. Both
    seats, non-vacuous (the two encodes DISAGREE on the damage column), v2.2 prefix frozen."""

    @staticmethod
    def _vocab():
        from pokezero.randbat_vocab import gen3_category_vocabulary

        return gen3_category_vocabulary(SHOWDOWN_ROOT, include_turn_merged=True)

    def _state(self, lines, *, player="p1"):
        replay = parse_showdown_replay(lines, battle_id="confuse-selfhit-encode")
        return normalize_for_player(
            replay,
            player_id=player,
            configured_showdown_slot=player,
            format_id="gen3randombattle",
            include_turn_merged=True,
        )

    def _encode(self, state, spec):
        observation = observation_from_player_state(state, category_vocab=self._vocab(), spec=spec)
        observation.validate(spec)
        return _legacy_v3_semantic_view(observation)

    def test_v3_corrects_the_damage_and_sets_the_flag_v2_2_keeps_the_folded_value(self) -> None:
        state = self._state(_CONFUSE_SELFHIT_LINES)
        turn1 = TRANSITION_TOKEN_OFFSET + 1  # lead pair at +0, turn 1 at +1
        v2_2 = self._encode(state, V2_2_REPLAY_OBSERVATION_SPEC)
        v3 = self._encode(state, V3_REPLAY_OBSERVATION_SPEC)
        # v2.2 (frozen): the self-hit is still folded into the move's damage column (0.27).
        self.assertAlmostEqual(v2_2.numeric_features[turn1][NUMERIC_TT_DAMAGE_FRACTION], 0.27)
        # v3: the move's OWN damage (self-hit removed) is 0.17, and the flag is set. This is a
        # NON-VACUOUS guard — without the fix v3 would read 0.27 too and this assertion fails.
        self.assertAlmostEqual(v3.numeric_features[turn1][NUMERIC_TT_DAMAGE_FRACTION], 0.17)
        self.assertEqual(v3.numeric_features[turn1][NUMERIC_TT_CONFUSION_SELFHIT], 1.0)
        # The flag is set on EXACTLY that one row.
        self.assertEqual(
            [i for i, row in enumerate(v3.numeric_features) if row[NUMERIC_TT_CONFUSION_SELFHIT]],
            [turn1],
        )

    def test_v2_2_is_frozen_and_v3_diverges_only_at_the_corrected_damage_column(self) -> None:
        # The semantic view makes the single intentional v3 rewrite explicit: v2.2 retains the
        # folded value, while v3 subtracts the confusion self-hit. The raw permutation-map test
        # owns physical positions; the v2.2 cmp-against-pristine battery owns byte identity.
        for player in ("p1", "p2"):
            state = self._state(_CONFUSE_SELFHIT_LINES, player=player)
            v2_2 = self._encode(state, V2_2_REPLAY_OBSERVATION_SPEC)
            v3 = self._encode(state, V3_REPLAY_OBSERVATION_SPEC)
            width = V2_2_REPLAY_OBSERVATION_SPEC.numeric_feature_count
            flagged = [i for i, row in enumerate(v3.numeric_features) if row[NUMERIC_TT_CONFUSION_SELFHIT]]
            self.assertEqual(len(flagged), 1, f"seat {player}: expected exactly one flagged row")
            flagged_row = flagged[0]
            prefix_diffs = {
                (ri, ci)
                for ri, (a, b) in enumerate(zip(v2_2.numeric_features, v3.numeric_features))
                for ci in range(width)
                if a[ci] != b[ci]
            }
            # The sole divergence on the shared surface is the intended damage correction.
            self.assertEqual(prefix_diffs, {(flagged_row, NUMERIC_TT_DAMAGE_FRACTION)})
            self.assertAlmostEqual(v2_2.numeric_features[flagged_row][NUMERIC_TT_DAMAGE_FRACTION], 0.27)
            self.assertAlmostEqual(v3.numeric_features[flagged_row][NUMERIC_TT_DAMAGE_FRACTION], 0.17)
            # Categoricals, masks, and token types are wholly untouched by change 10.
            self.assertEqual(
                [tuple(row) for row in v2_2.categorical_ids],
                [tuple(row) for row in v3.categorical_ids],
            )
            self.assertEqual(v2_2.attention_mask, v3.attention_mask)
            self.assertEqual(v2_2.token_type_ids, v3.token_type_ids)


if __name__ == "__main__":
    unittest.main()
