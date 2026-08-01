"""Observation spec v4 tests — the k0 feature pack (docs/observation_v4_spec.md).

Covers the schema-table wiring (fifth checkpoint-driven entry; v2.2 keeps the fresh default),
the grouped v4 column layout as an extension of v3's, the five Part-A pack signals
(A1 forced recharge, A2 last executed move, A3 Truant loaf phase, A4 the current TRACED ability,
A5 last-round damage dealt/taken), the three Part-B credit families (hazard credit, expected
remaining hazard value, items-removed credit), and the invariant that matters most for a new
contract: **v3 stays byte-frozen**. Part B item 3 (opponent switch propensity) is deliberately
absent — it is already in the observation and the plan's contribution was noting it needs no work.

``test_observation_spec_v3.py`` owns the v3 surface; this file only asserts v3 is unperturbed.
"""

import os
import unittest
from dataclasses import replace
from pathlib import Path

from pokezero.observation import (
    FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS,
    GROUPED_LAYOUT_OBSERVATION_SCHEMA_VERSIONS,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION_V2_2,
    OBSERVATION_SCHEMA_VERSION_V3,
    OBSERVATION_SCHEMA_VERSION_V4,
    SUPPORTED_OBSERVATION_SCHEMA_VERSIONS,
    TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS,
)
from pokezero.showdown import (
    CATEGORY_LAST_USED_MOVE,
    CATEGORY_TRACED_ABILITY,
    FIELD_TOKEN_OFFSET,
    LAST_USED_MOVE_SWITCH_SENTINEL,
    NUMERIC_ACTIVE,
    NUMERIC_LAST_DAMAGE_DEALT,
    NUMERIC_LAST_DAMAGE_TAKEN,
    NUMERIC_MUST_RECHARGE,
    NUMERIC_OPP_HAZARD_CREDIT,
    NUMERIC_OPP_HAZARD_EXPECTED,
    NUMERIC_OPP_HAZARDS,
    NUMERIC_OPP_ITEMS_REMOVED_CREDIT,
    NUMERIC_SELF_HAZARD_CREDIT,
    NUMERIC_SELF_HAZARD_EXPECTED,
    NUMERIC_SELF_ITEMS_REMOVED_CREDIT,
    NUMERIC_STALL_COUNTER,
    NUMERIC_TRUANT_LOAF,
    OPPONENT_POKEMON_TOKEN_OFFSET,
    REPLAY_OBSERVATION_SPECS_BY_SCHEMA,
    SELF_POKEMON_TOKEN_OFFSET,
    V3_NUMERIC_LAYOUT_GROUPS,
    V3_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT,
    V3_REPLAY_OBSERVATION_SPEC,
    V4_DROPPED_LEGACY_NUMERIC_INDICES,
    V4_NUMERIC_BASE,
    V4_NUMERIC_INDEX_BY_LEGACY_INDEX,
    V4_NUMERIC_LAYOUT_GROUPS,
    V4_NUMERIC_LEGACY_INDEX_BY_NEW_INDEX,
    V4_ONLY_NUMERIC_INDICES,
    V4_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT,
    V4_REPLAY_OBSERVATION_SPEC,
    _ReplayParser,
    normalize_for_player,
    numeric_index_for_schema,
    numeric_index_if_present_for_schema,
    observation_from_player_state,
    observation_schema_version_from_choice,
    observation_spec_for_schema,
    parse_showdown_replay,
    v3_numeric_index,
    v4_numeric_index,
)

SHOWDOWN_ROOT = Path(
    os.environ.get("POKEZERO_SHOWDOWN_ROOT", "/Users/scott/workspace/pokerena/vendor/pokemon-showdown")
)

_LEADS = [
    "|player|p1|Alice|",
    "|player|p2|Bob|",
    "|switch|p1a: Slaking|Slaking, L74, M|100/100",
    "|switch|p2a: Blissey|Blissey, L78, F|100/100",
    "|turn|1",
]


def _p1_request(active_ident: str, active_condition: str, *, bench: tuple[tuple[str, str, str], ...] = ()) -> str:
    """A minimal p1 ``|request|`` so p1's own mons land on SELF-side tokens.

    ``normalize_for_player`` builds the self team from the request, not from the public reveal
    stream, so any assertion about a self-side column needs one. ``bench`` lets a test declare
    additional party members — required by the Part-B expected-value column, whose whole input is
    how many BENCHED mons are still there to walk into the layers.
    """

    members = [
        f'{{"ident":"p1a: {active_ident}","details":"{_P1_DETAILS[active_ident]}",'
        f'"condition":"{active_condition}","active":true}}'
    ]
    members += [
        f'{{"ident":"p1: {ident}","details":"{details}","condition":"{condition}","active":false}}'
        for ident, details, condition in bench
    ]
    return (
        '|request|{"active":[{"moves":[{"move":"Body Slam","id":"bodyslam"}]}],'
        '"side":{"id":"p1","name":"Alice","pokemon":[' + ",".join(members) + "]}}"
    )


_P1_DETAILS = {
    "Slaking": "Slaking, L74, M",
    "Snorlax": "Snorlax, L78, M",
    "Gardevoir": "Gardevoir, L80, F",
    "Skarmory": "Skarmory, L76, M",
}

# A1 — Hyper Beam lands on turn 1, so the ``-mustrecharge`` lock is live across the turn-2
# decision and consumed by turn 2's ``cant``. Turn 3 is clean again.
_RECHARGE_LINES = _LEADS + [
    "|move|p1a: Slaking|Hyper Beam|p2a: Blissey",
    "|-damage|p2a: Blissey|60/100",
    "|-mustrecharge|p1a: Slaking",
    "|move|p2a: Blissey|Seismic Toss|p1a: Slaking",
    "|-damage|p1a: Slaking|80/100",
    "|upkeep",
    "|turn|2",
    "|cant|p1a: Slaking|recharge",
    "|move|p2a: Blissey|Seismic Toss|p1a: Slaking",
    "|-damage|p1a: Slaking|60/100",
    "|upkeep",
    "|turn|3",
]

# A MISSED Hyper Beam emits no ``-mustrecharge`` at all, so gen3's "a miss does not recharge"
# rule needs no special case in the tracker — the line simply never arrives.
_RECHARGE_MISS_LINES = _LEADS + [
    "|move|p1a: Slaking|Hyper Beam|p2a: Blissey",
    "|-miss|p1a: Slaking|p2a: Blissey",
    "|upkeep",
    "|turn|2",
]


@unittest.skipUnless(
    (SHOWDOWN_ROOT / "data" / "random-battles" / "gen3" / "sets.json").exists(),
    "requires a local Gen 3 Pokemon Showdown checkout",
)
class V4EncodeTestBase(unittest.TestCase):
    """Shared encode helpers: v4 rows are read through the schema-aware index map."""

    @staticmethod
    def _vocab(*, feature_pack: bool = True):
        from pokezero.randbat_vocab import gen3_category_vocabulary

        return gen3_category_vocabulary(
            SHOWDOWN_ROOT,
            include_turn_merged=True,
            include_feature_pack_v4=feature_pack,
        )

    @staticmethod
    def _dex():
        from pokezero.dex import load_showdown_dex_cached

        return load_showdown_dex_cached(SHOWDOWN_ROOT)

    def _state(self, lines, *, player="p1"):
        replay = parse_showdown_replay(lines, battle_id="v4-encode", complete_prefix=True)
        return normalize_for_player(
            replay,
            player_id=player,
            configured_showdown_slot=player,
            format_id="gen3randombattle",
            include_turn_merged=True,
        )

    def _encode(self, state, spec=None, *, dex=True):
        spec = spec if spec is not None else V4_REPLAY_OBSERVATION_SPEC
        feature_pack = spec.schema_version in FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS
        observation = observation_from_player_state(
            state,
            category_vocab=self._vocab(feature_pack=feature_pack),
            spec=spec,
            dex=self._dex() if dex else None,
        )
        observation.validate(spec)
        return observation

    @staticmethod
    def _active_token(observation, offset):
        """Index of the ACTIVE mon's token within a six-token team block."""
        active_column = v4_numeric_index(NUMERIC_ACTIVE)
        for index in range(offset, offset + 6):
            if observation.numeric_features[index][active_column] == 1.0:
                return index
        raise AssertionError("no active mon token in the block")

    def _pack(self, observation, offset, column):
        return observation.numeric_features[self._active_token(observation, offset)][
            v4_numeric_index(column)
        ]

    def _field(self, observation, column):
        return observation.numeric_features[FIELD_TOKEN_OFFSET][v4_numeric_index(column)]


class V4SchemaTableTest(unittest.TestCase):
    """The schema table, the CLI choice, and the specs the two resolve to."""

    def test_v4_is_supported_turn_merged_grouped_and_feature_packed_but_not_the_default(self) -> None:
        # Adding a schema must never move the fresh default: every running arm keeps collecting
        # under the schema its checkpoints were trained on.
        self.assertEqual(OBSERVATION_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION_V2_2)
        self.assertIn(OBSERVATION_SCHEMA_VERSION_V4, SUPPORTED_OBSERVATION_SCHEMA_VERSIONS)
        self.assertEqual(SUPPORTED_OBSERVATION_SCHEMA_VERSIONS[-1], OBSERVATION_SCHEMA_VERSION_V4)
        # V4 keeps v2.2's turn-merged transition surface and v3's grouped projection, and is the
        # only member of the feature-pack vocabulary family.
        self.assertIn(OBSERVATION_SCHEMA_VERSION_V4, TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS)
        self.assertIn(OBSERVATION_SCHEMA_VERSION_V4, GROUPED_LAYOUT_OBSERVATION_SCHEMA_VERSIONS)
        self.assertEqual(
            FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS, (OBSERVATION_SCHEMA_VERSION_V4,)
        )

    def test_cli_choice_and_spec_resolution(self) -> None:
        self.assertEqual(
            observation_schema_version_from_choice("v4"), OBSERVATION_SCHEMA_VERSION_V4
        )
        spec = observation_spec_for_schema(OBSERVATION_SCHEMA_VERSION_V4)
        self.assertIs(spec, V4_REPLAY_OBSERVATION_SPEC)
        self.assertIs(REPLAY_OBSERVATION_SPECS_BY_SCHEMA[OBSERVATION_SCHEMA_VERSION_V4], spec)
        # Ten numeric + two categorical columns on top of v3, and the SAME 64-row history tail:
        # the pack arm runs at budget 0, so resizing the tail would confound the comparison.
        self.assertEqual(
            spec.numeric_feature_count, V3_REPLAY_OBSERVATION_SPEC.numeric_feature_count + 10
        )
        self.assertEqual(
            spec.categorical_feature_count,
            V3_REPLAY_OBSERVATION_SPEC.categorical_feature_count + 2,
        )
        self.assertEqual(
            spec.transition_token_count, V3_REPLAY_OBSERVATION_SPEC.transition_token_count
        )


class V4LayoutTest(unittest.TestCase):
    """The grouped v4 layout: v3's table with the pack appended inside its semantic groups."""

    def test_layout_is_v3_plus_the_pack_in_semantic_groups(self) -> None:
        v3_groups = dict(V3_NUMERIC_LAYOUT_GROUPS)
        v4_groups = dict(V4_NUMERIC_LAYOUT_GROUPS)
        self.assertEqual(list(v4_groups), list(v3_groups))
        expected_additions = {
            "pokemon_state": (
                NUMERIC_MUST_RECHARGE,
                NUMERIC_TRUANT_LOAF,
                NUMERIC_LAST_DAMAGE_DEALT,
                NUMERIC_LAST_DAMAGE_TAKEN,
            ),
            "field": (
                NUMERIC_SELF_HAZARD_CREDIT,
                NUMERIC_OPP_HAZARD_CREDIT,
                NUMERIC_SELF_HAZARD_EXPECTED,
                NUMERIC_OPP_HAZARD_EXPECTED,
                NUMERIC_SELF_ITEMS_REMOVED_CREDIT,
                NUMERIC_OPP_ITEMS_REMOVED_CREDIT,
            ),
        }
        for name, v3_indices in v3_groups.items():
            self.assertEqual(
                v4_groups[name], v3_indices + expected_additions.get(name, ()), name
            )

    def test_every_writer_column_is_carried_or_explicitly_dropped(self) -> None:
        carried = set(V4_NUMERIC_LEGACY_INDEX_BY_NEW_INDEX)
        self.assertEqual(len(carried), len(V4_NUMERIC_LEGACY_INDEX_BY_NEW_INDEX))
        self.assertEqual(
            carried | V4_DROPPED_LEGACY_NUMERIC_INDICES,
            set(range(V4_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT)),
        )
        # V4 drops exactly what v3 dropped and nothing more: the pack adds signals, it does not
        # revisit the dead-field audit.
        self.assertEqual(
            V4_DROPPED_LEGACY_NUMERIC_INDICES, {24, 25, 35, 36, 48, 49, 50, 51, 52, 53, 54, 55, 103, 104}
        )
        self.assertEqual(
            V4_ONLY_NUMERIC_INDICES,
            frozenset(range(V4_NUMERIC_BASE, V4_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT)),
        )
        self.assertEqual(V4_NUMERIC_BASE, V3_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT)

    def test_v3_and_v4_indices_diverge_from_the_first_extended_group(self) -> None:
        # The pack is appended INSIDE pokemon_state, so every column laid out after that group
        # shifts. This is the concrete reason the two contracts can never share an artifact —
        # asserted rather than left implicit.
        self.assertEqual(v3_numeric_index(NUMERIC_STALL_COUNTER), v4_numeric_index(NUMERIC_STALL_COUNTER))
        self.assertNotEqual(v3_numeric_index(NUMERIC_OPP_HAZARDS), v4_numeric_index(NUMERIC_OPP_HAZARDS))
        self.assertEqual(
            v4_numeric_index(NUMERIC_OPP_HAZARDS), v3_numeric_index(NUMERIC_OPP_HAZARDS) + 4
        )

    def test_schema_aware_lookup_reports_pack_columns_as_absent_under_v3(self) -> None:
        for column in (NUMERIC_MUST_RECHARGE, NUMERIC_OPP_HAZARD_CREDIT):
            self.assertIsNone(
                numeric_index_if_present_for_schema(OBSERVATION_SCHEMA_VERSION_V3, column)
            )
            with self.assertRaisesRegex(ValueError, "not part of v3"):
                numeric_index_for_schema(OBSERVATION_SCHEMA_VERSION_V3, column)
            self.assertEqual(
                numeric_index_if_present_for_schema(OBSERVATION_SCHEMA_VERSION_V4, column),
                V4_NUMERIC_INDEX_BY_LEGACY_INDEX[column],
            )


class V4ParserTrackerTest(unittest.TestCase):
    """The pack's parser trackers, before any encode."""

    def test_must_recharge_is_set_by_the_line_and_consumed_by_the_cant(self) -> None:
        parser = _ReplayParser("recharge", complete_prefix=True)
        parser.feed(_LEADS)
        self.assertFalse(parser.snapshot().must_recharge["p1"])
        parser.feed(_RECHARGE_LINES[len(_LEADS) : _RECHARGE_LINES.index("|turn|2") + 1])
        # Live across the turn-2 decision: this is the boundary a k0 policy was blind at.
        self.assertTrue(parser.snapshot().must_recharge["p1"])
        parser.feed(["|cant|p1a: Slaking|recharge"])
        self.assertFalse(parser.snapshot().must_recharge["p1"])

    def test_missed_recharge_move_never_locks(self) -> None:
        replay = parse_showdown_replay(_RECHARGE_MISS_LINES, complete_prefix=True)
        self.assertFalse(replay.must_recharge["p1"])

    def test_recharge_lock_clears_when_the_mon_leaves(self) -> None:
        dragged = _RECHARGE_LINES[: _RECHARGE_LINES.index("|turn|2") + 1] + [
            "|drag|p1a: Snorlax|Snorlax, L78, M|100/100",
        ]
        replay = parse_showdown_replay(dragged, complete_prefix=True)
        self.assertFalse(replay.must_recharge["p1"])

    def test_must_recharge_round_trips_through_a_snapshot(self) -> None:
        prefix = _RECHARGE_LINES[: _RECHARGE_LINES.index("|turn|2") + 1]
        snapshot = parse_showdown_replay(prefix, complete_prefix=True)
        resumed = _ReplayParser.from_snapshot(snapshot)
        self.assertTrue(resumed.snapshot().must_recharge["p1"])
        resumed.feed(["|cant|p1a: Slaking|recharge"])
        self.assertFalse(resumed.snapshot().must_recharge["p1"])

    def test_damage_ledger_settles_at_the_turn_boundary_and_splits_dealt_from_taken(self) -> None:
        replay = parse_showdown_replay(
            _RECHARGE_LINES[: _RECHARGE_LINES.index("|turn|2") + 1], complete_prefix=True
        )
        # Turn 1: Slaking's Hyper Beam took Blissey to 60/100; Blissey's Seismic Toss took
        # Slaking to 80/100. DEALT is move-attributed, TAKEN is total.
        self.assertAlmostEqual(replay.last_damage_dealt["p1"], 0.40, places=6)
        self.assertAlmostEqual(replay.last_damage_taken["p1"], 0.20, places=6)
        self.assertAlmostEqual(replay.last_damage_dealt["p2"], 0.20, places=6)
        self.assertAlmostEqual(replay.last_damage_taken["p2"], 0.40, places=6)

    def test_damage_ledger_is_per_mon_and_resets_on_switch(self) -> None:
        lines = _RECHARGE_LINES[: _RECHARGE_LINES.index("|upkeep")] + [
            "|switch|p1a: Snorlax|Snorlax, L78, M|100/100",
            "|upkeep",
            "|turn|2",
        ]
        replay = parse_showdown_replay(lines, complete_prefix=True)
        # The mon that dealt the damage left; its replacement inherits no record.
        self.assertEqual(replay.last_damage_dealt["p1"], 0.0)
        self.assertEqual(replay.last_damage_taken["p1"], 0.0)
        # The other seat is untouched by our switch.
        self.assertAlmostEqual(replay.last_damage_taken["p2"], 0.40, places=6)

    def test_taken_counts_tagged_chip_that_dealt_never_does(self) -> None:
        lines = _LEADS + [
            "|move|p2a: Blissey|Toxic|p1a: Slaking",
            "|-status|p1a: Slaking|tox",
            "|-damage|p1a: Slaking|94/100|[from] psn",
            "|upkeep",
            "|turn|2",
        ]
        replay = parse_showdown_replay(lines, complete_prefix=True)
        self.assertAlmostEqual(replay.last_damage_taken["p1"], 0.06, places=6)
        # Chip is nobody's move damage: Blissey dealt none even though it applied the status.
        self.assertEqual(replay.last_damage_dealt["p2"], 0.0)

    def test_confusion_self_hit_is_not_credited_to_the_previous_mover(self) -> None:
        # The confused mon must be SLOWER, so the opponent's move window is still the most
        # recent one when the untagged self-damage lands. Without the ``-activate|confusion``
        # latch closing that window, this damage would be credited to p2's Seismic Toss.
        lines = _LEADS + [
            "|move|p2a: Blissey|Seismic Toss|p1a: Slaking",
            "|-damage|p1a: Slaking|80/100",
            "|-activate|p1a: Slaking|confusion",
            "|-damage|p1a: Slaking|65/100",
            "|upkeep",
            "|turn|2",
        ]
        replay = parse_showdown_replay(lines, complete_prefix=True)
        self.assertAlmostEqual(replay.last_damage_dealt["p2"], 0.20, places=6)
        # The self-hit still counts as HP this mon lost.
        self.assertAlmostEqual(replay.last_damage_taken["p1"], 0.35, places=6)

    def test_hazard_credit_accumulates_on_the_suffering_side_and_never_resets(self) -> None:
        lines = _LEADS + [
            "|move|p2a: Blissey|Spikes|p1a: Slaking",
            "|-sidestart|p1: Alice|Spikes",
            "|upkeep",
            "|turn|2",
            "|switch|p1a: Snorlax|Snorlax, L78, M|100/100",
            "|-damage|p1a: Snorlax|88/100|[from] Spikes",
            "|upkeep",
            "|turn|3",
            "|switch|p1a: Slaking|Slaking, L74, M|88/100",
            "|-damage|p1a: Slaking|76/100|[from] Spikes",
            "|upkeep",
            "|turn|4",
        ]
        replay = parse_showdown_replay(lines, complete_prefix=True)
        self.assertAlmostEqual(replay.hazard_damage_suffered["p1"], 0.24, places=6)
        self.assertEqual(replay.hazard_damage_suffered["p2"], 0.0)

    def test_knock_off_counts_but_an_eaten_berry_does_not(self) -> None:
        lines = _LEADS + [
            "|move|p2a: Blissey|Knock Off|p1a: Slaking",
            "|-damage|p1a: Slaking|96/100",
            "|-enditem|p1a: Slaking|Leftovers|[from] move: Knock Off|[of] p2a: Blissey",
            "|-enditem|p2a: Blissey|Salac Berry|[eat]",
            "|upkeep",
            "|turn|2",
        ]
        replay = parse_showdown_replay(lines, complete_prefix=True)
        self.assertEqual(replay.items_removed["p1"], 1)
        self.assertEqual(replay.items_removed["p2"], 0)


class V4PackEncodeTest(V4EncodeTestBase):
    """Part A: the per-mon pack columns on each side's ACTIVE token."""

    def test_recharge_bit_is_live_at_the_blind_decision_and_gone_after(self) -> None:
        # From the OPPONENT's seat (p2), Slaking's lock is the fact a k0 policy could not see.
        prefix = _RECHARGE_LINES[: _RECHARGE_LINES.index("|turn|2") + 1]
        locked = self._encode(self._state(prefix, player="p2"))
        self.assertEqual(
            self._pack(locked, OPPONENT_POKEMON_TOKEN_OFFSET, NUMERIC_MUST_RECHARGE), 1.0
        )
        # The SELF write path: the same lock on p1's own token (redundant with the request's
        # lone-legal-action collapse, and encoded anyway so the column is side-symmetric).
        own = self._encode(
            self._state(prefix + [_p1_request("Slaking", "80/100")], player="p1")
        )
        self.assertEqual(
            self._pack(own, SELF_POKEMON_TOKEN_OFFSET, NUMERIC_MUST_RECHARGE), 1.0
        )
        # Blissey, which used no recharge move, carries no bit on either side's write path.
        self.assertEqual(
            self._pack(own, OPPONENT_POKEMON_TOKEN_OFFSET, NUMERIC_MUST_RECHARGE), 0.0
        )
        cleared = self._encode(self._state(_RECHARGE_LINES, player="p2"))
        self.assertEqual(
            self._pack(cleared, OPPONENT_POKEMON_TOKEN_OFFSET, NUMERIC_MUST_RECHARGE), 0.0
        )

    def test_last_used_move_has_three_distinct_states(self) -> None:
        vocab = self._vocab()
        # Lead turn: both mons just came in -> the switch SENTINEL, not padding and not a move.
        lead = self._encode(self._state(_LEADS, player="p2"))
        token = self._active_token(lead, OPPONENT_POKEMON_TOKEN_OFFSET)
        self.assertEqual(
            lead.categorical_ids[token][CATEGORY_LAST_USED_MOVE],
            vocab.encode(LAST_USED_MOVE_SWITCH_SENTINEL),
        )
        # After executing a move: the move identity, sharing the action token's ``move:`` row.
        after = self._encode(
            self._state(_RECHARGE_LINES[: _RECHARGE_LINES.index("|turn|2") + 1], player="p2")
        )
        token = self._active_token(after, OPPONENT_POKEMON_TOKEN_OFFSET)
        self.assertEqual(
            after.categorical_ids[token][CATEGORY_LAST_USED_MOVE], vocab.encode("move:hyperbeam")
        )
        # The three states are genuinely three: sentinel != move identity != padding, where
        # padding means "this mon has never executed a move".
        self.assertNotEqual(
            vocab.encode("move:hyperbeam"), vocab.encode(LAST_USED_MOVE_SWITCH_SENTINEL)
        )
        self.assertNotEqual(vocab.encode(LAST_USED_MOVE_SWITCH_SENTINEL), 0)
        self.assertNotEqual(lead.categorical_ids[token][CATEGORY_LAST_USED_MOVE], 0)

    def test_last_damage_columns_are_mon_relative(self) -> None:
        # Encoded from p2's seat: the OPPONENT block is p1's Slaking (dealt 0.40, took 0.20),
        # the SELF block is p2's Blissey (dealt 0.20, took 0.40).
        observation = self._encode(
            self._state(
                _RECHARGE_LINES[: _RECHARGE_LINES.index("|turn|2") + 1], player="p2"
            )
        )
        self.assertAlmostEqual(
            self._pack(observation, OPPONENT_POKEMON_TOKEN_OFFSET, NUMERIC_LAST_DAMAGE_DEALT),
            0.40,
            places=5,
        )
        self.assertAlmostEqual(
            self._pack(observation, OPPONENT_POKEMON_TOKEN_OFFSET, NUMERIC_LAST_DAMAGE_TAKEN),
            0.20,
            places=5,
        )

    def test_truant_loaf_bit_tracks_the_free_running_toggle(self) -> None:
        # Slaking leads (acts on turn 1), so it LOAFS on turn 2's move attempt.
        lines = _LEADS + [
            "|move|p1a: Slaking|Body Slam|p2a: Blissey",
            "|-damage|p2a: Blissey|70/100",
            "|move|p2a: Blissey|Seismic Toss|p1a: Slaking",
            "|-damage|p1a: Slaking|90/100",
            "|upkeep",
            "|turn|2",
        ]
        observation = self._encode(
            self._state(lines + [_p1_request("Slaking", "90/100")], player="p1")
        )
        self.assertEqual(
            self._pack(observation, SELF_POKEMON_TOKEN_OFFSET, NUMERIC_TRUANT_LOAF), 1.0
        )
        # A non-holder never sets the bit — 0 covers "no holder" and "phase unknown" alike.
        self.assertEqual(
            self._pack(observation, OPPONENT_POKEMON_TOKEN_OFFSET, NUMERIC_TRUANT_LOAF), 0.0
        )

    def test_traced_ability_is_the_current_copy_and_clears_on_switch(self) -> None:
        vocab = self._vocab()
        traced = [
            "|player|p1|Alice|",
            "|player|p2|Bob|",
            "|switch|p1a: Gardevoir|Gardevoir, L80, F|100/100",
            "|switch|p2a: Claydol|Claydol, L80|100/100",
            "|-ability|p1a: Gardevoir|Levitate|Trace|[from] ability: Trace|[of] p2a: Claydol",
            "|turn|1",
        ]
        observation = self._encode(self._state(traced, player="p2"))
        token = self._active_token(observation, OPPONENT_POKEMON_TOKEN_OFFSET)
        self.assertEqual(
            observation.categorical_ids[token][CATEGORY_TRACED_ABILITY],
            vocab.encode("ability:levitate"),
        )
        # Trace drops its copy on switch-out — the stale-Levitate/Spikes-immunity bug this
        # column exists to avoid reproducing on the observation side.
        left = self._encode(
            self._state(
                traced
                + [
                    "|switch|p1a: Snorlax|Snorlax, L78, M|100/100",
                    "|upkeep",
                    "|turn|2",
                ],
                player="p2",
            )
        )
        token = self._active_token(left, OPPONENT_POKEMON_TOKEN_OFFSET)
        self.assertEqual(left.categorical_ids[token][CATEGORY_TRACED_ABILITY], 0)


class V4CreditEncodeTest(V4EncodeTestBase):
    """Part B: the field-token credit + expected-value columns."""

    # One layer of Spikes on OUR side; Snorlax walks into it, then Snorlax is active and
    # Slaking sits on the bench. The request declares the party the expected-value column
    # counts over: one grounded bench mon (Slaking) plus one Flying bench mon (Skarmory,
    # exempt) — the discrimination the grounding rule exists to make.
    _SPIKES_LINES = _LEADS + [
        "|move|p2a: Blissey|Spikes|p1a: Slaking",
        "|-sidestart|p1: Alice|Spikes",
        "|upkeep",
        "|turn|2",
        "|switch|p1a: Snorlax|Snorlax, L78, M|100/100",
        "|-damage|p1a: Snorlax|88/100|[from] Spikes",
        "|upkeep",
        "|turn|3",
        _p1_request(
            "Snorlax",
            "88/100",
            bench=(
                ("Slaking", "Slaking, L74, M", "100/100"),
                ("Skarmory", "Skarmory, L76, M", "100/100"),
            ),
        ),
    ]

    def test_hazard_credit_and_expected_value_read_as_a_spent_remaining_pair(self) -> None:
        observation = self._encode(self._state(self._SPIKES_LINES))
        # Realized: one 12% chip on our side, normalized by the six-mon team.
        self.assertAlmostEqual(
            self._field(observation, NUMERIC_SELF_HAZARD_CREDIT), 0.12 / 6.0, places=5
        )
        self.assertEqual(self._field(observation, NUMERIC_OPP_HAZARD_CREDIT), 0.0)
        # Remaining: layers are on OUR side, so the expectation is on the self column and the
        # opponent's is zero. One layer costs 1/8 of a max HP per grounded entry.
        self.assertGreater(self._field(observation, NUMERIC_SELF_HAZARD_EXPECTED), 0.0)
        self.assertEqual(self._field(observation, NUMERIC_OPP_HAZARD_EXPECTED), 0.0)

    def test_expected_value_counts_only_grounded_living_bench_mons(self) -> None:
        observation = self._encode(self._state(self._SPIKES_LINES))
        # Snorlax is ACTIVE (it already paid), Skarmory is Flying (exempt), so exactly one
        # grounded bench mon is still billable: 1 x 1/8 of a max HP, over the six-mon team.
        self.assertAlmostEqual(
            self._field(observation, NUMERIC_SELF_HAZARD_EXPECTED),
            (1 / 8.0) / 6.0,
            places=5,
        )

    def test_expected_value_is_zero_without_layers(self) -> None:
        observation = self._encode(
            self._state(
                _LEADS
                + [
                    _p1_request(
                        "Slaking",
                        "100/100",
                        bench=(("Snorlax", "Snorlax, L78, M", "100/100"),),
                    )
                ]
            )
        )
        for column in (NUMERIC_SELF_HAZARD_EXPECTED, NUMERIC_OPP_HAZARD_EXPECTED):
            self.assertEqual(self._field(observation, column), 0.0)

    def test_items_removed_credit_is_oriented_like_the_hazard_block(self) -> None:
        lines = _LEADS + [
            "|move|p2a: Blissey|Knock Off|p1a: Slaking",
            "|-damage|p1a: Slaking|96/100",
            "|-enditem|p1a: Slaking|Leftovers|[from] move: Knock Off|[of] p2a: Blissey",
            "|upkeep",
            "|turn|2",
        ]
        # From p1's seat the loss is ours: SELF_* is our own ground, matching SELF_HAZARDS.
        ours = self._encode(self._state(lines, player="p1"))
        self.assertAlmostEqual(
            self._field(ours, NUMERIC_SELF_ITEMS_REMOVED_CREDIT), 1 / 6.0, places=5
        )
        self.assertEqual(self._field(ours, NUMERIC_OPP_ITEMS_REMOVED_CREDIT), 0.0)
        # From p2's seat the same event is credit earned, on the OPP_* column.
        theirs = self._encode(self._state(lines, player="p2"))
        self.assertAlmostEqual(
            self._field(theirs, NUMERIC_OPP_ITEMS_REMOVED_CREDIT), 1 / 6.0, places=5
        )
        self.assertEqual(self._field(theirs, NUMERIC_SELF_ITEMS_REMOVED_CREDIT), 0.0)


class V4LastMoveAblationMaskTest(V4EncodeTestBase):
    """The A2 ablation switch: the plan's ``k0+pack`` vs ``k0+pack+lastmove`` arm pair."""

    def _encode_with_masks(self, lines, masks, *, player="p2"):
        return observation_from_player_state(
            self._state(lines, player=player),
            category_vocab=self._vocab(),
            spec=V4_REPLAY_OBSERVATION_SPEC,
            dex=self._dex(),
            feature_masks=masks,
        )

    def test_masking_a2_darkens_only_the_last_move_column(self) -> None:
        from pokezero.observation import ObservationFeatureMasks

        lines = _RECHARGE_LINES[: _RECHARGE_LINES.index("|turn|2") + 1]
        whole = self._encode_with_masks(lines, ObservationFeatureMasks())
        ablated = self._encode_with_masks(
            lines, ObservationFeatureMasks(feature_pack_last_move=False)
        )
        token = self._active_token(whole, OPPONENT_POKEMON_TOKEN_OFFSET)
        self.assertNotEqual(whole.categorical_ids[token][CATEGORY_LAST_USED_MOVE], 0)
        self.assertEqual(ablated.categorical_ids[token][CATEGORY_LAST_USED_MOVE], 0)
        # Two arms differing in EXACTLY this column: everything else, including the rest of
        # the pack, must be identical or the attribution read is confounded.
        self.assertEqual(whole.numeric_features, ablated.numeric_features)
        for row_index, (whole_row, ablated_row) in enumerate(
            zip(whole.categorical_ids, ablated.categorical_ids)
        ):
            for column, (a, b) in enumerate(zip(whole_row, ablated_row)):
                if column == CATEGORY_LAST_USED_MOVE:
                    continue
                self.assertEqual(a, b, f"row {row_index} categorical column {column}")

    def test_the_mask_is_inert_below_v4(self) -> None:
        from pokezero.observation import ObservationFeatureMasks

        # The column does not exist under v3, so toggling the switch cannot perturb a v3
        # encode — the property that makes it safe to add while v3 arms are running.
        state = self._state(_RECHARGE_LINES, player="p2")
        rows = [
            observation_from_player_state(
                state,
                category_vocab=self._vocab(feature_pack=False),
                spec=V3_REPLAY_OBSERVATION_SPEC,
                dex=self._dex(),
                feature_masks=ObservationFeatureMasks(feature_pack_last_move=flag),
            )
            for flag in (True, False)
        ]
        self.assertEqual(rows[0].categorical_ids, rows[1].categorical_ids)
        self.assertEqual(rows[0].numeric_features, rows[1].numeric_features)


class V4FreezesV3Test(V4EncodeTestBase):
    """The invariant a new contract lives or dies by: v3 output is untouched."""

    def _v3(self, lines, *, player="p1"):
        return self._encode(self._state(lines, player=player), V3_REPLAY_OBSERVATION_SPEC)

    def test_v3_encode_carries_no_pack_column_and_keeps_its_census(self) -> None:
        for lines in (_LEADS, _RECHARGE_LINES, V4CreditEncodeTest._SPIKES_LINES):
            observation = self._v3(lines)
            self.assertEqual(observation.schema_version, OBSERVATION_SCHEMA_VERSION_V3)
            self.assertEqual(
                len(observation.numeric_features[0]),
                V3_REPLAY_OBSERVATION_SPEC.numeric_feature_count,
            )
            self.assertEqual(
                len(observation.categorical_ids[0]),
                V3_REPLAY_OBSERVATION_SPEC.categorical_feature_count,
            )

    def test_v3_bytes_are_identical_with_and_without_pack_evidence_in_the_log(self) -> None:
        # The strongest available statement of the freeze: a replay FULL of pack signals
        # (a recharge lock, knocked-off item, hazard chip, damage on both sides) encodes at v3
        # to exactly the same rows as it did before the pack existed — because no v3 column
        # reads any of it. Compared against a v3 encode of the same state re-derived through the
        # v4 projection's shared writer surface.
        lines = _RECHARGE_LINES + [
            "|move|p2a: Blissey|Knock Off|p1a: Slaking",
            "|-enditem|p1a: Slaking|Leftovers|[from] move: Knock Off|[of] p2a: Blissey",
            "|upkeep",
            "|turn|4",
        ]
        state = self._state(lines)
        v3 = self._encode(state, V3_REPLAY_OBSERVATION_SPEC)
        v4 = self._encode(state, V4_REPLAY_OBSERVATION_SPEC)
        # Every column v3 carries appears in v4 with the same VALUE (at a different index).
        for row_index, (v3_row, v4_row) in enumerate(
            zip(v3.numeric_features, v4.numeric_features)
        ):
            for writer_index in range(V3_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT):
                if writer_index in V4_DROPPED_LEGACY_NUMERIC_INDICES:
                    continue
                self.assertEqual(
                    v3_row[v3_numeric_index(writer_index)],
                    v4_row[v4_numeric_index(writer_index)],
                    f"row {row_index} writer column {writer_index}",
                )

    def test_a_v4_spec_narrowed_to_the_v3_census_is_refused(self) -> None:
        narrowed = replace(
            V4_REPLAY_OBSERVATION_SPEC,
            numeric_feature_count=V3_REPLAY_OBSERVATION_SPEC.numeric_feature_count,
        )
        with self.assertRaisesRegex(ValueError, "grouped v4 layout requires exactly"):
            observation_from_player_state(
                self._state(_LEADS), category_vocab=self._vocab(), spec=narrowed
            )
        narrowed_categorical = replace(
            V4_REPLAY_OBSERVATION_SPEC,
            categorical_feature_count=V3_REPLAY_OBSERVATION_SPEC.categorical_feature_count,
        )
        with self.assertRaisesRegex(ValueError, "requires at least"):
            observation_from_player_state(
                self._state(_LEADS), category_vocab=self._vocab(), spec=narrowed_categorical
            )


@unittest.skipUnless(
    (SHOWDOWN_ROOT / "data" / "random-battles" / "gen3" / "sets.json").exists(),
    "requires a local Gen 3 Pokemon Showdown checkout",
)
class V4VocabularyTest(unittest.TestCase):
    """The feature-pack vocabulary latch: opt-in, and additive when on."""

    def test_pack_families_are_absent_unless_asked_for(self) -> None:
        from pokezero.randbat_vocab import gen3_category_vocabulary

        base = gen3_category_vocabulary(SHOWDOWN_ROOT, include_turn_merged=True)
        packed = gen3_category_vocabulary(
            SHOWDOWN_ROOT, include_turn_merged=True, include_feature_pack_v4=True
        )
        # Latch OFF: byte-for-byte the vocabulary every v2.2/v3 checkpoint was trained against.
        # This is what makes the pack safe to land while those arms are still running — their
        # enumeration, and therefore their embedding rows, cannot move.
        self.assertFalse(base.is_enumerated(LAST_USED_MOVE_SWITCH_SENTINEL))
        self.assertFalse(base.is_enumerated("ability:levitate"))
        # Latch ON: a strict superset by token SET. The enumeration is sorted, so the added rows
        # renumber the ones after them — harmless here and NOT an accident: a checkpoint stamps
        # its own ``category_vocab`` and resolves through that, never through a fresh build
        # (the row-drift bug ``category_vocab_from_model_config`` exists to prevent). A v4 arm
        # trains from game 0 against this enumeration.
        self.assertLess(set(base.tokens), set(packed.tokens))
        self.assertTrue(packed.is_enumerated(LAST_USED_MOVE_SWITCH_SENTINEL))
        self.assertTrue(packed.is_enumerated("ability:levitate"))

    def test_pack_reuses_the_move_family_rather_than_adding_one(self) -> None:
        from pokezero.randbat_vocab import gen3_category_vocabulary

        base = gen3_category_vocabulary(SHOWDOWN_ROOT, include_turn_merged=True)
        packed = gen3_category_vocabulary(
            SHOWDOWN_ROOT, include_turn_merged=True, include_feature_pack_v4=True
        )
        added = set(packed.tokens) - set(base.tokens)
        # A2's move identity shares the ACTION token's ``move:<id>`` row rather than getting a
        # private ``lastmove:<id>`` family — that sharing is why the pack costs one sentinel plus
        # the ability family and not one row per move.
        self.assertNotIn("lastmove:hyperbeam", added)
        self.assertFalse(packed.is_enumerated("lastmove:hyperbeam"))
        self.assertTrue(packed.is_enumerated("move:hyperbeam"))
        self.assertEqual(
            {token for token in added if not token.startswith("ability:")},
            {LAST_USED_MOVE_SWITCH_SENTINEL},
        )


if __name__ == "__main__":
    unittest.main()
