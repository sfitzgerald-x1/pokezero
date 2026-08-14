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

import unittest
from dataclasses import replace
from pathlib import Path

from pokezero.observation import (
    FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS,
    GROUPED_LAYOUT_OBSERVATION_SCHEMA_VERSIONS,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION_V2_1,
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
    CATEGORY_VOLATILE_OFFSET,
    MUST_RECHARGE_VOLATILE,
    NUMERIC_CHOICE_LOCKED,
    NUMERIC_ITEM_SWAPPED,
    LAST_USED_MOVE_BATON_PASS_SENTINEL,
    LAST_USED_MOVE_SWITCH_SENTINEL,
    VOLATILE_BUCKET_COUNT,
    NUMERIC_ACTIVE,
    NUMERIC_LAST_DAMAGE_DEALT,
    NUMERIC_LAST_DAMAGE_TAKEN,
    NUMERIC_MON_STAYED_VS_ACTIVE,
    NUMERIC_MON_SWITCHED_VS_ACTIVE,
    NUMERIC_OPP_HAZARD_CREDIT,
    NUMERIC_OPP_HAZARD_EXPECTED,
    NUMERIC_OPP_HAZARDS,
    NUMERIC_OPP_ITEMS_REMOVED_CREDIT,
    NUMERIC_SELF_HAZARD_CREDIT,
    NUMERIC_SELF_HAZARD_EXPECTED,
    NUMERIC_SELF_ITEMS_REMOVED_CREDIT,
    NUMERIC_STALL_COUNTER,
    NUMERIC_SUB_HP_FRACTION,
    NUMERIC_TIER2_CB_PINNED,
    NUMERIC_TIER2_INVESTMENT_PINNED,
    NUMERIC_TRUANT_LOAF,
    NUMERIC_TT_CB_BIT,
    NUMERIC_TT_INVESTMENT_BIT,
    OPPONENT_POKEMON_TOKEN_OFFSET,
    REPLAY_OBSERVATION_SPECS_BY_SCHEMA,
    SELF_POKEMON_TOKEN_OFFSET,
    V3_DROPPED_LEGACY_NUMERIC_INDICES,
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
from _showdown_root import showdown_root_str

SHOWDOWN_ROOT = Path(
    showdown_root_str()
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
        """v4's vocabulary: the feature-pack families, and NO turn-merged families.

        v4 carries no transition region, so the tt_phase/tt2_* rows would describe rows that do
        not exist. The two latches are independent for exactly this reason.
        """
        from pokezero.randbat_vocab import gen3_category_vocabulary

        return gen3_category_vocabulary(
            SHOWDOWN_ROOT,
            include_turn_merged=not feature_pack,
            include_feature_pack_v4=feature_pack,
        )

    @staticmethod
    def _dex():
        from pokezero.dex import load_showdown_dex_cached

        return load_showdown_dex_cached(SHOWDOWN_ROOT)

    def _state(self, lines, *, player="p1", turn_merged=False):
        """Normalize the way PRODUCTION v4 does: include_turn_merged=False.

        v4 is deliberately absent from TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS, so every
        harness builds v4 states without the merged stream. Hardcoding True here would have
        exercised a path v4 never takes; the v3-freeze test opts back in explicitly, because
        v3 does need it.
        """
        replay = parse_showdown_replay(lines, battle_id="v4-encode", complete_prefix=True)
        return normalize_for_player(
            replay,
            player_id=player,
            configured_showdown_slot=player,
            format_id="gen3randombattle",
            include_turn_merged=turn_merged,
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

    def test_v4_IS_the_default(self) -> None:
        """A CLEAN identity pin: one assertion, reading the process default and nothing else.

        Split out of test_v4_is_supported_turn_merged_grouped_and_feature_packed below, which
        asserted the default's identity AND four membership facts about tuples the rotation
        drill mutates (SUPPORTED, TURN_MERGED, GROUPED_LAYOUT, FEATURE_PACK). It therefore
        broke under a rotation whether or not its default assertion still bound, leaving the
        drill's EXPECTED-BUT-DID-NOT-BREAK detector blind to this pin going stale.

        Asserted POSITIVELY (default IS v4), and the reason is the same one that made the v2.2
        version of this pin positive rather than "v4 is not the default": a NEGATIVE form survives a
        rotation -- under a rotation to v5-drill, v3 is still not the default either -- so it would
        not break, and a non-breaking row in the drill's rubric is a false pin. The v2.2 pin's name
        carried a `_not_v4` clause because v4 was then the schema most likely to be mistaken for the
        default; now that it IS the default there is nothing left to negate, so the clause is gone.
        """
        self.assertEqual(OBSERVATION_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION_V4)

    def test_v4_is_supported_turn_merged_grouped_and_feature_packed(self) -> None:
        # This test is about SUPPORT and shape, not about which schema is default. Which one is
        # default is pinned by test_v4_IS_the_default above and is deliberately NOT re-asserted
        # here; see that test's docstring for why the two must stay separate.
        #
        # The comment this replaces read "Adding a schema must never move the fresh default" --
        # true of ADDING v4 (2026-07), and it stayed true right up until the deliberate rotation
        # that moved the default to v4 on 2026-08-13. Kept in the record because the invariant is
        # still the real one: no schema addition may move the default, only an explicit rotation.
        self.assertIn(OBSERVATION_SCHEMA_VERSION_V4, SUPPORTED_OBSERVATION_SCHEMA_VERSIONS)
        self.assertEqual(SUPPORTED_OBSERVATION_SCHEMA_VERSIONS[-1], OBSERVATION_SCHEMA_VERSION_V4)
        # V4 keeps v3's grouped projection but is NOT turn-merged — it has no transition
        # region for a turn-merged surface to live in.
        self.assertNotIn(OBSERVATION_SCHEMA_VERSION_V4, TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS)
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
        # The history region is GONE — not budgeted to zero, removed from the contract.
        self.assertEqual(spec.transition_token_count, 0)
        self.assertEqual(spec.token_count, 23)
        self.assertLess(spec.token_count, V3_REPLAY_OBSERVATION_SPEC.token_count)
        # v3's surface, minus the 34-column history group, minus the two retired pinned tier2
        # columns (NUMERIC_TIER2_CB_PINNED / NUMERIC_TIER2_INVESTMENT_PINNED), minus the 12
        # turn-merged categorical columns, plus the 13-column feature pack and its 2
        # categorical rows.
        self.assertEqual(spec.numeric_feature_count, 132)
        self.assertEqual(spec.categorical_feature_count, 41)
        # v4 is a grouped-layout schema but NOT a turn-merged one: the two axes are separate.
        self.assertNotIn(OBSERVATION_SCHEMA_VERSION_V4, TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS)


class V4LayoutTest(unittest.TestCase):
    """The grouped v4 layout: v3's table with the pack appended inside its semantic groups."""

    def test_layout_is_v3_plus_the_pack_in_semantic_groups(self) -> None:
        v3_groups = dict(V3_NUMERIC_LAYOUT_GROUPS)
        v4_groups = dict(V4_NUMERIC_LAYOUT_GROUPS)
        # Same groups as v3 except "history", which is gone with the region it described.
        self.assertEqual(list(v4_groups), [n for n in v3_groups if n != "history"])
        self.assertNotIn("history", v4_groups)
        expected_additions = {
            "pokemon_state": (
                NUMERIC_TRUANT_LOAF,
                NUMERIC_LAST_DAMAGE_DEALT,
                NUMERIC_LAST_DAMAGE_TAKEN,
                NUMERIC_CHOICE_LOCKED,
                NUMERIC_ITEM_SWAPPED,
            ),
            # The matchup-conditional pair joins the per-opponent-mon tendency triple, which
            # lives in "belief" — the two are read together, so they sit together.
            "belief": (
                NUMERIC_MON_SWITCHED_VS_ACTIVE,
                NUMERIC_MON_STAYED_VS_ACTIVE,
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
        # Two v3 current-state columns are RETIRED at v4 (not carried to a new index): the
        # pinned tier2 PAIR, whose evidence now narrows the belief candidate set instead. Both
        # sit in pokemon_state.
        retired = (NUMERIC_TIER2_CB_PINNED, NUMERIC_TIER2_INVESTMENT_PINNED)
        for column in retired:
            self.assertIn(column, dict(V3_NUMERIC_LAYOUT_GROUPS)["pokemon_state"])
            self.assertNotIn(column, v4_groups["pokemon_state"])
        for name, v3_indices in v3_groups.items():
            if name == "history":
                continue
            carried = tuple(i for i in v3_indices if i not in retired)
            self.assertEqual(
                v4_groups[name], carried + expected_additions.get(name, ()), name
            )

    def test_every_writer_column_is_carried_or_explicitly_dropped(self) -> None:
        carried = set(V4_NUMERIC_LEGACY_INDEX_BY_NEW_INDEX)
        self.assertEqual(len(carried), len(V4_NUMERIC_LEGACY_INDEX_BY_NEW_INDEX))
        self.assertEqual(
            carried | V4_DROPPED_LEGACY_NUMERIC_INDICES,
            set(range(V4_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT)),
        )
        # V4 drops everything v3 dropped, PLUS the whole history group, PLUS the two retired
        # current-state columns.
        self.assertLess(
            {24, 25, 35, 36, 48, 49, 50, 51, 52, 53, 54, 55, 103, 104},
            set(V4_DROPPED_LEGACY_NUMERIC_INDICES),
        )
        history = {i for n, idx in V3_NUMERIC_LAYOUT_GROUPS if n == "history" for i in idx}
        self.assertTrue(history <= set(V4_DROPPED_LEGACY_NUMERIC_INDICES))
        for column in (NUMERIC_TIER2_CB_PINNED, NUMERIC_TIER2_INVESTMENT_PINNED):
            self.assertIn(column, V4_DROPPED_LEGACY_NUMERIC_INDICES)
            self.assertNotIn(column, history)
        # ...and nothing else. The retirement is exactly the pinned tier2 pair, two columns
        # wide. Their as-of-strike history twins (119/120) come out with the region, not here.
        self.assertEqual(
            set(V4_DROPPED_LEGACY_NUMERIC_INDICES)
            - set(V3_DROPPED_LEGACY_NUMERIC_INDICES)
            - history,
            {NUMERIC_TIER2_CB_PINNED, NUMERIC_TIER2_INVESTMENT_PINNED},
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
        #
        # Columns BEFORE the retired pinned tier2 pair (138/139) still align — sub HP at 137 is
        # the last of them; everything after shifts down two, then up by the group insertions.
        self.assertEqual(
            v3_numeric_index(NUMERIC_SUB_HP_FRACTION), v4_numeric_index(NUMERIC_SUB_HP_FRACTION)
        )
        self.assertEqual(
            v4_numeric_index(NUMERIC_STALL_COUNTER),
            v3_numeric_index(NUMERIC_STALL_COUNTER) - 2,
        )
        self.assertNotEqual(v3_numeric_index(NUMERIC_OPP_HAZARDS), v4_numeric_index(NUMERIC_OPP_HAZARDS))
        # -2 for the retired pair, +5 from the pokemon_state insertions and +2 from the
        # belief insertions, all laid out before the field group.
        self.assertEqual(
            v4_numeric_index(NUMERIC_OPP_HAZARDS), v3_numeric_index(NUMERIC_OPP_HAZARDS) + 5
        )

    def test_schema_aware_lookup_reports_pack_columns_as_absent_under_v3(self) -> None:
        for column in (NUMERIC_TRUANT_LOAF, NUMERIC_OPP_HAZARD_CREDIT):
            self.assertIsNone(
                numeric_index_if_present_for_schema(OBSERVATION_SCHEMA_VERSION_V3, column)
            )
            with self.assertRaisesRegex(ValueError, "not part of v3"):
                numeric_index_for_schema(OBSERVATION_SCHEMA_VERSION_V3, column)
            self.assertEqual(
                numeric_index_if_present_for_schema(OBSERVATION_SCHEMA_VERSION_V4, column),
                V4_NUMERIC_INDEX_BY_LEGACY_INDEX[column],
            )

    def test_the_pinned_tier2_columns_are_retired_from_v4_only(self) -> None:
        """v2.1/v2.2/v3 checkpoints have columns 138/139 in their input layout; v4 never did."""

        for column in (NUMERIC_TIER2_CB_PINNED, NUMERIC_TIER2_INVESTMENT_PINNED):
            self.assertIsNone(
                numeric_index_if_present_for_schema(OBSERVATION_SCHEMA_VERSION_V4, column), column
            )
            with self.assertRaisesRegex(ValueError, "dropped from v4"):
                numeric_index_for_schema(OBSERVATION_SCHEMA_VERSION_V4, column)
            for schema in (
                OBSERVATION_SCHEMA_VERSION_V2_1,
                OBSERVATION_SCHEMA_VERSION_V2_2,
                OBSERVATION_SCHEMA_VERSION_V3,
            ):
                self.assertIsNotNone(
                    numeric_index_if_present_for_schema(schema, column), (schema, column)
                )

    def test_the_as_of_strike_tier2_twins_leave_with_the_history_region(self) -> None:
        """119/120 need no retirement clause — they are history columns.

        Worth pinning explicitly: it is the reason v4 carries NO encoded surface at all for
        either tier2 conclusion, so the belief narrowing is their only consumer there.
        """

        for column in (NUMERIC_TT_CB_BIT, NUMERIC_TT_INVESTMENT_BIT):
            history = {i for n, idx in V3_NUMERIC_LAYOUT_GROUPS if n == "history" for i in idx}
            self.assertIn(column, history)
            self.assertIsNone(
                numeric_index_if_present_for_schema(OBSERVATION_SCHEMA_VERSION_V4, column), column
            )
            self.assertIsNotNone(
                numeric_index_if_present_for_schema(OBSERVATION_SCHEMA_VERSION_V3, column), column
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

    def _volatiles(self, observation, offset):
        """The volatile bag on a side's ACTIVE mon token, as decoded vocabulary rows."""
        token = self._active_token(observation, offset)
        row = observation.categorical_ids[token]
        return {
            row[CATEGORY_VOLATILE_OFFSET + i]
            for i in range(VOLATILE_BUCKET_COUNT)
            if row[CATEGORY_VOLATILE_OFFSET + i]
        }

    def test_recharge_volatile_is_live_at_the_blind_decision_and_gone_after(self) -> None:
        # A1 is a VOLATILE at v4, not a numeric column: categorical columns are summed into the
        # token embedding, so a bag entry and a dedicated 0/1 column are the same function —
        # and the bag costs no column and matches ``volatile:solarbeam``, the charge half of
        # this very move family.
        vocab = self._vocab()
        want = vocab.encode(f"volatile:{MUST_RECHARGE_VOLATILE}")
        prefix = _RECHARGE_LINES[: _RECHARGE_LINES.index("|turn|2") + 1]

        # From the OPPONENT's seat (p2), Slaking's lock is the fact a k0 policy could not see.
        locked = self._encode(self._state(prefix, player="p2"))
        self.assertIn(want, self._volatiles(locked, OPPONENT_POKEMON_TOKEN_OFFSET))

        # The SELF write path, plus the negative: Blissey used no recharge move.
        own = self._encode(
            self._state(prefix + [_p1_request("Slaking", "80/100")], player="p1")
        )
        self.assertIn(want, self._volatiles(own, SELF_POKEMON_TOKEN_OFFSET))
        self.assertNotIn(want, self._volatiles(own, OPPONENT_POKEMON_TOKEN_OFFSET))

        # Consumed by the following turn's ``cant``.
        cleared = self._encode(self._state(_RECHARGE_LINES, player="p2"))
        self.assertNotIn(want, self._volatiles(cleared, OPPONENT_POKEMON_TOKEN_OFFSET))

    def test_the_recharge_volatile_row_exists_only_under_the_feature_pack_latch(self) -> None:
        # It is NOT in TRACKED_VOLATILES (the sim emits a bespoke ``|-mustrecharge|``, never a
        # ``|-start|``), so it does not fall out of GEN3_VOLATILES and must be enumerated by the
        # pack latch — which is also what keeps it out of every v2.2/v3 vocabulary.
        from pokezero.showdown import TRACKED_VOLATILES

        self.assertNotIn(MUST_RECHARGE_VOLATILE, TRACKED_VOLATILES)
        self.assertTrue(self._vocab().is_enumerated(f"volatile:{MUST_RECHARGE_VOLATILE}"))
        self.assertFalse(
            self._vocab(feature_pack=False).is_enumerated(
                f"volatile:{MUST_RECHARGE_VOLATILE}"
            )
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
        state = self._state(_RECHARGE_LINES, player="p2", turn_merged=True)
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


class V4MatchupSwitchTendencyTest(V4EncodeTestBase):
    """The matchup-conditional switch pair: the marginal triple's missing conditioning."""

    # Against Magneton: Skarmory bails BOTH times it is sent in and never attacks, while
    # Blissey bails once and then stands its ground once. Encoded from p1's seat, so
    # Magneton is "our current active".
    _LINES = [
        "|player|p1|Alice|",
        "|player|p2|Bob|",
        "|switch|p1a: Magneton|Magneton, L80|100/100",
        "|switch|p2a: Skarmory|Skarmory, L76, M|100/100",
        "|turn|1",
        "|switch|p2a: Blissey|Blissey, L78, F|100/100",           # Skarmory bails (1)
        "|move|p1a: Magneton|Thunderbolt|p2a: Blissey",
        "|-damage|p2a: Blissey|70/100",
        "|upkeep",
        "|turn|2",
        "|switch|p2a: Skarmory|Skarmory, L76, M|100/100",         # Blissey bails (1)
        "|move|p1a: Magneton|Thunderbolt|p2a: Skarmory",
        "|-damage|p2a: Skarmory|40/100",
        "|upkeep",
        "|turn|3",
        "|switch|p2a: Blissey|Blissey, L78, F|70/100",            # Skarmory bails (2)
        "|move|p1a: Magneton|Thunderbolt|p2a: Blissey",
        "|-damage|p2a: Blissey|40/100",
        "|upkeep",
        "|turn|4",
        "|move|p2a: Blissey|Seismic Toss|p1a: Magneton",          # Blissey STAYS (1)
        "|-damage|p1a: Magneton|70/100",
        "|move|p1a: Magneton|Thunderbolt|p2a: Blissey",
        "|-damage|p2a: Blissey|20/100",
        "|upkeep",
        "|turn|5",
        '|request|{"active":[{"moves":[{"move":"Thunderbolt","id":"thunderbolt"}]}],'
        '"side":{"id":"p1","name":"Alice","pokemon":['
        '{"ident":"p1a: Magneton","details":"Magneton, L80","condition":"70/100","active":true}]}}',
    ]

    def _by_species(self, observation, species):
        vocab = self._vocab()
        want = vocab.encode(f"species:{species}")
        for index in range(OPPONENT_POKEMON_TOKEN_OFFSET, OPPONENT_POKEMON_TOKEN_OFFSET + 6):
            if observation.categorical_ids[index][0] == want:
                row = observation.numeric_features[index]
                return (
                    row[v4_numeric_index(NUMERIC_MON_SWITCHED_VS_ACTIVE)],
                    row[v4_numeric_index(NUMERIC_MON_STAYED_VS_ACTIVE)],
                )
        raise AssertionError(f"no opponent token for {species}")

    def test_conditioning_separates_two_mons_the_marginal_count_would_not(self) -> None:
        state = self._state(self._LINES)
        # (bailed, stood its ground) against Magneton. Skarmory always runs; Blissey ran
        # once and then fought. A bare "bailed N times" cannot express that difference —
        # which is the whole point of conditioning on what they were facing.
        self.assertEqual(
            dict(state.opponent_matchup_switch_evidence),
            {"skarmory": (2, 0), "blissey": (1, 1)},
        )
        observation = self._encode(state)
        # /8 evidence mass, count and opportunity on the same scale so the model reads a rate.
        self.assertAlmostEqual(self._by_species(observation, "Skarmory")[0], 2 / 8, places=6)
        self.assertEqual(self._by_species(observation, "Skarmory")[1], 0.0)
        self.assertAlmostEqual(self._by_species(observation, "Blissey")[0], 1 / 8, places=6)
        self.assertAlmostEqual(self._by_species(observation, "Blissey")[1], 1 / 8, places=6)

    def test_an_unvisited_matchup_reads_zero_rather_than_borrowing_the_marginal(self) -> None:
        # Same game, but our active is Snorlax — a matchup neither of their mons has faced.
        # The pair must read (0, 0): no history HERE, not "no history at all".
        lines = self._LINES[:-1] + [
            "|switch|p1a: Snorlax|Snorlax, L78, M|100/100",
            "|turn|6",
            '|request|{"active":[{"moves":[{"move":"Body Slam","id":"bodyslam"}]}],'
            '"side":{"id":"p1","name":"Alice","pokemon":['
            '{"ident":"p1a: Snorlax","details":"Snorlax, L78, M","condition":"100/100",'
            '"active":true}]}}',
        ]
        state = self._state(lines)
        self.assertEqual(dict(state.opponent_matchup_switch_evidence), {})
        observation = self._encode(state)
        self.assertEqual(self._by_species(observation, "Skarmory"), (0.0, 0.0))
        # The MARGINAL triple on the same token is untouched and still carries the fallback.
        marginal = observation.numeric_features[
            self._active_token(observation, OPPONENT_POKEMON_TOKEN_OFFSET)
        ]
        self.assertGreaterEqual(len(marginal), V4_REPLAY_OBSERVATION_SPEC.numeric_feature_count)

    def test_the_pair_rides_the_tendency_mask(self) -> None:
        from pokezero.observation import ObservationFeatureMasks

        observation = observation_from_player_state(
            self._state(self._LINES),
            category_vocab=self._vocab(),
            spec=V4_REPLAY_OBSERVATION_SPEC,
            dex=self._dex(),
            feature_masks=ObservationFeatureMasks(opponent_tendency_stats_block=False),
        )
        # Same channel as the marginal triple, conditioned — so the same mask darkens it.
        self.assertEqual(self._by_species(observation, "Skarmory"), (0.0, 0.0))

    def test_v3_does_not_carry_the_pair(self) -> None:
        for column in (NUMERIC_MON_SWITCHED_VS_ACTIVE, NUMERIC_MON_STAYED_VS_ACTIVE):
            self.assertIsNone(
                numeric_index_if_present_for_schema(OBSERVATION_SCHEMA_VERSION_V3, column)
            )


class V4BatonPassSentinelTest(V4EncodeTestBase):
    """A2's fourth state: a Baton-Pass arrival is not an ordinary switch-in."""

    _BP = [
        "|player|p1|Alice|",
        "|player|p2|Bob|",
        "|switch|p1a: Slaking|Slaking, L74, M|100/100",
        "|switch|p2a: Ninjask|Ninjask, L78, M|100/100",
        "|turn|1",
        "|move|p2a: Ninjask|Swords Dance|p2a: Ninjask",
        "|-boost|p2a: Ninjask|atk|2",
        "|upkeep",
        "|turn|2",
        "|move|p2a: Ninjask|Baton Pass|p2a: Ninjask",
        "|switch|p2a: Blissey|Blissey, L78, F|100/100|[from] Baton Pass",
        "|upkeep",
        "|turn|3",
    ]
    _PLAIN = [
        "|player|p1|Alice|",
        "|player|p2|Bob|",
        "|switch|p1a: Slaking|Slaking, L74, M|100/100",
        "|switch|p2a: Ninjask|Ninjask, L78, M|100/100",
        "|turn|1",
        "|switch|p2a: Blissey|Blissey, L78, F|100/100",
        "|upkeep",
        "|turn|2",
    ]

    def _sentinel(self, lines):
        observation = self._encode(self._state(lines, player="p1"))
        token = self._active_token(observation, OPPONENT_POKEMON_TOKEN_OFFSET)
        return observation.categorical_ids[token][CATEGORY_LAST_USED_MOVE]

    def test_baton_pass_arrival_gets_its_own_sentinel(self) -> None:
        vocab = self._vocab()
        self.assertEqual(
            self._sentinel(self._BP), vocab.encode(LAST_USED_MOVE_BATON_PASS_SENTINEL)
        )
        self.assertEqual(
            self._sentinel(self._PLAIN), vocab.encode(LAST_USED_MOVE_SWITCH_SENTINEL)
        )
        # Three distinguishable arrival/never-moved states, none of them the padding row.
        self.assertNotEqual(self._sentinel(self._BP), self._sentinel(self._PLAIN))
        self.assertNotEqual(self._sentinel(self._BP), 0)

    def test_the_engine_fact_is_unchanged_only_the_observation_is_richer(self) -> None:
        # The parser still records the plain ``switch`` sentinel in last_used_move — that is
        # what the WORLD reads, and gen3 gives a Baton-Pass recipient a null lastMove exactly
        # like any other switch-in. Only the encoded label distinguishes them.
        state = self._state(self._BP, player="p1")
        self.assertEqual(state.opponent_last_used_move, "switch")
        self.assertTrue(state.opponent_arrived_by_baton_pass)
        self.assertFalse(self._state(self._PLAIN, player="p1").opponent_arrived_by_baton_pass)


class V4ChoiceLockTest(V4EncodeTestBase):
    """The silent choicelock, reconstructed — and its valence discriminator."""

    # Furret always holds a Choice Band in gen3 randbats (teams.ts:
    # ``if (moves.has('trick')) return 'Choice Band'``). It Tricks the band onto Blissey, who
    # then uses Calm Mind and is locked into it for the rest of its stay.
    _TRICK = [
        "|player|p1|Alice|",
        "|player|p2|Bob|",
        "|switch|p1a: Furret|Furret, L84, M|100/100",
        "|switch|p2a: Blissey|Blissey, L78, F|100/100",
        "|turn|1",
        "|move|p1a: Furret|Trick|p2a: Blissey",
        "|-activate|p1a: Furret|move: Trick|[of] p2a: Blissey",
        "|-item|p2a: Blissey|Choice Band|[from] move: Trick",
        "|-item|p1a: Furret|Leftovers|[from] move: Trick",
        "|upkeep",
        "|turn|2",
        "|move|p2a: Blissey|Calm Mind|p2a: Blissey",
        "|-boost|p2a: Blissey|spa|1",
        "|upkeep",
        "|turn|3",
    ]

    def _bits(self, lines, offset=OPPONENT_POKEMON_TOKEN_OFFSET, player="p1"):
        observation = self._encode(self._state(lines, player=player))
        return (
            self._pack(observation, offset, NUMERIC_CHOICE_LOCKED),
            self._pack(observation, offset, NUMERIC_ITEM_SWAPPED),
        )

    def test_the_lock_attaches_to_the_first_move_after_the_item_arrives(self) -> None:
        # Choice Band's onStart REMOVES any choicelock when the item lands, and onModifyMove
        # re-adds it on the next move used — so immediately after the Trick there is no lock.
        before = self._TRICK[: self._TRICK.index("|turn|2") + 1]
        self.assertEqual(self._bits(before), (0.0, 1.0))
        # …and after Calm Mind, the lock is on.
        self.assertEqual(self._bits(self._TRICK), (1.0, 1.0))

    def test_the_lock_names_its_move_through_A2(self) -> None:
        # Lock bit + last-used-move fully specify the lock, exactly as volatile:encore + A2
        # specify an Encore. This is the pair that was missing at v3.
        observation = self._encode(self._state(self._TRICK, player="p1"))
        token = self._active_token(observation, OPPONENT_POKEMON_TOKEN_OFFSET)
        self.assertEqual(
            observation.categorical_ids[token][CATEGORY_LAST_USED_MOVE],
            self._vocab().encode("move:calmmind"),
        )

    def test_in_this_pool_a_lock_is_always_a_tricked_lock(self) -> None:
        """The pair is collinear in gen3 randbats, and that is worth pinning explicitly.

        A NATIVE Choice Band is never announced — gen3 has no Frisk-style reveal and Choice
        Band is the only ``isChoice`` item — so ``choice_item_public`` can only be set by a
        Trick's ``|-item|`` line. An earlier version of this test manufactured the native case
        with ``[from] ability: Frisk``, which the gen 3 sim cannot emit; it asserted a state
        that does not exist. Pin the real invariant instead: locked implies swapped here.
        """
        locked, swapped = self._bits(self._TRICK)
        self.assertEqual((locked, swapped), (1.0, 1.0))
        # And the columns stay independent in the encoding, so the day a native reveal surface
        # appears the model does not have to unlearn a conflation.
        self.assertNotEqual(
            v4_numeric_index(NUMERIC_CHOICE_LOCKED), v4_numeric_index(NUMERIC_ITEM_SWAPPED)
        )

    def test_losing_the_item_clears_both_bits(self) -> None:
        knocked = self._TRICK + [
            "|move|p1a: Furret|Knock Off|p2a: Blissey",
            "|-enditem|p2a: Blissey|Choice Band|[from] move: Knock Off|[of] p1a: Furret",
            "|upkeep",
            "|turn|4",
        ]
        self.assertEqual(self._bits(knocked), (0.0, 0.0))


class V4HistoryIsGoneTest(V4EncodeTestBase):
    """The region trim: v4 carries no transition history at all."""

    def test_no_transition_tokens_and_no_history_columns(self) -> None:
        observation = self._encode(self._state(_RECHARGE_LINES))
        self.assertEqual(len(observation.numeric_features), 23)
        self.assertEqual(V4_REPLAY_OBSERVATION_SPEC.transition_token_count, 0)
        # Every history writer column resolves to "absent", not to a physical index.
        from pokezero.showdown import NUMERIC_TT_DAMAGE_FRACTION, NUMERIC_TT_KO, NUMERIC_TM2_MISS

        for column in (NUMERIC_TT_DAMAGE_FRACTION, NUMERIC_TT_KO, NUMERIC_TM2_MISS):
            self.assertIsNone(
                numeric_index_if_present_for_schema(OBSERVATION_SCHEMA_VERSION_V4, column)
            )

    def test_encoding_needs_no_turn_merged_stream_or_vocabulary(self) -> None:
        # v3 raises without them; v4 must not, because it never reads either.
        replay = parse_showdown_replay(_RECHARGE_LINES, complete_prefix=True)
        state = normalize_for_player(
            replay,
            player_id="p1",
            configured_showdown_slot="p1",
            format_id="gen3randombattle",
            include_turn_merged=False,
        )
        observation = observation_from_player_state(
            state,
            category_vocab=self._vocab(),
            spec=V4_REPLAY_OBSERVATION_SPEC,
            dex=self._dex(),
        )
        observation.validate(V4_REPLAY_OBSERVATION_SPEC)

    def test_the_tier2_derivation_survives_the_trim_even_though_its_columns_do_not(self) -> None:
        # v4 retires BOTH pinned tier2 columns (the conclusions narrow the belief candidate set
        # there instead), and the as-of-strike twins left with the history region — so v4 has no
        # encoded tier2 surface at all.
        from pokezero.showdown import NUMERIC_TIER2_CB_PINNED, NUMERIC_TIER2_INVESTMENT_PINNED

        for column in (NUMERIC_TIER2_CB_PINNED, NUMERIC_TIER2_INVESTMENT_PINNED):
            self.assertIsNone(
                numeric_index_if_present_for_schema(OBSERVATION_SCHEMA_VERSION_V4, column), column
            )
        # Column absence is not evidence the DERIVATION went away: it feeds the belief narrowing
        # now, and that reads the SAME stream. The conclusions come off state.transition_tokens,
        # which normalize still populates at include_turn_merged=False — assert that stream is
        # non-empty on the production path, since an empty one would leave the tier2 trackers
        # with nothing to assess and the narrowing silently inert.
        state = self._state(_RECHARGE_LINES)
        self.assertTrue(
            state.transition_tokens,
            "v4 normalize produced no transition tokens; the tier2 and tendency derivations "
            "read that stream even though v4 encodes none of its rows",
        )
        self.assertIsNotNone(state.tendency_stats)


class V4VolatileOverflowTest(unittest.TestCase):
    """Overflow past the six buckets is loud, counted, and never fatal."""

    def test_overflow_warns_counts_and_still_produces_a_valid_row(self) -> None:
        import warnings as _warnings

        from pokezero.showdown import (
            VolatileBucketOverflowWarning,
            _encode_active_volatiles,
        )
        import pokezero.showdown as showdown_module

        row = [""] * 41
        before = showdown_module.VOLATILE_BUCKET_OVERFLOWS
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            _encode_active_volatiles(
                row,
                [
                    "confusion", "curse", "encore", "leechseed",
                    "substitute", "yawn", "perish3", MUST_RECHARGE_VOLATILE,
                ],
            )
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, VolatileBucketOverflowWarning)
        self.assertEqual(showdown_module.VOLATILE_BUCKET_OVERFLOWS, before + 1)
        # NON-FATAL: the row is still exactly six filled buckets and the right width, so no
        # run, cache, or sample can be broken by the condition.
        self.assertEqual(len(row), 41)
        self.assertEqual(sum(1 for value in row if value), VOLATILE_BUCKET_COUNT)

    def test_a_bag_within_budget_is_silent(self) -> None:
        import warnings as _warnings

        from pokezero.showdown import _encode_active_volatiles

        row = [""] * 41
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            _encode_active_volatiles(row, ["confusion", "substitute"])
        self.assertEqual(caught, [])


class V4FreezesV3Test(V4EncodeTestBase):
    """The invariant a new contract lives or dies by: v3 output is untouched."""

    def _v3(self, lines, *, player="p1"):
        # v3 IS turn-merged, so it needs the merged stream; v4 is not and does not.
        return self._encode(
            self._state(lines, player=player, turn_merged=True), V3_REPLAY_OBSERVATION_SPEC
        )

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

    def test_every_v3_column_carries_the_same_value_under_v4(self) -> None:
        # NOTE ON WHAT THIS DOES AND DOES NOT PROVE. It shows the two projections agree
        # column-for-column on a replay full of pack signals, i.e. no pack write leaked into a
        # shared v3-gated writer. It does NOT establish "identical to before the pack existed"
        # — both sides move together if a shared writer changes. The evidence for the true
        # freeze is the committed golden corpus: arrays.npz is byte-unchanged on this branch.
        lines = _RECHARGE_LINES + [
            "|move|p2a: Blissey|Knock Off|p1a: Slaking",
            "|-enditem|p1a: Slaking|Leftovers|[from] move: Knock Off|[of] p2a: Blissey",
            "|upkeep",
            "|turn|4",
        ]
        # One state that satisfies BOTH: v3 needs the merged stream, v4 ignores it.
        state = self._state(lines, turn_merged=True)
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
        # v4's categorical census is NARROWER than v3's (the turn-merged block is gone), so
        # BOTH directions must be refused. A floor-only check would accept a stale v3 width as
        # a permissible superset and silently emit ten dead columns on a v4-stamped tensor.
        for count in (
            V4_REPLAY_OBSERVATION_SPEC.categorical_feature_count - 1,
            V3_REPLAY_OBSERVATION_SPEC.categorical_feature_count,
        ):
            narrowed_categorical = replace(
                V4_REPLAY_OBSERVATION_SPEC, categorical_feature_count=count
            )
            with self.assertRaises(ValueError):
                observation_from_player_state(
                    self._state(_LEADS),
                    category_vocab=self._vocab(),
                    spec=narrowed_categorical,
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
            {
                LAST_USED_MOVE_SWITCH_SENTINEL,
                LAST_USED_MOVE_BATON_PASS_SENTINEL,
                f"volatile:{MUST_RECHARGE_VOLATILE}",
            },
        )




class V4ExactSpreadsTest(V4EncodeTestBase):
    """v4 asks the generator's spread core instead of re-deriving its rules.

    The two approximations it replaces were measurably wrong: the trimmed-HP bound jumped to
    ev=0 (a full 85-EV strip) where the generator removes 4 EVs at a time and stops at the first
    value satisfying its modular condition, and the zeroed-Atk bound hardcoded iv=0, missing the
    Hidden-Power `-28` that leaves IV 3. Because the emitted band is min/max over survivors, a
    PERFECTLY PINNED set still reported the wrong HP -- a plausible number in the right units,
    ~6% off, which is the class a model cannot detect.
    """

    SHOWDOWN_ROOT = showdown_root_str()

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from pokezero.dex import load_showdown_dex_cached
            from pokezero.randbat import load_gen3_randbat_source_cached

            cls.dex = load_showdown_dex_cached(cls.SHOWDOWN_ROOT)
            cls.source = load_gen3_randbat_source_cached(cls.SHOWDOWN_ROOT)
        except Exception as exc:  # pragma: no cover - no checkout here
            raise unittest.SkipTest(f"needs a pokemon-showdown checkout: {exc}")

    def test_every_pool_variant_matches_the_generator_exactly(self) -> None:
        """COMPARE, do not merely resolve. The old bug produced a value for every variant too.

        Checks the emitted HP and Atk against randbats_spread_details for all ~1682 variants,
        and separately asserts the legacy approximations disagree on the families the fix is
        about -- so a regression to either one fails here rather than looking plausible.
        """
        from pokezero.gen3_damage import randbats_spread_details
        from pokezero.showdown import _gen3_stat, _variant_spread_stats
        from pokezero.tier2 import variant_has_physical_attack

        pinch = {"Salac Berry", "Petaya Berry", "Liechi Berry"}
        checked = legacy_hp_wrong = legacy_atk_wrong = 0
        for key, universe in self.source.universes.items():
            info = self.dex.species_info(key)
            if info is None or not info.base_stats:
                continue
            for variant in universe.variants:
                has_physical = variant_has_physical_attack(variant.moves, self.dex)
                got = _variant_spread_stats(
                    info.base_stats,
                    variant.level,
                    {"moves": list(variant.moves), "item": str(variant.item)},
                    has_physical,
                )
                self.assertIsNotNone(got)
                truth = randbats_spread_details(
                    info.base_stats,
                    level=variant.level,
                    moves=variant.moves,
                    item=variant.item,
                    has_physical_attack=has_physical,
                )
                self.assertEqual(got["hp"], int(truth.stats["hp"]), key)
                self.assertEqual(got["atk"], int(truth.stats["atk"]), key)
                checked += 1

                moves = set(variant.moves)
                trimmed = "bellydrum" in moves or (
                    "substitute" in moves
                    and (moves & {"flail", "reversal"} or str(variant.item) in pinch)
                )
                if trimmed and _gen3_stat(
                    info.base_stats["hp"], variant.level, ev=0, iv=31, hp=True
                ) != got["hp"]:
                    legacy_hp_wrong += 1
                if not has_physical and _gen3_stat(
                    info.base_stats["atk"], variant.level, ev=0, iv=0, hp=False
                ) != got["atk"]:
                    legacy_atk_wrong += 1

        self.assertGreater(checked, 1000, "pool did not load")
        # The whole point: the legacy derivations were wrong on real, common families.
        self.assertGreater(legacy_hp_wrong, 40, "trimmed-HP families no longer differ")
        self.assertGreater(legacy_atk_wrong, 200, "Hidden-Power Atk families no longer differ")

    def test_an_unevaluable_candidate_abandons_the_band_rather_than_inventing_one(self) -> None:
        """A WRONG belief costs more than an ABSENT one -- so degrade, never fabricate.

        Substituting the baseline for an unevaluable candidate and then taking min/max emits a
        bound partly derived from a value no real variant has, and the model reads that exactly
        as confidently as a true one. The correct degradation is the documented no-set-source
        state: low == high == baseline, an honest "unknown".
        """
        from unittest import mock

        from pokezero import showdown
        from pokezero.belief import RevealedPokemonBelief

        variants = (
            {"moves": ["substitute", "flail"], "item": "Salac Berry", "level": 80},
            {"moves": ["tackle"], "item": "Leftovers", "level": 80},
        )
        belief = RevealedPokemonBelief(
            showdown_slot="p2", species="Charizard", candidate_variants=variants
        )

        def _encode(side_effect=None):
            row = [0.0] * 300
            ctx = (
                mock.patch.object(showdown, "_variant_spread_stats", side_effect=side_effect)
                if side_effect
                else mock.patch.object(
                    showdown, "_variant_spread_stats", wraps=showdown._variant_spread_stats
                )
            )
            with ctx:
                showdown._encode_expected_stats(
                    row,
                    self.dex,
                    base_species="Charizard",
                    battle_species="Charizard",
                    details="Charizard, L80, M",
                    belief=belief,
                    exact_spreads=True,
                )
            return (
                row[showdown.NUMERIC_EXPECTED_HP],
                row[showdown.NUMERIC_EXPECTED_HP_LOW],
                row[showdown.NUMERIC_EXPECTED_HP_HIGH],
            )

        base, low, high = _encode()
        self.assertNotEqual(low, high, "fixture must produce a real band when all are evaluable")

        # One candidate unevaluable -> the band must COLLAPSE to the baseline, not straddle a
        # fabricated value.
        base2, low2, high2 = _encode(side_effect=lambda *a, **k: None)
        self.assertEqual(low2, base2)
        self.assertEqual(high2, base2)

    def test_only_v4_gets_the_corrected_spreads(self) -> None:
        """BEHAVIOURAL scoping. Three lineages train against v2.1/v2.2/v3 encodes right now.

        These are frozen legacy positions, so correcting them for the older schemas would shift
        a live input distribution mid-run. v4 is unlaunched and can simply start correct. Applying
        the fix to every schema passes every other test in this file, so the scoping is pinned
        here by comparing what the two encoders actually emit, not by reading the source.
        """
        from pokezero import showdown
        from pokezero.belief import RevealedPokemonBelief

        # A trim-eligible variant: the family where the legacy derivation is wrong by +14..+17.
        belief = RevealedPokemonBelief(
            showdown_slot="p2",
            species="Charizard",
            candidate_variants=({"moves": ["substitute", "flail"], "item": "Salac Berry", "level": 80},),
        )

        def _hp_low(exact: bool) -> float:
            row = [0.0] * 300
            showdown._encode_expected_stats(
                row,
                self.dex,
                base_species="Charizard",
                battle_species="Charizard",
                details="Charizard, L80, M",
                belief=belief,
                exact_spreads=exact,
            )
            return row[showdown.NUMERIC_EXPECTED_HP_LOW]

        legacy, corrected = _hp_low(False), _hp_low(True)
        self.assertNotEqual(
            legacy, corrected, "the fix must actually change this family, or it proves nothing"
        )
        # v4 takes the corrected value; every earlier schema keeps the legacy one.
        self.assertGreater(corrected, legacy, "the generator trims only a few EVs, not all 85")

    def test_only_v4_reaches_the_corrected_path(self) -> None:
        """Catches the scoping regression every value-level test misses.

        Passing `exact_spreads=True` unconditionally changes what three LIVE lineages are fed,
        and it passes every other assertion here because those exercise the flag directly. This
        asserts the CALL SITE, and asserts reachability first -- an earlier version of this test
        passed vacuously because its fixture had no candidate variants, so the path was never
        reached under either schema and the mutation sailed through.
        """
        from unittest import mock

        from pokezero import showdown
        from pokezero.belief import RevealedPokemonBelief

        belief = RevealedPokemonBelief(
            showdown_slot="p2",
            species="Charizard",
            candidate_variants=({"moves": ["substitute", "flail"], "item": "Salac Berry", "level": 80},),
        )

        def _calls(exact: bool) -> int:
            with mock.patch.object(
                showdown, "_variant_spread_stats", wraps=showdown._variant_spread_stats
            ) as spy:
                showdown._encode_expected_stats(
                    [0.0] * 300,
                    self.dex,
                    base_species="Charizard",
                    battle_species="Charizard",
                    details="Charizard, L80, M",
                    belief=belief,
                    exact_spreads=exact,
                )
                return spy.call_count

        # REACHABILITY first: without this the "not called" assertion below proves nothing.
        self.assertGreater(_calls(True), 0, "fixture never reaches the corrected path")
        self.assertEqual(_calls(False), 0, "the legacy path invoked the corrected derivation")

        # And the call site must pass the SCHEMA, not a constant. Reaching this behaviourally
        # needs a set-source-backed belief inside a full v3 encode, which this harness does not
        # build; the assertions above already pin what the flag DOES, so this pins only who
        # supplies it. Hardcoding True here changes the input distribution of three live
        # lineages and is otherwise invisible to every test in this file.
        import inspect

        self.assertIn(
            "exact_spreads=schema_v4", inspect.getsource(showdown._encode_pokemon_tokens)
        )

    def test_an_illegal_spread_actually_raises(self) -> None:
        """Assert the BEHAVIOUR, not the constants.

        `ev=0` is not a legal HP EV -- the generator can never strip the stat -- so the original
        bug produced a spread outside the reachable set. Emitting a plausible-but-wrong stat is
        the failure being removed, so an out-of-set spread must raise rather than degrade.
        """
        from unittest import mock

        from pokezero import showdown

        class _Bogus:
            evs = {"hp": 0, "atk": 85}
            ivs = {"hp": 31}
            stats = {"hp": 1, "atk": 1}

        with mock.patch("pokezero.gen3_damage.randbats_spread_details", return_value=_Bogus()):
            with self.assertRaises(ValueError) as caught:
                showdown._variant_spread_stats(
                    {"hp": 100, "atk": 100}, 80, {"moves": ["tackle"], "item": ""}, True
                )
        self.assertIn("legal set", str(caught.exception))

    def test_a_malformed_candidate_degrades_instead_of_breaking_the_encode(self) -> None:
        """An illegal SPREAD raises; a malformed CANDIDATE must not take down an encode."""
        from pokezero.showdown import _variant_spread_stats

        self.assertIsNone(
            _variant_spread_stats({"hp": 100, "atk": 100}, 80, {"moves": None, "item": ""}, True)
        )


if __name__ == "__main__":  # pragma: no cover
    # At the END. It sat at line 1295, stranding V4ExactSpreadsTest
    # from direct execution -- found by the repo-wide structural guard in
    # tests/test_public_invariant.py.
    unittest.main()
