"""Observation spec v2.1 tests (checkpoint-driven dual schema).

Covers the schema-keyed spec table, the v2-prefix invariance of the dual encoder, defender
identity on move transition tokens (CATEGORY_MOVE_PRIORITY reuse), the revealed-move
PP-validity bits (the revealed-at-0-PP collision fix), the substitute HP fraction, the
carried-forward investment reserve, and the config-side dual-schema resolution.
"""

import unittest

from pokezero.category_vocab import build_category_vocabulary
from pokezero.observation import (
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION_V2,
    OBSERVATION_SCHEMA_VERSION_V2_1,
    SUPPORTED_OBSERVATION_SCHEMA_VERSIONS,
    ObservationFeatureMasks,
)
from pokezero.showdown import (
    BELIEF_MOVE_BUCKET_COUNT,
    CATEGORY_BELIEF_MOVE_OFFSET,
    CATEGORY_MOVE_PRIORITY,
    DEFAULT_REPLAY_OBSERVATION_SPEC,
    NUMERIC_OPP_MOVE_PP_OFFSET,
    NUMERIC_OPP_MOVE_PP_VALID_OFFSET,
    NUMERIC_SUB_HP_FRACTION,
    NUMERIC_TIER2_CB_PINNED,
    NUMERIC_TIER2_INVESTMENT_PINNED,
    NUMERIC_TT_CB_BIT,
    NUMERIC_TT_INVESTMENT_BIT,
    OPPONENT_POKEMON_TOKEN_OFFSET,
    SELF_POKEMON_TOKEN_OFFSET,
    TRANSITION_TOKEN_OFFSET,
    V2_1_REPLAY_OBSERVATION_SPEC,
    V2_REPLAY_OBSERVATION_SPEC,
    normalize_for_player,
    observation_from_player_state,
    observation_spec_for_schema,
    parse_showdown_replay,
)
from pokezero.transitions import (
    TOKEN_KIND_CANT,
    TOKEN_KIND_MOVE,
    TOKEN_KIND_SWITCH,
    extract_transition_tokens,
)

from test_observation_spec_v2 import _fake_dex

_VOCAB = build_category_vocabulary(
    [
        "species:Charizard", "species:Xatu", "species:Snorlax", "species:Kingdra",
        "move:psychic", "move:spore", "move:substitute", "move:earthquake",
        "stats", "transition:self", "transition:opponent",
        "tt_kind:move", "tt_kind:switch", "tt_kind:cant",
        "tt_outcome:normal", "tt_effectiveness:neutral", "tt_side_effect:none",
        "cant:slp",
    ]
)

# Max HP 221 makes the floor in floor(maxhp/4)/maxhp observable (55/221 != 0.25).
_BASE_LINES = [
    "|player|p1|Us|",
    "|player|p2|Them|",
    '|request|{"active":[{"moves":[{"move":"Flamethrower","id":"flamethrower"}]}],"side":{"id":"p1","name":"Us","pokemon":[{"ident":"p1a: Charizard","details":"Charizard, L78","condition":"221/221","active":true}]}}',
    "|switch|p1a: Charizard|Charizard, L78|221/221",
    "|switch|p2a: Xatu|Xatu, L78|100/100",
    "|turn|1",
]


def _state(lines, **kwargs):
    replay = parse_showdown_replay(_BASE_LINES + lines, battle_id="battle-1")
    return normalize_for_player(replay, player_id="agent", player_name="Us", **kwargs)


def _encode(state, spec, **kwargs):
    return observation_from_player_state(
        state, category_vocab=_VOCAB, spec=spec, dex=_fake_dex(), **kwargs
    )


class SpecTableTest(unittest.TestCase):
    def test_schema_keyed_censuses(self) -> None:
        self.assertEqual(V2_REPLAY_OBSERVATION_SPEC.numeric_feature_count, 121)
        self.assertEqual(V2_REPLAY_OBSERVATION_SPEC.schema_version, OBSERVATION_SCHEMA_VERSION_V2)
        self.assertEqual(V2_1_REPLAY_OBSERVATION_SPEC.numeric_feature_count, 140)
        self.assertEqual(
            V2_1_REPLAY_OBSERVATION_SPEC.schema_version, OBSERVATION_SCHEMA_VERSION_V2_1
        )
        # Fresh (checkpoint-free) encodes and fresh trains default to v2.1.
        self.assertEqual(DEFAULT_REPLAY_OBSERVATION_SPEC, V2_1_REPLAY_OBSERVATION_SPEC)
        self.assertEqual(OBSERVATION_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION_V2_1)
        # Both widths share every other dimension: the schemas differ ONLY in numeric census
        # (plus the schema-conditioned encode branches).
        self.assertEqual(
            V2_REPLAY_OBSERVATION_SPEC.categorical_feature_count,
            V2_1_REPLAY_OBSERVATION_SPEC.categorical_feature_count,
        )
        self.assertEqual(
            V2_REPLAY_OBSERVATION_SPEC.token_count, V2_1_REPLAY_OBSERVATION_SPEC.token_count
        )

    def test_v2_1_column_layout(self) -> None:
        # Investment reserve carries forward unchanged; the new columns follow the v2 census.
        self.assertEqual(NUMERIC_TT_INVESTMENT_BIT, 120)
        self.assertEqual(NUMERIC_OPP_MOVE_PP_VALID_OFFSET, 121)
        self.assertEqual(
            NUMERIC_SUB_HP_FRACTION, NUMERIC_OPP_MOVE_PP_VALID_OFFSET + BELIEF_MOVE_BUCKET_COUNT
        )
        self.assertEqual(NUMERIC_SUB_HP_FRACTION, 137)
        self.assertEqual(NUMERIC_TIER2_CB_PINNED, 138)
        self.assertEqual(NUMERIC_TIER2_INVESTMENT_PINNED, 139)

    def test_spec_for_schema_is_loud_on_unknown_versions(self) -> None:
        for version in SUPPORTED_OBSERVATION_SCHEMA_VERSIONS:
            self.assertEqual(observation_spec_for_schema(version).schema_version, version)
        with self.assertRaisesRegex(ValueError, r"v2.*v2\.1|v2\.1.*v2"):
            observation_spec_for_schema("pokezero.observation.v1")
        with self.assertRaisesRegex(ValueError, "No replay observation spec"):
            observation_spec_for_schema("pokezero.observation.v3")


class DualEncodeTest(unittest.TestCase):
    _LINES = [
        "|move|p2a: Xatu|Psychic|p1a: Charizard",
        "|-damage|p1a: Charizard|150/221",
        "|turn|2",
        "|move|p1a: Charizard|Substitute|p1a: Charizard",
        "|-start|p1a: Charizard|Substitute",
        "|-damage|p1a: Charizard|95/221",
        "|turn|3",
    ]

    def test_v2_encode_is_a_byte_prefix_of_v2_1(self) -> None:
        state = _state(self._LINES)
        v2 = _encode(state, V2_REPLAY_OBSERVATION_SPEC)
        v2_1 = _encode(state, V2_1_REPLAY_OBSERVATION_SPEC)
        self.assertEqual(v2.schema_version, OBSERVATION_SCHEMA_VERSION_V2)
        self.assertEqual(v2_1.schema_version, OBSERVATION_SCHEMA_VERSION_V2_1)
        width = V2_REPLAY_OBSERVATION_SPEC.numeric_feature_count
        for row_index, (v2_row, v21_row) in enumerate(
            zip(v2.numeric_features, v2_1.numeric_features)
        ):
            self.assertEqual(len(v2_row), 121)
            self.assertEqual(len(v21_row), 140)
            self.assertEqual(tuple(v2_row), tuple(v21_row[:width]), f"numeric row {row_index}")
        # Categorical rows agree everywhere except the defender slot on move transition rows.
        for row_index, (v2_row, v21_row) in enumerate(
            zip(v2.categorical_ids, v2_1.categorical_ids)
        ):
            for column, (a, b) in enumerate(zip(v2_row, v21_row)):
                if column == CATEGORY_MOVE_PRIORITY and row_index >= TRANSITION_TOKEN_OFFSET:
                    continue
                self.assertEqual(a, b, f"categorical row {row_index} column {column}")
        self.assertEqual(v2.attention_mask, v2_1.attention_mask)
        self.assertEqual(v2.token_type_ids, v2_1.token_type_ids)

    def test_encode_refuses_undeclared_schema(self) -> None:
        from dataclasses import replace

        state = _state([])
        bad_spec = replace(V2_1_REPLAY_OBSERVATION_SPEC, schema_version="pokezero.observation.v3")
        with self.assertRaisesRegex(ValueError, "unsupported spec schema"):
            _encode(state, bad_spec)

    def test_validate_refuses_cross_schema_pairing(self) -> None:
        state = _state([])
        v2 = _encode(state, V2_REPLAY_OBSERVATION_SPEC)
        v2.validate(V2_REPLAY_OBSERVATION_SPEC)
        with self.assertRaisesRegex(ValueError, "must never be mixed"):
            v2.validate(V2_1_REPLAY_OBSERVATION_SPEC)
        v2_1 = _encode(state, V2_1_REPLAY_OBSERVATION_SPEC)
        v2_1.validate(V2_1_REPLAY_OBSERVATION_SPEC)
        with self.assertRaisesRegex(ValueError, "must never be mixed"):
            v2_1.validate(V2_REPLAY_OBSERVATION_SPEC)


class DefenderIdentityTest(unittest.TestCase):
    _LINES = [
        "|move|p2a: Xatu|Psychic|p1a: Charizard",
        "|-damage|p1a: Charizard|150/221",
        "|turn|2",
        "|switch|p1a: Snorlax|Snorlax, L80|100/100",
        "|move|p2a: Xatu|Psychic|p1a: Snorlax",
        "|turn|3",
        "|cant|p2a: Xatu|slp",
        "|turn|4",
    ]

    def test_extractor_records_defender_base_species_per_strike(self) -> None:
        replay = parse_showdown_replay(_BASE_LINES + self._LINES, battle_id="battle-1")
        tokens = extract_transition_tokens(replay, perspective_slot="p1")
        moves = [token for token in tokens if token.kind == TOKEN_KIND_MOVE]
        self.assertEqual(
            [(token.action, token.defender_species) for token in moves],
            [("psychic", "Charizard"), ("psychic", "Snorlax")],
        )
        for token in tokens:
            if token.kind in {TOKEN_KIND_SWITCH, TOKEN_KIND_CANT}:
                self.assertIsNone(token.defender_species)

    def test_extractor_uses_occupant_species_for_nicknamed_idents(self) -> None:
        lines = [
            "|switch|p2a: Birdy|Xatu, L78|100/100",
            "|move|p1a: Charizard|Flamethrower|p2a: Birdy",
            "|turn|2",
        ]
        replay = parse_showdown_replay(_BASE_LINES + lines, battle_id="battle-1")
        tokens = extract_transition_tokens(replay, perspective_slot="p1")
        move = next(token for token in tokens if token.kind == TOKEN_KIND_MOVE)
        # The slot occupant's switch-in details carry the base species; the nicknamed
        # target ident does not.
        self.assertEqual(move.defender_species, "Xatu")

    def test_self_targeted_move_records_the_actor_as_defender(self) -> None:
        lines = [
            "|move|p1a: Charizard|Substitute|p1a: Charizard",
            "|-start|p1a: Charizard|Substitute",
            "|turn|2",
        ]
        replay = parse_showdown_replay(_BASE_LINES + lines, battle_id="battle-1")
        tokens = extract_transition_tokens(replay, perspective_slot="p1")
        move = next(token for token in tokens if token.kind == TOKEN_KIND_MOVE)
        self.assertEqual(move.defender_species, "Charizard")

    def test_v2_1_encodes_defender_in_move_priority_slot_and_v2_does_not(self) -> None:
        state = _state(self._LINES)
        v2_1 = _encode(state, V2_1_REPLAY_OBSERVATION_SPEC)
        v2 = _encode(state, V2_REPLAY_OBSERVATION_SPEC)
        kinds = [token.kind for token in state.transition_tokens]
        charizard_row = _VOCAB.encode("species:Charizard")
        snorlax_row = _VOCAB.encode("species:Snorlax")
        defender_rows = [
            v2_1.categorical_ids[TRANSITION_TOKEN_OFFSET + index][CATEGORY_MOVE_PRIORITY]
            for index in range(len(kinds))
        ]
        move_rows = [
            row for row, kind in zip(defender_rows, kinds) if kind == TOKEN_KIND_MOVE
        ]
        self.assertEqual(move_rows, [charizard_row, snorlax_row])
        for row, kind in zip(defender_rows, kinds):
            if kind != TOKEN_KIND_MOVE:
                self.assertEqual(row, 0)
        # Under v2 the slot stays padding on EVERY transition row (byte-identity).
        for index in range(len(kinds)):
            self.assertEqual(
                v2.categorical_ids[TRANSITION_TOKEN_OFFSET + index][CATEGORY_MOVE_PRIORITY], 0
            )


class PPValidityTest(unittest.TestCase):
    @staticmethod
    def _uses(count: int, start_turn: int = 2) -> list[str]:
        lines: list[str] = []
        for offset in range(count):
            lines.append("|move|p2a: Xatu|Psychic|p1a: Charizard")
            lines.append(f"|turn|{start_turn + offset}")
        return lines

    def test_revealed_at_0_pp_is_distinguishable_under_v2_1(self) -> None:
        # Psychic: base pp 10 -> catalog max 16. Sixteen uses ledger it to exactly 0 PP.
        state = _state(self._uses(16))
        v2_1 = _encode(state, V2_1_REPLAY_OBSERVATION_SPEC)
        row = v2_1.numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET]
        self.assertEqual(row[NUMERIC_OPP_MOVE_PP_OFFSET], 0.0)  # confirmed empty...
        self.assertEqual(row[NUMERIC_OPP_MOVE_PP_VALID_OFFSET], 1.0)  # ...and confirmed known
        for column in range(1, BELIEF_MOVE_BUCKET_COUNT):
            self.assertEqual(row[NUMERIC_OPP_MOVE_PP_VALID_OFFSET + column], 0.0)
        # Under v2 the documented collision stands: this row is indistinguishable from an
        # unrevealed bucket in the PP channel (and the row is 121 wide - no validity bits).
        v2 = _encode(state, V2_REPLAY_OBSERVATION_SPEC)
        v2_row = v2.numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET]
        self.assertEqual(len(v2_row), 121)
        self.assertEqual(v2_row[NUMERIC_OPP_MOVE_PP_OFFSET], 0.0)

    def test_validity_bit_is_bucket_aligned_and_pp_independent(self) -> None:
        state = _state(self._uses(2))
        v2_1 = _encode(state, V2_1_REPLAY_OBSERVATION_SPEC)
        row = v2_1.numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET]
        self.assertAlmostEqual(row[NUMERIC_OPP_MOVE_PP_OFFSET], 14 / 16)
        self.assertEqual(row[NUMERIC_OPP_MOVE_PP_VALID_OFFSET], 1.0)
        cat_row = v2_1.categorical_ids[OPPONENT_POKEMON_TOKEN_OFFSET]
        self.assertEqual(
            cat_row[CATEGORY_BELIEF_MOVE_OFFSET], _VOCAB.encode("belief:possible_move:psychic")
        )

    def test_validity_bit_fires_even_without_dex_max_pp(self) -> None:
        # Spore is revealed but absent from the fake dex: no PP fraction can be encoded,
        # yet the reveal is protocol ground truth and the confirmed-move flag must fire.
        lines = [
            "|move|p2a: Xatu|Spore|p1a: Charizard",
            "|turn|2",
            "|move|p2a: Xatu|Psychic|p1a: Charizard",
            "|turn|3",
        ]
        state = _state(lines)
        v2_1 = _encode(state, V2_1_REPLAY_OBSERVATION_SPEC)
        row = v2_1.numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET]
        # Buckets sort alphabetically: psychic=0, spore=1.
        self.assertAlmostEqual(row[NUMERIC_OPP_MOVE_PP_OFFSET], 15 / 16)
        self.assertEqual(row[NUMERIC_OPP_MOVE_PP_OFFSET + 1], 0.0)
        self.assertEqual(row[NUMERIC_OPP_MOVE_PP_VALID_OFFSET], 1.0)
        self.assertEqual(row[NUMERIC_OPP_MOVE_PP_VALID_OFFSET + 1], 1.0)

    def test_exact_state_mask_darkens_validity_bits(self) -> None:
        state = _state(self._uses(2))
        masked = _encode(
            state,
            V2_1_REPLAY_OBSERVATION_SPEC,
            feature_masks=ObservationFeatureMasks(exact_state=False),
        )
        row = masked.numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET]
        for column in range(BELIEF_MOVE_BUCKET_COUNT):
            self.assertEqual(row[NUMERIC_OPP_MOVE_PP_OFFSET + column], 0.0)
            self.assertEqual(row[NUMERIC_OPP_MOVE_PP_VALID_OFFSET + column], 0.0)


class SubstituteHPTest(unittest.TestCase):
    def test_self_sub_uses_exact_floor_fraction_from_request_max_hp(self) -> None:
        lines = [
            "|move|p1a: Charizard|Substitute|p1a: Charizard",
            "|-start|p1a: Charizard|Substitute",
            "|-damage|p1a: Charizard|166/221",
            "|turn|2",
        ]
        state = _state(lines)
        v2_1 = _encode(state, V2_1_REPLAY_OBSERVATION_SPEC)
        row = v2_1.numeric_features[SELF_POKEMON_TOKEN_OFFSET]
        # Gen 3 sub HP = floor(221/4) = 55; the floor is observable (55/221 != 0.25).
        self.assertAlmostEqual(row[NUMERIC_SUB_HP_FRACTION], 55 / 221)

    def test_opponent_sub_uses_quarter_baseline(self) -> None:
        lines = [
            "|move|p2a: Xatu|Substitute|p2a: Xatu",
            "|-start|p2a: Xatu|Substitute",
            "|-damage|p2a: Xatu|75/100",
            "|turn|2",
        ]
        state = _state(lines)
        v2_1 = _encode(state, V2_1_REPLAY_OBSERVATION_SPEC)
        row = v2_1.numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET]
        self.assertAlmostEqual(row[NUMERIC_SUB_HP_FRACTION], 0.25)
        self.assertEqual(
            v2_1.numeric_features[SELF_POKEMON_TOKEN_OFFSET][NUMERIC_SUB_HP_FRACTION], 0.0
        )

    def test_sub_break_clears_the_column(self) -> None:
        lines = [
            "|move|p2a: Xatu|Substitute|p2a: Xatu",
            "|-start|p2a: Xatu|Substitute",
            "|turn|2",
            "|move|p1a: Charizard|Flamethrower|p2a: Xatu",
            "|-end|p2a: Xatu|Substitute",
            "|turn|3",
        ]
        state = _state(lines)
        v2_1 = _encode(state, V2_1_REPLAY_OBSERVATION_SPEC)
        row = v2_1.numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET]
        self.assertEqual(row[NUMERIC_SUB_HP_FRACTION], 0.0)

    def test_exact_state_mask_darkens_the_column(self) -> None:
        lines = [
            "|move|p1a: Charizard|Substitute|p1a: Charizard",
            "|-start|p1a: Charizard|Substitute",
            "|turn|2",
        ]
        state = _state(lines)
        masked = _encode(
            state,
            V2_1_REPLAY_OBSERVATION_SPEC,
            feature_masks=ObservationFeatureMasks(exact_state=False),
        )
        self.assertEqual(
            masked.numeric_features[SELF_POKEMON_TOKEN_OFFSET][NUMERIC_SUB_HP_FRACTION], 0.0
        )


class PinnedTier2ConclusionTest(unittest.TestCase):
    """Per-mon pinned Tier-2 conclusions on the opp-mon token surface (v2.1): derived from
    tier2-annotated tokens, tier2_residuals-gated, switch-persistent, and never a Tier-1
    belief mutation (layer separation — the belief columns are untouched)."""

    _LINES = [
        "|move|p2a: Xatu|Psychic|p1a: Charizard",
        "|-damage|p1a: Charizard|150/221",
        "|turn|2",
        "|switch|p2a: Snorlax|Snorlax, L80|100/100",
        "|turn|3",
    ]

    @staticmethod
    def _annotated_state(lines):
        """State whose Xatu strike token carries the tier2 as-of-strike CB bit, as the
        env's Tier2LiveTracker.annotate / batch apply_residuals would stamp it."""
        from dataclasses import replace as dc_replace

        state = _state(lines)
        tokens = list(state.transition_tokens)
        for index, token in enumerate(tokens):
            if token.kind == TOKEN_KIND_MOVE and token.actor_slot == "p2":
                tokens[index] = dc_replace(token, cb_bit=True)
        return dc_replace(state, transition_tokens=tuple(tokens))

    def test_pinned_bit_fires_on_the_concluded_mon_only(self) -> None:
        state = self._annotated_state(self._LINES)
        v2_1 = _encode(state, V2_1_REPLAY_OBSERVATION_SPEC)
        xatu_index = next(
            index for index, mon in enumerate(state.opponent_team) if mon.species == "Xatu"
        )
        snorlax_index = next(
            index for index, mon in enumerate(state.opponent_team) if mon.species == "Snorlax"
        )
        xatu_row = v2_1.numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET + xatu_index]
        snorlax_row = v2_1.numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET + snorlax_index]
        # Xatu has SWITCHED OUT (Snorlax is active) — the pinned bit is a per-mon fact
        # and persists on the benched mon's row.
        self.assertEqual(xatu_row[NUMERIC_TIER2_CB_PINNED], 1.0)
        self.assertEqual(snorlax_row[NUMERIC_TIER2_CB_PINNED], 0.0)
        # Investment twin: a true reserve, zero everywhere.
        self.assertEqual(xatu_row[NUMERIC_TIER2_INVESTMENT_PINNED], 0.0)
        self.assertEqual(snorlax_row[NUMERIC_TIER2_INVESTMENT_PINNED], 0.0)

    def test_tier2_mask_darkens_the_pinned_bit(self) -> None:
        state = self._annotated_state(self._LINES)
        masked = _encode(
            state,
            V2_1_REPLAY_OBSERVATION_SPEC,
            feature_masks=ObservationFeatureMasks(tier2_residuals=False),
        )
        for index in range(len(state.opponent_team)):
            row = masked.numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET + index]
            self.assertEqual(row[NUMERIC_TIER2_CB_PINNED], 0.0)

    def test_unannotated_tokens_leave_the_pinned_bit_dark(self) -> None:
        # The belief-source double-gate travels with the tokens: a plain extraction
        # (no tier2 tracker/inference ran) never sets the column, mask on or not.
        state = _state(self._LINES)
        v2_1 = _encode(state, V2_1_REPLAY_OBSERVATION_SPEC)
        for index in range(len(state.opponent_team)):
            row = v2_1.numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET + index]
            self.assertEqual(row[NUMERIC_TIER2_CB_PINNED], 0.0)

    def test_layer_separation_belief_columns_identical_with_and_without_conclusion(self) -> None:
        # The Tier-2 conclusion must not leak into the Tier-1 belief channel: every
        # categorical column (incl. all belief-fact buckets) and every numeric column
        # outside the two DECLARED tier2 surfaces — the as-of-strike tt cb_bit and the
        # per-mon pinned bit — is identical between the annotated and unannotated
        # encodes of the same state.
        plain = _encode(_state(self._LINES), V2_1_REPLAY_OBSERVATION_SPEC)
        pinned = _encode(self._annotated_state(self._LINES), V2_1_REPLAY_OBSERVATION_SPEC)
        self.assertEqual(plain.categorical_ids, pinned.categorical_ids)
        for row_index, (plain_row, pinned_row) in enumerate(
            zip(plain.numeric_features, pinned.numeric_features)
        ):
            for column, (a, b) in enumerate(zip(plain_row, pinned_row)):
                if column in {NUMERIC_TIER2_CB_PINNED, NUMERIC_TT_CB_BIT}:
                    continue
                self.assertEqual(a, b, f"row {row_index} column {column}")

    def test_v2_encode_ignores_annotated_tokens_beyond_the_tt_columns(self) -> None:
        state = self._annotated_state(self._LINES)
        v2 = _encode(state, V2_REPLAY_OBSERVATION_SPEC)
        for index in range(len(state.opponent_team)):
            self.assertEqual(
                len(v2.numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET + index]), 121
            )


class ConfigDualSchemaTest(unittest.TestCase):
    """Checkpoint-driven resolution on the model-config side (torch-free: pure dataclass)."""

    @staticmethod
    def _config(**kwargs):
        from pokezero.neural_policy import TransformerPolicyConfig

        return TransformerPolicyConfig.compact_category(
            category_vocab=("species:a",), category_oov_buckets=2, **kwargs
        )

    def test_fresh_config_stamps_v2_1_and_its_width(self) -> None:
        config = self._config()
        self.assertEqual(config.observation_schema_version, OBSERVATION_SCHEMA_VERSION_V2_1)
        self.assertEqual(config.numeric_feature_count, 140)

    def test_v2_stamped_config_still_constructs_no_refusal(self) -> None:
        config = self._config(
            observation_schema_version=OBSERVATION_SCHEMA_VERSION_V2, numeric_feature_count=121
        )
        self.assertEqual(config.observation_schema_version, OBSERVATION_SCHEMA_VERSION_V2)
        self.assertEqual(config.numeric_feature_count, 121)

    def test_v1_and_unversioned_configs_still_refuse(self) -> None:
        with self.assertRaisesRegex(ValueError, "pinned tag"):
            self._config(observation_schema_version="pokezero.observation.v1")
        with self.assertRaisesRegex(ValueError, "pinned tag"):
            self._config(observation_schema_version="")

    def test_from_dict_width_defaults_are_schema_keyed(self) -> None:
        from pokezero.neural_policy import TransformerPolicyConfig

        v2_payload = self._config(
            observation_schema_version=OBSERVATION_SCHEMA_VERSION_V2, numeric_feature_count=121
        ).to_dict()
        v2_payload.pop("numeric_feature_count")
        restored = TransformerPolicyConfig.from_dict(v2_payload)
        self.assertEqual(restored.numeric_feature_count, 121)
        v21_payload = self._config().to_dict()
        v21_payload.pop("numeric_feature_count")
        self.assertEqual(TransformerPolicyConfig.from_dict(v21_payload).numeric_feature_count, 140)

    def test_observation_spec_from_model_config_resolves_schema_and_width(self) -> None:
        from pokezero.neural_policy import observation_spec_from_model_config

        v2_spec = observation_spec_from_model_config(
            self._config(
                observation_schema_version=OBSERVATION_SCHEMA_VERSION_V2,
                numeric_feature_count=121,
            )
        )
        self.assertEqual(v2_spec, V2_REPLAY_OBSERVATION_SPEC)
        self.assertEqual(
            observation_spec_from_model_config(self._config()), V2_1_REPLAY_OBSERVATION_SPEC
        )
        # Intra-schema width narrowing survives (the pre-CB/investment 119-column v2 family):
        narrowed = observation_spec_from_model_config(
            self._config(
                observation_schema_version=OBSERVATION_SCHEMA_VERSION_V2,
                numeric_feature_count=119,
            )
        )
        self.assertEqual(narrowed.numeric_feature_count, 119)
        self.assertEqual(narrowed.schema_version, OBSERVATION_SCHEMA_VERSION_V2)

    def test_v2_data_into_v2_1_model_census_error_names_both_schemas(self) -> None:
        from types import SimpleNamespace

        from pokezero.neural_policy import _validate_tensor_shapes

        config = SimpleNamespace(
            window_size=1, token_count=4, categorical_feature_count=3, numeric_feature_count=140
        )

        def fake(shape):
            return SimpleNamespace(shape=shape)

        with self.assertRaisesRegex(ValueError, r"observation\.v2.*observation\.v2\.1"):
            _validate_tensor_shapes(
                fake((2, 1, 4, 3)),
                fake((2, 1, 4, 121)),  # v2-census data meeting a v2.1 model
                fake((2, 1, 4)),
                fake((2, 1, 4)),
                fake((2, 1)),
                config,
            )


if __name__ == "__main__":
    unittest.main()
