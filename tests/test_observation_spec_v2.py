"""Observation spec v2 (WS-1 C) tests: durations, exact-state fields, stats/transition
blocks, feature masks, serialization round-trip, and legacy-checkpoint refusal."""

import unittest

from pokezero.belief import CandidateSetSummary
from pokezero.category_vocab import build_category_vocabulary
from pokezero.dex import MoveInfo, ShowdownDex, SpeciesInfo
from pokezero.observation import (
    ACTION_CANDIDATE_TOKEN_COUNT,
    FIELD_TOKEN_COUNT,
    OPPONENT_POKEMON_TOKEN_COUNT,
    SELF_POKEMON_TOKEN_COUNT,
    STATS_TOKEN_COUNT,
    TRANSITION_TOKEN_COUNT,
    ObservationFeatureMasks,
)
from pokezero.showdown import (
    _TT_KIND_CANT,
    _TT_KIND_MOVE,
    _TT_KIND_SWITCH,
    BELIEF_MOVE_BUCKET_COUNT,
    DEFAULT_REPLAY_OBSERVATION_SPEC,
    NUMERIC_EXPECTED_ATK,
    NUMERIC_EXPECTED_ATK_HIGH,
    NUMERIC_EXPECTED_ATK_LOW,
    NUMERIC_EXPECTED_DEF,
    NUMERIC_EXPECTED_HP,
    NUMERIC_EXPECTED_HP_LOW,
    NUMERIC_EXPECTED_SPE,
    NUMERIC_OPP_MOVE_PP_OFFSET,
    NUMERIC_OPP_SLEEP_CLAUSE,
    NUMERIC_OPP_WISH_PENDING,
    NUMERIC_REST_SLEEP,
    NUMERIC_SELF_SLEEP_CLAUSE,
    NUMERIC_SELF_WISH_PENDING,
    NUMERIC_SLEEP_TURNS,
    NUMERIC_STAT_OPP_DECISION_OPPORTUNITIES,
    NUMERIC_STAT_OPP_SWITCH_COUNT,
    NUMERIC_STAT_WEATHER_REVEAL_OFFSET,
    NUMERIC_TRAPPER_ALIVE,
    NUMERIC_TT_ABS_TURN,
    NUMERIC_TT_RESIDUAL,
    NUMERIC_TT_RESIDUAL_VALID,
    NUMERIC_TT_TURNS_AGO,
    NUMERIC_WAKE_KNOWN,
    NUMERIC_WEATHER_PERMANENT,
    NUMERIC_WEATHER_TURNS,
    OPPONENT_POKEMON_TOKEN_OFFSET,
    STATS_TOKEN_OFFSET,
    TRANSITION_TOKEN_OFFSET,
    _gen3_stat,
    _weather_duration_features,
    _wish_pending,
    normalize_for_player,
    observation_from_player_state,
    parse_showdown_replay,
)
from pokezero.trajectory import _observation_from_dict, _observation_to_dict
from pokezero.transitions import TOKEN_KIND_CANT, TOKEN_KIND_MOVE, TOKEN_KIND_SWITCH

_VOCAB = build_category_vocabulary(
    [
        "species:Charizard", "species:Xatu", "species:Dugtrio", "species:Snorlax",
        "move:psychic", "move:spore", "move:raindance",
        "stats", "transition:self", "transition:opponent",
        "tt_kind:move", "tt_kind:switch", "tt_kind:cant",
        "tt_outcome:normal", "tt_effectiveness:neutral", "tt_side_effect:none",
        "cant:slp",
    ]
)


def _fake_dex() -> ShowdownDex:
    def move(move_id: str, *, gen3_category: str, base_power: int, pp: int) -> MoveInfo:
        return MoveInfo(
            id=move_id, name=move_id, type="Normal", category=gen3_category,
            gen3_category=gen3_category, base_power=base_power, accuracy=100.0,
            priority=0, recoil=False, drain=False, heal=False, status=None,
            boosts={}, target="normal", selfdestruct=False, pp=pp,
        )

    xatu = SpeciesInfo(
        id="xatu", name="Xatu", types=("Psychic", "Flying"),
        base_stats={"hp": 65, "atk": 75, "def": 70, "spa": 95, "spd": 70, "spe": 95},
    )
    return ShowdownDex(
        moves={
            "psychic": move("psychic", gen3_category="Special", base_power=90, pp=10),
            "shadowball": move("shadowball", gen3_category="Physical", base_power=80, pp=15),
            "substitute": move("substitute", gen3_category="Status", base_power=0, pp=10),
            "bellydrum": move("bellydrum", gen3_category="Status", base_power=0, pp=10),
        },
        species={"xatu": xatu},
        type_chart={},
    )


_BASE_LINES = [
    "|player|p1|Us|",
    "|player|p2|Them|",
    '|request|{"active":[{"moves":[{"move":"Flamethrower","id":"flamethrower"}]}],"side":{"id":"p1","name":"Us","pokemon":[{"ident":"p1a: Charizard","details":"Charizard, L78","condition":"100/100","active":true}]}}',
    "|switch|p1a: Charizard|Charizard, L78|100/100",
    "|switch|p2a: Xatu|Xatu, L78|100/100",
    "|turn|1",
]


def _state(lines, **kwargs):
    replay = parse_showdown_replay(_BASE_LINES + lines, battle_id="battle-1")
    return normalize_for_player(replay, player_id="agent", player_name="Us", **kwargs)


class TransitionKindLockstepTest(unittest.TestCase):
    def test_encoder_kind_ids_match_transitions_module(self) -> None:
        # showdown.py cannot import transitions at module level (cycle); these literals must
        # stay identical to the extraction module's constants.
        self.assertEqual(_TT_KIND_MOVE, TOKEN_KIND_MOVE)
        self.assertEqual(_TT_KIND_SWITCH, TOKEN_KIND_SWITCH)
        self.assertEqual(_TT_KIND_CANT, TOKEN_KIND_CANT)

    def test_token_section_offsets(self) -> None:
        self.assertEqual(
            STATS_TOKEN_OFFSET,
            FIELD_TOKEN_COUNT + SELF_POKEMON_TOKEN_COUNT + OPPONENT_POKEMON_TOKEN_COUNT
            + ACTION_CANDIDATE_TOKEN_COUNT,
        )
        self.assertEqual(TRANSITION_TOKEN_OFFSET, STATS_TOKEN_OFFSET + STATS_TOKEN_COUNT)
        self.assertEqual(
            DEFAULT_REPLAY_OBSERVATION_SPEC.token_count,
            TRANSITION_TOKEN_OFFSET + TRANSITION_TOKEN_COUNT,
        )


class DurationTrackingTest(unittest.TestCase):
    def test_move_weather_counts_down_and_upkeep_does_not_reset(self) -> None:
        state = _state([
            "|move|p1a: Charizard|Rain Dance|p1a: Charizard",
            "|-weather|RainDance",
            "|turn|2",
            "|-weather|RainDance|[upkeep]",
            "|turn|3",
        ])
        self.assertEqual(state.weather, "raindance")
        self.assertEqual(state.weather_turns_remaining, 3)  # 5 - (turn 3 - set turn 1)
        self.assertFalse(state.weather_permanent)

    def test_ability_weather_is_permanent(self) -> None:
        state = _state([
            "|switch|p2a: Tyranitar|Tyranitar, L74|100/100",
            "|-weather|Sandstorm|[from] ability: Sand Stream|[of] p2a: Tyranitar",
            "|turn|2",
            "|turn|3",
        ])
        self.assertEqual(state.weather, "sandstorm")
        self.assertTrue(state.weather_permanent)
        self.assertEqual(state.weather_turns_remaining, 5)  # pinned, never decays

    def test_weather_none_clears_duration_tracking(self) -> None:
        replay = parse_showdown_replay(
            _BASE_LINES + ["|-weather|RainDance", "|turn|2", "|-weather|none"],
            battle_id="battle-1",
        )
        self.assertIsNone(replay.weather)
        self.assertEqual(_weather_duration_features(replay), (0, False))

    def test_timed_side_conditions_track_set_turn_and_clear_on_sideend(self) -> None:
        state = _state([
            "|move|p1a: Charizard|Reflect|p1a: Charizard",
            "|-sidestart|p1: Us|Reflect",
            "|move|p2a: Xatu|Safeguard|p2a: Xatu",
            "|-sidestart|p2: Them|Safeguard",
            "|turn|2",
            "|turn|3",
        ])
        self.assertEqual(state.self_timed_condition_turns.get("reflect"), 3)
        self.assertEqual(state.opponent_timed_condition_turns.get("safeguard"), 3)
        ended = _state([
            "|move|p1a: Charizard|Reflect|p1a: Charizard",
            "|-sidestart|p1: Us|Reflect",
            "|turn|2",
            "|-sideend|p1: Us|Reflect",
        ])
        self.assertNotIn("reflect", ended.self_timed_condition_turns)

    def test_wish_pending_lifecycle(self) -> None:
        pending = _state(["|move|p1a: Charizard|Wish|p1a: Charizard", "|turn|2"])
        self.assertTrue(pending.self_wish_pending)
        self.assertFalse(pending.opponent_wish_pending)
        landed = _state([
            "|move|p1a: Charizard|Wish|p1a: Charizard",
            "|turn|2",
            "|-heal|p1a: Charizard|100/100|[from] move: Wish|[wisher] Charizard",
            "|turn|3",
        ])
        self.assertFalse(landed.self_wish_pending)
        expired = parse_showdown_replay(
            _BASE_LINES + ["|move|p1a: Charizard|Wish|p1a: Charizard", "|turn|2", "|turn|3"],
            battle_id="battle-1",
        )
        self.assertFalse(_wish_pending(expired, "p1"))  # full-HP landing emits no heal

    def test_sleep_clause_bits_are_live(self) -> None:
        slept = _state([
            "|move|p2a: Xatu|Spore|p1a: Charizard",
            "|-status|p1a: Charizard|slp|[from] move: Spore",
        ])
        self.assertTrue(slept.opponent_sleep_clause_used)
        self.assertFalse(slept.self_sleep_clause_used)
        woke = _state([
            "|move|p2a: Xatu|Spore|p1a: Charizard",
            "|-status|p1a: Charizard|slp|[from] move: Spore",
            "|turn|2",
            "|-curestatus|p1a: Charizard|slp",
        ])
        self.assertFalse(woke.opponent_sleep_clause_used)


class ExactStateEncodingTest(unittest.TestCase):
    def test_field_token_carries_side_level_exact_state(self) -> None:
        state = _state([
            "|move|p2a: Xatu|Spore|p1a: Charizard",
            "|-status|p1a: Charizard|slp|[from] move: Spore",
            "|move|p2a: Xatu|Wish|p2a: Xatu",
            "|-weather|RainDance",
        ])
        observation = observation_from_player_state(state, category_vocab=_VOCAB)
        field_row = observation.numeric_features[0]
        self.assertEqual(field_row[NUMERIC_OPP_SLEEP_CLAUSE], 1.0)
        self.assertEqual(field_row[NUMERIC_SELF_SLEEP_CLAUSE], 0.0)
        self.assertEqual(field_row[NUMERIC_OPP_WISH_PENDING], 1.0)
        self.assertEqual(field_row[NUMERIC_SELF_WISH_PENDING], 0.0)
        self.assertEqual(field_row[NUMERIC_WEATHER_TURNS], 1.0)
        self.assertEqual(field_row[NUMERIC_WEATHER_PERMANENT], 0.0)

    def test_opponent_sleep_counters_and_ambiguous_rest_wake(self) -> None:
        # Xatu Rest-sleeps; Early Bird is a live candidate -> rest bit set, wake ambiguous.
        state = _state(
            [
                "|move|p2a: Xatu|Rest|p2a: Xatu",
                "|-status|p2a: Xatu|slp|[from] move: Rest",
                "|turn|2",
                "|cant|p2a: Xatu|slp",
                "|turn|3",
            ],
            format_id="gen3randombattle",
            set_source=_EarlyBirdSetSource(),
        )
        observation = observation_from_player_state(state, category_vocab=_VOCAB)
        xatu_row = observation.numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET]
        self.assertAlmostEqual(xatu_row[NUMERIC_SLEEP_TURNS], 1 / 5)
        self.assertEqual(xatu_row[NUMERIC_REST_SLEEP], 1.0)
        self.assertEqual(xatu_row[NUMERIC_WAKE_KNOWN], 0.0)

    def test_rest_wake_known_when_early_bird_absent_from_candidates(self) -> None:
        state = _state(
            [
                "|move|p2a: Xatu|Rest|p2a: Xatu",
                "|-status|p2a: Xatu|slp|[from] move: Rest",
            ],
            format_id="gen3randombattle",
            set_source=_SynchronizeOnlySetSource(),
        )
        observation = observation_from_player_state(state, category_vocab=_VOCAB)
        xatu_row = observation.numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET]
        self.assertEqual(xatu_row[NUMERIC_REST_SLEEP], 1.0)
        self.assertEqual(xatu_row[NUMERIC_WAKE_KNOWN], 1.0)

    def test_benched_revealed_trapper_sets_trapper_alive_bit(self) -> None:
        state = _state([
            "|switch|p2a: Dugtrio|Dugtrio, L76|100/100",
            "|-ability|p2a: Dugtrio|Arena Trap",
            "|switch|p2a: Xatu|Xatu, L78|100/100",
        ])
        observation = observation_from_player_state(state, category_vocab=_VOCAB)
        rows = {
            state.opponent_team[index].species: observation.numeric_features[
                OPPONENT_POKEMON_TOKEN_OFFSET + index
            ]
            for index in range(len(state.opponent_team))
        }
        self.assertEqual(rows["Dugtrio"][NUMERIC_TRAPPER_ALIVE], 1.0)
        self.assertEqual(rows["Xatu"][NUMERIC_TRAPPER_ALIVE], 0.0)  # active, not benched

    def test_opponent_pp_fraction_uses_catalog_max_pp(self) -> None:
        state = _state([
            "|move|p2a: Xatu|Psychic|p1a: Charizard",
            "|turn|2",
            "|move|p2a: Xatu|Psychic|p1a: Charizard",
        ])
        observation = observation_from_player_state(state, category_vocab=_VOCAB, dex=_fake_dex())
        xatu_row = observation.numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET]
        # Psychic: base pp 10 -> catalog max 16; two uses -> 14/16. Only one revealed move,
        # so it occupies bucket column 0.
        self.assertAlmostEqual(xatu_row[NUMERIC_OPP_MOVE_PP_OFFSET], 14 / 16)
        for column in range(1, BELIEF_MOVE_BUCKET_COUNT):
            self.assertEqual(xatu_row[NUMERIC_OPP_MOVE_PP_OFFSET + column], 0.0)

    def test_expected_stats_fixed_four_and_variant_conditioned_hp_atk(self) -> None:
        state = _state(
            [],
            format_id="gen3randombattle",
            set_source=_VariantSetSource(),
        )
        observation = observation_from_player_state(state, category_vocab=_VOCAB, dex=_fake_dex())
        xatu_row = observation.numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET]
        level = 78
        expected_def = _gen3_stat(70, level, ev=85, iv=31, hp=False) / 714.0
        expected_spe = _gen3_stat(95, level, ev=85, iv=31, hp=False) / 714.0
        self.assertAlmostEqual(xatu_row[NUMERIC_EXPECTED_DEF], expected_def)
        self.assertAlmostEqual(xatu_row[NUMERIC_EXPECTED_SPE], expected_spe)
        atk_baseline = _gen3_stat(75, level, ev=85, iv=31, hp=False)
        atk_zeroed = _gen3_stat(75, level, ev=0, iv=0, hp=False)
        hp_baseline = _gen3_stat(65, level, ev=85, iv=31, hp=True)
        hp_trimmed = _gen3_stat(65, level, ev=0, iv=31, hp=True)
        # Variant A has a physical attack (Shadow Ball, physical in gen 3) -> baseline Atk;
        # variant B (Sub + Belly Drum-free but pinch berry) zeroes Atk and trims HP.
        self.assertAlmostEqual(xatu_row[NUMERIC_EXPECTED_ATK], atk_baseline / 714.0)
        self.assertAlmostEqual(xatu_row[NUMERIC_EXPECTED_ATK_LOW], atk_zeroed / 714.0)
        self.assertAlmostEqual(xatu_row[NUMERIC_EXPECTED_ATK_HIGH], atk_baseline / 714.0)
        self.assertAlmostEqual(xatu_row[NUMERIC_EXPECTED_HP], hp_baseline / 714.0)
        self.assertAlmostEqual(xatu_row[NUMERIC_EXPECTED_HP_LOW], hp_trimmed / 714.0)

    def test_expected_stats_without_set_source_collapse_bounds_to_baseline(self) -> None:
        state = _state([])
        observation = observation_from_player_state(state, category_vocab=_VOCAB, dex=_fake_dex())
        xatu_row = observation.numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET]
        self.assertEqual(xatu_row[NUMERIC_EXPECTED_ATK], xatu_row[NUMERIC_EXPECTED_ATK_LOW])
        self.assertEqual(xatu_row[NUMERIC_EXPECTED_ATK], xatu_row[NUMERIC_EXPECTED_ATK_HIGH])
        self.assertEqual(xatu_row[NUMERIC_EXPECTED_HP], xatu_row[NUMERIC_EXPECTED_HP_LOW])


class StatsAndTransitionBlockTest(unittest.TestCase):
    def _history_state(self):
        return _state([
            "|move|p2a: Xatu|Psychic|p1a: Charizard",
            "|-damage|p1a: Charizard|70/100",
            "|move|p1a: Charizard|Flamethrower|p2a: Xatu",
            "|-damage|p2a: Xatu|60/100",
            "|turn|2",
            "|switch|p2a: Snorlax|Snorlax, L80|100/100",
            "|move|p1a: Charizard|Flamethrower|p2a: Snorlax",
            "|-damage|p2a: Snorlax|80/100",
            "|turn|3",
        ])

    def test_stats_token_counts_and_weather_reveals(self) -> None:
        state = _state([
            "|move|p2a: Xatu|Rain Dance|p2a: Xatu",
            "|-weather|RainDance",
            "|turn|2",
        ])
        observation = observation_from_player_state(state, category_vocab=_VOCAB)
        stats_row = observation.numeric_features[STATS_TOKEN_OFFSET]
        self.assertAlmostEqual(stats_row[NUMERIC_STAT_OPP_DECISION_OPPORTUNITIES], 1 / 64)
        # Rain reveal pair: (set-this-game=1, from-ability=0); order rain/sun/sand/hail.
        self.assertEqual(stats_row[NUMERIC_STAT_WEATHER_REVEAL_OFFSET], 1.0)
        self.assertEqual(stats_row[NUMERIC_STAT_WEATHER_REVEAL_OFFSET + 1], 0.0)

    def test_transition_positional_pair_and_reserved_tier2_slots(self) -> None:
        state = self._history_state()
        observation = observation_from_player_state(state, category_vocab=_VOCAB)
        # First transition after the leads: Xatu's turn-1 Psychic.
        first_move_row = observation.numeric_features[TRANSITION_TOKEN_OFFSET + 2]
        self.assertAlmostEqual(first_move_row[NUMERIC_TT_ABS_TURN], 1 / 1000)
        self.assertAlmostEqual(first_move_row[NUMERIC_TT_TURNS_AGO], (3 - 1) / 64)
        for row_index in range(TRANSITION_TOKEN_OFFSET, TRANSITION_TOKEN_OFFSET + 5):
            self.assertEqual(observation.numeric_features[row_index][NUMERIC_TT_RESIDUAL], 0.0)
            self.assertEqual(observation.numeric_features[row_index][NUMERIC_TT_RESIDUAL_VALID], 0.0)

    def test_transition_budget_mask_truncates_oldest_first(self) -> None:
        state = self._history_state()
        masks = ObservationFeatureMasks(transition_token_budget=2)
        observation = observation_from_player_state(state, category_vocab=_VOCAB, feature_masks=masks)
        self.assertEqual(len(state.transition_tokens), 6)
        # Only the two most recent actions (turn-2 switch + Flamethrower) are encoded.
        self.assertEqual(
            observation.categorical_ids[TRANSITION_TOKEN_OFFSET][3], _VOCAB.encode("tt_kind:switch")
        )
        self.assertEqual(
            observation.categorical_ids[TRANSITION_TOKEN_OFFSET + 1][3], _VOCAB.encode("tt_kind:move")
        )
        self.assertTrue(all(observation.attention_mask[TRANSITION_TOKEN_OFFSET : TRANSITION_TOKEN_OFFSET + 2]))
        self.assertFalse(any(observation.attention_mask[TRANSITION_TOKEN_OFFSET + 2 :]))

    def test_stats_block_mask_zeroes_and_hides_the_stats_token(self) -> None:
        state = self._history_state()
        masks = ObservationFeatureMasks(stats_block=False)
        observation = observation_from_player_state(state, category_vocab=_VOCAB, feature_masks=masks)
        self.assertFalse(observation.attention_mask[STATS_TOKEN_OFFSET])
        self.assertEqual(set(observation.numeric_features[STATS_TOKEN_OFFSET]), {0.0})
        self.assertEqual(set(observation.categorical_ids[STATS_TOKEN_OFFSET]), {0})

    def test_exact_state_mask_zeroes_exact_state_columns(self) -> None:
        state = _state([
            "|move|p2a: Xatu|Spore|p1a: Charizard",
            "|-status|p1a: Charizard|slp|[from] move: Spore",
            "|-weather|RainDance",
        ])
        masked = observation_from_player_state(
            state,
            category_vocab=_VOCAB,
            dex=_fake_dex(),
            feature_masks=ObservationFeatureMasks(exact_state=False),
        )
        field_row = masked.numeric_features[0]
        self.assertEqual(field_row[NUMERIC_OPP_SLEEP_CLAUSE], 0.0)
        self.assertEqual(field_row[NUMERIC_WEATHER_TURNS], 0.0)
        xatu_row = masked.numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET]
        self.assertEqual(xatu_row[NUMERIC_EXPECTED_DEF], 0.0)

    def test_masks_do_not_change_shapes_or_schema_version(self) -> None:
        state = self._history_state()
        default = observation_from_player_state(state, category_vocab=_VOCAB)
        masked = observation_from_player_state(
            state,
            category_vocab=_VOCAB,
            feature_masks=ObservationFeatureMasks(
                stats_block=False, exact_state=False, transition_token_budget=32
            ),
        )
        self.assertEqual(default.schema_version, masked.schema_version)
        self.assertEqual(len(default.categorical_ids), len(masked.categorical_ids))
        self.assertEqual(len(default.attention_mask), len(masked.attention_mask))
        default.validate(DEFAULT_REPLAY_OBSERVATION_SPEC)
        masked.validate(DEFAULT_REPLAY_OBSERVATION_SPEC)


class SerializationRoundTripTest(unittest.TestCase):
    def test_observation_round_trips_through_trajectory_payload(self) -> None:
        state = _state([
            "|move|p2a: Xatu|Psychic|p1a: Charizard",
            "|-damage|p1a: Charizard|70/100",
            "|turn|2",
        ])
        observation = observation_from_player_state(state, category_vocab=_VOCAB, dex=_fake_dex())
        observation.validate(DEFAULT_REPLAY_OBSERVATION_SPEC)
        decoded = _observation_from_dict(_observation_to_dict(observation))
        decoded.validate(DEFAULT_REPLAY_OBSERVATION_SPEC)
        self.assertEqual(decoded.schema_version, observation.schema_version)
        self.assertEqual(decoded.categorical_ids, observation.categorical_ids)
        self.assertEqual(decoded.numeric_features, observation.numeric_features)
        self.assertEqual(decoded.token_type_ids, observation.token_type_ids)
        self.assertEqual(decoded.attention_mask, observation.attention_mask)
        self.assertEqual(decoded.legal_action_mask, tuple(observation.legal_action_mask))
        self.assertEqual(decoded.perspective, observation.perspective)


class LegacyCheckpointRefusalTest(unittest.TestCase):
    def test_v1_model_config_refuses_with_pinned_tag_message(self) -> None:
        from pokezero.neural_policy import TransformerPolicyConfig

        payload = TransformerPolicyConfig.compact_category(
            category_vocab=("species:a",), category_oov_buckets=2
        ).to_dict()
        payload["observation_schema_version"] = "pokezero.observation.v1"
        with self.assertRaisesRegex(ValueError, "pinned tag"):
            TransformerPolicyConfig.from_dict(payload)

    def test_v1_checkpoint_file_load_refuses_cleanly(self) -> None:
        from pokezero.neural_policy import (
            NEURAL_POLICY_SCHEMA_VERSION,
            NEURAL_TRAINING_SCHEMA_VERSION,
            TransformerPolicyConfig,
            TransformerTrainingConfig,
            load_transformer_checkpoint,
            torch_available,
        )

        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        import tempfile
        from pathlib import Path

        import torch

        config_payload = TransformerPolicyConfig.compact_category(
            category_vocab=("species:a",), category_oov_buckets=2
        ).to_dict()
        # Simulate a checkpoint written before the spec bump.
        config_payload["observation_schema_version"] = "pokezero.observation.v1"
        config_payload["window_size"] = 4
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.pt"
            torch.save(
                {
                    "schema_version": NEURAL_POLICY_SCHEMA_VERSION,
                    "training_schema_version": NEURAL_TRAINING_SCHEMA_VERSION,
                    "model_config": config_payload,
                    "training_config": TransformerTrainingConfig().to_dict(),
                    "epochs": [],
                    "value_calibration_transform": None,
                    "belief_set_source_hash": None,
                    "state_dict": {},
                },
                path,
            )
            with self.assertRaisesRegex(ValueError, "pokezero.observation.v1.*pinned tag"):
                load_transformer_checkpoint(path)


class _EarlyBirdSetSource:
    def summarize(self, *, format_id, species, revealed_moves, **kwargs):
        return CandidateSetSummary(
            species=species,
            candidate_count=2,
            uncertainty=0.5,
            possible_abilities=("Early Bird", "Synchronize"),
        )


class _SynchronizeOnlySetSource:
    def summarize(self, *, format_id, species, revealed_moves, **kwargs):
        return CandidateSetSummary(
            species=species,
            candidate_count=1,
            uncertainty=0.0,
            possible_abilities=("Synchronize",),
        )


class _VariantSetSource:
    def summarize(self, *, format_id, species, revealed_moves, **kwargs):
        return CandidateSetSummary(
            species=species,
            candidate_count=2,
            uncertainty=0.5,
            possible_abilities=("Synchronize",),
            candidate_variants=(
                {
                    "variant_id": "xatu-1",
                    "moves": ["shadowball", "psychic"],
                    "ability": "Synchronize",
                    "item": "Leftovers",
                },
                {
                    "variant_id": "xatu-2",
                    "moves": ["substitute", "psychic"],
                    "ability": "Synchronize",
                    "item": "Salac Berry",
                },
            ),
        )


if __name__ == "__main__":
    unittest.main()
