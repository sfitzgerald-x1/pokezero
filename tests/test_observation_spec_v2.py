"""Observation spec v2 (WS-1 C) tests: durations, exact-state fields, stats/transition
blocks, feature masks, serialization round-trip, and legacy-checkpoint refusal."""

import unittest

from pokezero.belief import CandidateSetSummary
from pokezero.category_vocab import build_category_vocabulary
from pokezero.dex import MoveInfo, ShowdownDex, SpeciesInfo
from pokezero.observation import (
    ACTION_CANDIDATE_TOKEN_COUNT,
    FIELD_TOKEN_COUNT,
    OBSERVATION_SCHEMA_VERSION,
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
    NUMERIC_TIER2_INVESTMENT_PINNED,
    NUMERIC_TT_INVESTMENT_BIT,
    NUMERIC_TT_RESIDUAL,
    NUMERIC_TT_RESIDUAL_VALID,
    NUMERIC_TT_TURNS_AGO,
    NUMERIC_WAKE_KNOWN,
    NUMERIC_WEATHER_PERMANENT,
    NUMERIC_WEATHER_TURNS,
    OPPONENT_POKEMON_TOKEN_OFFSET,
    STATS_TOKEN_OFFSET,
    TRANSITION_TOKEN_OFFSET,
    V2_1_REPLAY_OBSERVATION_SPEC,
    V2_REPLAY_OBSERVATION_SPEC,
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


class InvestmentColumnEncodeTest(unittest.TestCase):
    """The investment column (120) is double-masked (tier2_residuals AND the
    default-OFF tier2_investment provenance switch) and SCHEMA-GATED: the matrix runs
    on the dual-schema fixture, and the legacy v2 encode path never writes the column
    even when a hand-crafted v2-schema config carries the mask (review MED-2a)."""

    _BOTH_SPECS = (V2_REPLAY_OBSERVATION_SPEC, V2_1_REPLAY_OBSERVATION_SPEC)

    def _state_with_code(self, code):
        from dataclasses import replace as dc_replace

        state = _state([
            "|move|p1a: Charizard|Flamethrower|p2a: Xatu",
            "|-damage|p2a: Xatu|60/100",
            "|turn|2",
        ])
        tokens = list(state.transition_tokens)
        index = next(
            i for i, t in enumerate(tokens) if t.kind == TOKEN_KIND_MOVE and t.actor_slot == "p1"
        )
        tokens[index] = dc_replace(tokens[index], investment=code)
        return dc_replace(state, transition_tokens=tuple(tokens)), index

    def test_default_masks_keep_the_column_zero(self) -> None:
        for spec in self._BOTH_SPECS:
            with self.subTest(schema=spec.schema_version):
                state, index = self._state_with_code(-1.0)
                observation = observation_from_player_state(
                    state, category_vocab=_VOCAB, spec=spec
                )
                row = observation.numeric_features[TRANSITION_TOKEN_OFFSET + index]
                self.assertEqual(row[NUMERIC_TT_INVESTMENT_BIT], 0.0)

    def test_investment_mask_populates_the_column_under_v2_1(self) -> None:
        for code in (-1.0, 1.0, 0.5, -0.5):
            state, index = self._state_with_code(code)
            masks = ObservationFeatureMasks(tier2_investment=True)
            observation = observation_from_player_state(
                state, category_vocab=_VOCAB, spec=V2_1_REPLAY_OBSERVATION_SPEC,
                feature_masks=masks,
            )
            row = observation.numeric_features[TRANSITION_TOKEN_OFFSET + index]
            self.assertEqual(row[NUMERIC_TT_INVESTMENT_BIT], code)
            # Other rows stay zero.
            for offset, token in enumerate(state.transition_tokens):
                if offset == index:
                    continue
                other = observation.numeric_features[TRANSITION_TOKEN_OFFSET + offset]
                self.assertEqual(other[NUMERIC_TT_INVESTMENT_BIT], 0.0)

    def test_v2_legacy_path_never_writes_the_column(self) -> None:
        # Review MED-2a: no v2 checkpoint was ever trained on a populated column 120,
        # so a (hand-crafted) v2-schema config carrying tier2_investment=True must be
        # a NO-OP on the legacy encode path — column 120 physically exists in the v2
        # row (it sits below the v2 census end) but stays 0.0 unconditionally.
        for code in (-1.0, 1.0, 0.5, -0.5):
            state, index = self._state_with_code(code)
            masks = ObservationFeatureMasks(tier2_investment=True)
            observation = observation_from_player_state(
                state, category_vocab=_VOCAB, spec=V2_REPLAY_OBSERVATION_SPEC,
                feature_masks=masks,
            )
            for offset in range(len(state.transition_tokens)):
                row = observation.numeric_features[TRANSITION_TOKEN_OFFSET + offset]
                self.assertEqual(row[NUMERIC_TT_INVESTMENT_BIT], 0.0)

    def test_residuals_mask_off_darkens_investment_too(self) -> None:
        for spec in self._BOTH_SPECS:
            with self.subTest(schema=spec.schema_version):
                state, index = self._state_with_code(1.0)
                masks = ObservationFeatureMasks(tier2_residuals=False, tier2_investment=True)
                observation = observation_from_player_state(
                    state, category_vocab=_VOCAB, spec=spec, feature_masks=masks
                )
                row = observation.numeric_features[TRANSITION_TOKEN_OFFSET + index]
                self.assertEqual(row[NUMERIC_TT_INVESTMENT_BIT], 0.0)
                if spec is V2_1_REPLAY_OBSERVATION_SPEC:
                    opp_row = observation.numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET]
                    self.assertEqual(opp_row[NUMERIC_TIER2_INVESTMENT_PINNED], 0.0)

    def test_per_mon_pinned_column_carries_the_code_under_v2_1(self) -> None:
        # NUMERIC_TIER2_INVESTMENT_PINNED (139): the authoritative current-state surface
        # — the struck mon's opp-row carries the annotated code under v2.1 + both masks
        # (CB_PINNED's derivation convention, inverted to token.defender_species);
        # default masks keep it dark.
        state, index = self._state_with_code(0.5)
        self.assertEqual(state.transition_tokens[index].defender_species, "Xatu")
        masks = ObservationFeatureMasks(tier2_investment=True)
        observation = observation_from_player_state(
            state, category_vocab=_VOCAB, spec=V2_1_REPLAY_OBSERVATION_SPEC,
            feature_masks=masks,
        )
        opp_row = observation.numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET]
        self.assertEqual(opp_row[NUMERIC_TIER2_INVESTMENT_PINNED], 0.5)
        dark = observation_from_player_state(
            state, category_vocab=_VOCAB, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
        self.assertEqual(
            dark.numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET][NUMERIC_TIER2_INVESTMENT_PINNED],
            0.0,
        )

    def test_per_mon_code_is_last_annotated_and_truncation_robust(self) -> None:
        # Monotone semantics: the LAST annotated strike's code wins (an HP conclusion
        # upgrading over an earlier defense-only pin). K-truncation robustness: with a
        # budget that truncates every annotated strike out of the tt block, the history
        # column goes dark but the per-mon pinned form still stands — it derives from
        # the FULL untruncated stream, exactly like CB_PINNED.
        from dataclasses import replace as dc_replace

        state = _state([
            "|move|p1a: Charizard|Flamethrower|p2a: Xatu",
            "|-damage|p2a: Xatu|60/100",
            "|turn|2",
            "|move|p1a: Charizard|Flamethrower|p2a: Xatu",
            "|-damage|p2a: Xatu|30/100",
            "|turn|3",
            "|move|p2a: Xatu|Psychic|p1a: Charizard",
            "|-damage|p1a: Charizard|70/100",
            "|turn|4",
        ])
        tokens = list(state.transition_tokens)
        own_moves = [
            i for i, t in enumerate(tokens)
            if t.kind == TOKEN_KIND_MOVE and t.actor_slot == "p1"
        ]
        self.assertEqual(len(own_moves), 2)
        tokens[own_moves[0]] = dc_replace(tokens[own_moves[0]], investment=0.5)
        tokens[own_moves[1]] = dc_replace(tokens[own_moves[1]], investment=-1.0)
        state = dc_replace(state, transition_tokens=tuple(tokens))
        # A later unannotated opponent token must exist so K=1 truncates both strikes.
        self.assertGreater(len(state.transition_tokens), own_moves[1] + 1)

        masks = ObservationFeatureMasks(tier2_investment=True, transition_token_budget=1)
        observation = observation_from_player_state(
            state, category_vocab=_VOCAB, spec=V2_1_REPLAY_OBSERVATION_SPEC,
            feature_masks=masks,
        )
        # The K=1 window holds only the final (unannotated, opponent) token: the
        # history column is dark everywhere...
        for offset in range(TRANSITION_TOKEN_COUNT):
            row = observation.numeric_features[TRANSITION_TOKEN_OFFSET + offset]
            self.assertEqual(row[NUMERIC_TT_INVESTMENT_BIT], 0.0)
        # ...but the pinned per-mon form survives, carrying the LAST code.
        opp_row = observation.numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET]
        self.assertEqual(opp_row[NUMERIC_TIER2_INVESTMENT_PINNED], -1.0)


class TransitionKindLockstepTest(unittest.TestCase):
    def test_encoder_kind_ids_match_transitions_module(self) -> None:
        # showdown.py cannot import transitions at module level (cycle); these literals must
        # stay identical to the extraction module's constants.
        self.assertEqual(_TT_KIND_MOVE, TOKEN_KIND_MOVE)
        self.assertEqual(_TT_KIND_SWITCH, TOKEN_KIND_SWITCH)
        self.assertEqual(_TT_KIND_CANT, TOKEN_KIND_CANT)

    def test_vocab_enum_tuples_match_transitions_constants(self) -> None:
        # The vocabulary's mirrored enum tuples must equal the extraction module's closed
        # value spaces EXACTLY — a hand-mirrored tuple that drifts silently OOVs live games.
        from pokezero import transitions
        from pokezero.randbat_vocab import (
            TRANSITION_EFFECTIVENESS,
            TRANSITION_KINDS,
            TRANSITION_OUTCOMES,
            TRANSITION_SIDE_EFFECTS,
        )

        self.assertEqual(
            set(TRANSITION_KINDS),
            {transitions.TOKEN_KIND_MOVE, transitions.TOKEN_KIND_SWITCH, transitions.TOKEN_KIND_CANT},
        )
        self.assertEqual(set(TRANSITION_OUTCOMES), set(transitions._OUTCOME_RANK))
        self.assertEqual(set(TRANSITION_SIDE_EFFECTS), set(transitions._SIDE_EFFECT_RANK))
        self.assertEqual(
            set(TRANSITION_EFFECTIVENESS),
            {
                transitions.EFFECTIVENESS_NEUTRAL,
                transitions.EFFECTIVENESS_SUPER,
                transitions.EFFECTIVENESS_RESISTED,
                transitions.EFFECTIVENESS_IMMUNE,
            },
        )

    def test_cant_reason_vocabulary_covers_audited_gen3_emitters(self) -> None:
        # Audited reachable |cant| reasons for the gen3 randbats pool (see GEN3_CANT_REASONS
        # comment): status, flinch, attract, recharge, suppressions, broken Focus Punch
        # (in-pool movesets), and the ability-sourced Truant/Damp.
        from pokezero.randbat_vocab import GEN3_CANT_REASONS

        required = {
            "slp", "frz", "par", "flinch", "attract", "recharge",
            "disable", "imprison", "taunt", "focuspunch", "truant", "damp",
        }
        self.assertLessEqual(required, set(GEN3_CANT_REASONS))

    def test_broken_focus_punch_emits_enumerated_cant_action(self) -> None:
        # |cant|POKEMON|Focus Punch|Focus Punch (data/moves.ts onMoveAborted, no gen3
        # override) must normalize to an enumerated action id, never an OOV hash.
        from pokezero.randbat_vocab import GEN3_CANT_REASONS
        from pokezero.transitions import extract_transition_tokens

        replay = parse_showdown_replay(
            _BASE_LINES
            + [
                "|move|p1a: Charizard|Flamethrower|p2a: Xatu",
                "|-damage|p2a: Xatu|60/100",
                "|cant|p2a: Xatu|Focus Punch|Focus Punch",
                "|turn|2",
            ],
            battle_id="battle-1",
        )
        tokens = extract_transition_tokens(replay, perspective_slot="p1")
        cant_token = next(token for token in tokens if token.kind == TOKEN_KIND_CANT)
        self.assertEqual(cant_token.action, "focuspunch")
        self.assertIn(cant_token.action, GEN3_CANT_REASONS)

    def test_combined_extraction_matches_independent_calls(self) -> None:
        from pokezero.transitions import (
            extract_tendency_stats,
            extract_transition_tokens,
            extract_transitions_and_tendencies,
        )

        replay = parse_showdown_replay(
            _BASE_LINES
            + [
                "|move|p2a: Xatu|Psychic|p1a: Charizard",
                "|-damage|p1a: Charizard|70/100",
                "|turn|2",
                "|switch|p2a: Snorlax|Snorlax, L80|100/100",
                "|turn|3",
            ],
            battle_id="battle-1",
        )
        for slot in ("p1", "p2"):
            tokens, stats = extract_transitions_and_tendencies(replay, perspective_slot=slot)
            self.assertEqual(tokens, extract_transition_tokens(replay, perspective_slot=slot))
            self.assertEqual(stats, extract_tendency_stats(replay, perspective_slot=slot))

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
        # Real Showdown emits one |-weather|<id>|[upkeep] at the END of every turn the weather is
        # up (before the next |turn| marker). The countdown is driven by those upkeep ticks, and an
        # [upkeep] line must NOT be mistaken for a fresh set that resets the counter (audit #9).
        state = _state([
            "|move|p1a: Charizard|Rain Dance|p1a: Charizard",
            "|-weather|RainDance",
            "|-weather|RainDance|[upkeep]",  # end of turn 1: first tick consumed at the set turn
            "|turn|2",
            "|-weather|RainDance|[upkeep]",  # end of turn 2: second tick
            "|turn|3",
        ])
        self.assertEqual(state.weather, "raindance")
        self.assertEqual(state.weather_turns_remaining, 3)  # 5 - 2 observed upkeep ticks
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
        observation = observation_from_player_state(
            state, category_vocab=_VOCAB, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
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
        observation = observation_from_player_state(
            state, category_vocab=_VOCAB, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
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
        observation = observation_from_player_state(
            state, category_vocab=_VOCAB, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
        xatu_row = observation.numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET]
        self.assertEqual(xatu_row[NUMERIC_REST_SLEEP], 1.0)
        self.assertEqual(xatu_row[NUMERIC_WAKE_KNOWN], 1.0)

    def test_benched_revealed_trapper_sets_trapper_alive_bit(self) -> None:
        state = _state([
            "|switch|p2a: Dugtrio|Dugtrio, L76|100/100",
            "|-ability|p2a: Dugtrio|Arena Trap",
            "|switch|p2a: Xatu|Xatu, L78|100/100",
        ])
        observation = observation_from_player_state(
            state, category_vocab=_VOCAB, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
        rows = {
            state.opponent_team[index].species: observation.numeric_features[
                OPPONENT_POKEMON_TOKEN_OFFSET + index
            ]
            for index in range(len(state.opponent_team))
        }
        self.assertEqual(rows["Dugtrio"][NUMERIC_TRAPPER_ALIVE], 1.0)
        self.assertEqual(rows["Xatu"][NUMERIC_TRAPPER_ALIVE], 0.0)  # active, not benched

    def test_unrevealed_singleton_candidate_trapper_sets_bit(self) -> None:
        # Audit bug C1: gen3 trap abilities are NEVER protocol-revealed, but all three
        # pool trappers are single-ability species — a singleton live candidate set is
        # certain knowledge and must set the opponent-side bit for a benched trapper.
        from pokezero.belief import CandidateSetSummary

        class SingletonTrapSource:
            def summarize(self, *, format_id, species, revealed_moves, **kwargs):
                if species.lower().startswith("dugtrio"):
                    return CandidateSetSummary(
                        species=species, candidate_count=1, uncertainty=0.5,
                        possible_abilities=("Arena Trap",),
                    )
                return None

        state = _state(
            [
                "|switch|p2a: Dugtrio|Dugtrio, L76|100/100",
                "|switch|p2a: Xatu|Xatu, L78|100/100",
            ],
            set_source=SingletonTrapSource(),
        )
        observation = observation_from_player_state(
            state, category_vocab=_VOCAB, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
        rows = {
            state.opponent_team[index].species: observation.numeric_features[
                OPPONENT_POKEMON_TOKEN_OFFSET + index
            ]
            for index in range(len(state.opponent_team))
        }
        self.assertEqual(rows["Dugtrio"][NUMERIC_TRAPPER_ALIVE], 1.0)

    def test_two_candidate_trap_ability_does_not_set_bit(self) -> None:
        # A hypothetical species where the trap ability is only ONE of two live
        # candidates is NOT certain — the bit must stay dark (hard-rule asymmetry).
        from pokezero.belief import CandidateSetSummary

        class AmbiguousTrapSource:
            def summarize(self, *, format_id, species, revealed_moves, **kwargs):
                if species.lower().startswith("dugtrio"):
                    return CandidateSetSummary(
                        species=species, candidate_count=2, uncertainty=1.0,
                        possible_abilities=("Arena Trap", "Sand Veil"),
                    )
                return None

        state = _state(
            [
                "|switch|p2a: Dugtrio|Dugtrio, L76|100/100",
                "|switch|p2a: Xatu|Xatu, L78|100/100",
            ],
            set_source=AmbiguousTrapSource(),
        )
        observation = observation_from_player_state(
            state, category_vocab=_VOCAB, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
        dugtrio_index = next(
            index for index, mon in enumerate(state.opponent_team) if mon.species == "Dugtrio"
        )
        row = observation.numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET + dugtrio_index]
        self.assertEqual(row[NUMERIC_TRAPPER_ALIVE], 0.0)

    def test_self_mon_uncertainty_is_zero(self) -> None:
        # Own mons are fully known: UNCERTAINTY must be 0.0 on self tokens (the audit
        # flagged the previous constant 1.0 as semantically inverted).
        from pokezero.showdown import NUMERIC_UNCERTAINTY, SELF_POKEMON_TOKEN_OFFSET

        state = _state([])
        observation = observation_from_player_state(
            state, category_vocab=_VOCAB, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
        row = observation.numeric_features[SELF_POKEMON_TOKEN_OFFSET]
        self.assertEqual(row[NUMERIC_UNCERTAINTY], 0.0)

    def test_opponent_pp_fraction_uses_catalog_max_pp(self) -> None:
        state = _state([
            "|move|p2a: Xatu|Psychic|p1a: Charizard",
            "|turn|2",
            "|move|p2a: Xatu|Psychic|p1a: Charizard",
        ])
        observation = observation_from_player_state(
            state, category_vocab=_VOCAB, dex=_fake_dex(), spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
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
        observation = observation_from_player_state(
            state, category_vocab=_VOCAB, dex=_fake_dex(), spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
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
        observation = observation_from_player_state(
            state, category_vocab=_VOCAB, dex=_fake_dex(), spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
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
        observation = observation_from_player_state(
            state, category_vocab=_VOCAB, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
        stats_row = observation.numeric_features[STATS_TOKEN_OFFSET]
        self.assertAlmostEqual(stats_row[NUMERIC_STAT_OPP_DECISION_OPPORTUNITIES], 1 / 64)
        # Rain reveal pair: (set-this-game=1, from-ability=0); order rain/sun/sand/hail.
        self.assertEqual(stats_row[NUMERIC_STAT_WEATHER_REVEAL_OFFSET], 1.0)
        self.assertEqual(stats_row[NUMERIC_STAT_WEATHER_REVEAL_OFFSET + 1], 0.0)

    def test_transition_positional_pair_and_reserved_tier2_slots(self) -> None:
        state = self._history_state()
        observation = observation_from_player_state(
            state, category_vocab=_VOCAB, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
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
        observation = observation_from_player_state(
            state, category_vocab=_VOCAB, feature_masks=masks, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
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
        observation = observation_from_player_state(
            state, category_vocab=_VOCAB, feature_masks=masks, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
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
            spec=V2_1_REPLAY_OBSERVATION_SPEC,
        )
        field_row = masked.numeric_features[0]
        self.assertEqual(field_row[NUMERIC_OPP_SLEEP_CLAUSE], 0.0)
        self.assertEqual(field_row[NUMERIC_WEATHER_TURNS], 0.0)
        xatu_row = masked.numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET]
        self.assertEqual(xatu_row[NUMERIC_EXPECTED_DEF], 0.0)

    def test_masks_do_not_change_shapes_or_schema_version(self) -> None:
        state = self._history_state()
        default = observation_from_player_state(
            state, category_vocab=_VOCAB, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
        masked = observation_from_player_state(
            state,
            category_vocab=_VOCAB,
            feature_masks=ObservationFeatureMasks(
                stats_block=False, exact_state=False, transition_token_budget=32
            ),
            spec=V2_1_REPLAY_OBSERVATION_SPEC,
        )
        self.assertEqual(default.schema_version, masked.schema_version)
        self.assertEqual(len(default.categorical_ids), len(masked.categorical_ids))
        self.assertEqual(len(default.attention_mask), len(masked.attention_mask))
        default.validate(V2_1_REPLAY_OBSERVATION_SPEC)
        masked.validate(V2_1_REPLAY_OBSERVATION_SPEC)


_TRANSFORM_REQUEST = (
    '|request|{"active":[{"moves":[{"move":"Flamethrower","id":"flamethrower"}]}],'
    '"side":{"id":"p1","name":"Us","pokemon":[{"ident":"p1a: Charizard",'
    '"details":"Charizard, L78","condition":"250/250","active":true,'
    '"stats":{"atk":200,"def":180,"spa":220,"spd":190,"spe":210}}]}}'
)

_TRANSFORM_LINES = [
    "|player|p1|Us|",
    "|player|p2|Them|",
    _TRANSFORM_REQUEST,
    "|switch|p1a: Charizard|Charizard, L78|250/250",
    "|switch|p2a: Ditto|Ditto, L88|100/100",
    "|turn|1",
    "|move|p2a: Ditto|Transform|p1a: Charizard",
    "|-transform|p2a: Ditto|p1a: Charizard",
    "|turn|2",
]


def _transform_dex() -> ShowdownDex:
    return ShowdownDex(
        moves={},
        species={
            "ditto": SpeciesInfo(
                id="ditto", name="Ditto", types=("Normal",),
                base_stats={"hp": 48, "atk": 48, "def": 48, "spa": 48, "spd": 48, "spe": 48},
            ),
            "charizard": SpeciesInfo(
                id="charizard", name="Charizard", types=("Fire", "Flying"),
                base_stats={"hp": 78, "atk": 84, "def": 78, "spa": 109, "spd": 85, "spe": 100},
            ),
        },
        type_chart={},
    )


class TransformExpectedStatsTest(unittest.TestCase):
    """Engine-verified rule (vendored sim/pokemon.ts transformInto, no gen3 override):
    Transform copies the TARGET's stored stat VALUES for every non-HP stat and never HP."""

    def test_transformed_opponent_copies_target_actual_non_hp_stats(self) -> None:
        replay = parse_showdown_replay(_TRANSFORM_LINES, battle_id="battle-1")
        state = normalize_for_player(replay, player_id="agent", player_name="Us")
        observation = observation_from_player_state(
            state, category_vocab=_VOCAB, dex=_transform_dex(), spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
        ditto_row = observation.numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET]
        # Non-HP expected stats are the copy TARGET's actual (player-known) values — never
        # Ditto's own variant conditioning applied to the copied species.
        self.assertAlmostEqual(ditto_row[NUMERIC_EXPECTED_DEF], 180 / 714)
        self.assertAlmostEqual(ditto_row[NUMERIC_EXPECTED_SPE], 210 / 714)
        for slot in (NUMERIC_EXPECTED_ATK, NUMERIC_EXPECTED_ATK_LOW, NUMERIC_EXPECTED_ATK_HIGH):
            self.assertAlmostEqual(ditto_row[slot], 200 / 714)
        # HP is never copied: Ditto's own species at Ditto's level (L88), collapsed bounds.
        hp_expected = _gen3_stat(48, 88, ev=85, iv=31, hp=True) / 714
        self.assertAlmostEqual(ditto_row[NUMERIC_EXPECTED_HP], hp_expected)
        self.assertAlmostEqual(ditto_row[NUMERIC_EXPECTED_HP_LOW], hp_expected)

    def test_unidentifiable_transform_target_leaves_expected_block_zero(self) -> None:
        # No request -> self team unknown -> the copy target cannot be identified. Per the
        # asymmetry principle the block must stay ZERO, not a wrong deterministic value.
        lines = [line for line in _TRANSFORM_LINES if not line.startswith("|request|")]
        replay = parse_showdown_replay(lines, battle_id="battle-1")
        state = normalize_for_player(replay, player_id="agent", configured_showdown_slot="p1")
        observation = observation_from_player_state(
            state, category_vocab=_VOCAB, dex=_transform_dex(), spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
        ditto_row = observation.numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET]
        for slot in (
            NUMERIC_EXPECTED_HP, NUMERIC_EXPECTED_HP_LOW, NUMERIC_EXPECTED_ATK,
            NUMERIC_EXPECTED_ATK_LOW, NUMERIC_EXPECTED_ATK_HIGH, NUMERIC_EXPECTED_DEF,
            NUMERIC_EXPECTED_SPE,
        ):
            self.assertEqual(ditto_row[slot], 0.0)


class TransitionTokenFieldGateTest(unittest.TestCase):
    def test_n_hits_is_zero_on_switch_and_cant_tokens(self) -> None:
        from pokezero.showdown import NUMERIC_TT_N_HITS

        state = _state([
            "|move|p1a: Charizard|Flamethrower|p2a: Xatu",
            "|-damage|p2a: Xatu|60/100",
            "|cant|p2a: Xatu|slp",
            "|turn|2",
            "|switch|p2a: Snorlax|Snorlax, L80|100/100",
            "|turn|3",
        ])
        observation = observation_from_player_state(
            state, category_vocab=_VOCAB, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
        kinds = [token.kind for token in state.transition_tokens]
        for index, kind in enumerate(kinds):
            n_hits = observation.numeric_features[TRANSITION_TOKEN_OFFSET + index][NUMERIC_TT_N_HITS]
            if kind == "move":
                self.assertAlmostEqual(n_hits, 1 / 5)
            else:
                self.assertEqual(n_hits, 0.0)


class WishRearmGuardTest(unittest.TestCase):
    def test_failed_second_wish_does_not_extend_pending(self) -> None:
        # A Wish declared while one is pending FAILS in gen 3; the pending window must not
        # be extended by the failed re-declaration.
        replay = parse_showdown_replay(
            _BASE_LINES
            + [
                "|move|p1a: Charizard|Wish|p1a: Charizard",
                "|turn|2",
                "|move|p1a: Charizard|Wish|p1a: Charizard",
                "|-fail|p1a: Charizard",
                "|turn|3",
            ],
            battle_id="battle-1",
        )
        self.assertFalse(_wish_pending(replay, "p1"))

    def test_wish_can_rearm_after_the_previous_wish_expires(self) -> None:
        replay = parse_showdown_replay(
            _BASE_LINES
            + [
                "|move|p1a: Charizard|Wish|p1a: Charizard",
                "|turn|2",
                "|turn|3",
                "|move|p1a: Charizard|Wish|p1a: Charizard",
                "|turn|4",
            ],
            battle_id="battle-1",
        )
        self.assertTrue(_wish_pending(replay, "p1"))


class DataSideOneWayDoorTest(unittest.TestCase):
    def test_rollout_and_cache_schema_strings_bumped_with_the_spec(self) -> None:
        from pokezero.collection import ROLLOUT_RECORD_SCHEMA_VERSION
        from pokezero.dataset import TRAINING_CACHE_SCHEMA_VERSION

        self.assertEqual(ROLLOUT_RECORD_SCHEMA_VERSION, "pokezero.rollout_record.v2")
        self.assertEqual(TRAINING_CACHE_SCHEMA_VERSION, "pokezero.training_cache.v2")

    def _record_with_schema(self, schema_version: str):
        from dataclasses import replace as dc_replace

        from pokezero.collection import RolloutRecord
        from pokezero.env import TerminalState
        from pokezero.trajectory import BattleTrajectory, TrajectoryStep

        state = _state([])
        observation = observation_from_player_state(
            state, category_vocab=_VOCAB, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
        observation = dc_replace(observation, schema_version=schema_version)
        trajectory = BattleTrajectory(battle_id="battle-1", format_id="gen3randombattle", seed=1)
        action_index = next(
            index for index, legal in enumerate(observation.legal_action_mask) if legal
        )
        trajectory.append(
            TrajectoryStep(
                player_id="p1",
                turn_index=0,
                observation=observation,
                legal_action_mask=tuple(observation.legal_action_mask),
                action_index=action_index,
            )
        )
        terminal = TerminalState(winner="p1", turn_count=1)
        trajectory.record_terminal(terminal)
        return RolloutRecord(
            battle_id="battle-1",
            seed=1,
            format_id="gen3randombattle",
            policy_ids={"p1": "test"},
            decision_round_count=1,
            elapsed_seconds=0.0,
            terminal=terminal,
            trajectory=trajectory,
        )

    def test_examples_from_record_refuses_v1_observations_cleanly(self) -> None:
        from pokezero.dataset import examples_from_record

        record = self._record_with_schema("pokezero.observation.v1")
        with self.assertRaisesRegex(ValueError, "pinned tag"):
            list(examples_from_record(record))

    def test_examples_from_record_accepts_current_schema(self) -> None:
        from pokezero.dataset import examples_from_record

        record = self._record_with_schema(OBSERVATION_SCHEMA_VERSION)
        self.assertTrue(list(examples_from_record(record)))

    def test_examples_from_record_accepts_v2_during_dual_schema_window(self) -> None:
        # v2 rollouts stay ingestible while the live v2 training runs produce them; pairing
        # them with the RIGHT model is enforced by the schema-keyed numeric-census guard.
        from pokezero.dataset import examples_from_record

        record = self._record_with_schema("pokezero.observation.v2")
        self.assertTrue(list(examples_from_record(record)))

    def test_missing_observation_schema_version_is_refused_not_assumed_current(self) -> None:
        from pokezero.trajectory import _observation_from_dict as obs_from_dict
        from pokezero.trajectory import _observation_to_dict as obs_to_dict

        state = _state([])
        observation = observation_from_player_state(
            state, category_vocab=_VOCAB, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
        payload = dict(obs_to_dict(observation))
        payload.pop("schema_version")
        decoded = obs_from_dict(payload)
        self.assertEqual(decoded.schema_version, "pokezero.observation.unversioned")
        with self.assertRaisesRegex(ValueError, "pinned tag"):
            decoded.validate(DEFAULT_REPLAY_OBSERVATION_SPEC)

    def test_missing_checkpoint_schema_version_is_refused(self) -> None:
        from pokezero.neural_policy import TransformerPolicyConfig

        payload = TransformerPolicyConfig.compact_category(
            category_vocab=("species:a",), category_oov_buckets=2
        ).to_dict()
        payload.pop("observation_schema_version")
        with self.assertRaisesRegex(ValueError, "pinned tag"):
            TransformerPolicyConfig.from_dict(payload)


class SerializationRoundTripTest(unittest.TestCase):
    def test_observation_round_trips_through_trajectory_payload(self) -> None:
        state = _state([
            "|move|p2a: Xatu|Psychic|p1a: Charizard",
            "|-damage|p1a: Charizard|70/100",
            "|turn|2",
        ])
        observation = observation_from_player_state(
            state, category_vocab=_VOCAB, dex=_fake_dex(), spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
        observation.validate(V2_1_REPLAY_OBSERVATION_SPEC)
        decoded = _observation_from_dict(_observation_to_dict(observation))
        decoded.validate(V2_1_REPLAY_OBSERVATION_SPEC)
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
